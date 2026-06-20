"""Train TRUE176/CSS8 128-IP DeepONet with LE + Sobolev B supervision.

This is the DeepONet-aligned version of the current NNSE launcher contract:

    input  = shape4[4] + q48_raw[48]
    trunk  = deterministic 128-IP geometry/ID features from shape4
    output = LE_norm[target_ips, 6]
    B      = d(LE_norm)/d(q48_norm), supervised by normalized Abaqus B

The script intentionally keeps data/log artifacts out of git.  It accepts a list
of compact npz files or a text file containing compact paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .models import True176Shape4QrawDeepONet
from .true176_data import (
    SobolevArrayDataset,
    build_point_features,
    load_compacts,
    parse_int_list,
    parse_target_ips,
    split_indices,
    stats,
)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return str(obj)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def rel_np(pred: np.ndarray, true: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(true, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(p - t) / max(np.linalg.norm(t), 1.0e-300))


def cos_np(pred: np.ndarray, true: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(true, dtype=np.float64).reshape(-1)
    den = float(np.linalg.norm(p) * np.linalg.norm(t))
    return float(np.dot(p, t) / den) if den > 1.0e-300 else float("nan")


def distributed_context(args: argparse.Namespace) -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = bool(args.ddp) or world_size > 1
    if distributed and bool(args.cuda) and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=str(args.ddp_backend))
    if distributed:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    return {"distributed": distributed, "world_size": world_size, "rank": rank, "local_rank": local_rank, "is_main": rank == 0}


def ddp_barrier(ctx: dict[str, Any]) -> None:
    if bool(ctx.get("distributed")) and dist.is_initialized():
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[int(ctx.get("local_rank", 0))])
        else:
            dist.barrier()


def ddp_mean(value: float, device: torch.device, ctx: dict[str, Any]) -> float:
    if not bool(ctx.get("distributed")):
        return float(value)
    t = torch.as_tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / max(int(ctx.get("world_size", 1)), 1)
    return float(t.detach().cpu())


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def model_q_slice(model: nn.Module) -> tuple[int, int]:
    base = unwrap_model(model)
    return int(getattr(base, "q_start", 4)), int(getattr(base, "q_dim", 48))


def ad_jacobian_forward(model: nn.Module, xb: torch.Tensor, pb: torch.Tensor, columns: list[int], *, create_graph: bool) -> torch.Tensor:
    if not columns:
        return torch.zeros((xb.shape[0], pb.shape[1], 6, 0), dtype=xb.dtype, device=xb.device)
    x_req = xb.detach().clone().requires_grad_(True)
    q_start, _q_dim = model_q_slice(model)

    def fn(x_in: torch.Tensor) -> torch.Tensor:
        return model(x_in, pb)

    cols: list[torch.Tensor] = []
    for col in columns:
        tangent = torch.zeros_like(x_req)
        tangent[:, q_start + int(col)] = 1.0
        _le, jvp = torch.autograd.functional.jvp(fn, x_req, tangent, create_graph=bool(create_graph), strict=False)
        cols.append(jvp)
    return torch.stack(cols, dim=-1)


def ad_jacobian_reverse(model: nn.Module, xb: torch.Tensor, pb: torch.Tensor, columns: list[int], *, create_graph: bool) -> torch.Tensor:
    if not columns:
        return torch.zeros((xb.shape[0], pb.shape[1], 6, 0), dtype=xb.dtype, device=xb.device)
    x_req = xb.detach().clone().requires_grad_(True)
    le = model(x_req, pb)
    q_start, _q_dim = model_q_slice(model)
    q_cols = torch.as_tensor([q_start + int(c) for c in columns], dtype=torch.long, device=xb.device)
    out_ips: list[torch.Tensor] = []
    for ip in range(int(le.shape[1])):
        rows: list[torch.Tensor] = []
        for comp in range(6):
            grad_x = torch.autograd.grad(
                le[:, ip, comp].sum(),
                x_req,
                create_graph=bool(create_graph),
                retain_graph=True,
                only_inputs=True,
            )[0]
            rows.append(grad_x.index_select(-1, q_cols))
        out_ips.append(torch.stack(rows, dim=1))
    return torch.stack(out_ips, dim=1)


def ad_jacobian(model: nn.Module, xb: torch.Tensor, pb: torch.Tensor, columns: list[int], *, create_graph: bool, method: str) -> torch.Tensor:
    key = str(method).strip().lower()
    if key == "forward":
        return ad_jacobian_forward(model, xb, pb, columns, create_graph=create_graph)
    if key == "reverse":
        return ad_jacobian_reverse(model, xb, pb, columns, create_graph=create_graph)
    raise ValueError(f"unknown jacobian method: {method}")


def sample_columns(columns: list[int], count: int, rng: np.random.Generator) -> list[int]:
    if not columns:
        return []
    n = int(count)
    if n <= 0 or n >= len(columns):
        return list(columns)
    picked = rng.choice(np.asarray(columns, dtype=np.int64), size=n, replace=False)
    return [int(v) for v in picked.tolist()]


def j_norm_to_b_phys(j_norm: torch.Tensor, le_std: torch.Tensor, q_std_cols: torch.Tensor) -> torch.Tensor:
    q_scale = torch.clamp(q_std_cols, min=1.0e-12).reshape(1, 1, 1, -1)
    le_scale = torch.clamp(le_std, min=1.0e-12).reshape(1, j_norm.shape[1], 6, 1)
    return j_norm * le_scale / q_scale


def physical_j_loss(
    j_pred_norm: torch.Tensor,
    j_true_norm: torch.Tensor,
    le_std: torch.Tensor,
    q_std_cols: torch.Tensor,
    *,
    b_global_rms: float,
    rel_eps_scale: float,
    abs_weight: float,
    rel_weight: float,
    action_weight: float,
    action_directions: int,
    rng: torch.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    b_pred = j_norm_to_b_phys(j_pred_norm, le_std, q_std_cols)
    b_true = j_norm_to_b_phys(j_true_norm, le_std, q_std_cols)
    b_scale = torch.as_tensor(max(float(b_global_rms), 1.0e-30), dtype=b_pred.dtype, device=b_pred.device)
    eps_b = torch.clamp(torch.as_tensor(float(rel_eps_scale), dtype=b_pred.dtype, device=b_pred.device) * b_scale, min=1.0e-30)
    err = b_pred - b_true
    abs_loss = torch.mean((err / b_scale) ** 2)
    rel_eps_loss = torch.mean((err / (torch.abs(b_true) + eps_b)) ** 2)
    dcount = int(action_directions)
    if dcount > 0 and b_pred.shape[-1] > 0:
        dirs = torch.randn((b_pred.shape[0], dcount, b_pred.shape[-1]), generator=rng, device=b_pred.device, dtype=b_pred.dtype)
        dirs = dirs / torch.clamp(torch.linalg.norm(dirs, dim=-1, keepdim=True), min=1.0e-12)
        err_v = torch.einsum("bpaj,bdj->bpda", err, dirs)
        true_v = torch.einsum("bpaj,bdj->bpda", b_true, dirs)
        action_loss = torch.mean((err_v ** 2) / (true_v ** 2 + eps_b ** 2))
    else:
        action_loss = torch.zeros((), dtype=b_pred.dtype, device=b_pred.device)
    total = float(abs_weight) * abs_loss + float(rel_weight) * rel_eps_loss + float(action_weight) * action_loss
    return total, {"j_loss_abs_normed_mse": abs_loss, "j_loss_rel_eps_mse": rel_eps_loss, "j_loss_action_mse": action_loss}


def evaluate(
    model: nn.Module,
    x_norm: np.ndarray,
    point_norm: np.ndarray,
    le_raw: np.ndarray,
    b_raw: np.ndarray,
    j_norm_target: np.ndarray,
    indices: np.ndarray,
    norms: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    columns: list[int],
    jacobian_method: str,
    prefix: str,
) -> dict[str, Any]:
    base = unwrap_model(model)
    idx = np.asarray(indices, dtype=np.int64)
    le_rows: list[np.ndarray] = []
    jn_rows: list[np.ndarray] = []
    bp_rows: list[np.ndarray] = []
    q_start = int(np.asarray(norms.get("q_start", 4)).reshape(-1)[0])
    q_dim = int(np.asarray(norms.get("q_dim", 48)).reshape(-1)[0])
    q_std = norms["x_std"].reshape(-1)[q_start : q_start + q_dim].astype(np.float32)
    le_std = norms["le_std"].astype(np.float32)
    q_std_cols = q_std[np.asarray(columns, dtype=np.int64)]
    model.eval()
    for start in range(0, idx.size, int(batch_size)):
        sub = idx[start : start + int(batch_size)]
        xb = torch.as_tensor(x_norm[sub], dtype=torch.float32, device=device)
        pb = torch.as_tensor(point_norm[sub], dtype=torch.float32, device=device)
        with torch.no_grad():
            le_norm = model(xb, pb).detach().cpu().numpy()
        le_rows.append(le_norm * le_std + norms["le_mean"])
        with torch.enable_grad():
            jn = ad_jacobian(base, xb, pb, columns, create_graph=False, method=jacobian_method)
        jn_np = jn.detach().cpu().numpy().astype(np.float64)
        jn_rows.append(jn_np)
        b_phys = jn_np * le_std.reshape(1, le_std.shape[1], 6, 1) / q_std_cols.reshape(1, 1, 1, -1)
        bp_rows.append(b_phys)
    le_pred = np.concatenate(le_rows, axis=0).astype(np.float64)
    j_pred = np.concatenate(jn_rows, axis=0).astype(np.float64)
    b_pred = np.concatenate(bp_rows, axis=0).astype(np.float64)
    le_true = le_raw[idx].astype(np.float64, copy=False)
    j_true = j_norm_target[idx][:, :, :, columns].astype(np.float64, copy=False)
    b_true = b_raw[idx][:, :, :, columns].astype(np.float64, copy=False)
    return {
        f"{prefix}_frames": int(idx.size),
        f"{prefix}_LE_rel": rel_np(le_pred, le_true),
        f"{prefix}_LE_cos": cos_np(le_pred, le_true),
        f"{prefix}_AD_B_norm_rel": rel_np(j_pred, j_true),
        f"{prefix}_AD_B_norm_cos": cos_np(j_pred, j_true),
        f"{prefix}_AD_B_rel": rel_np(b_pred, b_true),
        f"{prefix}_AD_B_cos": cos_np(b_pred, b_true),
    }


def write_loss_history(out_dir: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    keys = [
        "epoch", "loss", "le_loss_norm_mse", "j_loss_objective", "j_loss_norm_mse",
        "j_loss_abs_normed_mse", "j_loss_rel_eps_mse", "j_loss_action_mse", "lambda_j",
        "train_LE_rel", "train_AD_B_rel", "train_AD_B_norm_rel", "val_LE_rel", "val_AD_B_rel", "score",
    ]
    with (out_dir / "loss_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in keys})


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> dict[str, Any] | None:
    text = str(checkpoint_path).strip()
    if not text:
        return None
    path = Path(text).resolve()
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt.get("model", ckpt))
    own = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in own and tuple(own[k].shape) == tuple(v.shape)}
    model.load_state_dict(compatible, strict=False)
    return {"init_checkpoint": str(path), "loaded_tensors": int(len(compatible)), "skipped_or_missing_tensors": int(len(own) - len(compatible))}


def compact_paths_from_args(args: argparse.Namespace) -> list[str]:
    paths = [str(v) for v in args.compact]
    if str(args.compact_list).strip():
        with Path(args.compact_list).open("r", encoding="utf-8") as f:
            paths.extend([line.strip() for line in f if line.strip() and not line.strip().startswith("#")])
    if not paths:
        raise ValueError("provide --compact or --compact-list")
    return paths


def train(args: argparse.Namespace) -> dict[str, Any]:
    ctx = distributed_context(args)
    rank = int(ctx["rank"])
    is_main = bool(ctx["is_main"])
    np.random.seed(int(args.seed) + rank)
    torch.manual_seed(int(args.seed) + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed) + rank)
    out_dir = Path(args.out_dir).resolve()
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    ddp_barrier(ctx)

    target_ips = parse_target_ips(str(args.target_ips))
    data = load_compacts(compact_paths_from_args(args), frame_stride=int(args.frame_stride), max_frames_per_compact=int(args.max_frames_per_compact))
    train_idx, val_idx = split_indices(data, float(args.val_fraction), int(args.seed), str(args.val_cases))
    if int(args.max_eval_frames) > 0:
        train_eval_idx = train_idx[: min(train_idx.size, int(args.max_eval_frames))]
        val_eval_idx = val_idx[: min(val_idx.size, int(args.max_eval_frames))]
    else:
        train_eval_idx = train_idx
        val_eval_idx = val_idx

    x_raw = np.concatenate([data.shape4.astype(np.float32), data.q48_raw.astype(np.float32)], axis=1)
    le_raw = data.le[:, target_ips, :].astype(np.float32)
    b_raw = data.b[:, target_ips, :, :].astype(np.float32)
    x_mean, x_std = stats(x_raw[train_idx], axis=0)
    le_mean, le_std = stats(le_raw[train_idx], axis=0)
    q_std = x_std.reshape(-1)[4:52]
    j_norm_target = (b_raw * q_std.reshape(1, 1, 1, 48) / le_std.reshape(1, len(target_ips), 6, 1)).astype(np.float32)
    skip_init = np.mean(j_norm_target[train_idx], axis=0).astype(np.float32)
    b_global_rms = float(np.sqrt(np.mean(np.asarray(b_raw, dtype=np.float64)[train_idx] ** 2)))

    point_all, point_meta = build_point_features(data.shape4, include_id_features=bool(args.include_id_features))
    point_raw = point_all[:, target_ips, :].astype(np.float32)
    point_mean, point_std = stats(point_raw[train_idx].reshape(-1, point_raw.shape[-1]), axis=0)
    point_norm = ((point_raw - point_mean.reshape(1, 1, -1)) / point_std.reshape(1, 1, -1)).astype(np.float32)
    x_norm = ((x_raw - x_mean) / x_std).astype(np.float32)
    le_norm = ((le_raw - le_mean) / le_std).astype(np.float32)
    norms = {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "le_mean": le_mean.astype(np.float32),
        "le_std": le_std.astype(np.float32),
        "point_mean": point_mean.astype(np.float32),
        "point_std": point_std.astype(np.float32),
        "q_start": np.asarray(4, dtype=np.int64),
        "q_dim": np.asarray(48, dtype=np.int64),
    }

    if bool(args.cuda) and torch.cuda.is_available():
        if bool(ctx["distributed"]):
            torch.cuda.set_device(int(ctx["local_rank"]))
            device = torch.device("cuda", int(ctx["local_rank"]))
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_columns_all = parse_int_list(str(args.jacobian_columns), default=list(range(48)))
    eval_columns = parse_int_list(str(args.eval_columns), default=train_columns_all)
    if not eval_columns:
        eval_columns = list(train_columns_all)

    train_set = SobolevArrayDataset(x_norm, point_norm, le_norm, j_norm_target, train_idx)
    train_sampler = DistributedSampler(train_set, num_replicas=int(ctx["world_size"]), rank=rank, shuffle=True, seed=int(args.seed)) if bool(ctx["distributed"]) else None
    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=train_sampler is None, sampler=train_sampler, drop_last=False)

    model = True176Shape4QrawDeepONet(
        input_dim=52,
        point_dim=int(point_norm.shape[-1]),
        ip_count=len(target_ips),
        basis_dim=int(args.basis_dim),
        hidden_dim=int(args.hidden_dim),
        branch_depth=int(args.branch_depth),
        trunk_depth=int(args.trunk_depth),
        activation=str(args.activation),
        skip_init=torch.as_tensor(skip_init, dtype=torch.float32),
        train_skip=not bool(args.freeze_skip),
    ).to(device)
    init_report = load_checkpoint(model, str(args.init_checkpoint))
    if bool(ctx["distributed"]):
        model = DDP(model, device_ids=[int(ctx["local_rank"])] if device.type == "cuda" else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(args.lr_decay))
    col_rng = np.random.default_rng(int(args.seed) + 311 + 1009 * rank)
    action_rng = torch.Generator(device=device)
    action_rng.manual_seed(int(args.seed) + 911 + 1009 * rank)
    le_std_t = torch.as_tensor(le_std, dtype=torch.float32, device=device)
    q_std_np = x_std.reshape(-1)[4:52].astype(np.float32)

    if is_main:
        write_json(
            out_dir / "config.json",
            {
                "args": vars(args),
                "target_ips": target_ips,
                "compact_paths": data.compact_paths,
                "train_frames": int(train_idx.size),
                "val_frames": int(val_idx.size),
                "point_meta": point_meta,
                "b_global_rms": b_global_rms,
                "init_report": init_report,
            },
        )

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        sums = {"loss": 0.0, "le": 0.0, "j": 0.0, "j_norm": 0.0, "j_abs": 0.0, "j_rel": 0.0, "j_action": 0.0}
        count = 0
        t0 = time.time()
        for xb, pb, leb, jb in train_loader:
            xb = xb.to(device)
            pb = pb.to(device)
            leb = leb.to(device)
            jb = jb.to(device)
            optimizer.zero_grad(set_to_none=True)
            le_pred = model(xb, pb)
            le_loss = nn.functional.mse_loss(le_pred, leb)
            columns = sample_columns(train_columns_all, int(args.jacobian_columns_per_batch), col_rng)
            if columns:
                # Use the DDP wrapper during training so the Sobolev-AD branch participates in normal DDP gradient synchronization.
                j_pred = ad_jacobian(model, xb, pb, columns, create_graph=True, method=str(args.jacobian_method))
                j_true = jb[:, :, :, columns]
                j_norm_loss = nn.functional.mse_loss(j_pred, j_true)
                q_std_cols = torch.as_tensor(q_std_np[np.asarray(columns, dtype=np.int64)], dtype=torch.float32, device=device)
                j_phys_loss, j_parts = physical_j_loss(
                    j_pred,
                    j_true,
                    le_std_t,
                    q_std_cols,
                    b_global_rms=b_global_rms,
                    rel_eps_scale=float(args.physical_j_rel_eps_scale),
                    abs_weight=float(args.physical_j_abs_weight),
                    rel_weight=float(args.physical_j_rel_weight),
                    action_weight=float(args.physical_j_action_weight),
                    action_directions=int(args.physical_j_action_directions),
                    rng=action_rng,
                )
                if str(args.j_loss_mode) == "norm":
                    j_obj = j_norm_loss
                elif str(args.j_loss_mode) == "physical":
                    j_obj = j_phys_loss
                else:
                    j_obj = j_norm_loss + float(args.physical_j_aux_weight) * j_phys_loss
            else:
                j_norm_loss = torch.zeros((), dtype=xb.dtype, device=device)
                j_obj = torch.zeros((), dtype=xb.dtype, device=device)
                j_parts = {"j_loss_abs_normed_mse": j_obj, "j_loss_rel_eps_mse": j_obj, "j_loss_action_mse": j_obj}
            lambda_j = float(args.initial_jacobian_weight)
            loss = le_loss + lambda_j * j_obj
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            batch_n = int(xb.shape[0])
            count += batch_n
            sums["loss"] += float(loss.detach().cpu()) * batch_n
            sums["le"] += float(le_loss.detach().cpu()) * batch_n
            sums["j"] += float(j_obj.detach().cpu()) * batch_n
            sums["j_norm"] += float(j_norm_loss.detach().cpu()) * batch_n
            sums["j_abs"] += float(j_parts["j_loss_abs_normed_mse"].detach().cpu()) * batch_n
            sums["j_rel"] += float(j_parts["j_loss_rel_eps_mse"].detach().cpu()) * batch_n
            sums["j_action"] += float(j_parts["j_loss_action_mse"].detach().cpu()) * batch_n
        scheduler.step()
        denom = max(count, 1)
        row: dict[str, Any] = {
            "epoch": epoch,
            "loss": ddp_mean(sums["loss"] / denom, device, ctx),
            "le_loss_norm_mse": ddp_mean(sums["le"] / denom, device, ctx),
            "j_loss_objective": ddp_mean(sums["j"] / denom, device, ctx),
            "j_loss_norm_mse": ddp_mean(sums["j_norm"] / denom, device, ctx),
            "j_loss_abs_normed_mse": ddp_mean(sums["j_abs"] / denom, device, ctx),
            "j_loss_rel_eps_mse": ddp_mean(sums["j_rel"] / denom, device, ctx),
            "j_loss_action_mse": ddp_mean(sums["j_action"] / denom, device, ctx),
            "lambda_j": lambda_j,
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - t0,
        }
        do_eval = epoch == 1 or epoch % int(args.eval_every) == 0 or epoch == int(args.epochs)
        if do_eval and is_main:
            eval_model = unwrap_model(model)
            row.update(evaluate(eval_model, x_norm, point_norm, le_raw, b_raw, j_norm_target, train_eval_idx, norms, device, int(args.eval_batch_size), eval_columns, str(args.jacobian_method), "train"))
            row.update(evaluate(eval_model, x_norm, point_norm, le_raw, b_raw, j_norm_target, val_eval_idx, norms, device, int(args.eval_batch_size), eval_columns, str(args.jacobian_method), "val"))
            row["score"] = float(row.get("val_LE_rel", row["loss"])) + float(row.get("val_AD_B_rel", 0.0))
            if float(row["score"]) < best_score:
                best_score = float(row["score"])
                torch.save(
                    {
                        "model_state": eval_model.state_dict(),
                        "norms": norms,
                        "args": vars(args),
                        "target_ips": target_ips,
                        "point_meta": point_meta,
                        "best_score": best_score,
                        "epoch": epoch,
                    },
                    out_dir / "best.pt",
                )
        if is_main:
            history.append(row)
            write_json(out_dir / "latest_metrics.json", row)
            write_loss_history(out_dir, history)
            if epoch % int(args.log_every) == 0 or do_eval:
                print(json.dumps(row, sort_keys=True), flush=True)
    if bool(ctx["distributed"]) and dist.is_initialized():
        dist.destroy_process_group()
    return {"best_score": best_score, "out_dir": str(out_dir)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compact", action="append", default=[])
    p.add_argument("--compact-list", default="")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--target-ips", default=",".join(str(i) for i in range(128)))
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames-per-compact", type=int, default=0)
    p.add_argument("--max-eval-frames", type=int, default=512)
    p.add_argument("--basis-dim", type=int, default=96)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--branch-depth", type=int, default=5)
    p.add_argument("--trunk-depth", type=int, default=5)
    p.add_argument("--residual-layers", type=int, default=5, help="Compatibility alias; use --branch-depth/--trunk-depth for DeepONet.")
    p.add_argument("--activation", default="tanh")
    p.add_argument("--include-id-features", action="store_true")
    p.add_argument("--jacobian-columns", default="all")
    p.add_argument("--jacobian-columns-per-batch", type=int, default=8)
    p.add_argument("--jacobian-method", default="forward", choices=["forward", "reverse"])
    p.add_argument("--eval-columns", default="0,1,2,3,4,5,6,7,8,9,10,11")
    p.add_argument("--j-loss-mode", default="norm-plus-physical", choices=["norm", "physical", "norm-plus-physical"])
    p.add_argument("--physical-j-aux-weight", type=float, default=0.05)
    p.add_argument("--physical-j-abs-weight", type=float, default=1.0)
    p.add_argument("--physical-j-rel-weight", type=float, default=0.02)
    p.add_argument("--physical-j-action-weight", type=float, default=0.05)
    p.add_argument("--physical-j-rel-eps-scale", type=float, default=0.02)
    p.add_argument("--physical-j-action-directions", type=int, default=4)
    p.add_argument("--initial-jacobian-weight", type=float, default=1.0)
    p.add_argument("--initial-tangent-weight", type=float, default=0.0, help="Accepted for launcher compatibility; currently unused.")
    p.add_argument("--tangent-directions", type=int, default=0, help="Accepted for launcher compatibility; currently unused.")
    p.add_argument("--lr", type=float, default=8.0e-5)
    p.add_argument("--lr-decay", type=float, default=0.9995)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--weight-decay", type=float, default=1.0e-5)
    p.add_argument("--val-fraction", type=float, default=0.0)
    p.add_argument("--val-cases", default="")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260620)
    p.add_argument("--init-checkpoint", default="")
    p.add_argument("--freeze-skip", action="store_true")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--ddp", action="store_true")
    p.add_argument("--ddp-backend", default="nccl")
    return p.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
