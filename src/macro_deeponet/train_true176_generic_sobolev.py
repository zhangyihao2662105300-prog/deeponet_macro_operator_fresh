"""Train TRUE176/CSS8 DeepONet using generic real integration-point features.

Unlike the legacy shape4-derived route, this trainer does not assume that the
standard 4x4 CSS8 row map is aligned with the Abaqus/real LE-B labels.  By
default it requires point/trunk features stored in the compact data itself.

Accepted point data is documented in ``macro_deeponet.point_features``.
"""

from __future__ import annotations

import argparse
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
from .point_features import load_point_features_from_compacts
from .train_true176_deeponet_sobolev import (
    ad_jacobian,
    compact_paths_from_args,
    ddp_barrier,
    ddp_mean,
    evaluate,
    load_checkpoint,
    physical_j_loss,
    sample_columns,
    write_json,
    write_loss_history,
)
from .true176_data import (
    SobolevArrayDataset,
    load_compacts,
    parse_int_list,
    parse_target_ips,
    split_indices,
    stats,
)


def _distributed_context(args: argparse.Namespace) -> dict[str, Any]:
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
    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_main": rank == 0,
    }


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def train(args: argparse.Namespace) -> dict[str, Any]:
    ctx = _distributed_context(args)
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

    compact_paths = compact_paths_from_args(args)
    target_ips = parse_target_ips(str(args.target_ips))
    data = load_compacts(
        compact_paths,
        frame_stride=int(args.frame_stride),
        max_frames_per_compact=int(args.max_frames_per_compact),
    )
    train_idx, val_idx = split_indices(data, float(args.val_fraction), int(args.seed), str(args.val_cases))
    train_eval_idx = train_idx[: min(train_idx.size, int(args.max_eval_frames))] if int(args.max_eval_frames) > 0 else train_idx
    val_eval_idx = val_idx[: min(val_idx.size, int(args.max_eval_frames))] if int(args.max_eval_frames) > 0 else val_idx

    x_raw = np.concatenate([data.shape4.astype(np.float32), data.q48_raw.astype(np.float32)], axis=1)
    le_raw = data.le[:, target_ips, :].astype(np.float32)
    b_raw = data.b[:, target_ips, :, :].astype(np.float32)

    x_mean, x_std = stats(x_raw[train_idx], axis=0)
    le_mean, le_std = stats(le_raw[train_idx], axis=0)
    q_std = x_std.reshape(-1)[4:52]
    j_norm_target = (b_raw * q_std.reshape(1, 1, 1, 48) / le_std.reshape(1, len(target_ips), 6, 1)).astype(np.float32)
    skip_init = np.mean(j_norm_target[train_idx], axis=0).astype(np.float32)
    b_global_rms = float(np.sqrt(np.mean(np.asarray(b_raw, dtype=np.float64)[train_idx] ** 2)))

    point_raw, point_meta = load_point_features_from_compacts(
        compact_paths=data.compact_paths,
        source_index=data.source_index,
        source_row=data.source_row,
        shape4=data.shape4,
        target_ips=target_ips,
        source=str(args.point_feature_source),
        include_id_features=bool(args.include_id_features),
        allow_shape4_fallback=bool(args.allow_shape4_point_feature_fallback),
    )
    if point_raw.shape[:2] != le_raw.shape[:2]:
        raise ValueError(f"point features {point_raw.shape[:2]} do not align with LE labels {le_raw.shape[:2]}")

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
    train_sampler = (
        DistributedSampler(train_set, num_replicas=int(ctx["world_size"]), rank=rank, shuffle=True, seed=int(args.seed))
        if bool(ctx["distributed"])
        else None
    )
    train_loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
    )

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
                "generic_point_contract": True,
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
            eval_model = _unwrap(model)
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
    p.add_argument("--point-feature-source", default="data", choices=["data", "auto", "shape4-audited", "shape4"])
    p.add_argument("--allow-shape4-point-feature-fallback", action="store_true")
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
    p.add_argument("--initial-tangent-weight", type=float, default=0.0)
    p.add_argument("--tangent-directions", type=int, default=0)
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
