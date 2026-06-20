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

from .models import FELinearResidualDeepONet, NOEMStyleMIONet, True176Shape4QrawDeepONet
from .point_features import load_point_features_from_compacts, transform_point_features_for_scale
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
    build_branch_features,
    canonical_scale_mode,
    load_compacts,
    parse_int_list,
    parse_target_ips,
    split_indices,
    stats,
    transform_b_target_for_q_coordinate,
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


def validate_point_feature_source_for_scale(
    *,
    scale_mode: str,
    requested_source: str,
    point_meta: dict[str, Any],
    allow_physical_shape4_trunk: bool = False,
) -> None:
    """Prevent physical arbitrary geometry from silently using shape4 Trunk fields."""

    if canonical_scale_mode(scale_mode) != "physical":
        return
    requested = str(requested_source).strip().lower().replace("_", "-")
    meta_source = str(point_meta.get("point_feature_source", "")).strip().lower()
    uses_shape4 = requested in {"shape4", "shape4-audited"} or "shape4" in meta_source
    if uses_shape4 and not bool(allow_physical_shape4_trunk):
        raise ValueError(
            "scale_mode=physical cannot use shape4-audited/shape4-generated Trunk features by default. "
            "For arbitrary physical geometry, provide data/raw point fields such as ip_xyz/ip_J/ip_invJ/ip_detJ "
            "or prebuilt dimensionless *_hat point_features. Use --allow-physical-shape4-trunk only when the "
            "explicit physical X_keep/X_macro is exactly an H-scaled TRUE176 shape4 geometry."
        )


def build_model(
    args: argparse.Namespace,
    *,
    input_dim: int,
    point_dim: int,
    ip_count: int,
    q_start: int,
    q_dim: int,
    skip_init: np.ndarray,
) -> nn.Module:
    style = str(args.model_style).strip().lower().replace("_", "-")
    common = {
        "input_dim": int(input_dim),
        "point_dim": int(point_dim),
        "ip_count": int(ip_count),
        "basis_dim": int(args.basis_dim),
        "hidden_dim": int(args.hidden_dim),
        "branch_depth": int(args.branch_depth),
        "trunk_depth": int(args.trunk_depth),
        "activation": str(args.activation),
        "skip_init": torch.as_tensor(skip_init, dtype=torch.float32),
        "train_skip": not bool(args.freeze_skip),
        "residual_scale": float(args.residual_scale),
        "q_start": int(q_start),
        "q_dim": int(q_dim),
    }
    if style in {"fe-linear-residual", "fe-residual", "linear-residual"}:
        return FELinearResidualDeepONet(
            **common,
            baseline_scale=float(args.fe_baseline_scale),
            train_point_baseline=not bool(args.freeze_fe_point_baseline),
            zero_init_residual=not bool(args.no_zero_init_residual),
        )
    if style in {"noem-mionet", "mionet", "noem"}:
        common.pop("residual_scale")
        return NOEMStyleMIONet(
            **common,
            use_q_skip=bool(args.use_q_skip),
            product_scale=str(args.mionet_product_scale),
        )
    if style in {"concat-skip", "legacy", "concat"}:
        return True176Shape4QrawDeepONet(**common)
    raise ValueError("model_style must be one of: fe-linear-residual, noem-mionet, concat-skip")


def model_meta(model: nn.Module, branch_meta: dict[str, Any]) -> dict[str, Any]:
    base = _unwrap(model)
    if isinstance(base, FELinearResidualDeepONet):
        style = "fe-linear-residual"
    elif isinstance(base, NOEMStyleMIONet):
        style = "noem-mionet"
    else:
        style = "concat-skip"
    meta = {
        "model_style": style,
        "input_dim": int(getattr(base, "input_dim")),
        "point_dim": int(getattr(base, "point_dim")),
        "ip_count": int(getattr(base, "ip_count")),
        "q_start": int(getattr(base, "q_start")),
        "q_dim": int(getattr(base, "q_dim")),
        "basis_dim": int(getattr(base, "basis_dim")),
        "strain_dim": int(getattr(base, "strain_dim")),
    }
    if isinstance(base, FELinearResidualDeepONet):
        meta.update(
            {
                "architecture_principle": (
                    "FE-like linear B baseline plus standard branch/trunk DeepONet residual."
                ),
                "linear_baseline": "B_base_norm(point_features) @ q48_norm",
                "static_baseline": "initialized from mean training d(LE_norm)/d(q48_norm)",
                "point_baseline": "learned from Trunk/IP features only",
                "residual": "standard single-branch DeepONet over [q48, geometry] and point features",
                "baseline_scale": float(base.baseline_scale),
                "residual_scale": float(base.residual_scale),
                "train_point_baseline": bool(base.train_point_baseline),
                "zero_init_residual": bool(base.zero_init_residual),
            }
        )
    elif isinstance(base, NOEMStyleMIONet):
        meta.update(
            {
                "architecture_principle": (
                    "NOEM/MIONet: separate q and geometry branches are multiplied in latent space, "
                    "then segmented-dot-producted with the trunk."
                ),
                "branch_q": f"q48 slice [{base.q_start}:{base.q_start + base.q_dim}]",
                "branch_geometry": f"remaining branch coordinates, dim={base.geometry_dim}",
                "geometry_input": branch_meta.get("geometry_input"),
                "use_q_skip": bool(base.use_q_skip),
                "product_scale": str(base.product_scale),
            }
        )
    else:
        meta.update(
            {
                "architecture_principle": (
                    "Standard single-branch DeepONet with a learned q-linear baseline and "
                    "branch/trunk residual."
                ),
                "branch": "single MLP over the full standardized branch vector [q48, geometry]",
                "use_q_skip": True,
                "residual_scale": float(base.residual_scale),
            }
        )
    return meta


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
    scale_mode = canonical_scale_mode(str(args.scale_mode))
    data = load_compacts(
        compact_paths,
        frame_stride=int(args.frame_stride),
        max_frames_per_compact=int(args.max_frames_per_compact),
        target_ips=target_ips,
    )
    if scale_mode == "physical" and "implicit_H_1" in str(data.length_scale_source).split("+"):
        raise ValueError(
            "scale_mode=physical requires an explicit H/length_scale field in every compact. "
            "Use scale_mode=normalized for already dimensionless TRUE176 data."
        )
    train_idx, val_idx = split_indices(data, float(args.val_fraction), int(args.seed), str(args.val_cases))
    train_eval_idx = train_idx[: min(train_idx.size, int(args.max_eval_frames))] if int(args.max_eval_frames) > 0 else train_idx
    val_eval_idx = val_idx[: min(val_idx.size, int(args.max_eval_frames))] if int(args.max_eval_frames) > 0 else val_idx

    x_raw, branch_meta = build_branch_features(
        shape4=data.shape4,
        q48_raw=data.q48_raw,
        mode=str(args.branch_feature_mode),
        keep_node_coords=data.keep_node_coords,
        macro_nodes=data.macro_nodes,
        length_scale=data.length_scale,
        scale_mode=scale_mode,
    )
    le_raw = data.le.astype(np.float32, copy=False)
    b_raw = data.b.astype(np.float32, copy=False)
    if le_raw.shape[1] != len(target_ips) or b_raw.shape[1] != len(target_ips):
        raise ValueError(
            f"loaded LE/B point count ({le_raw.shape[1]}, {b_raw.shape[1]}) "
            f"does not match target_ips count {len(target_ips)}"
        )
    b_train, b_scale_meta = transform_b_target_for_q_coordinate(
        b_raw,
        data.length_scale,
        scale_mode=scale_mode,
        b_label_coordinate=str(args.b_label_coordinate),
    )

    x_mean, x_std = stats(x_raw[train_idx], axis=0)
    le_mean, le_std = stats(le_raw[train_idx], axis=0)
    q_start = int(branch_meta["q_start"])
    q_dim = int(branch_meta["q_dim"])
    q_std = x_std.reshape(-1)[q_start : q_start + q_dim]
    j_norm_target = (b_train * q_std.reshape(1, 1, 1, 48) / le_std.reshape(1, len(target_ips), 6, 1)).astype(np.float32)
    skip_init = np.mean(j_norm_target[train_idx], axis=0).astype(np.float32)
    b_global_rms = float(np.sqrt(np.mean(np.asarray(b_train, dtype=np.float64)[train_idx] ** 2)))

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
    validate_point_feature_source_for_scale(
        scale_mode=scale_mode,
        requested_source=str(args.point_feature_source),
        point_meta=point_meta,
        allow_physical_shape4_trunk=bool(args.allow_physical_shape4_trunk),
    )
    point_feature_names = list(point_meta.get("feature_names", [f"point_feature_{i}" for i in range(point_raw.shape[-1])]))
    point_raw, point_scale_meta = transform_point_features_for_scale(
        point_raw,
        point_feature_names,
        data.length_scale,
        scale_mode=scale_mode,
        detj_scale_dim=int(args.detj_scale_dim),
    )
    point_meta = {**point_meta, "scale_transform": point_scale_meta}
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
        "q_start": np.asarray(q_start, dtype=np.int64),
        "q_dim": np.asarray(q_dim, dtype=np.int64),
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

    model = build_model(
        args,
        input_dim=int(x_norm.shape[-1]),
        point_dim=int(point_norm.shape[-1]),
        ip_count=len(target_ips),
        q_start=q_start,
        q_dim=q_dim,
        skip_init=skip_init,
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
    q_std_np = x_std.reshape(-1)[q_start : q_start + q_dim].astype(np.float32)

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
                "branch_meta": branch_meta,
                "model_meta": model_meta(model, branch_meta),
                "scale_meta": {
                    "scale_mode": scale_mode,
                    "length_scale_source": data.length_scale_source,
                    "length_scale_min": float(np.min(data.length_scale)),
                    "length_scale_max": float(np.max(data.length_scale)),
                    "q_branch_coordinate": branch_meta.get("q_coordinate"),
                    "b_label_coordinate": str(args.b_label_coordinate),
                    "b_scale_meta": b_scale_meta,
                    "physical_normalization": (
                        "normalized: identity; physical: q_hat=q/H, X_hat=X/H, "
                        "J_hat=J/H, invJ_hat=H*invJ, detJ_hat=detJ/H^detj_scale_dim, B_hat=H*B_phys"
                    ),
                },
                "b_global_rms": b_global_rms,
                "jacobian_target_relation": "J_norm = B_train * q_std / LE_std; B_train = J_norm * LE_std / q_std",
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
        sums = {
            "loss": 0.0,
            "le": 0.0,
            "j": 0.0,
            "j_norm": 0.0,
            "j_abs": 0.0,
            "j_rel": 0.0,
            "j_action": 0.0,
            "baseline_j": 0.0,
            "baseline_j_norm": 0.0,
            "baseline_j_abs": 0.0,
            "baseline_j_rel": 0.0,
            "baseline_j_action": 0.0,
        }
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
                baseline_j_obj = torch.zeros((), dtype=xb.dtype, device=device)
                baseline_j_norm_loss = torch.zeros((), dtype=xb.dtype, device=device)
                baseline_j_parts = {
                    "j_loss_abs_normed_mse": baseline_j_obj,
                    "j_loss_rel_eps_mse": baseline_j_obj,
                    "j_loss_action_mse": baseline_j_obj,
                }
                base_model = _unwrap(model)
                if isinstance(base_model, FELinearResidualDeepONet) and float(args.baseline_jacobian_weight) > 0.0:
                    b_base = base_model._linear_b_norm(pb)[:, :, :, columns]
                    baseline_j_norm_loss = nn.functional.mse_loss(b_base, j_true)
                    baseline_j_phys_loss, baseline_j_parts = physical_j_loss(
                        b_base,
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
                    if str(args.baseline_j_loss_mode) == "norm":
                        baseline_j_obj = baseline_j_norm_loss
                    elif str(args.baseline_j_loss_mode) == "physical":
                        baseline_j_obj = baseline_j_phys_loss
                    else:
                        baseline_j_obj = baseline_j_norm_loss + float(args.physical_j_aux_weight) * baseline_j_phys_loss
            else:
                j_norm_loss = torch.zeros((), dtype=xb.dtype, device=device)
                j_obj = torch.zeros((), dtype=xb.dtype, device=device)
                j_parts = {"j_loss_abs_normed_mse": j_obj, "j_loss_rel_eps_mse": j_obj, "j_loss_action_mse": j_obj}
                baseline_j_obj = torch.zeros((), dtype=xb.dtype, device=device)
                baseline_j_norm_loss = torch.zeros((), dtype=xb.dtype, device=device)
                baseline_j_parts = {
                    "j_loss_abs_normed_mse": baseline_j_obj,
                    "j_loss_rel_eps_mse": baseline_j_obj,
                    "j_loss_action_mse": baseline_j_obj,
                }
            lambda_j = float(args.initial_jacobian_weight)
            loss = le_loss + lambda_j * j_obj + float(args.baseline_jacobian_weight) * baseline_j_obj
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
            sums["baseline_j"] += float(baseline_j_obj.detach().cpu()) * batch_n
            sums["baseline_j_norm"] += float(baseline_j_norm_loss.detach().cpu()) * batch_n
            sums["baseline_j_abs"] += float(baseline_j_parts["j_loss_abs_normed_mse"].detach().cpu()) * batch_n
            sums["baseline_j_rel"] += float(baseline_j_parts["j_loss_rel_eps_mse"].detach().cpu()) * batch_n
            sums["baseline_j_action"] += float(baseline_j_parts["j_loss_action_mse"].detach().cpu()) * batch_n

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
            "baseline_j_loss_objective": ddp_mean(sums["baseline_j"] / denom, device, ctx),
            "baseline_j_loss_norm_mse": ddp_mean(sums["baseline_j_norm"] / denom, device, ctx),
            "baseline_j_loss_abs_normed_mse": ddp_mean(sums["baseline_j_abs"] / denom, device, ctx),
            "baseline_j_loss_rel_eps_mse": ddp_mean(sums["baseline_j_rel"] / denom, device, ctx),
            "baseline_j_loss_action_mse": ddp_mean(sums["baseline_j_action"] / denom, device, ctx),
            "lambda_j": lambda_j,
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - t0,
        }
        do_eval = epoch == 1 or epoch % int(args.eval_every) == 0 or epoch == int(args.epochs)
        if do_eval and is_main:
            eval_model = _unwrap(model)
            row.update(evaluate(eval_model, x_norm, point_norm, le_raw, b_train, j_norm_target, train_eval_idx, norms, device, int(args.eval_batch_size), eval_columns, str(args.jacobian_method), "train"))
            row.update(evaluate(eval_model, x_norm, point_norm, le_raw, b_train, j_norm_target, val_eval_idx, norms, device, int(args.eval_batch_size), eval_columns, str(args.jacobian_method), "val"))
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
                        "branch_meta": branch_meta,
                        "model_meta": model_meta(eval_model, branch_meta),
                        "scale_meta": {
                            "scale_mode": scale_mode,
                            "length_scale_source": data.length_scale_source,
                            "b_scale_meta": b_scale_meta,
                        },
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
    p.add_argument("--branch-feature-mode", default="xkeep-qraw", choices=["shape4-qraw", "xkeep-qraw", "xnodes-qraw"])
    p.add_argument("--point-feature-source", default="data", choices=["data", "auto", "shape4-audited", "shape4"])
    p.add_argument("--allow-shape4-point-feature-fallback", action="store_true")
    p.add_argument("--allow-physical-shape4-trunk", action="store_true")
    p.add_argument("--scale-mode", default="normalized", choices=["normalized", "physical"])
    p.add_argument("--b-label-coordinate", default="auto", choices=["auto", "physical", "dimensionless"])
    p.add_argument("--detj-scale-dim", type=int, default=3)
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
    p.add_argument("--model-style", default="fe-linear-residual", choices=["fe-linear-residual", "noem-mionet", "concat-skip"])
    p.add_argument("--mionet-product-scale", default="none", choices=["none", "sqrt", "basis"])
    p.add_argument("--use-q-skip", action="store_true")
    p.add_argument("--residual-scale", type=float, default=1.0)
    p.add_argument("--fe-baseline-scale", type=float, default=1.0)
    p.add_argument("--freeze-fe-point-baseline", action="store_true")
    p.add_argument("--no-zero-init-residual", action="store_true")
    p.add_argument("--baseline-jacobian-weight", type=float, default=1.0)
    p.add_argument("--baseline-j-loss-mode", default="norm-plus-physical", choices=["norm", "physical", "norm-plus-physical"])
    p.add_argument("--include-id-features", action="store_true")
    p.add_argument("--jacobian-columns", default="all")
    p.add_argument("--jacobian-columns-per-batch", type=int, default=8)
    p.add_argument("--jacobian-method", default="forward", choices=["forward", "reverse"])
    p.add_argument("--eval-columns", default="0,1,2,3,4,5,6,7,8,9,10,11")
    p.add_argument("--j-loss-mode", default="norm-plus-physical", choices=["norm", "physical", "norm-plus-physical"])
    p.add_argument("--physical-j-aux-weight", type=float, default=0.5)
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
