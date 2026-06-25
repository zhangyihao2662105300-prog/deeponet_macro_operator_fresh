#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose Macro16 tiny-overfit AD-B failure modes.

This script is read-only with respect to model code and compact data.  It does
not train a model, change q48/LE ordering, or change the 128-point rule.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macro_deeponet.macro16_geometry import macro16_source128_point_table  # noqa: E402
from macro_deeponet.models import Macro16BoundaryDeepONetWithLE0  # noqa: E402
from macro_deeponet.train_macro16_boundary_sobolev import (  # noqa: E402
    _le_std_scale,
    build_physical_b_loss_scale,
    compact_paths_from_args,
    j_norm_to_b_qhat_torch,
    load_macro16_compacts,
    physical_balanced_b_loss_from_j_norm,
    read_path_list,
    split_indices,
    standardize,
)
from macro_deeponet.train_true176_deeponet_sobolev import ad_jacobian  # noqa: E402
from macro_deeponet.true176_data import stats  # noqa: E402


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return str(obj)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def rel_np(pred: np.ndarray, true: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(true, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(p - t) / max(float(np.linalg.norm(t)), 1.0e-300))


def cos_np(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    den = max(float(np.linalg.norm(aa) * np.linalg.norm(bb)), 1.0e-300)
    return float(np.dot(aa, bb) / den)


def grad_norms(model: nn.Module) -> dict[str, dict[str, float]]:
    groups = {
        "static_b_norm": [],
        "point_b_net": [],
        "branch": [],
        "trunk": [],
        "le0": [],
        "other": [],
    }
    for name, param in model.named_parameters():
        if param.grad is None:
            norm = 0.0
        else:
            norm = float(torch.linalg.norm(param.grad.detach()).cpu())
        if name == "static_b_norm" or name == "global_b_norm":
            key = "static_b_norm"
        elif name.startswith("point_b_net."):
            key = "point_b_net"
        elif name.startswith("branch."):
            key = "branch"
        elif name.startswith("trunk."):
            key = "trunk"
        elif name.startswith("point_le0_net.") or name == "static_le0_norm":
            key = "le0"
        else:
            key = "other"
        groups[key].append(norm)
    out: dict[str, dict[str, float]] = {}
    for key, vals in groups.items():
        arr = np.asarray(vals, dtype=np.float64)
        out[key] = {
            "param_tensors": int(arr.size),
            "nonzero_grad_tensors": int(np.count_nonzero(arr > 0.0)),
            "grad_norm_sum": float(np.sum(arr)) if arr.size else 0.0,
            "grad_norm_max": float(np.max(arr)) if arr.size else 0.0,
        }
    return out


def rank_report(q: np.ndarray) -> dict[str, Any]:
    vals = np.asarray(q, dtype=np.float64)
    centered = vals - np.mean(vals, axis=0, keepdims=True)
    s_raw = np.linalg.svd(vals, compute_uv=False)
    s_ctr = np.linalg.svd(centered, compute_uv=False)

    def pack(s: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
        tol = max(ref.shape) * np.finfo(np.float64).eps * (float(s[0]) if s.size else 0.0)
        energy = s * s
        total = float(np.sum(energy))
        return {
            "rank": int(np.sum(s > tol)),
            "tol": float(tol),
            "singular_values": s[: min(8, s.size)].tolist(),
            "energy_share": (energy[: min(8, s.size)] / max(total, 1.0e-300)).tolist(),
            "condition_nonzero": float(s[0] / max(float(s[np.sum(s > tol) - 1]), 1.0e-300)) if np.sum(s > tol) else None,
        }

    return {
        "frame_count": int(vals.shape[0]),
        "dof_count": int(vals.shape[1]),
        "raw": pack(s_raw, vals),
        "centered": pack(s_ctr, centered),
        "per_column_std_min": float(np.min(np.std(vals, axis=0))),
        "per_column_std_median": float(np.median(np.std(vals, axis=0))),
        "per_column_std_max": float(np.max(np.std(vals, axis=0))),
    }


def b_variation_report(b: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    vals = np.asarray(b, dtype=np.float64)
    mean_b = np.mean(vals, axis=0, keepdims=True)
    centered = vals - mean_b
    qn = np.asarray(q, dtype=np.float64)
    q_center = qn - np.mean(qn, axis=0, keepdims=True)
    return {
        "B_rms": float(np.sqrt(np.mean(vals * vals))),
        "B_abs_max": float(np.max(np.abs(vals))),
        "constant_B_mean_rel": rel_np(np.broadcast_to(mean_b, vals.shape), vals),
        "B_centered_rel_to_B": rel_np(centered, vals),
        "B_frame_norms": np.linalg.norm(vals.reshape(vals.shape[0], -1), axis=1).tolist(),
        "B_centered_frame_norms": np.linalg.norm(centered.reshape(vals.shape[0], -1), axis=1).tolist(),
        "q_frame_norms": np.linalg.norm(qn, axis=1).tolist(),
        "q_centered_frame_norms": np.linalg.norm(q_center, axis=1).tolist(),
    }


def build_model(
    *,
    branch_norm: np.ndarray,
    point_norm: np.ndarray,
    branch_mean: np.ndarray,
    branch_std: np.ndarray,
    le_mean: np.ndarray,
    le_std: np.ndarray,
    j_norm_target: np.ndarray,
    le0_init_norm: np.ndarray,
    train_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Macro16BoundaryDeepONetWithLE0:
    skip_init = np.mean(j_norm_target[train_idx], axis=0).astype(np.float32)
    if bool(args.global_b_prior):
        skip_init = np.mean(skip_init, axis=0)
    return Macro16BoundaryDeepONetWithLE0(
        input_dim=int(branch_norm.shape[-1]),
        point_dim=int(point_norm.shape[-1]),
        ip_count=int(point_norm.shape[1]),
        q_start=0,
        q_dim=48,
        basis_dim=int(args.basis_dim),
        hidden_dim=int(args.hidden_dim),
        branch_depth=int(args.branch_depth),
        trunk_depth=int(args.trunk_depth),
        activation="tanh",
        skip_init=torch.as_tensor(skip_init, dtype=torch.float32),
        train_skip=not bool(args.freeze_skip),
        residual_scale=float(args.residual_scale),
        baseline_scale=float(args.fe_baseline_scale),
        train_point_baseline=not bool(args.freeze_fe_point_baseline),
        q_zero_norm=((0.0 - branch_mean.reshape(-1)[:48]) / branch_std.reshape(-1)[:48]).astype(np.float32),
        q_raw_mean=branch_mean.reshape(-1)[:48].astype(np.float32),
        q_raw_std=branch_std.reshape(-1)[:48].astype(np.float32),
        gate_q0=float(args.anchored_residual_gate_q0),
        le0_init_norm=torch.as_tensor(le0_init_norm, dtype=torch.float32),
        le0_scale=float(args.le0_scale),
        train_le0_static=not bool(args.freeze_le0_static),
        train_le0_point=not bool(args.freeze_le0_point),
    ).to(device)


def b_from_j(j: np.ndarray, le_std: np.ndarray, q_std: np.ndarray) -> np.ndarray:
    return np.asarray(j, dtype=np.float64) * _le_std_scale(le_std, j.shape[1]) / q_std.reshape(1, 1, 1, -1)


def evaluate_decomposition(
    model: Macro16BoundaryDeepONetWithLE0,
    *,
    branch_norm: np.ndarray,
    point_norm: np.ndarray,
    le_mean: np.ndarray,
    le_std: np.ndarray,
    q_std: np.ndarray,
    b_true: np.ndarray,
    le_true: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    xb = torch.as_tensor(branch_norm[indices], dtype=torch.float32, device=device)
    pb = torch.as_tensor(point_norm[indices], dtype=torch.float32, device=device)
    cols = list(range(48))
    with torch.no_grad():
        pred_norm = model(xb, pb).detach().cpu().numpy()
        b_base_norm = model._linear_b_norm(pb).detach().cpu().numpy()
    with torch.enable_grad():
        j_full = ad_jacobian(model, xb, pb, cols, create_graph=False, method="forward").detach().cpu().numpy()
    b_full = b_from_j(j_full, le_std, q_std)
    b_base = b_from_j(b_base_norm, le_std, q_std)
    b_residual = b_full - b_base
    le_pred = pred_norm * le_std + le_mean
    target_b = np.asarray(b_true[indices], dtype=np.float64)
    target_le = np.asarray(le_true[indices], dtype=np.float64)
    return {
        "LE_rel": rel_np(le_pred, target_le),
        "LE_cos": cos_np(le_pred, target_le),
        "full_B_rel": rel_np(b_full, target_b),
        "full_B_cos": cos_np(b_full, target_b),
        "linear_baseline_B_rel": rel_np(b_base, target_b),
        "linear_baseline_B_cos": cos_np(b_base, target_b),
        "residual_B_norm_over_target": float(np.linalg.norm(b_residual.reshape(-1)) / max(float(np.linalg.norm(target_b.reshape(-1))), 1.0e-300)),
        "full_minus_baseline_B_norm_over_target": float(np.linalg.norm((b_full - b_base).reshape(-1)) / max(float(np.linalg.norm(target_b.reshape(-1))), 1.0e-300)),
    }


def load_checkpoint_into_model(
    model: Macro16BoundaryDeepONetWithLE0,
    checkpoint_path: Path,
    device: torch.device,
) -> Macro16BoundaryDeepONetWithLE0:
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device("cuda" if bool(args.cuda) and torch.cuda.is_available() else "cpu")
    compact_paths = compact_paths_from_args(args)
    data = load_macro16_compacts(
        compact_paths,
        point_table=macro16_source128_point_table(),
        frame_stride=1,
        max_frames_per_compact=0,
        scale_mode="normalized",
        b_label_coordinate="auto",
    )
    train_idx, val_idx, split_meta = split_indices(
        data.case_id,
        val_fraction=float(args.val_fraction),
        val_cases=str(args.val_cases),
        seed=int(args.seed),
    )
    branch_raw = np.concatenate(
        [data.q48_hat, data.x16_hat.reshape(data.x16_hat.shape[0], -1), data.length_scale],
        axis=1,
    )
    branch_mean, branch_std = stats(branch_raw[train_idx], axis=0)
    point_mean, point_std = stats(data.point_features_hat[train_idx].reshape(-1, data.point_features_hat.shape[-1]), axis=0)
    le_mean, le_std = stats(data.le[train_idx], axis=0)
    branch_norm = standardize(branch_raw, branch_mean, branch_std)
    point_norm = standardize(data.point_features_hat, point_mean.reshape(1, 1, -1), point_std.reshape(1, 1, -1))
    le_norm = standardize(data.le, le_mean, le_std)
    q_std = branch_std.reshape(-1)[:48]
    j_norm_target = (data.b * q_std.reshape(1, 1, 1, 48) / _le_std_scale(le_std, data.le.shape[1])).astype(np.float32)
    le0_star = data.le - np.einsum("npaj,nj->npa", data.b, data.q48_hat)
    le0_init_norm = np.mean(((le0_star[train_idx] - le_mean) / le_std).astype(np.float32), axis=0)
    b_loss_scale = build_physical_b_loss_scale(data.b[train_idx])

    model = build_model(
        branch_norm=branch_norm,
        point_norm=point_norm,
        branch_mean=branch_mean,
        branch_std=branch_std,
        le_mean=le_mean,
        le_std=le_std,
        j_norm_target=j_norm_target,
        le0_init_norm=le0_init_norm,
        train_idx=train_idx,
        args=args,
        device=device,
    )
    sub = train_idx[: int(args.batch_size)]
    xb = torch.as_tensor(branch_norm[sub], dtype=torch.float32, device=device)
    pb = torch.as_tensor(point_norm[sub], dtype=torch.float32, device=device)
    leb = torch.as_tensor(le_norm[sub], dtype=torch.float32, device=device)
    jb = torch.as_tensor(j_norm_target[sub], dtype=torch.float32, device=device)
    cols = list(range(48))
    le_std_scale_t = torch.as_tensor(_le_std_scale(le_std, data.le.shape[1]), dtype=torch.float32, device=device)
    q_std_t = torch.as_tensor(q_std, dtype=torch.float32, device=device)
    b_scale_t = torch.as_tensor(b_loss_scale, dtype=torch.float32, device=device)

    pred = model(xb, pb)
    le_loss = nn.functional.mse_loss(pred, leb)
    model.zero_grad(set_to_none=True)
    le_loss.backward()
    le_grads = grad_norms(model)

    model.zero_grad(set_to_none=True)
    j_pred = ad_jacobian(model, xb, pb, cols, create_graph=True, method="forward")
    j_loss = physical_balanced_b_loss_from_j_norm(
        j_pred,
        jb,
        le_std_scale=le_std_scale_t,
        q_std_cols=q_std_t,
        b_scale=b_scale_t,
    )
    j_loss.backward()
    b_grads = grad_norms(model)
    b_pred = j_norm_to_b_qhat_torch(j_pred.detach(), le_std_scale_t, q_std_t).cpu().numpy()

    initial_decomp = evaluate_decomposition(
        model,
        branch_norm=branch_norm,
        point_norm=point_norm,
        le_mean=le_mean,
        le_std=le_std,
        q_std=q_std,
        b_true=data.b,
        le_true=data.le,
        indices=train_idx,
        device=device,
    )
    checkpoint_decomp = None
    if str(args.checkpoint).strip():
        model_ckpt = build_model(
            branch_norm=branch_norm,
            point_norm=point_norm,
            branch_mean=branch_mean,
            branch_std=branch_std,
            le_mean=le_mean,
            le_std=le_std,
            j_norm_target=j_norm_target,
            le0_init_norm=le0_init_norm,
            train_idx=train_idx,
            args=args,
            device=device,
        )
        model_ckpt = load_checkpoint_into_model(model_ckpt, Path(args.checkpoint), device)
        checkpoint_decomp = evaluate_decomposition(
            model_ckpt,
            branch_norm=branch_norm,
            point_norm=point_norm,
            le_mean=le_mean,
            le_std=le_std,
            q_std=q_std,
            b_true=data.b,
            le_true=data.le,
            indices=train_idx,
            device=device,
        )

    return {
        "status": "done",
        "device": str(device),
        "compact_paths": compact_paths,
        "split": {
            **split_meta,
            "train_indices": train_idx.astype(int).tolist(),
            "val_indices": val_idx.astype(int).tolist(),
            "train_case_ids": sorted(np.unique(data.case_id[train_idx]).astype(int).tolist()),
            "val_case_ids": sorted(np.unique(data.case_id[val_idx]).astype(int).tolist()),
        },
        "data": {
            "frame_count": int(data.q48_hat.shape[0]),
            "point_count": int(data.le.shape[1]),
            "model_visible_q": data.point_meta.get("model_visible_q"),
            "model_visible_B": data.point_meta.get("model_visible_B"),
            "case_ids": sorted(np.unique(data.case_id).astype(int).tolist()),
        },
        "q_rank": {
            "train": rank_report(data.q48_hat[train_idx]),
            "all": rank_report(data.q48_hat),
        },
        "b_variation": {
            "train": b_variation_report(data.b[train_idx], data.q48_hat[train_idx]),
            "all": b_variation_report(data.b, data.q48_hat),
        },
        "le0_star": {
            "train_rel_to_LE": rel_np(le0_star[train_idx], data.le[train_idx]),
            "val_rel_to_LE": rel_np(le0_star[val_idx], data.le[val_idx]),
            "train_centered_rel": rel_np(le0_star[train_idx] - np.mean(le0_star[train_idx], axis=0, keepdims=True), data.le[train_idx] - le_mean),
        },
        "initial_losses": {
            "LE_norm_mse": float(le_loss.detach().cpu()),
            "physical_balanced_B_loss": float(j_loss.detach().cpu()),
            "batch_AD_B_rel": rel_np(b_pred, data.b[sub]),
        },
        "initial_decomposition_train": initial_decomp,
        "checkpoint_decomposition_train": checkpoint_decomp,
        "grad_norms_from_LE_loss": le_grads,
        "grad_norms_from_physical_B_loss": b_grads,
        "model_options": {
            "basis_dim": int(args.basis_dim),
            "hidden_dim": int(args.hidden_dim),
            "branch_depth": int(args.branch_depth),
            "trunk_depth": int(args.trunk_depth),
            "residual_scale": float(args.residual_scale),
            "fe_baseline_scale": float(args.fe_baseline_scale),
            "freeze_skip": bool(args.freeze_skip),
            "freeze_fe_point_baseline": bool(args.freeze_fe_point_baseline),
            "le0_scale": float(args.le0_scale),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-list", default="")
    parser.add_argument("--compact", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-cases", default="")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--basis-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--branch-depth", type=int, default=3)
    parser.add_argument("--trunk-depth", type=int, default=3)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--fe-baseline-scale", type=float, default=1.0)
    parser.add_argument("--freeze-skip", action="store_true")
    parser.add_argument("--freeze-fe-point-baseline", action="store_true")
    parser.add_argument("--global-b-prior", action="store_true")
    parser.add_argument("--anchored-residual-gate-q0", type=float, default=0.0)
    parser.add_argument("--le0-scale", type=float, default=1.0)
    parser.add_argument("--freeze-le0-static", action="store_true")
    parser.add_argument("--freeze-le0-point", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if args.compact_list:
        _ = read_path_list(Path(args.compact_list))
    payload = diagnose(args)
    write_json(Path(args.out), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "q_centered_rank": payload["q_rank"]["train"]["centered"]["rank"],
                "B_mean_rel": payload["b_variation"]["train"]["constant_B_mean_rel"],
                "initial_B_rel": payload["initial_decomposition_train"]["full_B_rel"],
                "checkpoint_B_rel": (
                    None
                    if payload["checkpoint_decomposition_train"] is None
                    else payload["checkpoint_decomposition_train"]["full_B_rel"]
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
