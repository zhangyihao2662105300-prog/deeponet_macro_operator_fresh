"""Diagnose Macro16 overfit AD-B gradient flow on a tiny compact.

This script is a Gate 08 diagnostic.  It does not train the network, change the
model structure, change q48/LE ordering, or change the 128-point rule.
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

from macro_deeponet.macro16_geometry import macro16_standard_point_table  # noqa: E402
from macro_deeponet.models import Macro16BoundaryDeepONetWithLE0  # noqa: E402
from macro_deeponet.train_macro16_boundary_sobolev import (  # noqa: E402
    _le_std_scale,
    compact_paths_from_args,
    load_macro16_compacts,
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


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device("cuda" if bool(args.cuda) and torch.cuda.is_available() else "cpu")
    point_table = macro16_standard_point_table(
        plane_order=int(args.plane_gauss_order),
        thickness_order=int(args.thickness_gauss_order),
    )
    compact_paths = compact_paths_from_args(args)
    data = load_macro16_compacts(
        compact_paths,
        point_table=point_table,
        frame_stride=1,
        max_frames_per_compact=0,
        scale_mode="normalized",
        b_label_coordinate="auto",
    )
    train_idx, val_idx, split_meta = split_indices(
        data.case_id,
        val_fraction=0.2,
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
    skip_init = np.mean(j_norm_target[train_idx], axis=0).astype(np.float32)
    le0_star = data.le - np.einsum("npaj,nj->npa", data.b, data.q48_hat)
    le0_init_norm = np.mean(((le0_star[train_idx] - le_mean) / le_std).astype(np.float32), axis=0)

    model = Macro16BoundaryDeepONetWithLE0(
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
        gate_q0=0.0,
        le0_init_norm=torch.as_tensor(le0_init_norm, dtype=torch.float32),
        le0_scale=1.0,
        train_le0_static=True,
        train_le0_point=True,
    ).to(device)

    sub = train_idx[: int(args.batch_size)]
    xb = torch.as_tensor(branch_norm[sub], dtype=torch.float32, device=device)
    pb = torch.as_tensor(point_norm[sub], dtype=torch.float32, device=device)
    leb = torch.as_tensor(le_norm[sub], dtype=torch.float32, device=device)
    jb = torch.as_tensor(j_norm_target[sub], dtype=torch.float32, device=device)
    cols = list(range(48))

    pred = model(xb, pb)
    le_loss = nn.functional.mse_loss(pred, leb)
    model.zero_grad(set_to_none=True)
    le_loss.backward()
    le_grads = grad_norms(model)

    model.zero_grad(set_to_none=True)
    j_pred = ad_jacobian(model, xb, pb, cols, create_graph=True, method="forward")
    j_loss = nn.functional.mse_loss(j_pred, jb[:, :, :, cols])
    j_loss.backward()
    j_grads = grad_norms(model)

    j_np = j_pred.detach().cpu().numpy().astype(np.float64)
    b_pred = j_np * _le_std_scale(le_std, j_np.shape[1]) / q_std.reshape(1, 1, 1, -1)
    b_true = data.b[sub].astype(np.float64)
    j_true = j_norm_target[sub].astype(np.float64)
    pred_raw = pred.detach().cpu().numpy() * le_std + le_mean

    return {
        "status": "done",
        "device": str(device),
        "compact_paths": compact_paths,
        "split": split_meta,
        "train_indices": train_idx.astype(int).tolist(),
        "val_indices": val_idx.astype(int).tolist(),
        "batch_indices": sub.astype(int).tolist(),
        "train_case_ids": sorted(np.unique(data.case_id[train_idx]).astype(int).tolist()),
        "val_case_ids": sorted(np.unique(data.case_id[val_idx]).astype(int).tolist()),
        "shape": {
            "frames": int(data.q48_hat.shape[0]),
            "points": int(data.le.shape[1]),
            "branch_dim": int(branch_norm.shape[-1]),
            "point_dim": int(point_norm.shape[-1]),
        },
        "scale": {
            "q_std_min": float(np.min(q_std)),
            "q_std_max": float(np.max(q_std)),
            "le_std_min": float(np.min(le_std)),
            "le_std_max": float(np.max(le_std)),
            "B_rms": float(np.sqrt(np.mean(np.asarray(data.b, dtype=np.float64) ** 2))),
            "B_abs_max": float(np.max(np.abs(data.b))),
            "j_norm_target_rms": float(np.sqrt(np.mean(np.asarray(j_norm_target, dtype=np.float64) ** 2))),
            "j_norm_target_abs_max": float(np.max(np.abs(j_norm_target))),
        },
        "initial_losses": {
            "LE_norm_mse": float(le_loss.detach().cpu()),
            "J_norm_mse": float(j_loss.detach().cpu()),
            "batch_LE_rel": rel_np(pred_raw, data.le[sub]),
            "batch_AD_B_norm_rel": rel_np(j_np, j_true),
            "batch_AD_B_rel": rel_np(b_pred, b_true),
        },
        "grad_norms_from_LE_loss": le_grads,
        "grad_norms_from_AD_B_loss": j_grads,
        "model_options": {
            "basis_dim": int(args.basis_dim),
            "hidden_dim": int(args.hidden_dim),
            "branch_depth": int(args.branch_depth),
            "trunk_depth": int(args.trunk_depth),
            "freeze_skip": bool(args.freeze_skip),
            "freeze_fe_point_baseline": bool(args.freeze_fe_point_baseline),
            "residual_scale": float(args.residual_scale),
            "fe_baseline_scale": float(args.fe_baseline_scale),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-list", default="")
    parser.add_argument("--compact", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-cases", default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--basis-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--branch-depth", type=int, default=3)
    parser.add_argument("--trunk-depth", type=int, default=3)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--fe-baseline-scale", type=float, default=1.0)
    parser.add_argument("--freeze-skip", action="store_true")
    parser.add_argument("--freeze-fe-point-baseline", action="store_true")
    parser.add_argument("--plane-gauss-order", type=int, default=3)
    parser.add_argument("--thickness-gauss-order", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if args.compact_list:
        _ = read_path_list(Path(args.compact_list))
    payload = diagnose(args)
    write_json(Path(args.out), payload)
    print(json.dumps(payload["initial_losses"], sort_keys=True), flush=True)
    print(json.dumps(payload["grad_norms_from_AD_B_loss"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
