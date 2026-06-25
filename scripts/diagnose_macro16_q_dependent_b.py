#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose q-dependent Macro16 B variation and residual AD-B capacity.

This is a Gate 16 read-only diagnostic.  It does not train a model, change the
model architecture, change q48/LE ordering, or change the 128-point rule.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_macro16_trained_force_closure import make_model, torch_load_cross_platform  # noqa: E402
from macro_deeponet.macro16_geometry import MACRO16_CONTRACT_VERSION, macro16_source128_point_table  # noqa: E402
from macro_deeponet.train_macro16_boundary_sobolev import (  # noqa: E402
    _le_std_scale,
    ad_jacobian,
    load_macro16_compacts,
    parse_int_list,
    read_path_list,
    split_indices,
    standardize,
)
from macro_deeponet.true176_data import stats  # noqa: E402


class _CrossPlatformPosixPath(pathlib.PurePosixPath):
    pass


class _CrossPlatformWindowsPath(pathlib.PureWindowsPath):
    pass


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


def cos_np(pred: np.ndarray, true: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(true, dtype=np.float64).reshape(-1)
    den = float(np.linalg.norm(p) * np.linalg.norm(t))
    return float(np.dot(p, t) / den) if den > 1.0e-300 else float("nan")


def rms_np(vals: np.ndarray) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr))) if arr.size else 0.0


def rank_report(matrix: np.ndarray) -> dict[str, Any]:
    vals = np.asarray(matrix, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    flat = vals.reshape(vals.shape[0], -1)
    if flat.shape[0] == 0 or flat.shape[1] == 0:
        return {"rows": int(flat.shape[0]), "cols": int(flat.shape[1]), "rank": 0}
    s = np.linalg.svd(flat, compute_uv=False)
    tol = max(flat.shape) * np.finfo(np.float64).eps * (float(s[0]) if s.size else 0.0)
    energy = s * s
    total = max(float(np.sum(energy)), 1.0e-300)
    return {
        "rows": int(flat.shape[0]),
        "cols": int(flat.shape[1]),
        "rank": int(np.sum(s > tol)),
        "tol": float(tol),
        "singular_values_first8": s[: min(8, s.size)].tolist(),
        "energy_share_first8": (energy[: min(8, s.size)] / total).tolist(),
    }


def metric_block(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    return {
        "rel": rel_np(pred, true),
        "cos": cos_np(pred, true),
        "pred_rms": rms_np(pred),
        "true_rms": rms_np(true),
        "diff_rms": rms_np(np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)),
    }


def parse_checkpoint_arg(text: str) -> tuple[str, Path]:
    raw = str(text).strip()
    if not raw:
        raise ValueError("empty checkpoint argument")
    if "=" in raw:
        label, path = raw.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"checkpoint label is empty: {text!r}")
        return label, Path(path.strip())
    path = Path(raw)
    return path.parent.name or path.stem, path


def geometry_group_report(x16_hat: np.ndarray, q: np.ndarray, b: np.ndarray, case_id: np.ndarray) -> dict[str, Any]:
    rounded = np.round(np.asarray(x16_hat, dtype=np.float64).reshape(x16_hat.shape[0], -1), decimals=10)
    groups: dict[tuple[float, ...], list[int]] = {}
    for i, row in enumerate(rounded):
        groups.setdefault(tuple(row.tolist()), []).append(int(i))
    entries: list[dict[str, Any]] = []
    for gid, idx in enumerate(groups.values()):
        rows = np.asarray(idx, dtype=np.int64)
        b_rows = b[rows]
        q_rows = q[rows]
        mean_b = np.mean(b_rows, axis=0, keepdims=True)
        entries.append(
            {
                "group_id": int(gid),
                "frame_count": int(rows.size),
                "case_ids": sorted(np.unique(np.asarray(case_id, dtype=np.int64)[rows]).astype(int).tolist()),
                "constant_B_mean_rel": rel_np(np.broadcast_to(mean_b, b_rows.shape), b_rows),
                "B_centered_rank": rank_report(b_rows - mean_b),
                "q_rank": rank_report(q_rows - np.mean(q_rows, axis=0, keepdims=True)),
            }
        )
    return {
        "unique_geometry_count": int(len(groups)),
        "group_frame_counts": [int(len(v)) for v in groups.values()],
        "groups": entries,
    }


def split_metric(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    all_idx: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        "train": metric_block(pred[train_idx], true[train_idx]),
        "val": metric_block(pred[val_idx], true[val_idx]),
        "all": metric_block(pred[all_idx], true[all_idx]),
    }


def q_linear_fit_report(
    q: np.ndarray,
    b: np.ndarray,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    all_idx: np.ndarray,
    ridge: float,
) -> dict[str, Any]:
    q_arr = np.asarray(q, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    target = b_arr.reshape(b_arr.shape[0], -1)
    q_mean = np.mean(q_arr[train_idx], axis=0, keepdims=True)
    x = q_arr - q_mean
    design = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    x_train = design[train_idx]
    y_train = target[train_idx]
    normal = x_train.T @ x_train
    if float(ridge) > 0.0:
        normal = normal + float(ridge) * np.eye(normal.shape[0], dtype=np.float64)
    coef = np.linalg.solve(normal, x_train.T @ y_train)
    pred = (design @ coef).reshape(b_arr.shape)
    mean_b = np.mean(b_arr[train_idx], axis=0, keepdims=True)
    pred_mean = np.broadcast_to(mean_b, b_arr.shape)
    return {
        "ridge": float(ridge),
        "q_design_rank_train": rank_report(x_train),
        "constant_point_only": split_metric(pred_mean, b_arr, train_idx=train_idx, val_idx=val_idx, all_idx=all_idx),
        "linear_q_correction": split_metric(pred, b_arr, train_idx=train_idx, val_idx=val_idx, all_idx=all_idx),
        "linear_minus_constant_rel": split_metric(pred - pred_mean, b_arr - pred_mean, train_idx=train_idx, val_idx=val_idx, all_idx=all_idx),
    }


def b_variation_report(q: np.ndarray, b: np.ndarray, *, indices: np.ndarray) -> dict[str, Any]:
    q_rows = np.asarray(q[indices], dtype=np.float64)
    b_rows = np.asarray(b[indices], dtype=np.float64)
    mean_b = np.mean(b_rows, axis=0, keepdims=True)
    b_center = b_rows - mean_b
    q_center = q_rows - np.mean(q_rows, axis=0, keepdims=True)
    return {
        "frame_count": int(indices.size),
        "q_rms": rms_np(q_rows),
        "q_centered_rms": rms_np(q_center),
        "q_rank": rank_report(q_center),
        "B_rms": rms_np(b_rows),
        "B_centered_rms": rms_np(b_center),
        "B_centered_rel_to_B": rel_np(b_center, b_rows),
        "constant_B_mean_rel": rel_np(np.broadcast_to(mean_b, b_rows.shape), b_rows),
        "B_centered_rank": rank_report(b_center),
        "q_frame_norms": np.linalg.norm(q_rows, axis=1).tolist(),
        "B_frame_norms": np.linalg.norm(b_rows.reshape(b_rows.shape[0], -1), axis=1).tolist(),
    }


def b_from_j(j_norm: np.ndarray, le_std: np.ndarray, q_std: np.ndarray, point_count: int) -> np.ndarray:
    return np.asarray(j_norm, dtype=np.float64) * _le_std_scale(le_std, point_count) / q_std.reshape(1, 1, 1, -1)


def checkpoint_decomposition(
    *,
    label: str,
    checkpoint_path: Path,
    branch_raw: np.ndarray,
    point_features: np.ndarray,
    b_true: np.ndarray,
    le_true: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    all_idx: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch_load_cross_platform(checkpoint_path.resolve(), device)
    if str(checkpoint.get("contract_version", "")) != MACRO16_CONTRACT_VERSION:
        raise ValueError(f"{checkpoint_path}: unexpected contract version {checkpoint.get('contract_version')}")
    norms = {key: np.asarray(value) for key, value in checkpoint["norms"].items()}
    branch_norm = standardize(branch_raw, norms["branch_mean"], norms["branch_std"])
    point_norm = standardize(
        point_features,
        norms["point_mean"].reshape(1, 1, -1),
        norms["point_std"].reshape(1, 1, -1),
    )
    q_std = np.asarray(norms["branch_std"], dtype=np.float64).reshape(-1)[:48]
    le_std = np.asarray(norms["le_std"], dtype=np.float64)
    le_mean = np.asarray(norms["le_mean"], dtype=np.float64)
    model = make_model(checkpoint, norms, device)
    columns = list(range(48))
    le_rows: list[np.ndarray] = []
    j_rows: list[np.ndarray] = []
    base_rows: list[np.ndarray] = []
    for start in range(0, all_idx.size, int(batch_size)):
        sub = all_idx[start : start + int(batch_size)]
        xb = torch.as_tensor(branch_norm[sub], dtype=torch.float32, device=device)
        pb = torch.as_tensor(point_norm[sub], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_norm = model(xb, pb).detach().cpu().numpy()
            base_norm = model._linear_b_norm(pb).detach().cpu().numpy()
        with torch.enable_grad():
            j_norm = ad_jacobian(model, xb, pb, columns, create_graph=False, method="forward")
        le_rows.append(pred_norm * le_std + le_mean)
        j_rows.append(j_norm.detach().cpu().numpy().astype(np.float64))
        base_rows.append(base_norm.astype(np.float64))
    le_pred = np.concatenate(le_rows, axis=0)
    j_full = np.concatenate(j_rows, axis=0)
    j_base = np.concatenate(base_rows, axis=0)
    b_full = b_from_j(j_full, le_std, q_std, b_true.shape[1])
    b_base = b_from_j(j_base, le_std, q_std, b_true.shape[1])
    b_residual = b_full - b_base
    b_target_minus_base = np.asarray(b_true[all_idx], dtype=np.float64) - b_base

    order_map = {int(v): i for i, v in enumerate(all_idx.tolist())}
    local_train = np.asarray([order_map[int(v)] for v in train_idx], dtype=np.int64)
    local_val = np.asarray([order_map[int(v)] for v in val_idx], dtype=np.int64)
    local_all = np.asarray([order_map[int(v)] for v in all_idx], dtype=np.int64)

    return {
        "label": label,
        "checkpoint": str(checkpoint_path),
        "args": {
            key: checkpoint.get("args", {}).get(key)
            for key in [
                "epochs",
                "residual_scale",
                "jacobian_loss_weight",
                "global_b_lr_scale",
                "point_b_lr_scale",
                "freeze_b_prior_after_warmstart",
            ]
        },
        "frame_count": int(all_idx.size),
        "LE": split_metric(le_pred, le_true[all_idx], train_idx=local_train, val_idx=local_val, all_idx=local_all),
        "B_full_AD": split_metric(b_full, b_true[all_idx], train_idx=local_train, val_idx=local_val, all_idx=local_all),
        "B_point_baseline": split_metric(b_base, b_true[all_idx], train_idx=local_train, val_idx=local_val, all_idx=local_all),
        "B_residual_vs_target_minus_baseline": split_metric(
            b_residual,
            b_target_minus_base,
            train_idx=local_train,
            val_idx=local_val,
            all_idx=local_all,
        ),
        "residual_B_norm_fraction": {
            "train": float(
                np.linalg.norm(b_residual[local_train].reshape(-1))
                / max(float(np.linalg.norm(b_full[local_train].reshape(-1))), 1.0e-300)
            ),
            "val": float(
                np.linalg.norm(b_residual[local_val].reshape(-1))
                / max(float(np.linalg.norm(b_full[local_val].reshape(-1))), 1.0e-300)
            ),
            "all": float(
                np.linalg.norm(b_residual.reshape(-1))
                / max(float(np.linalg.norm(b_full.reshape(-1))), 1.0e-300)
            ),
        },
        "target_minus_baseline_norm_fraction": {
            "train": float(
                np.linalg.norm(b_target_minus_base[local_train].reshape(-1))
                / max(float(np.linalg.norm(b_true[all_idx][local_train].reshape(-1))), 1.0e-300)
            ),
            "val": float(
                np.linalg.norm(b_target_minus_base[local_val].reshape(-1))
                / max(float(np.linalg.norm(b_true[all_idx][local_val].reshape(-1))), 1.0e-300)
            ),
            "all": float(
                np.linalg.norm(b_target_minus_base.reshape(-1))
                / max(float(np.linalg.norm(b_true[all_idx].reshape(-1))), 1.0e-300)
            ),
        },
        "residual_B_rank": {
            "train": rank_report(b_residual[local_train]),
            "val": rank_report(b_residual[local_val]),
            "all": rank_report(b_residual),
        },
        "target_minus_baseline_rank": {
            "train": rank_report(b_target_minus_base[local_train]),
            "val": rank_report(b_target_minus_base[local_val]),
            "all": rank_report(b_target_minus_base),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if bool(args.cuda) and torch.cuda.is_available() else "cpu")
    compact_paths = read_path_list(Path(args.compact_list))
    data = load_macro16_compacts(
        compact_paths,
        point_table=macro16_source128_point_table(),
        frame_stride=1,
        max_frames_per_compact=0,
        scale_mode="normalized",
        b_label_coordinate="auto",
    )
    all_idx = np.arange(data.q48_hat.shape[0], dtype=np.int64)
    case_list = set(parse_int_list(str(args.case_list))) if str(args.case_list).strip() else set()
    if case_list:
        all_idx = all_idx[np.isin(data.case_id[all_idx], np.asarray(sorted(case_list), dtype=np.int64))]
    if int(args.max_frames) > 0:
        all_idx = all_idx[: int(args.max_frames)]
    if all_idx.size < 2:
        raise ValueError("Gate 16 diagnosis needs at least two frames")

    train_full, val_full, split_meta = split_indices(
        data.case_id,
        val_fraction=float(args.val_fraction),
        val_cases=str(args.val_cases),
        seed=int(args.seed),
    )
    train_idx = np.intersect1d(all_idx, train_full, assume_unique=False)
    val_idx = np.intersect1d(all_idx, val_full, assume_unique=False)
    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError("selected frames produced empty train or val split")

    branch_raw = np.concatenate(
        [data.q48_hat, data.x16_hat.reshape(data.x16_hat.shape[0], -1), data.length_scale],
        axis=1,
    )
    branch_mean, branch_std = stats(branch_raw[train_idx], axis=0)
    point_mean, point_std = stats(data.point_features_hat[train_idx].reshape(-1, data.point_features_hat.shape[-1]), axis=0)
    le_mean, le_std = stats(data.le[train_idx], axis=0)
    point_norm = standardize(data.point_features_hat, point_mean.reshape(1, 1, -1), point_std.reshape(1, 1, -1))
    q_std = branch_std.reshape(-1)[:48]
    j_target = data.b * q_std.reshape(1, 1, 1, 48) / _le_std_scale(le_std, data.le.shape[1])
    j_mean_train = np.mean(j_target[train_idx], axis=0, keepdims=True)
    b_mean_train = np.mean(data.b[train_idx], axis=0, keepdims=True)
    point_only_j_mean_b = b_from_j(
        np.broadcast_to(j_mean_train, j_target.shape),
        le_std,
        q_std,
        data.b.shape[1],
    )
    point_only_b_mean = np.broadcast_to(b_mean_train, data.b.shape)
    linear_probe = q_linear_fit_report(
        data.q48_hat,
        data.b,
        train_idx=train_idx,
        val_idx=val_idx,
        all_idx=all_idx,
        ridge=float(args.ridge),
    )
    checkpoints = [parse_checkpoint_arg(text) for text in getattr(args, "checkpoint", [])]
    checkpoint_reports = []
    for label, checkpoint_path in checkpoints:
        checkpoint_reports.append(
            checkpoint_decomposition(
                label=label,
                checkpoint_path=checkpoint_path,
                branch_raw=branch_raw,
                point_features=data.point_features_hat,
                b_true=data.b,
                le_true=data.le,
                train_idx=train_idx,
                val_idx=val_idx,
                all_idx=all_idx,
                batch_size=int(args.batch_size),
                device=device,
            )
        )

    return {
        "task": "Gate 16 q-dependent B diagnosis",
        "contract": {
            "q_order_changed": False,
            "LE_order_changed": False,
            "point_rule": "macro16_source128",
            "training_started": False,
            "model_structure_changed": False,
        },
        "data": {
            "compact_list": str(Path(args.compact_list)),
            "compact_count": int(len(compact_paths)),
            "frame_count_selected": int(all_idx.size),
            "train_frame_count": int(train_idx.size),
            "val_frame_count": int(val_idx.size),
            "case_ids_selected": sorted(np.unique(data.case_id[all_idx]).astype(int).tolist()),
            "train_case_ids": sorted(np.unique(data.case_id[train_idx]).astype(int).tolist()),
            "val_case_ids": sorted(np.unique(data.case_id[val_idx]).astype(int).tolist()),
            "split": split_meta,
            "point_count": int(data.b.shape[1]),
            "point_feature_dim": int(data.point_features_hat.shape[-1]),
            "q_dim": 48,
        },
        "geometry_groups": geometry_group_report(
            data.x16_hat[all_idx],
            data.q48_hat[all_idx],
            data.b[all_idx],
            data.case_id[all_idx],
        ),
        "variation": {
            "train": b_variation_report(data.q48_hat, data.b, indices=train_idx),
            "val": b_variation_report(data.q48_hat, data.b, indices=val_idx),
            "all": b_variation_report(data.q48_hat, data.b, indices=all_idx),
        },
        "point_only_baselines": {
            "mean_B_macro_qdef": split_metric(
                point_only_b_mean,
                data.b,
                train_idx=train_idx,
                val_idx=val_idx,
                all_idx=all_idx,
            ),
            "mean_J_norm_converted_to_B": split_metric(
                point_only_j_mean_b,
                data.b,
                train_idx=train_idx,
                val_idx=val_idx,
                all_idx=all_idx,
            ),
        },
        "linear_q_probe": linear_probe,
        "target_j_norm_variation": {
            "train": b_variation_report(data.q48_hat, j_target, indices=train_idx),
            "val": b_variation_report(data.q48_hat, j_target, indices=val_idx),
            "all": b_variation_report(data.q48_hat, j_target, indices=all_idx),
        },
        "checkpoint_decomposition": checkpoint_reports,
        "diagnosis": {
            "point_only_B_has_irreducible_q_dependent_error": bool(
                rel_np(point_only_b_mean[all_idx], data.b[all_idx]) > float(args.point_only_fail_rel)
            ),
            "linear_q_probe_train_can_fit": bool(
                linear_probe["linear_q_correction"]["train"]["rel"]
                < rel_np(point_only_b_mean[train_idx], data.b[train_idx])
            ),
            "large_training_release_allowed": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--val-cases", default="")
    parser.add_argument("--case-list", default="")
    parser.add_argument("--checkpoint", action="append", default=[], help="label=path or path; may be repeated")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--ridge", type=float, default=1.0e-8)
    parser.add_argument("--point-only-fail-rel", type=float, default=0.05)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    write_json(Path(args.out), payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
