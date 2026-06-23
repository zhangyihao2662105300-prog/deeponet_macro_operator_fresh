#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze v2 multi-case compact scales before formal model design.

This script is read-only with respect to v2 compacts.  It does not train a
model and does not use old TRUE176 LE/B values as v2 labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


B_USEFUL_KEYS = ("B_standard_useful", "B_local_useful")


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
    return str(obj)


def scalar_text(value: Any, default: str = "unknown") -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return str(default)
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def scalar_float_np(value: Any, default: float | None = None) -> float | None:
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return default
        return float(arr.reshape(-1)[0])
    except Exception:
        return default


def scalar_int_np(value: Any, default: int | None = None) -> int | None:
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return default
        return int(arr.reshape(-1)[0])
    except Exception:
        return default


def rms(value: np.ndarray) -> float:
    vals = np.asarray(value, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(vals * vals)))


def rel_norm(num: np.ndarray, den: np.ndarray) -> float:
    n = float(np.linalg.norm(np.asarray(num, dtype=np.float64).reshape(-1)))
    d = float(np.linalg.norm(np.asarray(den, dtype=np.float64).reshape(-1)))
    return n / max(d, 1.0e-30)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = max(float(np.linalg.norm(aa) * np.linalg.norm(bb)), 1.0e-30)
    return float(np.dot(aa, bb) / denom)


def finite_ratio(max_val: float, min_val: float) -> float | None:
    if not math.isfinite(max_val) or not math.isfinite(min_val) or abs(min_val) <= 1.0e-30:
        return None
    return float(max_val / min_val)


def parse_case_id(path: Path, z: np.lib.npyio.NpzFile | None = None) -> int:
    if z is not None:
        for key in ("v2b_pilot_case_id", "v2a_pilot_case_id", "case_id"):
            if key in z.files:
                val = scalar_int_np(z[key])
                if val is not None:
                    return int(val)
    match = re.search(r"case[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot parse case id from {path}")
    return int(match.group(1))


def find_b_key(files: list[str]) -> str:
    for key in B_USEFUL_KEYS:
        if key in files:
            return key
    raise KeyError(f"missing one of {B_USEFUL_KEYS}")


def read_compact_list(path: Path) -> list[Path]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            out.append(Path(item).resolve())
    return out


def parse_cases(text: str) -> list[int]:
    vals = []
    for part in str(text).replace(";", ",").split(","):
        item = part.strip()
        if item:
            vals.append(int(item))
    return vals


@dataclass
class CaseData:
    case_id: int
    compact_path: str
    q: np.ndarray
    le: np.ndarray
    b: np.ndarray
    ip_xi: np.ndarray
    t_q: np.ndarray
    t_eps: np.ndarray
    b_raw: np.ndarray
    q_removed_rel: float | None
    b_rigid_residual_rel: float | None


def load_case(path: Path) -> CaseData:
    with np.load(str(path), allow_pickle=True) as z:
        required = {
            "q_useful",
            "LE128_local",
            "ip_xi",
            "T_q_raw_to_useful",
            "T_eps_to_abq",
            "B_LE128_forward",
        }
        missing = sorted(required.difference(z.files))
        if missing:
            raise KeyError(f"{path}: missing required v2 stats fields {missing}")
        b_key = find_b_key(list(z.files))
        case_id = parse_case_id(path, z)
        q = np.asarray(z["q_useful"], dtype=np.float64)
        le = np.asarray(z["LE128_local"], dtype=np.float64)
        b = np.asarray(z[b_key], dtype=np.float64)
        ip_xi = np.asarray(z["ip_xi"], dtype=np.float64)
        if ip_xi.ndim == 3:
            ip_xi = ip_xi[0]
        t_q = np.asarray(z["T_q_raw_to_useful"], dtype=np.float64)
        t_eps = np.asarray(z["T_eps_to_abq"], dtype=np.float64)
        b_raw = np.asarray(z["B_LE128_forward"], dtype=np.float64)
        b_q_coord = scalar_text(z["B_label_q_coordinate"]) if "B_label_q_coordinate" in z.files else "unknown"
        b_out_coord = scalar_text(z["B_label_output_coordinate"]) if "B_label_output_coordinate" in z.files else "unknown"
        old_true176 = bool(np.asarray(z["uses_old_true176_labels_as_v2_labels"]).reshape(-1)[0]) if "uses_old_true176_labels_as_v2_labels" in z.files else False
        q_removed_rel = scalar_float_np(z["q_useful_removed_rigid_rel"]) if "q_useful_removed_rigid_rel" in z.files else None
        b_rigid_residual_rel = scalar_float_np(z["B_rigid_residual_rel"]) if "B_rigid_residual_rel" in z.files else None

    if old_true176:
        raise ValueError(f"{path}: uses_old_true176_labels_as_v2_labels is true")
    if b_q_coord != "q_useful":
        raise ValueError(f"{path}: B_label_q_coordinate must be q_useful, got {b_q_coord!r}")
    if b_out_coord != "local_jacobian_frame":
        raise ValueError(f"{path}: B_label_output_coordinate must be local_jacobian_frame, got {b_out_coord!r}")
    if q.ndim != 2 or q.shape[1] != 42:
        raise ValueError(f"{path}: q_useful must be [N,42], got {q.shape}")
    if le.shape != (q.shape[0], 128, 6):
        raise ValueError(f"{path}: LE128_local must be [N,128,6], got {le.shape}")
    if b.shape != (q.shape[0], 128, 6, 42):
        raise ValueError(f"{path}: {b_key} must be [N,128,6,42], got {b.shape}")
    if ip_xi.shape != (128, 3):
        raise ValueError(f"{path}: ip_xi must be [128,3], got {ip_xi.shape}")
    if t_q.shape != (42, 48):
        raise ValueError(f"{path}: T_q_raw_to_useful must be [42,48], got {t_q.shape}")
    if t_eps.shape not in {(128, 6, 6), (q.shape[0], 128, 6, 6)}:
        raise ValueError(f"{path}: T_eps_to_abq must be [128,6,6] or [N,128,6,6], got {t_eps.shape}")
    if b_raw.shape != (q.shape[0], 128, 6, 48):
        raise ValueError(f"{path}: B_LE128_forward must be [N,128,6,48], got {b_raw.shape}")

    return CaseData(
        case_id=int(case_id),
        compact_path=str(path),
        q=q,
        le=le,
        b=b,
        ip_xi=ip_xi,
        t_q=t_q,
        t_eps=t_eps,
        b_raw=b_raw,
        q_removed_rel=q_removed_rel,
        b_rigid_residual_rel=b_rigid_residual_rel,
    )


def array_stats(vals: np.ndarray) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "rms": rms(arr),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "max_abs": float(np.max(np.abs(arr))),
    }


def bq_oracle(q: np.ndarray, b: np.ndarray, le: np.ndarray) -> tuple[np.ndarray, float, float]:
    pred = np.einsum("npak,nk->npa", b, q)
    return pred, rel_norm(pred - le, le), cosine(pred, le)


def corr(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    xa = np.asarray([p[0] for p in pairs], dtype=np.float64)
    ya = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(np.std(xa)) <= 1.0e-30 or float(np.std(ya)) <= 1.0e-30:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def split_name(case_id: int, val_cases: set[int]) -> str:
    return "val" if int(case_id) in val_cases else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--val-cases", default="44,46")
    parser.add_argument("--near-zero-rel-tol", type=float, default=1.0e-6)
    args = parser.parse_args()

    compact_list = Path(args.compact_list).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cases = [load_case(path) for path in read_compact_list(compact_list)]
    if not cases:
        raise SystemExit("no cases loaded")
    val_cases = set(parse_cases(args.val_cases))
    train_cases = sorted(c.case_id for c in cases if c.case_id not in val_cases)
    unknown_val = sorted(val_cases.difference(c.case_id for c in cases))
    if unknown_val:
        raise SystemExit(f"val cases not present: {unknown_val}")

    q_all = np.concatenate([c.q for c in cases], axis=0)
    le_all = np.concatenate([c.le for c in cases], axis=0)
    b_all = np.concatenate([c.b for c in cases], axis=0)
    q_norm_all = np.linalg.norm(q_all, axis=1)
    q_component_std = np.std(q_all, axis=0)
    q_component_rms = np.sqrt(np.mean(q_all * q_all, axis=0))
    q_component_max_abs = np.max(np.abs(q_all), axis=0)
    near_zero_threshold = float(args.near_zero_rel_tol) * max(float(np.max(q_component_std)), 1.0e-30)
    q_near_zero_count = int(np.sum(q_component_std <= near_zero_threshold))

    q_centered = q_all - np.mean(q_all, axis=0, keepdims=True)
    q_cov = (q_centered.T @ q_centered) / max(q_centered.shape[0] - 1, 1)
    q_svals = np.linalg.svd(q_cov, compute_uv=False)
    q_rank_tol = max(float(q_svals[0]) * 1.0e-10, 1.0e-30) if q_svals.size else 1.0e-30
    q_rank = int(np.sum(q_svals > q_rank_tol))
    q_condition = finite_ratio(float(q_svals[0]), float(q_svals[q_rank - 1])) if q_rank > 0 else None

    le_component_rms = np.sqrt(np.mean(le_all * le_all, axis=(0, 1)))
    b_component_rms = np.sqrt(np.mean(b_all * b_all, axis=(0, 1, 3)))
    b_q_column_rms = np.sqrt(np.mean(b_all * b_all, axis=(0, 1, 2)))
    b_point_rms = np.sqrt(np.mean(b_all * b_all, axis=(0, 2, 3)))
    b_q_near_zero_threshold = float(args.near_zero_rel_tol) * max(float(np.max(b_q_column_rms)), 1.0e-30)
    b_q_near_zero_count = int(np.sum(b_q_column_rms <= b_q_near_zero_threshold))

    train_b = np.concatenate([c.b for c in cases if c.case_id in train_cases], axis=0)
    train_b_mean = np.mean(train_b, axis=0)

    per_case_rows: list[dict[str, Any]] = []
    bq_rows: list[dict[str, Any]] = []
    for c in cases:
        pred, bq_rel, bq_cos = bq_oracle(c.q, c.b, c.le)
        q_norm = np.linalg.norm(c.q, axis=1)
        le_component_case = np.sqrt(np.mean(c.le * c.le, axis=(0, 1)))
        b_component_case = np.sqrt(np.mean(c.b * c.b, axis=(0, 1, 3)))
        b_q_case = np.sqrt(np.mean(c.b * c.b, axis=(0, 1, 2)))
        row = {
            "case_id": c.case_id,
            "split": split_name(c.case_id, val_cases),
            "compact_path": c.compact_path,
            "frame_count": int(c.q.shape[0]),
            "q_useful_rms": rms(c.q),
            "q_norm_mean": float(np.mean(q_norm)),
            "q_norm_max": float(np.max(q_norm)),
            "q_norm_min": float(np.min(q_norm)),
            "LE_local_rms": rms(c.le),
            "LE_local_max_abs": float(np.max(np.abs(c.le))),
            "B_local_rms": rms(c.b),
            "B_local_max_abs": float(np.max(np.abs(c.b))),
            "q_useful_removed_rigid_rel": c.q_removed_rel,
            "B_rigid_residual_rel": c.b_rigid_residual_rel,
            "LE_local_Bq_rel": bq_rel,
            "LE_local_Bq_cos": bq_cos,
            "LE_local_Bq_rms": rms(pred),
            "B_train_mean_rel": rel_norm(np.broadcast_to(train_b_mean.reshape(1, 128, 6, 42), c.b.shape) - c.b, c.b),
            "LE_component_rms_min": float(np.min(le_component_case)),
            "LE_component_rms_max": float(np.max(le_component_case)),
            "LE_component_rms_ratio": finite_ratio(float(np.max(le_component_case)), float(np.min(le_component_case))),
            "B_component_rms_min": float(np.min(b_component_case)),
            "B_component_rms_max": float(np.max(b_component_case)),
            "B_component_rms_ratio": finite_ratio(float(np.max(b_component_case)), float(np.min(b_component_case))),
            "B_q_column_rms_min": float(np.min(b_q_case)),
            "B_q_column_rms_max": float(np.max(b_q_case)),
            "B_q_column_rms_ratio": finite_ratio(float(np.max(b_q_case)), float(np.min(b_q_case))),
        }
        per_case_rows.append(row)
        bq_rows.append(
            {
                "case_id": c.case_id,
                "split": row["split"],
                "LE_local_Bq_rel": bq_rel,
                "LE_local_Bq_cos": bq_cos,
                "q_norm_mean": row["q_norm_mean"],
                "LE_local_rms": row["LE_local_rms"],
                "q_useful_removed_rigid_rel": c.q_removed_rel,
                "B_train_mean_rel": row["B_train_mean_rel"],
            }
        )

    component_rows: list[dict[str, Any]] = []
    for i in range(42):
        component_rows.append(
            {
                "kind": "q_useful",
                "index": i,
                "mean": float(np.mean(q_all[:, i])),
                "std": float(q_component_std[i]),
                "rms": float(q_component_rms[i]),
                "min": float(np.min(q_all[:, i])),
                "max": float(np.max(q_all[:, i])),
                "max_abs": float(q_component_max_abs[i]),
                "near_zero": bool(q_component_std[i] <= near_zero_threshold),
            }
        )
    for i in range(6):
        component_rows.append(
            {
                "kind": "LE_local",
                "index": i,
                "mean": float(np.mean(le_all[:, :, i])),
                "std": float(np.std(le_all[:, :, i])),
                "rms": float(le_component_rms[i]),
                "min": float(np.min(le_all[:, :, i])),
                "max": float(np.max(le_all[:, :, i])),
                "max_abs": float(np.max(np.abs(le_all[:, :, i]))),
                "near_zero": False,
            }
        )
        component_rows.append(
            {
                "kind": "B_local_strain_component",
                "index": i,
                "mean": float(np.mean(b_all[:, :, i, :])),
                "std": float(np.std(b_all[:, :, i, :])),
                "rms": float(b_component_rms[i]),
                "min": float(np.min(b_all[:, :, i, :])),
                "max": float(np.max(b_all[:, :, i, :])),
                "max_abs": float(np.max(np.abs(b_all[:, :, i, :]))),
                "near_zero": False,
            }
        )

    q_column_rows = []
    for i in range(42):
        vals = b_all[:, :, :, i]
        q_column_rows.append(
            {
                "q_column": i,
                "B_q_column_rms": float(b_q_column_rms[i]),
                "B_q_column_std": float(np.std(vals)),
                "B_q_column_max_abs": float(np.max(np.abs(vals))),
                "near_zero": bool(b_q_column_rms[i] <= b_q_near_zero_threshold),
            }
        )

    def split_arrays(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        selected = [c for c in cases if split_name(c.case_id, val_cases) == split]
        return (
            np.concatenate([c.q for c in selected], axis=0),
            np.concatenate([c.le for c in selected], axis=0),
            np.concatenate([c.b for c in selected], axis=0),
            [row for row in per_case_rows if row["split"] == split],
        )

    split_stats = {}
    for split in ("train", "val"):
        q_split, le_split, b_split, case_rows = split_arrays(split)
        le_comp = np.sqrt(np.mean(le_split * le_split, axis=(0, 1)))
        b_comp = np.sqrt(np.mean(b_split * b_split, axis=(0, 1, 3)))
        bq_rel_vals = [float(row["LE_local_Bq_rel"]) for row in case_rows]
        split_stats[split] = {
            "case_count": len(case_rows),
            "frame_count": int(q_split.shape[0]),
            "q_useful_rms": rms(q_split),
            "q_norm_mean": float(np.mean(np.linalg.norm(q_split, axis=1))),
            "q_norm_max": float(np.max(np.linalg.norm(q_split, axis=1))),
            "LE_local_rms": rms(le_split),
            "LE_component_rms": le_comp.tolist(),
            "LE_component_rms_min": float(np.min(le_comp)),
            "LE_component_rms_max": float(np.max(le_comp)),
            "LE_component_rms_ratio": finite_ratio(float(np.max(le_comp)), float(np.min(le_comp))),
            "B_local_rms": rms(b_split),
            "B_component_rms": b_comp.tolist(),
            "B_component_rms_min": float(np.min(b_comp)),
            "B_component_rms_max": float(np.max(b_comp)),
            "B_component_rms_ratio": finite_ratio(float(np.max(b_comp)), float(np.min(b_comp))),
            "LE_local_Bq_rel_mean": float(np.mean(bq_rel_vals)),
            "LE_local_Bq_rel_max": float(np.max(bq_rel_vals)),
        }

    q_removed = [row["q_useful_removed_rigid_rel"] for row in per_case_rows]
    bq_rel = [row["LE_local_Bq_rel"] for row in per_case_rows]
    q_norm_mean = [row["q_norm_mean"] for row in per_case_rows]
    le_rms_case = [row["LE_local_rms"] for row in per_case_rows]
    correlations = {
        "Bq_rel_vs_q_norm_mean": corr(bq_rel, q_norm_mean),
        "Bq_rel_vs_LE_local_rms": corr(bq_rel, le_rms_case),
        "Bq_rel_vs_q_removed_rel": corr(bq_rel, q_removed),
    }

    summary = {
        "audit_name": "v2f_multi_case_stats",
        "compact_list": str(compact_list),
        "out_root": str(out_root),
        "case_count": len(cases),
        "case_ids": sorted(c.case_id for c in cases),
        "train_cases": train_cases,
        "val_cases": sorted(val_cases),
        "validation_is_overlapping": False,
        "q_useful": {
            "global": array_stats(q_all),
            "norm_mean": float(np.mean(q_norm_all)),
            "norm_max": float(np.max(q_norm_all)),
            "component_std_min": float(np.min(q_component_std)),
            "component_std_max": float(np.max(q_component_std)),
            "component_std_ratio": finite_ratio(float(np.max(q_component_std)), float(np.min(q_component_std))),
            "near_zero_component_count": q_near_zero_count,
            "near_zero_component_threshold": near_zero_threshold,
            "covariance_singular_values": q_svals.tolist(),
            "covariance_rank_estimate": q_rank,
            "covariance_condition_estimate": q_condition,
        },
        "LE_local": {
            "global": array_stats(le_all),
            "component_rms": le_component_rms.tolist(),
            "component_rms_min": float(np.min(le_component_rms)),
            "component_rms_max": float(np.max(le_component_rms)),
            "component_scale_ratio": finite_ratio(float(np.max(le_component_rms)), float(np.min(le_component_rms))),
        },
        "B_local": {
            "global": array_stats(b_all),
            "component_rms": b_component_rms.tolist(),
            "component_rms_min": float(np.min(b_component_rms)),
            "component_rms_max": float(np.max(b_component_rms)),
            "component_scale_ratio": finite_ratio(float(np.max(b_component_rms)), float(np.min(b_component_rms))),
            "q_column_rms_min": float(np.min(b_q_column_rms)),
            "q_column_rms_max": float(np.max(b_q_column_rms)),
            "q_column_scale_ratio": finite_ratio(float(np.max(b_q_column_rms)), float(np.min(b_q_column_rms))),
            "q_column_near_zero_count": b_q_near_zero_count,
            "pointwise_rms_min": float(np.min(b_point_rms)),
            "pointwise_rms_max": float(np.max(b_point_rms)),
            "pointwise_rms_mean": float(np.mean(b_point_rms)),
        },
        "Bq_oracle": {
            "global_rel": rel_norm(np.concatenate([bq_oracle(c.q, c.b, c.le)[0] for c in cases], axis=0) - le_all, le_all),
            "global_cos": cosine(np.concatenate([bq_oracle(c.q, c.b, c.le)[0] for c in cases], axis=0), le_all),
            "per_case_rel_min": float(np.min([row["LE_local_Bq_rel"] for row in per_case_rows])),
            "per_case_rel_max": float(np.max([row["LE_local_Bq_rel"] for row in per_case_rows])),
            "per_case_cos_min": float(np.min([row["LE_local_Bq_cos"] for row in per_case_rows])),
            "per_case_cos_max": float(np.max([row["LE_local_Bq_cos"] for row in per_case_rows])),
            "correlations": correlations,
        },
        "split_stats": split_stats,
        "B_train_mean_prior": {
            "per_case_rel_min": float(np.min([row["B_train_mean_rel"] for row in per_case_rows])),
            "per_case_rel_max": float(np.max([row["B_train_mean_rel"] for row in per_case_rows])),
            "val_case_rel": {str(row["case_id"]): row["B_train_mean_rel"] for row in per_case_rows if row["split"] == "val"},
        },
        "model_training_performed": False,
        "uses_old_true176_labels_as_v2_labels": False,
    }

    paths = {
        "stats_summary": out_root / "stats_summary.json",
        "per_case_stats": out_root / "per_case_stats.csv",
        "component_stats": out_root / "component_stats.csv",
        "q_column_stats": out_root / "q_column_stats.csv",
        "bq_oracle_stats": out_root / "bq_oracle_stats.csv",
    }
    paths["stats_summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    for key, rows in (
        ("per_case_stats", per_case_rows),
        ("component_stats", component_rows),
        ("q_column_stats", q_column_rows),
        ("bq_oracle_stats", bq_rows),
    ):
        with paths[key].open("w", newline="", encoding="utf-8") as f:
            fields = sorted({field for row in rows for field in row.keys()})
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps({**summary, "output_paths": {k: str(v) for k, v in paths.items()}}, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
