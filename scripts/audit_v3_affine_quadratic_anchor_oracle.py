#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only oracle for a v3 affine-quadratic value anchor.

This script does not train a neural network. It fits a closed-form scalar
amplitude correction on top of the affine anchor:

    s = <q_useful_hat - q_mean(case), q_dir(case)>
    LE_anchor = LE_mean(case,point)
              + B_mean(case,point) @ (q_useful_hat - q_mean(case))
              + C0(case,point) + C1(case,point) * s + C2(case,point) * s^2

The derivative of this anchor with respect to q_useful_hat is audited against
B_local_useful_stack_hat. This answers whether the proposed value anchor fixes
case031 without hiding an AD-B inconsistency.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


CONTRACT_VERSION = "v3-css8-standard-operator-001"


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


def rms(value: np.ndarray) -> float:
    vals = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(vals * vals))) if vals.size else 0.0


def rel_norm(diff: np.ndarray, ref: np.ndarray) -> float:
    den = max(float(np.linalg.norm(np.asarray(ref, dtype=np.float64).reshape(-1))), 1.0e-30)
    return float(np.linalg.norm(np.asarray(diff, dtype=np.float64).reshape(-1)) / den)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    den = max(float(np.linalg.norm(aa) * np.linalg.norm(bb)), 1.0e-30)
    return float(np.dot(aa, bb) / den)


def read_path_list(path: Path) -> list[Path]:
    return [
        Path(line.strip()).resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def load_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.compact_list:
        paths.extend(read_path_list(Path(args.compact_list).resolve()))
    paths.extend(Path(item).resolve() for item in args.compact)
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    if not out:
        raise SystemExit("Provide --compact-list or at least one --compact")
    return out


def load_one(path: Path) -> dict[str, Any]:
    with np.load(str(path), allow_pickle=True) as z:
        required = {
            "q_useful_hat",
            "LE_local_stack",
            "B_local_useful_stack_hat",
            "standard_operator_contract_version",
            "case_id",
        }
        missing = sorted(required.difference(z.files))
        if missing:
            raise KeyError(f"{path}: missing required v3 fields {missing}")
        version = scalar_text(z["standard_operator_contract_version"])
        if version != CONTRACT_VERSION:
            raise ValueError(f"{path}: expected {CONTRACT_VERSION}, got {version}")
        old_labels = bool(np.asarray(z["uses_old_true176_labels_as_v3_labels"]).reshape(-1)[0]) if "uses_old_true176_labels_as_v3_labels" in z.files else False
        if old_labels:
            raise ValueError(f"{path}: uses_old_true176_labels_as_v3_labels is true")
        return {
            "path": str(path),
            "case_id": int(np.asarray(z["case_id"]).reshape(-1)[0]),
            "q": np.asarray(z["q_useful_hat"], dtype=np.float64),
            "le": np.asarray(z["LE_local_stack"], dtype=np.float64),
            "b": np.asarray(z["B_local_useful_stack_hat"], dtype=np.float64),
        }


def case_q_direction(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_norm = np.linalg.norm(q, axis=1)
    q_dir_frame = q / np.maximum(q_norm.reshape(-1, 1), 1.0e-30)
    q_dir = np.mean(q_dir_frame, axis=0)
    q_dir = q_dir / max(float(np.linalg.norm(q_dir)), 1.0e-30)
    return q_dir, q_dir_frame @ q_dir


def fit_affine_quadratic_anchor(q: np.ndarray, le: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    le_mean = np.mean(le, axis=0, keepdims=True)
    b_mean = np.mean(b, axis=0)
    q_mean = np.mean(q, axis=0)
    q_dir, internal_cos = case_q_direction(q)
    dq = q - q_mean.reshape(1, -1)
    signed_amp = dq @ q_dir
    affine = le_mean + np.einsum("pak,nk->npa", b_mean, dq)
    residual = le - affine
    s2 = signed_amp * signed_amp
    denom = max(float(np.sum(s2 * s2)), 1.0e-30)
    c2_only = np.einsum("n,npa->pa", s2, residual) / denom
    quadratic_s2_only = affine + s2.reshape(-1, 1, 1) * c2_only.reshape(1, c2_only.shape[0], c2_only.shape[1])
    xmat = np.stack([np.ones_like(signed_amp), signed_amp, s2], axis=1)
    coef = np.linalg.lstsq(xmat, residual.reshape(residual.shape[0], -1), rcond=None)[0]
    coef = coef.reshape(3, residual.shape[1], residual.shape[2])
    quadratic_full = affine + np.einsum("nm,mpa->npa", xmat, coef)
    b_linear = np.broadcast_to(b_mean.reshape(1, *b_mean.shape), b.shape)
    b_quadratic_s2_only = b_linear + 2.0 * signed_amp.reshape(-1, 1, 1, 1) * c2_only.reshape(1, c2_only.shape[0], c2_only.shape[1], 1) * q_dir.reshape(1, 1, 1, -1)
    slope = coef[1].reshape(1, coef.shape[1], coef.shape[2], 1) + 2.0 * signed_amp.reshape(-1, 1, 1, 1) * coef[2].reshape(1, coef.shape[1], coef.shape[2], 1)
    b_quadratic_full = b_linear + slope * q_dir.reshape(1, 1, 1, -1)
    return {
        "q_norm": np.linalg.norm(q, axis=1),
        "q_dir": q_dir,
        "internal_cos": internal_cos,
        "signed_amp": signed_amp,
        "le_mean": le_mean,
        "b_mean": b_mean,
        "affine": affine,
        "quadratic_s2_only": quadratic_s2_only,
        "quadratic_full": quadratic_full,
        "b_linear": b_linear,
        "b_quadratic_s2_only": b_quadratic_s2_only,
        "b_quadratic_full": b_quadratic_full,
        "c2_only": c2_only,
        "quadratic_coeff": coef,
    }


def audit_case(row: dict[str, Any]) -> dict[str, Any]:
    q = row["q"]
    le = row["le"]
    b = row["b"]
    fit = fit_affine_quadratic_anchor(q, le, b)
    linear = np.einsum("pak,nk->npa", fit["b_mean"], q)
    mean_le = fit["le_mean"]
    case_id = int(row["case_id"])
    b_linear = fit["b_linear"]
    b_quadratic_s2_only = fit["b_quadratic_s2_only"]
    b_quadratic_full = fit["b_quadratic_full"]
    return {
        "case_id": case_id,
        "compact_path": row["path"],
        "frame_count": int(q.shape[0]),
        "point_count": int(le.shape[1]),
        "q_norm_min": float(np.min(fit["q_norm"])),
        "q_norm_max": float(np.max(fit["q_norm"])),
        "q_norm_mean": float(np.mean(fit["q_norm"])),
        "LE_rms_mean": rms(le),
        "B_local_useful_rms_mean": rms(b),
        "case_internal_q_direction_cos_min": float(np.min(fit["internal_cos"])),
        "case_internal_q_direction_cos_mean": float(np.mean(fit["internal_cos"])),
        "C2_s2_only_rms": rms(fit["c2_only"]),
        "quadratic_C0_rms": rms(fit["quadratic_coeff"][0]),
        "quadratic_C1_rms": rms(fit["quadratic_coeff"][1]),
        "quadratic_C2_rms": rms(fit["quadratic_coeff"][2]),
        "mean_LE_rel": rel_norm(np.broadcast_to(mean_le, le.shape) - le, le),
        "linear_Bmean_at_q_LE_rel": rel_norm(linear - le, le),
        "affine_Bmean_LE_rel": rel_norm(fit["affine"] - le, le),
        "affine_quadratic_s2_only_LE_rel": rel_norm(fit["quadratic_s2_only"] - le, le),
        "affine_quadratic_full_LE_rel": rel_norm(fit["quadratic_full"] - le, le),
        "affine_AD_B_rel": rel_norm(b_linear - b, b),
        "affine_AD_B_cos": cosine(b_linear, b),
        "affine_quadratic_s2_only_AD_B_rel": rel_norm(b_quadratic_s2_only - b, b),
        "affine_quadratic_s2_only_AD_B_cos": cosine(b_quadratic_s2_only, b),
        "affine_quadratic_full_AD_B_rel": rel_norm(b_quadratic_full - b, b),
        "affine_quadratic_full_AD_B_cos": cosine(b_quadratic_full, b),
        "quadratic_s2_only_LE_improvement_over_affine": rel_norm(fit["affine"] - le, le) - rel_norm(fit["quadratic_s2_only"] - le, le),
        "quadratic_full_LE_improvement_over_affine": rel_norm(fit["affine"] - le, le) - rel_norm(fit["quadratic_full"] - le, le),
        "quadratic_s2_only_AD_B_delta_vs_affine": rel_norm(b_quadratic_s2_only - b_linear, b),
        "quadratic_full_AD_B_delta_vs_affine": rel_norm(b_quadratic_full - b_linear, b),
        "uses_old_true176_labels_as_v3_labels": False,
    }


def add_pool_context(rows: list[dict[str, Any]]) -> None:
    q_median = float(np.median([float(row["q_norm_mean"]) for row in rows]))
    le_median = float(np.median([float(row["LE_rms_mean"]) for row in rows]))
    for rank, row in enumerate(sorted(rows, key=lambda item: float(item["q_norm_mean"]), reverse=True), start=1):
        row["q_norm_mean_rank_desc"] = int(rank)
    for rank, row in enumerate(sorted(rows, key=lambda item: float(item["LE_rms_mean"]), reverse=True), start=1):
        row["LE_rms_mean_rank_desc"] = int(rank)
    for row in rows:
        row["q_norm_mean_ratio_to_pool_median"] = float(row["q_norm_mean"] / max(q_median, 1.0e-30))
        row["LE_rms_mean_ratio_to_pool_median"] = float(row["LE_rms_mean"] / max(le_median, 1.0e-30))


def aggregate(rows_raw: list[dict[str, Any]], case_rows: list[dict[str, Any]], key: str, target_key: str) -> float:
    num = 0.0
    den = 0.0
    for raw, metrics in zip(rows_raw, case_rows):
        # Recompute only the squared norms needed for true pooled weighting.
        q = raw["q"]
        le = raw["le"]
        b = raw["b"]
        fit = fit_affine_quadratic_anchor(q, le, b)
        if key == "linear_Bmean_at_q_LE_rel":
            pred = np.einsum("pak,nk->npa", fit["b_mean"], q)
            ref = le
        elif key == "affine_Bmean_LE_rel":
            pred = fit["affine"]
            ref = le
        elif key == "affine_quadratic_s2_only_LE_rel":
            pred = fit["quadratic_s2_only"]
            ref = le
        elif key == "affine_quadratic_full_LE_rel":
            pred = fit["quadratic_full"]
            ref = le
        elif key == "affine_AD_B_rel":
            pred = fit["b_linear"]
            ref = b
        elif key == "affine_quadratic_s2_only_AD_B_rel":
            pred = fit["b_quadratic_s2_only"]
            ref = b
        elif key == "affine_quadratic_full_AD_B_rel":
            pred = fit["b_quadratic_full"]
            ref = b
        else:
            raise KeyError(key)
        diff = pred - ref
        num += float(np.sum(diff * diff))
        den += float(np.sum(ref * ref))
    return float(np.sqrt(num) / max(np.sqrt(den), 1.0e-30))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    preferred = [
        "case_id",
        "q_norm_mean",
        "q_norm_mean_ratio_to_pool_median",
        "LE_rms_mean",
        "LE_rms_mean_ratio_to_pool_median",
        "linear_Bmean_at_q_LE_rel",
        "affine_Bmean_LE_rel",
        "affine_quadratic_s2_only_LE_rel",
        "affine_quadratic_full_LE_rel",
        "affine_AD_B_rel",
        "affine_quadratic_s2_only_AD_B_rel",
        "affine_quadratic_full_AD_B_rel",
        "affine_quadratic_full_AD_B_cos",
        "quadratic_C2_rms",
        "case_internal_q_direction_cos_min",
    ]
    fields = {key for row in rows for key in row}
    ordered = [key for key in preferred if key in fields]
    ordered.extend(sorted(fields.difference(ordered)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", default="")
    parser.add_argument("--compact", action="append", default=[])
    parser.add_argument("--focus-case", type=int, default=31)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows_raw = [load_one(path) for path in load_paths(args)]
    rows_raw = sorted(rows_raw, key=lambda row: int(row["case_id"]))
    case_rows = [audit_case(row) for row in rows_raw]
    add_pool_context(case_rows)
    focus = next((row for row in case_rows if int(row["case_id"]) == int(args.focus_case)), None)
    if focus is None:
        raise SystemExit(f"focus case {args.focus_case} not found")
    pooled = {
        key: aggregate(rows_raw, case_rows, key, "")
        for key in [
            "linear_Bmean_at_q_LE_rel",
            "affine_Bmean_LE_rel",
            "affine_quadratic_s2_only_LE_rel",
            "affine_quadratic_full_LE_rel",
            "affine_AD_B_rel",
            "affine_quadratic_s2_only_AD_B_rel",
            "affine_quadratic_full_AD_B_rel",
        ]
    }
    summary = {
        "audit_name": "v3_affine_quadratic_anchor_oracle",
        "out_root": str(out_root),
        "case_count": len(case_rows),
        "focus_case": int(args.focus_case),
        "focus_case_summary": focus,
        "pooled_metrics": pooled,
        "interpretation": {
            "focus_case_quadratic_full_LE_improves_affine": bool(float(focus["affine_quadratic_full_LE_rel"]) < float(focus["affine_Bmean_LE_rel"]) * 0.5),
            "focus_case_quadratic_full_AD_B_not_worse_than_affine": bool(float(focus["affine_quadratic_full_AD_B_rel"]) <= float(focus["affine_AD_B_rel"]) * 1.2),
            "focus_case_is_large_amplitude": bool(int(focus["q_norm_mean_rank_desc"]) == 1 and float(focus["q_norm_mean_ratio_to_pool_median"]) > 10.0),
        },
        "formal_training": False,
        "uses_old_true176_labels_as_v3_labels": False,
    }
    case_csv = out_root / "v3_affine_quadratic_anchor_oracle_case_metrics.csv"
    case_json = out_root / "v3_affine_quadratic_anchor_oracle_case_metrics.json"
    summary_json = out_root / "v3_affine_quadratic_anchor_oracle_summary.json"
    write_csv(case_csv, case_rows)
    case_json.write_text(json.dumps({"cases": case_rows}, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    print(json.dumps({**summary, "case_csv": str(case_csv), "case_json": str(case_json), "summary_json": str(summary_json)}, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
