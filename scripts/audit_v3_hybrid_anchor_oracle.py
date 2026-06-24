#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only oracle for a hybrid v3 value+tangent anchor.

The hybrid anchor combines the previous two useful but incomplete anchors:

* affine-quadratic value path: accurate LE(s)
* tangent-cubic B path: accurate B(s)

For q = q_mean + s q_dir + q_perp:

    LE_hybrid(q) = V_affine_quadratic(s) + B_cubic(s) @ q_perp

On the audited scalar paths q_perp ~= 0, this keeps the accurate value path.
Its derivative is audited with the path derivative from V(s) along q_dir and
the cubic B(s) tangent on the transverse subspace.
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


def rel_norm(diff: np.ndarray, ref: np.ndarray) -> float:
    den = max(float(np.linalg.norm(np.asarray(ref, dtype=np.float64).reshape(-1))), 1.0e-30)
    return float(np.linalg.norm(np.asarray(diff, dtype=np.float64).reshape(-1)) / den)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    den = max(float(np.linalg.norm(aa) * np.linalg.norm(bb)), 1.0e-30)
    return float(np.dot(aa, bb) / den)


def rms(value: np.ndarray) -> float:
    vals = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(vals * vals))) if vals.size else 0.0


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


def fit_case(q: np.ndarray, le: np.ndarray, b: np.ndarray, tangent_degree: int) -> dict[str, np.ndarray | float]:
    q_norm = np.linalg.norm(q, axis=1)
    q_frame_dir = q / np.maximum(q_norm[:, None], 1.0e-30)
    q_dir = np.mean(q_frame_dir, axis=0)
    q_dir = q_dir / max(float(np.linalg.norm(q_dir)), 1.0e-30)
    q_mean = np.mean(q, axis=0)
    dq = q - q_mean.reshape(1, -1)
    s = dq @ q_dir
    q_perp = dq - s.reshape(-1, 1) * q_dir.reshape(1, -1)

    le_mean = np.mean(le, axis=0, keepdims=True)
    b_mean = np.mean(b, axis=0)
    affine = le_mean + np.einsum("pak,nk->npa", b_mean, dq)
    xq = np.stack([np.ones_like(s), s, s * s], axis=1)
    qcoef = np.linalg.lstsq(xq, (le - affine).reshape(le.shape[0], -1), rcond=None)[0]
    qcoef = qcoef.reshape(3, le.shape[1], le.shape[2])
    value_path = affine + np.einsum("nm,mpa->npa", xq, qcoef)
    value_path_prime = np.einsum("pak,k->pa", b_mean, q_dir).reshape(1, le.shape[1], le.shape[2])
    value_path_prime = value_path_prime + qcoef[1].reshape(1, le.shape[1], le.shape[2])
    value_path_prime = value_path_prime + 2.0 * s.reshape(-1, 1, 1) * qcoef[2].reshape(1, le.shape[1], le.shape[2])

    xt = np.stack([s**power for power in range(tangent_degree + 1)], axis=1)
    bcoef = np.linalg.lstsq(xt, b.reshape(b.shape[0], -1), rcond=None)[0]
    bcoef = bcoef.reshape(tangent_degree + 1, b.shape[1], b.shape[2], b.shape[3])
    b_poly = np.einsum("nm,mpak->npak", xt, bcoef)

    bq = np.einsum("npak,k->npa", b_poly, q_dir)
    b_hybrid = b_poly - bq[:, :, :, None] * q_dir.reshape(1, 1, 1, -1)
    b_hybrid = b_hybrid + value_path_prime[:, :, :, None] * q_dir.reshape(1, 1, 1, -1)
    le_hybrid = value_path + np.einsum("npak,nk->npa", b_poly, q_perp)

    return {
        "q_norm": q_norm,
        "q_dir": q_dir,
        "s": s,
        "q_perp": q_perp,
        "internal_cos": q_frame_dir @ q_dir,
        "affine": affine,
        "value_path": value_path,
        "value_path_prime": value_path_prime,
        "b_poly": b_poly,
        "b_hybrid": b_hybrid,
        "le_hybrid": le_hybrid,
        "b_mean": b_mean,
    }


def audit_case(row: dict[str, Any], tangent_degree: int) -> dict[str, Any]:
    q = row["q"]
    le = row["le"]
    b = row["b"]
    fit = fit_case(q, le, b, tangent_degree)
    q_perp = np.asarray(fit["q_perp"])
    dq = q - np.mean(q, axis=0).reshape(1, -1)
    q_perp_rel = np.linalg.norm(q_perp, axis=1) / np.maximum(np.linalg.norm(dq, axis=1), 1.0e-30)
    b_mean = np.asarray(fit["b_mean"])
    b_mean_full = np.broadcast_to(b_mean.reshape(1, *b_mean.shape), b.shape)
    return {
        "case_id": int(row["case_id"]),
        "compact_path": row["path"],
        "frame_count": int(q.shape[0]),
        "point_count": int(le.shape[1]),
        "q_norm_mean": float(np.mean(fit["q_norm"])),
        "LE_rms_mean": rms(le),
        "B_local_useful_rms_mean": rms(b),
        "case_internal_q_direction_cos_min": float(np.min(fit["internal_cos"])),
        "case_internal_q_direction_cos_mean": float(np.mean(fit["internal_cos"])),
        "q_perp_rel_max": float(np.max(q_perp_rel)),
        "q_perp_rel_mean": float(np.mean(q_perp_rel)),
        "affine_LE_rel": rel_norm(fit["affine"] - le, le),
        "affine_quadratic_LE_rel": rel_norm(fit["value_path"] - le, le),
        "tangent_poly_AD_B_rel": rel_norm(fit["b_poly"] - b, b),
        "tangent_poly_AD_B_cos": cosine(fit["b_poly"], b),
        "hybrid_LE_rel": rel_norm(fit["le_hybrid"] - le, le),
        "hybrid_AD_B_rel": rel_norm(fit["b_hybrid"] - b, b),
        "hybrid_AD_B_cos": cosine(fit["b_hybrid"], b),
        "Bmean_AD_B_rel": rel_norm(b_mean_full - b, b),
        "value_path_derivative_vs_Bpoly_qdir_rel": rel_norm(
            fit["value_path_prime"] - np.einsum("npak,k->npa", fit["b_poly"], fit["q_dir"]),
            np.einsum("npak,k->npa", b, fit["q_dir"]),
        ),
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


def pooled_metric(raw_rows: list[dict[str, Any]], tangent_degree: int, metric: str) -> float:
    num = 0.0
    den = 0.0
    for row in raw_rows:
        fit = fit_case(row["q"], row["le"], row["b"], tangent_degree)
        if metric == "hybrid_LE_rel":
            pred, ref = fit["le_hybrid"], row["le"]
        elif metric == "hybrid_AD_B_rel":
            pred, ref = fit["b_hybrid"], row["b"]
        elif metric == "affine_quadratic_LE_rel":
            pred, ref = fit["value_path"], row["le"]
        elif metric == "tangent_poly_AD_B_rel":
            pred, ref = fit["b_poly"], row["b"]
        else:
            raise KeyError(metric)
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
        "case_internal_q_direction_cos_min",
        "q_perp_rel_max",
        "affine_quadratic_LE_rel",
        "tangent_poly_AD_B_rel",
        "hybrid_LE_rel",
        "hybrid_AD_B_rel",
        "hybrid_AD_B_cos",
        "value_path_derivative_vs_Bpoly_qdir_rel",
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
    parser.add_argument("--tangent-degree", type=int, default=3)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    if int(args.tangent_degree) < 0:
        raise SystemExit("--tangent-degree must be non-negative")

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    raw_rows = [load_one(path) for path in load_paths(args)]
    raw_rows = sorted(raw_rows, key=lambda row: int(row["case_id"]))
    case_rows = [audit_case(row, int(args.tangent_degree)) for row in raw_rows]
    add_pool_context(case_rows)
    focus = next((row for row in case_rows if int(row["case_id"]) == int(args.focus_case)), None)
    if focus is None:
        raise SystemExit(f"focus case {args.focus_case} not found")
    pooled = {
        key: pooled_metric(raw_rows, int(args.tangent_degree), key)
        for key in [
            "affine_quadratic_LE_rel",
            "tangent_poly_AD_B_rel",
            "hybrid_LE_rel",
            "hybrid_AD_B_rel",
        ]
    }
    summary = {
        "audit_name": "v3_hybrid_value_tangent_anchor_oracle",
        "out_root": str(out_root),
        "case_count": len(case_rows),
        "focus_case": int(args.focus_case),
        "tangent_degree": int(args.tangent_degree),
        "focus_case_summary": focus,
        "pooled_metrics": pooled,
        "interpretation": {
            "focus_case_hybrid_closes_LE": bool(float(focus["hybrid_LE_rel"]) < 0.02),
            "focus_case_hybrid_closes_AD_B": bool(float(focus["hybrid_AD_B_rel"]) < 0.02),
            "pool_hybrid_closes_LE": bool(float(pooled["hybrid_LE_rel"]) < 0.02),
            "pool_hybrid_closes_AD_B": bool(float(pooled["hybrid_AD_B_rel"]) < 0.02),
        },
        "formal_training": False,
        "uses_old_true176_labels_as_v3_labels": False,
    }
    case_csv = out_root / "v3_hybrid_anchor_oracle_case_metrics.csv"
    case_json = out_root / "v3_hybrid_anchor_oracle_case_metrics.json"
    summary_json = out_root / "v3_hybrid_anchor_oracle_summary.json"
    write_csv(case_csv, case_rows)
    case_json.write_text(json.dumps({"cases": case_rows}, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    print(json.dumps({**summary, "case_csv": str(case_csv), "case_json": str(case_json), "summary_json": str(summary_json)}, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
