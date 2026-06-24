#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only tangent/amplitude oracle for v3 CSS8 scalar q paths.

This script does not train a neural network. It audits whether the true tangent
label B_local_useful_stack_hat is explained by a scalar amplitude coordinate

    s = <q_useful_hat - q_mean(case), q_dir(case)>

and whether a derivative-consistent path anchor can close both LE and B:

    q_perp = q - q_mean - s * q_dir
    B_poly(s) = B0 + B1*s + B2*s^2 + ...
    F'(s) = B_poly(s) @ q_dir
    LE_anchor(q) = C + integral(F'(s), ds) + B_poly(s) @ q_perp

For samples on the scalar path, q_perp ~= 0 and dLE_anchor/dq = B_poly(s).
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


def path_coordinates(q: np.ndarray) -> dict[str, np.ndarray | float]:
    q_norm = np.linalg.norm(q, axis=1)
    q_frame_dir = q / np.maximum(q_norm[:, None], 1.0e-30)
    q_dir = np.mean(q_frame_dir, axis=0)
    q_dir = q_dir / max(float(np.linalg.norm(q_dir)), 1.0e-30)
    q_mean = np.mean(q, axis=0)
    dq = q - q_mean.reshape(1, -1)
    s = dq @ q_dir
    q_perp = dq - s.reshape(-1, 1) * q_dir.reshape(1, -1)
    dq_norm = np.linalg.norm(dq, axis=1)
    q_perp_norm = np.linalg.norm(q_perp, axis=1)
    return {
        "q_norm": q_norm,
        "q_dir": q_dir,
        "q_mean": q_mean,
        "dq": dq,
        "s": s,
        "q_perp": q_perp,
        "internal_cos": q_frame_dir @ q_dir,
        "q_perp_rel_max": float(np.max(q_perp_norm / np.maximum(dq_norm, 1.0e-30))),
        "q_perp_rel_mean": float(np.mean(q_perp_norm / np.maximum(dq_norm, 1.0e-30))),
    }


def vandermonde(s: np.ndarray, degree: int) -> np.ndarray:
    return np.stack([s**p for p in range(degree + 1)], axis=1)


def fit_poly(s: np.ndarray, y: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray]:
    x = vandermonde(s, degree)
    coef = np.linalg.lstsq(x, y.reshape(y.shape[0], -1), rcond=None)[0]
    pred = (x @ coef).reshape(y.shape)
    return coef.reshape((degree + 1, *y.shape[1:])), pred


def derivative_consistent_le_from_b(
    s: np.ndarray,
    q_dir: np.ndarray,
    le: np.ndarray,
    b_coeff: np.ndarray,
) -> np.ndarray:
    # path_slope_coeff[m,p,a] = B_m[p,a,k] q_dir[k]
    path_slope_coeff = np.einsum("mpak,k->mpa", b_coeff, q_dir)
    integ = np.zeros_like(le)
    for m in range(b_coeff.shape[0]):
        integ = integ + (s ** (m + 1)).reshape(-1, 1, 1) * path_slope_coeff[m].reshape(1, *path_slope_coeff[m].shape) / float(m + 1)
    const = np.mean(le - integ, axis=0, keepdims=True)
    return const + integ


def anchor_from_b_poly(
    s: np.ndarray,
    q_perp: np.ndarray,
    q_dir: np.ndarray,
    le: np.ndarray,
    b_coeff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    le_path = derivative_consistent_le_from_b(s, q_dir, le, b_coeff)
    b_pred = np.einsum("nm,mpak->npak", vandermonde(s, b_coeff.shape[0] - 1), b_coeff)
    le_anchor = le_path + np.einsum("npak,nk->npa", b_pred, q_perp)
    return le_anchor, b_pred


def secant_projection_stats(s: np.ndarray, le: np.ndarray, b: np.ndarray, q_dir: np.ndarray) -> dict[str, float]:
    rels: list[float] = []
    for i in range(len(s) - 1):
        ds = float(s[i + 1] - s[i])
        if abs(ds) < 1.0e-30:
            continue
        dle_ds = (le[i + 1] - le[i]) / ds
        b_mid = 0.5 * (b[i + 1] + b[i])
        proj = np.einsum("pak,k->pa", b_mid, q_dir)
        rels.append(rel_norm(proj - dle_ds, dle_ds))
    if not rels:
        return {"secant_Bproj_rel_mean": float("nan"), "secant_Bproj_rel_max": float("nan")}
    return {
        "secant_Bproj_rel_mean": float(np.mean(rels)),
        "secant_Bproj_rel_max": float(np.max(rels)),
    }


def audit_case(row: dict[str, Any], max_degree: int) -> dict[str, Any]:
    case_id = int(row["case_id"])
    q = row["q"]
    le = row["le"]
    b = row["b"]
    path = path_coordinates(q)
    s = np.asarray(path["s"], dtype=np.float64)
    q_dir = np.asarray(path["q_dir"], dtype=np.float64)
    q_perp = np.asarray(path["q_perp"], dtype=np.float64)
    b_mean = np.mean(b, axis=0, keepdims=True)
    result: dict[str, Any] = {
        "case_id": case_id,
        "compact_path": row["path"],
        "frame_count": int(q.shape[0]),
        "point_count": int(le.shape[1]),
        "q_norm_min": float(np.min(path["q_norm"])),
        "q_norm_max": float(np.max(path["q_norm"])),
        "q_norm_mean": float(np.mean(path["q_norm"])),
        "s_min": float(np.min(s)),
        "s_max": float(np.max(s)),
        "s_range": float(np.max(s) - np.min(s)),
        "LE_rms_mean": rms(le),
        "B_local_useful_rms_mean": rms(b),
        "case_internal_q_direction_cos_min": float(np.min(path["internal_cos"])),
        "case_internal_q_direction_cos_mean": float(np.mean(path["internal_cos"])),
        "q_perp_rel_max": float(path["q_perp_rel_max"]),
        "q_perp_rel_mean": float(path["q_perp_rel_mean"]),
        "Bmean_AD_B_rel": rel_norm(np.broadcast_to(b_mean, b.shape) - b, b),
        "Bmean_AD_B_cos": cosine(np.broadcast_to(b_mean, b.shape), b),
        **secant_projection_stats(s, le, b, q_dir),
        "uses_old_true176_labels_as_v3_labels": False,
    }
    for degree in range(0, max_degree + 1):
        b_coeff, b_pred = fit_poly(s, b, degree)
        le_anchor, b_anchor = anchor_from_b_poly(s, q_perp, q_dir, le, b_coeff)
        result[f"B_poly_deg{degree}_AD_B_rel"] = rel_norm(b_pred - b, b)
        result[f"B_poly_deg{degree}_AD_B_cos"] = cosine(b_pred, b)
        result[f"B_poly_deg{degree}_integrated_LE_rel"] = rel_norm(le_anchor - le, le)
        result[f"B_poly_deg{degree}_integrated_LE_rms"] = rms(le_anchor - le)
        result[f"B_poly_deg{degree}_anchor_AD_B_rel"] = rel_norm(b_anchor - b, b)
    best_degree = min(range(0, max_degree + 1), key=lambda deg: float(result[f"B_poly_deg{deg}_AD_B_rel"]))
    result["best_B_poly_degree_by_AD_B"] = int(best_degree)
    result["best_B_poly_AD_B_rel"] = float(result[f"B_poly_deg{best_degree}_AD_B_rel"])
    result["best_B_poly_integrated_LE_rel"] = float(result[f"B_poly_deg{best_degree}_integrated_LE_rel"])
    return result


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
        "Bmean_AD_B_rel",
        "B_poly_deg1_AD_B_rel",
        "B_poly_deg2_AD_B_rel",
        "B_poly_deg3_AD_B_rel",
        "B_poly_deg1_integrated_LE_rel",
        "B_poly_deg2_integrated_LE_rel",
        "B_poly_deg3_integrated_LE_rel",
        "secant_Bproj_rel_mean",
        "best_B_poly_degree_by_AD_B",
        "best_B_poly_AD_B_rel",
        "best_B_poly_integrated_LE_rel",
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
    parser.add_argument("--max-degree", type=int, default=3)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    if int(args.max_degree) < 0:
        raise SystemExit("--max-degree must be non-negative")

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows_raw = [load_one(path) for path in load_paths(args)]
    rows_raw = sorted(rows_raw, key=lambda row: int(row["case_id"]))
    case_rows = [audit_case(row, int(args.max_degree)) for row in rows_raw]
    add_pool_context(case_rows)
    focus = next((row for row in case_rows if int(row["case_id"]) == int(args.focus_case)), None)
    if focus is None:
        raise SystemExit(f"focus case {args.focus_case} not found")
    summary = {
        "audit_name": "v3_tangent_amplitude_oracle",
        "out_root": str(out_root),
        "case_count": len(case_rows),
        "focus_case": int(args.focus_case),
        "max_degree": int(args.max_degree),
        "focus_case_summary": focus,
        "interpretation": {
            "focus_case_is_scalar_path": bool(float(focus["case_internal_q_direction_cos_min"]) > 0.999 and float(focus["q_perp_rel_max"]) < 1.0e-6),
            "focus_case_B_poly_deg2_improves_Bmean": bool(float(focus.get("B_poly_deg2_AD_B_rel", float("inf"))) < float(focus["Bmean_AD_B_rel"]) * 0.5),
            "focus_case_integrated_B_poly_deg2_closes_LE": bool(float(focus.get("B_poly_deg2_integrated_LE_rel", float("inf"))) < 0.02),
            "focus_case_best_B_poly_rel_below_0p02": bool(float(focus["best_B_poly_AD_B_rel"]) < 0.02),
        },
        "formal_training": False,
        "uses_old_true176_labels_as_v3_labels": False,
    }
    case_csv = out_root / "v3_tangent_amplitude_oracle_case_metrics.csv"
    case_json = out_root / "v3_tangent_amplitude_oracle_case_metrics.json"
    summary_json = out_root / "v3_tangent_amplitude_oracle_summary.json"
    write_csv(case_csv, case_rows)
    case_json.write_text(json.dumps({"cases": case_rows}, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    print(json.dumps({**summary, "case_csv": str(case_csv), "case_json": str(case_json), "summary_json": str(summary_json)}, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
