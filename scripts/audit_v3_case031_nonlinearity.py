#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused v3 CSS8 amplitude/nonlinearity audit for case031.

This is a read-only diagnostic.  It does not train a model.  It audits each
selected v3 compact frame-by-frame and compares case031 against the rest of the
10-case pool.

The goal is to decide whether case031 is primarily:

* a large-amplitude nonlinear path,
* an abnormal q/LE/B compact,
* or simply a value-anchor design problem.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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


def polynomial_fit_rel(x: np.ndarray, y: np.ndarray, degree: int) -> tuple[float, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y2 = np.asarray(y, dtype=np.float64).reshape(x.shape[0], -1)
    cols = [np.ones_like(x)]
    for p in range(1, degree + 1):
        cols.append(x**p)
    xmat = np.stack(cols, axis=1)
    coef = np.linalg.lstsq(xmat, y2, rcond=None)[0]
    pred = (xmat @ coef).reshape(y.shape)
    return rel_norm(pred - y, y), pred


def quadratic_correction_rel(base: np.ndarray, q: np.ndarray, le: np.ndarray) -> float:
    q_norm = np.linalg.norm(q, axis=1)
    q_mean = np.mean(q, axis=0)
    dq_norm2 = np.sum((q - q_mean.reshape(1, -1)) ** 2, axis=1)
    residual = le - base
    xmat = np.stack([np.ones_like(q_norm), q_norm, q_norm * q_norm, dq_norm2], axis=1)
    y = residual.reshape(residual.shape[0], -1)
    coef = np.linalg.lstsq(xmat, y, rcond=None)[0]
    pred = base + (xmat @ coef).reshape(le.shape)
    return rel_norm(pred - le, le)


def secant_stats(q: np.ndarray, le: np.ndarray, b_mean: np.ndarray) -> dict[str, float]:
    rels: list[float] = []
    sec_norms: list[float] = []
    dq_norms: list[float] = []
    for i in range(q.shape[0] - 1):
        dq = q[i + 1] - q[i]
        dle = le[i + 1] - le[i]
        pred = np.einsum("pak,k->pa", b_mean, dq)
        rels.append(rel_norm(pred - dle, dle))
        sec_norms.append(rms(dle) / max(float(np.linalg.norm(dq)), 1.0e-30))
        dq_norms.append(float(np.linalg.norm(dq)))
    if not rels:
        return {
            "secant_Bmean_rel_mean": float("nan"),
            "secant_Bmean_rel_max": float("nan"),
            "secant_slope_rms_over_dq_mean": float("nan"),
            "dq_step_norm_mean": float("nan"),
            "dq_step_norm_max": float("nan"),
        }
    return {
        "secant_Bmean_rel_mean": float(np.mean(rels)),
        "secant_Bmean_rel_max": float(np.max(rels)),
        "secant_slope_rms_over_dq_mean": float(np.mean(sec_norms)),
        "dq_step_norm_mean": float(np.mean(dq_norms)),
        "dq_step_norm_max": float(np.max(dq_norms)),
    }


def load_one(path: Path) -> dict[str, Any]:
    with np.load(str(path), allow_pickle=True) as z:
        version = scalar_text(z["standard_operator_contract_version"])
        if version != CONTRACT_VERSION:
            raise ValueError(f"{path}: expected {CONTRACT_VERSION}, got {version}")
        if "uses_old_true176_labels_as_v3_labels" in z.files and bool(np.asarray(z["uses_old_true176_labels_as_v3_labels"]).reshape(-1)[0]):
            raise ValueError(f"{path}: uses_old_true176_labels_as_v3_labels is true")
        return {
            "path": str(path),
            "case_id": int(np.asarray(z["case_id"]).reshape(-1)[0]),
            "q": np.asarray(z["q_useful_hat"], dtype=np.float64),
            "le": np.asarray(z["LE_local_stack"], dtype=np.float64),
            "b": np.asarray(z["B_local_useful_stack_hat"], dtype=np.float64),
            "b_raw": np.asarray(z["B_LE128_forward"], dtype=np.float64),
        }


def audit_case(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_id = int(row["case_id"])
    q = row["q"]
    le = row["le"]
    b = row["b"]
    b_raw = row["b_raw"]
    q_norm = np.linalg.norm(q, axis=1)
    q_dir = q / np.maximum(q_norm.reshape(-1, 1), 1.0e-30)
    q_rep = np.mean(q_dir, axis=0)
    q_rep_norm = np.linalg.norm(q_rep)
    q_rep = q_rep / max(float(q_rep_norm), 1.0e-30)
    signed_amp = q @ q_rep
    internal_cos = q_dir @ q_rep

    b_mean = np.mean(b, axis=0)
    le_mean = np.mean(le, axis=0, keepdims=True)
    q_mean = np.mean(q, axis=0)
    pred_bmean = np.einsum("pak,nk->npa", b_mean, q)
    pred_affine = le_mean + np.einsum("pak,nk->npa", b_mean, q - q_mean.reshape(1, -1))
    pred_frame_b = np.einsum("npak,nk->npa", b, q)
    pred_frame_b_affine = le_mean + np.einsum("npak,nk->npa", b, q - q_mean.reshape(1, -1))
    qnorm_lin_rel, _ = polynomial_fit_rel(q_norm, le, degree=1)
    qnorm_quad_rel, _ = polynomial_fit_rel(q_norm, le, degree=2)
    signed_lin_rel, _ = polynomial_fit_rel(signed_amp, le, degree=1)
    signed_quad_rel, _ = polynomial_fit_rel(signed_amp, le, degree=2)
    quad_corr_rel = quadratic_correction_rel(pred_affine, q, le)

    frame_rows: list[dict[str, Any]] = []
    for i in range(q.shape[0]):
        frame_rows.append(
            {
                "case_id": case_id,
                "frame_index": int(i),
                "q_norm": float(q_norm[i]),
                "signed_amplitude": float(signed_amp[i]),
                "q_direction_cos_to_case_mean": float(internal_cos[i]),
                "LE_rms": rms(le[i]),
                "B_local_useful_rms": rms(b[i]),
                "B_raw_rms": rms(b_raw[i]),
                "Bmean_at_q_frame_LE_rel": rel_norm(pred_bmean[i] - le[i], le[i]),
                "affine_Bmean_frame_LE_rel": rel_norm(pred_affine[i] - le[i], le[i]),
                "frame_B_at_q_frame_LE_rel": rel_norm(pred_frame_b[i] - le[i], le[i]),
                "frame_B_affine_frame_LE_rel": rel_norm(pred_frame_b_affine[i] - le[i], le[i]),
            }
        )

    sec = secant_stats(q, le, b_mean)
    summary = {
        "case_id": case_id,
        "compact_path": row["path"],
        "frame_count": int(q.shape[0]),
        "point_count": int(le.shape[1]),
        "q_norm_min": float(np.min(q_norm)),
        "q_norm_max": float(np.max(q_norm)),
        "q_norm_mean": float(np.mean(q_norm)),
        "q_norm_median": float(np.median(q_norm)),
        "q_norm_range": float(np.max(q_norm) - np.min(q_norm)),
        "signed_amp_min": float(np.min(signed_amp)),
        "signed_amp_max": float(np.max(signed_amp)),
        "signed_amp_mean": float(np.mean(signed_amp)),
        "case_internal_q_direction_cos_min": float(np.min(internal_cos)),
        "case_internal_q_direction_cos_mean": float(np.mean(internal_cos)),
        "LE_rms_mean": rms(le),
        "LE_rms_min_frame": float(min(rms(le[i]) for i in range(le.shape[0]))),
        "LE_rms_max_frame": float(max(rms(le[i]) for i in range(le.shape[0]))),
        "B_local_useful_rms_mean": rms(b),
        "B_raw_rms_mean": rms(b_raw),
        "Bmean_at_q_LE_rel": rel_norm(pred_bmean - le, le),
        "affine_Bmean_LE_rel": rel_norm(pred_affine - le, le),
        "frame_B_at_q_LE_rel": rel_norm(pred_frame_b - le, le),
        "frame_B_affine_LE_rel": rel_norm(pred_frame_b_affine - le, le),
        "qnorm_linear_LE_fit_rel": qnorm_lin_rel,
        "qnorm_quadratic_LE_fit_rel": qnorm_quad_rel,
        "signed_amp_linear_LE_fit_rel": signed_lin_rel,
        "signed_amp_quadratic_LE_fit_rel": signed_quad_rel,
        "quadratic_correction_over_affine_rel": quad_corr_rel,
        **sec,
    }
    summary["affine_improvement_over_Bmean"] = float(summary["Bmean_at_q_LE_rel"] - summary["affine_Bmean_LE_rel"])
    summary["quadratic_improvement_over_affine"] = float(summary["affine_Bmean_LE_rel"] - summary["quadratic_correction_over_affine_rel"])
    summary["likely_direction_constant"] = bool(summary["case_internal_q_direction_cos_min"] > 0.999)
    summary["likely_scalar_amplitude_path"] = bool(summary["case_internal_q_direction_cos_min"] > 0.999 and summary["q_norm_range"] > 0.0)
    return summary, frame_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "case_id",
        "frame_index",
        "q_norm",
        "signed_amplitude",
        "LE_rms",
        "Bmean_at_q_LE_rel",
        "affine_Bmean_LE_rel",
        "quadratic_correction_over_affine_rel",
        "qnorm_quadratic_LE_fit_rel",
        "signed_amp_quadratic_LE_fit_rel",
        "Bmean_at_q_frame_LE_rel",
        "affine_Bmean_frame_LE_rel",
    ]
    ordered = [key for key in preferred if key in fields]
    ordered.extend([key for key in fields if key not in ordered])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def rank_case(rows: list[dict[str, Any]], case_id: int, key: str, *, reverse: bool = True) -> int | None:
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=reverse)
    for idx, row in enumerate(ordered, start=1):
        if int(row["case_id"]) == int(case_id):
            return idx
    return None


def safe_ratio(value: float, ref: float) -> float:
    return float(value / max(float(ref), 1.0e-30))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", default="")
    parser.add_argument("--compact", action="append", default=[])
    parser.add_argument("--focus-case", type=int, default=31)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    case_summaries: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for path in load_paths(args):
        summary, frames = audit_case(load_one(path))
        case_summaries.append(summary)
        frame_rows.extend(frames)
    case_summaries = sorted(case_summaries, key=lambda row: int(row["case_id"]))
    frame_rows = sorted(frame_rows, key=lambda row: (int(row["case_id"]), int(row["frame_index"])))
    focus = next((row for row in case_summaries if int(row["case_id"]) == int(args.focus_case)), None)
    if focus is None:
        raise SystemExit(f"focus case {args.focus_case} not found")
    q_norms = np.asarray([row["q_norm_mean"] for row in case_summaries], dtype=np.float64)
    le_rms = np.asarray([row["LE_rms_mean"] for row in case_summaries], dtype=np.float64)
    q_norm_median = float(np.median(q_norms))
    le_rms_median = float(np.median(le_rms))
    for row in case_summaries:
        case_id = int(row["case_id"])
        row["q_norm_mean_rank_desc"] = rank_case(case_summaries, case_id, "q_norm_mean", reverse=True)
        row["LE_rms_mean_rank_desc"] = rank_case(case_summaries, case_id, "LE_rms_mean", reverse=True)
        row["q_norm_mean_ratio_to_pool_median"] = safe_ratio(float(row["q_norm_mean"]), q_norm_median)
        row["LE_rms_mean_ratio_to_pool_median"] = safe_ratio(float(row["LE_rms_mean"]), le_rms_median)
        row["likely_large_amplitude"] = bool(
            row["q_norm_mean_rank_desc"] == 1 and row["q_norm_mean_ratio_to_pool_median"] > 10.0
        )
        row["likely_large_LE"] = bool(
            row["LE_rms_mean_rank_desc"] == 1 and row["LE_rms_mean_ratio_to_pool_median"] > 10.0
        )
    summary = {
        "audit_name": "v3_case031_amplitude_nonlinearity_audit",
        "focus_case": int(args.focus_case),
        "out_root": str(out_root),
        "case_count": len(case_summaries),
        "focus_case_summary": focus,
        "focus_case_rankings": {
            "q_norm_mean_desc": rank_case(case_summaries, int(args.focus_case), "q_norm_mean", reverse=True),
            "LE_rms_mean_desc": rank_case(case_summaries, int(args.focus_case), "LE_rms_mean", reverse=True),
            "Bmean_at_q_LE_rel_desc": rank_case(case_summaries, int(args.focus_case), "Bmean_at_q_LE_rel", reverse=True),
            "affine_Bmean_LE_rel_desc": rank_case(case_summaries, int(args.focus_case), "affine_Bmean_LE_rel", reverse=True),
            "quadratic_correction_over_affine_rel_desc": rank_case(case_summaries, int(args.focus_case), "quadratic_correction_over_affine_rel", reverse=True),
        },
        "pool_stats": {
            "q_norm_mean_min": float(np.min(q_norms)),
            "q_norm_mean_median": q_norm_median,
            "q_norm_mean_max": float(np.max(q_norms)),
            "LE_rms_mean_min": float(np.min(le_rms)),
            "LE_rms_mean_median": le_rms_median,
            "LE_rms_mean_max": float(np.max(le_rms)),
        },
        "interpretation": {
            "focus_case_is_largest_amplitude": bool(rank_case(case_summaries, int(args.focus_case), "q_norm_mean", reverse=True) == 1),
            "focus_case_is_largest_LE": bool(rank_case(case_summaries, int(args.focus_case), "LE_rms_mean", reverse=True) == 1),
            "focus_case_direction_constant": bool(focus["likely_direction_constant"]),
            "focus_case_scalar_amplitude_path": bool(focus["likely_scalar_amplitude_path"]),
            "affine_anchor_large_improvement": bool(focus["affine_improvement_over_Bmean"] > 0.1),
            "quadratic_correction_large_improvement": bool(focus["quadratic_improvement_over_affine"] > 0.02),
        },
        "formal_training": False,
        "uses_old_true176_labels_as_v3_labels": False,
    }
    summary_path = out_root / "v3_case031_nonlinearity_summary.json"
    case_csv = out_root / "v3_case031_nonlinearity_case_summary.csv"
    frame_csv = out_root / "v3_case031_nonlinearity_frame_manifest.csv"
    case_json = out_root / "v3_case031_nonlinearity_case_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    case_json.write_text(json.dumps({"cases": case_summaries}, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    write_csv(case_csv, case_summaries)
    write_csv(frame_csv, frame_rows)
    print(json.dumps({**summary, "summary_path": str(summary_path), "case_csv": str(case_csv), "frame_csv": str(frame_csv), "case_json": str(case_json)}, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
