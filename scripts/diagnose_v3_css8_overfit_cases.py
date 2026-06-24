#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose v3 CSS8 10-case training-set overfit difficulty.

This is not formal training.  It audits the selected v3 standard-operator
compacts and an optional multi-case smoke history to answer:

    Why is LE_local_stack overfit uneven across training cases?

The script computes per-case scale metrics and a few closed-form value anchors:

* zero prediction
* per-case mean LE
* per-case mean B(point) @ q_useful_hat
* per-frame least-squares scalar rescale of Bmean @ q

It can also merge the latest per-case rows from
``v3_multi_case_overfit_case_history.csv`` so the initial oracle and final smoke
metrics are compared side by side.
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


def rel_norm(diff: np.ndarray, ref: np.ndarray) -> float:
    den = max(float(np.linalg.norm(np.asarray(ref, dtype=np.float64).reshape(-1))), 1.0e-30)
    return float(np.linalg.norm(np.asarray(diff, dtype=np.float64).reshape(-1)) / den)


def rms(value: np.ndarray) -> float:
    vals = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(vals * vals))) if vals.size else 0.0


def norm_stats(value: np.ndarray) -> dict[str, float]:
    vals = np.asarray(value, dtype=np.float64)
    norms = np.linalg.norm(vals.reshape(vals.shape[0], -1), axis=1)
    return {
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
        "mean": float(np.mean(norms)),
        "median": float(np.median(norms)),
    }


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


def read_smoke_case_history(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    latest: dict[int, dict[str, Any]] = {}
    for row in rows:
        case_id = int(float(row["case_id"]))
        step = int(float(row["step"]))
        cur = latest.get(case_id)
        if cur is None or step >= int(cur["step"]):
            latest[case_id] = row
    return latest


def float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def diagnose_one(path: Path, smoke_latest: dict[int, dict[str, Any]]) -> dict[str, Any]:
    with np.load(str(path), allow_pickle=True) as z:
        version = scalar_text(z["standard_operator_contract_version"])
        if version != CONTRACT_VERSION:
            raise ValueError(f"{path}: expected {CONTRACT_VERSION}, got {version}")
        if "uses_old_true176_labels_as_v3_labels" in z.files and bool(np.asarray(z["uses_old_true176_labels_as_v3_labels"]).reshape(-1)[0]):
            raise ValueError(f"{path}: uses_old_true176_labels_as_v3_labels is true")
        case_id = int(np.asarray(z["case_id"]).reshape(-1)[0])
        q = np.asarray(z["q_useful_hat"], dtype=np.float64)
        le = np.asarray(z["LE_local_stack"], dtype=np.float64)
        b = np.asarray(z["B_local_useful_stack_hat"], dtype=np.float64)
        b_raw = np.asarray(z["B_LE128_forward"], dtype=np.float64)
        geom = np.asarray(z["geometry_global_hat"], dtype=np.float64).reshape(-1)
        trunk = np.asarray(z["trunk_features_hat"], dtype=np.float64)

    b_mean = np.mean(b, axis=0)
    pred_bmean = np.einsum("pak,nk->npa", b_mean, q)
    pred_mean_le = np.mean(le, axis=0, keepdims=True)
    pred_zero = np.zeros_like(le)
    pred_frame_scalar = np.empty_like(le)
    scalar_values: list[float] = []
    for frame in range(le.shape[0]):
        a = pred_bmean[frame].reshape(-1)
        y = le[frame].reshape(-1)
        denom = float(np.dot(a, a))
        scale = float(np.dot(a, y) / denom) if denom > 1.0e-30 else 0.0
        scalar_values.append(scale)
        pred_frame_scalar[frame] = scale * pred_bmean[frame]

    le_abs_rmse_bmean = rms(pred_bmean - le)
    le_abs_rmse_mean = rms(pred_mean_le - le)
    le_abs_rmse_zero = rms(pred_zero - le)
    le_abs_rmse_frame_scalar = rms(pred_frame_scalar - le)
    le_rms = rms(le)

    smoke = smoke_latest.get(case_id, {})
    smoke_final_le_rel = float_or_nan(smoke.get("train_LE_local_stack_rel", "nan"))
    smoke_final_ad_rel = float_or_nan(smoke.get("train_AD_B_local_useful_hat_rel", "nan"))
    smoke_final_b_raw_rel = float_or_nan(smoke.get("B_model_raw_rel", "nan"))
    smoke_final_step = int(float_or_nan(smoke.get("step", "nan"))) if smoke else None

    bmean_le_rel = rel_norm(pred_bmean - le, le)
    row = {
        "case_id": int(case_id),
        "compact_path": str(path),
        "frame_count": int(q.shape[0]),
        "point_count": int(le.shape[1]),
        "q_useful_hat_dim": int(q.shape[1]),
        "geometry_global_hat_dim": int(geom.shape[0]),
        "trunk_features_hat_dim": int(trunk.shape[1]),
        "LE_local_rms": le_rms,
        "LE_local_max_abs": float(np.max(np.abs(le))),
        "B_local_useful_rms": rms(b),
        "B_raw_rms": rms(b_raw),
        "q_useful_hat_rms": rms(q),
        "q_useful_hat_norm_min": norm_stats(q)["min"],
        "q_useful_hat_norm_max": norm_stats(q)["max"],
        "q_useful_hat_norm_mean": norm_stats(q)["mean"],
        "zero_LE_rel": rel_norm(pred_zero - le, le),
        "mean_LE_rel": rel_norm(pred_mean_le - le, le),
        "Bmean_at_q_LE_rel": bmean_le_rel,
        "Bmean_at_q_LE_abs_rmse": le_abs_rmse_bmean,
        "Bmean_at_q_LE_rmse_over_global_LE_rms": le_abs_rmse_bmean / max(le_rms, 1.0e-30),
        "frame_scalar_Bmean_at_q_LE_rel": rel_norm(pred_frame_scalar - le, le),
        "frame_scalar_Bmean_at_q_LE_abs_rmse": le_abs_rmse_frame_scalar,
        "frame_scalar_min": float(np.min(scalar_values)),
        "frame_scalar_max": float(np.max(scalar_values)),
        "frame_scalar_mean": float(np.mean(scalar_values)),
        "mean_LE_abs_rmse": le_abs_rmse_mean,
        "zero_LE_abs_rmse": le_abs_rmse_zero,
        "smoke_final_step": smoke_final_step,
        "smoke_final_LE_rel": smoke_final_le_rel,
        "smoke_final_AD_B_rel": smoke_final_ad_rel,
        "smoke_final_B_raw_rel": smoke_final_b_raw_rel,
        "smoke_final_minus_Bmean_LE_rel": smoke_final_le_rel - bmean_le_rel if math.isfinite(smoke_final_le_rel) else float("nan"),
        "likely_relative_error_inflated_by_small_LE": bool(le_rms < 1.0e-4 and bmean_le_rel > 0.5),
        "Bmean_anchor_bad": bool(bmean_le_rel > 0.1),
        "smoke_degraded_from_Bmean_anchor": bool(math.isfinite(smoke_final_le_rel) and smoke_final_le_rel > bmean_le_rel * 2.0),
    }
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    preferred = [
        "case_id",
        "LE_local_rms",
        "q_useful_hat_norm_mean",
        "B_local_useful_rms",
        "Bmean_at_q_LE_rel",
        "frame_scalar_Bmean_at_q_LE_rel",
        "mean_LE_rel",
        "zero_LE_rel",
        "smoke_final_LE_rel",
        "smoke_final_AD_B_rel",
        "smoke_final_B_raw_rel",
        "smoke_final_minus_Bmean_LE_rel",
        "Bmean_anchor_bad",
        "smoke_degraded_from_Bmean_anchor",
    ]
    fields = {key for row in rows for key in row}
    ordered = [key for key in preferred if key in fields]
    ordered.extend(sorted(fields.difference(ordered)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], *, out_root: Path, smoke_history: Path | None) -> dict[str, Any]:
    le_rel = np.asarray([float(row["Bmean_at_q_LE_rel"]) for row in rows], dtype=np.float64)
    smoke_rel = np.asarray([float(row["smoke_final_LE_rel"]) for row in rows], dtype=np.float64)
    finite_smoke = smoke_rel[np.isfinite(smoke_rel)]
    anchor_bad = [row["case_id"] for row in rows if bool(row["Bmean_anchor_bad"])]
    degraded = [row["case_id"] for row in rows if bool(row["smoke_degraded_from_Bmean_anchor"])]
    worst_bmean = sorted(rows, key=lambda row: float(row["Bmean_at_q_LE_rel"]), reverse=True)[:5]
    worst_smoke = sorted(rows, key=lambda row: float(row["smoke_final_LE_rel"]) if math.isfinite(float(row["smoke_final_LE_rel"])) else -1.0, reverse=True)[:5]
    return {
        "audit_name": "v3_css8_overfit_case_diagnosis",
        "out_root": str(out_root),
        "case_count": len(rows),
        "smoke_case_history": str(smoke_history) if smoke_history is not None else None,
        "Bmean_at_q_LE_rel_min": float(np.min(le_rel)),
        "Bmean_at_q_LE_rel_median": float(np.median(le_rel)),
        "Bmean_at_q_LE_rel_max": float(np.max(le_rel)),
        "smoke_final_LE_rel_min": float(np.min(finite_smoke)) if finite_smoke.size else None,
        "smoke_final_LE_rel_median": float(np.median(finite_smoke)) if finite_smoke.size else None,
        "smoke_final_LE_rel_max": float(np.max(finite_smoke)) if finite_smoke.size else None,
        "Bmean_anchor_bad_case_ids": anchor_bad,
        "smoke_degraded_from_Bmean_anchor_case_ids": degraded,
        "worst_Bmean_anchor_cases": [
            {"case_id": int(row["case_id"]), "Bmean_at_q_LE_rel": float(row["Bmean_at_q_LE_rel"]), "LE_local_rms": float(row["LE_local_rms"])}
            for row in worst_bmean
        ],
        "worst_smoke_final_cases": [
            {
                "case_id": int(row["case_id"]),
                "smoke_final_LE_rel": float(row["smoke_final_LE_rel"]),
                "Bmean_at_q_LE_rel": float(row["Bmean_at_q_LE_rel"]),
                "smoke_final_minus_Bmean_LE_rel": float(row["smoke_final_minus_Bmean_LE_rel"]),
            }
            for row in worst_smoke
        ],
        "formal_training": False,
        "uses_old_true176_labels_as_v3_labels": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", default="")
    parser.add_argument("--compact", action="append", default=[])
    parser.add_argument("--smoke-case-history", default="")
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    smoke_history = Path(args.smoke_case_history).resolve() if args.smoke_case_history else None
    smoke_latest = read_smoke_case_history(smoke_history)
    rows = [diagnose_one(path, smoke_latest) for path in load_paths(args)]
    rows = sorted(rows, key=lambda row: int(row["case_id"]))
    summary = summarize(rows, out_root=out_root, smoke_history=smoke_history)
    manifest_csv = out_root / "v3_overfit_case_diagnosis_manifest.csv"
    manifest_json = out_root / "v3_overfit_case_diagnosis_manifest.json"
    summary_json = out_root / "v3_overfit_case_diagnosis_summary.json"
    write_csv(manifest_csv, rows)
    manifest_json.write_text(json.dumps({"cases": rows}, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    print(json.dumps({**summary, "manifest_csv": str(manifest_csv), "manifest_json": str(manifest_json), "summary_json": str(summary_json)}, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
