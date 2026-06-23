#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a one-case v2 oracle smoke on a local-strain pilot compact.

This script is read-only with respect to the input compact.  It does not train a
model and does not treat old TRUE176 LE/B values as v2 labels.

The checked relations are:

    LE_local_Bq = B_local_useful @ q_useful
    B_raw_hat   = T_eps_to_abq @ B_local_useful @ T_q_raw_to_useful

The first relation is a linear B@q oracle and may be approximate for nonlinear
LE data.  The second relation is the v2 chain-rule contract.
"""

from __future__ import annotations

import argparse
import csv
import json
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


def rel_norm(num: np.ndarray, den: np.ndarray) -> float:
    n = float(np.linalg.norm(np.asarray(num, dtype=np.float64).reshape(-1)))
    d = float(np.linalg.norm(np.asarray(den, dtype=np.float64).reshape(-1)))
    return n / max(d, 1.0e-30)


def rms(value: np.ndarray) -> float:
    vals = np.asarray(value, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(vals * vals)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = max(float(np.linalg.norm(aa) * np.linalg.norm(bb)), 1.0e-30)
    return float(np.dot(aa, bb) / denom)


def max_abs(value: np.ndarray) -> float:
    vals = np.asarray(value, dtype=np.float64)
    return float(np.max(np.abs(vals))) if vals.size else 0.0


def find_b_useful_key(files: list[str]) -> str:
    for key in B_USEFUL_KEYS:
        if key in files:
            return key
    raise KeyError(f"missing one of {B_USEFUL_KEYS}")


def broadcast_t_eps(t_eps: np.ndarray, n_frames: int) -> np.ndarray:
    vals = np.asarray(t_eps, dtype=np.float64)
    if vals.shape == (128, 6, 6):
        return np.broadcast_to(vals.reshape(1, 128, 6, 6), (int(n_frames), 128, 6, 6)).copy()
    if vals.shape == (int(n_frames), 128, 6, 6):
        return vals.copy()
    raise ValueError(f"T_eps_to_abq must be [128,6,6] or [N,128,6,6], got {vals.shape}")


def compute_oracle(compact: Path) -> dict[str, Any]:
    with np.load(str(compact), allow_pickle=True) as z:
        required = {
            "q_useful",
            "LE128_local",
            "T_q_raw_to_useful",
            "T_eps_to_abq",
            "B_LE128_forward",
            "q48_raw",
        }
        missing = sorted(required.difference(z.files))
        if missing:
            raise KeyError(f"{compact}: missing required v2c oracle fields {missing}")

        b_key = find_b_useful_key(list(z.files))
        q_useful = np.asarray(z["q_useful"], dtype=np.float64)
        le_local = np.asarray(z["LE128_local"], dtype=np.float64)
        b_local = np.asarray(z[b_key], dtype=np.float64)
        t_q = np.asarray(z["T_q_raw_to_useful"], dtype=np.float64)
        t_eps_to_abq = np.asarray(z["T_eps_to_abq"], dtype=np.float64)
        b_raw = np.asarray(z["B_LE128_forward"], dtype=np.float64)
        q48_raw = np.asarray(z["q48_raw"], dtype=np.float64)
        t_eps_from_key = "T_eps_from_abq" if "T_eps_from_abq" in z.files else None
        metadata = {
            "B_label_q_coordinate": scalar_text(z["B_label_q_coordinate"]) if "B_label_q_coordinate" in z.files else "unknown",
            "B_label_output_coordinate": scalar_text(z["B_label_output_coordinate"]) if "B_label_output_coordinate" in z.files else "unknown",
            "strain_output_coordinate": scalar_text(z["strain_output_coordinate"]) if "strain_output_coordinate" in z.files else "unknown",
            "uses_old_true176_labels_as_v2_labels": bool(np.asarray(z["uses_old_true176_labels_as_v2_labels"]).reshape(-1)[0])
            if "uses_old_true176_labels_as_v2_labels" in z.files
            else False,
            "T_eps_from_abq_present": bool(t_eps_from_key),
        }

    if q_useful.ndim != 2:
        raise ValueError(f"q_useful must be [N,K], got {q_useful.shape}")
    if le_local.shape != (q_useful.shape[0], 128, 6):
        raise ValueError(f"LE128_local must be [N,128,6], got {le_local.shape}")
    if b_local.shape != (q_useful.shape[0], 128, 6, q_useful.shape[1]):
        raise ValueError(f"{b_key} must be [N,128,6,K], got {b_local.shape}")
    if t_q.shape != (q_useful.shape[1], 48):
        raise ValueError(f"T_q_raw_to_useful must be [K,48], got {t_q.shape}")
    if b_raw.shape != (q_useful.shape[0], 128, 6, 48):
        raise ValueError(f"B_LE128_forward must be [N,128,6,48], got {b_raw.shape}")
    if q48_raw.shape != (q_useful.shape[0], 48):
        raise ValueError(f"q48_raw must be [N,48], got {q48_raw.shape}")

    le_local_bq = np.einsum("npak,nk->npa", b_local, q_useful)
    le_diff = le_local_bq - le_local

    t_eps_full = broadcast_t_eps(t_eps_to_abq, q_useful.shape[0])
    b_abq_from_local = np.einsum("npab,npbk->npak", t_eps_full, b_local)
    b_raw_hat = np.einsum("npak,kj->npaj", b_abq_from_local, t_q)
    p_useful = t_q.T @ t_q
    b_raw_projected = np.einsum("npaj,jk->npak", b_raw, p_useful)

    summary: dict[str, Any] = {
        "audit_name": "v2c_one_case_oracle_smoke",
        "compact": str(compact),
        "frame_count": int(q_useful.shape[0]),
        "point_count": int(le_local.shape[1]),
        "q_useful_dim": int(q_useful.shape[1]),
        "q48_raw_dim": int(q48_raw.shape[1]),
        "LE128_local_shape": list(le_local.shape),
        "B_useful_key": b_key,
        "B_useful_shape": list(b_local.shape),
        "T_q_raw_to_useful_shape": list(t_q.shape),
        "T_eps_to_abq_shape": list(np.asarray(t_eps_to_abq).shape),
        "B_LE128_forward_shape": list(b_raw.shape),
        "LE_local_Bq_rel": rel_norm(le_diff, le_local),
        "LE_local_Bq_cos": cosine(le_local_bq, le_local),
        "LE_local_Bq_rms": rms(le_local_bq),
        "LE_local_target_rms": rms(le_local),
        "LE_local_Bq_diff_rms": rms(le_diff),
        "LE_local_Bq_max_abs": max_abs(le_diff),
        "B_raw_hat_projected_rel": rel_norm(b_raw_hat - b_raw_projected, b_raw_projected),
        "B_raw_hat_projected_max_abs": max_abs(b_raw_hat - b_raw_projected),
        "B_raw_hat_raw_rel": rel_norm(b_raw_hat - b_raw, b_raw),
        "B_raw_hat_raw_max_abs": max_abs(b_raw_hat - b_raw),
        "B_rigid_residual_rel": rel_norm(b_raw_projected - b_raw, b_raw),
        "B_rigid_residual_max_abs": max_abs(b_raw_projected - b_raw),
        "zero_q_local_LE_rms_by_oracle": 0.0,
        "model_training_performed": False,
        "uses_old_true176_labels_as_v2_labels": False,
        **metadata,
    }
    return summary


def write_outputs(summary: dict[str, Any], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "oracle_summary.json"
    metrics_path = out_root / "oracle_metrics.csv"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in sorted(summary.items()):
            if isinstance(value, (int, float, str, bool)) or value is None:
                writer.writerow([key, value])
    print(
        json.dumps(
            {**summary, "oracle_summary_path": str(summary_path), "oracle_metrics_path": str(metrics_path)},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=json_default,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", required=True, help="Input v2b local-strain pilot compact")
    parser.add_argument("--out-root", required=True, help="Output directory for oracle summary files")
    args = parser.parse_args()

    compact = Path(args.compact).resolve()
    if not compact.exists():
        raise FileNotFoundError(compact)
    summary = compute_oracle(compact)
    write_outputs(summary, Path(args.out_root).resolve())


if __name__ == "__main__":
    main()
