#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit v2g formal prototype sanity without training a new formal model.

This script checks normalization roundtrips and no-training B@q/B-prior
baselines for the v2g formal prototype split.  It is diagnostic only: no old
TRUE176 labels are used, no checkpoints are written, and no formal training
claim is made.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

import train_v2_formal_prototype as v2g


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


def rel_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a).reshape(-1)) / max(np.linalg.norm(np.asarray(b).reshape(-1)), 1.0e-300))


def cos_np(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a).reshape(-1)
    bb = np.asarray(b).reshape(-1)
    return float(np.dot(aa, bb) / max(np.linalg.norm(aa) * np.linalg.norm(bb), 1.0e-300))


def rmse_np(diff: np.ndarray) -> float:
    vals = np.asarray(diff).reshape(-1)
    return float(np.sqrt(np.mean(vals * vals)))


def rms_np(vals: np.ndarray) -> float:
    arr = np.asarray(vals).reshape(-1)
    return float(np.sqrt(np.mean(arr * arr)))


def parse_cases(text: str) -> list[int]:
    return v2g.parse_cases(text)


def load_split(compact_list: Path, val_cases: list[int]) -> tuple[list[v2g.CaseData], list[int], list[int], dict[str, np.ndarray], dict[str, np.ndarray]]:
    cases = [v2g.load_case(path) for path in v2g.read_compact_list(compact_list)]
    case_ids = sorted(c.case_id for c in cases)
    train_cases = [case for case in case_ids if case not in set(val_cases)]
    missing = sorted(set(val_cases).difference(case_ids))
    if missing:
        raise SystemExit(f"val cases not found: {missing}")
    train_np = v2g.concat_cases([c for c in cases if c.case_id in set(train_cases)])
    val_np = v2g.concat_cases([c for c in cases if c.case_id in set(val_cases)])
    return cases, train_cases, val_cases, train_np, val_np


def normalization_from_v2g_summary(summary: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "q_std_floor": np.asarray(summary["q_std_floor"] if "q_std_floor" in summary else summary.get("q_std", []), dtype=np.float64),
        "le_std_floor": np.asarray(summary["LE_std_floor"], dtype=np.float64),
    }


def roundtrip_metrics(data: dict[str, np.ndarray], q_std: np.ndarray, le_std: np.ndarray) -> dict[str, float]:
    q = np.asarray(data["q"], dtype=np.float64)
    le = np.asarray(data["le"], dtype=np.float64)
    b = np.asarray(data["b"], dtype=np.float64)
    q_norm = q / q_std.reshape(1, 42)
    q_back = q_norm * q_std.reshape(1, 42)
    le_norm = le / le_std.reshape(1, 1, 6)
    le_back = le_norm * le_std.reshape(1, 1, 6)
    b_norm = b * q_std.reshape(1, 1, 1, 42) / le_std.reshape(1, 1, 6, 1)
    b_back = b_norm * le_std.reshape(1, 1, 6, 1) / q_std.reshape(1, 1, 1, 42)
    return {
        "q_roundtrip_rel": rel_np(q_back - q, q),
        "q_roundtrip_max_abs": float(np.max(np.abs(q_back - q))),
        "LE_roundtrip_rel": rel_np(le_back - le, le),
        "LE_roundtrip_max_abs": float(np.max(np.abs(le_back - le))),
        "B_roundtrip_rel": rel_np(b_back - b, b),
        "B_roundtrip_max_abs": float(np.max(np.abs(b_back - b))),
    }


def project_raw(b_local: np.ndarray, t_eps: np.ndarray, t_q: np.ndarray) -> np.ndarray:
    b_abq = np.einsum("npab,npbk->npak", t_eps, b_local)
    return np.einsum("npak,nkj->npaj", b_abq, t_q)


def baseline_metrics(data: dict[str, np.ndarray], b_hat: np.ndarray, le_hat: np.ndarray) -> dict[str, float]:
    le = np.asarray(data["le"], dtype=np.float64)
    b = np.asarray(data["b"], dtype=np.float64)
    b_raw = np.asarray(data["b_raw"], dtype=np.float64)
    b_model_raw = project_raw(b_hat, np.asarray(data["t_eps"], dtype=np.float64), np.asarray(data["t_q"], dtype=np.float64))
    t_q = np.asarray(data["t_q"], dtype=np.float64)
    p_useful = np.einsum("fki,fkj->fij", t_q, t_q)
    b_raw_projected = np.einsum("fpaj,fjk->fpak", b_raw, p_useful)
    return {
        "LE_local_rel": rel_np(le_hat - le, le),
        "LE_local_rmse": rmse_np(le_hat - le),
        "LE_target_rms": rms_np(le),
        "AD_B_local_rel": rel_np(b_hat - b, b),
        "AD_B_local_cos": cos_np(b_hat, b),
        "B_raw_projected_rel": rel_np(b_model_raw - b_raw_projected, b_raw_projected),
        "B_raw_rel": rel_np(b_model_raw - b_raw, b_raw),
    }


def exact_bq_baseline(data: dict[str, np.ndarray]) -> dict[str, float]:
    q = np.asarray(data["q"], dtype=np.float64)
    b = np.asarray(data["b"], dtype=np.float64)
    le_hat = np.einsum("npak,nk->npa", b, q)
    return {
        "LE_local_rel": rel_np(le_hat - data["le"], data["le"]),
        "LE_local_rmse": rmse_np(le_hat - data["le"]),
        "LE_target_rms": rms_np(data["le"]),
    }


def train_mean_b_baseline(train_data: dict[str, np.ndarray], data: dict[str, np.ndarray]) -> dict[str, float]:
    b_mean = np.asarray(train_data["b"], dtype=np.float64).mean(axis=0)
    q = np.asarray(data["q"], dtype=np.float64)
    b_hat = np.broadcast_to(b_mean.reshape(1, 128, 6, 42), (q.shape[0], 128, 6, 42)).copy()
    le_hat = np.einsum("npak,nk->npa", b_hat, q)
    return baseline_metrics(data, b_hat, le_hat)


def normalized_mean_b_baseline(
    train_data: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    q_std: np.ndarray,
    le_std: np.ndarray,
) -> dict[str, float]:
    b_train = np.asarray(train_data["b"], dtype=np.float64)
    b_norm_train = b_train * q_std.reshape(1, 1, 1, 42) / le_std.reshape(1, 1, 6, 1)
    b_norm_mean = b_norm_train.mean(axis=0)
    q_norm = np.asarray(data["q"], dtype=np.float64) / q_std.reshape(1, 42)
    le_norm_hat = np.einsum("pak,nk->npa", b_norm_mean, q_norm)
    le_hat = le_norm_hat * le_std.reshape(1, 1, 6)
    b_hat = np.broadcast_to((b_norm_mean * le_std.reshape(6, 1) / q_std.reshape(1, 42)).reshape(1, 128, 6, 42), (q_norm.shape[0], 128, 6, 42)).copy()
    return baseline_metrics(data, b_hat, le_hat)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pick_v2g_metrics(v2g_root: Path) -> dict[str, Any]:
    summary = read_json(v2g_root / "training_summary.json")
    return {
        "initial": read_json(v2g_root / "metrics_history.csv.json") if (v2g_root / "metrics_history.csv.json").exists() else None,
        "best_combined": summary["best_combined"],
        "best_LE": summary["best_LE"],
        "best_B": summary["best_B"],
        "best_raw_projected": summary["best_raw_projected"],
        "latest": summary["latest"],
    }


def first_history_row(v2g_root: Path) -> dict[str, Any]:
    with (v2g_root / "metrics_history.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    out: dict[str, Any] = {}
    for key, value in row.items():
        if value == "":
            continue
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    if "step" in out:
        out["step"] = int(out["step"])
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    compact_list = Path(args.compact_list).resolve()
    v2g_root = Path(args.v2g_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    val_cases = parse_cases(args.val_cases)
    cases, train_cases, val_cases, train_raw, val_raw = load_split(compact_list, val_cases)

    norm_summary = read_json(v2g_root / "normalization_summary.json")
    recomputed_norm = v2g.compute_normalization(train_raw)
    q_std = np.asarray(recomputed_norm["q_std_floor"], dtype=np.float64)
    le_std = np.asarray(recomputed_norm["le_std_floor"], dtype=np.float64)
    # v2g stores scalar normalization audit stats, not full q_std vectors, so
    # the exact vectors are recomputed here with the same train-only helper.
    norm_vector_source = "recomputed_train_only"

    roundtrip = {
        "train": roundtrip_metrics(train_raw, q_std, le_std),
        "val": roundtrip_metrics(val_raw, q_std, le_std),
    }
    exact_bq = {
        "train": exact_bq_baseline(train_raw),
        "val": exact_bq_baseline(val_raw),
    }
    train_mean_b = {
        "train": train_mean_b_baseline(train_raw, train_raw),
        "val": train_mean_b_baseline(train_raw, val_raw),
    }
    normalized_mean_b = {
        "train": normalized_mean_b_baseline(train_raw, train_raw, q_std, le_std),
        "val": normalized_mean_b_baseline(train_raw, val_raw, q_std, le_std),
    }

    v2g_metrics = pick_v2g_metrics(v2g_root)
    v2g_metrics["initial"] = first_history_row(v2g_root)

    rows: list[dict[str, Any]] = []
    for name, payload in [
        ("exact_Bq", exact_bq),
        ("train_mean_B", train_mean_b),
        ("normalized_train_mean_B", normalized_mean_b),
    ]:
        for split, metrics in payload.items():
            rows.append({"baseline": name, "split": split, **metrics})
    with (out_root / "baseline_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "audit_name": "v2h_formal_prototype_sanity",
        "compact_list": str(compact_list),
        "v2g_root": str(v2g_root),
        "out_root": str(out_root),
        "compact_count": len(cases),
        "train_cases": train_cases,
        "val_cases": val_cases,
        "normalization_vector_source": norm_vector_source,
        "normalization_train_only": True,
        "roundtrip": roundtrip,
        "exact_Bq_baseline": exact_bq,
        "train_mean_B_prior_baseline": train_mean_b,
        "normalized_train_mean_B_prior_baseline": normalized_mean_b,
        "v2g_metrics": v2g_metrics,
        "uses_old_true176_labels_as_v2_labels": False,
        "formal_training_result": False,
        "output_files": {
            "summary": str(out_root / "sanity_summary.json"),
            "baseline_metrics": str(out_root / "baseline_metrics.csv"),
        },
    }
    (out_root / "sanity_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", required=True)
    parser.add_argument("--v2g-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--val-cases", default="44,46")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
