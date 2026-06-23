#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose value-field correction failures for the v2 formal prototype.

This script is diagnostic only.  It computes B-prior value-anchor baselines and
optionally summarizes v2i training runs.  It does not train a model, does not
use old TRUE176 labels, and does not write checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import audit_v2_formal_prototype_sanity as sanity
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_run_roots(text: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in str(text).split(";"):
        part = item.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--run-roots entry must be name=path, got {part!r}")
        name, path = part.split("=", 1)
        out[name.strip()] = Path(path.strip()).resolve()
    return out


def load_case_data(compact_list: Path, train_cases: list[int], val_cases: list[int]) -> tuple[list[v2g.CaseData], dict[str, np.ndarray], dict[str, np.ndarray]]:
    cases = [v2g.load_case(path) for path in v2g.read_compact_list(compact_list)]
    case_ids = {c.case_id for c in cases}
    missing = sorted(set(train_cases + val_cases).difference(case_ids))
    if missing:
        raise SystemExit(f"cases not found: {missing}")
    train = v2g.concat_cases([c for c in cases if c.case_id in set(train_cases)])
    val = v2g.concat_cases([c for c in cases if c.case_id in set(val_cases)])
    return cases, train, val


def subset_by_case(data: dict[str, np.ndarray], case_id: int) -> dict[str, np.ndarray]:
    mask = np.asarray(data["case_ids"]) == int(case_id)
    return {key: (value[mask] if isinstance(value, np.ndarray) and value.shape[:1] == mask.shape else value) for key, value in data.items()}


def cluster_labels_by_amp(data: dict[str, np.ndarray], threshold: float) -> np.ndarray:
    q_amp = np.linalg.norm(np.asarray(data["q"], dtype=np.float64), axis=1)
    return (q_amp > float(threshold)).astype(np.int64)


def cluster_b_prior_baseline(train_data: dict[str, np.ndarray], data: dict[str, np.ndarray], threshold: float) -> dict[str, float]:
    train_labels = cluster_labels_by_amp(train_data, threshold)
    labels = cluster_labels_by_amp(data, threshold)
    b_train = np.asarray(train_data["b"], dtype=np.float64)
    b_means: dict[int, np.ndarray] = {}
    global_mean = b_train.mean(axis=0)
    for label in (0, 1):
        idx = train_labels == label
        b_means[label] = b_train[idx].mean(axis=0) if np.any(idx) else global_mean
    b_hat = np.stack([b_means[int(label)] for label in labels], axis=0)
    le_hat = np.einsum("npak,nk->npa", b_hat, np.asarray(data["q"], dtype=np.float64))
    metrics = sanity.baseline_metrics(data, b_hat, le_hat)
    metrics["amp_threshold"] = float(threshold)
    metrics["low_frame_count"] = int(np.sum(labels == 0))
    metrics["high_frame_count"] = int(np.sum(labels == 1))
    return metrics


def per_case_b_prior_upper_bound(data: dict[str, np.ndarray]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    case_ids = sorted(set(int(v) for v in np.asarray(data["case_ids"]).tolist()))
    b_hat_chunks = []
    le_hat_chunks = []
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        sub = subset_by_case(data, case_id)
        b_mean = np.asarray(sub["b"], dtype=np.float64).mean(axis=0)
        b_hat = np.broadcast_to(b_mean[None, ...], (sub["q"].shape[0],) + b_mean.shape).copy()
        le_hat = np.einsum("npak,nk->npa", b_hat, np.asarray(sub["q"], dtype=np.float64))
        metrics = sanity.baseline_metrics(sub, b_hat, le_hat)
        rows.append({"case_id": case_id, **metrics})
        b_hat_chunks.append(b_hat)
        le_hat_chunks.append(le_hat)
    b_all = np.concatenate(b_hat_chunks, axis=0)
    le_all = np.concatenate(le_hat_chunks, axis=0)
    return sanity.baseline_metrics(data, b_all, le_all), rows


def per_case_oracle_rows(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for case_id in sorted(set(int(v) for v in np.asarray(data["case_ids"]).tolist())):
        sub = subset_by_case(data, case_id)
        exact = sanity.exact_bq_baseline(sub)
        mean_metrics = sanity.train_mean_b_baseline(sub, sub)
        rows.append(
            {
                "case_id": int(case_id),
                "frame_count": int(sub["q"].shape[0]),
                "q_amp_mean": float(np.mean(np.linalg.norm(sub["q"], axis=1))),
                "LE_target_rms": sanity.rms_np(sub["le"]),
                "exact_Bq_LE_rel": exact["LE_local_rel"],
                "case_mean_B_LE_rel": mean_metrics["LE_local_rel"],
                "case_mean_B_AD_B_rel": mean_metrics["AD_B_local_rel"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(run_roots: dict[str, Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, root in run_roots.items():
        summary_path = root / "training_summary.json"
        if not summary_path.exists():
            out[name] = {"missing": str(summary_path)}
            continue
        data = read_json(summary_path)
        item: dict[str, Any] = {
            "root": str(root),
            "best_combined": data.get("best_combined"),
            "latest": data.get("latest"),
            "train_cases": data.get("train_cases"),
            "val_cases": data.get("val_cases"),
            "use_q_dir": data.get("use_q_dir"),
            "use_amp_regime_descriptor": data.get("use_amp_regime_descriptor"),
            "le_weight": None,
            "b_weight": None,
        }
        per_case_path = root / "per_case_attribution.csv"
        if per_case_path.exists():
            with per_case_path.open(newline="", encoding="utf-8") as f:
                item["per_case_attribution"] = list(csv.DictReader(f))
        history_path = root / "per_case_history.csv"
        if history_path.exists():
            with history_path.open(newline="", encoding="utf-8") as f:
                item["per_case_history_tail"] = list(csv.DictReader(f))[-20:]
        out[name] = item
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    compact_list = Path(args.compact_list).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    train_cases = v2g.parse_cases(args.train_cases)
    val_cases = v2g.parse_cases(args.val_cases)
    cases, train_data, val_data = load_case_data(compact_list, train_cases, val_cases)

    q_amp_train = np.linalg.norm(np.asarray(train_data["q"], dtype=np.float64), axis=1)
    amp_median = float(np.median(q_amp_train))
    amp_p75 = float(np.percentile(q_amp_train, 75.0))
    clustered_median = {
        "train": cluster_b_prior_baseline(train_data, train_data, amp_median),
        "val": cluster_b_prior_baseline(train_data, val_data, amp_median),
    }
    clustered_p75 = {
        "train": cluster_b_prior_baseline(train_data, train_data, amp_p75),
        "val": cluster_b_prior_baseline(train_data, val_data, amp_p75),
    }
    train_case_upper, train_case_rows = per_case_b_prior_upper_bound(train_data)
    val_exact = sanity.exact_bq_baseline(val_data)

    rows = []
    for split, metrics in clustered_median.items():
        rows.append({"baseline": "clustered_B_amp_median", "split": split, **metrics})
    for split, metrics in clustered_p75.items():
        rows.append({"baseline": "clustered_B_amp_p75", "split": split, **metrics})
    rows.append({"baseline": "per_case_B_upper_bound", "split": "train", **train_case_upper})
    write_csv(out_root / "value_anchor_baselines.csv", rows)
    write_csv(out_root / "per_case_oracle_rows.csv", per_case_oracle_rows(train_data))
    write_csv(out_root / "per_case_B_upper_bound_rows.csv", train_case_rows)

    run_roots = parse_run_roots(args.run_roots)
    summary = {
        "audit_name": "v2i_value_field_correction_diagnostics",
        "compact_list": str(compact_list),
        "out_root": str(out_root),
        "case_ids": sorted(c.case_id for c in cases),
        "train_cases": train_cases,
        "val_cases": val_cases,
        "q_amp_train_median": amp_median,
        "q_amp_train_p75": amp_p75,
        "clustered_B_prior_amp_median": clustered_median,
        "clustered_B_prior_amp_p75": clustered_p75,
        "per_case_B_prior_train_upper_bound": train_case_upper,
        "val_exact_Bq": val_exact,
        "run_summaries": summarize_runs(run_roots),
        "uses_old_true176_labels_as_v2_labels": False,
        "formal_training_result": False,
        "output_files": {
            "summary": str(out_root / "value_correction_summary.json"),
            "value_anchor_baselines": str(out_root / "value_anchor_baselines.csv"),
            "per_case_oracle_rows": str(out_root / "per_case_oracle_rows.csv"),
            "per_case_B_upper_bound_rows": str(out_root / "per_case_B_upper_bound_rows.csv"),
        },
    }
    (out_root / "value_correction_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--train-cases", default="19,25,31,41,43,45,49,50")
    parser.add_argument("--val-cases", default="44,46")
    parser.add_argument("--run-roots", default="")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
