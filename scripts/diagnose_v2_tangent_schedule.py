#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run and summarize v2k tangent / training-schedule diagnostics.

This script orchestrates short prototype runs only.  It does not use old
TRUE176 labels and does not write checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


RUN_SPECS = [
    {
        "name": "A_full_output_joint",
        "b_loss_target_mode": "full_output",
        "training_schedule": "joint",
        "steps": None,
    },
    {
        "name": "B_anchor_only_joint",
        "b_loss_target_mode": "anchor_only",
        "training_schedule": "joint",
        "steps": None,
    },
    {
        "name": "C_residual_only_joint",
        "b_loss_target_mode": "residual_only",
        "training_schedule": "joint",
        "steps": None,
    },
    {
        "name": "D_full_output_le_warmup",
        "b_loss_target_mode": "full_output",
        "training_schedule": "le_warmup_then_b",
        "steps": None,
    },
    {
        "name": "E_detached_anchor_plus_residual",
        "b_loss_target_mode": "detached_anchor_plus_residual",
        "training_schedule": "anchor_then_residual",
        "steps": None,
    },
]


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


def pick_metrics(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "step",
        "train_LE_local_rel",
        "val_LE_local_rel",
        "train_AD_B_local_rel",
        "val_AD_B_local_rel",
        "train_AD_B_local_cos",
        "val_AD_B_local_cos",
        "train_B_model_raw_projected_rel",
        "val_B_model_raw_projected_rel",
        "train_zero_q_LE_local_rms",
        "val_zero_q_LE_local_rms",
    ]
    return {key: row.get(key) for key in keys if key in row}


def read_per_case(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def flatten_summary_rows(run_summaries: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, data in run_summaries.items():
        meta = {
            "run": name,
            "b_loss_target_mode": data["b_loss_target_mode"],
            "training_schedule": data["training_schedule"],
            "warmup_steps": data["warmup_steps"],
        }
        for label in ("best_combined", "best_LE", "best_B", "best_raw_projected", "latest"):
            row = data.get(label, {})
            rows.append({"metric_set": label, **meta, **pick_metrics(row)})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "run",
        "metric_set",
        "b_loss_target_mode",
        "training_schedule",
        "warmup_steps",
        "step",
        "train_LE_local_rel",
        "val_LE_local_rel",
        "train_AD_B_local_rel",
        "val_AD_B_local_rel",
        "train_AD_B_local_cos",
        "val_AD_B_local_cos",
        "train_B_model_raw_projected_rel",
        "val_B_model_raw_projected_rel",
        "val_zero_q_LE_local_rms",
    ]
    ordered = [key for key in preferred if key in fields]
    ordered.extend(key for key in fields if key not in ordered)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def run_one(args: argparse.Namespace, spec: dict[str, Any], script_path: Path) -> dict[str, Any]:
    out_root = Path(args.out_root).resolve() / spec["name"]
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.out_root).resolve() / f"{spec['name']}.log"
    steps = int(spec["steps"] or args.steps)
    cmd = [
        sys.executable,
        str(script_path),
        "--compact-list",
        str(Path(args.compact_list).resolve()),
        "--out-root",
        str(out_root),
        "--train-cases",
        str(args.train_cases),
        "--val-cases",
        str(args.val_cases),
        "--steps",
        str(steps),
        "--eval-every",
        str(args.eval_every),
        "--seed",
        str(args.seed),
        "--use-q-amp",
        "--b-prior-mode",
        "amp_p75_cluster",
        "--detach-b-prior-regime-weight",
        "--b-loss-target-mode",
        str(spec["b_loss_target_mode"]),
        "--training-schedule",
        str(spec["training_schedule"]),
        "--warmup-steps",
        str(args.warmup_steps),
    ]
    if args.device:
        cmd.extend(["--device", str(args.device)])
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=str(script_path.parent.parent), stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = read_json(out_root / "training_summary.json")
    return {
        "name": spec["name"],
        "out_root": str(out_root),
        "log_path": str(log_path),
        "b_loss_target_mode": spec["b_loss_target_mode"],
        "training_schedule": spec["training_schedule"],
        "warmup_steps": int(args.warmup_steps),
        "anchor_metrics": summary.get("B_prior_anchor_metrics", {}),
        "best_combined": summary.get("best_combined", {}),
        "best_LE": summary.get("best_LE", {}),
        "best_B": summary.get("best_B", {}),
        "best_raw_projected": summary.get("best_raw_projected", {}),
        "latest": summary.get("latest", {}),
        "training_summary": str(out_root / "training_summary.json"),
        "metrics_history": str(out_root / "metrics_history.csv"),
        "per_case_attribution": read_per_case(out_root / "per_case_attribution.csv"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve().parent / "train_v2_formal_prototype.py"
    run_summaries: dict[str, Any] = {}
    selected = {item.strip() for item in str(args.only).split(",") if item.strip()}
    for spec in RUN_SPECS:
        if selected and spec["name"] not in selected:
            continue
        run_summaries[spec["name"]] = run_one(args, spec, script_path)

    rows = flatten_summary_rows(run_summaries)
    write_csv(out_root / "v2k_tangent_schedule_summary.csv", rows)
    summary = {
        "audit_name": "v2k_tangent_schedule_diagnostic",
        "compact_list": str(Path(args.compact_list).resolve()),
        "out_root": str(out_root),
        "train_cases": [int(v) for v in str(args.train_cases).split(",") if v.strip()],
        "val_cases": [int(v) for v in str(args.val_cases).split(",") if v.strip()],
        "fixed_b_prior_mode": "amp_p75_cluster",
        "fixed_detach_b_prior_regime_weight": True,
        "fixed_use_q_amp": True,
        "steps": int(args.steps),
        "eval_every": int(args.eval_every),
        "seed": int(args.seed),
        "warmup_steps": int(args.warmup_steps),
        "run_summaries": run_summaries,
        "formal_training_result": False,
        "uses_old_true176_labels_as_v2_labels": False,
        "checkpoint_written": False,
        "output_files": {
            "summary": str(out_root / "v2k_tangent_schedule_summary.json"),
            "summary_csv": str(out_root / "v2k_tangent_schedule_summary.csv"),
        },
    }
    (out_root / "v2k_tangent_schedule_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--train-cases", default="19,25,31,41,43,45,49,50")
    parser.add_argument("--val-cases", default="44,46")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--device", default="")
    parser.add_argument("--only", default="", help="optional comma-separated run names")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
