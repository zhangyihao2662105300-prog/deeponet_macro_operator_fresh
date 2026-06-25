#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Gate 32 wind-shell generalization diagnosis.

This wrapper is read-only. It does not train a network, change model
structure, change q48/LE ordering, or change the 128-point rule.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_macro16_force_stiffness import json_default, write_json  # noqa: E402
from diagnose_macro16_wind_shell_generalization import run as run_diagnosis  # noqa: E402


DEFAULT_RUN_ROOT = Path("runs/gate31_force_residual_16case_qdef")


def find_checkpoint(train_dir: Path) -> Path:
    candidates = [
        train_dir / "best.pt",
        train_dir / "best.pth",
        train_dir / "checkpoint_best.pt",
        train_dir / "checkpoint_best.pth",
        train_dir / "latest.pt",
        train_dir / "latest.pth",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    hits = sorted(list(train_dir.glob("*.pt")) + list(train_dir.glob("*.pth")))
    if hits:
        return hits[0].resolve()
    raise FileNotFoundError(f"no checkpoint found under {train_dir}")


def resolve_existing(path: Path, name: str) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{name} not found: {resolved}")
    return resolved


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root).resolve()
    compact_list = resolve_existing(Path(args.compact_list) if str(args.compact_list) else run_root / "enriched_16case_compact_list.txt", "compact list")
    train_dir = resolve_existing(Path(args.train_dir) if str(args.train_dir) else run_root / "train" / str(args.run_name), "train dir")
    checkpoint = resolve_existing(Path(args.checkpoint), "checkpoint") if str(args.checkpoint) else find_checkpoint(train_dir)
    audit_dir = Path(args.audit_dir) if str(args.audit_dir) else run_root / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out) if str(args.out) else audit_dir / "gate32_wind_shell_generalization_diagnosis.json"
    report_out = Path(args.report_out) if str(args.report_out) else ROOT / "reports" / "32_wind_shell_generalization_diagnosis.md"
    audit_glob = str(args.audit_json_glob) if str(args.audit_json_glob) else str(audit_dir / f"{args.run_name}_case*.json")

    diagnosis_args = argparse.Namespace(
        compact_list=compact_list,
        out=out_json,
        report_out=report_out,
        train_cases=str(args.train_cases),
        wind_cases=str(args.wind_cases),
        special_cases=str(args.special_cases),
        checkpoint=str(checkpoint),
        audit_dir="",
        audit_json_glob=[audit_glob],
        source_path_map=list(args.source_path_map),
        plane_gauss_order=int(args.plane_gauss_order),
        thickness_gauss_order=int(args.thickness_gauss_order),
    )
    summary = run_diagnosis(diagnosis_args)
    wrapper_summary = {
        "script": "run_gate32_wind_shell_generalization_diagnosis.py",
        "run_root": str(run_root),
        "run_name": str(args.run_name),
        "compact_list": str(compact_list),
        "train_dir": str(train_dir),
        "checkpoint": str(checkpoint),
        "audit_glob": audit_glob,
        "out": str(out_json.resolve()),
        "report_out": str(report_out.resolve()),
        "diagnosis": summary.get("diagnosis", {}),
    }
    summary_out = Path(args.summary_out) if str(args.summary_out) else audit_dir / "gate32_wrapper_summary.json"
    write_json(summary_out.resolve(), wrapper_summary)
    wrapper_summary["summary_out"] = str(summary_out.resolve())
    return wrapper_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--run-name", default="f01_lr8e5")
    parser.add_argument("--compact-list", default="")
    parser.add_argument("--train-dir", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--audit-dir", default="")
    parser.add_argument("--audit-json-glob", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--report-out", default="")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--train-cases", default="19,25,31,41,43,44,45,46,49,50,60,61")
    parser.add_argument("--wind-cases", default="70,71,72,73")
    parser.add_argument("--special-cases", default="60")
    parser.add_argument("--source-path-map", action="append", default=[])
    parser.add_argument("--plane-gauss-order", type=int, default=3)
    parser.add_argument("--thickness-gauss-order", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
