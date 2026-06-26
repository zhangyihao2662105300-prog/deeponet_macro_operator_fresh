#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plan a multi-machine wind-shell Abaqus data campaign.

The generated CSV is consumed by ``run_gate04_wind_shell_generality_audit.py``.
It keeps the current boundary-control route: TRUE176 full48 displacement
templates are transferred to target shell keep-node frames, then Abaqus exports
fresh LE/B labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


FAMILIES = (
    "cylindrical_shell",
    "conical_shell",
    "thickness_varying_shell",
    "mild_double_curvature_shell",
    "distorted_shell",
)


def ps_quote(path: str | Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def geometry_rows(*, variants_per_family: int, templates_per_geometry: int, template_pool: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    machine_for_family = {
        "cylindrical_shell": "winA",
        "conical_shell": "winA",
        "thickness_varying_shell": "winB",
        "mild_double_curvature_shell": "winB",
        "distorted_shell": "winA",
    }
    amp_cycle = [0.75, 1.0, 1.25]
    for family_i, family in enumerate(FAMILIES):
        for gi in range(int(variants_per_family)):
            params: dict[str, float] = {}
            if family == "cylindrical_shell":
                params = {
                    "theta": 0.34 + 0.055 * gi,
                    "lam": 0.90 + 0.035 * (gi % 4),
                    "tau": 0.016 + 0.002 * (gi % 5),
                }
            elif family == "conical_shell":
                params = {
                    "theta": 0.32 + 0.050 * gi,
                    "lam": 0.95 + 0.030 * (gi % 4),
                    "tau": 0.016 + 0.002 * (gi % 5),
                    "mu": -0.18 + 0.06 * (gi % 7),
                }
            elif family == "thickness_varying_shell":
                params = {
                    "theta": 0.30 + 0.045 * gi,
                    "lam": 0.92 + 0.035 * (gi % 4),
                    "tau0": 0.017 + 0.002 * (gi % 5),
                    "grad_u": -0.24 + 0.08 * (gi % 7),
                    "grad_v": -0.12 + 0.06 * (gi % 5),
                }
            elif family == "mild_double_curvature_shell":
                params = {
                    "lam": 0.92 + 0.035 * (gi % 5),
                    "tau": 0.016 + 0.002 * (gi % 5),
                    "cu": -0.060 + 0.025 * (gi % 7),
                    "cv": -0.045 + 0.020 * ((gi + 2) % 7),
                    "cxy": -0.030 + 0.015 * (gi % 5),
                    "twist": -0.025 + 0.0125 * ((gi + 1) % 5),
                }
            elif family == "distorted_shell":
                params = {
                    "lam": 0.94 + 0.030 * (gi % 5),
                    "tau": 0.016 + 0.002 * (gi % 5),
                    "distort_amp": 0.012 + 0.004 * (gi % 8),
                    "distort_u": 1.0 + 0.25 * (gi % 3),
                    "distort_v": 1.0 + 0.25 * ((gi + 1) % 3),
                }
            geometry_id = f"{family}_g{gi:02d}"
            for ti in range(int(templates_per_geometry)):
                template_case = int(template_pool[(family_i * variants_per_family * templates_per_geometry + gi * templates_per_geometry + ti) % len(template_pool)])
                case_id = f"{geometry_id}_t176{template_case:03d}_a{ti}"
                rows.append(
                    {
                        "case_id": case_id,
                        "geometry_id": geometry_id,
                        "family": family,
                        "machine_group": machine_for_family[family] if not (family == "distorted_shell" and gi % 2) else "winB",
                        "template_case_id": template_case,
                        "template_amplitude_scale": amp_cycle[ti % len(amp_cycle)],
                        **params,
                        "geometry_params_json": json.dumps(params, sort_keys=True),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "geometry_id",
        "family",
        "machine_group",
        "template_case_id",
        "template_amplitude_scale",
        "theta",
        "lam",
        "tau",
        "mu",
        "tau0",
        "grad_u",
        "grad_v",
        "cu",
        "cv",
        "cxy",
        "twist",
        "distort_amp",
        "distort_u",
        "distort_v",
        "geometry_params_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def worker_script(*, repo: Path, spec: Path, out_root: Path, machine_group: str, inner_workers: int, post_workers: int) -> str:
    return f"""$ErrorActionPreference = 'Stop'
cd {ps_quote(repo)}
$env:PYTHONPATH = 'src;scripts'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
py -3 scripts\\run_gate04_wind_shell_generality_audit.py `
  --out-root {ps_quote(out_root)} `
  --geometry-spec {ps_quote(spec)} `
  --machine-group {machine_group} `
  --q-source-mode true176-template `
  --increments 100 `
  --inner-workers {int(inner_workers)} `
  --post-workers {int(post_workers)} `
  --skip-existing
"""


def linux_train_script(*, github_copy: Path, remote_data_root: str, out_dir: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd "{github_copy.as_posix()}"
export PYTHONPATH=code/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

torchrun --nproc_per_node=2 -m macro_deeponet.train_true176_generic_sobolev \\
  --compact-list "{remote_data_root}/complete_compact_list_for_training.txt" \\
  --out-dir "{out_dir}" \\
  --epochs 2000 \\
  --cuda --ddp --ddp-backend nccl \\
  --branch-feature-mode xkeep-qraw \\
  --q-input-mode rigid-removed \\
  --point-feature-source data \\
  --trunk-input-mode query-physical-xyz \\
  --b-base-point-input-mode same-as-trunk \\
  --scale-mode physical \\
  --geometry-frame-mode centered-global \\
  --model-style fe-linear-residual \\
  --basis-dim 96 --hidden-dim 384 --branch-depth 5 --trunk-depth 5 --activation tanh \\
  --jacobian-columns all --jacobian-columns-per-batch 48 \\
  --j-loss-mode balanced-ip-col --baseline-j-loss-mode balanced-ip-col \\
  --jacobian-target-mode residual-b --baseline-jacobian-weight 0.0 \\
  --initial-jacobian-weight 5.0 \\
  --batch-size 8 --eval-batch-size 1 --weight-decay 1e-5 \\
  --eval-every 10 --max-eval-frames 512
"""


def parse_template_pool(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).replace(";", ",").split(","):
        item = part.strip()
        if not item:
            continue
        if ":" in item:
            lo, hi = [int(v.strip()) for v in item.split(":", 1)]
            out.extend(range(lo, hi))
        else:
            out.append(int(item))
    if not out:
        raise ValueError("empty template pool")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs" / "wind_shell_abaqus_campaign")
    parser.add_argument("--variants-per-family", type=int, default=8)
    parser.add_argument("--templates-per-geometry", type=int, default=3)
    parser.add_argument("--template-pool", default="1:25")
    parser.add_argument("--inner-workers", type=int, default=4)
    parser.add_argument("--post-workers", type=int, default=8)
    parser.add_argument("--remote-data-root", default="/home/ydh/zhangyihao/wind_shell_campaign")
    parser.add_argument("--linux-github-copy", type=Path, default=Path("/home/ydh/zhangyihao/deeponet_2000epoch_remote_snapshot_20260624_github_copy"))
    parser.add_argument("--linux-out-dir", default="/home/ydh/zhangyihao/run_logs/wind_shell_query_xyz_2000")
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    spec = out_root / "wind_shell_geometry_spec.csv"
    rows = geometry_rows(
        variants_per_family=int(args.variants_per_family),
        templates_per_geometry=int(args.templates_per_geometry),
        template_pool=parse_template_pool(str(args.template_pool)),
    )
    write_csv(spec, rows)
    write_counts = {
        "total_tasks": len(rows),
        "by_machine": {},
        "by_family": {},
        "abaqus_jobs_per_task": 49,
        "estimated_abaqus_jobs": 49 * len(rows),
    }
    for row in rows:
        write_counts["by_machine"][row["machine_group"]] = int(write_counts["by_machine"].get(row["machine_group"], 0)) + 1
        write_counts["by_family"][row["family"]] = int(write_counts["by_family"].get(row["family"], 0)) + 1
    (out_root / "campaign_summary.json").write_text(json.dumps(write_counts, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    (out_root / "run_winA.ps1").write_text(
        worker_script(
            repo=ROOT,
            spec=spec,
            out_root=out_root / "winA",
            machine_group="winA",
            inner_workers=int(args.inner_workers),
            post_workers=int(args.post_workers),
        ),
        encoding="utf-8",
    )
    (out_root / "run_winB.ps1").write_text(
        worker_script(
            repo=ROOT,
            spec=spec,
            out_root=out_root / "winB",
            machine_group="winB",
            inner_workers=int(args.inner_workers),
            post_workers=int(args.post_workers),
        ),
        encoding="utf-8",
    )
    (out_root / "train_linux_2000.sh").write_text(
        linux_train_script(
            github_copy=Path(args.linux_github_copy),
            remote_data_root=str(args.remote_data_root),
            out_dir=str(args.linux_out_dir),
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out_root": str(out_root), "spec": str(spec), **write_counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
