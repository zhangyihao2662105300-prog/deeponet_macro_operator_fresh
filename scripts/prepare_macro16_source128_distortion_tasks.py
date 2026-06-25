#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare missing distorted-geometry tasks for the Macro16 source128 audit.

This script does not train a network.  It writes independent q48 path tasks for
the missing medium and strong-but-not-flipped geometries so Abaqus can generate
fresh teacher compacts for the standard 128-point Macro16 audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from macro_deeponet.macro16_geometry import Macro16GeometryMap, macro16_source128_point_table  # noqa: E402
from plan_macro16_source128_generality_audit import q_modes_for_geometry, q_program_definitions  # noqa: E402

DEFAULT_OLD_SRC_ROOT = Path(r"D:\IS-FEM\NNSE_css8_push_tmp")
DEFAULT_REMOTE_ABAQUS = Path(r"D:\SIMULIA\Commands\abaqus.bat")

TRUE176_SURFACE_MACRO_TO_KEEP = np.asarray([0, 1, 2, 4, 7, 6, 5, 3], dtype=np.int64)
TRUE176_MACRO_TO_KEEP_NODE = np.concatenate([TRUE176_SURFACE_MACRO_TO_KEEP, TRUE176_SURFACE_MACRO_TO_KEEP + 8])

DISTORTION_CASES = [
    {
        "geometry_role": "moderately_distorted",
        "case_prefix": "case060",
        "shape4": [1.0, 0.02, 1.3, 0.0],
        "note": "medium distortion, positive detJ at standard source128 points",
    },
    {
        "geometry_role": "strongly_distorted_not_flipped",
        "case_prefix": "case061",
        "shape4": [1.0, 0.02, 1.3, 0.2],
        "note": "stronger distortion, positive detJ at standard source128 points",
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def load_old_shape4_helpers(old_src_root: Path):
    scripts_dir = Path(old_src_root).resolve() / "scripts"
    if not scripts_dir.exists():
        raise FileNotFoundError(f"old TRUE176 script directory not found: {scripts_dir}")
    sys.path.insert(0, str(scripts_dir))
    from css8_128_unified_shell_shape4 import (  # type: ignore
        SHAPE4_KEYS,
        audit_shape4,
        boundary_keep_nodes,
        build_shape4_node_dict,
        derived_geometry_from_shape4,
    )
    from run_css8_shape4_nonzero_smoke import build_t_boundary  # type: ignore

    return {
        "SHAPE4_KEYS": SHAPE4_KEYS,
        "audit_shape4": audit_shape4,
        "boundary_keep_nodes": boundary_keep_nodes,
        "build_shape4_node_dict": build_shape4_node_dict,
        "derived_geometry_from_shape4": derived_geometry_from_shape4,
        "build_t_boundary": build_t_boundary,
    }


def macro16_x16_from_shape4(shape4: np.ndarray, helpers: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    nodes = helpers["build_shape4_node_dict"](shape4)
    _boundary_nodes, keep_nodes = helpers["boundary_keep_nodes"]()
    x_keep = np.vstack([nodes[int(n)] for n in keep_nodes]).astype(np.float64)
    x16 = x_keep[TRUE176_MACRO_TO_KEEP_NODE].astype(np.float64)
    return x16, x_keep, [int(v) for v in keep_nodes]


def geometry_metrics(x16: np.ndarray) -> dict[str, Any]:
    geom = Macro16GeometryMap(np.asarray(x16, dtype=np.float64).reshape(16, 3))
    fields = geom.eval_points(macro16_source128_point_table())
    det = np.asarray(fields["detJ_hat"], dtype=np.float64)
    thickness = np.asarray(fields["thickness_hat"], dtype=np.float64)
    metric = np.asarray(fields["metric_hat"], dtype=np.float64)
    det_ratio = float(np.max(det) / max(float(np.min(det)), 1.0e-30))
    thickness_ratio = float(np.max(thickness) / max(float(np.min(thickness)), 1.0e-30))
    diag = np.sqrt(np.maximum(np.einsum("pii->pi", metric), 1.0e-30))
    skews = [
        np.abs(metric[:, 0, 1]) / np.maximum(diag[:, 0] * diag[:, 1], 1.0e-30),
        np.abs(metric[:, 0, 2]) / np.maximum(diag[:, 0] * diag[:, 2], 1.0e-30),
        np.abs(metric[:, 1, 2]) / np.maximum(diag[:, 1] * diag[:, 2], 1.0e-30),
    ]
    skew = float(np.max(np.stack(skews, axis=1)))
    score = math.log(max(det_ratio, 1.0)) + 0.25 * math.log(max(thickness_ratio, 1.0)) + skew
    return {
        "distortion_score": float(score),
        "detJ_min": float(np.min(det)),
        "detJ_max": float(np.max(det)),
        "detJ_ratio": det_ratio,
        "thickness_ratio": thickness_ratio,
        "metric_skew_max": skew,
        "L_ref": float(geom.l_ref),
        "span": geom.span.astype(float).tolist(),
        "center": geom.center.astype(float).tolist(),
    }


def write_q48_linear_frames(path: Path, *, case_id: str, program: dict[str, Any], q_final: np.ndarray, increments: int) -> None:
    q = np.asarray(q_final, dtype=np.float64).reshape(48)
    fields = ["case_id", "frame_index", "alpha_factor", "alpha", "q_norm", "q_program_id", "q_family"] + [f"q{i}" for i in range(1, 49)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for frame in range(1, int(increments) + 1):
            factor = float(frame) / float(increments)
            cur = factor * q
            row = {
                "case_id": case_id,
                "frame_index": int(frame),
                "alpha_factor": factor,
                "alpha": float(np.linalg.norm(cur)),
                "q_norm": float(np.linalg.norm(cur)),
                "q_program_id": program["q_program_id"],
                "q_family": program["family"],
            }
            row.update({f"q{i + 1}": float(cur[i]) for i in range(48)})
            writer.writerow(row)


def write_boundary_csv(path: Path, *, case_id: str, q48_frames: np.ndarray, t_boundary: np.ndarray, boundary_nodes: list[int]) -> None:
    qb = np.asarray(q48_frames, dtype=np.float64).reshape(-1, 48) @ np.asarray(t_boundary, dtype=np.float64).reshape(96, 48).T
    fields = ["case_id", "frame_index", "alpha", "q_norm", "boundary_node_index", "abaqus_node_label", "ux", "uy", "uz"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for frame_idx, flat in enumerate(qb.reshape(-1, 32, 3), start=1):
            q = q48_frames[frame_idx - 1]
            for node_i, node_label in enumerate(boundary_nodes):
                writer.writerow(
                    {
                        "case_id": case_id,
                        "frame_index": frame_idx,
                        "alpha": float(np.linalg.norm(q)),
                        "q_norm": float(np.linalg.norm(q)),
                        "boundary_node_index": node_i + 1,
                        "abaqus_node_label": int(node_label),
                        "ux": float(flat[node_i, 0]),
                        "uy": float(flat[node_i, 1]),
                        "uz": float(flat[node_i, 2]),
                    }
                )


def read_q48_csv(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([float(row[f"q{i}"]) for i in range(1, 49)])
    return np.asarray(rows, dtype=np.float64)


def ps_quote(value: Any) -> str:
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def command_for_task(
    task: dict[str, Any],
    args: argparse.Namespace,
    *,
    case_dir: Path | str | None = None,
    old_src_root: Path | str | None = None,
    export_lib: Path | str | None = None,
    abaqus: Path | str | None = None,
) -> str:
    case_dir = Path(case_dir if case_dir is not None else task["case_dir"])
    old_src_root = Path(old_src_root if old_src_root is not None else args.old_src_root)
    export_lib = Path(export_lib if export_lib is not None else args.export_lib)
    abaqus = Path(abaqus if abaqus is not None else args.abaqus)
    return (
        "py -3 scripts\\run_v1_2_pilot_case041_fresh_abaqus.py "
        f"--case-id {task['case_id']} "
        f"--q48-frames-csv {ps_quote(case_dir / 'q48_frames.csv')} "
        f"--boundary96-frames-csv {ps_quote(case_dir / 'boundary96_frames.csv')} "
        f"--bridge-contract {ps_quote(case_dir / 'boundary_contract_arrays.npz')} "
        f"--out-root {ps_quote(case_dir / 'fresh_run')} "
        f"--old-src-root {ps_quote(old_src_root)} "
        f"--export-lib {ps_quote(export_lib)} "
        f"--abaqus {ps_quote(abaqus)} "
        f"--shape4 {','.join(str(v) for v in task['shape4'])} "
        f"--increments {int(args.increments)} "
        f"--delta {float(args.delta):.12g} "
        f"--inner-workers {int(args.inner_workers)} "
        "--run-abaqus --run-complete-export --run-strict-audit --skip-existing"
    )


def remote_case_dir_for(task: dict[str, Any], out_root: Path, remote_root: str) -> str:
    case_rel = Path(task["case_dir"]).relative_to(out_root)
    return str(remote_root).rstrip("\\/") + "\\" + str(case_rel).replace("/", "\\")


def write_task_script(path: Path, tasks: list[dict[str, Any]], *, header: list[str]) -> None:
    lines = list(header)
    if lines and lines[-1] != "":
        lines.append("")
    for task in tasks:
        lines.append(f"Write-Output '[{task['case_id']}] start'")
        lines.append(task["command"])
        lines.append(f"if ($LASTEXITCODE -ne 0) {{ throw '[{task['case_id']}] failed with exit code ' + $LASTEXITCODE }}")
        lines.append(f"Write-Output '[{task['case_id']}] done'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def with_command(task: dict[str, Any], command: str) -> dict[str, Any]:
    row = dict(task)
    row["command"] = command
    return row


def prepare_tasks(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    helpers = load_old_shape4_helpers(Path(args.old_src_root))
    t_boundary, _t_rows = helpers["build_t_boundary"]()
    boundary_nodes, keep_nodes = helpers["boundary_keep_nodes"]()
    programs = q_program_definitions()

    tasks: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    compact_list_entries: list[str] = []

    for geom_idx, geom_def in enumerate(DISTORTION_CASES):
        shape4 = np.asarray(geom_def["shape4"], dtype=np.float64).reshape(4)
        audit = helpers["audit_shape4"](shape4)
        if not audit["passed"]:
            raise ValueError(f"{geom_def['geometry_role']} invalid shape4: {audit}")
        x16, x_keep, keep_node_ids = macro16_x16_from_shape4(shape4, helpers)
        geom_metrics = geometry_metrics(x16)
        modes = q_modes_for_geometry(x16, q_scale=float(args.q_scale), random_seed=int(args.random_seed) + geom_idx)
        geom_root = out_root / str(geom_def["geometry_role"])
        geom_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            geom_root / "geometry_contract.npz",
            shape4=shape4.astype(np.float64),
            shape4_keys=np.asarray(list(helpers["SHAPE4_KEYS"]), dtype=object),
            X16=x16.astype(np.float64),
            X_keep=x_keep.astype(np.float64),
            keep_nodes=np.asarray(keep_node_ids, dtype=np.int64),
            boundary_nodes=np.asarray(boundary_nodes, dtype=np.int64),
            T_boundary_96x48=np.asarray(t_boundary, dtype=np.float64),
        )
        geometry_rows.append(
            {
                **geom_def,
                "shape4": shape4.astype(float).tolist(),
                "shape4_keys": list(helpers["SHAPE4_KEYS"]),
                "shape4_audit_pass": bool(audit["passed"]),
                "derived_geometry": helpers["derived_geometry_from_shape4"](shape4),
                "X16": x16.astype(float).tolist(),
                **geom_metrics,
            }
        )
        for program_idx, program in enumerate(programs):
            qid = str(program["q_program_id"])
            q_final = np.asarray(modes[qid], dtype=np.float64).reshape(48)
            case_id = f"{geom_def['case_prefix']}_{program_idx:02d}_{qid}"
            case_dir = geom_root / case_id
            q48_csv = case_dir / "q48_frames.csv"
            boundary_csv = case_dir / "boundary96_frames.csv"
            write_q48_linear_frames(q48_csv, case_id=case_id, program=program, q_final=q_final, increments=int(args.increments))
            q_frames = read_q48_csv(q48_csv)
            write_boundary_csv(boundary_csv, case_id=case_id, q48_frames=q_frames, t_boundary=t_boundary, boundary_nodes=boundary_nodes)
            np.savez_compressed(
                case_dir / "boundary_contract_arrays.npz",
                shape4=shape4.astype(np.float64),
                shape4_keys=np.asarray(list(helpers["SHAPE4_KEYS"]), dtype=object),
                path_name=np.asarray([case_id], dtype=object),
                T_boundary_96x48=np.asarray(t_boundary, dtype=np.float64),
                X_keep_ref=x_keep.astype(np.float64),
                X_boundary_ref=np.vstack([helpers["build_shape4_node_dict"](shape4)[int(n)] for n in boundary_nodes]).astype(np.float64),
                q48_frames=np.vstack([np.zeros((1, 48), dtype=np.float64), q_frames]).astype(np.float64),
                q_boundary_frames=np.vstack([np.zeros((1, 96), dtype=np.float64), q_frames @ np.asarray(t_boundary, dtype=np.float64).T]).astype(np.float64),
                frame_times=np.linspace(0.0, 1.0, int(args.increments) + 1, dtype=np.float64),
                keep_nodes=np.asarray(keep_nodes, dtype=np.int64),
                boundary_nodes=np.asarray(boundary_nodes, dtype=np.int64),
                nlgeom=np.asarray([str(args.nlgeom).strip().upper()], dtype=object),
            )
            task = {
                "task_id": case_id,
                "case_id": case_id,
                "geometry_role": geom_def["geometry_role"],
                "q_program_id": qid,
                "q_family": program["family"],
                "q_norm_final": float(np.linalg.norm(q_final)),
                "rigid": bool(program["rigid"]),
                "shape4": shape4.astype(float).tolist(),
                "case_dir": str(case_dir),
                "q48_frames_csv": str(q48_csv),
                "boundary96_frames_csv": str(boundary_csv),
                "boundary_contract": str(case_dir / "boundary_contract_arrays.npz"),
                "complete_compact_expected": str(case_dir / "fresh_run" / "complete" / f"complete_{case_id}_training_ready.npz"),
                "uses_X_macro_as_model_input": False,
                "uses_internal_fine_grid_nodes_as_model_input": False,
                "network_training": False,
                "command": "",
            }
            task["command"] = command_for_task(task, args)
            compact_list_entries.append(task["complete_compact_expected"])
            tasks.append(task)

    write_json(out_root / "distorted_geometry_cases.json", {"geometries": geometry_rows})
    write_json(out_root / "distorted_task_manifest.json", {"task_count": len(tasks), "tasks": tasks})
    with (out_root / "distorted_task_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "task_id",
            "geometry_role",
            "q_program_id",
            "q_family",
            "q_norm_final",
            "rigid",
            "q48_frames_csv",
            "boundary96_frames_csv",
            "boundary_contract",
            "complete_compact_expected",
            "command",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(tasks)
    (out_root / "expected_complete_compact_list.txt").write_text("\n".join(compact_list_entries) + "\n", encoding="utf-8")

    remote_root = str(args.remote_task_root).rstrip("\\/")
    remote_tasks: list[dict[str, Any]] = []
    for task in tasks:
        remote_case_dir = remote_case_dir_for(task, out_root, remote_root)
        remote_tasks.append(
            with_command(
                task,
                command_for_task(
                    task,
                    args,
                    case_dir=remote_case_dir,
                    old_src_root=args.remote_old_src_root,
                    export_lib=args.remote_export_lib,
                    abaqus=args.remote_abaqus,
                ),
            )
        )

    local_header = [
        "# Run on a Windows Abaqus worker from the repository root.",
        "$ErrorActionPreference = 'Stop'",
    ]
    remote_header = [
        "# Run on the remote Windows Abaqus worker from the repository root.",
        "$ErrorActionPreference = 'Stop'",
        f"$TaskRoot = \"{remote_root}\"",
    ]
    write_task_script(out_root / "run_distorted_tasks_windows.ps1", tasks, header=local_header)
    write_task_script(out_root / "run_distorted_tasks_remote_windows.ps1", remote_tasks, header=remote_header)

    local_medium = [task for task in tasks if task["geometry_role"] == "moderately_distorted"]
    local_strong = [task for task in tasks if task["geometry_role"] == "strongly_distorted_not_flipped"]
    remote_medium = [task for task in remote_tasks if task["geometry_role"] == "moderately_distorted"]
    remote_strong = [task for task in remote_tasks if task["geometry_role"] == "strongly_distorted_not_flipped"]
    probe_ids = {"case060_00_q_zero", "case060_01_axial_tension", "case061_00_q_zero", "case061_01_axial_tension"}
    write_task_script(out_root / "run_distorted_tasks_local_moderate_windows.ps1", local_medium, header=local_header)
    write_task_script(out_root / "run_distorted_tasks_local_strong_windows.ps1", local_strong, header=local_header)
    write_task_script(out_root / "run_distorted_tasks_remote_moderate_windows.ps1", remote_medium, header=remote_header)
    write_task_script(out_root / "run_distorted_tasks_remote_strong_windows.ps1", remote_strong, header=remote_header)
    write_task_script(out_root / "run_distorted_tasks_probe_windows.ps1", [task for task in tasks if task["case_id"] in probe_ids], header=local_header)
    write_task_script(out_root / "run_distorted_tasks_remote_probe_windows.ps1", [task for task in remote_tasks if task["case_id"] in probe_ids], header=remote_header)

    summary = {
        "script": "prepare_macro16_source128_distortion_tasks",
        "out_root": str(out_root),
        "geometry_count": len(geometry_rows),
        "task_count": len(tasks),
        "q_program_count_per_geometry": len(programs),
        "increments_per_task": int(args.increments),
        "q_scale": float(args.q_scale),
        "inner_workers_per_task": int(args.inner_workers),
        "standard_macro16_contract": {
            "geometry_input": "X16 only",
            "displacement_input": "q48 only",
            "point_rule": "fixed 128 parent-domain source128 rule",
            "weight_rule": "generated from X16 by Macro16 isoparametric map after complete compact export",
            "uses_X_macro_as_model_input": False,
            "uses_internal_fine_grid_nodes_as_model_input": False,
            "network_training": False,
        },
        "outputs": [
            "distorted_geometry_cases.json",
            "distorted_task_manifest.json",
            "distorted_task_manifest.csv",
            "expected_complete_compact_list.txt",
            "run_distorted_tasks_windows.ps1",
            "run_distorted_tasks_remote_windows.ps1",
            "run_distorted_tasks_local_moderate_windows.ps1",
            "run_distorted_tasks_remote_strong_windows.ps1",
            "run_distorted_tasks_probe_windows.ps1",
            "run_distorted_tasks_remote_probe_windows.ps1",
        ],
        "worker_assignment": {
            "local_windows": "moderately_distorted",
            "remote_windows": "strongly_distorted_not_flipped",
            "linux": "postprocess source128 compact build and force/stiffness audit after complete compacts are collected",
        },
        "remote_paths": {
            "remote_task_root": str(args.remote_task_root),
            "remote_old_src_root": str(args.remote_old_src_root),
            "remote_export_lib": str(args.remote_export_lib),
            "remote_abaqus": str(args.remote_abaqus),
        },
    }
    write_json(out_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, default=json_default))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(Path("runs") / "macro16_source128_generality_plan" / "distorted_generation_tasks"), type=Path)
    parser.add_argument("--old-src-root", default=str(DEFAULT_OLD_SRC_ROOT), type=Path)
    parser.add_argument("--export-lib", default=r"D:\IS-FEM\SRCv3.1 - 1\SRCv3.1\scripts\export_three_element_bfk_arrays.py", type=Path)
    parser.add_argument("--abaqus", default=r"D:\Program Files\SIMULIA\Commands\abaqus.bat", type=Path)
    parser.add_argument("--remote-task-root", default=r"D:\IS-FEM\outputs\macro16_source128_generality_distorted_tasks")
    parser.add_argument("--remote-old-src-root", default=str(DEFAULT_OLD_SRC_ROOT), type=Path)
    parser.add_argument("--remote-export-lib", default=r"D:\IS-FEM\SRCv3.1 - 1\SRCv3.1\scripts\export_three_element_bfk_arrays.py", type=Path)
    parser.add_argument("--remote-abaqus", default=str(DEFAULT_REMOTE_ABAQUS), type=Path)
    parser.add_argument("--increments", type=int, default=10)
    parser.add_argument("--q-scale", type=float, default=1.0e-3)
    parser.add_argument("--random-seed", type=int, default=20260625)
    parser.add_argument("--delta", type=float, default=1.0e-6)
    parser.add_argument("--inner-workers", type=int, default=4)
    parser.add_argument("--nlgeom", default="YES", choices=("YES", "NO", "yes", "no"))
    return parser.parse_args()


def main() -> None:
    prepare_tasks(parse_args())


if __name__ == "__main__":
    main()
