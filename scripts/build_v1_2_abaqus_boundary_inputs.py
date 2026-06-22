#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Abaqus boundary-displacement input tables from v1.2 q48 plans.

This script applies an audited TRUE176 bridge:

    q_boundary96 = T_boundary_96x48 @ q48

It does not run Abaqus, does not generate ODB files, and does not use legacy
LE/B as labels.  The output is a lightweight boundary-displacement input plan
for fresh Abaqus runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


def case_label(num: int) -> str:
    return f"case{int(num):03d}"


def read_q48_csv(path: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows: list[dict[str, Any]] = []
    q_rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"case_id", "frame_index", "alpha", "q_norm"} | {f"q{i}" for i in range(1, 49)}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            rows.append(row)
            q_rows.append([float(row[f"q{i}"]) for i in range(1, 49)])
    if not q_rows:
        raise ValueError(f"{path}: no q48 rows")
    q48 = np.asarray(q_rows, dtype=np.float64)
    if q48.ndim != 2 or q48.shape[1] != 48:
        raise ValueError(f"{path}: expected q48 [N,48], got {q48.shape}")
    return rows, q48


def load_bridge_contract(path: Path, tol: float) -> dict[str, Any]:
    with np.load(str(path), allow_pickle=True) as z:
        required = {"T_boundary_96x48", "boundary_nodes", "keep_nodes"}
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"{path}: missing bridge arrays {sorted(missing)}")
        T = np.asarray(z["T_boundary_96x48"], dtype=np.float64)
        boundary_nodes = np.asarray(z["boundary_nodes"], dtype=np.int64).reshape(-1)
        keep_nodes = np.asarray(z["keep_nodes"], dtype=np.int64).reshape(-1)
        if T.shape != (96, 48):
            raise ValueError(f"{path}: expected T_boundary_96x48 shape (96,48), got {T.shape}")
        if boundary_nodes.shape != (32,):
            raise ValueError(f"{path}: expected 32 boundary node labels, got {boundary_nodes.shape}")
        if keep_nodes.shape != (16,):
            raise ValueError(f"{path}: expected 16 keep node labels, got {keep_nodes.shape}")
        bridge_validation = None
        if "q48_frames" in z.files and "q_boundary_frames" in z.files:
            q_ref = np.asarray(z["q48_frames"], dtype=np.float64).reshape(-1, 48)
            qb_ref = np.asarray(z["q_boundary_frames"], dtype=np.float64).reshape(-1, 96)
            pred = q_ref @ T.T
            max_abs = float(np.max(np.abs(pred - qb_ref))) if pred.size else 0.0
            rel = float(np.linalg.norm(pred - qb_ref) / max(np.linalg.norm(qb_ref), 1.0e-30))
            bridge_validation = {
                "q48_frames_shape": list(q_ref.shape),
                "q_boundary_frames_shape": list(qb_ref.shape),
                "Tq_vs_q_boundary_max_abs": max_abs,
                "Tq_vs_q_boundary_rel": rel,
                "pass": bool(max_abs <= float(tol)),
            }
            if max_abs > float(tol):
                raise ValueError(f"{path}: T @ q48 does not reproduce q_boundary_frames, max_abs={max_abs:g}")
        return {
            "path": path,
            "T": T,
            "boundary_nodes": boundary_nodes,
            "keep_nodes": keep_nodes,
            "shape4": np.asarray(z["shape4"], dtype=np.float64).reshape(-1).tolist() if "shape4" in z.files else None,
            "shape4_keys": np.asarray(z["shape4_keys"]).astype(str).reshape(-1).tolist() if "shape4_keys" in z.files else None,
            "T_sha256": hashlib.sha256(np.ascontiguousarray(T).tobytes()).hexdigest(),
            "bridge_validation": bridge_validation,
        }


def write_boundary_csv(path: Path, case_id: str, rows: list[dict[str, Any]], qb: np.ndarray, boundary_nodes: np.ndarray) -> None:
    fields = [
        "case_id",
        "frame_index",
        "alpha",
        "q_norm",
        "boundary_node_index",
        "abaqus_node_label",
        "ux",
        "uy",
        "uz",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for frame_i, row in enumerate(rows):
            flat = qb[frame_i].reshape(32, 3)
            for node_i, node_label in enumerate(boundary_nodes):
                writer.writerow(
                    {
                        "case_id": case_id,
                        "frame_index": int(row["frame_index"]),
                        "alpha": float(row["alpha"]),
                        "q_norm": float(row["q_norm"]),
                        "boundary_node_index": node_i + 1,
                        "abaqus_node_label": int(node_label),
                        "ux": float(flat[node_i, 0]),
                        "uy": float(flat[node_i, 1]),
                        "uz": float(flat[node_i, 2]),
                    }
                )


def amplitude_lines(name: str, values: list[tuple[float, float]], per_line: int = 4) -> list[str]:
    tokens: list[str] = []
    for t, v in values:
        tokens.extend([f"{t:.12g}", f"{v:.12e}"])
    lines = [f"*Amplitude, name={name}, time=TOTAL TIME"]
    chunk = per_line * 2
    for i in range(0, len(tokens), chunk):
        lines.append(", ".join(tokens[i : i + chunk]))
    return lines


def write_abaqus_template(path: Path, case_id: str, rows: list[dict[str, Any]], qb: np.ndarray, boundary_nodes: np.ndarray) -> None:
    times = [0.0] + [float(row["alpha_factor"]) if "alpha_factor" in row else float(row["frame_index"]) / len(rows) for row in rows]
    lines = [
        f"** Boundary displacement template for {case_id}",
        "** Generated from q_boundary96 = T_boundary_96x48 @ q48.",
        "** Template only: insert into the matching fresh Abaqus input model/step.",
        "** Legacy LE/B are not labels and are not used here.",
        "",
    ]
    for node_i, node_label in enumerate(boundary_nodes):
        for dof_i, dof_name in enumerate(("U1", "U2", "U3"), start=1):
            amp_name = f"{case_id.upper()}_N{int(node_label):03d}_{dof_name}"
            vals = [(0.0, 0.0)]
            vals.extend((times[i + 1], float(qb[i].reshape(32, 3)[node_i, dof_i - 1])) for i in range(len(rows)))
            lines.extend(amplitude_lines(amp_name, vals))
            lines.append(f"*Boundary, amplitude={amp_name}")
            lines.append(f"{int(node_label)}, {dof_i}, {dof_i}, 1.")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_node_json(path: Path, case_id: str, bridge: dict[str, Any], q48_source: Path, boundary_csv: Path) -> None:
    payload = {
        "case_id": case_id,
        "bridge_contract_path": str(bridge["path"]),
        "bridge_source": "boundary_contract_arrays.npz:T_boundary_96x48",
        "T_boundary_96x48_shape": list(bridge["T"].shape),
        "T_boundary_96x48_sha256": bridge["T_sha256"],
        "bridge_validation": bridge["bridge_validation"],
        "keep_nodes": bridge["keep_nodes"].tolist(),
        "boundary_nodes": bridge["boundary_nodes"].tolist(),
        "q48_source_csv": str(q48_source),
        "boundary96_frames_csv": str(boundary_csv),
        "flattening": "boundary96[3*i+0:3*i+3] maps to boundary_nodes[i] U1,U2,U3",
        "uses_legacy_LE_B_as_labels": False,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")


def build_inputs(args: argparse.Namespace) -> dict[str, Any]:
    plan_root = Path(args.q48_plan_root).resolve()
    out_root = Path(args.out_root).resolve()
    bridge_path = Path(args.bridge_contract).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    bridge = load_bridge_contract(bridge_path, float(args.bridge_tol))
    T = bridge["T"]
    boundary_nodes = bridge["boundary_nodes"]

    case_summaries: list[dict[str, Any]] = []
    for num in range(int(args.case_start), int(args.case_end) + 1):
        case_id = case_label(num)
        q48_path = plan_root / f"{case_id}_q48_frames.csv"
        if not q48_path.exists():
            raise FileNotFoundError(f"missing q48 frame CSV: {q48_path}")
        rows, q48 = read_q48_csv(q48_path)
        if int(q48.shape[0]) != int(args.expected_frame_count):
            raise ValueError(f"{q48_path}: expected {args.expected_frame_count} frames, got {q48.shape[0]}")
        for row, q in zip(rows, q48):
            q_norm = float(np.linalg.norm(q))
            alpha = float(row["alpha"])
            if abs(q_norm - alpha) > float(args.q_norm_tol) * max(1.0, abs(alpha)):
                raise ValueError(f"{q48_path}: frame {row['frame_index']} q_norm {q_norm:g} != alpha {alpha:g}")
        qb = q48 @ T.T
        if qb.shape != (q48.shape[0], 96):
            raise ValueError(f"{case_id}: expected boundary output [N,96], got {qb.shape}")
        boundary_csv = out_root / f"{case_id}_boundary96_frames.csv"
        node_json = out_root / f"{case_id}_boundary_nodes.json"
        template_txt = out_root / f"{case_id}_abaqus_bc_commands.txt"
        write_boundary_csv(boundary_csv, case_id, rows, qb, boundary_nodes)
        write_node_json(node_json, case_id, bridge, q48_path, boundary_csv)
        write_abaqus_template(template_txt, case_id, rows, qb, boundary_nodes)
        case_summaries.append(
            {
                "case_id": case_id,
                "q48_source_csv": str(q48_path),
                "boundary96_frames_csv": str(boundary_csv),
                "boundary_nodes_json": str(node_json),
                "abaqus_bc_template": str(template_txt),
                "frame_count": int(q48.shape[0]),
                "boundary_node_count": int(boundary_nodes.size),
                "boundary_dof_count": 96,
                "q48_norm_min": float(np.min(np.linalg.norm(q48, axis=1))),
                "q48_norm_max": float(np.max(np.linalg.norm(q48, axis=1))),
                "boundary96_norm_min": float(np.min(np.linalg.norm(qb, axis=1))),
                "boundary96_norm_max": float(np.max(np.linalg.norm(qb, axis=1))),
                "status": "resolved",
            }
        )

    summary = {
        "audit_name": "v1_2_abaqus_boundary_input_bridge",
        "case_count": len(case_summaries),
        "case_ids": [row["case_id"] for row in case_summaries],
        "q48_plan_root": str(plan_root),
        "out_root": str(out_root),
        "bridge_contract_path": str(bridge_path),
        "bridge_source": "T_boundary_96x48 from boundary_contract_arrays.npz",
        "bridge_validation": bridge["bridge_validation"],
        "T_boundary_96x48_sha256": bridge["T_sha256"],
        "keep_nodes": bridge["keep_nodes"].tolist(),
        "boundary_nodes": boundary_nodes.tolist(),
        "output_shape": "[frame, 32, 3] flattened to 96 DOF",
        "uses_legacy_LE_B_as_labels": False,
        "requires_fresh_abaqus_run": True,
        "requires_matching_sobolev_B_compact": True,
        "requires_strict_v1_2_pass": True,
        "cases": case_summaries,
    }
    (out_root / "fresh_abaqus_boundary_inputs_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    with (out_root / "fresh_abaqus_boundary_inputs_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "case_id",
            "frame_count",
            "boundary_node_count",
            "boundary_dof_count",
            "q48_norm_min",
            "q48_norm_max",
            "boundary96_norm_min",
            "boundary96_norm_max",
            "boundary96_frames_csv",
            "abaqus_bc_template",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in case_summaries:
            writer.writerow(row)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q48-plan-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--bridge-contract", required=True)
    parser.add_argument("--case-start", type=int, default=40)
    parser.add_argument("--case-end", type=int, default=51)
    parser.add_argument("--expected-frame-count", type=int, default=10)
    parser.add_argument("--bridge-tol", type=float, default=1.0e-12)
    parser.add_argument("--q-norm-tol", type=float, default=1.0e-10)
    args = parser.parse_args()

    summary = build_inputs(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
