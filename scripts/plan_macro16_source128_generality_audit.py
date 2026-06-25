#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plan the Macro16 source128 standard-weight generality audit.

This script does not run Abaqus and does not train a network.  It creates a
reproducible task manifest for checking whether a standard Macro16 128-point
integration rule remains force/stiffness closed across geometry distortion
levels and boundary q48 programs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_macro16_from_128_teacher import (  # noqa: E402
    TRUE176_MACRO_TO_KEEP_NODE,
    infer_n,
    load_x16,
    resolve_source_node_order,
)
from macro_deeponet.macro16_geometry import Macro16GeometryMap, macro16_source128_point_table  # noqa: E402


GEOMETRY_ROLES = [
    {
        "role": "regular",
        "label": "regular geometry",
        "worker": "local_windows",
        "selection": "lowest distortion score",
    },
    {
        "role": "lightly_distorted",
        "label": "lightly distorted geometry",
        "worker": "local_windows",
        "selection": "low nonzero distortion score",
    },
    {
        "role": "moderately_distorted",
        "label": "moderately distorted geometry",
        "worker": "remote_windows_abaqus",
        "selection": "middle-high distortion score without detJ failure",
    },
    {
        "role": "strongly_distorted_not_flipped",
        "label": "strongly distorted but not flipped geometry",
        "worker": "remote_windows_abaqus",
        "selection": "highest distortion score with positive detJ",
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


def read_path_list(path: Path) -> list[Path]:
    return [
        Path(line.strip()).resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def collect_paths(compacts: Iterable[Path], compact_lists: Iterable[Path], *, case_limit: int) -> list[Path]:
    paths = [Path(path).resolve() for path in compacts]
    for list_path in compact_lists:
        paths.extend(read_path_list(Path(list_path)))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if int(case_limit) > 0:
        unique = unique[: int(case_limit)]
    return unique


def case_id_from_path(path: Path) -> int:
    match = re.search(r"case(\d+)", path.name.lower())
    return int(match.group(1)) if match else -1


def row_select_or_broadcast(arr: np.ndarray, rows: np.ndarray, n_total: int, tail: tuple[int, ...]) -> np.ndarray:
    vals = np.asarray(arr)
    if vals.shape == tail:
        return np.broadcast_to(vals.reshape((1,) + tail), (rows.size,) + tail).copy()
    if vals.shape == (1,) + tail:
        return np.broadcast_to(vals.reshape((1,) + tail), (rows.size,) + tail).copy()
    if vals.shape == (int(n_total),) + tail:
        return vals[rows]
    if vals.size == int(n_total) * int(np.prod(tail)):
        return vals.reshape((int(n_total),) + tail)[rows]
    raise ValueError(f"cannot broadcast array of shape {vals.shape} to rows with tail {tail}")


def load_case_ids(z: np.lib.npyio.NpzFile, path: Path, n_total: int, rows: np.ndarray) -> np.ndarray:
    if "case_id" not in z.files:
        return np.full(rows.size, case_id_from_path(path), dtype=np.int64)
    vals = np.asarray(z["case_id"], dtype=np.int64)
    if vals.shape == ():
        return np.full(rows.size, int(vals), dtype=np.int64)
    if vals.size == 1:
        return np.full(rows.size, int(vals.reshape(-1)[0]), dtype=np.int64)
    return vals.reshape(int(n_total))[rows].astype(np.int64)


def load_shape4_rows(z: np.lib.npyio.NpzFile, n_total: int, rows: np.ndarray) -> list[list[float]] | None:
    if "shape4" not in z.files:
        return None
    arr = np.asarray(z["shape4"], dtype=np.float64)
    if arr.shape == (4,):
        vals = np.broadcast_to(arr.reshape(1, 4), (rows.size, 4)).copy()
    elif arr.shape == (1, 4):
        vals = np.broadcast_to(arr.reshape(1, 4), (rows.size, 4)).copy()
    elif arr.shape == (int(n_total), 4):
        vals = arr[rows]
    elif arr.size == int(n_total) * 4:
        vals = arr.reshape(int(n_total), 4)[rows]
    else:
        return None
    return vals.astype(float).tolist()


def selected_rows(n_total: int, *, frame_stride: int, max_rows_per_compact: int) -> np.ndarray:
    rows = np.arange(0, int(n_total), max(1, int(frame_stride)), dtype=np.int64)
    if int(max_rows_per_compact) > 0:
        rows = rows[: int(max_rows_per_compact)]
    return rows


def load_geometry_samples(
    paths: list[Path],
    *,
    frame_stride: int,
    max_rows_per_compact: int,
    source_node_order: str,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for source_index, path in enumerate(paths):
        with np.load(str(path), allow_pickle=True) as z:
            n_total = infer_n(z, path)
            rows = selected_rows(n_total, frame_stride=frame_stride, max_rows_per_compact=max_rows_per_compact)
            x16, x16_key = load_x16(z, path, n_total, rows)
            order = resolve_source_node_order(source_node_order, x16_key)
            if order == "true176-keep":
                x16 = np.asarray(x16, dtype=np.float64)[:, TRUE176_MACRO_TO_KEEP_NODE, :]
            elif order != "macro16":
                raise ValueError(f"{path}: unsupported source node order {order!r}")
            case_ids = load_case_ids(z, path, n_total, rows)
            shape4_rows = load_shape4_rows(z, n_total, rows)
            for local_i, row in enumerate(rows.tolist()):
                samples.append(
                    {
                        "source_index": int(source_index),
                        "source_compact": str(path),
                        "source_row": int(row),
                        "case_id": int(case_ids[local_i]),
                        "x16_source_key": x16_key,
                        "source_node_order": order,
                        "shape4": None if shape4_rows is None else shape4_rows[local_i],
                        "X16": np.asarray(x16[local_i], dtype=np.float64),
                    }
                )
    return samples


def geometry_metrics(x16: np.ndarray) -> dict[str, Any]:
    try:
        geom = Macro16GeometryMap(np.asarray(x16, dtype=np.float64).reshape(16, 3))
        fields = geom.eval_points(macro16_source128_point_table())
        det = np.asarray(fields["detJ_hat"], dtype=np.float64)
        thickness = np.asarray(fields["thickness_hat"], dtype=np.float64)
        metric = np.asarray(fields["metric_hat"], dtype=np.float64)
        det_min = float(np.min(det))
        det_max = float(np.max(det))
        det_ratio = det_max / max(det_min, 1.0e-30)
        thickness_ratio = float(np.max(thickness) / max(float(np.min(thickness)), 1.0e-30))
        diag = np.sqrt(np.maximum(np.einsum("pii->pi", metric), 1.0e-30))
        skew01 = np.abs(metric[:, 0, 1]) / np.maximum(diag[:, 0] * diag[:, 1], 1.0e-30)
        skew02 = np.abs(metric[:, 0, 2]) / np.maximum(diag[:, 0] * diag[:, 2], 1.0e-30)
        skew12 = np.abs(metric[:, 1, 2]) / np.maximum(diag[:, 1] * diag[:, 2], 1.0e-30)
        skew_max = float(np.max(np.stack([skew01, skew02, skew12], axis=1)))
        score = math.log(max(det_ratio, 1.0)) + 0.25 * math.log(max(thickness_ratio, 1.0)) + skew_max
        return {
            "geometry_valid": True,
            "distortion_score": float(score),
            "detJ_min": det_min,
            "detJ_max": det_max,
            "detJ_ratio": float(det_ratio),
            "thickness_ratio": thickness_ratio,
            "metric_skew_max": skew_max,
            "L_ref": float(geom.l_ref),
            "center": geom.center.astype(float).tolist(),
            "span": geom.span.astype(float).tolist(),
        }
    except Exception as exc:
        return {
            "geometry_valid": False,
            "distortion_score": float("inf"),
            "error": str(exc),
        }


def geometry_key(x16: np.ndarray, decimals: int) -> tuple[float, ...]:
    geom = Macro16GeometryMap(np.asarray(x16, dtype=np.float64).reshape(16, 3))
    return tuple(float(v) for v in np.round(geom.x16_hat.reshape(-1), int(decimals)).tolist())


def group_geometries(samples: list[dict[str, Any]], *, decimals: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, ...], list[dict[str, Any]]] = {}
    for sample in samples:
        try:
            key = geometry_key(np.asarray(sample["X16"], dtype=np.float64), decimals)
        except Exception:
            key = tuple(float(v) for v in np.round(np.asarray(sample["X16"], dtype=np.float64).reshape(-1), int(decimals)).tolist())
        grouped.setdefault(key, []).append(sample)

    rows: list[dict[str, Any]] = []
    for group_id, group_samples in enumerate(grouped.values()):
        x16 = np.asarray(group_samples[0]["X16"], dtype=np.float64)
        metrics = geometry_metrics(x16)
        source_rows = [
            {
                "source_index": int(item["source_index"]),
                "source_compact": item["source_compact"],
                "source_row": int(item["source_row"]),
                "case_id": int(item["case_id"]),
            }
            for item in group_samples
        ]
        rows.append(
            {
                "geometry_group_id": int(group_id),
                "status": "available" if metrics["geometry_valid"] else "invalid_geometry",
                "source_count": int(len(group_samples)),
                "source_rows": source_rows,
                "representative_source_compact": group_samples[0]["source_compact"],
                "representative_source_row": int(group_samples[0]["source_row"]),
                "case_ids": sorted({int(item["case_id"]) for item in group_samples}),
                "shape4": group_samples[0].get("shape4"),
                "X16": x16.astype(float).tolist(),
                **metrics,
            }
        )
    return sorted(rows, key=lambda item: (float(item.get("distortion_score", float("inf"))), int(item["geometry_group_id"])))


def unique_indices(candidates: list[int]) -> list[int]:
    out: list[int] = []
    for idx in candidates:
        if idx not in out:
            out.append(idx)
    return out


def select_geometry_roles(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [group for group in groups if group.get("status") == "available" and bool(group.get("geometry_valid"))]
    n = len(valid)
    if n >= 4:
        indices = unique_indices([0, max(1, round((n - 1) / 3)), max(2, round(2 * (n - 1) / 3)), n - 1])
        while len(indices) < 4:
            for i in range(n):
                if i not in indices:
                    indices.append(i)
                if len(indices) == 4:
                    break
    elif n == 3:
        indices = [0, 1, 2]
    elif n == 2:
        indices = [0, 1]
    elif n == 1:
        indices = [0]
    else:
        indices = []

    rows: list[dict[str, Any]] = []
    for role_i, role_def in enumerate(GEOMETRY_ROLES):
        if role_i < len(indices):
            group = valid[indices[role_i]]
            row = {
                **role_def,
                **{key: value for key, value in group.items() if key != "status"},
                "status": "available",
                "needs_generation": False,
            }
        else:
            row = {
                **role_def,
                "status": "needs_generation",
                "needs_generation": True,
                "geometry_valid": None,
                "distortion_score": None,
                "target_requirement": "Generate or select an X16 geometry for this role, then rerun this planner. detJ must remain positive at all 128 standard parent points.",
            }
        rows.append(row)
    return rows


def characteristic_length(x16: np.ndarray) -> float:
    nodes = np.asarray(x16, dtype=np.float64).reshape(16, 3)
    span = np.max(nodes, axis=0) - np.min(nodes, axis=0)
    l_ref = float(np.max(span))
    if not math.isfinite(l_ref) or l_ref <= 1.0e-14:
        center = np.mean(nodes, axis=0)
        l_ref = float(np.sqrt(np.mean(np.sum((nodes - center.reshape(1, 3)) ** 2, axis=1))))
    if not math.isfinite(l_ref) or l_ref <= 1.0e-14:
        raise ValueError("cannot compute L_ref from X16")
    return l_ref


def normalize_mode(mode: np.ndarray, target_norm: float) -> np.ndarray:
    vals = np.asarray(mode, dtype=np.float64).reshape(48)
    norm = float(np.linalg.norm(vals))
    if norm <= 1.0e-30:
        return np.zeros(48, dtype=np.float64)
    return vals * (float(target_norm) / norm)


def q_modes_for_geometry(x16: np.ndarray, *, q_scale: float, random_seed: int) -> dict[str, np.ndarray]:
    nodes = np.asarray(x16, dtype=np.float64).reshape(16, 3)
    center = np.mean(nodes, axis=0)
    rel = nodes - center.reshape(1, 3)
    l_ref = characteristic_length(nodes)
    target_norm = float(q_scale) * l_ref
    rel_hat = rel / l_ref
    x = rel_hat[:, 0]
    y = rel_hat[:, 1]
    z = rel_hat[:, 2]
    modes: dict[str, np.ndarray] = {}
    modes["q_zero"] = np.zeros((16, 3), dtype=np.float64)

    axial = np.zeros((16, 3), dtype=np.float64)
    axial[:, 0] = x
    modes["axial_tension"] = normalize_mode(axial.reshape(48), target_norm).reshape(16, 3)
    modes["axial_compression"] = -modes["axial_tension"]

    shear = np.zeros((16, 3), dtype=np.float64)
    shear[:, 0] = y
    modes["inplane_shear_pos"] = normalize_mode(shear.reshape(48), target_norm).reshape(16, 3)
    modes["inplane_shear_neg"] = -modes["inplane_shear_pos"]

    bending = np.zeros((16, 3), dtype=np.float64)
    bending[:, 2] = x * x - float(np.mean(x * x))
    modes["bending_pos"] = normalize_mode(bending.reshape(48), target_norm).reshape(16, 3)
    modes["bending_neg"] = -modes["bending_pos"]

    torsion = np.zeros((16, 3), dtype=np.float64)
    torsion[:, 0] = -z * y
    torsion[:, 1] = z * x
    modes["torsion_pos"] = normalize_mode(torsion.reshape(48), target_norm).reshape(16, 3)
    modes["torsion_neg"] = -modes["torsion_pos"]

    rng = np.random.default_rng(int(random_seed))
    for idx in range(1, 4):
        raw = rng.normal(size=(16, 3))
        raw -= np.mean(raw, axis=0, keepdims=True)
        key_pos = f"random_48_small_pos_{idx:03d}"
        key_neg = f"random_48_small_neg_{idx:03d}"
        modes[key_pos] = normalize_mode(raw.reshape(48), target_norm).reshape(16, 3)
        modes[key_neg] = -modes[key_pos]

    for axis, vec in (("x", [1.0, 0.0, 0.0]), ("y", [0.0, 1.0, 0.0]), ("z", [0.0, 0.0, 1.0])):
        trans = np.broadcast_to(np.asarray(vec, dtype=np.float64).reshape(1, 3), (16, 3)).copy()
        modes[f"rigid_translation_{axis}"] = normalize_mode(trans.reshape(48), target_norm).reshape(16, 3)

    for axis, omega in (
        ("x", np.asarray([1.0, 0.0, 0.0], dtype=np.float64)),
        ("y", np.asarray([0.0, 1.0, 0.0], dtype=np.float64)),
        ("z", np.asarray([0.0, 0.0, 1.0], dtype=np.float64)),
    ):
        rot = np.cross(np.broadcast_to(omega.reshape(1, 3), (16, 3)), rel)
        modes[f"rigid_rotation_{axis}"] = normalize_mode(rot.reshape(48), target_norm).reshape(16, 3)

    return {key: value.reshape(48).astype(np.float64) for key, value in modes.items()}


def q_program_definitions() -> list[dict[str, Any]]:
    return [
        {"q_program_id": "q_zero", "family": "zero", "sign": 0, "pair_id": "", "rigid": False},
        {"q_program_id": "axial_tension", "family": "single_axis", "sign": 1, "pair_id": "axial", "rigid": False},
        {"q_program_id": "axial_compression", "family": "single_axis", "sign": -1, "pair_id": "axial", "rigid": False},
        {"q_program_id": "inplane_shear_pos", "family": "inplane_shear", "sign": 1, "pair_id": "inplane_shear", "rigid": False},
        {"q_program_id": "inplane_shear_neg", "family": "inplane_shear", "sign": -1, "pair_id": "inplane_shear", "rigid": False},
        {"q_program_id": "bending_pos", "family": "bending", "sign": 1, "pair_id": "bending", "rigid": False},
        {"q_program_id": "bending_neg", "family": "bending", "sign": -1, "pair_id": "bending", "rigid": False},
        {"q_program_id": "torsion_pos", "family": "torsion", "sign": 1, "pair_id": "torsion", "rigid": False},
        {"q_program_id": "torsion_neg", "family": "torsion", "sign": -1, "pair_id": "torsion", "rigid": False},
        {"q_program_id": "random_48_small_pos_001", "family": "random_48_small", "sign": 1, "pair_id": "random_001", "rigid": False},
        {"q_program_id": "random_48_small_neg_001", "family": "random_48_small", "sign": -1, "pair_id": "random_001", "rigid": False},
        {"q_program_id": "random_48_small_pos_002", "family": "random_48_small", "sign": 1, "pair_id": "random_002", "rigid": False},
        {"q_program_id": "random_48_small_neg_002", "family": "random_48_small", "sign": -1, "pair_id": "random_002", "rigid": False},
        {"q_program_id": "random_48_small_pos_003", "family": "random_48_small", "sign": 1, "pair_id": "random_003", "rigid": False},
        {"q_program_id": "random_48_small_neg_003", "family": "random_48_small", "sign": -1, "pair_id": "random_003", "rigid": False},
        {"q_program_id": "rigid_translation_x", "family": "rigid_translation", "sign": 1, "pair_id": "", "rigid": True},
        {"q_program_id": "rigid_translation_y", "family": "rigid_translation", "sign": 1, "pair_id": "", "rigid": True},
        {"q_program_id": "rigid_translation_z", "family": "rigid_translation", "sign": 1, "pair_id": "", "rigid": True},
        {"q_program_id": "rigid_rotation_x", "family": "rigid_rotation", "sign": 1, "pair_id": "", "rigid": True},
        {"q_program_id": "rigid_rotation_y", "family": "rigid_rotation", "sign": 1, "pair_id": "", "rigid": True},
        {"q_program_id": "rigid_rotation_z", "family": "rigid_rotation", "sign": 1, "pair_id": "", "rigid": True},
    ]


def write_q48_csv(path: Path, geometry_role: dict[str, Any], programs: list[dict[str, Any]], *, q_scale: float, random_seed: int) -> None:
    if geometry_role.get("status") != "available" or "X16" not in geometry_role:
        return
    modes = q_modes_for_geometry(np.asarray(geometry_role["X16"], dtype=np.float64), q_scale=q_scale, random_seed=random_seed)
    fields = ["geometry_role", "q_program_id", "family", "sign", "q_norm"] + [f"q{i}" for i in range(1, 49)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for program in programs:
            qid = str(program["q_program_id"])
            q = modes[qid]
            row = {
                "geometry_role": geometry_role["role"],
                "q_program_id": qid,
                "family": program["family"],
                "sign": program["sign"],
                "q_norm": float(np.linalg.norm(q)),
            }
            row.update({f"q{i + 1}": float(q[i]) for i in range(48)})
            writer.writerow(row)


def build_task_rows(geometry_roles: list[dict[str, Any]], programs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for geom in geometry_roles:
        role = str(geom["role"])
        worker = str(geom["worker"])
        for program in programs:
            qid = str(program["q_program_id"])
            rows.append(
                {
                    "task_id": f"gen_{role}_{qid}",
                    "phase": "abaqus_generation",
                    "worker": worker,
                    "geometry_role": role,
                    "geometry_status": geom["status"],
                    "q_program_id": qid,
                    "q_family": program["family"],
                    "requires_abaqus": True,
                    "requires_network_training": False,
                    "uses_X_macro": False,
                    "uses_internal_fine_grid_nodes_as_input": False,
                    "output_needed": "fresh TRUE176 complete compact with RF, RF_plus, LE128, B128, IVOL, X16/X_keep, q48",
                }
            )
        rows.append(
            {
                "task_id": f"post_{role}_source128_x16_weight_audit",
                "phase": "linux_postprocess_audit",
                "worker": "linux_postprocess",
                "geometry_role": role,
                "geometry_status": geom["status"],
                "q_program_id": "all_programs",
                "q_family": "all",
                "requires_abaqus": False,
                "requires_network_training": False,
                "uses_X_macro": False,
                "uses_internal_fine_grid_nodes_as_input": False,
                "output_needed": "Macro16 source128 compact with macro16-x16 weights and force/stiffness audit JSON",
            }
        )
    rows.append(
        {
            "task_id": "global_acceptance_summary",
            "phase": "acceptance",
            "worker": "linux_postprocess",
            "geometry_role": "all",
            "geometry_status": "after_all_roles_available",
            "q_program_id": "all_programs",
            "q_family": "all",
            "requires_abaqus": False,
            "requires_network_training": False,
            "uses_X_macro": False,
            "uses_internal_fine_grid_nodes_as_input": False,
            "output_needed": "force_rel_max < 0.02, stiffness_rel_max < 0.02, no strong-distortion blow-up, rigid programs add no extra strain/force",
        }
    )
    return rows


def write_task_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def worker_commands(
    *,
    out_root: Path,
    compact_list: str,
    remote_windows_host: str,
    linux_host: str,
    strain_coordinate_mode: str,
    force_threshold: float,
    stiffness_threshold: float,
) -> dict[str, str]:
    compact_text = compact_list or "<fresh_complete_compact_list.txt>"
    source128_root = out_root / "macro16_source128_x16weights"
    audit_json = out_root / "macro16_source128_x16weights_force_stiffness_audit.json"
    local = f"""# Local Windows worker
# Roles: regular, lightly_distorted
# Goal: run fresh Abaqus jobs for all q48 programs in q48_programs/*.csv.
# Do not train a network. Do not use X_macro as model input.

py -3 scripts\\plan_macro16_source128_generality_audit.py --compact-list {compact_text} --out-root {out_root}

# After fresh complete compacts are exported, add their paths to a compact list
# and let the Linux worker run the standard Macro16 source128 postprocess.
"""
    remote = f"""# Remote Windows Abaqus worker: {remote_windows_host}
# Roles: moderately_distorted, strongly_distorted_not_flipped
# Goal: generate or select non-flipped X16 geometries, run the same q48 programs,
# and export fresh TRUE176 complete compacts with RF/RF_plus/LE128/B128/IVOL.

ssh {remote_windows_host} hostname

# Copy this plan folder to the remote machine, generate the missing geometries,
# then return the exported complete compact paths to the Linux worker.
"""
    linux = f"""# Linux postprocess worker: {linux_host}
# Goal: convert fresh complete compacts into standard Macro16 source128 compacts
# using X16-generated weights, then audit force and stiffness.
# Sync the plan folder and fresh compact list to the Linux machine first, then
# set PLAN_ROOT and COMPACT_LIST to the Linux paths.

PLAN_ROOT="${{PLAN_ROOT:-runs/macro16_source128_generality_plan}}"
COMPACT_LIST="${{COMPACT_LIST:-<synced_fresh_complete_compact_list.txt>}}"

python scripts/build_macro16_source128_teacher.py \\
  --compact-list "$COMPACT_LIST" \\
  --out-root "$PLAN_ROOT/macro16_source128_x16weights" \\
  --strain-coordinate-mode {strain_coordinate_mode} \\
  --source-node-order auto \\
  --volume-weight-mode macro16-x16

python scripts/audit_macro16_force_stiffness.py \\
  --compact-list "$PLAN_ROOT/macro16_source128_x16weights/macro16_source128_teacher_compact_list.txt" \\
  --out "$PLAN_ROOT/macro16_source128_x16weights_force_stiffness_audit.json" \\
  --weight-mode auto \\
  --max-force-rel {force_threshold:.12g} \\
  --max-stiffness-rel {stiffness_threshold:.12g} \\
  --strict
"""
    return {
        "worker_local_windows.txt": local,
        "worker_remote_windows.txt": remote,
        "worker_linux_postprocess.txt": linux,
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    paths = collect_paths(args.compact, args.compact_list, case_limit=int(args.case_limit))
    samples = load_geometry_samples(
        paths,
        frame_stride=int(args.frame_stride),
        max_rows_per_compact=int(args.max_rows_per_compact),
        source_node_order=str(args.source_node_order),
    ) if paths else []
    groups = group_geometries(samples, decimals=int(args.geometry_round_decimals)) if samples else []
    geometry_roles = select_geometry_roles(groups)
    programs = q_program_definitions()

    q_dir = out_root / "q48_programs"
    for role in geometry_roles:
        write_q48_csv(q_dir / f"{role['role']}_q48_programs.csv", role, programs, q_scale=float(args.q_scale), random_seed=int(args.random_seed))

    tasks = build_task_rows(geometry_roles, programs)
    write_task_manifest(out_root / "task_manifest.csv", tasks)
    write_json(out_root / "geometry_candidate_pool.json", {"geometry_count": len(groups), "geometries": groups})
    write_json(out_root / "geometry_cases.json", {"geometry_roles": geometry_roles})
    write_json(
        out_root / "q_programs.json",
        {
            "q_scale": float(args.q_scale),
            "random_seed": int(args.random_seed),
            "q48_order": "Macro16 node order, q[3*i+dof] for 16 boundary control nodes and 3 translations",
            "program_count": len(programs),
            "programs": programs,
        },
    )

    compact_list_text = str(Path(args.compact_list[0]).resolve()) if getattr(args, "compact_list", []) else ""
    commands = worker_commands(
        out_root=out_root,
        compact_list=compact_list_text,
        remote_windows_host=str(args.remote_windows_host),
        linux_host=str(args.linux_host),
        strain_coordinate_mode=str(args.strain_coordinate_mode),
        force_threshold=float(args.force_threshold),
        stiffness_threshold=float(args.stiffness_threshold),
    )
    for filename, text in commands.items():
        (out_root / filename).write_text(text, encoding="utf-8")

    available_roles = [role["role"] for role in geometry_roles if role.get("status") == "available"]
    missing_roles = [role["role"] for role in geometry_roles if role.get("status") != "available"]
    summary = {
        "script": "plan_macro16_source128_generality_audit",
        "out_root": str(out_root),
        "input_compact_count": len(paths),
        "geometry_group_count": len(groups),
        "available_geometry_roles": available_roles,
        "missing_geometry_roles": missing_roles,
        "q_program_count": len(programs),
        "task_count": len(tasks),
        "workers": {
            "local_windows": "regular and lightly_distorted Abaqus generation",
            "remote_windows_abaqus": "moderately_distorted and strongly_distorted_not_flipped Abaqus generation",
            "linux_postprocess": "source128 X16-weight build and force/stiffness audit",
        },
        "standard_macro16_contract": {
            "geometry_input": "X16 only",
            "displacement_input": "q48 only",
            "point_rule": "fixed 128 parent-domain source128 rule",
            "weight_rule": "generated from X16 by Macro16 isoparametric map",
            "strain_coordinate_mode": str(args.strain_coordinate_mode),
            "uses_X_macro": False,
            "uses_internal_fine_grid_nodes_as_input": False,
            "network_training": False,
        },
        "acceptance": {
            "force_rel_max": float(args.force_threshold),
            "stiffness_rel_max": float(args.stiffness_threshold),
            "strong_distortion_no_sudden_blow_up": True,
            "rigid_programs_no_extra_strain_or_force": True,
        },
        "outputs": [
            "geometry_candidate_pool.json",
            "geometry_cases.json",
            "q_programs.json",
            "task_manifest.csv",
            "q48_programs/*_q48_programs.csv for available geometries",
            "worker_local_windows.txt",
            "worker_remote_windows.txt",
            "worker_linux_postprocess.txt",
        ],
    }
    write_json(out_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, default=json_default))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="append", type=Path, default=[])
    parser.add_argument("--compact-list", action="append", type=Path, default=[])
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-rows-per-compact", type=int, default=1)
    parser.add_argument("--source-node-order", default="auto", choices=("auto", "macro16", "true176-keep"))
    parser.add_argument("--geometry-round-decimals", type=int, default=8)
    parser.add_argument("--q-scale", type=float, default=1.0e-3)
    parser.add_argument("--random-seed", type=int, default=20260625)
    parser.add_argument("--strain-coordinate-mode", default="global-to-macro-local", choices=("global-to-macro-local", "source-local"))
    parser.add_argument("--force-threshold", type=float, default=2.0e-2)
    parser.add_argument("--stiffness-threshold", type=float, default=2.0e-2)
    parser.add_argument("--remote-windows-host", default="Administrator@100.110.69.33")
    parser.add_argument("--linux-host", default="lab-gpu-ts")
    return parser.parse_args()


def main() -> None:
    build_plan(parse_args())


if __name__ == "__main__":
    main()
