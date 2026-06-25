#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate and audit Gate 04 wind-turbine shell Macro16 source128 data.

This script does not train a network, does not change the Macro16 model, and
does not change the fixed 128-point source rule.  It creates four structured
4x4 CSS8 teacher patches, runs one q48 path plus 48 forward perturbations for
each patch, exports complete TRUE176/CSS8 compacts, converts them to Macro16
source128 compacts, and runs the Gate 04 data-contract and force audits.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from macro_deeponet.macro16_geometry import Macro16GeometryMap, macro16_source128_point_table  # noqa: E402
from build_macro16_from_128_teacher import TRUE176_MACRO_TO_KEEP_NODE  # noqa: E402
from build_macro16_source128_teacher import build_all as build_source128_all  # noqa: E402
from audit_macro16_rigid_preprocessing import run_audit as run_contract_audit  # noqa: E402
from audit_macro16_force_stiffness import run_audit as run_force_audit  # noqa: E402

DEFAULT_OLD_SRC_ROOT = Path(r"D:\IS-FEM\NNSE_css8_push_tmp")
DEFAULT_EXPORT_LIB = Path(r"D:\IS-FEM\SRCv3.1 - 1\SRCv3.1\scripts\export_three_element_bfk_arrays.py")
DEFAULT_ABAQUS = Path(r"D:\Program Files\SIMULIA\Commands\abaqus.bat")

NX = 4
NY = 4
GAUSS_FAMILIES = ("cylindrical_shell", "conical_shell", "thickness_varying_shell", "mild_double_curvature_shell")
NODE_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


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
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def node_id(i: int, j: int, k: int) -> int:
    return 1 + int(k) * (NX + 1) * (NY + 1) + int(j) * (NX + 1) + int(i)


def node_ijk(nid: int) -> tuple[int, int, int]:
    zero = int(nid) - 1
    per_layer = (NX + 1) * (NY + 1)
    k = zero // per_layer
    rem = zero % per_layer
    j = rem // (NX + 1)
    i = rem % (NX + 1)
    return int(i), int(j), int(k)


def boundary_keep_nodes() -> tuple[list[int], list[int]]:
    boundary_nodes: list[int] = []
    keep_nodes: list[int] = []
    keep_ij = {(0, 0), (NX // 2, 0), (NX, 0), (NX, NY // 2), (NX, NY), (NX // 2, NY), (0, NY), (0, NY // 2)}
    for k in range(2):
        for j in range(NY + 1):
            for i in range(NX + 1):
                if i == 0 or i == NX or j == 0 or j == NY:
                    nid = node_id(i, j, k)
                    boundary_nodes.append(nid)
                    if (i, j) in keep_ij:
                        keep_nodes.append(nid)
    return boundary_nodes, keep_nodes


def css8_elements() -> dict[int, list[int]]:
    elements: dict[int, list[int]] = {}
    eid = 1
    for j in range(NY):
        for i in range(NX):
            elements[eid] = [
                node_id(i, j, 0),
                node_id(i + 1, j, 0),
                node_id(i + 1, j + 1, 0),
                node_id(i, j + 1, 0),
                node_id(i, j, 1),
                node_id(i + 1, j, 1),
                node_id(i + 1, j + 1, 1),
                node_id(i, j + 1, 1),
            ]
            eid += 1
    return elements


def build_t_boundary() -> tuple[np.ndarray, list[dict[str, Any]]]:
    boundary_nodes, keep_nodes = boundary_keep_nodes()
    keep_index = {node: idx for idx, node in enumerate(keep_nodes)}
    tmat = np.zeros((len(boundary_nodes) * 3, len(keep_nodes) * 3), dtype=np.float64)
    rows: list[dict[str, Any]] = []

    def lagrange3(t: float) -> tuple[float, float, float]:
        return ((1.0 - 2.0 * t) * (1.0 - t), 4.0 * t * (1.0 - t), t * (2.0 * t - 1.0))

    for bi, node in enumerate(boundary_nodes):
        i, j, k = node_ijk(node)
        if j == 0:
            masters_ijk = ((0, 0, k), (NX // 2, 0, k), (NX, 0, k), float(i) / float(NX))
        elif i == NX:
            masters_ijk = ((NX, 0, k), (NX, NY // 2, k), (NX, NY, k), float(j) / float(NY))
        elif j == NY:
            masters_ijk = ((NX, NY, k), (NX // 2, NY, k), (0, NY, k), float(NX - i) / float(NX))
        elif i == 0:
            masters_ijk = ((0, NY, k), (0, NY // 2, k), (0, 0, k), float(NY - j) / float(NY))
        else:
            raise RuntimeError(f"node {node} is not on the perimeter")
        t = float(masters_ijk[3])
        weights = lagrange3(t)
        masters = [node_id(ii, jj, kk) for ii, jj, kk in masters_ijk[:3]]
        for comp in range(3):
            rr = 3 * bi + comp
            for master, weight in zip(masters, weights):
                cc = 3 * keep_index[int(master)] + comp
                tmat[rr, cc] += float(weight)
        rows.append({"boundary_node": int(node), "masters": masters, "weights": [float(v) for v in weights], "ijk": [int(i), int(j), int(k)]})
    return tmat, rows


def normalize(vec: np.ndarray) -> np.ndarray:
    vals = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vals))
    if norm <= 1.0e-14 or not math.isfinite(norm):
        raise ValueError(f"degenerate normal vector norm {norm:g}")
    return vals / norm


def cylindrical_point(u: float, v: float, w: float, *, theta: float = 0.55, lam: float = 1.0, tau: float = 0.02) -> np.ndarray:
    radius = lam / theta
    phi = -theta * float(v)
    normal = np.asarray([math.cos(phi), 0.0, math.sin(phi)], dtype=np.float64)
    mid = np.asarray([radius * math.cos(phi), float(u), radius * math.sin(phi)], dtype=np.float64)
    return mid + float(w) * tau * normal


def conical_point(u: float, v: float, w: float, *, theta: float = 0.50, lam: float = 1.05, tau: float = 0.02, mu: float = 0.14) -> np.ndarray:
    root = math.sqrt(1.0 + mu * mu)
    radius_mid = lam / theta
    radius = radius_mid + mu * float(u)
    phi = -theta * float(v)
    normal = np.asarray([math.cos(phi) / root, -mu / root, math.sin(phi) / root], dtype=np.float64)
    mid = np.asarray([radius * math.cos(phi), float(u), radius * math.sin(phi)], dtype=np.float64)
    return mid + float(w) * tau * normal


def thickness_varying_point(u: float, v: float, w: float, *, theta: float = 0.45, lam: float = 1.0, tau0: float = 0.022) -> np.ndarray:
    radius = lam / theta
    phi = -theta * float(v)
    normal = np.asarray([math.cos(phi), 0.0, math.sin(phi)], dtype=np.float64)
    thickness = tau0 * (1.0 + 0.30 * float(u) + 0.12 * float(v))
    mid = np.asarray([radius * math.cos(phi), float(u), radius * math.sin(phi)], dtype=np.float64)
    return mid + float(w) * thickness * normal


def mild_double_curvature_point(u: float, v: float, w: float, *, lam: float = 1.0, tau: float = 0.02) -> np.ndarray:
    cu = 0.080
    cv = 0.055
    cxy = 0.030
    x = lam * float(v)
    y = float(u)
    z = cu * (float(u) ** 2 - 1.0 / 12.0) + cv * (float(v) ** 2 - 1.0 / 12.0) + cxy * float(u) * float(v)
    p_v = np.asarray([lam, 0.0, 2.0 * cv * float(v) + cxy * float(u)], dtype=np.float64)
    p_u = np.asarray([0.0, 1.0, 2.0 * cu * float(u) + cxy * float(v)], dtype=np.float64)
    normal = normalize(np.cross(p_v, p_u))
    return np.asarray([x, y, z], dtype=np.float64) + float(w) * tau * normal


def family_point_function(family: str) -> Callable[[float, float, float], np.ndarray]:
    funcs: dict[str, Callable[[float, float, float], np.ndarray]] = {
        "cylindrical_shell": cylindrical_point,
        "conical_shell": conical_point,
        "thickness_varying_shell": thickness_varying_point,
        "mild_double_curvature_shell": mild_double_curvature_point,
    }
    if family not in funcs:
        raise ValueError(f"unknown wind shell family {family!r}")
    return funcs[family]


def build_nodes(family: str) -> dict[int, np.ndarray]:
    fn = family_point_function(family)
    nodes: dict[int, np.ndarray] = {}
    for k in range(2):
        w = -0.5 + float(k)
        for j in range(NY + 1):
            u = -0.5 + float(j) / float(NY)
            for i in range(NX + 1):
                v = -0.5 + float(i) / float(NX)
                nodes[node_id(i, j, k)] = fn(u, v, w).astype(np.float64)
    return nodes


def x16_macro_from_nodes(nodes: dict[int, np.ndarray]) -> np.ndarray:
    _boundary, keep = boundary_keep_nodes()
    x_keep = np.vstack([nodes[int(n)] for n in keep]).astype(np.float64)
    return x_keep[TRUE176_MACRO_TO_KEEP_NODE].astype(np.float64)


def geometry_metrics(family: str) -> dict[str, Any]:
    nodes = build_nodes(family)
    x16 = x16_macro_from_nodes(nodes)
    geom = Macro16GeometryMap(x16)
    fields = geom.eval_points(macro16_source128_point_table())
    det = np.asarray(fields["detJ_hat"], dtype=np.float64)
    thickness = np.asarray(fields["thickness_hat"], dtype=np.float64)
    return {
        "family": family,
        "detJ_positive": bool(np.min(det) > 0.0),
        "detJ_min": float(np.min(det)),
        "detJ_max": float(np.max(det)),
        "detJ_ratio": float(np.max(det) / max(float(np.min(det)), 1.0e-30)),
        "thickness_min": float(np.min(thickness)),
        "thickness_max": float(np.max(thickness)),
        "thickness_ratio": float(np.max(thickness) / max(float(np.min(thickness)), 1.0e-30)),
        "L_ref": float(geom.l_ref),
        "X16": x16.astype(float).tolist(),
    }


def mixed_q48_keep(nodes: dict[int, np.ndarray], *, q_scale: float) -> np.ndarray:
    _boundary, keep = boundary_keep_nodes()
    x_keep = np.vstack([nodes[int(n)] for n in keep]).astype(np.float64)
    center = np.mean(x_keep, axis=0)
    rel = x_keep - center.reshape(1, 3)
    span = np.max(x_keep, axis=0) - np.min(x_keep, axis=0)
    l_ref = float(np.max(span))
    if l_ref <= 1.0e-14:
        raise ValueError("degenerate keep-node span")
    h = rel / l_ref
    x = h[:, 0]
    y = h[:, 1]
    z = h[:, 2]
    q = np.zeros((16, 3), dtype=np.float64)
    q[:, 0] = 0.45 * y + 0.20 * x * z
    q[:, 1] = -0.25 * x + 0.18 * y * z
    q[:, 2] = 0.55 * (x * x - np.mean(x * x)) + 0.25 * x * y
    q -= np.mean(q, axis=0, keepdims=True)
    norm = float(np.linalg.norm(q.reshape(-1)))
    if norm <= 1.0e-30:
        raise ValueError("generated q48 path is zero")
    return (q.reshape(48) * (float(q_scale) * l_ref / norm)).astype(np.float64)


def amplitude_lines(name: str, times: np.ndarray, values: np.ndarray) -> list[str]:
    pairs: list[str] = []
    for t, value in zip(times.reshape(-1), values.reshape(-1)):
        pairs.extend(["%.12g" % float(t), "%.12e" % float(value)])
    lines = [f"*Amplitude, name={name}, time=TOTAL TIME"]
    for start in range(0, len(pairs), 8):
        lines.append(", ".join(pairs[start : start + 8]))
    return lines


def fmt_set(values: list[int]) -> list[str]:
    return [", ".join(str(v) for v in values[start : start + 16]) for start in range(0, len(values), 16)]


def q48_frames(q48_final: np.ndarray, *, increments: int, perturb_direction: int | None, delta: float) -> tuple[np.ndarray, np.ndarray]:
    times = np.linspace(0.0, 1.0, int(increments) + 1, dtype=np.float64)
    q = np.zeros((int(increments) + 1, 48), dtype=np.float64)
    final = np.asarray(q48_final, dtype=np.float64).reshape(48)
    for frame in range(1, int(increments) + 1):
        q[frame] = times[frame] * final
        if perturb_direction is not None:
            q[frame, int(perturb_direction)] += float(delta)
    return times, q


def write_generic_css8_shell_inp(
    out_path: Path,
    *,
    family: str,
    q48_final: np.ndarray,
    increments: int,
    perturb_direction: int | None,
    delta: float,
    nlgeom: str,
) -> dict[str, Any]:
    nodes = build_nodes(family)
    boundary_nodes, keep_nodes = boundary_keep_nodes()
    tmat, t_rows = build_t_boundary()
    times, q48 = q48_frames(q48_final, increments=increments, perturb_direction=perturb_direction, delta=delta)
    q_boundary = q48 @ tmat.T
    elements = css8_elements()
    nlgeom_norm = str(nlgeom).strip().upper()
    lines = [
        "*Heading",
        f"** Gate 04 wind shell family = {family}",
        "*Preprint, echo=NO, model=NO, history=NO, contact=NO",
        "*Node",
    ]
    for nid in sorted(nodes):
        x, y, z = nodes[int(nid)]
        lines.append("%d, %.12g, %.12g, %.12g" % (int(nid), float(x), float(y), float(z)))
    lines.append("*Element, type=CSS8")
    for eid in sorted(elements):
        lines.append("%d, %s" % (int(eid), ", ".join(str(int(v)) for v in elements[int(eid)])))
    lines.extend(["*Nset, nset=BOUNDARY_NODES"])
    lines.extend(fmt_set(boundary_nodes))
    lines.extend(["*Nset, nset=KEEP_NODES"])
    lines.extend(fmt_set(keep_nodes))
    lines.extend(["*Elset, elset=EALL, generate", "1, %d, 1" % len(elements)])
    lines.extend(["*Solid Section, elset=EALL, material=MAT_ELASTIC", ","])
    lines.extend(["*Material, name=MAT_ELASTIC", "*Elastic", "210000., 0.3"])
    amp_names: list[str] = []
    for row in range(q_boundary.shape[1]):
        name = f"A{row:03d}"
        amp_names.append(name)
        lines.extend(amplitude_lines(name, times, q_boundary[:, row]))
    lines.extend(
        [
            f"*Step, name=STEP_BASE, nlgeom={nlgeom_norm}, inc={int(increments)}",
            "*Static",
            "%.12g, 1., %.12g, %.12g" % (1.0 / float(increments), 1.0 / float(increments), 1.0 / float(increments)),
        ]
    )
    for bi, node in enumerate(boundary_nodes):
        for dof in range(1, 4):
            row = 3 * bi + (dof - 1)
            lines.append(f"*Boundary, amplitude={amp_names[row]}")
            lines.append("%d, %d, %d, 1." % (int(node), dof, dof))
    lines.extend(
        [
            "*Output, field, number interval=%d, time marks=YES" % int(increments),
            "*Node Output",
            "U, RF, COORD",
            "*Element Output, directions=YES",
            "S, E, LE, COORD, SENER, IVOL, EVOL",
            "*End Step",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    x_keep = np.vstack([nodes[int(n)] for n in keep_nodes]).astype(np.float64)
    x_boundary = np.vstack([nodes[int(n)] for n in boundary_nodes]).astype(np.float64)
    return {
        "inp": str(out_path),
        "family": family,
        "nodes": nodes,
        "boundary_nodes": boundary_nodes,
        "keep_nodes": keep_nodes,
        "T_boundary_96x48": tmat,
        "T_rows": t_rows,
        "X_keep_ref": x_keep,
        "X_boundary_ref": x_boundary,
        "q48_frames": q48,
        "q_boundary_frames": q_boundary,
        "frame_times": times,
        "increments": int(increments),
        "perturb_direction": None if perturb_direction is None else int(perturb_direction),
        "delta": float(delta),
        "nlgeom": nlgeom_norm,
    }


def short_job(case_id: str, suffix: str) -> str:
    import hashlib
    import re

    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(case_id)).strip("_") or "case"
    digest = hashlib.sha1((str(case_id) + "_" + str(suffix)).encode("utf-8")).hexdigest()[:7]
    stem = safe[: max(1, 24 - len(digest) - 4)].strip("_") or "case"
    return f"g_{stem}_{digest}"


def setup_old_imports(old_src_root: Path) -> dict[str, Any]:
    old_scripts = Path(old_src_root).resolve() / "scripts"
    if not old_scripts.exists():
        raise FileNotFoundError(f"old source scripts not found: {old_scripts}")
    sys.path.insert(0, str(old_scripts))
    from css8_128_input_preprocess import preprocess_q48_raw  # type: ignore
    from run_css8_128_tower11_allframes_parallel_generation import _run_export_job  # type: ignore
    from run_css8_128_native_parallel_generation import _cos, _dmat_engineering, _infer_ivol_from_g_dle, _rel  # type: ignore

    return {
        "preprocess_q48_raw": preprocess_q48_raw,
        "run_export_job": _run_export_job,
        "cos": _cos,
        "dmat_engineering": _dmat_engineering,
        "infer_ivol_from_g_dle": _infer_ivol_from_g_dle,
        "rel": _rel,
    }


def export_status_ok(row: dict[str, Any]) -> bool:
    return str(row.get("export", {}).get("status")) in {"OK", "SKIPPED_EXISTING_NPZ"}


def run_jobs(jobs: list[dict[str, Any]], *, run_export_job: Callable[..., dict[str, Any]], args: Any, logs: Path, workers: int) -> list[dict[str, Any]]:
    if int(workers) <= 1:
        return [run_export_job(args=args, logs=logs, attempt=1, **job) for job in jobs]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = [pool.submit(run_export_job, args=args, logs=logs, attempt=1, **job) for job in jobs]
        for fut in as_completed(futures):
            rows.append(fut.result())
    return rows


def frame_positions(frame_indices: np.ndarray, selected: list[int]) -> list[int]:
    index = {int(v): i for i, v in enumerate(np.asarray(frame_indices, dtype=np.int64).reshape(-1).tolist())}
    return [int(index[int(v)]) for v in selected]


def explicit_force(le: np.ndarray, b: np.ndarray, ivol: np.ndarray, dmat: np.ndarray) -> np.ndarray:
    stress = np.einsum("nac,dc->nad", le, dmat)
    g = stress * ivol[:, :, None]
    return np.einsum("naci,nac->ni", b, g)


def build_b_compact(sample_root: Path, case_id: str, args: Any, perturb_jobs: list[dict[str, Any]], old: dict[str, Any]) -> dict[str, Any]:
    candidate = str(args.candidate)
    ykey = f"candidate_{candidate}_y"
    gkey = f"candidate_{candidate}_g"
    kkey = f"candidate_{candidate}_keys"
    base = np.load(str(sample_root / "base_frames.npz"), allow_pickle=True)
    contract = np.load(str(sample_root / "boundary_contract" / "boundary_contract_arrays.npz"), allow_pickle=True)
    for key in (ykey, gkey, kkey):
        if key not in base.files:
            raise KeyError(f"base export missing {key}")
    base_frame_indices = np.asarray(base["frame_indices"], dtype=np.int64).reshape(-1)
    base_frame_values = np.asarray(base["frame_values"], dtype=np.float64).reshape(-1)
    directions = sorted(int(job["direction"]) for job in perturb_jobs)
    common = set(int(v) for v in base_frame_indices.tolist())
    plus_npz: dict[int, Any] = {}
    plus_frame_indices: dict[int, np.ndarray] = {}
    for job in perturb_jobs:
        direction = int(job["direction"])
        z = np.load(str(Path(job["npz"]).resolve()), allow_pickle=True)
        plus_npz[direction] = z
        idx = np.asarray(z["frame_indices"], dtype=np.int64).reshape(-1)
        plus_frame_indices[direction] = idx
        common.intersection_update(int(v) for v in idx.tolist())
    selected_common = [int(v) for v in base_frame_indices.tolist() if int(v) in common]
    if not selected_common:
        raise RuntimeError(f"{case_id}: base/plus exports have no common frames")
    if (not bool(args.allow_partial_frames)) and selected_common != base_frame_indices.tolist():
        raise RuntimeError(f"{case_id}: partial frame intersection detected")
    base_pos = frame_positions(base_frame_indices, selected_common)
    plus_pos = {direction: frame_positions(plus_frame_indices[direction], selected_common) for direction in directions}
    frame_indices = np.asarray(selected_common, dtype=np.int64)
    frame_values = base_frame_values[base_pos]
    q48_all = np.asarray(contract["q48_frames"], dtype=np.float64)
    q48 = q48_all[frame_indices]
    t_boundary = np.asarray(base["T_boundary_keep"], dtype=np.float64).reshape(96, 48)
    t_local = np.asarray(contract["T_boundary_96x48"], dtype=np.float64).reshape(96, 48)
    x_boundary = np.asarray(contract["X_boundary_ref"], dtype=np.float64).reshape(32, 3)
    pre = old["preprocess_q48_raw"](q48, t_boundary, x_boundary)

    n_frame = int(frame_indices.size)
    q_clean_plus = np.zeros((n_frame, len(directions), 96), dtype=np.float64)
    for col, direction in enumerate(directions):
        qp = q48.copy()
        qp[:, direction] += float(args.delta)
        q_clean_plus[:, col] = np.asarray(old["preprocess_q48_raw"](qp, t_boundary, x_boundary).q_boundary_clean, dtype=np.float64).reshape(n_frame, 96)
    le_base = np.asarray(base[ykey], dtype=np.float64).reshape(base_frame_indices.size, 128, 6)[base_pos]
    g_abq = np.asarray(base[gkey], dtype=np.float64).reshape(base_frame_indices.size, 128, 6)[base_pos]
    ip_keys = np.asarray(base[kkey], dtype=np.int64).reshape(128, 3)
    rf_all = np.asarray(base["rf_projected"], dtype=np.float64).reshape(base_frame_indices.size, -1)[base_pos]
    rf = rf_all[:, directions]
    dmat = old["dmat_engineering"](float(args.elastic_e), float(args.nu))
    ivol = np.zeros((n_frame, 128), dtype=np.float64)
    stress_dle = np.zeros((n_frame, 128, 6), dtype=np.float64)
    g_dle = np.zeros((n_frame, 128, 6), dtype=np.float64)
    for f in range(n_frame):
        ivol[f], stress_dle[f], g_dle[f] = old["infer_ivol_from_g_dle"](le_base[f], g_abq[f], dmat)
    le_plus = np.zeros((n_frame, len(directions), 128, 6), dtype=np.float64)
    ivol_plus = np.zeros((n_frame, len(directions), 128), dtype=np.float64)
    rf_plus = np.zeros((n_frame, len(directions), len(directions)), dtype=np.float64)
    for col, direction in enumerate(directions):
        z = plus_npz[direction]
        count = int(np.asarray(z["frame_indices"], dtype=np.int64).reshape(-1).size)
        le_dir = np.asarray(z[ykey], dtype=np.float64).reshape(count, 128, 6)[plus_pos[direction]]
        g_dir = np.asarray(z[gkey], dtype=np.float64).reshape(count, 128, 6)[plus_pos[direction]]
        le_plus[:, col] = le_dir
        rf_plus[:, col] = np.asarray(z["rf_projected"], dtype=np.float64).reshape(count, -1)[plus_pos[direction]][:, directions]
        for f in range(n_frame):
            ivol_plus[f, col], _, _ = old["infer_ivol_from_g_dle"](le_dir[f], g_dir[f], dmat)
    delta = float(args.delta)
    b_le = np.transpose((le_plus - le_base[:, None, :, :]) / delta, (0, 2, 3, 1))
    f_dle_ivol = explicit_force(le_base, b_le, ivol, dmat)
    out_path = sample_root / "b_compact" / f"{case_id}_wind_shell_B_compact.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        sample_id=np.asarray([case_id], dtype=object),
        geometry_family=np.asarray(str(contract["geometry_family"].reshape(-1)[0]), dtype=object),
        frame_indices=frame_indices,
        frame_values=frame_values,
        perturb_directions=np.asarray(directions, dtype=np.int64),
        is_full_48_direction_sample=np.asarray([int(directions == list(range(48)))], dtype=np.int64),
        q48_raw=q48,
        q_boundary_raw=np.asarray(pre.q_boundary_raw, dtype=np.float64).reshape(n_frame, 32, 3),
        q_boundary_clean=np.asarray(pre.q_boundary_clean, dtype=np.float64).reshape(n_frame, 32, 3),
        q_boundary_clean_plus=q_clean_plus.reshape(n_frame, len(directions), 32, 3),
        T_boundary_96x48=t_boundary,
        T_boundary_local_96x48=t_local,
        X_keep_ref=np.asarray(contract["X_keep_ref"], dtype=np.float64).reshape(16, 3),
        X_boundary_ref=x_boundary,
        keep_nodes=np.asarray(base["keep_nodes"], dtype=np.int64),
        boundary_nodes=np.asarray(base["boundary_nodes"], dtype=np.int64),
        LE128_base=le_base,
        LE128_plus=le_plus,
        DLE128_forward=le_plus - le_base[:, None, :, :],
        B_LE128_forward=b_le,
        g_abq_S_IVOL=g_abq,
        IVOL128_inferred_from_DLE=ivol,
        IVOL128_plus_inferred_from_DLE=ivol_plus,
        S_DLE=stress_dle,
        g_DLE_IVOL=g_dle,
        ip_keys=ip_keys,
        RF_projected=rf,
        RF_projected_plus=rf_plus,
        F_DLE_IVOL=f_dle_ivol,
        elastic_D=dmat,
        delta=np.asarray([delta], dtype=np.float64),
        strain_field=np.asarray("LE", dtype=object),
        B_label_strain_field=np.asarray("LE", dtype=object),
        strain_label_key=np.asarray("LE128_base", dtype=object),
        B_label_key=np.asarray("B_LE128_forward", dtype=object),
    )
    summary = {
        "case_id": case_id,
        "b_compact": str(out_path),
        "frame_count": n_frame,
        "perturb_direction_count": len(directions),
        "F_DLE_IVOL_vs_RF_rel": old["rel"](f_dle_ivol, rf),
        "F_DLE_IVOL_vs_RF_cosine": old["cos"](f_dle_ivol, rf),
        "T_boundary_export_vs_local_rel": old["rel"](t_boundary, t_local),
        "ivol_min": float(np.min(ivol)),
        "ivol_max": float(np.max(ivol)),
        "field_shapes": {
            "q48_raw": list(q48.shape),
            "LE128_base": list(le_base.shape),
            "B_LE128_forward": list(b_le.shape),
            "RF_projected": list(rf.shape),
            "RF_projected_plus": list(rf_plus.shape),
        },
    }
    write_json(out_path.with_suffix(out_path.suffix + ".json"), summary)
    return summary


def run_subprocess(cmd: list[str], cwd: Path, log_path: Path, timeout_s: int) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: " + " ".join(str(v) for v in cmd) + "\n")
        log.flush()
        proc = subprocess.run([str(v) for v in cmd], cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True, timeout=int(timeout_s), check=False)
    return {"cmd": [str(v) for v in cmd], "cwd": str(cwd), "log": str(log_path), "returncode": int(proc.returncode)}


def export_complete(sample_root: Path, case_id: str, b_compact: Path, base_odb: Path, args: Any) -> dict[str, Any]:
    complete = sample_root / "complete" / f"complete_{case_id}_training_ready.npz"
    cmd = [
        str(Path(args.abaqus).resolve()),
        "python",
        str(ROOT / "scripts" / "export_abaqus_true176_complete_compact.py"),
        "--odb",
        str(base_odb),
        "--out",
        str(complete),
        "--frames",
        ",".join(str(i) for i in range(1, int(args.increments) + 1)),
        "--merge-compact",
        str(b_compact),
        "--strain-field",
        "LE",
        "--require-b",
        "--require-merge-ip-keys",
        "--require-ip-audit",
        "--merge-q-tol",
        str(args.merge_q_tol),
        "--merge-le-tol",
        str(args.merge_le_tol),
    ]
    res = run_subprocess(cmd, ROOT, sample_root / "logs" / f"{case_id}.complete_export.log", int(args.complete_export_timeout_s))
    res["complete_compact"] = str(complete)
    res["complete_compact_exists"] = bool(complete.exists())
    if int(res["returncode"]) != 0 or not complete.exists():
        raise RuntimeError(f"{case_id}: complete export failed; see {res['log']}")
    return res


def run_family(family: str, case_id: str, args: Any, old: dict[str, Any]) -> dict[str, Any]:
    sample_root = Path(args.out_root).resolve() / family / case_id
    sample_root.mkdir(parents=True, exist_ok=True)
    nodes = build_nodes(family)
    q_final = mixed_q48_keep(nodes, q_scale=float(args.q_scale))
    metrics = geometry_metrics(family)
    if not metrics["detJ_positive"]:
        raise ValueError(f"{family}: Macro16 detJ is not positive")
    write_json(sample_root / "geometry_metrics.json", metrics)
    np.save(sample_root / "q48_final.npy", q_final)

    run_args = SimpleNamespace(
        skip_existing=bool(args.skip_existing),
        abaqus=str(Path(args.abaqus).resolve()),
        abaqus_memory=str(args.abaqus_memory),
        job_timeout_s=int(args.job_timeout_s),
        export_timeout_s=int(args.export_timeout_s),
        allow_partial_frames=bool(args.allow_partial_frames),
        frame_start=1,
        frame_end=int(args.increments),
        candidate=str(args.candidate),
        export_lib=str(Path(args.export_lib).resolve()),
        src_root=str(Path(args.old_src_root).resolve()),
    )
    base_job = short_job(case_id, "base")
    base_dir = sample_root / "abaqus_run" / "base"
    base_inp = base_dir / f"{base_job}.inp"
    base_info = write_generic_css8_shell_inp(
        base_inp,
        family=family,
        q48_final=q_final,
        increments=int(args.increments),
        perturb_direction=None,
        delta=float(args.delta),
        nlgeom=str(args.nlgeom),
    )
    bc_dir = sample_root / "boundary_contract"
    bc_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(bc_dir / "boundary_contract_arrays.npz"),
        geometry_family=np.asarray(family, dtype=object),
        T_boundary_96x48=np.asarray(base_info["T_boundary_96x48"], dtype=np.float64),
        X_keep_ref=np.asarray(base_info["X_keep_ref"], dtype=np.float64),
        X_boundary_ref=np.asarray(base_info["X_boundary_ref"], dtype=np.float64),
        q48_frames=np.asarray(base_info["q48_frames"], dtype=np.float64),
        q_boundary_frames=np.asarray(base_info["q_boundary_frames"], dtype=np.float64),
        frame_times=np.asarray(base_info["frame_times"], dtype=np.float64),
        keep_nodes=np.asarray(base_info["keep_nodes"], dtype=np.int64),
        boundary_nodes=np.asarray(base_info["boundary_nodes"], dtype=np.int64),
        nlgeom=np.asarray([str(args.nlgeom).strip().upper()], dtype=object),
    )
    write_json(bc_dir / "boundary_contract.json", {k: v for k, v in base_info.items() if k not in {"nodes", "T_boundary_96x48", "q48_frames", "q_boundary_frames", "X_keep_ref", "X_boundary_ref"}})
    logs = sample_root / "logs"
    base_npz = sample_root / "base_frames.npz"
    base_res = old["run_export_job"](
        job_name=base_job,
        inp=base_inp,
        odb=base_dir / f"{base_job}.odb",
        npz=base_npz,
        args=run_args,
        logs=logs,
        attempt=1,
    )
    if not export_status_ok(base_res) or not base_npz.exists():
        raise RuntimeError(f"{case_id}: base job/export failed")

    perturb_root = sample_root / "abaqus_run" / "perturb"
    perturb_jobs: list[dict[str, Any]] = []
    for direction in range(48):
        job_name = short_job(case_id, f"d{direction:02d}")
        job_dir = perturb_root / job_name
        inp = job_dir / f"{job_name}.inp"
        write_generic_css8_shell_inp(
            inp,
            family=family,
            q48_final=q_final,
            increments=int(args.increments),
            perturb_direction=direction,
            delta=float(args.delta),
            nlgeom=str(args.nlgeom),
        )
        perturb_jobs.append({"direction": int(direction), "job_name": job_name, "inp": inp, "odb": job_dir / f"{job_name}.odb", "npz": job_dir / f"{job_name}.npz"})
    perturb_results = run_jobs(
        perturb_jobs,
        run_export_job=old["run_export_job"],
        args=run_args,
        logs=logs,
        workers=int(args.inner_workers),
    )
    failed = [row for row in perturb_results if not export_status_ok(row)]
    if failed:
        raise RuntimeError(f"{case_id}: {len(failed)} perturb exports failed")
    b_summary = build_b_compact(sample_root, case_id, args, perturb_jobs, old)
    complete_summary = export_complete(sample_root, case_id, Path(b_summary["b_compact"]), Path(base_res["odb"]), args)
    summary = {
        "family": family,
        "case_id": case_id,
        "sample_root": str(sample_root),
        "geometry_metrics": metrics,
        "q48_final_norm": float(np.linalg.norm(q_final)),
        "base_result": base_res,
        "perturb_job_count": len(perturb_results),
        "b_compact_summary": b_summary,
        "complete_export": complete_summary,
        "complete_compact": complete_summary["complete_compact"],
    }
    write_json(sample_root / "family_generation_summary.json", summary)
    return summary


def build_source128_and_audit(family_rows: list[dict[str, Any]], args: Any) -> dict[str, Any]:
    out_root = Path(args.out_root).resolve()
    complete_list = out_root / "complete_compact_list.txt"
    complete_list.write_text("\n".join(str(row["complete_compact"]) for row in family_rows) + "\n", encoding="utf-8")
    source_root = out_root / "macro16_source128"
    build_summary = build_source128_all(
        SimpleNamespace(
            compact=[],
            compact_list=[complete_list],
            out_root=source_root,
            case_limit=0,
            frame_stride=1,
            max_frames_per_compact=0,
            allow_missing_ip_keys=False,
            strain_coordinate_mode="global-to-macro-local",
            source_node_order="auto",
            volume_weight_mode="macro16-x16",
            workers=int(args.post_workers),
        )
    )
    all_macro_list = Path(build_summary["compact_list"]).resolve()
    macro_paths = read_path_list(all_macro_list)
    family_to_macro: dict[str, list[str]] = {}
    for path in macro_paths:
        with np.load(str(path), allow_pickle=True) as z:
            source = str(np.asarray(z["source_compact"]).reshape(-1)[0])
        matched = next((row["family"] for row in family_rows if str(row["complete_compact"]) == source), None)
        if matched is None:
            matched = "unknown"
        family_to_macro.setdefault(matched, []).append(str(path))

    audits: dict[str, Any] = {}
    for family, paths in sorted(family_to_macro.items()):
        family_dir = out_root / "audits" / family
        family_dir.mkdir(parents=True, exist_ok=True)
        list_path = family_dir / f"{family}_macro16_source128_compact_list.txt"
        list_path.write_text("\n".join(paths) + "\n", encoding="utf-8")
        contract_json = family_dir / f"{family}_data_contract_audit.json"
        force_json = family_dir / f"{family}_selected_material_only_force_audit.json"
        contract_summary = run_contract_audit(
            SimpleNamespace(
                compact=[],
                compact_list=[list_path],
                compact_glob=[],
                out=contract_json,
                case_limit=0,
                frame_stride=1,
                max_frames_per_compact=0,
                skip_synthetic=True,
                synthetic_tol=1.0e-10,
                max_scale_rel=1.0e-4,
                max_scale_abs=1.0e-5,
                max_projection_rel=1.0e-4,
                max_projection_abs=1.0e-5,
                max_rigid_closure_rel=1.0e-6,
                max_rigid_closure_abs=1.0e-6,
                max_rotation_matrix_orthogonality=1.0e-5,
                max_qdef_translation_orthogonality=0.0,
                max_qdef_rotation_orthogonality=0.0,
                strict=True,
            )
        )
        force_summary = run_force_audit(
            SimpleNamespace(
                compact=[],
                compact_list=[list_path],
                compact_glob=[],
                out=force_json,
                case_limit=0,
                weight_mode="auto",
                volume_mode="selected-frame",
                tangent_mode="material-only",
                max_force_rel=float(args.force_threshold),
                max_stiffness_rel=0.0,
                max_macro_stiffness_symmetry_rel=1.0e-10,
                max_active_tangent_symmetry_rel=0.0,
                max_plus_kdq_rel=0.0,
                max_source_force_rel=0.0,
                max_source_128_force_rel=0.0,
                workers=int(args.post_workers),
                strict=True,
            )
        )
        audits[family] = {
            "compact_list": str(list_path),
            "data_contract_audit": str(contract_json),
            "force_audit": str(force_json),
            "data_contract_summary": contract_summary,
            "force_summary": force_summary,
        }
    return {
        "complete_compact_list": str(complete_list),
        "macro16_source128_compact_list": str(all_macro_list),
        "build_summary": build_summary,
        "family_to_macro_compacts": family_to_macro,
        "audits": audits,
    }


def collect_report_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    generation_by_family = {row["family"]: row for row in summary["generation"]}
    for family, audit in sorted(summary["postprocess"]["audits"].items()):
        gen = generation_by_family[family]
        force_agg = audit["force_summary"]["aggregate"]
        contract_agg = audit["data_contract_summary"]["aggregate"]
        rows.append(
            {
                "family": family,
                "geometry_count": 1,
                "frame_count": int(force_agg.get("force_rel_count") or 0),
                "detJ_positive": bool(gen["geometry_metrics"]["detJ_positive"]),
                "detJ_min": float(gen["geometry_metrics"]["detJ_min"]),
                "data_contract_pass": bool(audit["data_contract_summary"]["strict_pass"]),
                "force_mean": float(force_agg["force_rel_mean"]),
                "force_max": float(force_agg["force_rel_max"]),
                "material_only_k_mean": float(force_agg["material_only_stiffness_rel_mean"]),
                "material_only_k_max": float(force_agg["material_only_stiffness_rel_max"]),
                "enter_next_class": bool(float(force_agg["force_rel_max"]) < 0.02 and audit["data_contract_summary"]["strict_pass"]),
                "compact_count": int(contract_agg["compact_count"]),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    old = setup_old_imports(Path(args.old_src_root))
    families = GAUSS_FAMILIES if str(args.families).strip().lower() == "all" else tuple(v.strip() for v in str(args.families).split(",") if v.strip())
    case_ids = {
        "cylindrical_shell": "case070_cylindrical_shell",
        "conical_shell": "case071_conical_shell",
        "thickness_varying_shell": "case072_thickness_varying_shell",
        "mild_double_curvature_shell": "case073_mild_double_curvature_shell",
    }
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    generation: list[dict[str, Any]] = []
    for family in families:
        generation.append(run_family(family, case_ids[family], args, old))
    post = build_source128_and_audit(generation, args)
    summary = {
        "script": "run_gate04_wind_shell_generality_audit",
        "out_root": str(out_root),
        "families": list(families),
        "network_training": False,
        "model_changed": False,
        "integration_rule": "fixed standard Macro16 source128 128-point rule",
        "abaqus_parallel_inner_workers": int(args.inner_workers),
        "post_workers": int(args.post_workers),
        "generation": generation,
        "postprocess": post,
    }
    summary["report_rows"] = collect_report_rows(summary)
    write_json(out_root / "gate04_wind_shell_summary.json", summary)
    print(json.dumps({"summary": str(out_root / "gate04_wind_shell_summary.json"), "rows": summary["report_rows"]}, ensure_ascii=False, sort_keys=True, default=json_default))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("runs") / "gate04_wind_shell_generality")
    parser.add_argument("--families", default="all")
    parser.add_argument("--old-src-root", type=Path, default=DEFAULT_OLD_SRC_ROOT)
    parser.add_argument("--export-lib", type=Path, default=DEFAULT_EXPORT_LIB)
    parser.add_argument("--abaqus", type=Path, default=DEFAULT_ABAQUS)
    parser.add_argument("--candidate", default="css8_le6_engineering")
    parser.add_argument("--increments", type=int, default=10)
    parser.add_argument("--q-scale", type=float, default=1.0e-3)
    parser.add_argument("--delta", type=float, default=1.0e-6)
    parser.add_argument("--nlgeom", default="YES", choices=("YES", "NO", "yes", "no"))
    parser.add_argument("--elastic-e", type=float, default=210000.0)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--abaqus-memory", default="512mb")
    parser.add_argument("--job-timeout-s", type=int, default=3600)
    parser.add_argument("--export-timeout-s", type=int, default=1200)
    parser.add_argument("--complete-export-timeout-s", type=int, default=1200)
    parser.add_argument("--inner-workers", type=int, default=4)
    parser.add_argument("--post-workers", type=int, default=8)
    parser.add_argument("--merge-q-tol", default="1e-8")
    parser.add_argument("--merge-le-tol", default="1e-8")
    parser.add_argument("--force-threshold", type=float, default=2.0e-2)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--allow-partial-frames", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
