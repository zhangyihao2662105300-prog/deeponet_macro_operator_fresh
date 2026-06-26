#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate and audit Gate 04 wind-turbine shell Macro16 source128 data.

This script does not train a network, does not change the Macro16 model, and
does not change the fixed 128-point source rule.

The production data route uses the same displacement-template method as the
2000-epoch TRUE176 shape4 training data:

    TRUE176 full48_vector.npy template
      -> legacy local-frame/H normalization
      -> target shell keep-node local frames
      -> 100-frame Abaqus displacement history
      -> base + 48 forward perturbation jobs
      -> LE/B compact export

The older synthetic mixed q path is retained only as an explicit compatibility
smoke mode.  Do not use it for formal wind-shell training data.
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
DEFAULT_TRUE176_ROOT = Path(
    r"D:\IS-FEM\NNSE-NeuralNetworkShellElement_git"
    r"\run_logs\true176_10x10_range_force176_linear_mid"
)
DEFAULT_TRUE176_DOMAIN = "mid_free_deform"
DEFAULT_TRUE176_VECTOR_KIND = "full48"
DEFAULT_TEMPLATE_CASES = "1,2,3,4"

NX = 4
NY = 4
GAUSS_FAMILIES = ("cylindrical_shell", "conical_shell", "thickness_varying_shell", "mild_double_curvature_shell")
OLD_INDEX_FOR_SHAPE4_INDEX = np.asarray(
    [0, 3, 5, 1, 6, 2, 4, 7, 8, 11, 13, 9, 14, 10, 12, 15],
    dtype=np.int64,
)
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


def parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for item in str(text).replace(";", ",").split(","):
        part = item.strip()
        if not part:
            continue
        if ":" in part:
            lo, hi = [int(v.strip()) for v in part.split(":", 1)]
            out.extend(range(lo, hi))
        else:
            out.append(int(part))
    if not out:
        raise ValueError("empty integer list")
    return out


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


def load_csv_dict(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def true176_case_dir(root: Path, domain: str, case_id: int) -> Path:
    return Path(root) / str(domain) / f"case{int(case_id):03d}"


def load_true176_q48(root: Path, domain: str, case_id: int, vector_kind: str) -> np.ndarray:
    path = true176_case_dir(root, domain, int(case_id)) / f"{vector_kind}_vector.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(np.load(str(path)), dtype=np.float64).reshape(16, 3)


def load_legacy_meta(root: Path, domain: str, case_id: int) -> dict[str, Any]:
    path = true176_case_dir(root, domain, int(case_id)) / "model_meta.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def legacy_h(meta: dict[str, Any]) -> float:
    geom = dict(meta.get("geometry", {}))
    h = float(geom.get("H2", 1.0))
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("bad legacy H2 in model_meta.json")
    return h


def load_legacy_master_rows(root: Path, domain: str, case_id: int) -> list[dict[str, str]]:
    path = true176_case_dir(root, domain, int(case_id)) / "full48_u.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = load_csv_dict(path)
    rows.sort(key=lambda r: int(float(r["full48_node_index0"])))
    if len(rows) != 16:
        raise ValueError(f"expected 16 full48 rows in {path}, got {len(rows)}")
    for i, row in enumerate(rows):
        if int(float(row["full48_node_index0"])) != i:
            raise ValueError(f"full48 rows are not indexed 0..15 in {path}")
    return rows


def legacy_coords_from_rows(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=np.float64)


def legacy_width_axial_radial_frames(coords: np.ndarray) -> np.ndarray:
    """Return old TRUE176 local axes as rows: width_css8, axial, radial_outward."""

    xyz = np.asarray(coords, dtype=np.float64).reshape(16, 3)
    theta = np.arctan2(xyz[:, 2], xyz[:, 0])
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    e_width = np.stack([sin_t, np.zeros_like(theta), -cos_t], axis=1)
    e_axial = np.broadcast_to(np.asarray([0.0, 1.0, 0.0], dtype=np.float64), e_width.shape)
    e_radial = np.stack([cos_t, np.zeros_like(theta), sin_t], axis=1)
    return np.stack([e_width, e_axial, e_radial], axis=1)


def frame_quality(frames: np.ndarray) -> dict[str, float]:
    ff = np.asarray(frames, dtype=np.float64).reshape(-1, 3, 3)
    gram = np.einsum("nai,nbi->nab", ff, ff, optimize=True)
    ident = np.eye(3, dtype=np.float64).reshape(1, 3, 3)
    det = np.linalg.det(ff)
    return {
        "orthogonality_error": float(np.max(np.abs(gram - ident))),
        "det_min": float(np.min(det)),
        "det_max": float(np.max(det)),
        "det_abs_error": float(np.max(np.abs(det - 1.0))),
    }


def target_keep_frames(nodes: dict[int, np.ndarray]) -> np.ndarray:
    """Return target keep-node frames as rows: width_css8, axial, normal_outward."""

    _boundary_nodes, keep_nodes = boundary_keep_nodes()
    frames: list[np.ndarray] = []
    for nid in keep_nodes:
        i, j, k = node_ijk(int(nid))
        i0 = max(0, i - 1)
        i1 = min(NX, i + 1)
        j0 = max(0, j - 1)
        j1 = min(NY, j + 1)
        if i0 == i1:
            raise ValueError(f"cannot build width tangent for node {nid}")
        if j0 == j1:
            raise ValueError(f"cannot build axial tangent for node {nid}")
        width = normalize(nodes[node_id(i1, j, k)] - nodes[node_id(i0, j, k)])
        axial_raw = normalize(nodes[node_id(i, j1, k)] - nodes[node_id(i, j0, k)])
        normal_raw = np.cross(width, axial_raw)
        if float(np.linalg.norm(normal_raw)) <= 1.0e-14:
            other_k = 1 - int(k)
            normal_raw = nodes[node_id(i, j, other_k)] - nodes[int(nid)]
            if int(k) == 1:
                normal_raw = -normal_raw
        normal = normalize(normal_raw)
        axial = normalize(np.cross(normal, width))
        width = normalize(np.cross(axial, normal))
        frames.append(np.stack([width, axial, normal], axis=0))
    return np.asarray(frames, dtype=np.float64).reshape(16, 3, 3)


def template_transfer_context(args: Any, template_case_id: int | None = None) -> dict[str, Any]:
    true176_root = Path(args.true176_root).resolve()
    domain = str(args.true176_domain)
    meta_case = int(template_case_id if template_case_id is not None else args.true176_meta_case)
    meta = load_legacy_meta(true176_root, domain, meta_case)
    h_ref = legacy_h(meta)
    rows = load_legacy_master_rows(true176_root, domain, meta_case)
    coords = legacy_coords_from_rows(rows)
    frames = legacy_width_axial_radial_frames(coords)
    return {
        "true176_root": true176_root,
        "true176_domain": domain,
        "true176_vector_kind": str(args.true176_vector_kind),
        "legacy_meta_case": int(meta_case),
        "legacy_h_ref": float(h_ref),
        "legacy_frames": frames,
        "legacy_frame_quality": frame_quality(frames),
        "legacy_full48_u_csv": str(true176_case_dir(true176_root, domain, meta_case) / "full48_u.csv"),
        "legacy_model_meta": str(true176_case_dir(true176_root, domain, meta_case) / "model_meta.json"),
    }


def transfer_true176_template_to_family(
    *,
    family: str,
    template_case_id: int,
    args: Any,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ctx = template_transfer_context(args) if context is None else context
    nodes = build_nodes(family)
    q_old = load_true176_q48(
        Path(ctx["true176_root"]),
        str(ctx["true176_domain"]),
        int(template_case_id),
        str(ctx["true176_vector_kind"]),
    )
    old_idx = OLD_INDEX_FOR_SHAPE4_INDEX
    q_old_shape_order = np.asarray(q_old, dtype=np.float64).reshape(16, 3)[old_idx]
    old_frames_shape_order = np.asarray(ctx["legacy_frames"], dtype=np.float64).reshape(16, 3, 3)[old_idx]
    q_local_hat = np.einsum("nai,ni->na", old_frames_shape_order, q_old_shape_order, optimize=True) / float(ctx["legacy_h_ref"])
    target_frames = target_keep_frames(nodes)
    geom_l_ref = float(Macro16GeometryMap(x16_macro_from_nodes(nodes)).l_ref)
    q_target = float(args.template_amplitude_scale) * geom_l_ref * np.einsum("na,nai->ni", q_local_hat, target_frames, optimize=True)
    q_target_local_hat = np.einsum("nai,ni->na", target_frames, q_target, optimize=True) / geom_l_ref
    local_rel = float(np.linalg.norm(q_target_local_hat - float(args.template_amplitude_scale) * q_local_hat) / max(np.linalg.norm(q_local_hat), 1.0e-300))
    return {
        "q48_final": q_target.reshape(48).astype(np.float64),
        "source": {
            "q_generation_method": "true176-template-local-frame-transfer",
            "true176_case_id": int(template_case_id),
            "true176_vector_kind": str(ctx["true176_vector_kind"]),
            "true176_vector_path": str(true176_case_dir(Path(ctx["true176_root"]), str(ctx["true176_domain"]), int(template_case_id)) / f"{ctx['true176_vector_kind']}_vector.npy"),
            "true176_root": str(ctx["true176_root"]),
            "true176_domain": str(ctx["true176_domain"]),
            "legacy_meta_case": int(ctx["legacy_meta_case"]),
            "legacy_h_ref": float(ctx["legacy_h_ref"]),
            "target_L_ref": geom_l_ref,
            "template_amplitude_scale": float(args.template_amplitude_scale),
            "uses_path_scale": False,
            "old_index_for_shape4_index": OLD_INDEX_FOR_SHAPE4_INDEX.tolist(),
            "legacy_frame_quality": ctx["legacy_frame_quality"],
            "target_frame_quality": frame_quality(target_frames),
            "local_template_preservation_rel": local_rel,
            "notes": (
                "Uses the 2000-epoch TRUE176 displacement-template route: complete full48 q template, "
                "legacy local-frame/H normalization, then rebuild in target shell keep-node frames. "
                "TRUE176 LE/B labels are not reused; target LE/B must be re-exported from Abaqus."
            ),
        },
    }


def select_template_case(family: str, args: Any) -> int:
    cases = parse_int_list(str(args.template_cases))
    families = GAUSS_FAMILIES if str(args.families).strip().lower() == "all" else tuple(v.strip() for v in str(args.families).split(",") if v.strip())
    try:
        idx = list(families).index(str(family))
    except ValueError:
        idx = list(GAUSS_FAMILIES).index(str(family)) if str(family) in GAUSS_FAMILIES else 0
    return int(cases[idx % len(cases)])


def build_q48_source(
    family: str,
    args: Any,
    context: dict[str, Any] | None = None,
    template_case_id: int | None = None,
) -> dict[str, Any]:
    mode = str(args.q_source_mode).strip().lower()
    nodes = build_nodes(family)
    if mode == "true176-template":
        template_case = select_template_case(family, args) if template_case_id is None else int(template_case_id)
        return transfer_true176_template_to_family(
            family=family,
            template_case_id=template_case,
            args=args,
            context=context,
        )
    if mode == "synthetic-mixed":
        q_final = mixed_q48_keep(nodes, q_scale=float(args.q_scale))
        return {
            "q48_final": q_final,
            "source": {
                "q_generation_method": "synthetic-mixed-compatibility-smoke",
                "q_scale": float(args.q_scale),
                "uses_path_scale": True,
                "notes": "Compatibility smoke only. Do not use as formal wind-shell training data.",
            },
        }
    raise ValueError(f"unknown q source mode {args.q_source_mode!r}")


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
        q_generation_method=np.asarray(str(contract["q_generation_method"].reshape(-1)[0]), dtype=object)
        if "q_generation_method" in contract.files
        else np.asarray("", dtype=object),
        q_source_true176_case_id=np.asarray(contract["q_source_true176_case_id"], dtype=np.int64)
        if "q_source_true176_case_id" in contract.files
        else np.asarray([-1], dtype=np.int64),
        q_source_true176_vector_path=np.asarray(str(contract["q_source_true176_vector_path"].reshape(-1)[0]), dtype=object)
        if "q_source_true176_vector_path" in contract.files
        else np.asarray("", dtype=object),
        q_source_legacy_h_ref=np.asarray(contract["q_source_legacy_h_ref"], dtype=np.float64)
        if "q_source_legacy_h_ref" in contract.files
        else np.asarray([np.nan], dtype=np.float64),
        q_source_target_l_ref=np.asarray(contract["q_source_target_l_ref"], dtype=np.float64)
        if "q_source_target_l_ref" in contract.files
        else np.asarray([np.nan], dtype=np.float64),
        q_source_template_amplitude_scale=np.asarray(contract["q_source_template_amplitude_scale"], dtype=np.float64)
        if "q_source_template_amplitude_scale" in contract.files
        else np.asarray([np.nan], dtype=np.float64),
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


def run_family(
    family: str,
    case_id: str,
    args: Any,
    old: dict[str, Any],
    q_context: dict[str, Any] | None = None,
    template_case_id: int | None = None,
) -> dict[str, Any]:
    sample_root = Path(args.out_root).resolve() / family / case_id
    sample_root.mkdir(parents=True, exist_ok=True)
    q_source = build_q48_source(family, args, q_context, template_case_id=template_case_id)
    q_final = np.asarray(q_source["q48_final"], dtype=np.float64).reshape(48)
    metrics = geometry_metrics(family)
    if not metrics["detJ_positive"]:
        raise ValueError(f"{family}: Macro16 detJ is not positive")
    existing_base_npz = sample_root / "base_frames.npz"
    existing_q_path = sample_root / "q48_final.npy"
    if bool(args.skip_existing) and existing_base_npz.exists():
        if not existing_q_path.exists():
            raise RuntimeError(
                f"{case_id}: --skip-existing found an existing base export but no q48_final.npy; "
                "refusing to reuse unknown displacement data"
            )
        existing_q = np.asarray(np.load(str(existing_q_path)), dtype=np.float64).reshape(48)
        if not np.allclose(existing_q, q_final, rtol=1.0e-12, atol=1.0e-12):
            raise RuntimeError(
                f"{case_id}: --skip-existing would reuse an existing Abaqus export with a different q48_final. "
                "Use a fresh out-root or remove the stale sample directory."
            )
    write_json(sample_root / "geometry_metrics.json", metrics)
    np.save(sample_root / "q48_final.npy", q_final)
    write_json(sample_root / "q48_source.json", dict(q_source["source"]))

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
        q_generation_method=np.asarray(str(q_source["source"]["q_generation_method"]), dtype=object),
        q_source_true176_case_id=np.asarray([int(q_source["source"].get("true176_case_id", -1))], dtype=np.int64),
        q_source_true176_vector_path=np.asarray(str(q_source["source"].get("true176_vector_path", "")), dtype=object),
        q_source_legacy_h_ref=np.asarray([float(q_source["source"].get("legacy_h_ref", np.nan))], dtype=np.float64),
        q_source_target_l_ref=np.asarray([float(q_source["source"].get("target_L_ref", np.nan))], dtype=np.float64),
        q_source_template_amplitude_scale=np.asarray([float(q_source["source"].get("template_amplitude_scale", np.nan))], dtype=np.float64),
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
    boundary_json = {k: v for k, v in base_info.items() if k not in {"nodes", "T_boundary_96x48", "q48_frames", "q_boundary_frames", "X_keep_ref", "X_boundary_ref"}}
    boundary_json["q_source"] = q_source["source"]
    write_json(bc_dir / "boundary_contract.json", boundary_json)
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
        "q_source": q_source["source"],
        "q48_final_norm": float(np.linalg.norm(q_final)),
        "q48_final_abs_max": float(np.max(np.abs(q_final))),
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
    generation_by_family: dict[str, list[dict[str, Any]]] = {}
    for row in summary["generation"]:
        generation_by_family.setdefault(str(row["family"]), []).append(row)
    for family, audit in sorted(summary["postprocess"]["audits"].items()):
        generated = generation_by_family[family]
        gen = generated[0]
        force_agg = audit["force_summary"]["aggregate"]
        contract_agg = audit["data_contract_summary"]["aggregate"]
        rows.append(
            {
                "family": family,
                "geometry_count": len(generated),
                "frame_count": int(force_agg.get("force_rel_count") or 0),
                "detJ_positive": bool(gen["geometry_metrics"]["detJ_positive"]),
                "detJ_min": float(gen["geometry_metrics"]["detJ_min"]),
                "q_generation_methods": sorted({str(item["q_source"]["q_generation_method"]) for item in generated}),
                "template_case_ids": sorted(
                    int(item["q_source"]["true176_case_id"])
                    for item in generated
                    if "true176_case_id" in item["q_source"]
                ),
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


def generation_tasks(families: tuple[str, ...], args: argparse.Namespace) -> list[dict[str, Any]]:
    case_ids = {
        "cylindrical_shell": "case070_cylindrical_shell",
        "conical_shell": "case071_conical_shell",
        "thickness_varying_shell": "case072_thickness_varying_shell",
        "mild_double_curvature_shell": "case073_mild_double_curvature_shell",
    }
    mode = str(args.q_source_mode).strip().lower()
    if mode != "true176-template":
        return [
            {
                "family": family,
                "case_id": case_ids[family],
                "template_case_id": None,
            }
            for family in families
        ]

    template_cases = parse_int_list(str(args.template_cases))
    assignment = str(args.template_assignment).strip().lower()
    tasks: list[dict[str, Any]] = []
    if assignment == "one-per-family":
        for idx, family in enumerate(families):
            case = int(template_cases[idx % len(template_cases)])
            tasks.append(
                {
                    "family": family,
                    "case_id": f"{case_ids[family]}_t176{case:03d}",
                    "template_case_id": case,
                }
            )
    elif assignment == "cross-product":
        for family in families:
            for case in template_cases:
                tasks.append(
                    {
                        "family": family,
                        "case_id": f"{case_ids[family]}_t176{int(case):03d}",
                        "template_case_id": int(case),
                    }
                )
    else:
        raise ValueError(f"unknown template assignment {args.template_assignment!r}")
    return tasks


def run(args: argparse.Namespace) -> dict[str, Any]:
    old = setup_old_imports(Path(args.old_src_root))
    families = GAUSS_FAMILIES if str(args.families).strip().lower() == "all" else tuple(v.strip() for v in str(args.families).split(",") if v.strip())
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    q_context = template_transfer_context(args) if str(args.q_source_mode).strip().lower() == "true176-template" else None
    tasks = generation_tasks(families, args)
    generation: list[dict[str, Any]] = []
    for task in tasks:
        generation.append(
            run_family(
                str(task["family"]),
                str(task["case_id"]),
                args,
                old,
                q_context,
                template_case_id=task.get("template_case_id"),
            )
        )
    post = build_source128_and_audit(generation, args)
    summary = {
        "script": "run_gate04_wind_shell_generality_audit",
        "out_root": str(out_root),
        "families": list(families),
        "generation_task_count": len(tasks),
        "q_source_mode": str(args.q_source_mode),
        "template_assignment": str(args.template_assignment),
        "template_cases": parse_int_list(str(args.template_cases)) if str(args.q_source_mode).strip().lower() == "true176-template" else [],
        "true176_template_context": {k: v for k, v in (q_context or {}).items() if k != "legacy_frames"},
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("runs") / "gate04_wind_shell_generality")
    parser.add_argument("--families", default="all")
    parser.add_argument("--old-src-root", type=Path, default=DEFAULT_OLD_SRC_ROOT)
    parser.add_argument("--export-lib", type=Path, default=DEFAULT_EXPORT_LIB)
    parser.add_argument("--abaqus", type=Path, default=DEFAULT_ABAQUS)
    parser.add_argument("--candidate", default="css8_le6_engineering")
    parser.add_argument("--increments", type=int, default=100)
    parser.add_argument("--q-source-mode", default="true176-template", choices=("true176-template", "synthetic-mixed"))
    parser.add_argument("--true176-root", type=Path, default=DEFAULT_TRUE176_ROOT)
    parser.add_argument("--true176-domain", default=DEFAULT_TRUE176_DOMAIN)
    parser.add_argument("--true176-vector-kind", default=DEFAULT_TRUE176_VECTOR_KIND)
    parser.add_argument("--true176-meta-case", type=int, default=1)
    parser.add_argument("--template-cases", default=DEFAULT_TEMPLATE_CASES)
    parser.add_argument("--template-assignment", default="one-per-family", choices=("one-per-family", "cross-product"))
    parser.add_argument("--template-amplitude-scale", type=float, default=1.0)
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
    return parser.parse_args(argv)


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
