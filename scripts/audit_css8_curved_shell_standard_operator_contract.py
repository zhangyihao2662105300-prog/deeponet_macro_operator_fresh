#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit the CSS8 curved-shell interpretation of the v2/v3 standard operator.

This is a geometry/contract audit only.  It does not train a model and does not
use old TRUE176 labels.  The audit makes several non-ambiguous checks for the
current CSS8 macro route:

* ``ip_xi`` in current compacts is local CSS8 parent ``(r,s,t)``, not the
  4x4 macro coordinate.
* A required macro trunk coordinate can be derived as ``ip_macro_xi`` from the
  4x4 CSS8 row map.
* Each CSS8 subelement admits an exact midsurface + director decomposition:
  ``x(r,s,t) = x_mid(r,s) + 0.5 * t * director(r,s)``.
* The final CSS8 local strain frame should be stack-director based:
  local-3 = normalized ``g_t = dX/dt``.
* The existing v2 local frame is compared against the stack-director frame so
  the contract gap is quantified rather than hand-waved.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_v2_local_strain_pilot_compact import strain_transform_matrices  # noqa: E402
from macro_deeponet.true176_data import (  # noqa: E402
    GAUSS_POINTS,
    css8_elements,
    css8_shape,
    frame_at_params,
    standard_css8_row_map,
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


def read_compact_list(path: Path) -> list[Path]:
    return [
        Path(line.strip()).resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def parse_case_id(path: Path, files: np.lib.npyio.NpzFile | None = None) -> int:
    if files is not None:
        for key in ("v2b_pilot_case_id", "v2a_pilot_case_id", "case_id"):
            if key in files.files:
                arr = np.asarray(files[key])
                if arr.size:
                    return int(arr.reshape(-1)[0])
    match = re.search(r"case[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot parse case id from {path}")
    return int(match.group(1))


def scalar_text(value: Any, default: str = "unknown") -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return str(default)
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def max_abs(value: np.ndarray) -> float:
    vals = np.asarray(value, dtype=np.float64)
    return float(np.max(np.abs(vals))) if vals.size else 0.0


def rel_norm(diff: np.ndarray, ref: np.ndarray) -> float:
    den = max(float(np.linalg.norm(np.asarray(ref, dtype=np.float64).reshape(-1))), 1.0e-30)
    return float(np.linalg.norm(np.asarray(diff, dtype=np.float64).reshape(-1)) / den)


def normalize(vec: np.ndarray, *, eps: float = 1.0e-14) -> np.ndarray:
    vals = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vals))
    if not math.isfinite(norm) or norm <= float(eps):
        raise ValueError(f"degenerate vector norm {norm:g}")
    return vals / norm


def stack_director_frame_from_j(jmat: np.ndarray) -> np.ndarray:
    """Return Q columns [e1,e2,e3] with e3 aligned to CSS8 stack/t direction."""

    j = np.asarray(jmat, dtype=np.float64).reshape(3, 3)
    g_r, g_s, g_t = j[0], j[1], j[2]
    e3 = normalize(g_t)
    g_r_projected = g_r - float(np.dot(g_r, e3)) * e3
    if float(np.linalg.norm(g_r_projected)) <= 1.0e-14:
        g_r_projected = g_s - float(np.dot(g_s, e3)) * e3
    e1 = normalize(g_r_projected)
    e2 = normalize(np.cross(e3, e1))
    if float(np.dot(e2, g_s)) < 0.0:
        e1 = -e1
        e2 = normalize(np.cross(e3, e1))
    return np.stack([e1, e2, e3], axis=1)


def surface_normal_frame_from_j(jmat: np.ndarray) -> np.ndarray:
    """Mirror the v2b Gram-Schmidt frame: e1 from g_r, e2 from g_s, e3 normal."""

    j = np.asarray(jmat, dtype=np.float64).reshape(3, 3)
    g_r, g_s, g_t = j[0], j[1], j[2]
    e1 = normalize(g_r)
    e2_raw = g_s - float(np.dot(g_s, e1)) * e1
    e2 = normalize(e2_raw)
    e3 = normalize(np.cross(e1, e2))
    if float(np.dot(e3, g_t)) < 0.0:
        e2 = -e2
        e3 = normalize(np.cross(e1, e2))
    return np.stack([e1, e2, e3], axis=1)


def bilinear_face_shape(r: float, s: float) -> np.ndarray:
    signs = np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=np.float64)
    out = np.empty(4, dtype=np.float64)
    for idx, (ra, sa) in enumerate(signs):
        out[idx] = 0.25 * (1.0 + ra * float(r)) * (1.0 + sa * float(s))
    return out


def audit_one(path: Path, *, tol: float) -> dict[str, Any]:
    row = standard_css8_row_map()
    local_gauss = np.asarray([GAUSS_POINTS[int(gp)] for gp in row[:, 3].astype(np.int64)], dtype=np.float64)
    ip_macro_xi = row[:, 9:12].astype(np.float64)
    elements = css8_elements()

    with np.load(str(path), allow_pickle=True) as z:
        required = {"ip_xi", "ip_J", "ip_detJ", "X_macro"}
        missing = sorted(required.difference(z.files))
        if missing:
            raise KeyError(f"{path}: missing CSS8 audit fields {missing}")
        case_id = parse_case_id(path, z)
        ip_xi = np.asarray(z["ip_xi"], dtype=np.float64)
        ip_j = np.asarray(z["ip_J"], dtype=np.float64)
        ip_detj = np.asarray(z["ip_detJ"], dtype=np.float64).reshape(128)
        x_macro = np.asarray(z["X_macro"], dtype=np.float64)
        ip_xyz = np.asarray(z["ip_xyz"], dtype=np.float64) if "ip_xyz" in z.files else None
        local_frame_q = np.asarray(z["local_frame_Q"], dtype=np.float64) if "local_frame_Q" in z.files else None
        shape4 = np.asarray(z["shape4"], dtype=np.float64).reshape(-1, 4)[0] if "shape4" in z.files else None
        point_feature_names = [str(v) for v in np.asarray(z["point_feature_names"]).reshape(-1)] if "point_feature_names" in z.files else []
        ip_j_convention = scalar_text(z["ip_J_convention"], "") if "ip_J_convention" in z.files else ""

    if ip_xi.shape != (128, 3):
        raise ValueError(f"{path}: ip_xi must be [128,3], got {ip_xi.shape}")
    if ip_j.shape != (128, 3, 3):
        raise ValueError(f"{path}: ip_J must be [128,3,3], got {ip_j.shape}")
    if x_macro.shape != (50, 3):
        raise ValueError(f"{path}: X_macro must be [50,3], got {x_macro.shape}")
    if local_frame_q is not None and local_frame_q.shape != (128, 3, 3):
        raise ValueError(f"{path}: local_frame_Q must be [128,3,3], got {local_frame_q.shape}")

    ip_xi_local_max_abs = max_abs(ip_xi - local_gauss)
    macro_x_unique = int(np.unique(np.round(ip_macro_xi[:, 0], 12)).size)
    macro_y_unique = int(np.unique(np.round(ip_macro_xi[:, 1], 12)).size)
    macro_z_unique = int(np.unique(np.round(ip_macro_xi[:, 2], 12)).size)
    has_macro_trunk_features = any(
        name in {"xi_fine", "eta_fine", "zeta_fine", "ip_macro_xi", "ip_macro_eta", "ip_macro_zeta"}
        for name in point_feature_names
    )

    det_from_j = np.linalg.det(ip_j)
    stack_frames = np.asarray([stack_director_frame_from_j(jmat) for jmat in ip_j], dtype=np.float64)
    surface_frames = np.asarray([surface_normal_frame_from_j(jmat) for jmat in ip_j], dtype=np.float64)
    stack_qtq = np.einsum("pji,pjk->pik", stack_frames, stack_frames)
    stack_det = np.linalg.det(stack_frames)
    gt_unit = ip_j[:, 2, :] / np.linalg.norm(ip_j[:, 2, :], axis=1, keepdims=True)

    stack_e3_dot_gt = np.einsum("pi,pi->p", stack_frames[:, :, 2], gt_unit)
    surface_e3_dot_gt = np.einsum("pi,pi->p", surface_frames[:, :, 2], gt_unit)
    current_e3_dot_stack = None
    current_q_rel_to_stack = None
    if local_frame_q is not None:
        current_e3_dot_stack_arr = np.einsum("pi,pi->p", local_frame_q[:, :, 2], stack_frames[:, :, 2])
        current_e3_dot_stack = {
            "min": float(np.min(current_e3_dot_stack_arr)),
            "median": float(np.median(current_e3_dot_stack_arr)),
            "max": float(np.max(current_e3_dot_stack_arr)),
        }
        current_q_rel_to_stack = rel_norm(local_frame_q - stack_frames, stack_frames)

    analytic_normal_dot_stack = None
    if shape4 is not None:
        analytic_normals = np.asarray([frame_at_params(shape4, 0.5 * rr[10], 0.5 * rr[9])[2] for rr in row], dtype=np.float64)
        analytic_normals /= np.linalg.norm(analytic_normals, axis=1, keepdims=True)
        vals = np.einsum("pi,pi->p", stack_frames[:, :, 2], analytic_normals)
        analytic_normal_dot_stack = {
            "min": float(np.min(vals)),
            "median": float(np.median(vals)),
            "max": float(np.max(vals)),
        }

    x_iso_errors: list[float] = []
    x_mid_director_errors: list[float] = []
    x_compact_errors: list[float] = []
    thicknesses: list[float] = []
    director_norms: list[float] = []
    for ri, rr in enumerate(row):
        elem = int(rr[1])
        gp = int(rr[3])
        r, s, t = GAUSS_POINTS[gp]
        conn = np.asarray(elements[elem], dtype=np.int64) - 1
        nodes = x_macro[conn]
        nshape, _dndr = css8_shape(r, s, t)
        x_iso = nshape @ nodes
        n2d = bilinear_face_shape(r, s)
        bottom = nodes[:4]
        top = nodes[4:]
        x_mid = n2d @ (0.5 * (bottom + top))
        director = n2d @ (top - bottom)
        x_from_mid = x_mid + 0.5 * float(t) * director
        x_iso_errors.append(max_abs(x_iso - x_from_mid))
        x_mid_director_errors.append(max_abs(x_from_mid - x_iso))
        thicknesses.append(2.0 * float(np.linalg.norm(ip_j[ri, 2])))
        director_norms.append(float(np.linalg.norm(director)))
        if ip_xyz is not None:
            x_compact_errors.append(max_abs(x_iso - ip_xyz[ri]))

    t_from_stack, t_to_stack = strain_transform_matrices(stack_frames)
    roundtrip_abq = np.einsum("pab,pbc->pac", t_to_stack, t_from_stack)
    roundtrip_local = np.einsum("pab,pbc->pac", t_from_stack, t_to_stack)
    eye6 = np.eye(6, dtype=np.float64)
    t_eps_roundtrip_abq_rel = rel_norm(roundtrip_abq - eye6.reshape(1, 6, 6), eye6)
    t_eps_roundtrip_local_rel = rel_norm(roundtrip_local - eye6.reshape(1, 6, 6), eye6)

    failures: list[str] = []
    if ip_xi_local_max_abs >= tol:
        failures.append("ip_xi_does_not_match_local_css8_gauss")
    if max_abs(det_from_j - ip_detj) >= 5.0e-7:
        failures.append("ip_detJ_does_not_match_det_ip_J")
    if max_abs(np.asarray(x_mid_director_errors)) >= 5.0e-10:
        failures.append("midsurface_director_decomposition_not_exact")
    if np.min(stack_e3_dot_gt) < 1.0 - 1.0e-12:
        failures.append("stack_frame_e3_not_aligned_to_g_t")
    if max_abs(stack_qtq - np.eye(3)) >= 5.0e-12:
        failures.append("stack_frame_not_orthonormal")
    if np.min(stack_det) < 1.0 - 5.0e-12:
        failures.append("stack_frame_not_right_handed")
    if t_eps_roundtrip_abq_rel >= 5.0e-12 or t_eps_roundtrip_local_rel >= 5.0e-12:
        failures.append("stack_T_eps_roundtrip_failed")

    return {
        "case_id": int(case_id),
        "compact_path": str(path),
        "strict_css8_geometry_pass": len(failures) == 0,
        "strict_failures": failures,
        "ip_J_convention": ip_j_convention,
        "ip_xi_current_meaning": "local_css8_parent_r_s_t",
        "ip_xi_local_gauss_max_abs": ip_xi_local_max_abs,
        "ip_macro_xi_required_for_trunk": True,
        "ip_macro_xi_currently_stored": False,
        "ip_macro_xi_shape_if_added": [128, 3],
        "ip_macro_xi_unique_counts": {"xi": macro_x_unique, "eta": macro_y_unique, "zeta": macro_z_unique},
        "point_features_include_macro_xi": bool(has_macro_trunk_features),
        "final_recommended_local_frame": "css8_stack_director_frame",
        "existing_v2_local_frame": "surface_normal_gram_schmidt_oriented_by_g_t",
        "stack_frame_orthonormal_max": max_abs(stack_qtq - np.eye(3)),
        "stack_frame_det_min": float(np.min(stack_det)),
        "stack_frame_det_max": float(np.max(stack_det)),
        "stack_e3_dot_g_t_unit_min": float(np.min(stack_e3_dot_gt)),
        "stack_e3_dot_g_t_unit_median": float(np.median(stack_e3_dot_gt)),
        "surface_e3_dot_g_t_unit_min": float(np.min(surface_e3_dot_gt)),
        "surface_e3_dot_g_t_unit_median": float(np.median(surface_e3_dot_gt)),
        "current_e3_dot_stack_e3": current_e3_dot_stack,
        "current_Q_rel_to_stack_Q": current_q_rel_to_stack,
        "analytic_shape4_normal_dot_stack_e3": analytic_normal_dot_stack,
        "ip_detJ_vs_det_ip_J_max_abs": max_abs(det_from_j - ip_detj),
        "detJ_min": float(np.min(det_from_j)),
        "detJ_abs_min": float(np.min(np.abs(det_from_j))),
        "midsurface_director_identity_max_abs": max_abs(np.asarray(x_mid_director_errors)),
        "x_iso_vs_compact_ip_xyz_max_abs": max_abs(np.asarray(x_compact_errors)) if x_compact_errors else None,
        "thickness_from_2_norm_g_t_min": float(np.min(thicknesses)),
        "thickness_from_2_norm_g_t_max": float(np.max(thicknesses)),
        "director_norm_min": float(np.min(director_norms)),
        "director_norm_max": float(np.max(director_norms)),
        "T_eps_stack_roundtrip_abq_rel": t_eps_roundtrip_abq_rel,
        "T_eps_stack_roundtrip_local_rel": t_eps_roundtrip_local_rel,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tol", type=float, default=1.0e-7)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    paths = read_compact_list(args.compact_list)
    cases = [audit_one(path, tol=float(args.tol)) for path in paths]
    pass_count = sum(1 for row in cases if row["strict_css8_geometry_pass"])
    q_rel_vals = [row["current_Q_rel_to_stack_Q"] for row in cases if row["current_Q_rel_to_stack_Q"] is not None]
    summary = {
        "audit_name": "css8_curved_shell_standard_operator_contract",
        "compact_list": str(args.compact_list),
        "compact_count": len(cases),
        "strict_css8_geometry_pass_count": int(pass_count),
        "strict_css8_geometry_fail_count": int(len(cases) - pass_count),
        "all_strict_css8_geometry_pass": pass_count == len(cases),
        "contract_decisions": {
            "macro_shape": "piecewise_4x4_css8_parent_domain_not_single_hex8",
            "current_ip_xi_role": "local_css8_parent_r_s_t",
            "required_trunk_coordinate": "ip_macro_xi_or_equivalent_subcell_id_plus_local_r_s_t",
            "final_local_frame": "stack_director_frame_with_local_3_parallel_to_g_t",
            "network_output_coordinate": "LE_local_in_css8_stack_director_frame",
            "B_postprocess": "B_raw_hat = T_eps_to_abq(stack_frame) @ B_local_useful_hat @ T_q_raw_to_useful",
        },
        "current_contract_gap": {
            "ip_macro_xi_not_stored_in_current_v2_standard_compacts": True,
            "current_v2_local_frame_is_surface_normal_not_exact_stack_director": True,
            "gap_is_small_for_current_shape4_pool_but_should_not_define_final_contract": True,
            "requires_v3_css8_standard_operator_compact_rebuild": True,
        },
        "current_Q_rel_to_stack_Q_min": min(q_rel_vals) if q_rel_vals else None,
        "current_Q_rel_to_stack_Q_max": max(q_rel_vals) if q_rel_vals else None,
        "cases": cases,
        "trained_model": False,
        "used_old_true176_labels": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))
    if args.strict and pass_count != len(cases):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
