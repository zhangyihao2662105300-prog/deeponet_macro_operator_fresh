#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build v3 CSS8 standard-operator compacts from fresh/v2b Abaqus compacts.

This is a preprocessing utility only.  It does not train a model.

The v3 model-visible contract is:

    (q_useful_hat, geometry_global_hat, ip_macro_xi, local_geometry_features_hat)
        -> LE_local_stack

where q_useful_hat is dimensionless and rigid-motion-free, and LE_local_stack is
in the CSS8 stack-director frame.  Raw Abaqus LE/B are kept only for audit and
postprocessing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from v3_css8_standard_operator_common import (
    AUDIT_POSTPROCESS_FIELDS,
    CONTRACT_VERSION,
    MODEL_VISIBLE_FIELDS,
    apply_t_eps_to_b,
    apply_t_eps_to_strain,
    apply_t_q,
    characteristic_length,
    collect_paths,
    css8_standard_coordinates,
    first_key,
    json_default,
    local_geometry_features,
    parse_case_id,
    project_raw_b_to_useful_subspace,
    rel_norm,
    right_multiply_t_q_transpose,
    scalar_bool,
    scalar_text,
    stack_frames_from_ip_j,
    strain_transform_matrices,
    useful_to_raw_projected,
    write_json,
)


def as_float64(z: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    return np.asarray(z[key], dtype=np.float64)


def build_geometry_global_hat(
    *,
    x_macro_hat: np.ndarray,
    span_hat: np.ndarray,
    shape4: np.ndarray | None,
) -> tuple[np.ndarray, list[str]]:
    parts = [np.asarray(x_macro_hat, dtype=np.float64).reshape(-1)]
    names = [f"X_macro_hat_{idx}_{axis}" for idx in range(x_macro_hat.shape[0]) for axis in ("x", "y", "z")]
    parts.append(np.asarray(span_hat, dtype=np.float64).reshape(3))
    names.extend(["span_hat_x", "span_hat_y", "span_hat_z"])
    if shape4 is not None:
        parts.append(np.asarray(shape4, dtype=np.float64).reshape(-1))
        names.extend([f"shape4_{idx}" for idx in range(np.asarray(shape4).reshape(-1).shape[0])])
    return np.concatenate(parts, axis=0), names


def audit_arrays(
    *,
    q48_raw: np.ndarray,
    q_useful_hat: np.ndarray,
    le_abq: np.ndarray,
    le_local_stack: np.ndarray,
    b_raw: np.ndarray,
    b_local_useful_stack_hat: np.ndarray,
    t_q: np.ndarray,
    t_q_hat: np.ndarray,
    t_eps_to_abq_stack: np.ndarray,
    l_ref: float,
    ip_j: np.ndarray,
    ip_j_hat: np.ndarray,
    ip_invj: np.ndarray,
    ip_invj_hat: np.ndarray,
    ip_detj: np.ndarray,
    ip_detj_hat: np.ndarray,
    q_stack: np.ndarray,
    strict: bool,
    tol: float,
) -> dict[str, Any]:
    failures: list[str] = []

    q_useful_hat_check = apply_t_q(q48_raw, t_q_hat)
    q_diff = q_useful_hat_check - q_useful_hat
    le_abq_recovered = apply_t_eps_to_strain(t_eps_to_abq_stack, le_local_stack)
    le_diff = le_abq_recovered - le_abq

    b_raw_hat_projected = useful_to_raw_projected(b_local_useful_stack_hat, t_eps_to_abq_stack, t_q_hat)
    b_raw_projected = project_raw_b_to_useful_subspace(b_raw, t_q)
    b_projected_diff = b_raw_hat_projected - b_raw_projected
    b_raw_diff = b_raw_hat_projected - b_raw

    qtq = np.einsum("pji,pjk->pik", q_stack, q_stack)
    q_det = np.linalg.det(q_stack)
    gt_unit = ip_j[:, 2, :] / np.linalg.norm(ip_j[:, 2, :], axis=1, keepdims=True)
    stack_e3_dot_gt = np.einsum("pi,pi->p", q_stack[:, :, 2], gt_unit)

    ip_j_hat_check = ip_j / float(l_ref)
    ip_invj_hat_check = ip_invj * float(l_ref)
    ip_detj_hat_check = ip_detj / (float(l_ref) ** 3)
    det_from_j = np.linalg.det(ip_j)

    q_rel = rel_norm(q_diff, q_useful_hat)
    le_rel = rel_norm(le_diff, le_abq)
    b_rel = rel_norm(b_projected_diff, b_raw_projected)
    j_hat_rel = rel_norm(ip_j_hat - ip_j_hat_check, ip_j_hat_check)
    invj_hat_rel = rel_norm(ip_invj_hat - ip_invj_hat_check, ip_invj_hat_check)
    detj_hat_rel = rel_norm(ip_detj_hat - ip_detj_hat_check, ip_detj_hat_check)
    detj_rel = rel_norm(det_from_j - ip_detj, ip_detj)

    if q_useful_hat.shape != (q48_raw.shape[0], 42):
        failures.append(f"q_useful_hat_shape_{q_useful_hat.shape}")
    if le_local_stack.shape != (q48_raw.shape[0], ip_j.shape[0], 6):
        failures.append(f"LE_local_stack_shape_{le_local_stack.shape}")
    if b_local_useful_stack_hat.shape != (q48_raw.shape[0], ip_j.shape[0], 6, 42):
        failures.append(f"B_local_useful_stack_hat_shape_{b_local_useful_stack_hat.shape}")
    if strict:
        if q_rel >= tol:
            failures.append(f"q_useful_hat_transform_rel_ge_{tol:g}")
        if le_rel >= tol:
            failures.append(f"LE_local_stack_to_abq_roundtrip_rel_ge_{tol:g}")
        if b_rel >= tol:
            failures.append(f"B_raw_projected_rel_ge_{tol:g}")
        if j_hat_rel >= tol:
            failures.append(f"ip_J_hat_scaling_rel_ge_{tol:g}")
        if invj_hat_rel >= tol:
            failures.append(f"ip_invJ_hat_scaling_rel_ge_{tol:g}")
        if detj_hat_rel >= tol:
            failures.append(f"ip_detJ_hat_scaling_rel_ge_{tol:g}")
        if detj_rel >= 1.0e-6:
            failures.append("ip_detJ_vs_det_ip_J_rel_ge_1e-6")
        if np.max(np.abs(qtq - np.eye(3))) >= 5.0e-12:
            failures.append("Q_stack_not_orthonormal")
        if np.min(q_det) < 1.0 - 5.0e-12:
            failures.append("Q_stack_not_right_handed")
        if np.min(stack_e3_dot_gt) < 1.0 - 1.0e-12:
            failures.append("Q_stack_e3_not_aligned_to_g_t")

    return {
        "strict_pass": len(failures) == 0,
        "strict_failures": failures,
        "q_useful_hat_transform_rel": q_rel,
        "q_useful_hat_transform_max_abs": float(np.max(np.abs(q_diff))) if q_diff.size else 0.0,
        "LE_local_stack_to_abq_roundtrip_rel": le_rel,
        "LE_local_stack_to_abq_roundtrip_max_abs": float(np.max(np.abs(le_diff))) if le_diff.size else 0.0,
        "B_raw_projected_rel": b_rel,
        "B_raw_projected_max_abs": float(np.max(np.abs(b_projected_diff))) if b_projected_diff.size else 0.0,
        "B_rigid_residual_rel": rel_norm(b_raw_diff, b_raw),
        "B_rigid_residual_max_abs": float(np.max(np.abs(b_raw_diff))) if b_raw_diff.size else 0.0,
        "ip_J_hat_scaling_rel": j_hat_rel,
        "ip_invJ_hat_scaling_rel": invj_hat_rel,
        "ip_detJ_hat_scaling_rel": detj_hat_rel,
        "ip_detJ_vs_det_ip_J_rel": detj_rel,
        "Q_stack_orthonormal_max": float(np.max(np.abs(qtq - np.eye(3)))),
        "Q_stack_det_min": float(np.min(q_det)),
        "Q_stack_det_max": float(np.max(q_det)),
        "Q_stack_e3_dot_g_t_min": float(np.min(stack_e3_dot_gt)),
        "Q_stack_e3_dot_g_t_median": float(np.median(stack_e3_dot_gt)),
    }


def build_one(path: Path, out_root: Path, *, strict: bool, tol: float, nx: int, ny: int) -> dict[str, Any]:
    with np.load(str(path), allow_pickle=True) as z:
        files = list(z.files)
        required = {
            "q48_raw",
            "B_LE128_forward",
            "T_q_raw_to_useful",
            "X_macro",
            "ip_J",
            "ip_invJ",
            "ip_detJ",
            "ip_xyz",
        }
        missing = sorted(required.difference(files))
        if missing:
            raise KeyError(f"{path}: missing required fields {missing}")
        if "uses_old_true176_labels_as_v2_labels" in files and scalar_bool(z["uses_old_true176_labels_as_v2_labels"]):
            raise ValueError(f"{path}: old TRUE176 labels are marked as v2/v3 labels")

        le_key = first_key(files, ("LE_abq", "LE128_base", "le"))
        if le_key is None:
            raise KeyError(f"{path}: missing LE_abq/LE128_base/le")

        case_id = parse_case_id(path, z)
        case_name = f"case{case_id:03d}"
        q48_raw = as_float64(z, "q48_raw")
        le_abq = as_float64(z, le_key)
        b_raw = as_float64(z, "B_LE128_forward")
        t_q = as_float64(z, "T_q_raw_to_useful")
        x_macro = as_float64(z, "X_macro")
        ip_j = as_float64(z, "ip_J")
        ip_invj = as_float64(z, "ip_invJ")
        ip_detj = as_float64(z, "ip_detJ").reshape(-1)
        ip_xyz = as_float64(z, "ip_xyz")
        shape4 = as_float64(z, "shape4")[0] if "shape4" in files else None

    if q48_raw.ndim != 2 or q48_raw.shape[1] != 48:
        raise ValueError(f"{path}: q48_raw must be [N,48], got {q48_raw.shape}")
    n_frames = int(q48_raw.shape[0])
    row_map, ip_macro_xi, ip_local_rst = css8_standard_coordinates(nx=nx, ny=ny)
    point_count = int(ip_macro_xi.shape[0])
    if point_count != ip_j.shape[0]:
        raise ValueError(f"{path}: row-map point count {point_count} does not match ip_J {ip_j.shape}")
    if le_abq.shape != (n_frames, point_count, 6):
        raise ValueError(f"{path}: LE must be [N,{point_count},6], got {le_abq.shape}")
    if b_raw.shape != (n_frames, point_count, 6, 48):
        raise ValueError(f"{path}: B_LE128_forward must be [N,{point_count},6,48], got {b_raw.shape}")

    l_ref, x_center, x_span = characteristic_length(x_macro)
    x_macro_hat = (x_macro - x_center.reshape(1, 3)) / float(l_ref)
    ip_xyz_hat = (ip_xyz - x_center.reshape(1, 3)) / float(l_ref)
    ip_j_hat = ip_j / float(l_ref)
    ip_invj_hat = ip_invj * float(l_ref)
    ip_detj_hat = ip_detj / (float(l_ref) ** 3)
    q_stack = stack_frames_from_ip_j(ip_j)
    t_eps_from_abq_stack, t_eps_to_abq_stack = strain_transform_matrices(q_stack)

    q_useful = apply_t_q(q48_raw, t_q)
    q_useful_hat = q_useful / float(l_ref)
    t_q_hat = t_q / float(l_ref)
    t_q_useful_hat_to_raw_projected = t_q * float(l_ref)

    le_local_stack = apply_t_eps_to_strain(t_eps_from_abq_stack, le_abq)
    b_local_raw = apply_t_eps_to_b(t_eps_from_abq_stack, b_raw)
    b_local_useful_stack_hat = float(l_ref) * right_multiply_t_q_transpose(b_local_raw, t_q)

    geometry_global_hat, geometry_global_names = build_geometry_global_hat(
        x_macro_hat=x_macro_hat,
        span_hat=x_span / float(l_ref),
        shape4=shape4,
    )
    local_geom, local_geom_names, trunk, trunk_names = local_geometry_features(
        ip_macro_xi=ip_macro_xi,
        ip_local_rst=ip_local_rst,
        ip_xyz_hat=ip_xyz_hat,
        q_stack=q_stack,
        ip_j_hat=ip_j_hat,
        ip_invj_hat=ip_invj_hat,
        ip_detj_hat=ip_detj_hat,
    )

    audit = audit_arrays(
        q48_raw=q48_raw,
        q_useful_hat=q_useful_hat,
        le_abq=le_abq,
        le_local_stack=le_local_stack,
        b_raw=b_raw,
        b_local_useful_stack_hat=b_local_useful_stack_hat,
        t_q=t_q,
        t_q_hat=t_q_hat,
        t_eps_to_abq_stack=t_eps_to_abq_stack,
        l_ref=float(l_ref),
        ip_j=ip_j,
        ip_j_hat=ip_j_hat,
        ip_invj=ip_invj,
        ip_invj_hat=ip_invj_hat,
        ip_detj=ip_detj,
        ip_detj_hat=ip_detj_hat,
        q_stack=q_stack,
        strict=strict,
        tol=tol,
    )

    metadata = {
        "standard_operator_contract_version": CONTRACT_VERSION,
        "macro_shape": f"piecewise_{nx}x{ny}_css8_parent_domain_not_single_hex8",
        "model_operator": "(q_useful_hat, geometry_global_hat, trunk_features_hat) -> LE_local_stack",
        "branch_input": "q_useful_hat = (T_q_raw_to_useful @ q48_raw) / L_ref",
        "geometry_input": "geometry_global_hat plus local_geometry_features_hat",
        "trunk_input": "ip_macro_xi + ip_local_rst + local_geometry_features_hat",
        "output": "LE_local_stack in Q_stack(point)",
        "ad_target": "dLE_local_stack/dq_useful_hat",
        "raw_backprojection": "B_raw_hat = T_eps_to_abq_stack @ B_local_useful_stack_hat @ (T_q_raw_to_useful / L_ref)",
        "preprocess_B": "B_local_useful_stack_hat = L_ref * T_eps_from_abq_stack @ B_raw @ T_q_raw_to_useful.T",
        "L_ref_definition": "max axis-aligned span of X_macro",
        "Q_stack_definition": "columns [e1,e2,e3], e3=normalize(dX/dt)",
        "model_visible_fields": list(MODEL_VISIBLE_FIELDS),
        "audit_postprocess_only_fields": list(AUDIT_POSTPROCESS_FIELDS),
        "trained_model": False,
        "used_old_true176_labels": False,
    }

    case_dir = out_root / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    out_npz = case_dir / f"{case_name}_v3_css8_standard_operator.npz"
    out_summary = case_dir / f"{case_name}_v3_css8_standard_operator.summary.json"

    arrays: dict[str, Any] = {
        "q_useful_hat": q_useful_hat.astype(np.float64),
        "geometry_global_hat": geometry_global_hat.astype(np.float64),
        "geometry_global_feature_names": np.asarray(geometry_global_names, dtype=object),
        "ip_macro_xi": ip_macro_xi.astype(np.float32),
        "ip_local_rst": ip_local_rst.astype(np.float32),
        "local_geometry_features_hat": local_geom.astype(np.float64),
        "local_geometry_feature_names": np.asarray(local_geom_names, dtype=object),
        "trunk_features_hat": trunk.astype(np.float64),
        "trunk_feature_names": np.asarray(trunk_names, dtype=object),
        "LE_local_stack": le_local_stack.astype(np.float64),
        "B_local_useful_stack_hat": b_local_useful_stack_hat.astype(np.float64),
        "q48_raw": q48_raw.astype(np.float64),
        "q_useful": q_useful.astype(np.float64),
        "LE_abq": le_abq.astype(np.float64),
        "B_LE128_forward": b_raw.astype(np.float64),
        "T_q_raw_to_useful": t_q.astype(np.float64),
        "T_q_raw_to_useful_hat": t_q_hat.astype(np.float64),
        "T_q_useful_hat_to_raw_projected": t_q_useful_hat_to_raw_projected.astype(np.float64),
        "Q_stack": q_stack.astype(np.float64),
        "T_eps_to_abq_stack": t_eps_to_abq_stack.astype(np.float64),
        "T_eps_from_abq_stack": t_eps_from_abq_stack.astype(np.float64),
        "X_macro": x_macro.astype(np.float64),
        "X_macro_hat": x_macro_hat.astype(np.float64),
        "X_center": x_center.astype(np.float64),
        "X_span": x_span.astype(np.float64),
        "L_ref": np.asarray(float(l_ref), dtype=np.float64),
        "ip_J": ip_j.astype(np.float64),
        "ip_J_hat": ip_j_hat.astype(np.float64),
        "ip_invJ": ip_invj.astype(np.float64),
        "ip_invJ_hat": ip_invj_hat.astype(np.float64),
        "ip_detJ": ip_detj.astype(np.float64),
        "ip_detJ_hat": ip_detj_hat.astype(np.float64),
        "ip_xyz": ip_xyz.astype(np.float64),
        "ip_xyz_hat": ip_xyz_hat.astype(np.float64),
        "css8_row_map": row_map.astype(np.float64),
        "case_id": np.asarray(case_id, dtype=np.int64),
        "source_compact": np.asarray(str(path), dtype=object),
        "source_LE_abq_key": np.asarray(le_key, dtype=object),
        "metadata": np.asarray(json.dumps(metadata, ensure_ascii=False, default=json_default), dtype=object),
        "standard_operator_contract_version": np.asarray(CONTRACT_VERSION, dtype=object),
        "macro_shape": np.asarray(metadata["macro_shape"], dtype=object),
        "model_operator": np.asarray(metadata["model_operator"], dtype=object),
        "branch_input": np.asarray(metadata["branch_input"], dtype=object),
        "geometry_input": np.asarray(metadata["geometry_input"], dtype=object),
        "trunk_input": np.asarray(metadata["trunk_input"], dtype=object),
        "model_output_quantity": np.asarray("LE_local_stack", dtype=object),
        "model_derivative": np.asarray("dLE_local_stack/dq_useful_hat", dtype=object),
        "raw_backprojection": np.asarray(metadata["raw_backprojection"], dtype=object),
        "preprocess_B_formula": np.asarray(metadata["preprocess_B"], dtype=object),
        "L_ref_definition": np.asarray(metadata["L_ref_definition"], dtype=object),
        "Q_stack_definition": np.asarray(metadata["Q_stack_definition"], dtype=object),
        "model_visible_fields": np.asarray(MODEL_VISIBLE_FIELDS, dtype=object),
        "audit_postprocess_only_fields": np.asarray(AUDIT_POSTPROCESS_FIELDS, dtype=object),
        "trained_model": np.asarray(False),
        "uses_old_true176_labels_as_v3_labels": np.asarray(False),
    }
    if shape4 is not None:
        arrays["shape4"] = shape4.astype(np.float32)

    np.savez_compressed(out_npz, **arrays)

    summary = {
        "case_id": int(case_id),
        "case_name": case_name,
        "source_compact": str(path),
        "v3_standard_compact": str(out_npz),
        "summary_path": str(out_summary),
        "standard_operator_contract_version": CONTRACT_VERSION,
        "macro_shape": metadata["macro_shape"],
        "frame_count": int(n_frames),
        "point_count": int(point_count),
        "L_ref": float(l_ref),
        "X_span": x_span.tolist(),
        "q_useful_hat_shape": list(q_useful_hat.shape),
        "geometry_global_hat_shape": list(geometry_global_hat.shape),
        "trunk_features_hat_shape": list(trunk.shape),
        "LE_local_stack_shape": list(le_local_stack.shape),
        "B_local_useful_stack_hat_shape": list(b_local_useful_stack_hat.shape),
        "model_visible_fields": list(MODEL_VISIBLE_FIELDS),
        "audit_postprocess_only_fields": list(AUDIT_POSTPROCESS_FIELDS),
        "strict_requested": bool(strict),
        **audit,
        "trained_model": False,
        "used_old_true176_labels": False,
    }
    write_json(out_summary, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="append", type=Path, default=[])
    parser.add_argument("--compact-list", action="append", type=Path, default=[])
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--tol", type=float, default=1.0e-10)
    args = parser.parse_args()

    paths = collect_paths(args.compact, args.compact_list, case_limit=args.case_limit)
    if not paths:
        raise SystemExit("no compact inputs provided")
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows = [build_one(path, args.out_root, strict=args.strict, tol=float(args.tol), nx=args.nx, ny=args.ny) for path in paths]
    list_path = args.out_root / "v3_css8_standard_operator_compact_list.txt"
    list_path.write_text("\n".join(str(row["v3_standard_compact"]) for row in rows) + "\n", encoding="utf-8")

    pass_count = sum(1 for row in rows if row["strict_pass"])
    summary = {
        "standard_operator_contract_version": CONTRACT_VERSION,
        "macro_shape": f"piecewise_{args.nx}x{args.ny}_css8_parent_domain_not_single_hex8",
        "compact_count": len(rows),
        "strict_requested": bool(args.strict),
        "strict_pass_count": int(pass_count),
        "strict_fail_count": int(len(rows) - pass_count),
        "v3_standard_operator_compact_list": str(list_path),
        "model_visible_fields": list(MODEL_VISIBLE_FIELDS),
        "audit_postprocess_only_fields": list(AUDIT_POSTPROCESS_FIELDS),
        "trained_model": False,
        "used_old_true176_labels": False,
        "cases": rows,
    }
    summary_path = args.out_root / "v3_css8_standard_operator_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({
        "v3_css8_standard_operator_summary": str(summary_path),
        "v3_standard_operator_compact_list": str(list_path),
        "compact_count": len(rows),
        "strict_pass_count": int(pass_count),
        "strict_fail_count": int(len(rows) - pass_count),
    }, indent=2, ensure_ascii=False, default=json_default))
    if args.strict and pass_count != len(rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
