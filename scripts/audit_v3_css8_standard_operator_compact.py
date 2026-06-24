#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit v3 CSS8 standard-operator compacts.

This audit is read-only.  It verifies the preprocessing/postprocessing contract
before any v3 training is allowed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from v3_css8_standard_operator_common import (
    CONTRACT_VERSION,
    MODEL_VISIBLE_FIELDS,
    apply_t_eps_to_strain,
    apply_t_q,
    collect_paths,
    json_default,
    project_raw_b_to_useful_subspace,
    rel_norm,
    scalar_bool,
    scalar_text,
    useful_to_raw_projected,
    write_json,
)


def max_abs(value: np.ndarray) -> float:
    vals = np.asarray(value, dtype=np.float64)
    return float(np.max(np.abs(vals))) if vals.size else 0.0


def shape_list(value: np.ndarray | None) -> list[int] | None:
    if value is None:
        return None
    return [int(v) for v in value.shape]


def load_required(z: np.lib.npyio.NpzFile, key: str, failures: list[str]) -> np.ndarray | None:
    if key not in z.files:
        failures.append(f"missing_{key}")
        return None
    return np.asarray(z[key])


def audit_one(path: Path, *, strict: bool, tol: float) -> dict[str, Any]:
    failures: list[str] = []
    if not path.exists():
        return {"compact_path": str(path), "strict_pass": False, "strict_failures": ["compact_missing"]}

    with np.load(str(path), allow_pickle=True) as z:
        keys = set(z.files)
        for key in MODEL_VISIBLE_FIELDS:
            if key not in keys:
                failures.append(f"missing_model_visible_{key}")

        q_useful_hat = load_required(z, "q_useful_hat", failures)
        geometry_global_hat = load_required(z, "geometry_global_hat", failures)
        ip_macro_xi = load_required(z, "ip_macro_xi", failures)
        ip_local_rst = load_required(z, "ip_local_rst", failures)
        trunk_features_hat = load_required(z, "trunk_features_hat", failures)
        le_local_stack = load_required(z, "LE_local_stack", failures)
        b_local_useful_stack_hat = load_required(z, "B_local_useful_stack_hat", failures)
        q48_raw = load_required(z, "q48_raw", failures)
        le_abq = load_required(z, "LE_abq", failures)
        b_raw = load_required(z, "B_LE128_forward", failures)
        t_q = load_required(z, "T_q_raw_to_useful", failures)
        t_q_hat = load_required(z, "T_q_raw_to_useful_hat", failures)
        q_stack = load_required(z, "Q_stack", failures)
        t_eps_to_abq_stack = load_required(z, "T_eps_to_abq_stack", failures)
        t_eps_from_abq_stack = load_required(z, "T_eps_from_abq_stack", failures)
        ip_j = load_required(z, "ip_J", failures)
        ip_j_hat = load_required(z, "ip_J_hat", failures)
        ip_invj = load_required(z, "ip_invJ", failures)
        ip_invj_hat = load_required(z, "ip_invJ_hat", failures)
        ip_detj = load_required(z, "ip_detJ", failures)
        ip_detj_hat = load_required(z, "ip_detJ_hat", failures)
        l_ref_arr = load_required(z, "L_ref", failures)

        if failures:
            return {
                "compact_path": str(path),
                "strict_requested": bool(strict),
                "strict_pass": False,
                "strict_failures": failures,
            }

        assert q_useful_hat is not None
        assert geometry_global_hat is not None
        assert ip_macro_xi is not None
        assert ip_local_rst is not None
        assert trunk_features_hat is not None
        assert le_local_stack is not None
        assert b_local_useful_stack_hat is not None
        assert q48_raw is not None
        assert le_abq is not None
        assert b_raw is not None
        assert t_q is not None
        assert t_q_hat is not None
        assert q_stack is not None
        assert t_eps_to_abq_stack is not None
        assert t_eps_from_abq_stack is not None
        assert ip_j is not None
        assert ip_j_hat is not None
        assert ip_invj is not None
        assert ip_invj_hat is not None
        assert ip_detj is not None
        assert ip_detj_hat is not None
        assert l_ref_arr is not None

        q_useful_hat = np.asarray(q_useful_hat, dtype=np.float64)
        q48_raw = np.asarray(q48_raw, dtype=np.float64)
        le_abq = np.asarray(le_abq, dtype=np.float64)
        b_raw = np.asarray(b_raw, dtype=np.float64)
        le_local_stack = np.asarray(le_local_stack, dtype=np.float64)
        b_local_useful_stack_hat = np.asarray(b_local_useful_stack_hat, dtype=np.float64)
        t_q = np.asarray(t_q, dtype=np.float64)
        t_q_hat = np.asarray(t_q_hat, dtype=np.float64)
        q_stack = np.asarray(q_stack, dtype=np.float64)
        t_eps_to_abq_stack = np.asarray(t_eps_to_abq_stack, dtype=np.float64)
        t_eps_from_abq_stack = np.asarray(t_eps_from_abq_stack, dtype=np.float64)
        ip_j = np.asarray(ip_j, dtype=np.float64)
        ip_j_hat = np.asarray(ip_j_hat, dtype=np.float64)
        ip_invj = np.asarray(ip_invj, dtype=np.float64)
        ip_invj_hat = np.asarray(ip_invj_hat, dtype=np.float64)
        ip_detj = np.asarray(ip_detj, dtype=np.float64).reshape(-1)
        ip_detj_hat = np.asarray(ip_detj_hat, dtype=np.float64).reshape(-1)
        l_ref = float(np.asarray(l_ref_arr, dtype=np.float64).reshape(-1)[0])

        frame_count = int(q48_raw.shape[0]) if q48_raw.ndim == 2 else None
        point_count = int(ip_macro_xi.shape[0]) if ip_macro_xi.ndim == 2 else None
        shape_failures: list[str] = []
        if q48_raw.shape != (frame_count, 48):
            shape_failures.append(f"q48_raw_shape_{q48_raw.shape}")
        if q_useful_hat.shape != (frame_count, 42):
            shape_failures.append(f"q_useful_hat_shape_{q_useful_hat.shape}")
        if ip_macro_xi.shape != (point_count, 3):
            shape_failures.append(f"ip_macro_xi_shape_{ip_macro_xi.shape}")
        if ip_local_rst.shape != (point_count, 3):
            shape_failures.append(f"ip_local_rst_shape_{ip_local_rst.shape}")
        if le_abq.shape != (frame_count, point_count, 6):
            shape_failures.append(f"LE_abq_shape_{le_abq.shape}")
        if le_local_stack.shape != (frame_count, point_count, 6):
            shape_failures.append(f"LE_local_stack_shape_{le_local_stack.shape}")
        if b_raw.shape != (frame_count, point_count, 6, 48):
            shape_failures.append(f"B_LE128_forward_shape_{b_raw.shape}")
        if b_local_useful_stack_hat.shape != (frame_count, point_count, 6, 42):
            shape_failures.append(f"B_local_useful_stack_hat_shape_{b_local_useful_stack_hat.shape}")
        if q_stack.shape != (point_count, 3, 3):
            shape_failures.append(f"Q_stack_shape_{q_stack.shape}")
        if t_eps_to_abq_stack.shape != (point_count, 6, 6):
            shape_failures.append(f"T_eps_to_abq_stack_shape_{t_eps_to_abq_stack.shape}")
        if trunk_features_hat.ndim != 2 or trunk_features_hat.shape[0] != point_count:
            shape_failures.append(f"trunk_features_hat_shape_{trunk_features_hat.shape}")
        failures.extend(shape_failures)

        q_check = apply_t_q(q48_raw, t_q_hat)
        q_diff = q_check - q_useful_hat
        le_abq_recovered = apply_t_eps_to_strain(t_eps_to_abq_stack, le_local_stack)
        le_diff = le_abq_recovered - le_abq
        b_raw_hat_projected = useful_to_raw_projected(b_local_useful_stack_hat, t_eps_to_abq_stack, t_q_hat)
        b_raw_projected = project_raw_b_to_useful_subspace(b_raw, t_q)
        b_projected_diff = b_raw_hat_projected - b_raw_projected
        b_raw_diff = b_raw_hat_projected - b_raw

        eye6 = np.eye(6)
        t_roundtrip_abq = np.einsum("pab,pbc->pac", t_eps_to_abq_stack, t_eps_from_abq_stack)
        t_roundtrip_local = np.einsum("pab,pbc->pac", t_eps_from_abq_stack, t_eps_to_abq_stack)
        qtq = np.einsum("pji,pjk->pik", q_stack, q_stack)
        q_det = np.linalg.det(q_stack)
        gt_unit = ip_j[:, 2, :] / np.linalg.norm(ip_j[:, 2, :], axis=1, keepdims=True)
        stack_e3_dot_gt = np.einsum("pi,pi->p", q_stack[:, :, 2], gt_unit)
        det_from_j = np.linalg.det(ip_j)

        metrics = {
            "q_useful_hat_transform_rel": rel_norm(q_diff, q_useful_hat),
            "q_useful_hat_transform_max_abs": max_abs(q_diff),
            "LE_local_stack_to_abq_roundtrip_rel": rel_norm(le_diff, le_abq),
            "LE_local_stack_to_abq_roundtrip_max_abs": max_abs(le_diff),
            "B_raw_projected_rel": rel_norm(b_projected_diff, b_raw_projected),
            "B_raw_projected_max_abs": max_abs(b_projected_diff),
            "B_rigid_residual_rel": rel_norm(b_raw_diff, b_raw),
            "B_rigid_residual_max_abs": max_abs(b_raw_diff),
            "T_eps_stack_roundtrip_abq_rel": rel_norm(t_roundtrip_abq - eye6.reshape(1, 6, 6), eye6),
            "T_eps_stack_roundtrip_local_rel": rel_norm(t_roundtrip_local - eye6.reshape(1, 6, 6), eye6),
            "Q_stack_orthonormal_max": max_abs(qtq - np.eye(3)),
            "Q_stack_det_min": float(np.min(q_det)),
            "Q_stack_det_max": float(np.max(q_det)),
            "Q_stack_e3_dot_g_t_min": float(np.min(stack_e3_dot_gt)),
            "ip_J_hat_scaling_rel": rel_norm(ip_j_hat - ip_j / l_ref, ip_j / l_ref),
            "ip_invJ_hat_scaling_rel": rel_norm(ip_invj_hat - ip_invj * l_ref, ip_invj * l_ref),
            "ip_detJ_hat_scaling_rel": rel_norm(ip_detj_hat - ip_detj / (l_ref**3), ip_detj / (l_ref**3)),
            "ip_detJ_vs_det_ip_J_rel": rel_norm(det_from_j - ip_detj, ip_detj),
        }

        if strict:
            strict_rules = {
                "q_useful_hat_transform_rel": tol,
                "LE_local_stack_to_abq_roundtrip_rel": tol,
                "B_raw_projected_rel": tol,
                "T_eps_stack_roundtrip_abq_rel": tol,
                "T_eps_stack_roundtrip_local_rel": tol,
                "ip_J_hat_scaling_rel": tol,
                "ip_invJ_hat_scaling_rel": tol,
                "ip_detJ_hat_scaling_rel": tol,
                "ip_detJ_vs_det_ip_J_rel": 1.0e-6,
            }
            for key, limit in strict_rules.items():
                if not np.isfinite(metrics[key]) or metrics[key] >= float(limit):
                    failures.append(f"{key}_ge_{limit:g}")
            if metrics["Q_stack_orthonormal_max"] >= 5.0e-12:
                failures.append("Q_stack_not_orthonormal")
            if metrics["Q_stack_det_min"] < 1.0 - 5.0e-12:
                failures.append("Q_stack_not_right_handed")
            if metrics["Q_stack_e3_dot_g_t_min"] < 1.0 - 1.0e-12:
                failures.append("Q_stack_e3_not_aligned_to_g_t")
            if scalar_bool(z["uses_old_true176_labels_as_v3_labels"], False) if "uses_old_true176_labels_as_v3_labels" in z.files else False:
                failures.append("uses_old_true176_labels_as_v3_labels_true")
            version = scalar_text(z["standard_operator_contract_version"], "unknown")
            if version != CONTRACT_VERSION:
                failures.append(f"contract_version_not_{CONTRACT_VERSION}")

        contract_version = scalar_text(z["standard_operator_contract_version"], "unknown")
        case_id = int(np.asarray(z["case_id"]).reshape(-1)[0]) if "case_id" in z.files else None
        macro_shape = scalar_text(z["macro_shape"], "unknown") if "macro_shape" in z.files else "unknown"

    return {
        "compact_path": str(path),
        "case_id": case_id,
        "standard_operator_contract_version": contract_version,
        "macro_shape": macro_shape,
        "strict_requested": bool(strict),
        "strict_pass": len(failures) == 0,
        "strict_failures": failures,
        "frame_count": frame_count,
        "point_count": point_count,
        "shapes": {
            "q_useful_hat": shape_list(q_useful_hat),
            "geometry_global_hat": shape_list(geometry_global_hat),
            "ip_macro_xi": shape_list(ip_macro_xi),
            "ip_local_rst": shape_list(ip_local_rst),
            "trunk_features_hat": shape_list(trunk_features_hat),
            "LE_local_stack": shape_list(le_local_stack),
            "B_local_useful_stack_hat": shape_list(b_local_useful_stack_hat),
        },
        **metrics,
        "B_rigid_residual_strict_failure": False,
        "trained_model": False,
        "used_old_true176_labels": False,
    }


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (list, dict, tuple)):
            out[key] = json.dumps(value, ensure_ascii=False, sort_keys=True, default=json_default)
        else:
            out[key] = value
    return out


def write_outputs(out_root: Path, rows: list[dict[str, Any]], *, strict: bool, tol: float) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    pass_count = sum(1 for row in rows if row.get("strict_pass"))
    failure_counts: dict[str, int] = {}
    for row in rows:
        for failure in row.get("strict_failures", []):
            failure_counts[str(failure)] = failure_counts.get(str(failure), 0) + 1
    summary = {
        "audit_name": "v3_css8_standard_operator_compact",
        "standard_operator_contract_version": CONTRACT_VERSION,
        "compact_count": len(rows),
        "strict_requested": bool(strict),
        "strict_pass_count": int(pass_count),
        "strict_fail_count": int(len(rows) - pass_count),
        "strict_pass": bool(rows and pass_count == len(rows)),
        "tolerance": float(tol),
        "failure_counts": dict(sorted(failure_counts.items())),
        "trained_model": False,
        "used_old_true176_labels": False,
    }
    write_json(out_root / "v3_css8_standard_operator_audit_summary.json", summary)
    write_json(out_root / "v3_css8_standard_operator_audit_manifest.json", {"summary": summary, "cases": rows})
    csv_rows = [flatten_for_csv(row) for row in rows]
    if csv_rows:
        fields = sorted({key for row in csv_rows for key in row.keys()})
        with (out_root / "v3_css8_standard_operator_audit_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="append", type=Path, default=[])
    parser.add_argument("--compact-list", action="append", type=Path, default=[])
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--tol", type=float, default=1.0e-10)
    args = parser.parse_args()

    paths = collect_paths(args.compact, args.compact_list)
    if not paths:
        raise SystemExit("no v3 compact inputs provided")
    rows = [audit_one(path, strict=args.strict, tol=float(args.tol)) for path in paths]
    summary = write_outputs(args.out_root, rows, strict=bool(args.strict), tol=float(args.tol))
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))
    if args.strict and not summary["strict_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
