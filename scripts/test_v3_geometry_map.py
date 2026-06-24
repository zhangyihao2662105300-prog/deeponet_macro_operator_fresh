#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-test the v3 reference GeometryMap contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from macro_deeponet.true176_data import build_shape4_nodes  # noqa: E402
from v3_css8_standard_operator_common import (  # noqa: E402
    GeometryMap,
    css8_connectivity_zero_based,
    css8_point_table,
    rel_norm,
)


def main() -> int:
    shape4 = np.asarray([1.15, 0.08, 0.14, 0.2], dtype=np.float64)
    x_ref = build_shape4_nodes(shape4).astype(np.float64)
    point_table = css8_point_table()
    geom = GeometryMap(
        x_ref,
        css8_connectivity_zero_based(nx=4, ny=4),
        cell_type="CSS8",
        frame_mode="shell-normal",
    )
    local, local_names, trunk, trunk_names, fields = geom.build_trunk_features(point_table)
    phys = geom.physical_point_fields(fields)

    q = fields["Q"]
    qtq = np.einsum("pji,pjk->pik", q, q)
    normal = np.cross(fields["J_hat"][:, 0, :], fields["J_hat"][:, 1, :])
    normal_unit = normal / np.linalg.norm(normal, axis=1, keepdims=True)
    e3_dot_normal = np.einsum("pi,pi->p", q[:, :, 2], normal_unit)
    report = {
        "passed": True,
        "point_count": int(trunk.shape[0]),
        "local_feature_dim": int(local.shape[1]),
        "trunk_feature_dim": int(trunk.shape[1]),
        "local_name_count": int(len(local_names)),
        "trunk_name_count": int(len(trunk_names)),
        "X_phys_roundtrip_rel": rel_norm(phys["ip_xyz"] - (fields["x_hat"] * geom.L_ref + geom.center.reshape(1, 3)), phys["ip_xyz"]),
        "J_phys_roundtrip_rel": rel_norm(phys["ip_J"] - fields["J_hat"] * geom.L_ref, phys["ip_J"]),
        "invJ_phys_roundtrip_rel": rel_norm(phys["ip_invJ"] - fields["invJ_hat"] / geom.L_ref, phys["ip_invJ"]),
        "detJ_phys_roundtrip_rel": rel_norm(phys["ip_detJ"] - fields["detJ_hat"] * (geom.L_ref**3), phys["ip_detJ"]),
        "Q_orthonormal_max": float(np.max(np.abs(qtq - np.eye(3)))),
        "Q_det_min": float(np.min(np.linalg.det(q))),
        "Q_e3_dot_reference_normal_min": float(np.min(e3_dot_normal)),
    }
    tolerances = {
        "X_phys_roundtrip_rel": 1.0e-14,
        "J_phys_roundtrip_rel": 1.0e-14,
        "invJ_phys_roundtrip_rel": 1.0e-14,
        "detJ_phys_roundtrip_rel": 1.0e-14,
        "Q_orthonormal_max": 1.0e-12,
        "Q_e3_dot_reference_normal_min": 1.0 - 1.0e-12,
    }
    report["passed"] = (
        report["local_feature_dim"] == len(local_names)
        and report["trunk_feature_dim"] == len(trunk_names)
        and all(float(report[key]) <= limit for key, limit in tolerances.items() if key != "Q_e3_dot_reference_normal_min")
        and float(report["Q_e3_dot_reference_normal_min"]) >= tolerances["Q_e3_dot_reference_normal_min"]
        and float(report["Q_det_min"]) >= 1.0 - 1.0e-12
    )
    report["tolerances"] = tolerances
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
