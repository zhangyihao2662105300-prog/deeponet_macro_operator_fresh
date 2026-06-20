"""Validate the TRUE176 4x4 CSS8 macro-element isoparametric map.

The TRUE176 geometry used by the current trainer is a 4x4 grid of 16 Hex8/CSS8
sub-elements with 50 reference nodes.  This script checks the actual contract
implemented in ``macro_deeponet.true176_data``:

    shape4 -> X_macro[50,3] -> 16 CSS8 elements -> 128 Gauss rows

It validates, row by row, that the reconstructed integration-point geometry is
consistent with isoparametric interpolation, the local Jacobian, inverse
Jacobian, determinant, affine-field strain reproduction, B=dLE/dq for the local
8-node displacement field, and uniform H scaling.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macro_deeponet.true176_data import (  # noqa: E402
    GAUSS_POINTS,
    NODE_SIGNS,
    build_shape4_ip_geometry,
    build_shape4_nodes,
    css8_elements,
    css8_shape,
    keep_node_ids,
    standard_css8_row_map,
)


def max_abs(arr: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(arr, dtype=np.float64))))


def point_local(nodes: np.ndarray, rst: tuple[float, float, float]) -> np.ndarray:
    n, _dndr = css8_shape(*rst)
    return np.asarray(n @ nodes, dtype=np.float64)


def jacobian_local(nodes: np.ndarray, rst: tuple[float, float, float]) -> np.ndarray:
    _n, dndr = css8_shape(*rst)
    return np.asarray(dndr.T @ nodes, dtype=np.float64)


def engineering_strain_local(nodes: np.ndarray, q: np.ndarray, rst: tuple[float, float, float]) -> np.ndarray:
    _n, dndr = css8_shape(*rst)
    jac = dndr.T @ nodes
    # TRUE176 stores J as [natural, physical] = dX/d(r,s,t)^T.
    # For physical gradients, dN/dx = dN/d(r,s,t) * inv(J^T).
    dndx = dndr @ np.linalg.inv(jac.T)
    grad_u = q.T @ dndx
    return np.asarray(
        [
            grad_u[0, 0],
            grad_u[1, 1],
            grad_u[2, 2],
            grad_u[0, 1] + grad_u[1, 0],
            grad_u[0, 2] + grad_u[2, 0],
            grad_u[1, 2] + grad_u[2, 1],
        ],
        dtype=np.float64,
    )


def strain_b_matrix_local(nodes: np.ndarray, rst: tuple[float, float, float]) -> np.ndarray:
    _n, dndr = css8_shape(*rst)
    jac = dndr.T @ nodes
    dndx = dndr @ np.linalg.inv(jac.T)
    out = np.zeros((6, 24), dtype=np.float64)
    for a in range(8):
        gx, gy, gz = dndx[a]
        ux = 3 * a
        uy = ux + 1
        uz = ux + 2
        out[0, ux] = gx
        out[1, uy] = gy
        out[2, uz] = gz
        out[3, ux] = gy
        out[3, uy] = gx
        out[4, ux] = gz
        out[4, uz] = gx
        out[5, uy] = gz
        out[5, uz] = gy
    return out


def finite_difference_jacobian(nodes: np.ndarray, rst: tuple[float, float, float], eps: float) -> np.ndarray:
    base = np.asarray(rst, dtype=np.float64)
    rows: list[np.ndarray] = []
    for j in range(3):
        step = np.zeros(3, dtype=np.float64)
        step[j] = float(eps)
        plus = point_local(nodes, tuple((base + step).tolist()))
        minus = point_local(nodes, tuple((base - step).tolist()))
        rows.append((plus - minus) / (2.0 * float(eps)))
    return np.stack(rows, axis=0)


def inverse_map_newton(nodes: np.ndarray, x_target: np.ndarray) -> tuple[np.ndarray, float]:
    rst = np.zeros(3, dtype=np.float64)
    for _ in range(30):
        x_cur = point_local(nodes, tuple(rst.tolist()))
        residual = x_cur - x_target
        if max_abs(residual) < 1.0e-14:
            break
        jac = jacobian_local(nodes, tuple(rst.tolist()))
        # Row convention: dx_row = drst_row @ J, so J.T * drst_col = dx_col.
        rst = rst - np.linalg.solve(jac.T, residual)
    return rst, max_abs(point_local(nodes, tuple(rst.tolist())) - x_target)


def finite_difference_b(nodes: np.ndarray, q: np.ndarray, rst: tuple[float, float, float], eps: float) -> np.ndarray:
    cols: list[np.ndarray] = []
    for col in range(24):
        dq = np.zeros_like(q)
        dq.reshape(-1)[col] = float(eps)
        plus = engineering_strain_local(nodes, q + dq, rst)
        minus = engineering_strain_local(nodes, q - dq, rst)
        cols.append((plus - minus) / (2.0 * float(eps)))
    return np.stack(cols, axis=1)


def expected_affine_strain(fmat: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            fmat[0, 0],
            fmat[1, 1],
            fmat[2, 2],
            fmat[0, 1] + fmat[1, 0],
            fmat[0, 2] + fmat[2, 0],
            fmat[1, 2] + fmat[2, 1],
        ],
        dtype=np.float64,
    )


def random_shape4(rng: np.random.Generator) -> np.ndarray:
    lam = float(rng.uniform(0.65, 2.2))
    tau = float(rng.uniform(0.015, 0.12))
    if rng.random() < 0.25:
        chi = 0.0
        mu = 0.0
    else:
        chi = float(rng.uniform(0.01, 0.22))
        mu = float(rng.uniform(-0.45, 0.45))
    return np.asarray([lam, tau, chi, mu], dtype=np.float64)


def run_css8_macro_validation(*, seed: int, shapes: int, rows_per_shape: int) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    row_map = standard_css8_row_map()
    elements = css8_elements()
    eps = 1.0e-5
    errors: dict[str, float] = {
        "shape_function_partition_abs": 0.0,
        "shape_function_gradient_sum_abs": 0.0,
        "element_nodal_interpolation_abs": 0.0,
        "stored_ip_xyz_abs": 0.0,
        "stored_ip_J_abs": 0.0,
        "stored_ip_invJ_abs": 0.0,
        "stored_ip_detJ_abs": 0.0,
        "J_finite_difference_abs": 0.0,
        "J_inverse_identity_abs": 0.0,
        "physical_gradient_chain_rule_abs": 0.0,
        "inverse_map_local_abs": 0.0,
        "inverse_map_residual_abs": 0.0,
        "affine_displacement_strain_abs": 0.0,
        "B_times_q_minus_LE_abs": 0.0,
        "B_finite_difference_abs": 0.0,
        "x_phys_over_H_minus_x_hat_abs": 0.0,
        "J_phys_over_H_minus_J_hat_abs": 0.0,
        "H_invJ_phys_minus_invJ_hat_abs": 0.0,
        "detJ_phys_over_H3_minus_detJ_hat_abs": 0.0,
        "B_hat_minus_H_B_phys_abs": 0.0,
        "X_keep_phys_over_H_minus_X_keep_hat_abs": 0.0,
    }
    det_min = math.inf
    checked_rows = 0
    keep_ids = keep_node_ids()

    for _shape_idx in range(int(shapes)):
        shape4 = random_shape4(rng)
        geom = build_shape4_ip_geometry(shape4.reshape(1, 4))
        macro_nodes = build_shape4_nodes(shape4).astype(np.float64)
        h = float(np.exp(rng.uniform(np.log(0.2), np.log(8.0))))
        macro_nodes_phys = h * macro_nodes
        keep_hat = macro_nodes[keep_ids - 1]
        keep_phys = macro_nodes_phys[keep_ids - 1]
        errors["X_keep_phys_over_H_minus_X_keep_hat_abs"] = max(
            errors["X_keep_phys_over_H_minus_X_keep_hat_abs"],
            max_abs(keep_phys / h - keep_hat),
        )

        if int(rows_per_shape) >= 128:
            row_ids = np.arange(128, dtype=np.int64)
        else:
            row_ids = rng.choice(np.arange(128, dtype=np.int64), size=int(rows_per_shape), replace=False)

        for row_id in row_ids:
            row = row_map[int(row_id)]
            elem_id = int(row[1])
            gp = int(row[3])
            rst = tuple(float(v) for v in GAUSS_POINTS[gp])
            conn = np.asarray(elements[elem_id], dtype=np.int64) - 1
            nodes = macro_nodes[conn]
            nodes_phys = macro_nodes_phys[conn]
            n, dndr = css8_shape(*rst)
            x = n @ nodes
            jac = dndr.T @ nodes
            invj = np.linalg.inv(jac)
            detj = float(np.linalg.det(jac))
            det_min = min(det_min, abs(detj))

            errors["shape_function_partition_abs"] = max(errors["shape_function_partition_abs"], abs(float(np.sum(n) - 1.0)))
            errors["shape_function_gradient_sum_abs"] = max(errors["shape_function_gradient_sum_abs"], max_abs(np.sum(dndr, axis=0)))
            for local_idx, rst_node in enumerate(NODE_SIGNS):
                n_node, _dndr_node = css8_shape(*tuple(float(v) for v in rst_node))
                errors["element_nodal_interpolation_abs"] = max(
                    errors["element_nodal_interpolation_abs"],
                    max_abs(n_node @ nodes - nodes[local_idx]),
                )

            errors["stored_ip_xyz_abs"] = max(errors["stored_ip_xyz_abs"], max_abs(x - geom["ip_xyz"][0, int(row_id)]))
            errors["stored_ip_J_abs"] = max(errors["stored_ip_J_abs"], max_abs(jac - geom["ip_J"][0, int(row_id)]))
            errors["stored_ip_invJ_abs"] = max(errors["stored_ip_invJ_abs"], max_abs(invj - geom["ip_invJ"][0, int(row_id)]))
            errors["stored_ip_detJ_abs"] = max(errors["stored_ip_detJ_abs"], abs(detj - float(geom["ip_detJ"][0, int(row_id)])))
            errors["J_finite_difference_abs"] = max(errors["J_finite_difference_abs"], max_abs(finite_difference_jacobian(nodes, rst, eps) - jac))
            errors["J_inverse_identity_abs"] = max(errors["J_inverse_identity_abs"], max_abs(jac @ invj - np.eye(3)))
            dndx = dndr @ np.linalg.inv(jac.T)
            errors["physical_gradient_chain_rule_abs"] = max(errors["physical_gradient_chain_rule_abs"], max_abs(dndx @ jac.T - dndr))

            rst_back, residual = inverse_map_newton(nodes, x)
            errors["inverse_map_local_abs"] = max(errors["inverse_map_local_abs"], max_abs(rst_back - np.asarray(rst)))
            errors["inverse_map_residual_abs"] = max(errors["inverse_map_residual_abs"], residual)

            fmat = rng.normal(scale=0.25, size=(3, 3))
            affine_q = nodes @ fmat.T
            errors["affine_displacement_strain_abs"] = max(
                errors["affine_displacement_strain_abs"],
                max_abs(engineering_strain_local(nodes, affine_q, rst) - expected_affine_strain(fmat)),
            )

            q = rng.normal(size=(8, 3))
            le = engineering_strain_local(nodes, q, rst)
            bmat = strain_b_matrix_local(nodes, rst)
            errors["B_times_q_minus_LE_abs"] = max(errors["B_times_q_minus_LE_abs"], max_abs(bmat @ q.reshape(-1) - le))
            errors["B_finite_difference_abs"] = max(errors["B_finite_difference_abs"], max_abs(finite_difference_b(nodes, q, rst, eps) - bmat))

            x_phys = n @ nodes_phys
            jac_phys = dndr.T @ nodes_phys
            invj_phys = np.linalg.inv(jac_phys)
            detj_phys = float(np.linalg.det(jac_phys))
            b_phys = strain_b_matrix_local(nodes_phys, rst)
            errors["x_phys_over_H_minus_x_hat_abs"] = max(errors["x_phys_over_H_minus_x_hat_abs"], max_abs(x_phys / h - x))
            errors["J_phys_over_H_minus_J_hat_abs"] = max(errors["J_phys_over_H_minus_J_hat_abs"], max_abs(jac_phys / h - jac))
            errors["H_invJ_phys_minus_invJ_hat_abs"] = max(errors["H_invJ_phys_minus_invJ_hat_abs"], max_abs(h * invj_phys - invj))
            errors["detJ_phys_over_H3_minus_detJ_hat_abs"] = max(
                errors["detJ_phys_over_H3_minus_detJ_hat_abs"],
                abs(detj_phys / (h**3) - detj),
            )
            errors["B_hat_minus_H_B_phys_abs"] = max(errors["B_hat_minus_H_B_phys_abs"], max_abs(bmat - h * b_phys))
            checked_rows += 1

    tolerances = {
        "shape_function_partition_abs": 5.0e-14,
        "shape_function_gradient_sum_abs": 5.0e-14,
        "element_nodal_interpolation_abs": 5.0e-14,
        "stored_ip_xyz_abs": 5.0e-14,
        "stored_ip_J_abs": 5.0e-14,
        "stored_ip_invJ_abs": 5.0e-12,
        "stored_ip_detJ_abs": 5.0e-14,
        "J_finite_difference_abs": 5.0e-8,
        "J_inverse_identity_abs": 5.0e-13,
        "physical_gradient_chain_rule_abs": 5.0e-13,
        "inverse_map_local_abs": 5.0e-10,
        "inverse_map_residual_abs": 5.0e-13,
        "affine_displacement_strain_abs": 5.0e-11,
        "B_times_q_minus_LE_abs": 5.0e-11,
        "B_finite_difference_abs": 5.0e-8,
        "x_phys_over_H_minus_x_hat_abs": 5.0e-14,
        "J_phys_over_H_minus_J_hat_abs": 5.0e-14,
        "H_invJ_phys_minus_invJ_hat_abs": 1.0e-11,
        "detJ_phys_over_H3_minus_detJ_hat_abs": 5.0e-14,
        "B_hat_minus_H_B_phys_abs": 5.0e-11,
        "X_keep_phys_over_H_minus_X_keep_hat_abs": 5.0e-14,
    }
    passed = all(errors[key] <= tolerances[key] for key in errors)
    return {
        "passed": bool(passed),
        "seed": int(seed),
        "shape_count": int(shapes),
        "rows_per_shape": int(rows_per_shape),
        "checked_rows": int(checked_rows),
        "detJ_abs_min": float(det_min),
        "errors": errors,
        "tolerances": tolerances,
        "note": "B checks are for local 8-node CSS8 displacement DOFs, not the Abaqus q48 condensed label.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--shapes", type=int, default=16)
    parser.add_argument("--rows-per-shape", type=int, default=128)
    args = parser.parse_args()
    report = run_css8_macro_validation(seed=args.seed, shapes=args.shapes, rows_per_shape=args.rows_per_shape)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
