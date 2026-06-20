"""Numerically validate the Hex8 isoparametric mapping itself.

This is separate from similarity scaling.  It checks that the map

    x(xi) = sum_a N_a(xi) X_a

and the Jacobian used by the code have the expected finite-element meaning:

    J = dx/dxi
    dN/dx = dN/dxi * J^{-1}
    LE = B q

The validation uses random positive-orientation Hex8 elements and compares the
implemented formulas against finite differences, Newton inverse mapping, and
affine-field reproduction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from macro_deeponet.geometry import (  # noqa: E402
    HEX8_NATURAL,
    jacobian_hex8,
    map_to_physical,
    shape_function_gradients_hex8,
    shape_functions_hex8,
)
from validate_isoparametric_scaling import engineering_strain, random_positive_hex8, strain_b_matrix  # noqa: E402


def max_abs(value: torch.Tensor) -> float:
    return float(torch.max(torch.abs(value)).detach().cpu())


def shape_value(nodes: torch.Tensor, values: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    n = shape_functions_hex8(xi.reshape(1, 3))[0]
    return torch.dot(n, values)


def inverse_map_newton(nodes: torch.Tensor, x_target: torch.Tensor, *, max_iter: int = 20) -> tuple[torch.Tensor, float]:
    xi = torch.zeros(3, dtype=nodes.dtype)
    for _ in range(int(max_iter)):
        x_cur = map_to_physical(nodes.reshape(1, 8, 3), xi.reshape(1, 3))[0]
        residual = x_cur - x_target
        if max_abs(residual) < 1.0e-14:
            break
        jac = jacobian_hex8(nodes.reshape(1, 8, 3), xi.reshape(1, 3))[0]
        step = torch.linalg.solve(jac, residual)
        xi = xi - step
    final = map_to_physical(nodes.reshape(1, 8, 3), xi.reshape(1, 3))[0] - x_target
    return xi, max_abs(final)


def finite_difference_jacobian(nodes: torch.Tensor, xi: torch.Tensor, eps: float) -> torch.Tensor:
    cols: list[torch.Tensor] = []
    for j in range(3):
        delta = torch.zeros(3, dtype=nodes.dtype)
        delta[j] = float(eps)
        x_plus = map_to_physical(nodes.reshape(1, 8, 3), (xi + delta).reshape(1, 3))[0]
        x_minus = map_to_physical(nodes.reshape(1, 8, 3), (xi - delta).reshape(1, 3))[0]
        cols.append((x_plus - x_minus) / (2.0 * float(eps)))
    return torch.stack(cols, dim=1)


def finite_difference_scalar_direction(
    nodes: torch.Tensor,
    values: torch.Tensor,
    xi: torch.Tensor,
    physical_direction: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    jac = jacobian_hex8(nodes.reshape(1, 8, 3), xi.reshape(1, 3))[0]
    natural_step = torch.linalg.solve(jac, physical_direction)
    f_plus = shape_value(nodes, values, xi + float(eps) * natural_step)
    f_minus = shape_value(nodes, values, xi - float(eps) * natural_step)
    return (f_plus - f_minus) / (2.0 * float(eps))


def expected_engineering_strain_from_affine(fmat: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            fmat[0, 0],
            fmat[1, 1],
            fmat[2, 2],
            fmat[0, 1] + fmat[1, 0],
            fmat[0, 2] + fmat[2, 0],
            fmat[1, 2] + fmat[2, 1],
        ]
    )


def finite_difference_b_matrix(nodes: torch.Tensor, q: torch.Tensor, xi: torch.Tensor, eps: float) -> torch.Tensor:
    cols: list[torch.Tensor] = []
    for col in range(24):
        pert = torch.zeros_like(q)
        pert.reshape(-1)[col] = float(eps)
        le_plus = engineering_strain(nodes, q + pert, xi)
        le_minus = engineering_strain(nodes, q - pert, xi)
        cols.append((le_plus - le_minus) / (2.0 * float(eps)))
    return torch.stack(cols, dim=1)


def run_mapping_validation(*, seed: int, trials: int, points_per_trial: int) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    xi_probe = torch.as_tensor(rng.uniform(-0.85, 0.85, size=(96, 3)), dtype=torch.float64)
    eps = 1.0e-6
    errors: dict[str, float] = {
        "partition_unity_abs": 0.0,
        "natural_gradient_sum_abs": 0.0,
        "nodal_kronecker_abs": 0.0,
        "nodal_interpolation_abs": 0.0,
        "J_finite_difference_abs": 0.0,
        "J_inverse_identity_abs": 0.0,
        "physical_gradient_chain_rule_abs": 0.0,
        "physical_gradient_directional_fd_abs": 0.0,
        "inverse_map_xi_abs": 0.0,
        "inverse_map_residual_abs": 0.0,
        "affine_displacement_strain_abs": 0.0,
        "B_times_q_minus_LE_abs": 0.0,
        "B_finite_difference_abs": 0.0,
    }

    natural_nodes = HEX8_NATURAL.to(dtype=torch.float64)
    n_at_nodes = shape_functions_hex8(natural_nodes)
    errors["nodal_kronecker_abs"] = max_abs(n_at_nodes - torch.eye(8, dtype=torch.float64))

    for _trial in range(int(trials)):
        nodes = random_positive_hex8(rng, xi_probe)
        for node_id, xi_node in enumerate(natural_nodes):
            mapped = map_to_physical(nodes.reshape(1, 8, 3), xi_node.reshape(1, 3))[0]
            errors["nodal_interpolation_abs"] = max(errors["nodal_interpolation_abs"], max_abs(mapped - nodes[node_id]))

        for _point in range(int(points_per_trial)):
            xi = torch.as_tensor(rng.uniform(-0.75, 0.75, size=(3,)), dtype=torch.float64)
            n = shape_functions_hex8(xi.reshape(1, 3))[0]
            dndxi = shape_function_gradients_hex8(xi.reshape(1, 3))[0]
            jac = jacobian_hex8(nodes.reshape(1, 8, 3), xi.reshape(1, 3))[0]
            invj = torch.linalg.inv(jac)
            dndx = dndxi @ invj

            errors["partition_unity_abs"] = max(errors["partition_unity_abs"], abs(float(torch.sum(n) - 1.0)))
            errors["natural_gradient_sum_abs"] = max(errors["natural_gradient_sum_abs"], max_abs(torch.sum(dndxi, dim=0)))
            errors["J_finite_difference_abs"] = max(errors["J_finite_difference_abs"], max_abs(finite_difference_jacobian(nodes, xi, eps) - jac))
            errors["J_inverse_identity_abs"] = max(errors["J_inverse_identity_abs"], max_abs(jac @ invj - torch.eye(3, dtype=torch.float64)))
            errors["physical_gradient_chain_rule_abs"] = max(errors["physical_gradient_chain_rule_abs"], max_abs(dndx @ jac - dndxi))

            values = torch.as_tensor(rng.normal(size=(8,)), dtype=torch.float64)
            direction = torch.as_tensor(rng.normal(size=(3,)), dtype=torch.float64)
            direction = direction / torch.linalg.norm(direction)
            directional_formula = torch.dot(dndx.T @ values, direction)
            directional_fd = finite_difference_scalar_direction(nodes, values, xi, direction, eps)
            errors["physical_gradient_directional_fd_abs"] = max(
                errors["physical_gradient_directional_fd_abs"],
                abs(float(directional_formula - directional_fd)),
            )

            x = map_to_physical(nodes.reshape(1, 8, 3), xi.reshape(1, 3))[0]
            xi_back, residual = inverse_map_newton(nodes, x)
            errors["inverse_map_xi_abs"] = max(errors["inverse_map_xi_abs"], max_abs(xi_back - xi))
            errors["inverse_map_residual_abs"] = max(errors["inverse_map_residual_abs"], residual)

            fmat = torch.as_tensor(rng.normal(scale=0.25, size=(3, 3)), dtype=torch.float64)
            affine_q = nodes @ fmat.T
            le_affine = engineering_strain(nodes, affine_q, xi)
            errors["affine_displacement_strain_abs"] = max(
                errors["affine_displacement_strain_abs"],
                max_abs(le_affine - expected_engineering_strain_from_affine(fmat)),
            )

            q = torch.as_tensor(rng.normal(size=(8, 3)), dtype=torch.float64)
            le = engineering_strain(nodes, q, xi)
            bmat = strain_b_matrix(nodes, xi)
            errors["B_times_q_minus_LE_abs"] = max(errors["B_times_q_minus_LE_abs"], max_abs(bmat @ q.reshape(-1) - le))
            b_fd = finite_difference_b_matrix(nodes, q, xi, eps)
            errors["B_finite_difference_abs"] = max(errors["B_finite_difference_abs"], max_abs(b_fd - bmat))

    tolerances = {
        "partition_unity_abs": 5.0e-14,
        "natural_gradient_sum_abs": 5.0e-14,
        "nodal_kronecker_abs": 5.0e-14,
        "nodal_interpolation_abs": 5.0e-14,
        "J_finite_difference_abs": 5.0e-10,
        "J_inverse_identity_abs": 5.0e-13,
        "physical_gradient_chain_rule_abs": 5.0e-13,
        "physical_gradient_directional_fd_abs": 5.0e-9,
        "inverse_map_xi_abs": 5.0e-11,
        "inverse_map_residual_abs": 5.0e-13,
        "affine_displacement_strain_abs": 5.0e-12,
        "B_times_q_minus_LE_abs": 5.0e-12,
        "B_finite_difference_abs": 5.0e-9,
    }
    passed = all(errors[key] <= tolerances[key] for key in errors)
    return {
        "passed": bool(passed),
        "seed": int(seed),
        "trials": int(trials),
        "points_per_trial": int(points_per_trial),
        "total_points": int(trials) * int(points_per_trial),
        "errors": errors,
        "tolerances": tolerances,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--points-per-trial", type=int, default=8)
    args = parser.parse_args()
    report = run_mapping_validation(seed=args.seed, trials=args.trials, points_per_trial=args.points_per_trial)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
