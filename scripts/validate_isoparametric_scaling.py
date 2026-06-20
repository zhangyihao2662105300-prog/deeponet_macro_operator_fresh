"""Numerically validate isoparametric similarity-scaling rules.

The checked identities are for a 3D Hex8 isoparametric map under

    X_phys = H * X_hat
    q_phys = H * q_hat

at the same natural coordinate xi:

    x_phys / H       = x_hat
    J_phys / H       = J_hat
    H * invJ_phys    = invJ_hat
    detJ_phys / H^3  = detJ_hat
    LE(X_phys,q_phys)= LE(X_hat,q_hat)
    H * B_phys       = B_hat

where B = dLE/dq and LE is the linearized engineering strain vector.
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

from macro_deeponet.geometry import HEX8_NATURAL, jacobian_hex8, map_to_physical, shape_function_gradients_hex8  # noqa: E402
from macro_deeponet.point_features import transform_point_features_for_scale  # noqa: E402


def engineering_strain(nodes: torch.Tensor, q: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    dndxi = shape_function_gradients_hex8(xi.reshape(1, 3))[0]
    jac = jacobian_hex8(nodes.reshape(1, 8, 3), xi.reshape(1, 3))[0]
    dndx = dndxi @ torch.linalg.inv(jac)
    grad_u = q.T @ dndx
    return torch.stack(
        [
            grad_u[0, 0],
            grad_u[1, 1],
            grad_u[2, 2],
            grad_u[0, 1] + grad_u[1, 0],
            grad_u[0, 2] + grad_u[2, 0],
            grad_u[1, 2] + grad_u[2, 1],
        ]
    )


def strain_b_matrix(nodes: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    dndxi = shape_function_gradients_hex8(xi.reshape(1, 3))[0]
    jac = jacobian_hex8(nodes.reshape(1, 8, 3), xi.reshape(1, 3))[0]
    dndx = dndxi @ torch.linalg.inv(jac)
    out = torch.zeros((6, 24), dtype=nodes.dtype)
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


def max_abs(a: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a)).detach().cpu())


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.clamp(torch.max(torch.abs(b)), min=torch.as_tensor(1.0e-30, dtype=b.dtype))
    return float((torch.max(torch.abs(a - b)) / denom).detach().cpu())


def random_positive_hex8(rng: np.random.Generator, xi_probe: torch.Tensor) -> torch.Tensor:
    base = HEX8_NATURAL.to(dtype=torch.float64)
    for _attempt in range(1000):
        a = rng.normal(size=(3, 3))
        q, _r = np.linalg.qr(a)
        if np.linalg.det(q) < 0.0:
            q[:, 0] *= -1.0
        scales = np.diag(rng.uniform(0.35, 1.8, size=3))
        affine = q @ scales
        nodes = base @ torch.as_tensor(affine.T, dtype=torch.float64)
        nodes += torch.as_tensor(rng.normal(scale=0.05, size=(8, 3)), dtype=torch.float64)
        nodes += torch.as_tensor(rng.uniform(-0.5, 0.5, size=(1, 3)), dtype=torch.float64)
        repeated = nodes.reshape(1, 8, 3).expand(xi_probe.shape[0], -1, -1)
        det = torch.linalg.det(jacobian_hex8(repeated, xi_probe))
        if bool(torch.all(det > 1.0e-3)):
            return nodes
    raise RuntimeError("failed to sample a positive-orientation Hex8 element")


def point_feature_transform_error(records: list[dict[str, torch.Tensor]], h_values: list[float]) -> float:
    names = ["ip_xyz_x", "ip_xyz_y", "ip_xyz_z"]
    names += [f"ip_J_{i}{j}" for i in range(3) for j in range(3)]
    names += [f"ip_invJ_{i}{j}" for i in range(3) for j in range(3)]
    names += ["ip_detJ"]

    physical_rows: list[np.ndarray] = []
    expected_rows: list[np.ndarray] = []
    for rec in records:
        physical_rows.append(
            np.concatenate(
                [
                    rec["x_phys"].detach().cpu().numpy().reshape(-1),
                    rec["j_phys"].detach().cpu().numpy().reshape(-1),
                    rec["invj_phys"].detach().cpu().numpy().reshape(-1),
                    rec["det_phys"].detach().cpu().numpy().reshape(-1),
                ]
            )
        )
        expected_rows.append(
            np.concatenate(
                [
                    rec["x_hat"].detach().cpu().numpy().reshape(-1),
                    rec["j_hat"].detach().cpu().numpy().reshape(-1),
                    rec["invj_hat"].detach().cpu().numpy().reshape(-1),
                    rec["det_hat"].detach().cpu().numpy().reshape(-1),
                ]
            )
        )

    point = np.asarray(physical_rows, dtype=np.float32).reshape(len(physical_rows), 1, -1)
    expected = np.asarray(expected_rows, dtype=np.float32).reshape(len(expected_rows), 1, -1)
    scaled, _meta = transform_point_features_for_scale(
        point,
        names,
        np.asarray(h_values, dtype=np.float32).reshape(-1, 1),
        scale_mode="physical",
        detj_scale_dim=3,
    )
    return float(np.max(np.abs(scaled - expected)))


def run_validation(*, seed: int, trials: int, points_per_trial: int) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    xi_probe = torch.as_tensor(rng.uniform(-0.8, 0.8, size=(64, 3)), dtype=torch.float64)
    errors: dict[str, float] = {
        "x_phys_over_H_minus_x_hat_abs": 0.0,
        "J_phys_over_H_minus_J_hat_abs": 0.0,
        "H_invJ_phys_minus_invJ_hat_abs": 0.0,
        "detJ_phys_over_H3_minus_detJ_hat_abs": 0.0,
        "LE_scaled_similarity_abs": 0.0,
        "B_hat_minus_H_B_phys_abs": 0.0,
        "B_hat_minus_H_B_phys_rel": 0.0,
        "LE_unscaled_q_scales_as_1_over_H_abs": 0.0,
        "point_feature_transform_abs": 0.0,
    }
    records: list[dict[str, torch.Tensor]] = []
    h_values: list[float] = []

    for _trial in range(int(trials)):
        nodes_hat = random_positive_hex8(rng, xi_probe)
        h = float(np.exp(rng.uniform(np.log(0.2), np.log(8.0))))
        nodes_phys = h * nodes_hat
        for _point in range(int(points_per_trial)):
            xi = torch.as_tensor(rng.uniform(-0.8, 0.8, size=(3,)), dtype=torch.float64)
            q_hat = torch.as_tensor(rng.normal(size=(8, 3)), dtype=torch.float64)
            q_phys = h * q_hat

            x_hat = map_to_physical(nodes_hat.reshape(1, 8, 3), xi.reshape(1, 3))[0]
            x_phys = map_to_physical(nodes_phys.reshape(1, 8, 3), xi.reshape(1, 3))[0]
            j_hat = jacobian_hex8(nodes_hat.reshape(1, 8, 3), xi.reshape(1, 3))[0]
            j_phys = jacobian_hex8(nodes_phys.reshape(1, 8, 3), xi.reshape(1, 3))[0]
            invj_hat = torch.linalg.inv(j_hat)
            invj_phys = torch.linalg.inv(j_phys)
            det_hat = torch.linalg.det(j_hat).reshape(1)
            det_phys = torch.linalg.det(j_phys).reshape(1)

            le_hat = engineering_strain(nodes_hat, q_hat, xi)
            le_phys = engineering_strain(nodes_phys, q_phys, xi)
            le_unscaled_q = engineering_strain(nodes_phys, q_hat, xi)
            b_hat = strain_b_matrix(nodes_hat, xi)
            b_phys = strain_b_matrix(nodes_phys, xi)

            errors["x_phys_over_H_minus_x_hat_abs"] = max(errors["x_phys_over_H_minus_x_hat_abs"], max_abs(x_phys / h - x_hat))
            errors["J_phys_over_H_minus_J_hat_abs"] = max(errors["J_phys_over_H_minus_J_hat_abs"], max_abs(j_phys / h - j_hat))
            errors["H_invJ_phys_minus_invJ_hat_abs"] = max(errors["H_invJ_phys_minus_invJ_hat_abs"], max_abs(h * invj_phys - invj_hat))
            errors["detJ_phys_over_H3_minus_detJ_hat_abs"] = max(
                errors["detJ_phys_over_H3_minus_detJ_hat_abs"], max_abs(det_phys / (h**3) - det_hat)
            )
            errors["LE_scaled_similarity_abs"] = max(errors["LE_scaled_similarity_abs"], max_abs(le_phys - le_hat))
            errors["B_hat_minus_H_B_phys_abs"] = max(errors["B_hat_minus_H_B_phys_abs"], max_abs(b_hat - h * b_phys))
            errors["B_hat_minus_H_B_phys_rel"] = max(errors["B_hat_minus_H_B_phys_rel"], relative_error(h * b_phys, b_hat))
            errors["LE_unscaled_q_scales_as_1_over_H_abs"] = max(
                errors["LE_unscaled_q_scales_as_1_over_H_abs"], max_abs(h * le_unscaled_q - le_hat)
            )

            records.append(
                {
                    "x_phys": x_phys,
                    "j_phys": j_phys,
                    "invj_phys": invj_phys,
                    "det_phys": det_phys,
                    "x_hat": x_hat,
                    "j_hat": j_hat,
                    "invj_hat": invj_hat,
                    "det_hat": det_hat,
                }
            )
            h_values.append(h)

    errors["point_feature_transform_abs"] = point_feature_transform_error(records, h_values)
    passed = all(value < 5.0e-10 for key, value in errors.items() if key != "point_feature_transform_abs")
    passed = passed and errors["point_feature_transform_abs"] < 5.0e-5
    return {
        "passed": bool(passed),
        "seed": int(seed),
        "trials": int(trials),
        "points_per_trial": int(points_per_trial),
        "total_points": int(trials) * int(points_per_trial),
        "errors": errors,
        "tolerance_double_precision": 5.0e-10,
        "tolerance_point_feature_float32": 5.0e-5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--points-per-trial", type=int, default=8)
    args = parser.parse_args()

    report = run_validation(seed=args.seed, trials=args.trials, points_per_trial=args.points_per_trial)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
