#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random-geometry isoparametric invariance audit for the v2 route.

This script is a geometry/data-contract test only.  It does not train a model
and it does not use old TRUE176 labels.  It complements the fixed 10-case
standard-operator audit by sampling random positive-orientation Hex8 geometries
and checking:

* standard natural coordinates remain geometry-independent,
* x, J, invJ, detJ transform correctly under translation/rotation/scaling,
* dN/dx and the engineering-strain B matrix have the expected length scaling,
* local/global strain transforms round-trip for random local frames from J,
* B_local_useful projects back to the useful raw-B subspace by chain rule.
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from macro_deeponet.geometry import (  # noqa: E402
    jacobian_hex8,
    map_to_physical,
    shape_function_gradients_hex8,
    shape_functions_hex8,
)
from build_v2_local_strain_pilot_compact import (  # noqa: E402
    local_frame_from_j,
    strain_transform_matrices,
)
from validate_isoparametric_scaling import (  # noqa: E402
    engineering_strain,
    random_positive_hex8,
    strain_b_matrix,
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


def max_abs_t(value: torch.Tensor) -> float:
    return float(torch.max(torch.abs(value)).detach().cpu())


def max_abs_np(value: np.ndarray) -> float:
    vals = np.asarray(value, dtype=np.float64)
    return float(np.max(np.abs(vals))) if vals.size else 0.0


def rel_norm_np(diff: np.ndarray, ref: np.ndarray) -> float:
    den = max(float(np.linalg.norm(np.asarray(ref, dtype=np.float64).reshape(-1))), 1.0e-30)
    return float(np.linalg.norm(np.asarray(diff, dtype=np.float64).reshape(-1)) / den)


def random_rotation(rng: np.random.Generator) -> torch.Tensor:
    a = rng.normal(size=(3, 3))
    q, _r = np.linalg.qr(a)
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return torch.as_tensor(q, dtype=torch.float64)


def random_t_q(rng: np.random.Generator) -> np.ndarray:
    # Orthonormal 42-row projection from raw 48-DOF q to useful q.
    q, _r = np.linalg.qr(rng.normal(size=(48, 42)))
    return q.T.astype(np.float64)


def inverse_map_newton(nodes: torch.Tensor, x_target: torch.Tensor, *, max_iter: int = 24) -> tuple[torch.Tensor, float]:
    xi = torch.zeros(3, dtype=nodes.dtype)
    for _step in range(int(max_iter)):
        x_cur = map_to_physical(nodes.reshape(1, 8, 3), xi.reshape(1, 3))[0]
        residual = x_cur - x_target
        if max_abs_t(residual) < 1.0e-14:
            break
        jac = jacobian_hex8(nodes.reshape(1, 8, 3), xi.reshape(1, 3))[0]
        xi = xi - torch.linalg.solve(jac, residual)
    final = map_to_physical(nodes.reshape(1, 8, 3), xi.reshape(1, 3))[0] - x_target
    return xi, max_abs_t(final)


def tensor_to_voigt_np(tensor: np.ndarray) -> np.ndarray:
    t = np.asarray(tensor, dtype=np.float64).reshape(3, 3)
    return np.asarray([t[0, 0], t[1, 1], t[2, 2], t[0, 1], t[0, 2], t[1, 2]], dtype=np.float64)


def random_symmetric_voigt(rng: np.random.Generator, count: int) -> np.ndarray:
    raw = rng.normal(size=(int(count), 3, 3))
    sym = 0.5 * (raw + np.swapaxes(raw, -1, -2))
    return np.asarray([tensor_to_voigt_np(sym[i]) for i in range(int(count))], dtype=np.float64)


def update_max(errors: dict[str, float], key: str, value: float) -> None:
    errors[key] = max(float(errors.get(key, 0.0)), float(value))


def run_audit(
    *,
    seed: int,
    trials: int,
    points_per_trial: int,
    local_frame_convention: str,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    xi_probe = torch.as_tensor(rng.uniform(-0.82, 0.82, size=(96, 3)), dtype=torch.float64)
    errors: dict[str, float] = {
        "partition_unity_abs": 0.0,
        "natural_gradient_sum_abs": 0.0,
        "ip_xi_invariance_abs": 0.0,
        "positive_detJ_min_violation": 0.0,
        "translation_x_abs": 0.0,
        "translation_J_abs": 0.0,
        "rotation_scale_x_abs": 0.0,
        "rotation_scale_J_abs": 0.0,
        "rotation_scale_invJ_abs": 0.0,
        "rotation_scale_detJ_abs": 0.0,
        "dndx_rotation_scale_abs": 0.0,
        "B_scaling_abs": 0.0,
        "B_scaling_rel": 0.0,
        "LE_scaled_similarity_abs": 0.0,
        "B_times_q_minus_LE_abs": 0.0,
        "local_frame_orthonormal_abs": 0.0,
        "local_frame_det_abs": 0.0,
        "T_eps_roundtrip_abq_rel": 0.0,
        "T_eps_roundtrip_local_rel": 0.0,
        "LE_local_to_abq_roundtrip_rel": 0.0,
        "B_raw_projected_rel": 0.0,
        "B_rigid_residual_rel": 0.0,
    }
    det_min = math.inf
    frames: list[np.ndarray] = []

    convention = str(local_frame_convention).strip().lower()
    if convention not in {"rows", "columns", "cols"}:
        raise ValueError("--local-frame-convention must be rows or columns")
    if convention == "cols":
        convention = "columns"

    for _trial in range(int(trials)):
        nodes = random_positive_hex8(rng, xi_probe)
        translation = torch.as_tensor(rng.normal(scale=2.0, size=(3,)), dtype=torch.float64)
        rotation = random_rotation(rng)
        scale = float(np.exp(rng.uniform(np.log(0.25), np.log(6.0))))
        nodes_translated = nodes + translation.reshape(1, 3)
        nodes_rot_scaled = scale * (nodes @ rotation.T) + translation.reshape(1, 3)
        nodes_scaled = scale * nodes

        for _point in range(int(points_per_trial)):
            xi_np = rng.uniform(-0.78, 0.78, size=(3,))
            xi = torch.as_tensor(xi_np, dtype=torch.float64)
            n = shape_functions_hex8(xi.reshape(1, 3))[0]
            dndxi = shape_function_gradients_hex8(xi.reshape(1, 3))[0]
            jac = jacobian_hex8(nodes.reshape(1, 8, 3), xi.reshape(1, 3))[0]
            jac_translated = jacobian_hex8(nodes_translated.reshape(1, 8, 3), xi.reshape(1, 3))[0]
            jac_rot_scaled = jacobian_hex8(nodes_rot_scaled.reshape(1, 8, 3), xi.reshape(1, 3))[0]
            jac_scaled = jacobian_hex8(nodes_scaled.reshape(1, 8, 3), xi.reshape(1, 3))[0]
            det = float(torch.linalg.det(jac))
            det_min = min(det_min, det)
            if det <= 0.0:
                update_max(errors, "positive_detJ_min_violation", abs(det) + 1.0e-12)

            invj = torch.linalg.inv(jac)
            invj_rot_scaled = torch.linalg.inv(jac_rot_scaled)
            dndx = dndxi @ invj
            dndx_rot_scaled = dndxi @ invj_rot_scaled

            x = map_to_physical(nodes.reshape(1, 8, 3), xi.reshape(1, 3))[0]
            x_translated = map_to_physical(nodes_translated.reshape(1, 8, 3), xi.reshape(1, 3))[0]
            x_rot_scaled = map_to_physical(nodes_rot_scaled.reshape(1, 8, 3), xi.reshape(1, 3))[0]
            xi_back, xi_residual = inverse_map_newton(nodes, x)
            xi_translated_back, xi_translated_residual = inverse_map_newton(nodes_translated, x_translated)
            xi_rot_scaled_back, xi_rot_scaled_residual = inverse_map_newton(nodes_rot_scaled, x_rot_scaled)

            update_max(errors, "partition_unity_abs", abs(float(torch.sum(n) - 1.0)))
            update_max(errors, "natural_gradient_sum_abs", max_abs_t(torch.sum(dndxi, dim=0)))
            update_max(
                errors,
                "ip_xi_invariance_abs",
                max(
                    max_abs_t(xi_back - xi),
                    max_abs_t(xi_translated_back - xi),
                    max_abs_t(xi_rot_scaled_back - xi),
                    xi_residual,
                    xi_translated_residual,
                    xi_rot_scaled_residual,
                ),
            )
            update_max(errors, "translation_x_abs", max_abs_t(x_translated - (x + translation)))
            update_max(errors, "translation_J_abs", max_abs_t(jac_translated - jac))
            update_max(errors, "rotation_scale_x_abs", max_abs_t(x_rot_scaled - (scale * (rotation @ x) + translation)))
            update_max(errors, "rotation_scale_J_abs", max_abs_t(jac_rot_scaled - scale * (rotation @ jac)))
            update_max(errors, "rotation_scale_invJ_abs", max_abs_t(invj_rot_scaled - (invj @ rotation.T / scale)))
            update_max(
                errors,
                "rotation_scale_detJ_abs",
                abs(float(torch.linalg.det(jac_rot_scaled) - (scale**3) * torch.linalg.det(jac))),
            )
            update_max(errors, "dndx_rotation_scale_abs", max_abs_t(dndx_rot_scaled - (dndx @ rotation.T / scale)))

            q = torch.as_tensor(rng.normal(size=(8, 3)), dtype=torch.float64)
            q_scaled = scale * q
            b = strain_b_matrix(nodes, xi)
            b_scaled = strain_b_matrix(nodes_scaled, xi)
            le = engineering_strain(nodes, q, xi)
            le_scaled = engineering_strain(nodes_scaled, q_scaled, xi)

            update_max(errors, "B_scaling_abs", max_abs_t(b_scaled - b / scale))
            update_max(errors, "B_scaling_rel", max_abs_t((scale * b_scaled - b) / torch.clamp(torch.max(torch.abs(b)), min=1.0e-30)))
            update_max(errors, "LE_scaled_similarity_abs", max_abs_t(le_scaled - le))
            update_max(errors, "B_times_q_minus_LE_abs", max_abs_t(b @ q.reshape(-1) - le))

            frame_q, _orientation = local_frame_from_j(
                jac.detach().cpu().numpy(),
                convention=convention,
                eps=1.0e-12,
            )
            frames.append(frame_q)

    q_frames = np.asarray(frames, dtype=np.float64)
    qtq = np.einsum("...ia,...ib->...ab", q_frames, q_frames)
    update_max(errors, "local_frame_orthonormal_abs", max_abs_np(qtq - np.eye(3)))
    update_max(errors, "local_frame_det_abs", max_abs_np(np.linalg.det(q_frames) - 1.0))

    t_from, t_to = strain_transform_matrices(q_frames)
    eye6 = np.eye(6, dtype=np.float64)
    t_roundtrip_abq = np.einsum("pab,pbc->pac", t_to, t_from)
    t_roundtrip_local = np.einsum("pab,pbc->pac", t_from, t_to)
    errors["T_eps_roundtrip_abq_rel"] = rel_norm_np(t_roundtrip_abq - eye6.reshape(1, 6, 6), eye6)
    errors["T_eps_roundtrip_local_rel"] = rel_norm_np(t_roundtrip_local - eye6.reshape(1, 6, 6), eye6)

    point_count = q_frames.shape[0]
    le_abq = random_symmetric_voigt(rng, point_count)
    le_local = np.einsum("pab,pb->pa", t_from, le_abq)
    le_abq_recovered = np.einsum("pab,pb->pa", t_to, le_local)
    errors["LE_local_to_abq_roundtrip_rel"] = rel_norm_np(le_abq_recovered - le_abq, le_abq)

    b_raw = rng.normal(size=(point_count, 6, 48))
    t_q = random_t_q(rng)
    b_local_raw = np.einsum("pab,pbj->paj", t_from, b_raw)
    b_local_useful = np.einsum("paj,kj->pak", b_local_raw, t_q)
    b_abq_useful = np.einsum("pab,pbk->pak", t_to, b_local_useful)
    b_raw_hat_projected = np.einsum("pak,kj->paj", b_abq_useful, t_q)
    p_useful = t_q.T @ t_q
    b_raw_projected = np.einsum("paj,jk->pak", b_raw, p_useful)
    errors["B_raw_projected_rel"] = rel_norm_np(b_raw_hat_projected - b_raw_projected, b_raw_projected)
    errors["B_rigid_residual_rel"] = rel_norm_np(b_raw_hat_projected - b_raw, b_raw)

    tolerances = {
        "partition_unity_abs": 5.0e-14,
        "natural_gradient_sum_abs": 5.0e-14,
        "ip_xi_invariance_abs": 5.0e-11,
        "positive_detJ_min_violation": 0.0,
        "translation_x_abs": 5.0e-13,
        "translation_J_abs": 5.0e-13,
        "rotation_scale_x_abs": 5.0e-12,
        "rotation_scale_J_abs": 5.0e-12,
        "rotation_scale_invJ_abs": 5.0e-11,
        "rotation_scale_detJ_abs": 5.0e-11,
        "dndx_rotation_scale_abs": 5.0e-11,
        "B_scaling_abs": 5.0e-11,
        "B_scaling_rel": 5.0e-11,
        "LE_scaled_similarity_abs": 5.0e-12,
        "B_times_q_minus_LE_abs": 5.0e-12,
        "local_frame_orthonormal_abs": 5.0e-13,
        "local_frame_det_abs": 5.0e-13,
        "T_eps_roundtrip_abq_rel": 5.0e-13,
        "T_eps_roundtrip_local_rel": 5.0e-13,
        "LE_local_to_abq_roundtrip_rel": 5.0e-13,
        "B_raw_projected_rel": 5.0e-13,
        # B_rigid_residual_rel is informational: useful q intentionally removes
        # six raw-q modes, so raw residual is not a strict failure.
    }
    strict_keys = list(tolerances)
    passed = all(float(errors[key]) <= float(tolerances[key]) for key in strict_keys)
    return {
        "audit_name": "random_geometry_isoparametric_invariance",
        "passed": bool(passed),
        "seed": int(seed),
        "trials": int(trials),
        "points_per_trial": int(points_per_trial),
        "total_points": int(trials) * int(points_per_trial),
        "local_frame_convention": convention,
        "detJ_min": float(det_min),
        "errors": errors,
        "tolerances": tolerances,
        "strict_keys": strict_keys,
        "trained_model": False,
        "used_old_true176_labels": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--points-per-trial", type=int, default=8)
    parser.add_argument("--local-frame-convention", choices=("rows", "columns"), default="rows")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    report = run_audit(
        seed=args.seed,
        trials=args.trials,
        points_per_trial=args.points_per_trial,
        local_frame_convention=args.local_frame_convention,
    )
    text = json.dumps(report, indent=2, sort_keys=True, default=json_default)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
