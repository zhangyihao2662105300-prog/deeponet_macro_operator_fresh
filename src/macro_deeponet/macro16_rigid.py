"""Rigid-motion preprocessing for Macro16 boundary displacement vectors."""

from __future__ import annotations

from typing import Any

import numpy as np


def rigid_modes_from_x16(x16_raw: np.ndarray) -> np.ndarray:
    """Return 48x6 small-motion rigid modes for Macro16 nodes."""

    x = np.asarray(x16_raw, dtype=np.float64).reshape(16, 3)
    center = np.mean(x, axis=0)
    x0 = x - center.reshape(1, 3)
    modes = np.zeros((16, 3, 6), dtype=np.float64)
    modes[:, 0, 0] = 1.0
    modes[:, 1, 1] = 1.0
    modes[:, 2, 2] = 1.0
    axes = np.eye(3, dtype=np.float64)
    for i, axis in enumerate(axes):
        modes[:, :, 3 + i] = np.cross(axis.reshape(1, 3), x0, axis=1)
    return modes.reshape(48, 6)


def rigid_projection_matrix(x16_raw: np.ndarray, *, rtol: float = 1.0e-12) -> np.ndarray:
    """Return the 48x48 projector that removes linearized rigid modes.

    This is a local small-rotation projector, not the Jacobian of the nonlinear
    Kabsch preprocessing used by ``remove_rigid_motion``.  It is the first
    chain-rule approximation for B labels and can be replaced later by a
    numerical or analytic Kabsch Jacobian.
    """

    modes = rigid_modes_from_x16(x16_raw)
    u, s, _vt = np.linalg.svd(modes, full_matrices=False)
    if s.size == 0:
        raise ValueError("cannot build rigid projection from empty mode matrix")
    rank = int(np.sum(s > float(rtol) * max(float(s[0]), 1.0)))
    if rank <= 0:
        raise ValueError("rigid mode matrix has zero rank")
    q = u[:, :rank]
    eye = np.eye(48, dtype=np.float64)
    return eye - q @ q.T


def fit_rigid_transform(x16_raw: np.ndarray, q48_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the best proper rigid transform from X16_raw to X16_raw + q48_raw."""

    x = np.asarray(x16_raw, dtype=np.float64).reshape(16, 3)
    y = x + np.asarray(q48_raw, dtype=np.float64).reshape(16, 3)
    x_center = np.mean(x, axis=0)
    y_center = np.mean(y, axis=0)
    x0 = x - x_center.reshape(1, 3)
    y0 = y - y_center.reshape(1, 3)
    h = x0.T @ y0
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if float(np.linalg.det(r)) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = y_center - r @ x_center
    return r.astype(np.float64), t.astype(np.float64)


def remove_rigid_motion(x16_raw: np.ndarray, q48_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove best-fit finite rigid motion from a 48D Macro16 displacement."""

    x = np.asarray(x16_raw, dtype=np.float64).reshape(16, 3)
    q = np.asarray(q48_raw, dtype=np.float64).reshape(16, 3)
    r, t = fit_rigid_transform(x, q.reshape(48))
    x_rigid = (r @ x.T).T + t.reshape(1, 3)
    q_rigid = x_rigid - x
    q_def = q - q_rigid
    return q_def.reshape(48), q_rigid.reshape(48), r, t


def rigid_preprocess_batch(
    x16_raw: np.ndarray,
    q48_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply rigid removal and linear projection construction frame by frame."""

    x = np.asarray(x16_raw, dtype=np.float64).reshape(-1, 16, 3)
    q = np.asarray(q48_raw, dtype=np.float64).reshape(-1, 48)
    if x.shape[0] != q.shape[0]:
        raise ValueError(f"X16/q48 frame count mismatch: {x.shape[0]} != {q.shape[0]}")
    q_def = np.empty_like(q, dtype=np.float64)
    q_rigid = np.empty_like(q, dtype=np.float64)
    rotations = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    translations = np.empty((q.shape[0], 3), dtype=np.float64)
    projectors = np.empty((q.shape[0], 48, 48), dtype=np.float64)
    for i in range(q.shape[0]):
        q_def[i], q_rigid[i], rotations[i], translations[i] = remove_rigid_motion(x[i], q[i])
        projectors[i] = rigid_projection_matrix(x[i])
    return q_def, q_rigid, rotations, translations, projectors


def rigid_consistency_report(
    *,
    x16_raw: np.ndarray,
    q48_raw: np.ndarray,
    q48_rigid_raw: np.ndarray,
    q48_def_raw: np.ndarray,
    q48_def_hat: np.ndarray | None = None,
    l_ref: np.ndarray | None = None,
    rigid_projection_p: np.ndarray | None = None,
) -> dict[str, Any]:
    """Report raw/def closure and orthogonality to linear rigid modes."""

    x = np.asarray(x16_raw, dtype=np.float64).reshape(-1, 16, 3)
    q_raw = np.asarray(q48_raw, dtype=np.float64).reshape(-1, 48)
    q_rigid = np.asarray(q48_rigid_raw, dtype=np.float64).reshape(-1, 48)
    q_def = np.asarray(q48_def_raw, dtype=np.float64).reshape(-1, 48)
    if x.shape[0] != q_raw.shape[0] or q_raw.shape != q_rigid.shape or q_raw.shape != q_def.shape:
        raise ValueError("rigid consistency arrays must share frame count and 48D q shape")

    def rel(diff: np.ndarray, ref: np.ndarray) -> float:
        den = max(float(np.linalg.norm(np.asarray(ref, dtype=np.float64).reshape(-1))), 1.0e-30)
        return float(np.linalg.norm(np.asarray(diff, dtype=np.float64).reshape(-1)) / den)

    closure = q_rigid + q_def - q_raw
    trans_dot: list[float] = []
    rot_dot: list[float] = []
    proj_diff: list[float] = []
    for i in range(q_raw.shape[0]):
        modes = rigid_modes_from_x16(x[i])
        dots = modes.T @ q_def[i]
        trans_dot.append(float(np.max(np.abs(dots[:3]))))
        rot_dot.append(float(np.max(np.abs(dots[3:]))))
        if rigid_projection_p is not None:
            p = np.asarray(rigid_projection_p, dtype=np.float64).reshape(q_raw.shape[0], 48, 48)[i]
            proj_diff.append(float(np.max(np.abs(p @ q_raw[i] - q_def[i]))))

    out: dict[str, Any] = {
        "q48_raw_vs_rigid_plus_def_rel": rel(closure, q_raw),
        "q48_raw_vs_rigid_plus_def_max_abs": float(np.max(np.abs(closure))) if closure.size else 0.0,
        "q48_def_translation_orthogonality_max_abs": float(max(trans_dot, default=0.0)),
        "q48_def_rotation_orthogonality_max_abs": float(max(rot_dot, default=0.0)),
    }
    if proj_diff:
        out["linear_projection_P_qraw_vs_kabsch_qdef_max_abs"] = float(max(proj_diff))
        out["linear_projection_note"] = (
            "rigid_projection_P is a small-rotation chain-rule approximation; "
            "Kabsch rigid removal is nonlinear."
        )
    if q48_def_hat is not None and l_ref is not None:
        l_ref_arr = np.asarray(l_ref, dtype=np.float64).reshape(-1, 1)
        diff = np.asarray(q48_def_hat, dtype=np.float64).reshape(-1, 48) * l_ref_arr - q_def
        out["q48_def_hat_times_L_ref_vs_q48_def_raw_rel"] = rel(diff, q_def)
        out["q48_def_hat_times_L_ref_vs_q48_def_raw_max_abs"] = float(np.max(np.abs(diff))) if diff.size else 0.0
    return out
