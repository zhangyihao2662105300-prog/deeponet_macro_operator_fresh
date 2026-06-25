from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from macro_deeponet.macro16_rigid import remove_rigid_motion, rigid_projection_matrix, rigid_preprocess_batch


def flat_macro16_x16(thickness: float = 0.2) -> np.ndarray:
    bottom = np.asarray(
        [
            [-1.0, -1.0, -0.5 * thickness],
            [0.0, -1.0, -0.5 * thickness],
            [1.0, -1.0, -0.5 * thickness],
            [1.0, 0.0, -0.5 * thickness],
            [1.0, 1.0, -0.5 * thickness],
            [0.0, 1.0, -0.5 * thickness],
            [-1.0, 1.0, -0.5 * thickness],
            [-1.0, 0.0, -0.5 * thickness],
        ],
        dtype=np.float64,
    )
    top = bottom.copy()
    top[:, 2] = 0.5 * thickness
    return np.concatenate([bottom, top], axis=0)


def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    k = np.asarray([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]], dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    return eye + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def rigid_q(x16: np.ndarray, *, rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    r = np.eye(3, dtype=np.float64) if rotation is None else np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    t = np.zeros(3, dtype=np.float64) if translation is None else np.asarray(translation, dtype=np.float64).reshape(3)
    return ((r @ x16.T).T + t.reshape(1, 3) - x16).reshape(48)


def test_pure_translation_def_is_zero() -> None:
    x16 = flat_macro16_x16()
    q_raw = np.broadcast_to(np.asarray([0.3, -0.2, 0.15], dtype=np.float64), (16, 3)).reshape(48)
    q_def, q_rigid, r, t = remove_rigid_motion(x16, q_raw)
    assert q_def.shape == (48,)
    assert q_rigid.shape == (48,)
    assert np.allclose(q_def, 0.0, atol=1.0e-12)
    assert np.allclose(q_rigid, q_raw, atol=1.0e-12)
    assert np.allclose(r.T @ r, np.eye(3), atol=1.0e-12)
    assert np.linalg.det(r) > 0.0
    assert np.allclose(t, [0.3, -0.2, 0.15], atol=1.0e-12)


def test_pure_small_rotation_def_is_zero() -> None:
    x16 = flat_macro16_x16()
    r = rodrigues(np.asarray([0.2, -0.3, 1.0], dtype=np.float64), 2.0e-3)
    q_raw = rigid_q(x16, rotation=r)
    q_def, q_rigid, r_fit, _t = remove_rigid_motion(x16, q_raw)
    assert q_def.shape == (48,)
    assert np.allclose(q_def, 0.0, atol=1.0e-12)
    assert np.allclose(q_rigid, q_raw, atol=1.0e-12)
    assert np.allclose(r_fit.T @ r_fit, np.eye(3), atol=1.0e-12)
    assert np.linalg.det(r_fit) > 0.0


def test_rigid_plus_deformation_closes_raw_split() -> None:
    x16 = flat_macro16_x16()
    r = rodrigues(np.asarray([0.0, 0.0, 1.0], dtype=np.float64), 1.0e-3)
    q_rigid_in = rigid_q(x16, rotation=r, translation=np.asarray([0.1, 0.02, -0.04]))
    deformation = np.zeros((16, 3), dtype=np.float64)
    deformation[:, 2] = np.linspace(-0.01, 0.01, 16)
    q_raw = q_rigid_in + deformation.reshape(48)
    q_def, q_rigid, _r, _t = remove_rigid_motion(x16, q_raw)
    assert q_def.shape == (48,)
    assert q_rigid.shape == (48,)
    assert np.allclose(q_rigid + q_def, q_raw, atol=1.0e-12)
    assert np.linalg.norm(q_def) > 0.0


def test_batch_outputs_remain_48d_and_projection_is_not_42d() -> None:
    x16 = flat_macro16_x16()
    q = np.zeros((2, 48), dtype=np.float64)
    q[0] = np.broadcast_to(np.asarray([0.01, 0.02, 0.03]), (16, 3)).reshape(48)
    q[1, 5] = 0.02
    q_def, q_rigid, rotations, translations, projectors = rigid_preprocess_batch(
        np.broadcast_to(x16.reshape(1, 16, 3), (2, 16, 3)),
        q,
    )
    assert q_def.shape == (2, 48)
    assert q_rigid.shape == (2, 48)
    assert rotations.shape == (2, 3, 3)
    assert translations.shape == (2, 3)
    assert projectors.shape == (2, 48, 48)
    assert rigid_projection_matrix(x16).shape == (48, 48)
