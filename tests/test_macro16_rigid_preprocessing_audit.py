from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from audit_macro16_rigid_preprocessing import audit_synthetic_rigid, run_audit
from macro_deeponet.macro16_geometry import (
    MACRO16_CONTRACT_VERSION,
    Macro16GeometryMap,
    macro16_source128_point_table,
    normalize_macro16_x16_batch,
)
from macro_deeponet.macro16_rigid import rigid_preprocess_batch


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


def write_new_contract_compact(path: Path) -> None:
    n = 2
    point_table = macro16_source128_point_table()
    p = point_table.xi.shape[0]
    x16 = np.broadcast_to(flat_macro16_x16().reshape(1, 16, 3), (n, 16, 3)).copy()
    x16[1, :, 0] += 0.08 * x16[1, :, 1]
    x16_hat, x_center, l_ref = normalize_macro16_x16_batch(x16)
    q48 = np.zeros((n, 48), dtype=np.float64)
    q48[0] = np.broadcast_to(np.asarray([0.02, -0.01, 0.005], dtype=np.float64), (16, 3)).reshape(48)
    r = rodrigues(np.asarray([0.2, -0.1, 1.0], dtype=np.float64), 1.0e-3)
    q48[1] = ((r @ x16[1].T).T - x16[1]).reshape(48)
    q48[1, 2::3] += np.linspace(-0.002, 0.002, 16)
    q48_def, q48_rigid, rotations, translations, projectors = rigid_preprocess_batch(x16, q48)

    weights_hat = np.empty((n, p), dtype=np.float64)
    for i in range(n):
        fields = Macro16GeometryMap(x16[i]).eval_points(point_table)
        weights_hat[i] = np.asarray(fields["integration_weight_hat"], dtype=np.float64)
    b_qraw = np.zeros((n, p, 6, 48), dtype=np.float64)
    b_qraw[:, :, 0, 0] = 1.25
    b_qraw[:, :, 1, 1] = np.linspace(0.1, 0.2, p).reshape(1, p)
    b_qraw[:, :, 2, 2] = 0.5
    b_qdef_raw = np.einsum("npak,nkj->npaj", b_qraw, projectors)

    np.savez(
        path,
        standard_operator_contract_version=np.asarray(MACRO16_CONTRACT_VERSION, dtype=object),
        macro16_teacher_contract_version=np.asarray("macro16-source128-teacher-rigid-preprocess-test", dtype=object),
        X16_raw=x16,
        X_center=x_center,
        L_ref=l_ref,
        X16_hat=x16_hat,
        q48_raw=q48,
        q48_hat=q48 / l_ref,
        q48_rigid_raw=q48_rigid,
        q48_def_raw=q48_def,
        q48_def_hat=q48_def / l_ref,
        rigid_rotation_R=rotations,
        rigid_translation_t=translations,
        rigid_projection_P=projectors,
        LE_macro=np.zeros((n, p, 6), dtype=np.float64),
        B_macro_qraw=b_qraw,
        B_macro_qhat=b_qraw * l_ref.reshape(n, 1, 1, 1),
        B_macro_qdef_raw=b_qdef_raw,
        B_macro_qdef=b_qdef_raw * l_ref.reshape(n, 1, 1, 1),
        integration_weight_hat=weights_hat,
        integration_weight_phys=weights_hat * (l_ref**3),
        macro16_point_xi=point_table.xi,
        case_id=np.asarray([9, 9], dtype=np.int64),
    )


def test_synthetic_rigid_audit_keeps_48d_and_removes_pure_rigid_motion() -> None:
    audit = audit_synthetic_rigid(1.0e-10)
    assert audit["passed"] is True
    for row in audit["cases"]:
        assert row["output_remains_48d"] is True
        assert row["q48_def_raw_shape"] == [48]
        if row["pure_rigid_expected_zero_def"]:
            assert row["q_def_max_abs"] < 1.0e-10


def test_rigid_preprocessing_audit_verifies_loader_uses_def_hat_and_qdef(tmp_path: Path) -> None:
    compact = tmp_path / "macro16_source128_new_contract.npz"
    out = tmp_path / "rigid_audit.json"
    write_new_contract_compact(compact)
    args = SimpleNamespace(
        compact=[compact],
        compact_list=[],
        compact_glob=[],
        out=out,
        case_limit=0,
        frame_stride=1,
        max_frames_per_compact=0,
        skip_synthetic=False,
        synthetic_tol=1.0e-10,
        max_scale_rel=1.0e-10,
        max_scale_abs=1.0e-10,
        max_projection_rel=1.0e-10,
        max_projection_abs=1.0e-10,
        max_rigid_closure_rel=1.0e-10,
        max_rigid_closure_abs=1.0e-10,
        max_rotation_matrix_orthogonality=1.0e-10,
        max_qdef_translation_orthogonality=0.0,
        max_qdef_rotation_orthogonality=0.0,
        strict=True,
    )
    summary = run_audit(args)
    assert summary["strict_pass"] is True
    row = summary["compacts"][0]
    assert row["loader"]["model_visible_q"] == "q48_def_hat"
    assert row["loader"]["model_visible_B"] == "B_macro_qdef"
    assert row["loader"]["q48_hat_shape_loaded_for_model"] == [2, 48]
    assert row["dimension_checks"]["q_input_remains_48d"] is True
    assert row["dimension_checks"]["no_42d_input"] is True
