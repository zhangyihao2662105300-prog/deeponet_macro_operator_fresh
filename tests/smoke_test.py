from __future__ import annotations

import json
import math
import sys
import tempfile
import warnings
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from macro_deeponet.autograd import strain_jacobian_wrt_q
from macro_deeponet.data import SyntheticConfig, SyntheticMacroDataset
from macro_deeponet.geometry import (
    HEX8_NATURAL,
    det_jacobian_hex8,
    shape_function_gradients_hex8,
    shape_functions_hex8,
)
from macro_deeponet.models import (
    FELinearResidualDeepONet,
    MacroDeepONet,
    NOEMStyleMIONet,
    QueryFELinearResidualDeepONet,
    True176Shape4QrawDeepONet,
)
from macro_deeponet.point_features import load_point_features_from_compacts, transform_point_features_for_scale
from macro_deeponet.true176_data import (
    build_branch_features,
    build_keep_node_coords_unique,
    build_point_features,
    keep_node_ids,
    load_compacts,
    load_one_compact,
    split_indices,
    split_indices_with_meta,
    transform_b_target_for_q_coordinate,
)
from macro_deeponet.train_true176_deeponet_sobolev import ad_jacobian
from macro_deeponet.train_true176_generic_sobolev import (
    train as train_true176_generic,
    validate_point_feature_source_for_scale,
)

ROOT = Path(__file__).resolve().parents[1]
sys_path_text = str(ROOT / "scripts")
if sys_path_text not in sys.path:
    sys.path.insert(0, sys_path_text)
from validate_isoparametric_scaling import run_validation as run_isoparametric_scaling_validation
from validate_isoparametric_mapping import run_mapping_validation
from validate_true176_css8_macro_mapping import run_css8_macro_validation
from export_abaqus_true176_complete_compact import enforce_ip_audit, merge_existing_payload


def test_geometry_identities() -> None:
    points = torch.rand(32, 3) * 2.0 - 1.0
    n = shape_functions_hex8(points)
    dndxi = shape_function_gradients_hex8(points)
    assert torch.allclose(n.sum(dim=-1), torch.ones(32), atol=1.0e-6)
    assert torch.allclose(dndxi.sum(dim=-2), torch.zeros(32, 3), atol=1.0e-6)

    nodes = 0.5 * HEX8_NATURAL.unsqueeze(0).expand(32, -1, -1)
    detj = det_jacobian_hex8(nodes, points)
    assert torch.all(detj > 0.0)
    assert torch.allclose(detj, torch.full_like(detj, 0.125), atol=1.0e-6)


def test_synthetic_rigid_targets() -> None:
    dataset = SyntheticMacroDataset(
        SyntheticConfig(num_samples=128, seed=7, rigid_probability=1.0)
    )
    max_abs = dataset.epsilon.abs().max().item()
    assert max_abs < 1.0e-6, max_abs


def test_forward_and_autograd_shapes() -> None:
    dataset = SyntheticMacroDataset(SyntheticConfig(num_samples=16, seed=11))
    model = MacroDeepONet(hidden_dim=32, basis_dim=16, depth=3)
    q = dataset.q_b[:8]
    x = dataset.x_nodes[:8]
    xi = dataset.xi[:8]
    eps = model(q, x, xi)
    assert eps.shape == (8, 6)
    eps2, b_macro = strain_jacobian_wrt_q(model, q, x, xi)
    assert eps2.shape == (8, 6)
    assert b_macro.shape == (8, 6, 24)
    assert torch.isfinite(b_macro).all()


def test_true176_point_features_and_ad_shapes() -> None:
    shape4 = torch.tensor([[1.2, 0.05, 0.10, 0.00], [1.4, 0.04, 0.08, 0.20]], dtype=torch.float32).numpy()
    point, meta = build_point_features(shape4, include_id_features=True)
    assert point.shape[0] == 2
    assert point.shape[1] == 128
    assert point.shape[2] == meta["feature_dim"]
    assert point.shape[2] > 40

    model = True176Shape4QrawDeepONet(
        point_dim=point.shape[-1],
        ip_count=128,
        basis_dim=12,
        hidden_dim=32,
        branch_depth=2,
        trunk_depth=2,
    )
    x_norm = torch.randn(2, 52)
    p_norm = torch.as_tensor(point, dtype=torch.float32)
    le = model(x_norm, p_norm)
    assert le.shape == (2, 128, 6)
    j = ad_jacobian(model, x_norm, p_norm, [0, 3, 7], create_graph=False, method="forward")
    assert j.shape == (2, 128, 6, 3)
    assert torch.isfinite(j).all()


def test_true176_xkeep_branch_and_ad_shapes() -> None:
    shape4 = np.asarray([[1.2, 0.05, 0.10, 0.00], [1.4, 0.04, 0.08, 0.20]], dtype=np.float32)
    q48 = np.zeros((2, 48), dtype=np.float32)
    branch, branch_meta = build_branch_features(shape4=shape4, q48_raw=q48, mode="xkeep-qraw")
    point, _meta = build_point_features(shape4, include_id_features=False)
    assert branch.shape == (2, 96)
    assert branch_meta["q_start"] == 0
    assert branch_meta["geometry_input"] == "X_keep[16,3]"
    assert branch_meta["visibility"] == "macro-element branch sees only q48 control-node coordinates"

    model = True176Shape4QrawDeepONet(
        input_dim=branch.shape[-1],
        point_dim=point.shape[-1],
        ip_count=128,
        basis_dim=12,
        hidden_dim=32,
        branch_depth=2,
        trunk_depth=2,
        q_start=0,
        q_dim=48,
    )
    x_norm = torch.randn(2, branch.shape[-1])
    p_norm = torch.as_tensor(point, dtype=torch.float32)
    le = model(x_norm, p_norm)
    assert le.shape == (2, 128, 6)
    j = ad_jacobian(model, x_norm, p_norm, [0, 3, 7], create_graph=False, method="forward")
    assert j.shape == (2, 128, 6, 3)
    assert torch.isfinite(j).all()


def test_noem_style_mionet_splits_q_and_geometry() -> None:
    shape4 = np.asarray([[1.2, 0.05, 0.10, 0.00], [1.4, 0.04, 0.08, 0.20]], dtype=np.float32)
    q48 = np.zeros((2, 48), dtype=np.float32)
    branch, branch_meta = build_branch_features(shape4=shape4, q48_raw=q48, mode="xkeep-qraw")
    point, _meta = build_point_features(shape4, include_id_features=False)
    model = NOEMStyleMIONet(
        input_dim=branch.shape[-1],
        point_dim=point.shape[-1],
        ip_count=128,
        basis_dim=12,
        hidden_dim=32,
        branch_depth=2,
        trunk_depth=2,
        q_start=int(branch_meta["q_start"]),
        q_dim=int(branch_meta["q_dim"]),
    )
    assert model.geometry_dim == 48
    named = dict(model.named_parameters())
    assert any(k.startswith("q_branch.") for k in named)
    assert any(k.startswith("geometry_branch.") for k in named)
    assert any(k.startswith("trunk.") for k in named)
    assert "skip_weight" not in named
    assert model.product_scale == "none"

    x_norm = torch.randn(2, branch.shape[-1])
    p_norm = torch.as_tensor(point, dtype=torch.float32)
    le = model(x_norm, p_norm)
    assert le.shape == (2, 128, 6)
    j = ad_jacobian(model, x_norm, p_norm, [0, 3, 7], create_graph=False, method="forward")
    assert j.shape == (2, 128, 6, 3)
    assert torch.isfinite(j).all()


def test_fe_linear_residual_model_has_explicit_b_baseline() -> None:
    shape4 = np.asarray([[1.2, 0.05, 0.10, 0.00], [1.4, 0.04, 0.08, 0.20]], dtype=np.float32)
    q48 = np.zeros((2, 48), dtype=np.float32)
    branch, branch_meta = build_branch_features(shape4=shape4, q48_raw=q48, mode="xkeep-qraw")
    point, _meta = build_point_features(shape4, include_id_features=False)
    q_start = int(branch_meta["q_start"])
    q_dim = int(branch_meta["q_dim"])
    skip = torch.zeros((128, 6, q_dim), dtype=torch.float32)
    skip[:, :, 0] = 0.25
    model = FELinearResidualDeepONet(
        input_dim=branch.shape[-1],
        point_dim=point.shape[-1],
        ip_count=128,
        basis_dim=12,
        hidden_dim=32,
        branch_depth=2,
        trunk_depth=2,
        q_start=q_start,
        q_dim=q_dim,
        skip_init=skip,
        train_skip=False,
        residual_scale=0.0,
        train_point_baseline=False,
    )
    assert model.zero_init_residual
    x_norm = torch.zeros(2, branch.shape[-1])
    x_norm[:, q_start] = torch.tensor([2.0, -1.0])
    p_norm = torch.as_tensor(point, dtype=torch.float32)
    le = model(x_norm, p_norm)
    assert le.shape == (2, 128, 6)
    assert torch.allclose(le[0], torch.full((128, 6), 0.5), atol=1.0e-6)
    assert torch.allclose(le[1], torch.full((128, 6), -0.25), atol=1.0e-6)
    j = ad_jacobian(model, x_norm, p_norm, [0], create_graph=False, method="forward")
    assert j.shape == (2, 128, 6, 1)
    assert torch.allclose(j, torch.full_like(j, 0.25), atol=1.0e-6)


def test_query_fe_linear_residual_model_supports_dynamic_points() -> None:
    shape4 = np.asarray([[1.2, 0.05, 0.10, 0.00], [1.4, 0.04, 0.08, 0.20]], dtype=np.float32)
    q48 = np.zeros((2, 48), dtype=np.float32)
    branch, branch_meta = build_branch_features(shape4=shape4, q48_raw=q48, mode="xkeep-qraw")
    point, _meta = build_point_features(shape4, include_id_features=False)
    q_start = int(branch_meta["q_start"])
    q_dim = int(branch_meta["q_dim"])
    skip = torch.zeros((6, q_dim), dtype=torch.float32)
    skip[:, 0] = 0.25
    model = QueryFELinearResidualDeepONet(
        input_dim=branch.shape[-1],
        point_dim=point.shape[-1],
        ip_count=128,
        basis_dim=12,
        hidden_dim=32,
        branch_depth=2,
        trunk_depth=2,
        q_start=q_start,
        q_dim=q_dim,
        skip_init=skip,
        train_skip=False,
        residual_scale=0.0,
        train_point_baseline=False,
    )
    assert model.supports_dynamic_points
    x_norm = torch.zeros(2, branch.shape[-1])
    x_norm[:, q_start] = torch.tensor([2.0, -1.0])
    p_full = torch.as_tensor(point, dtype=torch.float32)
    le_full = model(x_norm, p_full)
    assert le_full.shape == (2, 128, 6)
    assert torch.allclose(le_full[0], torch.full((128, 6), 0.5), atol=1.0e-6)
    assert torch.allclose(le_full[1], torch.full((128, 6), -0.25), atol=1.0e-6)

    subset = [0, 5, 17, 63, 127]
    p_query = p_full[:, subset, :]
    le_query = model(x_norm, p_query)
    assert le_query.shape == (2, len(subset), 6)
    assert torch.allclose(le_query, le_full[:, subset, :], atol=1.0e-6)
    b_query = model._linear_b_norm(p_query)
    assert b_query.shape == (2, len(subset), 6, q_dim)
    j = ad_jacobian(model, x_norm, p_query, [0, 3, 7], create_graph=False, method="forward")
    assert j.shape == (2, len(subset), 6, 3)
    assert torch.allclose(j[:, :, :, 0], torch.full((2, len(subset), 6), 0.25), atol=1.0e-6)


def test_true176_keep_node_order_matches_q48_contract() -> None:
    expected = np.asarray([1, 3, 5, 11, 15, 21, 23, 25, 26, 28, 30, 36, 40, 46, 48, 50], dtype=np.int64)
    assert np.array_equal(keep_node_ids(), expected)

    shape4 = np.asarray([[1.0, 0.01, 0.0, 0.0]], dtype=np.float32)
    x_keep, meta = build_keep_node_coords_unique(shape4)
    assert x_keep.shape == (1, 16, 3)
    assert np.array_equal(meta["keep_node_ids"], expected)


def test_true176_loader_uses_explicit_xkeep_when_available() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "xkeep_compact.npz"
        n = 2
        shape4 = np.zeros((n, 4), dtype=np.float32)
        q48 = np.zeros((n, 48), dtype=np.float32)
        x_keep = np.arange(16 * 3, dtype=np.float32).reshape(16, 3)
        np.savez(
            path,
            shape4=shape4,
            q48_raw=q48,
            LE128_base=np.zeros((n, 128, 6), dtype=np.float32),
            B_LE128_forward=np.zeros((n, 128, 6, 48), dtype=np.float32),
            X_keep_ref=x_keep,
            sample_paths=np.asarray(["case001"], dtype=str),
        )
        data = load_compacts([str(path)])
        branch, branch_meta = build_branch_features(
            shape4=data.shape4,
            q48_raw=data.q48_raw,
            mode="xkeep-qraw",
            keep_node_coords=data.keep_node_coords,
        )
        assert branch.shape == (n, 96)
        assert branch_meta["macro_geometry_source"] == "compact_X_keep"
        assert np.allclose(branch[:, 48:].reshape(n, 16, 3), x_keep.reshape(1, 16, 3))


def test_true176_physical_scale_transforms_branch_and_b() -> None:
    shape4 = np.zeros((2, 4), dtype=np.float32)
    q48 = np.full((2, 48), 2.0, dtype=np.float32)
    x_keep = np.full((2, 16, 3), 4.0, dtype=np.float32)
    h = np.asarray([[2.0], [4.0]], dtype=np.float32)

    branch, branch_meta = build_branch_features(
        shape4=shape4,
        q48_raw=q48,
        mode="xkeep-qraw",
        keep_node_coords=x_keep,
        length_scale=h,
        scale_mode="physical",
    )
    assert branch_meta["scale_mode"] == "physical"
    assert branch_meta["q_coordinate"] == "q_hat=q48_raw/H"
    assert np.allclose(branch[0, :48], 1.0)
    assert np.allclose(branch[1, :48], 0.5)
    assert np.allclose(branch[0, 48:].reshape(16, 3), 2.0)
    assert np.allclose(branch[1, 48:].reshape(16, 3), 1.0)

    b_phys = np.full((2, 3, 6, 48), 3.0, dtype=np.float32)
    b_hat, meta = transform_b_target_for_q_coordinate(
        b_phys,
        h,
        scale_mode="physical",
        b_label_coordinate="physical",
    )
    assert meta["b_target_transform"] == "H * B_phys"
    assert np.allclose(b_hat[0], 6.0)
    assert np.allclose(b_hat[1], 12.0)


def test_isoparametric_similarity_scaling_laws() -> None:
    report = run_isoparametric_scaling_validation(seed=1234, trials=4, points_per_trial=3)
    assert report["passed"], report


def test_isoparametric_mapping_laws() -> None:
    report = run_mapping_validation(seed=4321, trials=4, points_per_trial=3)
    assert report["passed"], report


def test_true176_css8_macro_mapping_laws() -> None:
    report = run_css8_macro_validation(seed=2468, shapes=2, rows_per_shape=16)
    assert report["passed"], report


def test_true176_physical_scale_rejects_shape4_branch_geometry() -> None:
    shape4 = np.zeros((1, 4), dtype=np.float32)
    q48 = np.zeros((1, 48), dtype=np.float32)
    h = np.asarray([[2.0]], dtype=np.float32)

    for mode in ("shape4-qraw", "xkeep-qraw", "xnodes-qraw"):
        try:
            build_branch_features(
                shape4=shape4,
                q48_raw=q48,
                mode=mode,
                length_scale=h,
                scale_mode="physical",
            )
        except ValueError as exc:
            assert "shape4" in str(exc) or "explicit physical" in str(exc)
        else:
            raise AssertionError(f"physical scale accepted shape4-only geometry for {mode}")


def test_true176_physical_scale_rejects_shape4_trunk_by_default() -> None:
    point_meta = {"point_feature_source": "shape4_reconstructed_audited"}
    try:
        validate_point_feature_source_for_scale(
            scale_mode="physical",
            requested_source="shape4-audited",
            point_meta=point_meta,
        )
    except ValueError as exc:
        assert "cannot use shape4-audited" in str(exc)
    else:
        raise AssertionError("physical scale accepted shape4-audited Trunk without opt-in")

    validate_point_feature_source_for_scale(
        scale_mode="physical",
        requested_source="shape4-audited",
        point_meta=point_meta,
        allow_physical_shape4_trunk=True,
    )
    validate_point_feature_source_for_scale(
        scale_mode="physical",
        requested_source="data",
        point_meta={"point_feature_source": "data_generic"},
    )


def test_true176_physical_scale_transforms_raw_point_fields() -> None:
    point = np.asarray([[[2.0, 4.0, 0.25, 8.0]]], dtype=np.float32)
    names = ["ip_xyz_x", "ip_J_00", "ip_invJ_00", "ip_detJ"]
    scaled, meta = transform_point_features_for_scale(
        point,
        names,
        np.asarray([[2.0]], dtype=np.float32),
        scale_mode="physical",
        detj_scale_dim=3,
    )
    assert meta["transformed_feature_count"] == 4
    assert np.allclose(scaled[0, 0], [1.0, 2.0, 0.5, 1.0])


def test_true176_physical_scale_rejects_unnamed_point_features() -> None:
    point = np.asarray([[[1.0, 2.0]]], dtype=np.float32)
    names = ["point_features_0", "point_features_1"]
    try:
        transform_point_features_for_scale(
            point,
            names,
            np.asarray([[2.0]], dtype=np.float32),
            scale_mode="physical",
        )
    except ValueError as exc:
        assert "length units are unknown" in str(exc)
    else:
        raise AssertionError("physical scale accepted anonymous point features")

    hat, meta = transform_point_features_for_scale(
        point,
        ["X_hat_exact", "detJ_hat"],
        np.asarray([[2.0]], dtype=np.float32),
        scale_mode="physical",
    )
    assert meta["transformed_feature_count"] == 0
    assert meta["dimensionless_feature_count"] == 2
    assert np.allclose(hat, point)


def test_true176_full_xnodes_branch_is_available_as_ablation() -> None:
    shape4 = np.asarray([[1.2, 0.05, 0.10, 0.00]], dtype=np.float32)
    q48 = np.zeros((1, 48), dtype=np.float32)
    branch, branch_meta = build_branch_features(shape4=shape4, q48_raw=q48, mode="xnodes-qraw")
    assert branch.shape == (1, 198)
    assert branch_meta["geometry_input"] == "X_macro[50,3]"


def test_true176_loader_rejects_non_full48_b() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad_compact.npz"
        np.savez(
            path,
            shape4=np.zeros((2, 4), dtype=np.float32),
            q48_raw=np.zeros((2, 48), dtype=np.float32),
            LE128_base=np.zeros((2, 128, 6), dtype=np.float32),
            B_LE128_forward=np.zeros((2, 128, 6, 4), dtype=np.float32),
            sample_paths=np.asarray(["case001"], dtype=str),
        )
        try:
            load_one_compact(path)
        except ValueError as exc:
            assert "full q48/B48 compact data" in str(exc)
        else:
            raise AssertionError("loader accepted a compact file with B last dimension != 48")


def test_true176_loader_broadcasts_constant_shape4_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "constant_shape4_compact.npz"
        n = 3
        shape4 = np.asarray([1.0, 0.01, 0.0, 0.0], dtype=np.float32)
        np.savez(
            path,
            shape4=shape4,
            q48_raw=np.zeros((n, 48), dtype=np.float32),
            LE128_base=np.zeros((n, 128, 6), dtype=np.float32),
            B_LE128_forward=np.zeros((n, 128, 6, 48), dtype=np.float32),
            sample_paths=np.asarray(["case001"], dtype=str),
        )
        data = load_one_compact(path)
        assert data["shape4"].shape == (n, 4)
        assert np.allclose(data["shape4"], shape4.reshape(1, 4))


def test_true176_physical_train_requires_explicit_h() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        compact = tmp_path / "no_h_compact.npz"
        compact_list = tmp_path / "compact_paths.txt"
        n = 1
        np.savez(
            compact,
            shape4=np.zeros((n, 4), dtype=np.float32),
            q48_raw=np.zeros((n, 48), dtype=np.float32),
            LE128_base=np.zeros((n, 128, 6), dtype=np.float32),
            B_LE128_forward=np.zeros((n, 128, 6, 48), dtype=np.float32),
            X_keep=np.zeros((n, 16, 3), dtype=np.float32),
            ip_xi=np.zeros((n, 128, 3), dtype=np.float32),
        )
        compact_list.write_text(str(compact), encoding="utf-8")
        args = type(
            "Args",
            (),
            {
                "ddp": False,
                "cuda": False,
                "ddp_backend": "gloo",
                "seed": 1,
                "out_dir": tmp_path / "out",
                "compact": [],
                "compact_list": str(compact_list),
                "target_ips": "0",
                "scale_mode": "physical",
                "frame_stride": 1,
                "max_frames_per_compact": 0,
                "val_fraction": 0.0,
                "val_cases": "",
            },
        )()
        try:
            train_true176_generic(args)
        except ValueError as exc:
            assert "requires an explicit H" in str(exc)
        else:
            raise AssertionError("physical training accepted implicit H=1 compact")


def test_strict_split_uses_case_or_geometry_without_overlap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "split_compact.npz"
        n = 6
        sample_paths = np.asarray(["case001_frame", "case002_frame", "case003_frame"], dtype=str)
        shape4 = np.asarray(
            [[1.0, 0.01, 0.0, 0.0], [1.0, 0.01, 0.0, 0.0],
             [1.1, 0.01, 0.0, 0.0], [1.1, 0.01, 0.0, 0.0],
             [1.2, 0.01, 0.0, 0.0], [1.2, 0.01, 0.0, 0.0]],
            dtype=np.float32,
        )
        np.savez(
            path,
            shape4=shape4,
            q48_raw=np.zeros((n, 48), dtype=np.float32),
            LE128_base=np.zeros((n, 128, 6), dtype=np.float32),
            B_LE128_forward=np.zeros((n, 128, 6, 48), dtype=np.float32),
            sample_paths=sample_paths,
        )
        data = load_compacts([str(path)])
        train, val, meta = split_indices_with_meta(data, 0.34, 3, "", split_mode="case")
        assert not set(train.tolist()).intersection(set(val.tolist()))
        assert meta["validation_split_mode"] == "case"
        assert not set(meta["train_cases"]).intersection(set(meta["val_cases"]))

        train_g, val_g, meta_g = split_indices_with_meta(data, 0.34, 3, "", split_mode="geometry")
        assert not set(train_g.tolist()).intersection(set(val_g.tolist()))
        assert meta_g["validation_split_mode"] == "geometry"
        assert not set(meta_g["train_shape_indices"]).intersection(set(meta_g["val_shape_indices"]))

        try:
            split_indices_with_meta(data, 0.0, 3, "", split_mode="frame")
        except ValueError as exc:
            assert "requires --val-fraction > 0" in str(exc)
        else:
            raise AssertionError("frame split accepted val_fraction=0")

        try:
            split_indices_with_meta(data, 0.0, 3, "", split_mode="overlap-debug")
        except ValueError as exc:
            assert "--allow-overlap-val" in str(exc)
        else:
            raise AssertionError("overlap split did not require explicit debug flag")


def test_legacy_split_indices_warns_about_overlap_behavior() -> None:
    data = SimpleNamespace(
        shape4=np.zeros((5, 4), dtype=np.float32),
        case_id=np.arange(5, dtype=np.int64),
        shape_index=np.arange(5, dtype=np.int64),
    )
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        train, val = split_indices(data, 0.0, 11, "")
    assert train.size == 5
    assert val.size == 1
    assert np.intersect1d(train, val).size == 1
    assert any("split_indices() is legacy/debug behavior" in str(w.message) for w in seen)


def test_generic_point_feature_loader_reads_real_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "real_points_compact.npz"
        n = 3
        ip_xyz = np.stack(
            [
                np.linspace(-1.0, 1.0, 128, dtype=np.float32),
                np.linspace(0.0, 2.0, 128, dtype=np.float32),
                np.linspace(2.0, 3.0, 128, dtype=np.float32),
            ],
            axis=1,
        )
        ip_detj = np.linspace(0.1, 0.2, 128, dtype=np.float32)
        np.savez(
            path,
            shape4=np.zeros((n, 4), dtype=np.float32),
            q48_raw=np.zeros((n, 48), dtype=np.float32),
            LE128_base=np.zeros((n, 128, 6), dtype=np.float32),
            B_LE128_forward=np.zeros((n, 128, 6, 48), dtype=np.float32),
            ip_xyz=ip_xyz,
            ip_detJ=ip_detj,
        )
        points, meta = load_point_features_from_compacts(
            compact_paths=[str(path)],
            source_index=np.zeros(n, dtype=np.int64),
            source_row=np.arange(n, dtype=np.int64),
            shape4=np.zeros((n, 4), dtype=np.float32),
            target_ips=[0, 7, 127],
            source="data",
            include_id_features=False,
            allow_shape4_fallback=False,
        )
        assert points.shape == (n, 3, 4)
        assert meta["point_feature_source"] == "data_generic"
        assert meta["feature_names"] == ["ip_xyz_x", "ip_xyz_y", "ip_xyz_z", "ip_detJ"]
        assert np.allclose(points[:, 0, :3], ip_xyz[0])
        assert np.allclose(points[:, -1, 3], ip_detj[127])


def test_generic_point_feature_loader_rejects_name_order_mismatch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path_a = Path(tmp) / "points_a.npz"
        path_b = Path(tmp) / "points_b.npz"
        n = 1
        common = dict(
            shape4=np.zeros((n, 4), dtype=np.float32),
            q48_raw=np.zeros((n, 48), dtype=np.float32),
            LE128_base=np.zeros((n, 128, 6), dtype=np.float32),
            B_LE128_forward=np.zeros((n, 128, 6, 48), dtype=np.float32),
            point_features=np.zeros((n, 128, 2), dtype=np.float32),
        )
        np.savez(path_a, **common, point_feature_names=np.asarray(["a", "b"], dtype=object))
        np.savez(path_b, **common, point_feature_names=np.asarray(["b", "a"], dtype=object))
        try:
            load_point_features_from_compacts(
                compact_paths=[str(path_a), str(path_b)],
                source_index=np.asarray([0, 1], dtype=np.int64),
                source_row=np.asarray([0, 0], dtype=np.int64),
                shape4=np.zeros((2, 4), dtype=np.float32),
                target_ips=[0],
                source="data",
                include_id_features=False,
                allow_shape4_fallback=False,
            )
        except ValueError as exc:
            assert "names/order mismatch" in str(exc)
        else:
            raise AssertionError("loader accepted mismatched point feature name order")


def test_generic_point_feature_loader_carries_ip_keys() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "real_points_with_keys_compact.npz"
        n = 2
        ip_xi = np.zeros((n, 128, 3), dtype=np.float32)
        ip_keys = np.stack(
            [
                np.arange(1, 129, dtype=np.int64),
                np.arange(1001, 1129, dtype=np.int64),
                np.arange(2001, 2129, dtype=np.int64),
            ],
            axis=1,
        )
        np.savez(
            path,
            shape4=np.zeros((n, 4), dtype=np.float32),
            q48_raw=np.zeros((n, 48), dtype=np.float32),
            LE128_base=np.zeros((n, 128, 6), dtype=np.float32),
            B_LE128_forward=np.zeros((n, 128, 6, 48), dtype=np.float32),
            ip_xi=ip_xi,
            ip_keys=ip_keys,
        )
        target_ips = [0, 7, 127]
        points, meta = load_point_features_from_compacts(
            compact_paths=[str(path)],
            source_index=np.zeros(n, dtype=np.int64),
            source_row=np.arange(n, dtype=np.int64),
            shape4=np.zeros((n, 4), dtype=np.float32),
            target_ips=target_ips,
            source="data",
            include_id_features=False,
            allow_shape4_fallback=False,
        )
        assert points.shape == (n, len(target_ips), 3)
        assert meta["point_feature_target_ips"] == target_ips
        assert meta["point_feature_ip_keys_source"] == "compact"
        assert meta["point_feature_ip_keys_repeated_for_frames"] is True
        assert meta["point_feature_ip_keys_frame_count"] == n
        got = np.asarray(meta["point_feature_ip_keys"], dtype=np.int64)
        assert got.shape == (len(target_ips), 3)
        assert np.array_equal(got, ip_keys[target_ips])


def test_generic_point_feature_loader_uses_standard_ip_keys_for_spatial_id_features() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "standard_keys_compact.npz"
        n = 1
        ip_xi = np.zeros((n, 128, 3), dtype=np.float32)
        ip_keys = np.asarray([[elem, ip, 0] for elem in range(1, 17) for ip in range(1, 9)], dtype=np.int64)
        np.savez(
            path,
            shape4=np.zeros((n, 4), dtype=np.float32),
            q48_raw=np.zeros((n, 48), dtype=np.float32),
            LE128_base=np.zeros((n, 128, 6), dtype=np.float32),
            B_LE128_forward=np.zeros((n, 128, 6, 48), dtype=np.float32),
            ip_xi=ip_xi,
            ip_keys=ip_keys,
        )
        points, meta = load_point_features_from_compacts(
            compact_paths=[str(path)],
            source_index=np.zeros(n, dtype=np.int64),
            source_row=np.arange(n, dtype=np.int64),
            shape4=np.zeros((n, 4), dtype=np.float32),
            target_ips=[17],
            source="data",
            include_id_features=True,
            allow_shape4_fallback=False,
        )
        expected = np.asarray([(3 - 8.5) / 7.5, (2 - 4.5) / 3.5, -1.0 + 2.0 * (2.0 + 0.5) / 4.0, -1.0 + 2.0 * (0.0 + 0.5) / 4.0], dtype=np.float32)
        assert points.shape == (n, 1, 7)
        assert np.allclose(points[0, 0, -4:], expected)
        assert meta["sources_used"][0]["id_feature_source"] == "ip_keys"
        assert meta["sources_used"][0]["id_feature_rule"] == "standard_true176_elem_label_4x4_spatial"


def test_generic_point_feature_loader_nonstandard_ip_keys_use_rank_id_guard() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nonstandard_keys_compact.npz"
        n = 1
        ip_xi = np.zeros((n, 128, 3), dtype=np.float32)
        labels = np.repeat(np.asarray([101, 205, 309, 412], dtype=np.int64), 32)
        ip_keys = np.stack([labels, np.tile(np.arange(1, 9, dtype=np.int64), 16), np.zeros(128, dtype=np.int64)], axis=1)
        np.savez(
            path,
            shape4=np.zeros((n, 4), dtype=np.float32),
            q48_raw=np.zeros((n, 48), dtype=np.float32),
            LE128_base=np.zeros((n, 128, 6), dtype=np.float32),
            B_LE128_forward=np.zeros((n, 128, 6, 48), dtype=np.float32),
            ip_xi=ip_xi,
            ip_keys=ip_keys,
        )
        points, meta = load_point_features_from_compacts(
            compact_paths=[str(path)],
            source_index=np.zeros(n, dtype=np.int64),
            source_row=np.arange(n, dtype=np.int64),
            shape4=np.zeros((n, 4), dtype=np.float32),
            target_ips=[0],
            source="data",
            include_id_features=True,
            allow_shape4_fallback=False,
        )
        assert points.shape == (n, 1, 5)
        assert meta["sources_used"][0]["id_feature_rule"] == "nonstandard_elem_label_rank_only_no_4x4_spatial"
        assert meta["sources_used"][0]["id_feature_guard_passed"] is False


def test_generic_complete_compact_can_omit_shape4_when_geometry_is_explicit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "complete_no_shape4_compact.npz"
        n = 2
        ip_xi = np.zeros((128, 3), dtype=np.float32)
        ip_xyz = np.zeros((128, 3), dtype=np.float32)
        ip_keys = np.stack(
            [
                np.arange(1, 129, dtype=np.int64),
                np.arange(1001, 1129, dtype=np.int64),
                np.zeros(128, dtype=np.int64),
            ],
            axis=1,
        )
        ip_keys_by_frame = np.stack([ip_keys, ip_keys + np.asarray([1000, 0, 0], dtype=np.int64)], axis=0)
        np.savez(
            path,
            q48_raw=np.zeros((n, 48), dtype=np.float32),
            LE128_base=np.zeros((n, 128, 6), dtype=np.float32),
            B_LE128_forward=np.zeros((n, 128, 6, 48), dtype=np.float32),
            X_keep=np.zeros((16, 3), dtype=np.float32),
            ip_xi=ip_xi,
            ip_xyz=ip_xyz,
            ip_keys=ip_keys_by_frame,
        )
        data = load_compacts([str(path)], target_ips=[0, 127])
        assert data.shape4.shape == (n, 4)
        assert np.allclose(data.shape4, 0.0)
        assert data.keep_node_coords is not None
        assert data.le.shape == (n, 2, 6)
        assert data.b.shape == (n, 2, 6, 48)
        points, meta = load_point_features_from_compacts(
            compact_paths=data.compact_paths,
            source_index=data.source_index,
            source_row=data.source_row,
            shape4=data.shape4,
            target_ips=[0, 127],
            source="data",
            include_id_features=False,
            allow_shape4_fallback=False,
        )
        assert points.shape == (n, 2, 6)
        assert meta["point_feature_ip_keys_repeated_for_frames"] is False
        assert np.asarray(meta["point_feature_ip_keys"]).shape == (n, 2, 3)


def test_exporter_merge_guards_q_le_and_ip_keys() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "merge_source.npz"
        n = 2
        p = 128
        ip_keys = np.asarray([[elem, ip, 0] for elem in range(1, 17) for ip in range(1, 9)], dtype=np.int64)
        payload = {
            "q48_raw": np.zeros((n, 48), dtype=np.float32),
            "LE128_base": np.zeros((n, p, 6), dtype=np.float32),
            "ip_keys": ip_keys,
        }
        np.savez(
            path,
            q48_raw=np.ones((n, 48), dtype=np.float32),
            LE128_base=np.zeros((n, p, 6), dtype=np.float32),
            B_LE128_forward=np.zeros((n, p, 6, 48), dtype=np.float32),
            ip_keys=ip_keys,
        )
        try:
            merge_existing_payload(dict(payload), str(path), require_b=True, merge_q_tol=1.0e-8)
        except ValueError as exc:
            assert "q48_raw" in str(exc)
        else:
            raise AssertionError("merge accepted mismatched q48")

        np.savez(
            path,
            q48_raw=np.zeros((n, 48), dtype=np.float32),
            LE128_base=np.ones((n, p, 6), dtype=np.float32),
            B_LE128_forward=np.zeros((n, p, 6, 48), dtype=np.float32),
            ip_keys=ip_keys,
        )
        try:
            merge_existing_payload(dict(payload), str(path), require_b=True, merge_le_tol=1.0e-8)
        except ValueError as exc:
            assert "LE128_base" in str(exc)
        else:
            raise AssertionError("merge accepted mismatched LE")

        bad_keys = ip_keys.copy()
        bad_keys[[0, 1]] = bad_keys[[1, 0]]
        np.savez(
            path,
            q48_raw=np.zeros((n, 48), dtype=np.float32),
            LE128_base=np.zeros((n, p, 6), dtype=np.float32),
            B_LE128_forward=np.zeros((n, p, 6, 48), dtype=np.float32),
            ip_keys=bad_keys,
        )
        try:
            merge_existing_payload(dict(payload), str(path), require_b=True, require_merge_ip_keys=True)
        except ValueError as exc:
            assert "ip_keys" in str(exc)
        else:
            raise AssertionError("merge accepted mismatched ip_keys")

        np.savez(
            path,
            q48_raw=np.zeros((n, 48), dtype=np.float32),
            LE128_base=np.zeros((n, p, 6), dtype=np.float32),
            B_LE128_forward=np.zeros((n, p, 6, 48), dtype=np.float32),
        )
        try:
            merge_existing_payload(dict(payload), str(path), require_b=True, require_merge_ip_keys=True)
        except ValueError as exc:
            assert "missing ip_keys" in str(exc)
        else:
            raise AssertionError("merge accepted missing ip_keys in strict mode")


def test_exporter_merge_rejects_strain_field_mismatch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "merge_source.npz"
        n = 2
        p = 128
        ip_keys = np.asarray([[elem, ip, 0] for elem in range(1, 17) for ip in range(1, 9)], dtype=np.int64)
        payload = {
            "q48_raw": np.zeros((n, 48), dtype=np.float32),
            "LE128_base": np.zeros((n, p, 6), dtype=np.float32),
            "ip_keys": ip_keys,
            "strain_field": np.asarray("E", dtype=object),
            "B_label_strain_field": np.asarray("E", dtype=object),
        }
        np.savez(
            path,
            q48_raw=np.zeros((n, 48), dtype=np.float32),
            LE128_base=np.zeros((n, p, 6), dtype=np.float32),
            B_LE128_forward=np.zeros((n, p, 6, 48), dtype=np.float32),
            ip_keys=ip_keys,
            strain_field=np.asarray("LE", dtype=object),
            B_label_strain_field=np.asarray("LE", dtype=object),
        )
        try:
            merge_existing_payload(dict(payload), str(path), require_b=True)
        except ValueError as exc:
            assert "strain_field" in str(exc)
        else:
            raise AssertionError("merge accepted mismatched strain_field")

        merged = merge_existing_payload(dict(payload), str(path), require_b=True, allow_merge_mismatch=True)
        assert bool(merged["merge_strain_field_match"]) is False


def test_load_compacts_rejects_mixed_strain_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        n = 2
        p = 128
        common = dict(
            shape4=np.zeros((n, 4), dtype=np.float32),
            q48_raw=np.zeros((n, 48), dtype=np.float32),
            LE128_base=np.zeros((n, p, 6), dtype=np.float32),
            B_LE128_forward=np.zeros((n, p, 6, 48), dtype=np.float32),
        )
        path_e = tmp_path / "strain_e.npz"
        path_le = tmp_path / "strain_le.npz"
        np.savez(path_e, **common, strain_field=np.asarray("E", dtype=object), B_label_strain_field=np.asarray("E", dtype=object))
        np.savez(path_le, **common, strain_field=np.asarray("LE", dtype=object), B_label_strain_field=np.asarray("LE", dtype=object))
        try:
            load_compacts([str(path_e), str(path_le)])
        except ValueError as exc:
            assert "mixed compact strain_field" in str(exc)
        else:
            raise AssertionError("loader accepted mixed strain fields")


def test_load_compacts_keeps_legacy_missing_strain_field_unknown() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy_missing_strain.npz"
        n = 2
        p = 128
        np.savez(
            path,
            shape4=np.zeros((n, 4), dtype=np.float32),
            q48_raw=np.zeros((n, 48), dtype=np.float32),
            LE128_base=np.zeros((n, p, 6), dtype=np.float32),
            B_LE128_forward=np.zeros((n, p, 6, 48), dtype=np.float32),
        )
        data = load_compacts([str(path)])
        assert data.strain_field == "unknown"
        assert data.b_label_strain_field == "unknown"
        assert data.strain_meta["missing_strain_field_count"] == 1


def test_exporter_ip_audit_can_block_bad_geometry() -> None:
    p = 128
    payload = {
        "ip_xyz": np.zeros((p, 3), dtype=np.float32),
        "ip_detJ": np.ones((p,), dtype=np.float32),
        "ip_xyz_abaqus_coord": np.full((1, p, 3), 1.0e-3, dtype=np.float32),
        "ip_IVOL_abaqus": np.ones((1, p), dtype=np.float32),
    }
    try:
        enforce_ip_audit(payload, require_ip_audit=True, ip_xyz_tol=1.0e-6, detj_ivol_tol=1.0e-8, skip_ivol_audit=False)
    except ValueError as exc:
        assert "COORD audit failed" in str(exc)
    else:
        raise AssertionError("IP audit accepted bad COORD")

    payload2 = {
        "ip_xyz": np.zeros((p, 3), dtype=np.float32),
        "ip_detJ": np.ones((p,), dtype=np.float32),
        "ip_xyz_abaqus_coord": np.zeros((1, p, 3), dtype=np.float32),
        "ip_IVOL_abaqus": np.full((1, p), 2.0, dtype=np.float32),
    }
    try:
        enforce_ip_audit(payload2, require_ip_audit=True, ip_xyz_tol=1.0e-6, detj_ivol_tol=1.0e-8, skip_ivol_audit=False)
    except ValueError as exc:
        assert "detJ-vs-IVOL" in str(exc)
    else:
        raise AssertionError("IP audit accepted bad IVOL")

    payload3 = {"ip_xyz": np.zeros((p, 3), dtype=np.float32), "ip_detJ": np.ones((p,), dtype=np.float32)}
    try:
        enforce_ip_audit(payload3, require_ip_audit=True, ip_xyz_tol=1.0e-6, detj_ivol_tol=1.0e-8, skip_ivol_audit=True)
    except ValueError as exc:
        assert "COORD" in str(exc)
    else:
        raise AssertionError("IP audit accepted missing COORD in strict mode")


def test_exporter_ip_audit_uses_reference_coord_not_selected_deformed_coord() -> None:
    p = 128
    payload = {
        "ip_xyz": np.zeros((p, 3), dtype=np.float32),
        "ip_detJ": np.ones((p,), dtype=np.float32),
        "ip_xyz_abaqus_coord": np.zeros((1, p, 3), dtype=np.float32),
        "ip_xyz_abaqus_coord_selected_frames": np.full((2, p, 3), 1.0e-2, dtype=np.float32),
        "audit_selected_frame_coord_vs_reference_max_abs": np.asarray(1.0e-2, dtype=np.float64),
        "ip_IVOL_abaqus": np.ones((1, p), dtype=np.float32),
    }
    out = enforce_ip_audit(
        payload,
        require_ip_audit=True,
        ip_xyz_tol=1.0e-6,
        detj_ivol_tol=1.0e-8,
        skip_ivol_audit=False,
    )
    assert float(out["audit_ref_ip_xyz_vs_abaqus_coord_max_abs"]) == 0.0
    assert float(out["audit_selected_frame_coord_vs_reference_max_abs"]) == 1.0e-2


def test_generic_query_training_supports_global_le_norm_and_point_sampling() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        compact = tmp_path / "query_train_compact.npz"
        out_dir = tmp_path / "out"
        n = 4
        rng = np.random.default_rng(123)
        q48 = rng.normal(size=(n, 48)).astype(np.float32)
        ip_xi = rng.normal(size=(128, 3)).astype(np.float32)
        b = rng.normal(scale=0.01, size=(n, 128, 6, 48)).astype(np.float32)
        le = np.einsum("npaj,nj->npa", b, q48).astype(np.float32)
        np.savez(
            compact,
            q48_raw=q48,
            LE128_base=le,
            B_LE128_forward=b,
            X_keep=np.zeros((n, 16, 3), dtype=np.float32),
            ip_xi=ip_xi,
        )
        args = SimpleNamespace(
            ddp=False,
            cuda=False,
            ddp_backend="gloo",
            seed=7,
            out_dir=out_dir,
            compact=[str(compact)],
            compact_list="",
            target_ips="0,1,2,3",
            scale_mode="normalized",
            frame_stride=1,
            max_frames_per_compact=0,
            val_fraction=0.25,
            val_cases="",
            split_mode="frame",
            allow_overlap_val=False,
            max_eval_frames=2,
            branch_feature_mode="xkeep-qraw",
            b_label_coordinate="auto",
            point_feature_source="data",
            include_id_features=False,
            allow_shape4_point_feature_fallback=False,
            allow_physical_shape4_trunk=False,
            detj_scale_dim=3,
            le_normalization="global-component",
            epochs=1,
            batch_size=2,
            eval_batch_size=1,
            basis_dim=8,
            hidden_dim=16,
            branch_depth=2,
            trunk_depth=2,
            activation="tanh",
            model_style="query-fe-linear-residual",
            mionet_product_scale="none",
            use_q_skip=False,
            residual_scale=1.0,
            fe_baseline_scale=1.0,
            freeze_fe_point_baseline=False,
            no_zero_init_residual=False,
            baseline_jacobian_weight=0.0,
            baseline_j_loss_mode="norm-plus-physical",
            b_baseline_warmstart_steps=0,
            b_baseline_warmstart_lr=1.0e-3,
            b_baseline_warmstart_weight_decay=0.0,
            b_baseline_warmstart_source="train-only",
            jacobian_columns="0,1",
            jacobian_columns_per_batch=1,
            jacobian_method="forward",
            eval_columns="0,1",
            j_loss_mode="norm",
            physical_j_aux_weight=0.0,
            physical_j_abs_weight=1.0,
            physical_j_rel_weight=0.0,
            physical_j_action_weight=0.0,
            physical_j_rel_eps_scale=0.02,
            physical_j_action_directions=0,
            initial_jacobian_weight=1.0,
            initial_tangent_weight=0.0,
            tangent_directions=0,
            lr=1.0e-4,
            lr_decay=1.0,
            grad_clip=10.0,
            weight_decay=0.0,
            eval_every=1,
            log_every=1,
            init_checkpoint="",
            freeze_skip=False,
            train_point_sample_count=2,
        )
        train_true176_generic(args)
        summary = out_dir / "training_summary.json"
        assert summary.exists()
        text = summary.read_text(encoding="utf-8")
        assert '"le_normalization": "global-component"' in text
        assert '"train_point_sample_count": 2' in text


def test_query_b_baseline_warmstart_is_train_only_and_reduces_prior_loss() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        compact = tmp_path / "query_warmstart_compact.npz"
        out_dir = tmp_path / "out"
        n = 4
        rng = np.random.default_rng(321)
        q48 = rng.normal(scale=0.2, size=(n, 48)).astype(np.float32)
        ip_xi = np.zeros((128, 3), dtype=np.float32)
        ip_xi[:, 0] = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
        ip_xi[:, 1] = np.linspace(0.5, -0.5, 128, dtype=np.float32)
        b = np.zeros((n, 128, 6, 48), dtype=np.float32)
        for ip in range(128):
            b[:, ip, 0, 0] = 0.2 + 0.01 * float(ip)
            b[:, ip, 1, 1] = -0.1 + 0.005 * float(ip)
        le = np.einsum("npaj,nj->npa", b, q48).astype(np.float32)
        np.savez(
            compact,
            q48_raw=q48,
            LE128_base=le,
            B_LE128_forward=b,
            X_keep=np.zeros((n, 16, 3), dtype=np.float32),
            ip_xi=ip_xi,
            sample_paths=np.asarray(["case001_train", "case002_val"], dtype=str),
        )
        args = SimpleNamespace(
            ddp=False,
            cuda=False,
            ddp_backend="gloo",
            seed=9,
            out_dir=out_dir,
            compact=[str(compact)],
            compact_list="",
            target_ips="0,1,2,3",
            scale_mode="normalized",
            frame_stride=1,
            max_frames_per_compact=0,
            val_fraction=0.0,
            val_cases="2",
            split_mode="case",
            allow_overlap_val=False,
            max_eval_frames=4,
            branch_feature_mode="xkeep-qraw",
            b_label_coordinate="auto",
            point_feature_source="data",
            include_id_features=False,
            allow_shape4_point_feature_fallback=False,
            allow_physical_shape4_trunk=False,
            detj_scale_dim=3,
            le_normalization="global-component",
            epochs=1,
            batch_size=2,
            eval_batch_size=1,
            basis_dim=8,
            hidden_dim=16,
            branch_depth=2,
            trunk_depth=2,
            activation="tanh",
            model_style="query-fe-linear-residual",
            mionet_product_scale="none",
            use_q_skip=False,
            residual_scale=0.0,
            fe_baseline_scale=1.0,
            freeze_fe_point_baseline=False,
            no_zero_init_residual=False,
            baseline_jacobian_weight=0.0,
            baseline_j_loss_mode="norm",
            b_baseline_warmstart_steps=60,
            b_baseline_warmstart_lr=5.0e-3,
            b_baseline_warmstart_weight_decay=0.0,
            b_baseline_warmstart_source="train-only",
            jacobian_columns="0,1",
            jacobian_columns_per_batch=2,
            jacobian_method="forward",
            eval_columns="0,1",
            j_loss_mode="norm",
            physical_j_aux_weight=0.0,
            physical_j_abs_weight=1.0,
            physical_j_rel_weight=0.0,
            physical_j_action_weight=0.0,
            physical_j_rel_eps_scale=0.02,
            physical_j_action_directions=0,
            initial_jacobian_weight=0.0,
            initial_tangent_weight=0.0,
            tangent_directions=0,
            lr=1.0e-4,
            lr_decay=1.0,
            grad_clip=10.0,
            weight_decay=0.0,
            eval_every=1,
            log_every=1,
            init_checkpoint="",
            freeze_skip=False,
            train_point_sample_count=0,
        )
        train_true176_generic(args)
        summary = json.loads((out_dir / "training_summary.json").read_text(encoding="utf-8"))
        meta = summary["warmstart_meta"]
        assert meta["warmstart_enabled"] is True
        assert meta["warmstart_train_cases"] == [1]
        assert meta["warmstart_val_cases_excluded"] == [2]
        assert 2 not in meta["warmstart_train_cases"]
        assert meta["b_prior_after_train_rel"] < meta["b_prior_before_train_rel"]
        assert "b_prior_after_train_B_rel" in meta
        assert "b_prior_after_train_evalcols_B_rel" in meta
        assert "b_prior_current_train_evalcols_B_rel" in summary["history"][0]
        assert summary["history"][0]["point_b_correction_norm_ratio"] > 0.0


def test_generic_point_feature_loader_marks_shape4_fallback() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "missing_points_compact.npz"
        shape4 = np.asarray([[1.2, 0.1, 0.0, 0.0], [1.3, 0.08, 0.05, 0.1]], dtype=np.float32)
        np.savez(path, shape4=shape4)
        points, meta = load_point_features_from_compacts(
            compact_paths=[str(path)],
            source_index=np.zeros(2, dtype=np.int64),
            source_row=np.arange(2, dtype=np.int64),
            shape4=shape4,
            target_ips=[0, 1],
            source="auto",
            include_id_features=True,
            allow_shape4_fallback=True,
        )
        assert points.shape[0] == 2
        assert points.shape[1] == 2
        assert meta["point_feature_source"] == "shape4_generated_fallback"


def test_generic_point_feature_loader_shape4_audited() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shape4_contract_compact.npz"
        shape4 = np.asarray([[1.2, 0.1, 0.0, 0.0], [1.2, 0.1, 0.0, 0.0]], dtype=np.float32)
        np.savez(path, shape4=shape4)
        points, meta = load_point_features_from_compacts(
            compact_paths=[str(path)],
            source_index=np.zeros(2, dtype=np.int64),
            source_row=np.arange(2, dtype=np.int64),
            shape4=shape4,
            target_ips=[0, 7, 127],
            source="shape4-audited",
            include_id_features=True,
            allow_shape4_fallback=False,
        )
        assert points.shape[0] == 2
        assert points.shape[1] == 3
        assert meta["point_feature_source"] == "shape4_reconstructed_audited"
        assert meta["shape4_ip_contract"] == "standard_css8_elem_major_ip_major"
        assert meta["unique_shape4_count"] == 1


def test_tiny_overfit_loss_decreases() -> None:
    torch.manual_seed(19)
    dataset = SyntheticMacroDataset(SyntheticConfig(num_samples=256, seed=19))
    model = MacroDeepONet(hidden_dim=48, basis_dim=24, depth=3)
    opt = torch.optim.AdamW(model.parameters(), lr=3.0e-3)
    q = dataset.q_b
    x = dataset.x_nodes
    xi = dataset.xi
    y = dataset.epsilon

    def loss_value() -> torch.Tensor:
        return torch.nn.functional.mse_loss(model(q, x, xi), y)

    initial = loss_value().item()
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        loss = loss_value()
        loss.backward()
        opt.step()
    final = loss_value().item()
    assert math.isfinite(final)
    assert final < 0.75 * initial, (initial, final)


if __name__ == "__main__":
    test_geometry_identities()
    test_synthetic_rigid_targets()
    test_forward_and_autograd_shapes()
    test_true176_point_features_and_ad_shapes()
    test_true176_xkeep_branch_and_ad_shapes()
    test_noem_style_mionet_splits_q_and_geometry()
    test_fe_linear_residual_model_has_explicit_b_baseline()
    test_query_fe_linear_residual_model_supports_dynamic_points()
    test_true176_keep_node_order_matches_q48_contract()
    test_true176_loader_uses_explicit_xkeep_when_available()
    test_true176_physical_scale_transforms_branch_and_b()
    test_isoparametric_similarity_scaling_laws()
    test_isoparametric_mapping_laws()
    test_true176_css8_macro_mapping_laws()
    test_true176_physical_scale_rejects_shape4_branch_geometry()
    test_true176_physical_scale_rejects_shape4_trunk_by_default()
    test_true176_physical_scale_transforms_raw_point_fields()
    test_true176_physical_scale_rejects_unnamed_point_features()
    test_true176_full_xnodes_branch_is_available_as_ablation()
    test_true176_loader_rejects_non_full48_b()
    test_true176_physical_train_requires_explicit_h()
    test_strict_split_uses_case_or_geometry_without_overlap()
    test_generic_point_feature_loader_reads_real_fields()
    test_generic_point_feature_loader_rejects_name_order_mismatch()
    test_generic_point_feature_loader_carries_ip_keys()
    test_generic_point_feature_loader_uses_standard_ip_keys_for_spatial_id_features()
    test_generic_point_feature_loader_nonstandard_ip_keys_use_rank_id_guard()
    test_generic_complete_compact_can_omit_shape4_when_geometry_is_explicit()
    test_exporter_merge_guards_q_le_and_ip_keys()
    test_exporter_ip_audit_can_block_bad_geometry()
    test_generic_query_training_supports_global_le_norm_and_point_sampling()
    test_generic_point_feature_loader_marks_shape4_fallback()
    test_generic_point_feature_loader_shape4_audited()
    test_tiny_overfit_loss_decreases()
    print("smoke_test.py: all checks passed")
