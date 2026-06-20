from __future__ import annotations

import math
import tempfile
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
from macro_deeponet.models import MacroDeepONet, True176Shape4QrawDeepONet
from macro_deeponet.point_features import load_point_features_from_compacts
from macro_deeponet.true176_data import build_branch_features, build_point_features, load_one_compact
from macro_deeponet.train_true176_deeponet_sobolev import ad_jacobian


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


def test_true176_xnodes_branch_and_ad_shapes() -> None:
    shape4 = np.asarray([[1.2, 0.05, 0.10, 0.00], [1.4, 0.04, 0.08, 0.20]], dtype=np.float32)
    q48 = np.zeros((2, 48), dtype=np.float32)
    branch, branch_meta = build_branch_features(shape4=shape4, q48_raw=q48, mode="xnodes-qraw")
    point, _meta = build_point_features(shape4, include_id_features=False)
    assert branch.shape == (2, 198)
    assert branch_meta["q_start"] == 0
    assert branch_meta["geometry_input"] == "X_macro[50,3]"

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
    test_true176_xnodes_branch_and_ad_shapes()
    test_true176_loader_rejects_non_full48_b()
    test_generic_point_feature_loader_reads_real_fields()
    test_generic_point_feature_loader_marks_shape4_fallback()
    test_generic_point_feature_loader_shape4_audited()
    test_tiny_overfit_loss_decreases()
    print("smoke_test.py: all checks passed")
