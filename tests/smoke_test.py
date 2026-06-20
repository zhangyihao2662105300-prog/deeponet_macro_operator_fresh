from __future__ import annotations

import math

import torch

from macro_deeponet.autograd import strain_jacobian_wrt_q
from macro_deeponet.data import SyntheticConfig, SyntheticMacroDataset
from macro_deeponet.geometry import (
    HEX8_NATURAL,
    det_jacobian_hex8,
    shape_function_gradients_hex8,
    shape_functions_hex8,
)
from macro_deeponet.models import MacroDeepONet


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
    test_tiny_overfit_loss_decreases()
    print("smoke_test.py: all checks passed")

