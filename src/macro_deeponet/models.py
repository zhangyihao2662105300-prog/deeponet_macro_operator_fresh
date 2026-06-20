"""DeepONet models for macro-element strain fields."""

from __future__ import annotations

import torch
from torch import nn

from .geometry import jacobian_features, normalize_nodes


class MLP(nn.Module):
    """Simple fully connected network with SiLU activations."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        depth: int = 4,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        layers: list[nn.Module] = []
        last = in_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(activation())
            last = hidden_dim
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MacroDeepONet(nn.Module):
    """Geometry-conditioned DeepONet for ``(q_b, X)(xi) -> strain``."""

    def __init__(
        self,
        q_nodes: int = 8,
        geom_nodes: int = 8,
        strain_dim: int = 6,
        basis_dim: int = 64,
        hidden_dim: int = 128,
        depth: int = 4,
        use_jacobian_features: bool = True,
        normalize_geometry: bool = True,
    ) -> None:
        super().__init__()
        self.q_nodes = q_nodes
        self.geom_nodes = geom_nodes
        self.strain_dim = strain_dim
        self.basis_dim = basis_dim
        self.use_jacobian_features = use_jacobian_features
        self.normalize_geometry = normalize_geometry

        branch_dim = q_nodes * 3 + geom_nodes * 3
        trunk_dim = 3 + (10 if use_jacobian_features else 0)
        coeff_dim = strain_dim * basis_dim

        self.branch = MLP(branch_dim, coeff_dim, hidden_dim=hidden_dim, depth=depth)
        self.trunk = MLP(trunk_dim, coeff_dim, hidden_dim=hidden_dim, depth=depth)
        self.bias = nn.Parameter(torch.zeros(strain_dim))

    def _prepare_inputs(
        self,
        q_b: torch.Tensor,
        x_nodes: torch.Tensor,
        xi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q_b.ndim != 3 or x_nodes.ndim != 3 or xi.ndim != 2:
            raise ValueError("Expected q_b [B,n,3], x_nodes [B,n,3], xi [B,3]")
        if q_b.shape[-2:] != (self.q_nodes, 3):
            raise ValueError(f"q_b must have shape [B,{self.q_nodes},3]")
        if x_nodes.shape[-2:] != (self.geom_nodes, 3):
            raise ValueError(f"x_nodes must have shape [B,{self.geom_nodes},3]")
        if xi.shape[-1] != 3:
            raise ValueError("xi must have last dimension 3")

        if self.normalize_geometry:
            x_norm, _, scale = normalize_nodes(x_nodes)
            q_centered = q_b - q_b.mean(dim=-2, keepdim=True)
            q_norm = q_centered / scale
        else:
            x_norm = x_nodes
            q_norm = q_b - q_b.mean(dim=-2, keepdim=True)

        branch_in = torch.cat([q_norm.flatten(1), x_norm.flatten(1)], dim=-1)

        trunk_parts = [xi]
        if self.use_jacobian_features:
            trunk_parts.append(jacobian_features(x_norm, xi))
        trunk_in = torch.cat(trunk_parts, dim=-1)
        return branch_in, trunk_in

    def forward(self, q_b: torch.Tensor, x_nodes: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        branch_in, trunk_in = self._prepare_inputs(q_b, x_nodes, xi)
        b = self.branch(branch_in).view(-1, self.strain_dim, self.basis_dim)
        t = self.trunk(trunk_in).view(-1, self.strain_dim, self.basis_dim)
        return (b * t).sum(dim=-1) / (self.basis_dim**0.5) + self.bias
