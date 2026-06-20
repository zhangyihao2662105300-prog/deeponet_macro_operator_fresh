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


def activation_module(name: str) -> type[nn.Module]:
    key = str(name).strip().lower()
    if key == "tanh":
        return nn.Tanh
    if key == "silu":
        return nn.SiLU
    if key == "gelu":
        return nn.GELU
    if key == "relu":
        return nn.ReLU
    raise ValueError(f"unknown activation {name!r}")


class MacroDeepONet(nn.Module):
    """Geometry-conditioned DeepONet for ``(q_b, X)(xi) -> strain``.

    This is the original clean Hex8 prototype.  The TRUE176/CSS8 production-facing
    model is ``True176Shape4QrawDeepONet`` below.
    """

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


class True176Shape4QrawDeepONet(nn.Module):
    """DeepONet aligned with the current TRUE176/CSS8 128-IP Sobolev contract.

    Contract, matching the fc4117d launcher/trainer in NNSE-NeuralNetworkShellElement:

        branch input  x_norm    = standardized branch state -> [B, input_dim]
                                = [shape4,q48] legacy or [q48,X_keep] isoparametric
        trunk input   point_norm = standardized 128-IP geometry/features -> [B, P, F]
        output        LE_norm    = standardized LE -> [B, P, 6]

    The current macro-visible default is [q48_raw(48), X_keep(16,3)] with
    input_dim=96.  The older [shape4,q48] route and the full [q48,X_macro]
    ablation can still be selected by the data loader.

    The Sobolev/B label is not an independent output.  It is obtained by AD as
    d(LE_norm)/d(q48_norm), then de-normalized outside the model.

    A learnable q-skip term is included because the current 128-IP trainer uses a
    strong linear-in-q baseline before the nonlinear residual.  This keeps the
    DeepONet version compatible with the present Sobolev training behavior while
    still using branch/trunk operator factorization for the residual field.
    """

    def __init__(
        self,
        *,
        input_dim: int = 52,
        point_dim: int,
        ip_count: int = 128,
        strain_dim: int = 6,
        basis_dim: int = 96,
        hidden_dim: int = 384,
        branch_depth: int = 5,
        trunk_depth: int = 5,
        activation: str = "tanh",
        skip_init: torch.Tensor | None = None,
        train_skip: bool = True,
        residual_scale: float = 1.0,
        q_start: int = 4,
        q_dim: int = 48,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if point_dim <= 0:
            raise ValueError("point_dim must be positive for the 128-IP DeepONet trunk")
        if int(q_start) < 0 or int(q_dim) <= 0 or int(q_start) + int(q_dim) > int(input_dim):
            raise ValueError("q_start/q_dim must select a valid q slice inside the branch input")
        self.input_dim = int(input_dim)
        self.point_dim = int(point_dim)
        self.ip_count = int(ip_count)
        self.strain_dim = int(strain_dim)
        self.basis_dim = int(basis_dim)
        self.residual_scale = float(residual_scale)
        self.q_start = int(q_start)
        self.q_dim = int(q_dim)

        act = activation_module(activation)
        coeff_dim = self.strain_dim * self.basis_dim
        self.branch = MLP(self.input_dim, coeff_dim, hidden_dim=hidden_dim, depth=branch_depth, activation=act)
        self.trunk = MLP(self.point_dim, coeff_dim, hidden_dim=hidden_dim, depth=trunk_depth, activation=act)
        self.bias = nn.Parameter(torch.zeros(self.strain_dim))

        if skip_init is None:
            skip = torch.zeros(self.ip_count, self.strain_dim, self.q_dim, dtype=torch.float32)
        else:
            skip = torch.as_tensor(skip_init, dtype=torch.float32).reshape(self.ip_count, self.strain_dim, self.q_dim)
        self.skip_weight = nn.Parameter(skip, requires_grad=bool(train_skip))

    def forward(self, x_norm: torch.Tensor, point_norm: torch.Tensor) -> torch.Tensor:
        if x_norm.ndim != 2:
            raise ValueError(f"x_norm must have shape [B,{self.input_dim}]")
        if point_norm.ndim != 3:
            raise ValueError("point_norm must have shape [B,P,F]")
        if x_norm.shape[-1] != self.input_dim:
            raise ValueError(f"x_norm last dimension must be {self.input_dim}")
        if point_norm.shape[-1] != self.point_dim:
            raise ValueError(f"point_norm last dimension must be {self.point_dim}")
        if point_norm.shape[1] != self.ip_count:
            raise ValueError(f"point_norm point count must be {self.ip_count}")

        qn = x_norm[:, self.q_start : self.q_start + self.q_dim]
        skip = torch.einsum("paj,bj->bpa", self.skip_weight, qn)
        b = self.branch(x_norm).view(-1, self.strain_dim, self.basis_dim)
        t = self.trunk(point_norm.reshape(-1, self.point_dim)).view(
            x_norm.shape[0], self.ip_count, self.strain_dim, self.basis_dim
        )
        residual = torch.einsum("bak,bpak->bpa", b, t) / (self.basis_dim**0.5)
        return skip + self.residual_scale * residual + self.bias.view(1, 1, self.strain_dim)
