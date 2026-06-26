"""Coordinate-trunk DeepONet for Macro16 source128 experiments.

This file is intentionally independent from ``src/macro_deeponet/models.py``.
The trunk sees only dimensionless physical Gauss-point coordinates.
"""

from __future__ import annotations

import math

import torch
from torch import nn


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


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: int = 128,
        depth: int = 4,
        activation: str = "tanh",
        zero_last: bool = False,
    ) -> None:
        super().__init__()
        if int(depth) < 1:
            raise ValueError("depth must be >= 1")
        act = activation_module(activation)
        layers: list[nn.Module] = []
        last = int(in_dim)
        for _ in range(int(depth) - 1):
            layers.append(nn.Linear(last, int(hidden_dim)))
            layers.append(act())
            last = int(hidden_dim)
        final = nn.Linear(last, int(out_dim))
        if zero_last:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Macro16CoordinateDeepONet(nn.Module):
    """DeepONet mapping ``[q48_def_hat, g]`` and ``x_gp_hat`` to LE.

    Parameters
    ----------
    branch_dim:
        Dimension of the standardized branch vector. The first 48 coordinates
        must correspond to ``q48_def_hat``.
    trunk_dim:
        Must be 3 for ``x_gp_hat``.
    """

    def __init__(
        self,
        *,
        branch_dim: int = 62,
        trunk_dim: int = 3,
        strain_dim: int = 6,
        basis_dim: int = 64,
        hidden_dim: int = 128,
        branch_depth: int = 4,
        trunk_depth: int = 4,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if int(trunk_dim) != 3:
            raise ValueError("coordinate trunk uses exactly 3 x_gp_hat inputs")
        if int(branch_dim) <= 48:
            raise ValueError("branch_dim must contain q48_def_hat plus geometry parameters")
        self.branch_dim = int(branch_dim)
        self.trunk_dim = int(trunk_dim)
        self.strain_dim = int(strain_dim)
        self.basis_dim = int(basis_dim)

        coeff_dim = self.strain_dim * self.basis_dim
        self.branch = MLP(
            self.branch_dim,
            coeff_dim,
            hidden_dim=hidden_dim,
            depth=branch_depth,
            activation=activation,
        )
        self.trunk = MLP(
            self.trunk_dim,
            coeff_dim,
            hidden_dim=hidden_dim,
            depth=trunk_depth,
            activation=activation,
        )
        self.bias = nn.Parameter(torch.zeros(self.strain_dim))

    def forward(self, branch_norm: torch.Tensor, x_gp_hat: torch.Tensor) -> torch.Tensor:
        if branch_norm.ndim != 2:
            raise ValueError("branch_norm must have shape [B,F]")
        if x_gp_hat.ndim != 3:
            raise ValueError("x_gp_hat must have shape [B,P,3]")
        if branch_norm.shape[-1] != self.branch_dim:
            raise ValueError(f"branch_norm last dim must be {self.branch_dim}")
        if x_gp_hat.shape[-1] != self.trunk_dim:
            raise ValueError("x_gp_hat last dim must be 3")
        batch = int(branch_norm.shape[0])
        points = int(x_gp_hat.shape[1])
        b = self.branch(branch_norm).view(batch, self.strain_dim, self.basis_dim)
        t = self.trunk(x_gp_hat.reshape(batch * points, self.trunk_dim)).view(
            batch,
            points,
            self.strain_dim,
            self.basis_dim,
        )
        out = torch.einsum("bak,bpak->bpa", b, t) / math.sqrt(float(self.basis_dim))
        return out + self.bias.view(1, 1, self.strain_dim)


class Macro16CoordinateLinearResidualDeepONet(nn.Module):
    """Coordinate model with a q-linear LE baseline plus residual.

    The network still outputs only LE.  The linear coefficient is an internal
    coordinate/geometric prior, so the supervised B remains the autograd
    derivative of LE with respect to ``q48_def_hat``.
    """

    def __init__(
        self,
        *,
        branch_dim: int = 62,
        trunk_dim: int = 3,
        q_dim: int = 48,
        strain_dim: int = 6,
        basis_dim: int = 64,
        hidden_dim: int = 128,
        branch_depth: int = 4,
        trunk_depth: int = 4,
        activation: str = "tanh",
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        if int(trunk_dim) != 3:
            raise ValueError("coordinate trunk uses exactly 3 x_gp_hat inputs")
        if int(q_dim) != 48:
            raise ValueError("q_dim must stay 48")
        if int(branch_dim) <= int(q_dim):
            raise ValueError("branch_dim must contain q48_def_hat plus geometry parameters")
        self.branch_dim = int(branch_dim)
        self.trunk_dim = int(trunk_dim)
        self.q_dim = int(q_dim)
        self.g_dim = self.branch_dim - self.q_dim
        self.strain_dim = int(strain_dim)
        self.basis_dim = int(basis_dim)

        point_state_dim = self.g_dim + self.trunk_dim
        self.le0_net = MLP(
            point_state_dim,
            self.strain_dim,
            hidden_dim=hidden_dim,
            depth=trunk_depth,
            activation=activation,
        )
        self.b_net = MLP(
            point_state_dim,
            self.strain_dim * self.q_dim,
            hidden_dim=hidden_dim,
            depth=trunk_depth,
            activation=activation,
        )

        coeff_dim = self.strain_dim * self.basis_dim
        self.branch = MLP(
            self.branch_dim,
            coeff_dim,
            hidden_dim=hidden_dim,
            depth=branch_depth,
            activation=activation,
            zero_last=zero_init_residual,
        )
        self.trunk = MLP(
            self.trunk_dim,
            coeff_dim,
            hidden_dim=hidden_dim,
            depth=trunk_depth,
            activation=activation,
        )
        self.bias = nn.Parameter(torch.zeros(self.strain_dim))

    def forward(self, branch_norm: torch.Tensor, x_gp_hat: torch.Tensor) -> torch.Tensor:
        if branch_norm.ndim != 2:
            raise ValueError("branch_norm must have shape [B,F]")
        if x_gp_hat.ndim != 3:
            raise ValueError("x_gp_hat must have shape [B,P,3]")
        if branch_norm.shape[-1] != self.branch_dim:
            raise ValueError(f"branch_norm last dim must be {self.branch_dim}")
        if x_gp_hat.shape[-1] != self.trunk_dim:
            raise ValueError("x_gp_hat last dim must be 3")
        batch = int(branch_norm.shape[0])
        points = int(x_gp_hat.shape[1])
        qn = branch_norm[:, : self.q_dim]
        gn = branch_norm[:, self.q_dim :]
        point_state = torch.cat(
            [gn[:, None, :].expand(batch, points, self.g_dim), x_gp_hat],
            dim=-1,
        )
        le0 = self.le0_net(point_state.reshape(batch * points, -1)).view(batch, points, self.strain_dim)
        b_norm = self.b_net(point_state.reshape(batch * points, -1)).view(
            batch,
            points,
            self.strain_dim,
            self.q_dim,
        )
        linear = torch.einsum("bpaj,bj->bpa", b_norm, qn)
        b = self.branch(branch_norm).view(batch, self.strain_dim, self.basis_dim)
        t = self.trunk(x_gp_hat.reshape(batch * points, self.trunk_dim)).view(
            batch,
            points,
            self.strain_dim,
            self.basis_dim,
        )
        residual = torch.einsum("bak,bpak->bpa", b, t) / math.sqrt(float(self.basis_dim))
        return le0 + linear + residual + self.bias.view(1, 1, self.strain_dim)


class Macro16CoordinateStateLinearResidualDeepONet(nn.Module):
    """Coordinate model whose pointwise B prior also sees q state.

    This keeps the same external inputs as the coordinate route:
    ``q48_def_hat + geometry_g`` on the branch and ``x_gp_hat`` on the trunk.
    The difference from ``Macro16CoordinateLinearResidualDeepONet`` is that the
    internal pointwise B prior is a function of ``q, g, x`` instead of only
    ``g, x``.  That tests whether the observed q-dependent B field is the
    reason the geometry-only point prior collapses in overfit.
    """

    def __init__(
        self,
        *,
        branch_dim: int = 62,
        trunk_dim: int = 3,
        q_dim: int = 48,
        strain_dim: int = 6,
        basis_dim: int = 64,
        hidden_dim: int = 128,
        branch_depth: int = 4,
        trunk_depth: int = 4,
        activation: str = "tanh",
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        if int(trunk_dim) != 3:
            raise ValueError("coordinate trunk uses exactly 3 x_gp_hat inputs")
        if int(q_dim) != 48:
            raise ValueError("q_dim must stay 48")
        if int(branch_dim) <= int(q_dim):
            raise ValueError("branch_dim must contain q48_def_hat plus geometry parameters")
        self.branch_dim = int(branch_dim)
        self.trunk_dim = int(trunk_dim)
        self.q_dim = int(q_dim)
        self.g_dim = self.branch_dim - self.q_dim
        self.strain_dim = int(strain_dim)
        self.basis_dim = int(basis_dim)

        point_static_dim = self.g_dim + self.trunk_dim
        point_state_dim = self.q_dim + self.g_dim + self.trunk_dim
        self.le0_net = MLP(
            point_static_dim,
            self.strain_dim,
            hidden_dim=hidden_dim,
            depth=trunk_depth,
            activation=activation,
        )
        self.b_net = MLP(
            point_state_dim,
            self.strain_dim * self.q_dim,
            hidden_dim=hidden_dim,
            depth=trunk_depth,
            activation=activation,
        )

        coeff_dim = self.strain_dim * self.basis_dim
        self.branch = MLP(
            self.branch_dim,
            coeff_dim,
            hidden_dim=hidden_dim,
            depth=branch_depth,
            activation=activation,
            zero_last=zero_init_residual,
        )
        self.trunk = MLP(
            self.trunk_dim,
            coeff_dim,
            hidden_dim=hidden_dim,
            depth=trunk_depth,
            activation=activation,
        )
        self.bias = nn.Parameter(torch.zeros(self.strain_dim))

    def forward(self, branch_norm: torch.Tensor, x_gp_hat: torch.Tensor) -> torch.Tensor:
        if branch_norm.ndim != 2:
            raise ValueError("branch_norm must have shape [B,F]")
        if x_gp_hat.ndim != 3:
            raise ValueError("x_gp_hat must have shape [B,P,3]")
        if branch_norm.shape[-1] != self.branch_dim:
            raise ValueError(f"branch_norm last dim must be {self.branch_dim}")
        if x_gp_hat.shape[-1] != self.trunk_dim:
            raise ValueError("x_gp_hat last dim must be 3")
        batch = int(branch_norm.shape[0])
        points = int(x_gp_hat.shape[1])
        qn = branch_norm[:, : self.q_dim]
        gn = branch_norm[:, self.q_dim :]
        q_point = qn[:, None, :].expand(batch, points, self.q_dim)
        g_point = gn[:, None, :].expand(batch, points, self.g_dim)
        point_static = torch.cat([g_point, x_gp_hat], dim=-1)
        point_state = torch.cat([q_point, point_static], dim=-1)
        le0 = self.le0_net(point_static.reshape(batch * points, -1)).view(batch, points, self.strain_dim)
        b_norm = self.b_net(point_state.reshape(batch * points, -1)).view(
            batch,
            points,
            self.strain_dim,
            self.q_dim,
        )
        linear = torch.einsum("bpaj,bj->bpa", b_norm, qn)
        b = self.branch(branch_norm).view(batch, self.strain_dim, self.basis_dim)
        t = self.trunk(x_gp_hat.reshape(batch * points, self.trunk_dim)).view(
            batch,
            points,
            self.strain_dim,
            self.basis_dim,
        )
        residual = torch.einsum("bak,bpak->bpa", b, t) / math.sqrt(float(self.basis_dim))
        return le0 + linear + residual + self.bias.view(1, 1, self.strain_dim)
