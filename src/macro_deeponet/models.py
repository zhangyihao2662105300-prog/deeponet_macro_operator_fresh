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
        zero_last: bool = False,
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
        final = nn.Linear(last, out_dim)
        if zero_last:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        layers.append(final)
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


class FELinearResidualDeepONet(nn.Module):
    """TRUE176 model with an explicit FE-like linear B baseline.

    The model writes the standardized strain field as

        LE_norm = B_base_norm(point_features) @ q48_norm + DeepONet_residual.

    ``B_base_norm`` is predicted only from Trunk/point features, so its
    derivative with respect to q is explicit and can be supervised directly by
    raw/physical B loss.  A static per-IP baseline initialized from the training
    mean B is kept as a prior; the point-conditioned network learns the
    geometry/IP-dependent correction.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        point_dim: int,
        q_start: int,
        q_dim: int = 48,
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
        baseline_scale: float = 1.0,
        train_point_baseline: bool = True,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if point_dim <= 0:
            raise ValueError("point_dim must be positive")
        if int(q_start) < 0 or int(q_dim) <= 0 or int(q_start) + int(q_dim) > int(input_dim):
            raise ValueError("q_start/q_dim must select a valid q slice inside the branch input")
        self.input_dim = int(input_dim)
        self.point_dim = int(point_dim)
        self.q_start = int(q_start)
        self.q_dim = int(q_dim)
        self.ip_count = int(ip_count)
        self.strain_dim = int(strain_dim)
        self.basis_dim = int(basis_dim)
        self.residual_scale = float(residual_scale)
        self.baseline_scale = float(baseline_scale)
        self.train_point_baseline = bool(train_point_baseline)
        self.zero_init_residual = bool(zero_init_residual)

        act = activation_module(activation)
        coeff_dim = self.strain_dim * self.basis_dim
        self.branch = MLP(
            self.input_dim,
            coeff_dim,
            hidden_dim=hidden_dim,
            depth=branch_depth,
            activation=act,
            zero_last=self.zero_init_residual,
        )
        self.trunk = MLP(self.point_dim, coeff_dim, hidden_dim=hidden_dim, depth=trunk_depth, activation=act)
        self.point_b_net = MLP(
            self.point_dim,
            self.strain_dim * self.q_dim,
            hidden_dim=hidden_dim,
            depth=trunk_depth,
            activation=act,
            zero_last=True,
        )
        for p in self.point_b_net.parameters():
            p.requires_grad_(self.train_point_baseline)
        self.bias = nn.Parameter(torch.zeros(self.strain_dim))

        if skip_init is None:
            static = torch.zeros(self.ip_count, self.strain_dim, self.q_dim, dtype=torch.float32)
        else:
            static = torch.as_tensor(skip_init, dtype=torch.float32).reshape(self.ip_count, self.strain_dim, self.q_dim)
        self.static_b_norm = nn.Parameter(static, requires_grad=bool(train_skip))

    def _linear_b_norm(self, point_norm: torch.Tensor) -> torch.Tensor:
        delta = self.point_b_net(point_norm.reshape(-1, self.point_dim)).view(
            point_norm.shape[0], self.ip_count, self.strain_dim, self.q_dim
        )
        return self.static_b_norm.unsqueeze(0) + self.baseline_scale * delta

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
        b_base = self._linear_b_norm(point_norm)
        linear = torch.einsum("bpaj,bj->bpa", b_base, qn)
        b = self.branch(x_norm).view(-1, self.strain_dim, self.basis_dim)
        t = self.trunk(point_norm.reshape(-1, self.point_dim)).view(
            x_norm.shape[0], self.ip_count, self.strain_dim, self.basis_dim
        )
        residual = torch.einsum("bak,bpak->bpa", b, t) / (self.basis_dim**0.5)
        return linear + self.residual_scale * residual + self.bias.view(1, 1, self.strain_dim)


class QueryFELinearResidualDeepONet(nn.Module):
    """FE-linear-residual DeepONet that supports arbitrary query point counts.

    This keeps the current model's useful structure,

        LE_norm = B_base_norm(point_features) @ q48_norm + DeepONet_residual,

    but removes the fixed ``[128,6,48]`` per-IP parameter table.  The linear
    baseline is a global q-linear prior plus a point-conditioned correction, so
    the same model can be evaluated on any ``P`` query points whose point
    features follow the training contract.
    """

    supports_dynamic_points = True

    def __init__(
        self,
        *,
        input_dim: int,
        point_dim: int,
        q_start: int,
        q_dim: int = 48,
        ip_count: int = 0,
        strain_dim: int = 6,
        basis_dim: int = 96,
        hidden_dim: int = 384,
        branch_depth: int = 5,
        trunk_depth: int = 5,
        activation: str = "tanh",
        skip_init: torch.Tensor | None = None,
        train_skip: bool = True,
        residual_scale: float = 1.0,
        baseline_scale: float = 1.0,
        train_point_baseline: bool = True,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if point_dim <= 0:
            raise ValueError("point_dim must be positive")
        if int(q_start) < 0 or int(q_dim) <= 0 or int(q_start) + int(q_dim) > int(input_dim):
            raise ValueError("q_start/q_dim must select a valid q slice inside the branch input")
        self.input_dim = int(input_dim)
        self.point_dim = int(point_dim)
        self.q_start = int(q_start)
        self.q_dim = int(q_dim)
        self.ip_count = int(ip_count)
        self.strain_dim = int(strain_dim)
        self.basis_dim = int(basis_dim)
        self.residual_scale = float(residual_scale)
        self.baseline_scale = float(baseline_scale)
        self.train_point_baseline = bool(train_point_baseline)
        self.zero_init_residual = bool(zero_init_residual)

        act = activation_module(activation)
        coeff_dim = self.strain_dim * self.basis_dim
        self.branch = MLP(
            self.input_dim,
            coeff_dim,
            hidden_dim=hidden_dim,
            depth=branch_depth,
            activation=act,
            zero_last=self.zero_init_residual,
        )
        self.trunk = MLP(self.point_dim, coeff_dim, hidden_dim=hidden_dim, depth=trunk_depth, activation=act)
        self.point_b_net = MLP(
            self.point_dim,
            self.strain_dim * self.q_dim,
            hidden_dim=hidden_dim,
            depth=trunk_depth,
            activation=act,
            zero_last=True,
        )
        for p in self.point_b_net.parameters():
            p.requires_grad_(self.train_point_baseline)
        self.bias = nn.Parameter(torch.zeros(self.strain_dim))

        if skip_init is None:
            global_static = torch.zeros(self.strain_dim, self.q_dim, dtype=torch.float32)
        else:
            static = torch.as_tensor(skip_init, dtype=torch.float32)
            if static.shape == (self.strain_dim, self.q_dim):
                global_static = static
            elif static.ndim == 3 and static.shape[-2:] == (self.strain_dim, self.q_dim):
                global_static = static.mean(dim=0)
            else:
                raise ValueError(
                    "skip_init for QueryFELinearResidualDeepONet must have shape "
                    f"[{self.strain_dim},{self.q_dim}] or [P,{self.strain_dim},{self.q_dim}]"
                )
        self.global_b_norm = nn.Parameter(global_static, requires_grad=bool(train_skip))

    def _linear_b_norm(self, point_norm: torch.Tensor) -> torch.Tensor:
        if point_norm.ndim != 3:
            raise ValueError("point_norm must have shape [B,P,F]")
        batch, point_count, _ = point_norm.shape
        delta = self.point_b_net(point_norm.reshape(-1, self.point_dim)).view(
            batch, point_count, self.strain_dim, self.q_dim
        )
        return self.global_b_norm.view(1, 1, self.strain_dim, self.q_dim) + self.baseline_scale * delta

    def forward(self, x_norm: torch.Tensor, point_norm: torch.Tensor) -> torch.Tensor:
        if x_norm.ndim != 2:
            raise ValueError(f"x_norm must have shape [B,{self.input_dim}]")
        if point_norm.ndim != 3:
            raise ValueError("point_norm must have shape [B,P,F]")
        if x_norm.shape[-1] != self.input_dim:
            raise ValueError(f"x_norm last dimension must be {self.input_dim}")
        if point_norm.shape[-1] != self.point_dim:
            raise ValueError(f"point_norm last dimension must be {self.point_dim}")
        if point_norm.shape[0] != x_norm.shape[0]:
            raise ValueError("x_norm and point_norm batch dimensions must match")

        qn = x_norm[:, self.q_start : self.q_start + self.q_dim]
        b_base = self._linear_b_norm(point_norm)
        linear = torch.einsum("bpaj,bj->bpa", b_base, qn)
        b = self.branch(x_norm).view(-1, self.strain_dim, self.basis_dim)
        point_count = int(point_norm.shape[1])
        t = self.trunk(point_norm.reshape(-1, self.point_dim)).view(
            x_norm.shape[0], point_count, self.strain_dim, self.basis_dim
        )
        residual = torch.einsum("bak,bpak->bpa", b, t) / (self.basis_dim**0.5)
        return linear + self.residual_scale * residual + self.bias.view(1, 1, self.strain_dim)


class QueryFEAnchoredLinearResidualDeepONet(QueryFELinearResidualDeepONet):
    """Query FE-linear residual model with an explicit zero-displacement anchor.

    The model stays in normalized LE coordinates internally:

        LE_norm = B_base_norm(point) @ q_norm
                  + gate(q_raw) * (R_raw(q_norm, point) - R_raw(q0_norm, point)).

    ``q0_norm`` is the standardized branch vector for raw ``q=0``.  The
    residual subtraction guarantees ``R(q=0)=0``.  ``le_zero_norm`` is the
    normalized raw ``LE=0`` value, so the output also satisfies
    ``LE_hat(raw q=0)=0`` after de-normalization.
    """

    supports_dynamic_points = True
    anchored_le_head = True

    def __init__(
        self,
        *,
        input_dim: int,
        point_dim: int,
        q_start: int,
        q_dim: int = 48,
        ip_count: int = 0,
        strain_dim: int = 6,
        basis_dim: int = 96,
        hidden_dim: int = 384,
        branch_depth: int = 5,
        trunk_depth: int = 5,
        activation: str = "tanh",
        skip_init: torch.Tensor | None = None,
        train_skip: bool = True,
        residual_scale: float = 1.0,
        baseline_scale: float = 1.0,
        train_point_baseline: bool = True,
        zero_init_residual: bool = True,
        q_zero_norm: torch.Tensor | None = None,
        le_zero_norm: torch.Tensor | None = None,
        q_raw_mean: torch.Tensor | None = None,
        q_raw_std: torch.Tensor | None = None,
        gate_q0: float = 0.0,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            point_dim=point_dim,
            q_start=q_start,
            q_dim=q_dim,
            ip_count=ip_count,
            strain_dim=strain_dim,
            basis_dim=basis_dim,
            hidden_dim=hidden_dim,
            branch_depth=branch_depth,
            trunk_depth=trunk_depth,
            activation=activation,
            skip_init=skip_init,
            train_skip=train_skip,
            residual_scale=residual_scale,
            baseline_scale=baseline_scale,
            train_point_baseline=train_point_baseline,
            zero_init_residual=zero_init_residual,
        )
        if q_zero_norm is None:
            q_zero = torch.zeros(self.q_dim, dtype=torch.float32)
        else:
            q_zero = torch.as_tensor(q_zero_norm, dtype=torch.float32).reshape(self.q_dim)
        if le_zero_norm is None:
            le_zero = torch.zeros(self.strain_dim, dtype=torch.float32)
        else:
            le_zero = torch.as_tensor(le_zero_norm, dtype=torch.float32).reshape(-1, self.strain_dim)
            if le_zero.shape[0] == 1:
                le_zero = le_zero.reshape(self.strain_dim)
            elif le_zero.shape[0] != self.ip_count:
                raise ValueError("le_zero_norm must have shape [6], [1,6], or [P,6]")
        if q_raw_mean is None:
            q_mean = torch.zeros(self.q_dim, dtype=torch.float32)
        else:
            q_mean = torch.as_tensor(q_raw_mean, dtype=torch.float32).reshape(self.q_dim)
        if q_raw_std is None:
            q_std = torch.ones(self.q_dim, dtype=torch.float32)
        else:
            q_std = torch.clamp(torch.as_tensor(q_raw_std, dtype=torch.float32).reshape(self.q_dim), min=1.0e-12)
        self.register_buffer("q_zero_norm", q_zero)
        self.register_buffer("le_zero_norm", le_zero)
        self.register_buffer("q_raw_mean", q_mean)
        self.register_buffer("q_raw_std", q_std)
        self.gate_q0 = float(max(0.0, gate_q0))
        self.bias.requires_grad_(False)

    def _residual_raw(self, x_norm: torch.Tensor, point_norm: torch.Tensor) -> torch.Tensor:
        b = self.branch(x_norm).view(-1, self.strain_dim, self.basis_dim)
        point_count = int(point_norm.shape[1])
        t = self.trunk(point_norm.reshape(-1, self.point_dim)).view(
            x_norm.shape[0], point_count, self.strain_dim, self.basis_dim
        )
        return torch.einsum("bak,bpak->bpa", b, t) / (self.basis_dim**0.5)

    def _q_zero_branch(self, x_norm: torch.Tensor) -> torch.Tensor:
        x0 = x_norm.clone()
        x0[:, self.q_start : self.q_start + self.q_dim] = self.q_zero_norm.to(dtype=x_norm.dtype, device=x_norm.device)
        return x0

    def _residual_gate(self, x_norm: torch.Tensor) -> torch.Tensor:
        qn = x_norm[:, self.q_start : self.q_start + self.q_dim]
        q_raw = qn * self.q_raw_std.to(dtype=x_norm.dtype, device=x_norm.device) + self.q_raw_mean.to(
            dtype=x_norm.dtype,
            device=x_norm.device,
        )
        norm = torch.linalg.norm(q_raw, dim=-1, keepdim=True)
        if self.gate_q0 <= 0.0:
            return torch.ones_like(norm)
        return norm / (norm + torch.as_tensor(self.gate_q0, dtype=x_norm.dtype, device=x_norm.device))

    def zero_q_prediction(self, x_norm: torch.Tensor, point_norm: torch.Tensor) -> torch.Tensor:
        return self.forward(self._q_zero_branch(x_norm), point_norm)

    def residual_offset_norm(self, x_norm: torch.Tensor, point_norm: torch.Tensor) -> torch.Tensor:
        raw = self._residual_raw(x_norm, point_norm)
        raw0 = self._residual_raw(self._q_zero_branch(x_norm), point_norm)
        return self.residual_scale * self._residual_gate(x_norm).view(-1, 1, 1) * (raw - raw0)

    def forward(self, x_norm: torch.Tensor, point_norm: torch.Tensor) -> torch.Tensor:
        if x_norm.ndim != 2:
            raise ValueError(f"x_norm must have shape [B,{self.input_dim}]")
        if point_norm.ndim != 3:
            raise ValueError("point_norm must have shape [B,P,F]")
        if x_norm.shape[-1] != self.input_dim:
            raise ValueError(f"x_norm last dimension must be {self.input_dim}")
        if point_norm.shape[-1] != self.point_dim:
            raise ValueError(f"point_norm last dimension must be {self.point_dim}")
        if point_norm.shape[0] != x_norm.shape[0]:
            raise ValueError("x_norm and point_norm batch dimensions must match")

        qn = x_norm[:, self.q_start : self.q_start + self.q_dim]
        b_base = self._linear_b_norm(point_norm)
        q0 = self.q_zero_norm.to(dtype=x_norm.dtype, device=x_norm.device).view(1, self.q_dim)
        linear = torch.einsum("bpaj,bj->bpa", b_base, qn - q0)
        le0 = self.le_zero_norm.to(dtype=x_norm.dtype, device=x_norm.device)
        if le0.ndim == 1:
            anchor = le0.view(1, 1, self.strain_dim)
        else:
            anchor = le0.view(1, le0.shape[0], self.strain_dim)
        return anchor + linear + self.residual_offset_norm(x_norm, point_norm)


class Macro16BoundaryDeepONet(QueryFEAnchoredLinearResidualDeepONet):
    """Boundary-control Macro16 operator with full q48 input.

    The intended branch contract is

        [q48_hat, X16_hat.flatten(), optional global geometry scalars]

    where ``q48_hat`` keeps all 16 x 3 displacement components.  No rigid modes
    are projected out before the network.
    """

    macro16_contract = "v4-macro16-boundary-operator-001"

    def __init__(
        self,
        *,
        input_dim: int = 99,
        point_dim: int,
        q_start: int = 0,
        q_dim: int = 48,
        ip_count: int = 0,
        strain_dim: int = 6,
        basis_dim: int = 96,
        hidden_dim: int = 384,
        branch_depth: int = 5,
        trunk_depth: int = 5,
        activation: str = "tanh",
        skip_init: torch.Tensor | None = None,
        train_skip: bool = True,
        residual_scale: float = 1.0,
        baseline_scale: float = 1.0,
        train_point_baseline: bool = True,
        zero_init_residual: bool = True,
        q_zero_norm: torch.Tensor | None = None,
        le_zero_norm: torch.Tensor | None = None,
        q_raw_mean: torch.Tensor | None = None,
        q_raw_std: torch.Tensor | None = None,
        gate_q0: float = 0.0,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            point_dim=point_dim,
            q_start=q_start,
            q_dim=q_dim,
            ip_count=ip_count,
            strain_dim=strain_dim,
            basis_dim=basis_dim,
            hidden_dim=hidden_dim,
            branch_depth=branch_depth,
            trunk_depth=trunk_depth,
            activation=activation,
            skip_init=skip_init,
            train_skip=train_skip,
            residual_scale=residual_scale,
            baseline_scale=baseline_scale,
            train_point_baseline=train_point_baseline,
            zero_init_residual=zero_init_residual,
            q_zero_norm=q_zero_norm,
            le_zero_norm=le_zero_norm,
            q_raw_mean=q_raw_mean,
            q_raw_std=q_raw_std,
            gate_q0=gate_q0,
        )


class NOEMStyleMIONet(nn.Module):
    """NOEM/MIONet-style TRUE176 operator model.

    This follows the source-code principle used by NOEM's ``MIDeepONet``:
    separate input groups are processed by separate branch networks, their
    latent coefficients are multiplied, and the result is dot-producted with a
    single trunk network.  Here the natural split is

        branch_q(q48) * branch_geometry(X_keep or shape4) * trunk(point) -> LE.

    The model still accepts the trainer's full standardized branch vector so
    existing AD code can differentiate with respect to the q slice.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        point_dim: int,
        q_start: int,
        q_dim: int = 48,
        ip_count: int = 128,
        strain_dim: int = 6,
        basis_dim: int = 96,
        hidden_dim: int = 384,
        branch_depth: int = 5,
        trunk_depth: int = 5,
        activation: str = "tanh",
        use_q_skip: bool = False,
        skip_init: torch.Tensor | None = None,
        train_skip: bool = True,
        product_scale: str = "none",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if point_dim <= 0:
            raise ValueError("point_dim must be positive")
        if int(q_start) < 0 or int(q_dim) <= 0 or int(q_start) + int(q_dim) > int(input_dim):
            raise ValueError("q_start/q_dim must select a valid q slice inside the branch input")
        geom_dim = int(input_dim) - int(q_dim)
        if geom_dim <= 0:
            raise ValueError(
                "NOEMStyleMIONet needs a non-q branch group. Use branch-feature-mode xkeep-qraw, "
                "xnodes-qraw, or shape4-qraw so geometry/parameters are available separately from q48."
            )
        key = str(product_scale).strip().lower()
        if key not in {"none", "sqrt", "basis"}:
            raise ValueError("product_scale must be one of: none, sqrt, basis")

        self.input_dim = int(input_dim)
        self.point_dim = int(point_dim)
        self.q_start = int(q_start)
        self.q_dim = int(q_dim)
        self.ip_count = int(ip_count)
        self.strain_dim = int(strain_dim)
        self.basis_dim = int(basis_dim)
        self.geometry_dim = int(geom_dim)
        self.use_q_skip = bool(use_q_skip)
        self.product_scale = key

        act = activation_module(activation)
        coeff_dim = self.strain_dim * self.basis_dim
        self.q_branch = MLP(self.q_dim, coeff_dim, hidden_dim=hidden_dim, depth=branch_depth, activation=act)
        self.geometry_branch = MLP(self.geometry_dim, coeff_dim, hidden_dim=hidden_dim, depth=branch_depth, activation=act)
        self.trunk = MLP(self.point_dim, coeff_dim, hidden_dim=hidden_dim, depth=trunk_depth, activation=act)
        self.bias = nn.Parameter(torch.zeros(self.strain_dim))

        if skip_init is None:
            skip = torch.zeros(self.ip_count, self.strain_dim, self.q_dim, dtype=torch.float32)
        else:
            skip = torch.as_tensor(skip_init, dtype=torch.float32).reshape(self.ip_count, self.strain_dim, self.q_dim)
        if self.use_q_skip:
            self.skip_weight = nn.Parameter(skip, requires_grad=bool(train_skip))
        else:
            self.register_buffer("skip_weight", skip)

    def _split_branch(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = x_norm[:, self.q_start : self.q_start + self.q_dim]
        left = x_norm[:, : self.q_start]
        right = x_norm[:, self.q_start + self.q_dim :]
        geometry = torch.cat([left, right], dim=-1) if left.shape[-1] or right.shape[-1] else left
        return q, geometry

    def _scale(self) -> float:
        if self.product_scale == "sqrt":
            return float(self.basis_dim) ** 0.5
        if self.product_scale == "basis":
            return float(self.basis_dim)
        return 1.0

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

        qn, gn = self._split_branch(x_norm)
        qb = self.q_branch(qn).view(-1, self.strain_dim, self.basis_dim)
        gb = self.geometry_branch(gn).view(-1, self.strain_dim, self.basis_dim)
        t = self.trunk(point_norm.reshape(-1, self.point_dim)).view(
            x_norm.shape[0], self.ip_count, self.strain_dim, self.basis_dim
        )
        out = torch.einsum("bak,bak,bpak->bpa", qb, gb, t) / self._scale()
        if self.use_q_skip:
            out = out + torch.einsum("paj,bj->bpa", self.skip_weight, qn)
        return out + self.bias.view(1, 1, self.strain_dim)
