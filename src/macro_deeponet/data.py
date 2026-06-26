"""Synthetic data for the first fresh macro-operator implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .geometry import HEX8_NATURAL, det_jacobian_hex8


@dataclass(frozen=True)
class SyntheticConfig:
    num_samples: int = 4096
    seed: int = 1234
    min_length: float = 0.7
    max_length: float = 1.6
    geometry_shear: float = 0.22
    corner_warp: float = 0.05
    affine_grad: float = 0.08
    rigid_translation: float = 0.12
    rigid_rotation: float = 0.10
    rigid_probability: float = 0.25


class SyntheticMacroDataset(Dataset):
    """Pre-generated synthetic macro-element samples.

    Each row is one macro-element state and one query coordinate.
    """

    def __init__(self, config: SyntheticConfig) -> None:
        self.config = config
        gen = torch.Generator().manual_seed(config.seed)
        samples = [self._make_sample(gen) for _ in range(config.num_samples)]
        self.q_b = torch.stack([s["q_b"] for s in samples], dim=0)
        self.x_nodes = torch.stack([s["x_nodes"] for s in samples], dim=0)
        self.xi = torch.stack([s["xi"] for s in samples], dim=0)
        self.epsilon = torch.stack([s["epsilon"] for s in samples], dim=0)
        self.is_rigid = torch.stack([s["is_rigid"] for s in samples], dim=0)

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "q_b": self.q_b[idx],
            "x_nodes": self.x_nodes[idx],
            "xi": self.xi[idx],
            "epsilon": self.epsilon[idx],
            "is_rigid": self.is_rigid[idx],
        }

    def _make_geometry(self, gen: torch.Generator) -> torch.Tensor:
        signs = HEX8_NATURAL.clone()
        for _ in range(100):
            lengths = self.config.min_length + (
                self.config.max_length - self.config.min_length
            ) * torch.rand(3, generator=gen)
            base = 0.5 * signs * lengths

            shear = torch.eye(3)
            offdiag = self.config.geometry_shear * (2.0 * torch.rand(3, 3, generator=gen) - 1.0)
            shear = shear + offdiag * (1.0 - torch.eye(3))
            nodes = base @ shear.T

            warp = self.config.corner_warp * lengths.mean() * (
                2.0 * torch.rand(8, 3, generator=gen) - 1.0
            )
            nodes = nodes + warp
            test_points = torch.cat(
                [
                    torch.zeros(1, 3),
                    0.57735026919 * signs,
                ],
                dim=0,
            )
            repeated_nodes = nodes.unsqueeze(0).expand(test_points.shape[0], -1, -1)
            if det_jacobian_hex8(repeated_nodes, test_points).min() > 0.02:
                return nodes
        raise RuntimeError("Failed to generate a positive-Jacobian Hex8 geometry")

    def _make_sample(self, gen: torch.Generator) -> dict[str, torch.Tensor]:
        x_nodes = self._make_geometry(gen)
        xi = 2.0 * torch.rand(3, generator=gen) - 1.0

        mode_draw = torch.rand((), generator=gen).item()
        is_rigid = mode_draw < self.config.rigid_probability
        if is_rigid:
            if torch.rand((), generator=gen).item() < 0.5:
                grad_u = torch.zeros(3, 3)
            else:
                omega = self.config.rigid_rotation * (2.0 * torch.rand(3, generator=gen) - 1.0)
                grad_u = torch.tensor(
                    [
                        [0.0, -omega[2], omega[1]],
                        [omega[2], 0.0, -omega[0]],
                        [-omega[1], omega[0], 0.0],
                    ]
                )
            translation = self.config.rigid_translation * (2.0 * torch.rand(3, generator=gen) - 1.0)
        else:
            grad_u = self.config.affine_grad * (2.0 * torch.rand(3, 3, generator=gen) - 1.0)
            translation = self.config.rigid_translation * (2.0 * torch.rand(3, generator=gen) - 1.0)

        q_b = x_nodes @ grad_u.T + translation
        epsilon = engineering_strain_from_grad_u(grad_u)
        return {
            "q_b": q_b.float(),
            "x_nodes": x_nodes.float(),
            "xi": xi.float(),
            "epsilon": epsilon.float(),
            "is_rigid": torch.tensor(is_rigid, dtype=torch.bool),
        }


def engineering_strain_from_grad_u(grad_u: torch.Tensor) -> torch.Tensor:
    """Convert displacement gradient to 6 engineering strain components."""

    return torch.stack(
        [
            grad_u[0, 0],
            grad_u[1, 1],
            grad_u[2, 2],
            grad_u[0, 1] + grad_u[1, 0],
            grad_u[0, 2] + grad_u[2, 0],
            grad_u[1, 2] + grad_u[2, 1],
        ]
    )
