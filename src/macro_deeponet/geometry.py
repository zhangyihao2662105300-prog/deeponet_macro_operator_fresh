"""Hex8 isoparametric geometry utilities."""

from __future__ import annotations

import torch


HEX8_NATURAL = torch.tensor(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=torch.float32,
)


def _hex8_signs_like(points: torch.Tensor) -> torch.Tensor:
    return HEX8_NATURAL.to(device=points.device, dtype=points.dtype)


def shape_functions_hex8(points: torch.Tensor) -> torch.Tensor:
    """Return Hex8 shape functions at natural points.

    Args:
        points: Tensor with shape ``[..., 3]`` containing ``xi, eta, zeta``.

    Returns:
        Tensor with shape ``[..., 8]``.
    """

    signs = _hex8_signs_like(points)
    xi = points[..., 0:1]
    eta = points[..., 1:2]
    zeta = points[..., 2:3]
    r = signs[:, 0]
    s = signs[:, 1]
    t = signs[:, 2]
    return 0.125 * (1.0 + xi * r) * (1.0 + eta * s) * (1.0 + zeta * t)


def shape_function_gradients_hex8(points: torch.Tensor) -> torch.Tensor:
    """Return gradients dN/d(xi, eta, zeta) for Hex8.

    Args:
        points: Tensor with shape ``[..., 3]``.

    Returns:
        Tensor with shape ``[..., 8, 3]``.
    """

    signs = _hex8_signs_like(points)
    xi = points[..., 0:1]
    eta = points[..., 1:2]
    zeta = points[..., 2:3]

    r = signs[:, 0]
    s = signs[:, 1]
    t = signs[:, 2]

    dxi = 0.125 * r * (1.0 + eta * s) * (1.0 + zeta * t)
    deta = 0.125 * (1.0 + xi * r) * s * (1.0 + zeta * t)
    dzeta = 0.125 * (1.0 + xi * r) * (1.0 + eta * s) * t
    return torch.stack([dxi, deta, dzeta], dim=-1)


def jacobian_hex8(nodes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Return physical Jacobian dx/dnatural for one query point per sample.

    Args:
        nodes: Tensor with shape ``[..., 8, 3]``.
        points: Tensor with shape ``[..., 3]``.

    Returns:
        Tensor with shape ``[..., 3, 3]`` where rows are physical components
        and columns are natural-coordinate derivatives.
    """

    dndxi = shape_function_gradients_hex8(points)
    return torch.einsum("...ai,...aj->...ij", nodes, dndxi)


def det_jacobian_hex8(nodes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Return determinant of the Hex8 Jacobian."""

    return torch.linalg.det(jacobian_hex8(nodes, points))


def jacobian_features(nodes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Return flattened local geometry features ``[J(:), detJ]``."""

    jac = jacobian_hex8(nodes, points)
    det = torch.linalg.det(jac).unsqueeze(-1)
    return torch.cat([jac.reshape(*jac.shape[:-2], 9), det], dim=-1)


def normalize_nodes(nodes: torch.Tensor, eps: float = 1.0e-8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Center and scale nodes per sample.

    The returned scale is a scalar RMS radius per sample. This removes global
    translation and keeps geometry magnitudes in a stable range.
    """

    center = nodes.mean(dim=-2, keepdim=True)
    centered = nodes - center
    scale = centered.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(eps)
    return centered / scale, center, scale


def map_to_physical(nodes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Map natural points to physical coordinates with Hex8 interpolation."""

    n = shape_functions_hex8(points)
    return torch.einsum("...a,...ai->...i", n, nodes)

