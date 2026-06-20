"""Autograd helpers for macro-element tangent quantities."""

from __future__ import annotations

import torch


def strain_jacobian_wrt_q(
    model: torch.nn.Module,
    q_b: torch.Tensor,
    x_nodes: torch.Tensor,
    xi: torch.Tensor,
    create_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute strain and ``B = d epsilon / d q_b`` with autograd.

    Args:
        model: Module returning strain with shape ``[B, strain_dim]``.
        q_b: Boundary displacement tensor ``[B, q_nodes, 3]``.
        x_nodes: Geometry tensor ``[B, geom_nodes, 3]``.
        xi: Query natural coordinates ``[B, 3]``.
        create_graph: Keep derivative graph for higher-order stiffness terms.

    Returns:
        ``(epsilon, B)`` where ``B`` has shape
        ``[B, strain_dim, q_nodes * 3]``.
    """

    q_req = q_b.detach().clone().requires_grad_(True)
    eps = model(q_req, x_nodes, xi)
    rows = []
    for component in range(eps.shape[-1]):
        grad = torch.autograd.grad(
            eps[:, component].sum(),
            q_req,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=False,
        )[0]
        rows.append(grad.flatten(1))
    b_macro = torch.stack(rows, dim=1)
    return eps, b_macro

