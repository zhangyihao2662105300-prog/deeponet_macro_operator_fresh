"""Macro16 boundary-control solid-shell geometry utilities.

The Macro16 contract uses only the 16 boundary control nodes visible to the
operator: 8 serendipity nodes on the lower surface and the matching 8 nodes on
the upper surface.  No internal CSS8 grid nodes are used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

MACRO16_CONTRACT_VERSION = "v4-macro16-boundary-operator-001"

SURFACE8_NATURAL = np.asarray(
    [
        [-1.0, -1.0],
        [0.0, -1.0],
        [1.0, -1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 0.0],
    ],
    dtype=np.float64,
)
MACRO16_NATURAL = np.asarray(
    [[r, s, -1.0] for r, s in SURFACE8_NATURAL] + [[r, s, 1.0] for r, s in SURFACE8_NATURAL],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Macro16PointTable:
    xi: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        xi = np.asarray(self.xi, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if xi.ndim != 2 or xi.shape[1] != 3:
            raise ValueError(f"xi must have shape [P,3], got {xi.shape}")
        if weights.shape[0] != xi.shape[0]:
            raise ValueError(f"weights length {weights.shape[0]} does not match point count {xi.shape[0]}")
        object.__setattr__(self, "xi", xi)
        object.__setattr__(self, "weights", weights)


def macro16_standard_point_table(plane_order: int = 3, thickness_order: int = 2) -> Macro16PointTable:
    """Return full macro-element Gauss points in one parent domain."""

    if int(plane_order) < 1 or int(thickness_order) < 1:
        raise ValueError("Gauss orders must be positive")
    gp_r, gw_r = np.polynomial.legendre.leggauss(int(plane_order))
    gp_s, gw_s = np.polynomial.legendre.leggauss(int(plane_order))
    gp_t, gw_t = np.polynomial.legendre.leggauss(int(thickness_order))
    xi: list[list[float]] = []
    weights: list[float] = []
    for t, wt in zip(gp_t, gw_t):
        for s, ws in zip(gp_s, gw_s):
            for r, wr in zip(gp_r, gw_r):
                xi.append([float(r), float(s), float(t)])
                weights.append(float(wr * ws * wt))
    return Macro16PointTable(xi=np.asarray(xi, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64))


def _conventional_quad8_shape(rs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(rs, dtype=np.float64)
    if pts.shape[-1] != 2:
        raise ValueError(f"rs last dimension must be 2, got {pts.shape}")
    r = pts[..., 0]
    s = pts[..., 1]
    n = np.stack(
        [
            0.25 * (1.0 - r) * (1.0 - s) * (-r - s - 1.0),
            0.25 * (1.0 + r) * (1.0 - s) * (r - s - 1.0),
            0.25 * (1.0 + r) * (1.0 + s) * (r + s - 1.0),
            0.25 * (1.0 - r) * (1.0 + s) * (-r + s - 1.0),
            0.5 * (1.0 - r * r) * (1.0 - s),
            0.5 * (1.0 + r) * (1.0 - s * s),
            0.5 * (1.0 - r * r) * (1.0 + s),
            0.5 * (1.0 - r) * (1.0 - s * s),
        ],
        axis=-1,
    )
    dn_dr = np.stack(
        [
            0.25 * (1.0 - s) * (2.0 * r + s),
            0.25 * (1.0 - s) * (2.0 * r - s),
            0.25 * (1.0 + s) * (2.0 * r + s),
            0.25 * (1.0 + s) * (2.0 * r - s),
            -r * (1.0 - s),
            0.5 * (1.0 - s * s),
            -r * (1.0 + s),
            -0.5 * (1.0 - s * s),
        ],
        axis=-1,
    )
    dn_ds = np.stack(
        [
            0.25 * (1.0 - r) * (r + 2.0 * s),
            0.25 * (1.0 + r) * (-r + 2.0 * s),
            0.25 * (1.0 + r) * (r + 2.0 * s),
            0.25 * (1.0 - r) * (-r + 2.0 * s),
            -0.5 * (1.0 - r * r),
            -(1.0 + r) * s,
            0.5 * (1.0 - r * r),
            -(1.0 - r) * s,
        ],
        axis=-1,
    )
    grad = np.stack([dn_dr, dn_ds], axis=-1)
    return n, grad


def shape_functions_surface8(rs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Q8 serendipity shape values in the Macro16 surface-node order."""

    n_conv, grad_conv = _conventional_quad8_shape(rs)
    order = np.asarray([0, 4, 1, 5, 2, 6, 3, 7], dtype=np.int64)
    return n_conv[..., order], grad_conv[..., order, :]


def shape_functions_macro16(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Macro16 shape values and gradients dN/d(r,s,t)."""

    pts = np.asarray(xi, dtype=np.float64)
    if pts.shape[-1] != 3:
        raise ValueError(f"xi last dimension must be 3, got {pts.shape}")
    q8, dq8 = shape_functions_surface8(pts[..., :2])
    t = pts[..., 2]
    l_bot = 0.5 * (1.0 - t)
    l_top = 0.5 * (1.0 + t)
    n_bot = l_bot[..., None] * q8
    n_top = l_top[..., None] * q8
    n = np.concatenate([n_bot, n_top], axis=-1)

    grad = np.zeros(pts.shape[:-1] + (16, 3), dtype=np.float64)
    grad[..., :8, :2] = l_bot[..., None, None] * dq8
    grad[..., 8:, :2] = l_top[..., None, None] * dq8
    grad[..., :8, 2] = -0.5 * q8
    grad[..., 8:, 2] = 0.5 * q8
    return n, grad


def characteristic_length(x16: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    nodes = np.asarray(x16, dtype=np.float64).reshape(16, 3)
    center = np.mean(nodes, axis=0)
    span = np.max(nodes, axis=0) - np.min(nodes, axis=0)
    l_ref = float(np.max(span))
    if not math.isfinite(l_ref) or l_ref <= 1.0e-14:
        l_ref = float(np.sqrt(np.mean(np.sum((nodes - center.reshape(1, 3)) ** 2, axis=1))))
    if not math.isfinite(l_ref) or l_ref <= 1.0e-14:
        raise ValueError("cannot compute nondegenerate L_ref from X16")
    return l_ref, center.astype(np.float64), span.astype(np.float64)


def _normalize(vec: np.ndarray, *, eps: float = 1.0e-14) -> np.ndarray:
    vals = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vals))
    if not math.isfinite(norm) or norm <= float(eps):
        raise ValueError(f"degenerate vector norm {norm:g}")
    return vals / norm


def frame_from_j(jmat: np.ndarray) -> np.ndarray:
    """Return local frame columns e1 e2 e3 from the macro surface tangents."""

    j = np.asarray(jmat, dtype=np.float64).reshape(3, 3)
    g_r, g_s, g_t = j[0], j[1], j[2]
    e1 = _normalize(g_r)
    normal = np.cross(g_r, g_s)
    if float(np.linalg.norm(normal)) <= 1.0e-14:
        normal = g_t
    e3 = _normalize(normal)
    e2 = _normalize(np.cross(e3, e1))
    e1 = _normalize(np.cross(e2, e3))
    return np.stack([e1, e2, e3], axis=1)


class Macro16GeometryMap:
    """Single Macro16 isoparametric geometry map built only from X16."""

    def __init__(
        self,
        x16: np.ndarray,
        *,
        center: np.ndarray | None = None,
        l_ref: float | None = None,
    ) -> None:
        self.x16 = np.asarray(x16, dtype=np.float64).reshape(16, 3)
        if center is None or l_ref is None:
            computed_l_ref, computed_center, computed_span = characteristic_length(self.x16)
        else:
            computed_l_ref = float(l_ref)
            computed_center = np.asarray(center, dtype=np.float64).reshape(3)
            computed_span = np.max(self.x16, axis=0) - np.min(self.x16, axis=0)
        if not math.isfinite(computed_l_ref) or computed_l_ref <= 1.0e-14:
            raise ValueError(f"L_ref must be finite and positive, got {computed_l_ref:g}")
        self.l_ref = float(computed_l_ref)
        self.center = computed_center.astype(np.float64)
        self.span = computed_span.astype(np.float64)
        self.span_hat = self.span / self.l_ref
        self.x16_hat = (self.x16 - self.center.reshape(1, 3)) / self.l_ref

    def eval_points(self, point_table: Macro16PointTable | np.ndarray) -> dict[str, np.ndarray]:
        if isinstance(point_table, Macro16PointTable):
            xi = point_table.xi
            gauss_weights = point_table.weights
        else:
            xi = np.asarray(point_table, dtype=np.float64).reshape(-1, 3)
            gauss_weights = np.ones(xi.shape[0], dtype=np.float64)
        n, dndxi = shape_functions_macro16(xi)
        x_hat = np.einsum("pa,ai->pi", n, self.x16_hat)
        j_hat = np.einsum("pai,aj->pij", dndxi, self.x16_hat)
        detj_hat = np.linalg.det(j_hat)
        invj_hat = np.linalg.inv(j_hat)
        q_stack = np.asarray([frame_from_j(jmat) for jmat in j_hat], dtype=np.float64)
        metric = np.einsum("pij,pkj->pik", j_hat, j_hat)
        thickness = 2.0 * np.linalg.norm(j_hat[:, 2, :], axis=1)
        integration_weights = gauss_weights * np.abs(detj_hat)
        return {
            "xi": xi.astype(np.float64),
            "gauss_weight": gauss_weights.astype(np.float64),
            "x_hat": x_hat.astype(np.float64),
            "J_hat": j_hat.astype(np.float64),
            "invJ_hat": invj_hat.astype(np.float64),
            "detJ_hat": detj_hat.astype(np.float64),
            "Q": q_stack.astype(np.float64),
            "metric_hat": metric.astype(np.float64),
            "thickness_hat": thickness.astype(np.float64),
            "integration_weight_hat": integration_weights.astype(np.float64),
            "log_abs_detJ_hat": np.log(np.maximum(np.abs(detj_hat), 1.0e-30)).astype(np.float64),
        }

    def build_point_features(self, point_table: Macro16PointTable | np.ndarray) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
        fields = self.eval_points(point_table)
        xi = fields["xi"]
        j = fields["J_hat"]
        parts = [
            xi,
            xi * xi,
            fields["x_hat"],
            fields["Q"].reshape(xi.shape[0], 9),
            j.reshape(xi.shape[0], 9),
            fields["invJ_hat"].reshape(xi.shape[0], 9),
            fields["metric_hat"].reshape(xi.shape[0], 9),
            fields["thickness_hat"].reshape(xi.shape[0], 1),
            fields["detJ_hat"].reshape(xi.shape[0], 1),
            fields["log_abs_detJ_hat"].reshape(xi.shape[0], 1),
            fields["gauss_weight"].reshape(xi.shape[0], 1),
            fields["integration_weight_hat"].reshape(xi.shape[0], 1),
        ]
        names = (
            ["ip_macro_r", "ip_macro_s", "ip_macro_t"]
            + ["ip_macro_r_sq", "ip_macro_s_sq", "ip_macro_t_sq"]
            + ["ip_xyz_hat_x", "ip_xyz_hat_y", "ip_xyz_hat_z"]
            + [f"Q_stack_{i}{jcol}" for i in range(3) for jcol in range(3)]
            + [f"ip_J_hat_{i}{jcol}" for i in range(3) for jcol in range(3)]
            + [f"ip_invJ_hat_{i}{jcol}" for i in range(3) for jcol in range(3)]
            + [f"metric_hat_{i}{jcol}" for i in range(3) for jcol in range(3)]
            + ["thickness_hat", "ip_detJ_hat", "log_abs_ip_detJ_hat", "gauss_weight", "integration_weight_hat"]
        )
        return np.concatenate(parts, axis=1).astype(np.float32), names, fields

    def build_global_features(self) -> tuple[np.ndarray, list[str]]:
        names = [f"X16_hat_{idx}_{axis}" for idx in range(16) for axis in ("x", "y", "z")]
        names.extend(["span_hat_x", "span_hat_y", "span_hat_z"])
        return np.concatenate([self.x16_hat.reshape(-1), self.span_hat.reshape(3)], axis=0).astype(np.float32), names


def build_macro16_feature_batch(
    x16: np.ndarray,
    point_table: Macro16PointTable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return X16_hat, point features, weights, and L_ref for a frame batch."""

    nodes = np.asarray(x16, dtype=np.float64).reshape(-1, 16, 3)
    geom_hat = np.empty_like(nodes, dtype=np.float32)
    point_features: list[np.ndarray] = []
    weights = np.empty((nodes.shape[0], point_table.xi.shape[0]), dtype=np.float32)
    l_ref = np.empty((nodes.shape[0], 1), dtype=np.float32)
    center = np.empty((nodes.shape[0], 3), dtype=np.float32)
    names: list[str] | None = None
    for i, frame_nodes in enumerate(nodes):
        geom = Macro16GeometryMap(frame_nodes)
        feats, feat_names, fields = geom.build_point_features(point_table)
        geom_hat[i] = geom.x16_hat.astype(np.float32)
        point_features.append(feats)
        weights[i] = fields["integration_weight_hat"].astype(np.float32)
        l_ref[i, 0] = float(geom.l_ref)
        center[i] = geom.center.astype(np.float32)
        names = feat_names
    meta = {
        "standard_operator_contract_version": MACRO16_CONTRACT_VERSION,
        "geometry_source": "X16_boundary_control_nodes",
        "macro_shape": "single_macro16_serendipity_surface_linear_thickness",
        "point_count": int(point_table.xi.shape[0]),
        "point_feature_names": names or [],
        "point_feature_dim": int(point_features[0].shape[1]) if point_features else 0,
        "gauss_point_rule": "one_macro_parent_domain",
    }
    point = np.stack(point_features, axis=0).astype(np.float32) if point_features else np.empty((0, 0, 0), dtype=np.float32)
    return geom_hat, point, weights, l_ref, {"X_center": center, **meta}
