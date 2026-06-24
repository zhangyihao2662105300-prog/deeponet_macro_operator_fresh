#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the v3 CSS8 standard-operator compact contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_v2_local_strain_pilot_compact import strain_transform_matrices  # noqa: E402
from macro_deeponet.true176_data import GAUSS_POINTS, standard_css8_row_map  # noqa: E402


CONTRACT_VERSION = "v3-css8-standard-operator-001"
MODEL_VISIBLE_FIELDS = (
    "q_useful_hat",
    "geometry_global_hat",
    "ip_macro_xi",
    "ip_local_rst",
    "local_geometry_features_hat",
    "trunk_features_hat",
    "LE_local_stack",
    "B_local_useful_stack_hat",
)
HEX8_NODE_SIGNS = np.asarray(
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
    dtype=np.float64,
)


@dataclass(frozen=True)
class PointTable:
    cell_id: np.ndarray
    rst: np.ndarray
    xi_macro: np.ndarray | None = None
    row_map: np.ndarray | None = None

    def __post_init__(self) -> None:
        cell_id = np.asarray(self.cell_id, dtype=np.int64).reshape(-1)
        rst = np.asarray(self.rst, dtype=np.float64)
        if rst.ndim != 2 or rst.shape[1] != 3:
            raise ValueError(f"rst must be [P,3], got {rst.shape}")
        if cell_id.shape[0] != rst.shape[0]:
            raise ValueError(f"cell_id length {cell_id.shape[0]} does not match rst rows {rst.shape[0]}")
        if self.xi_macro is not None:
            xi = np.asarray(self.xi_macro, dtype=np.float64)
            if xi.shape != rst.shape:
                raise ValueError(f"xi_macro must match rst shape {rst.shape}, got {xi.shape}")
            object.__setattr__(self, "xi_macro", xi)
        if self.row_map is not None:
            object.__setattr__(self, "row_map", np.asarray(self.row_map, dtype=np.float64))
        object.__setattr__(self, "cell_id", cell_id)
        object.__setattr__(self, "rst", rst)


def hex8_shape(rst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Hex8 shape values and dN/d(r,s,t) for one parent-domain point."""

    r, s, t = [float(v) for v in np.asarray(rst, dtype=np.float64).reshape(3)]
    n = np.empty(8, dtype=np.float64)
    dndr = np.empty((8, 3), dtype=np.float64)
    for a, (ra, sa, ta) in enumerate(HEX8_NODE_SIGNS):
        n[a] = 0.125 * (1.0 + ra * r) * (1.0 + sa * s) * (1.0 + ta * t)
        dndr[a, 0] = 0.125 * ra * (1.0 + sa * s) * (1.0 + ta * t)
        dndr[a, 1] = 0.125 * sa * (1.0 + ra * r) * (1.0 + ta * t)
        dndr[a, 2] = 0.125 * ta * (1.0 + ra * r) * (1.0 + sa * s)
    return n, dndr


def geometry_scale(
    x_ref: np.ndarray,
    *,
    center: np.ndarray | None = None,
    l_ref: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the single reference-geometry center/span/scale convention."""

    x = np.asarray(x_ref, dtype=np.float64).reshape(-1, 3)
    if center is None or l_ref is None:
        computed_l_ref, computed_center, span = characteristic_length(x)
    else:
        computed_center = np.asarray(center, dtype=np.float64).reshape(3)
        span = np.max(x, axis=0) - np.min(x, axis=0)
        computed_l_ref = float(l_ref)
    if not math.isfinite(computed_l_ref) or computed_l_ref <= 1.0e-14:
        raise ValueError(f"L_ref must be finite and positive, got {computed_l_ref:g}")
    return float(computed_l_ref), computed_center.astype(np.float64), span.astype(np.float64)


def scale_reference_geometry(
    x_ref: np.ndarray,
    *,
    center: np.ndarray | None = None,
    l_ref: float | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Normalize reference coordinates once as X_hat=(X_ref-X_center)/L_ref."""

    l_scale, x_center, x_span = geometry_scale(x_ref, center=center, l_ref=l_ref)
    x_hat = (np.asarray(x_ref, dtype=np.float64).reshape(-1, 3) - x_center.reshape(1, 3)) / l_scale
    return x_hat, l_scale, x_center, x_span


def css8_connectivity_zero_based(nx: int = 4, ny: int = 4) -> np.ndarray:
    """Return structured CSS8 subcell connectivity as [nx*ny,8], zero-based."""

    nx_i = int(nx)
    ny_i = int(ny)
    if nx_i < 1 or ny_i < 1:
        raise ValueError("nx and ny must be positive")

    def node_id(i: int, j: int, k: int) -> int:
        return int(k) * (nx_i + 1) * (ny_i + 1) + int(j) * (nx_i + 1) + int(i)

    cells: list[list[int]] = []
    for j in range(ny_i):
        for i in range(nx_i):
            cells.append([
                node_id(i, j, 0),
                node_id(i + 1, j, 0),
                node_id(i + 1, j + 1, 0),
                node_id(i, j + 1, 0),
                node_id(i, j, 1),
                node_id(i + 1, j, 1),
                node_id(i + 1, j + 1, 1),
                node_id(i, j + 1, 1),
            ])
    return np.asarray(cells, dtype=np.int64)


def css8_point_table(nx: int = 4, ny: int = 4) -> PointTable:
    """Return the two-level CSS8 point table: macro xi plus local rst."""

    row, ip_macro_xi, ip_local_rst = css8_standard_coordinates(nx=nx, ny=ny)
    cell_id = row[:, 1].astype(np.int64) - 1
    return PointTable(cell_id=cell_id, rst=ip_local_rst, xi_macro=ip_macro_xi, row_map=row)


def shell_normal_frame_from_j(jmat: np.ndarray) -> np.ndarray:
    """Return Q columns [e1,e2,e3] from the reference surface normal."""

    j = np.asarray(jmat, dtype=np.float64).reshape(3, 3)
    g1, g2, g3 = j[0], j[1], j[2]
    e1 = normalize(g1)
    normal = np.cross(g1, g2)
    if float(np.linalg.norm(normal)) <= 1.0e-14:
        normal = g3
    e3 = normalize(normal)
    e2 = normalize(np.cross(e3, e1))
    e1 = normalize(np.cross(e2, e3))
    return np.stack([e1, e2, e3], axis=1)


def gram_schmidt_frame_from_j(jmat: np.ndarray) -> np.ndarray:
    """Return an orthonormal frame from the reference Jacobian rows."""

    j = np.asarray(jmat, dtype=np.float64).reshape(3, 3)
    g1, g2, _g3 = j[0], j[1], j[2]
    e1 = normalize(g1)
    g2_projected = g2 - float(np.dot(g2, e1)) * e1
    e2 = normalize(g2_projected)
    e3 = normalize(np.cross(e1, e2))
    return np.stack([e1, e2, e3], axis=1)


def reference_frame_from_j(jmat: np.ndarray, *, mode: str) -> np.ndarray:
    key = str(mode).strip().lower().replace("_", "-")
    if key in {"shell-normal", "surface-normal", "shell"}:
        return shell_normal_frame_from_j(jmat)
    if key in {"stack-director", "css8-stack", "director"}:
        return stack_director_frame_from_j(jmat)
    if key in {"gram-schmidt", "hex8", "solid"}:
        return gram_schmidt_frame_from_j(jmat)
    raise ValueError(f"unknown reference frame mode {mode!r}")


class GeometryMap:
    """Reference-configuration isoparametric geometry map.

    The map owns the single normalization convention for reference geometry and
    builds both point-level trunk features and global geometry features from
    ``X_hat``.  It does not depend on the current displacement state ``q``.
    """

    def __init__(
        self,
        X_ref: np.ndarray,
        conn: np.ndarray,
        *,
        cell_type: str = "HEX8",
        center: np.ndarray | None = None,
        L_ref: float | None = None,
        detj_scale_dim: int = 3,
        frame_mode: str = "shell-normal",
    ) -> None:
        self.X_ref = np.asarray(X_ref, dtype=np.float64).reshape(-1, 3)
        self.conn = np.asarray(conn, dtype=np.int64)
        if self.conn.ndim != 2:
            raise ValueError(f"conn must be [n_cell,n_enode], got {self.conn.shape}")
        if int(np.min(self.conn)) < 0 or int(np.max(self.conn)) >= self.X_ref.shape[0]:
            raise ValueError("conn indices must be zero-based and within X_ref")
        self.cell_type = str(cell_type).upper().replace("-", "_")
        if self.cell_type not in {"HEX8", "CSS8", "SHELL_LIKE"}:
            raise ValueError("cell_type must be HEX8, CSS8, or shell-like")
        if self.conn.shape[1] != 8:
            raise ValueError(f"{self.cell_type} expects 8-node cells, got {self.conn.shape[1]}")
        self.X_hat, self.L_ref, self.center, self.span = scale_reference_geometry(
            self.X_ref,
            center=center,
            l_ref=L_ref,
        )
        self.span_hat = self.span / float(self.L_ref)
        self.detj_scale_dim = int(detj_scale_dim)
        self.frame_mode = str(frame_mode)

    def shape(self, rst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return hex8_shape(rst)

    def eval_point(
        self,
        cell_id: int,
        rst: np.ndarray,
        xi_macro: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | float | int]:
        cell = int(cell_id)
        if cell < 0 or cell >= self.conn.shape[0]:
            raise IndexError(f"cell_id {cell} outside [0,{self.conn.shape[0]})")
        n, dndr = self.shape(rst)
        nodes_hat = self.X_hat[self.conn[cell]]
        x_hat = np.asarray(n @ nodes_hat, dtype=np.float64)
        j_hat = np.asarray(dndr.T @ nodes_hat, dtype=np.float64)
        invj_hat = np.linalg.inv(j_hat)
        detj_hat = float(np.linalg.det(j_hat))
        q_ref = reference_frame_from_j(j_hat, mode=self.frame_mode)
        metric = np.einsum("ij,kj->ik", j_hat, j_hat)
        thickness_hat = 2.0 * float(np.linalg.norm(j_hat[2]))
        return {
            "cell_id": cell,
            "rst": np.asarray(rst, dtype=np.float64).reshape(3),
            "xi_macro": None if xi_macro is None else np.asarray(xi_macro, dtype=np.float64).reshape(3),
            "x_hat": x_hat,
            "J_hat": j_hat,
            "invJ_hat": invj_hat,
            "detJ_hat": detj_hat,
            "Q": q_ref,
            "metric_hat": metric,
            "thickness_hat": thickness_hat,
            "log_abs_detJ_hat": float(np.log(max(abs(detj_hat), 1.0e-30))),
        }

    def eval_points(self, point_table: PointTable | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        table = coerce_point_table(point_table)
        point_count = int(table.rst.shape[0])
        x_hat = np.empty((point_count, 3), dtype=np.float64)
        j_hat = np.empty((point_count, 3, 3), dtype=np.float64)
        invj_hat = np.empty((point_count, 3, 3), dtype=np.float64)
        detj_hat = np.empty(point_count, dtype=np.float64)
        q_ref = np.empty((point_count, 3, 3), dtype=np.float64)
        metric = np.empty((point_count, 3, 3), dtype=np.float64)
        thickness = np.empty(point_count, dtype=np.float64)
        for idx in range(point_count):
            xi = None if table.xi_macro is None else table.xi_macro[idx]
            rec = self.eval_point(int(table.cell_id[idx]), table.rst[idx], xi_macro=xi)
            x_hat[idx] = np.asarray(rec["x_hat"], dtype=np.float64)
            j_hat[idx] = np.asarray(rec["J_hat"], dtype=np.float64)
            invj_hat[idx] = np.asarray(rec["invJ_hat"], dtype=np.float64)
            detj_hat[idx] = float(rec["detJ_hat"])
            q_ref[idx] = np.asarray(rec["Q"], dtype=np.float64)
            metric[idx] = np.asarray(rec["metric_hat"], dtype=np.float64)
            thickness[idx] = float(rec["thickness_hat"])
        return {
            "cell_id": table.cell_id.astype(np.int64),
            "rst": table.rst.astype(np.float64),
            "xi_macro": None if table.xi_macro is None else table.xi_macro.astype(np.float64),
            "x_hat": x_hat,
            "J_hat": j_hat,
            "invJ_hat": invj_hat,
            "detJ_hat": detj_hat,
            "Q": q_ref,
            "metric_hat": metric,
            "thickness_hat": thickness,
            "log_abs_detJ_hat": np.log(np.maximum(np.abs(detj_hat), 1.0e-30)),
        }

    def build_trunk_features(
        self,
        point_table: PointTable | dict[str, np.ndarray],
    ) -> tuple[np.ndarray, list[str], np.ndarray, list[str], dict[str, np.ndarray]]:
        table = coerce_point_table(point_table)
        fields = self.eval_points(table)
        local, local_names, trunk, trunk_names = local_geometry_features(
            ip_macro_xi=np.zeros_like(table.rst) if table.xi_macro is None else table.xi_macro,
            ip_local_rst=table.rst,
            ip_xyz_hat=fields["x_hat"],
            q_stack=fields["Q"],
            ip_j_hat=fields["J_hat"],
            ip_invj_hat=fields["invJ_hat"],
            ip_detj_hat=fields["detJ_hat"],
        )
        return local, local_names, trunk, trunk_names, fields

    def build_global_features(
        self,
        *,
        shape_params: np.ndarray | None = None,
        prefix: str = "X_macro_hat",
        shape_prefix: str = "shape_param",
    ) -> tuple[np.ndarray, list[str]]:
        parts = [self.X_hat.reshape(-1)]
        names = [f"{prefix}_{idx}_{axis}" for idx in range(self.X_hat.shape[0]) for axis in ("x", "y", "z")]
        parts.append(self.span_hat.reshape(3))
        names.extend(["span_hat_x", "span_hat_y", "span_hat_z"])
        if shape_params is not None:
            shape = np.asarray(shape_params, dtype=np.float64).reshape(-1)
            parts.append(shape)
            names.extend([f"{shape_prefix}_{idx}" for idx in range(shape.shape[0])])
        return np.concatenate(parts, axis=0), names

    def physical_point_fields(self, fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        l_ref = float(self.L_ref)
        return {
            "ip_xyz": fields["x_hat"] * l_ref + self.center.reshape(1, 3),
            "ip_J": fields["J_hat"] * l_ref,
            "ip_invJ": fields["invJ_hat"] / l_ref,
            "ip_detJ": fields["detJ_hat"] * (l_ref ** self.detj_scale_dim),
        }


def coerce_point_table(point_table: PointTable | dict[str, np.ndarray]) -> PointTable:
    if isinstance(point_table, PointTable):
        return point_table
    if not isinstance(point_table, dict):
        raise TypeError("point_table must be a PointTable or dict")
    return PointTable(
        cell_id=np.asarray(point_table["cell_id"], dtype=np.int64),
        rst=np.asarray(point_table["rst"], dtype=np.float64),
        xi_macro=None if point_table.get("xi_macro") is None else np.asarray(point_table["xi_macro"], dtype=np.float64),
        row_map=None if point_table.get("row_map") is None else np.asarray(point_table["row_map"], dtype=np.float64),
    )
AUDIT_POSTPROCESS_FIELDS = (
    "q48_raw",
    "q_useful",
    "T_q_raw_to_useful",
    "T_q_raw_to_useful_hat",
    "LE_abq",
    "B_LE128_forward",
    "Q_stack",
    "T_eps_to_abq_stack",
    "T_eps_from_abq_stack",
    "L_ref",
    "X_center",
    "X_macro",
    "X_macro_hat",
    "ip_J",
    "ip_J_hat",
    "ip_invJ",
    "ip_invJ_hat",
    "ip_detJ",
    "ip_detJ_hat",
)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def read_path_list(path: Path) -> list[Path]:
    return [
        Path(line.strip()).resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def collect_paths(compact: Iterable[Path], compact_list: Iterable[Path], *, case_limit: int | None = None) -> list[Path]:
    paths: list[Path] = [Path(p).resolve() for p in compact]
    for list_path in compact_list:
        paths.extend(read_path_list(Path(list_path)))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if case_limit is not None:
        unique = unique[: int(case_limit)]
    return unique


def scalar_text(value: Any, default: str = "unknown") -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return str(default)
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def scalar_bool(value: Any, default: bool = False) -> bool:
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return bool(default)
        return bool(arr.reshape(-1)[0])
    except Exception:
        return bool(default)


def scalar_int(value: Any, default: int | None = None) -> int | None:
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return default
        return int(arr.reshape(-1)[0])
    except Exception:
        return default


def parse_case_id(path: Path, files: np.lib.npyio.NpzFile | None = None) -> int:
    if files is not None:
        for key in ("v2b_pilot_case_id", "v2a_pilot_case_id", "case_id"):
            if key in files.files:
                value = scalar_int(files[key])
                if value is not None:
                    return int(value)
    match = re.search(r"case[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot parse case id from {path}")
    return int(match.group(1))


def first_key(files: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    present = set(files)
    for key in candidates:
        if key in present:
            return key
    return None


def max_abs(value: np.ndarray) -> float:
    vals = np.asarray(value, dtype=np.float64)
    return float(np.max(np.abs(vals))) if vals.size else 0.0


def rel_norm(diff: np.ndarray, ref: np.ndarray) -> float:
    den = max(float(np.linalg.norm(np.asarray(ref, dtype=np.float64).reshape(-1))), 1.0e-30)
    return float(np.linalg.norm(np.asarray(diff, dtype=np.float64).reshape(-1)) / den)


def normalize(vec: np.ndarray, *, eps: float = 1.0e-14) -> np.ndarray:
    vals = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vals))
    if not math.isfinite(norm) or norm <= float(eps):
        raise ValueError(f"degenerate vector norm {norm:g}")
    return vals / norm


def stack_director_frame_from_j(jmat: np.ndarray) -> np.ndarray:
    """Return Q columns [e1,e2,e3], with e3 aligned to CSS8 stack direction."""

    j = np.asarray(jmat, dtype=np.float64).reshape(3, 3)
    g_r, g_s, g_t = j[0], j[1], j[2]
    e3 = normalize(g_t)
    g_r_projected = g_r - float(np.dot(g_r, e3)) * e3
    if float(np.linalg.norm(g_r_projected)) <= 1.0e-14:
        g_r_projected = g_s - float(np.dot(g_s, e3)) * e3
    e1 = normalize(g_r_projected)
    e2 = normalize(np.cross(e3, e1))
    if float(np.dot(e2, g_s)) < 0.0:
        e1 = -e1
        e2 = normalize(np.cross(e3, e1))
    return np.stack([e1, e2, e3], axis=1)


def stack_frames_from_ip_j(ip_j: np.ndarray) -> np.ndarray:
    vals = np.asarray(ip_j, dtype=np.float64)
    if vals.ndim != 3 or vals.shape[1:] != (3, 3):
        raise ValueError(f"ip_J must be [P,3,3], got {vals.shape}")
    return np.asarray([stack_director_frame_from_j(jmat) for jmat in vals], dtype=np.float64)


def css8_standard_coordinates(nx: int = 4, ny: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row = standard_css8_row_map(nx=nx, ny=ny)
    gp = row[:, 3].astype(np.int64)
    ip_local_rst = np.asarray([GAUSS_POINTS[int(idx)] for idx in gp], dtype=np.float64)
    ip_macro_xi = row[:, 9:12].astype(np.float64)
    return row.astype(np.float64), ip_macro_xi, ip_local_rst


def characteristic_length(x_macro: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    x = np.asarray(x_macro, dtype=np.float64).reshape(-1, 3)
    center = np.mean(x, axis=0)
    span = np.max(x, axis=0) - np.min(x, axis=0)
    l_ref = float(np.max(span))
    if not math.isfinite(l_ref) or l_ref <= 1.0e-14:
        l_ref = float(np.sqrt(np.mean(np.sum((x - center) ** 2, axis=1))))
    if not math.isfinite(l_ref) or l_ref <= 1.0e-14:
        raise ValueError("cannot compute nondegenerate L_ref from X_macro")
    return l_ref, center.astype(np.float64), span.astype(np.float64)


def apply_t_q(q_raw: np.ndarray, t_q: np.ndarray) -> np.ndarray:
    q = np.asarray(q_raw, dtype=np.float64)
    t = np.asarray(t_q, dtype=np.float64)
    if t.shape == (42, 48):
        return q @ t.T
    if t.ndim == 3 and t.shape[0] == q.shape[0] and t.shape[1:] == (42, 48):
        return np.einsum("nki,ni->nk", t, q)
    raise ValueError(f"T_q_raw_to_useful must be [42,48] or [N,42,48], got {t.shape}")


def apply_t_eps_to_strain(t_eps: np.ndarray, strain: np.ndarray) -> np.ndarray:
    t = np.asarray(t_eps, dtype=np.float64)
    e = np.asarray(strain, dtype=np.float64)
    point_count = int(e.shape[1])
    if t.shape == (point_count, 6, 6):
        return np.einsum("pab,npb->npa", t, e)
    if t.shape == (e.shape[0], point_count, 6, 6):
        return np.einsum("npab,npb->npa", t, e)
    raise ValueError(f"T_eps shape must be [P,6,6] or [N,P,6,6], got {t.shape}")


def apply_t_eps_to_b(t_eps: np.ndarray, b: np.ndarray) -> np.ndarray:
    t = np.asarray(t_eps, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    point_count = int(bb.shape[1])
    if t.shape == (point_count, 6, 6):
        return np.einsum("pab,npbj->npaj", t, bb)
    if t.shape == (bb.shape[0], point_count, 6, 6):
        return np.einsum("npab,npbj->npaj", t, bb)
    raise ValueError(f"T_eps shape must be [P,6,6] or [N,P,6,6], got {t.shape}")


def right_multiply_t_q_transpose(b_raw_or_local: np.ndarray, t_q: np.ndarray) -> np.ndarray:
    b = np.asarray(b_raw_or_local, dtype=np.float64)
    t = np.asarray(t_q, dtype=np.float64)
    if t.shape == (42, 48):
        return np.einsum("npaj,kj->npak", b, t)
    if t.ndim == 3 and t.shape[0] == b.shape[0] and t.shape[1:] == (42, 48):
        return np.einsum("npaj,nkj->npak", b, t)
    raise ValueError(f"T_q_raw_to_useful must be [42,48] or [N,42,48], got {t.shape}")


def useful_to_raw_projected(b_local_useful: np.ndarray, t_eps_to_abq: np.ndarray, t_q_map: np.ndarray) -> np.ndarray:
    b_abq_useful = apply_t_eps_to_b(t_eps_to_abq, b_local_useful)
    t = np.asarray(t_q_map, dtype=np.float64)
    if t.shape == (42, 48):
        return np.einsum("npak,kj->npaj", b_abq_useful, t)
    if t.ndim == 3 and t.shape[0] == b_local_useful.shape[0] and t.shape[1:] == (42, 48):
        return np.einsum("npak,nkj->npaj", b_abq_useful, t)
    raise ValueError(f"T_q map must be [42,48] or [N,42,48], got {t.shape}")


def project_raw_b_to_useful_subspace(b_raw: np.ndarray, t_q: np.ndarray) -> np.ndarray:
    b = np.asarray(b_raw, dtype=np.float64)
    t = np.asarray(t_q, dtype=np.float64)
    if t.shape == (42, 48):
        p_useful = t.T @ t
        return np.einsum("npaj,jk->npak", b, p_useful)
    if t.ndim == 3 and t.shape[0] == b.shape[0] and t.shape[1:] == (42, 48):
        p_useful = np.einsum("nki,nkj->nij", t, t)
        return np.einsum("npaj,njk->npak", b, p_useful)
    raise ValueError(f"T_q_raw_to_useful must be [42,48] or [N,42,48], got {t.shape}")


def local_geometry_features(
    *,
    ip_macro_xi: np.ndarray,
    ip_local_rst: np.ndarray,
    ip_xyz_hat: np.ndarray,
    q_stack: np.ndarray,
    ip_j_hat: np.ndarray,
    ip_invj_hat: np.ndarray,
    ip_detj_hat: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    j = np.asarray(ip_j_hat, dtype=np.float64)
    metric = np.einsum("pij,pkj->pik", j, j)
    thickness_hat = 2.0 * np.linalg.norm(j[:, 2, :], axis=1)
    det_hat = np.asarray(ip_detj_hat, dtype=np.float64).reshape(-1)
    local_parts = [
        np.asarray(ip_xyz_hat, dtype=np.float64),
        np.asarray(q_stack, dtype=np.float64).reshape(j.shape[0], 9),
        j.reshape(j.shape[0], 9),
        np.asarray(ip_invj_hat, dtype=np.float64).reshape(j.shape[0], 9),
        metric.reshape(j.shape[0], 9),
        thickness_hat.reshape(j.shape[0], 1),
        det_hat.reshape(j.shape[0], 1),
        np.log(np.maximum(np.abs(det_hat), 1.0e-30)).reshape(j.shape[0], 1),
    ]
    local_names = (
        ["ip_xyz_hat_x", "ip_xyz_hat_y", "ip_xyz_hat_z"]
        + [f"Q_stack_{i}{k}" for i in range(3) for k in range(3)]
        + [f"ip_J_hat_{i}{k}" for i in range(3) for k in range(3)]
        + [f"ip_invJ_hat_{i}{k}" for i in range(3) for k in range(3)]
        + [f"metric_hat_{i}{k}" for i in range(3) for k in range(3)]
        + ["thickness_hat", "ip_detJ_hat", "log_abs_ip_detJ_hat"]
    )
    local = np.concatenate(local_parts, axis=1)

    trunk_parts = [np.asarray(ip_macro_xi, dtype=np.float64), np.asarray(ip_local_rst, dtype=np.float64), local]
    trunk_names = (
        ["ip_macro_xi", "ip_macro_eta", "ip_macro_zeta"]
        + ["ip_local_r", "ip_local_s", "ip_local_t"]
        + local_names
    )
    trunk = np.concatenate(trunk_parts, axis=1)
    return local, local_names, trunk, trunk_names


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
