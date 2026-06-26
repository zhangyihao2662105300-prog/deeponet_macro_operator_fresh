"""Prepare Macro16 coordinate-DeepONet arrays from source128 compacts.

The prepared trunk array is only ``x_gp_hat``:

    x_gp_hat = (x_gp_raw - X_center) / L_ref

When a compact does not store explicit physical GP coordinates, ``x_gp_raw`` is
rebuilt from the fixed Macro16 source128 parent points and ``X16_raw``.  Parent
coordinates are never included in the model input.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


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


GEOMETRY_PARAM_NAMES = [
    "length_scale_hat",
    "width_scale_hat",
    "thickness_scale_hat",
    "length_width_ratio",
    "thickness_length_ratio",
    "in_plane_angle_over_pi",
    "trapezoid_distortion",
    "curvature_r",
    "curvature_s",
    "twist_curvature",
    "thickness_mean_hat",
    "thickness_gradient_r",
    "thickness_gradient_s",
    "warping_rms_hat",
]

TRUE176_SURFACE_MACRO_TO_KEEP = np.asarray([0, 1, 2, 4, 7, 6, 5, 3], dtype=np.int64)
TRUE176_MACRO_TO_KEEP_NODE = np.concatenate([TRUE176_SURFACE_MACRO_TO_KEEP, TRUE176_SURFACE_MACRO_TO_KEEP + 8])
TRUE176_MACRO_TO_KEEP_FLAT = np.asarray(
    [int(node) * 3 + axis for node in TRUE176_MACRO_TO_KEEP_NODE for axis in range(3)],
    dtype=np.int64,
)


@dataclass(frozen=True)
class CoordinateArrays:
    compact_paths: list[str]
    q48_def_hat: np.ndarray
    geometry_g: np.ndarray
    x_gp_hat: np.ndarray
    le_macro: np.ndarray
    b_macro_qdef: np.ndarray
    q48_def_hat_plus: np.ndarray | None
    le_macro_plus: np.ndarray | None
    plus_macro_directions: np.ndarray | None
    plus_source_directions: np.ndarray | None
    plus_delta_raw: float | None
    case_id: np.ndarray
    source_index: np.ndarray
    source_row: np.ndarray
    x16_raw: np.ndarray
    x16_hat: np.ndarray
    x_center: np.ndarray
    l_ref: np.ndarray
    macro16_point_xi: np.ndarray
    gp_coordinate_source: str
    geometry_param_names: list[str]


def read_path_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def paths_from_args(args: argparse.Namespace) -> list[str]:
    paths = [str(v) for v in getattr(args, "compact", [])]
    if str(getattr(args, "compact_list", "")).strip():
        paths.extend(read_path_list(Path(args.compact_list)))
    unique: list[str] = []
    seen: set[str] = set()
    for item in paths:
        key = str(Path(item).resolve())
        if key not in seen:
            seen.add(key)
            unique.append(key)
    if not unique:
        raise ValueError("provide --compact or --compact-list")
    return unique


def _infer_n(z: np.lib.npyio.NpzFile, path: Path) -> int:
    for key in ("q48_def_hat", "LE_macro", "B_macro_qdef", "X16_raw", "X16"):
        if key in z.files:
            arr = np.asarray(z[key])
            if arr.ndim >= 1:
                return int(arr.shape[0])
    raise KeyError(f"{path}: cannot infer frame count")


def _scalar_text(z: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in z.files:
        return ""
    arr = np.asarray(z[key])
    if arr.size == 0:
        return ""
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _rows(n_total: int, *, frame_stride: int, max_frames: int) -> np.ndarray:
    rows = np.arange(0, int(n_total), max(1, int(frame_stride)), dtype=np.int64)
    if int(max_frames) > 0:
        rows = rows[: int(max_frames)]
    return rows


def _frame_array(
    z: np.lib.npyio.NpzFile,
    path: Path,
    n_total: int,
    rows: np.ndarray,
    key: str,
    tail: tuple[int, ...],
) -> np.ndarray:
    vals = np.asarray(z[key], dtype=np.float32)
    if vals.shape == (n_total,) + tuple(tail):
        return vals[rows]
    raise ValueError(f"{path}: {key} must have shape [{n_total},{tail}], got {vals.shape}")


def _as_x16_raw(z: np.lib.npyio.NpzFile, path: Path, n_total: int, rows: np.ndarray) -> np.ndarray:
    for key in ("X16_raw", "X16", "X16_ref", "X_keep", "X_keep_ref"):
        if key not in z.files:
            continue
        vals = np.asarray(z[key], dtype=np.float32)
        if vals.shape == (n_total, 16, 3):
            return vals[rows]
        if vals.shape == (16, 3):
            return np.broadcast_to(vals.reshape(1, 16, 3), (rows.size, 16, 3)).copy()
        raise ValueError(f"{path}: {key} must have shape [N,16,3] or [16,3], got {vals.shape}")
    raise KeyError(f"{path}: missing X16_raw/X16/X_keep; do not use X_macro")


def _compute_center_lref(x16_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = np.asarray(x16_raw, dtype=np.float64).reshape(-1, 16, 3)
    center = np.mean(nodes, axis=1)
    span = np.max(nodes, axis=1) - np.min(nodes, axis=1)
    l_ref = np.max(span, axis=1, keepdims=True)
    bad = ~np.isfinite(l_ref[:, 0]) | (l_ref[:, 0] <= 1.0e-14)
    if np.any(bad):
        rms = np.sqrt(np.mean(np.sum((nodes[bad] - center[bad, None, :]) ** 2, axis=2), axis=1))
        l_ref[bad, 0] = rms
    if np.any(~np.isfinite(l_ref)) or np.any(l_ref <= 1.0e-14):
        raise ValueError("cannot compute nondegenerate L_ref from X16")
    x16_hat = (nodes - center[:, None, :]) / l_ref[:, None, :]
    return center.astype(np.float32), l_ref.astype(np.float32), x16_hat.astype(np.float32)


def source128_point_xi() -> np.ndarray:
    gp, _gw = np.polynomial.legendre.leggauss(2)
    xi: list[list[float]] = []
    h = 2.0 / 4.0
    for ey in range(4):
        s_center = -1.0 + (float(ey) + 0.5) * h
        for ex in range(4):
            r_center = -1.0 + (float(ex) + 0.5) * h
            for ip in range(8):
                lr = ip % 2
                ls = (ip // 2) % 2
                lt = ip // 4
                r = r_center + 0.5 * h * float(gp[lr])
                s = s_center + 0.5 * h * float(gp[ls])
                t = float(gp[lt])
                xi.append([r, s, t])
    return np.asarray(xi, dtype=np.float64)


def _conventional_quad8_shape(rs: np.ndarray) -> np.ndarray:
    pts = np.asarray(rs, dtype=np.float64)
    r = pts[..., 0]
    s = pts[..., 1]
    n_conv = np.stack(
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
    order = np.asarray([0, 4, 1, 5, 2, 6, 3, 7], dtype=np.int64)
    return n_conv[..., order]


def shape_functions_macro16(xi: np.ndarray) -> np.ndarray:
    pts = np.asarray(xi, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"xi must have shape [P,3], got {pts.shape}")
    q8 = _conventional_quad8_shape(pts[:, :2])
    t = pts[:, 2]
    n_bot = 0.5 * (1.0 - t[:, None]) * q8
    n_top = 0.5 * (1.0 + t[:, None]) * q8
    return np.concatenate([n_bot, n_top], axis=1)


def map_x_gp_hat_from_x16_hat(x16_hat: np.ndarray, xi: np.ndarray) -> np.ndarray:
    n = shape_functions_macro16(xi)
    return np.einsum("pa,nai->npi", n, np.asarray(x16_hat, dtype=np.float64)).astype(np.float32)


def _normalize(v: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if not math.isfinite(norm) or norm <= eps:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(v, dtype=np.float64) / norm


def _safe_norm(v: np.ndarray) -> float:
    val = float(np.linalg.norm(v))
    return val if math.isfinite(val) else 0.0


def extract_geometry_params(x16_raw: np.ndarray, l_ref: np.ndarray) -> np.ndarray:
    nodes = np.asarray(x16_raw, dtype=np.float64).reshape(-1, 16, 3)
    l_refs = np.asarray(l_ref, dtype=np.float64).reshape(-1)
    out = np.zeros((nodes.shape[0], len(GEOMETRY_PARAM_NAMES)), dtype=np.float64)
    eps = 1.0e-12
    for i, cur in enumerate(nodes):
        ref = max(float(l_refs[i]), eps)
        bot = cur[:8]
        top = cur[8:]
        mid = 0.5 * (bot + top)
        thick_vec = top - bot
        thick_norm = np.linalg.norm(thick_vec, axis=1)
        center = np.mean(mid, axis=0)

        left = np.mean(mid[[6, 7, 0]], axis=0)
        right = np.mean(mid[[2, 3, 4]], axis=0)
        bottom_edge = np.mean(mid[[0, 1, 2]], axis=0)
        top_edge = np.mean(mid[[4, 5, 6]], axis=0)
        u = right - left
        v = top_edge - bottom_edge
        length = max(_safe_norm(u), eps)
        width = max(_safe_norm(v), eps)
        e1 = _normalize(u)
        v_perp = v - float(np.dot(v, e1)) * e1
        e2 = _normalize(v_perp)
        if _safe_norm(e2) <= eps:
            e2 = _normalize(v)
        e3 = _normalize(np.cross(e1, e2))
        if _safe_norm(e3) <= eps:
            e3 = _normalize(np.mean(thick_vec, axis=0))
        if _safe_norm(e2) <= eps or _safe_norm(e3) <= eps:
            e1 = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            e2 = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            e3 = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

        u_hat = _normalize(u)
        v_hat = _normalize(v)
        angle = math.acos(float(np.clip(np.dot(u_hat, v_hat), -1.0, 1.0))) / math.pi
        edge_bottom_len = _safe_norm(mid[2] - mid[0])
        edge_top_len = _safe_norm(mid[4] - mid[6])
        edge_left_len = _safe_norm(mid[6] - mid[0])
        edge_right_len = _safe_norm(mid[4] - mid[2])
        trapezoid = (abs(edge_top_len - edge_bottom_len) + abs(edge_right_len - edge_left_len)) / (length + width + eps)

        local = np.stack([(mid - center) @ e1 / ref, (mid - center) @ e2 / ref, (mid - center) @ e3 / ref], axis=1)
        xh = local[:, 0]
        yh = local[:, 1]
        zh = local[:, 2]
        design = np.stack([np.ones_like(xh), xh, yh, xh * xh, xh * yh, yh * yh], axis=1)
        coef, *_ = np.linalg.lstsq(design, zh, rcond=None)
        pred = design @ coef
        warping = float(np.sqrt(np.mean((zh - pred) ** 2)))

        th_hat = thick_norm / ref
        design_t = np.stack([np.ones_like(xh), xh, yh], axis=1)
        tcoef, *_ = np.linalg.lstsq(design_t, th_hat, rcond=None)

        thickness_mean = float(np.mean(thick_norm))
        thickness_scale = float(np.max(thick_norm) - np.min(thick_norm) + thickness_mean)
        out[i] = np.asarray(
            [
                length / ref,
                width / ref,
                thickness_scale / ref,
                length / (width + eps),
                thickness_mean / (length + eps),
                angle,
                trapezoid,
                2.0 * float(coef[3]),
                2.0 * float(coef[5]),
                float(coef[4]),
                thickness_mean / ref,
                float(tcoef[1]),
                float(tcoef[2]),
                warping,
            ],
            dtype=np.float64,
        )
    return out.astype(np.float32)


def _case_id_from_path(path: Path) -> int:
    import re

    match = re.search(r"case[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def _load_ip_xyz_hat(
    z: np.lib.npyio.NpzFile,
    n_total: int,
    rows: np.ndarray,
    *,
    point_count: int,
    x_center: np.ndarray,
    l_ref: np.ndarray,
) -> tuple[np.ndarray | None, str]:
    for key in ("ip_xyz", "ip_coords", "ip_coordinates", "integration_point_xyz", "gauss_xyz"):
        if key not in z.files:
            continue
        vals = np.asarray(z[key], dtype=np.float32)
        if vals.shape == (n_total, point_count, 3):
            raw = vals[rows]
            hat = (raw - x_center[:, None, :]) / np.maximum(l_ref[:, None, :], 1.0e-12)
            return hat.astype(np.float32), key
    return None, ""


def _load_plus_samples_from_source(
    z: np.lib.npyio.NpzFile,
    path: Path,
    n_total: int,
    rows: np.ndarray,
    *,
    q48_def_hat: np.ndarray,
    l_ref: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, float | None, str]:
    source_text = _scalar_text(z, "source_compact")
    if not source_text:
        return None, None, None, None, None, "missing_source_compact"
    source_path = Path(source_text)
    if not source_path.exists():
        return None, None, None, None, None, f"source_missing:{source_path}"
    if "rigid_projection_P" not in z.files:
        return None, None, None, None, None, "missing_rigid_projection_P"
    pmat = _frame_array(z, path, n_total, rows, "rigid_projection_P", (48, 48)).astype(np.float64)
    if "source_row" in z.files:
        source_rows_all = np.asarray(z["source_row"], dtype=np.int64)
        if source_rows_all.shape != (n_total,):
            return None, None, None, None, None, f"bad_source_row_shape:{source_rows_all.shape}"
        source_rows = source_rows_all[rows]
    else:
        source_rows = rows
    with np.load(str(source_path), allow_pickle=True) as src:
        required = {"LE128_plus", "perturb_directions", "delta"}
        if not required.issubset(src.files):
            return None, None, None, None, None, "source_has_no_LE128_plus"
        le_plus_all = np.asarray(src["LE128_plus"], dtype=np.float32)
        if le_plus_all.ndim != 4 or le_plus_all.shape[2:] != (128, 6):
            raise ValueError(f"{source_path}: LE128_plus must be [N,D,128,6], got {le_plus_all.shape}")
        directions = np.asarray(src["perturb_directions"], dtype=np.int64).reshape(-1)
        if directions.shape[0] != le_plus_all.shape[1]:
            raise ValueError(f"{source_path}: perturb_directions length does not match LE128_plus")
        if np.any(directions < 0) or np.any(directions >= 48):
            raise ValueError(f"{source_path}: perturb_directions must be in [0,47]")
        delta = float(np.asarray(src["delta"], dtype=np.float64).reshape(-1)[0])
        if not math.isfinite(delta) or delta == 0.0:
            raise ValueError(f"{source_path}: bad delta {delta:g}")
        source_order = _scalar_text(z, "source_node_order").strip().lower().replace("_", "-")
        node_transform = _scalar_text(z, "node_order_transform").strip().lower()
        if source_order == "true176-keep" or "true176_keep_to_macro16" in node_transform:
            macro_to_source = TRUE176_MACRO_TO_KEEP_FLAT
            le_plus = le_plus_all[source_rows][:, macro_to_source, :, :]
            source_directions = directions[macro_to_source]
            macro_directions = np.arange(48, dtype=np.int64)
        else:
            le_plus = le_plus_all[source_rows]
            source_directions = directions.copy()
            macro_directions = directions.copy()
        if macro_directions.shape != (48,) or np.any(np.sort(macro_directions) != np.arange(48)):
            return None, None, None, None, None, f"unsupported_plus_direction_set:{macro_directions.tolist()}"
    q_plus = np.repeat(q48_def_hat[:, None, :], macro_directions.shape[0], axis=1).astype(np.float64)
    l_ref_arr = np.maximum(np.asarray(l_ref, dtype=np.float64).reshape(-1, 1), 1.0e-12)
    for pos, macro_dir in enumerate(macro_directions.tolist()):
        q_plus[:, pos, :] += float(delta) * pmat[:, :, int(macro_dir)] / l_ref_arr
    return (
        q_plus.astype(np.float32),
        le_plus.astype(np.float32),
        macro_directions.astype(np.int64),
        source_directions.astype(np.int64),
        float(delta),
        f"LE128_plus_from:{source_path}",
    )


def load_coordinate_arrays(
    paths: Iterable[str],
    *,
    frame_stride: int = 1,
    max_frames_per_compact: int = 0,
) -> CoordinateArrays:
    q_chunks: list[np.ndarray] = []
    g_chunks: list[np.ndarray] = []
    xgp_chunks: list[np.ndarray] = []
    le_chunks: list[np.ndarray] = []
    b_chunks: list[np.ndarray] = []
    q_plus_chunks: list[np.ndarray] = []
    le_plus_chunks: list[np.ndarray] = []
    case_chunks: list[np.ndarray] = []
    source_chunks: list[np.ndarray] = []
    row_chunks: list[np.ndarray] = []
    x16_raw_chunks: list[np.ndarray] = []
    x16_hat_chunks: list[np.ndarray] = []
    center_chunks: list[np.ndarray] = []
    lref_chunks: list[np.ndarray] = []
    compact_paths: list[str] = []
    xi_ref: np.ndarray | None = None
    gp_sources: list[str] = []
    plus_macro_dirs_ref: np.ndarray | None = None
    plus_source_dirs_ref: np.ndarray | None = None
    plus_delta_ref: float | None = None
    plus_sources: list[str] = []

    for source_id, text in enumerate(paths):
        path = Path(text).resolve()
        with np.load(str(path), allow_pickle=True) as z:
            n_total = _infer_n(z, path)
            rows = _rows(n_total, frame_stride=frame_stride, max_frames=max_frames_per_compact)
            if "LE_macro" not in z.files:
                raise KeyError(f"{path}: missing LE_macro")
            point_count = int(np.asarray(z["LE_macro"]).shape[1])
            if point_count != 128:
                raise ValueError(f"{path}: this experiment requires source128 labels, got {point_count} points")
            xi = np.asarray(z["macro16_point_xi"], dtype=np.float64) if "macro16_point_xi" in z.files else source128_point_xi()
            if xi.shape != (128, 3):
                raise ValueError(f"{path}: macro16_point_xi must be [128,3], got {xi.shape}")
            if xi_ref is None:
                xi_ref = xi
            elif not np.allclose(xi_ref, xi, rtol=0.0, atol=1.0e-7):
                raise ValueError(f"{path}: 128 point rule differs from previous compact")

            x16_raw = _as_x16_raw(z, path, n_total, rows)
            computed_center, computed_l_ref, computed_x16_hat = _compute_center_lref(x16_raw)
            x_center = _frame_array(z, path, n_total, rows, "X_center", (3,)) if "X_center" in z.files else computed_center
            l_ref = _frame_array(z, path, n_total, rows, "L_ref", (1,)) if "L_ref" in z.files else computed_l_ref
            x16_hat = _frame_array(z, path, n_total, rows, "X16_hat", (16, 3)) if "X16_hat" in z.files else computed_x16_hat

            q = _frame_array(z, path, n_total, rows, "q48_def_hat", (48,))
            le = _frame_array(z, path, n_total, rows, "LE_macro", (128, 6))
            b = _frame_array(z, path, n_total, rows, "B_macro_qdef", (128, 6, 48))
            g = extract_geometry_params(x16_raw, l_ref)
            xgp_hat, gp_source = _load_ip_xyz_hat(
                z,
                n_total,
                rows,
                point_count=128,
                x_center=x_center,
                l_ref=l_ref,
            )
            if xgp_hat is None:
                xgp_hat = map_x_gp_hat_from_x16_hat(x16_hat, xi)
                gp_source = "macro16_isoparametric_from_X16"
            gp_sources.append(gp_source)
            q_plus, le_plus, plus_macro_dirs, plus_source_dirs, plus_delta, plus_source = _load_plus_samples_from_source(
                z,
                path,
                n_total,
                rows,
                q48_def_hat=q,
                l_ref=l_ref,
            )
            plus_sources.append(plus_source)
            if (
                q_plus is not None
                and le_plus is not None
                and plus_macro_dirs is not None
                and plus_source_dirs is not None
                and plus_delta is not None
            ):
                if plus_macro_dirs_ref is None:
                    plus_macro_dirs_ref = plus_macro_dirs
                    plus_source_dirs_ref = plus_source_dirs
                    plus_delta_ref = float(plus_delta)
                else:
                    if not np.array_equal(plus_macro_dirs_ref, plus_macro_dirs):
                        raise ValueError(f"{path}: plus perturb directions differ from previous compact")
                    if not np.array_equal(plus_source_dirs_ref, plus_source_dirs):
                        raise ValueError(f"{path}: plus source directions differ from previous compact")
                    if abs(float(plus_delta_ref) - float(plus_delta)) > 1.0e-15:
                        raise ValueError(f"{path}: plus delta differs from previous compact")
                q_plus_chunks.append(q_plus.astype(np.float32))
                le_plus_chunks.append(le_plus.astype(np.float32))

            if "case_id" in z.files:
                case_vals = np.asarray(z["case_id"], dtype=np.int64)
                if case_vals.shape == (n_total,):
                    case_id = case_vals[rows]
                elif case_vals.ndim == 0 or case_vals.shape == (1,):
                    case_id = np.full(rows.size, int(case_vals.reshape(-1)[0]), dtype=np.int64)
                else:
                    raise ValueError(f"{path}: unsupported case_id shape {case_vals.shape}")
            else:
                case_id = np.full(rows.size, _case_id_from_path(path), dtype=np.int64)

            compact_paths.append(str(path))
            q_chunks.append(q.astype(np.float32))
            g_chunks.append(g.astype(np.float32))
            xgp_chunks.append(xgp_hat.astype(np.float32))
            le_chunks.append(le.astype(np.float32))
            b_chunks.append(b.astype(np.float32))
            case_chunks.append(case_id.astype(np.int64))
            source_chunks.append(np.full(rows.size, int(source_id), dtype=np.int64))
            row_chunks.append(rows.astype(np.int64))
            x16_raw_chunks.append(x16_raw.astype(np.float32))
            x16_hat_chunks.append(x16_hat.astype(np.float32))
            center_chunks.append(x_center.astype(np.float32))
            lref_chunks.append(l_ref.astype(np.float32))

    if xi_ref is None or not q_chunks:
        raise ValueError("no frames loaded")
    unique_sources = sorted(set(gp_sources))
    gp_source_summary = unique_sources[0] if len(unique_sources) == 1 else "+".join(unique_sources)
    return CoordinateArrays(
        compact_paths=compact_paths,
        q48_def_hat=np.concatenate(q_chunks, axis=0),
        geometry_g=np.concatenate(g_chunks, axis=0),
        x_gp_hat=np.concatenate(xgp_chunks, axis=0),
        le_macro=np.concatenate(le_chunks, axis=0),
        b_macro_qdef=np.concatenate(b_chunks, axis=0),
        q48_def_hat_plus=np.concatenate(q_plus_chunks, axis=0) if q_plus_chunks else None,
        le_macro_plus=np.concatenate(le_plus_chunks, axis=0) if le_plus_chunks else None,
        plus_macro_directions=plus_macro_dirs_ref.astype(np.int64) if plus_macro_dirs_ref is not None else None,
        plus_source_directions=plus_source_dirs_ref.astype(np.int64) if plus_source_dirs_ref is not None else None,
        plus_delta_raw=float(plus_delta_ref) if plus_delta_ref is not None else None,
        case_id=np.concatenate(case_chunks, axis=0),
        source_index=np.concatenate(source_chunks, axis=0),
        source_row=np.concatenate(row_chunks, axis=0),
        x16_raw=np.concatenate(x16_raw_chunks, axis=0),
        x16_hat=np.concatenate(x16_hat_chunks, axis=0),
        x_center=np.concatenate(center_chunks, axis=0),
        l_ref=np.concatenate(lref_chunks, axis=0),
        macro16_point_xi=xi_ref.astype(np.float32),
        gp_coordinate_source=gp_source_summary,
        geometry_param_names=list(GEOMETRY_PARAM_NAMES),
    )


def save_prepared(path: Path, arrays: CoordinateArrays, *, meta: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "q48_def_hat": arrays.q48_def_hat,
        "geometry_g": arrays.geometry_g,
        "x_gp_hat": arrays.x_gp_hat,
        "LE_macro": arrays.le_macro,
        "B_macro_qdef": arrays.b_macro_qdef,
        "case_id": arrays.case_id,
        "source_index": arrays.source_index,
        "source_row": arrays.source_row,
        "X16_raw": arrays.x16_raw,
        "X16_hat": arrays.x16_hat,
        "X_center": arrays.x_center,
        "L_ref": arrays.l_ref,
        "macro16_point_xi": arrays.macro16_point_xi,
        "geometry_param_names": np.asarray(arrays.geometry_param_names, dtype=object),
        "compact_paths": np.asarray(arrays.compact_paths, dtype=object),
        "gp_coordinate_source": np.asarray(arrays.gp_coordinate_source, dtype=object),
        "meta_json": np.asarray(json.dumps(meta or {}, ensure_ascii=False), dtype=object),
    }
    if arrays.q48_def_hat_plus is not None and arrays.le_macro_plus is not None:
        payload.update(
            {
                "q48_def_hat_plus": arrays.q48_def_hat_plus,
                "LE_macro_plus": arrays.le_macro_plus,
                "plus_macro_directions": arrays.plus_macro_directions,
                "plus_source_directions": arrays.plus_source_directions,
                "plus_raw_directions": arrays.plus_macro_directions,
                "plus_delta_raw": np.asarray([float(arrays.plus_delta_raw)], dtype=np.float64),
            }
        )
    np.savez_compressed(path, **payload)


def write_summary(path: Path, arrays: CoordinateArrays, *, command: str) -> None:
    summary = {
        "command": command,
        "compact_count": len(arrays.compact_paths),
        "frame_count": int(arrays.q48_def_hat.shape[0]),
        "point_count": int(arrays.x_gp_hat.shape[1]),
        "branch_contract": "q48_def_hat[48] + geometry_g[14]",
        "trunk_contract": "x_gp_hat[3] only",
        "output_contract": "LE_macro[6]",
        "b_contract": "B from AD dLE/dq48_def_hat, target B_macro_qdef",
        "plus_contract": (
            "q48_def_hat_plus and LE_macro_plus use Macro16 q-column order; plus_source_directions records original source columns"
            if arrays.q48_def_hat_plus is not None
            else "not available"
        ),
        "gp_coordinate_source": arrays.gp_coordinate_source,
        "geometry_param_names": arrays.geometry_param_names,
        "shapes": {
            "q48_def_hat": list(arrays.q48_def_hat.shape),
            "geometry_g": list(arrays.geometry_g.shape),
            "x_gp_hat": list(arrays.x_gp_hat.shape),
            "LE_macro": list(arrays.le_macro.shape),
            "B_macro_qdef": list(arrays.b_macro_qdef.shape),
            "q48_def_hat_plus": list(arrays.q48_def_hat_plus.shape) if arrays.q48_def_hat_plus is not None else None,
            "LE_macro_plus": list(arrays.le_macro_plus.shape) if arrays.le_macro_plus is not None else None,
        },
        "case_ids": sorted({int(v) for v in arrays.case_id.reshape(-1)}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="append", default=[])
    parser.add_argument("--compact-list", default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=Path(""))
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-compact", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays = load_coordinate_arrays(
        paths_from_args(args),
        frame_stride=int(args.frame_stride),
        max_frames_per_compact=int(args.max_frames_per_compact),
    )
    save_prepared(args.out, arrays, meta={"script": str(Path(__file__).resolve())})
    summary_path = args.summary if str(args.summary) else args.out.with_suffix(".summary.json")
    write_summary(summary_path, arrays, command="prepare_coordinate_dataset.py")
    print(json.dumps({"out": str(args.out), "summary": str(summary_path), "frames": int(arrays.q48_def_hat.shape[0])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
