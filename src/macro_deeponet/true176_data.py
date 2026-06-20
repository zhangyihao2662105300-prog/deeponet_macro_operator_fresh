"""TRUE176/CSS8 compact-data helpers for the 128-IP DeepONet route.

This module mirrors the current NNSE training contract without depending on the
large NNSE repository at runtime:

    legacy branch = [shape4(4), q48_raw(48)]
    current generic branch = [q48_raw(48), X_keep(16,3)]
    LE     = LE128_base          -> [frames, 128, 6]
    B      = B_LE128_forward     -> [frames, 128, 6, 48]

Point features are deterministic functions of ``shape4`` and the fixed 4x4 CSS8
row order.  They are used as the DeepONet trunk input.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

NX = 4
NY = 4
GAUSS_1D = np.asarray([-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)], dtype=np.float64)
GAUSS_POINTS = tuple((r, s, t) for t in GAUSS_1D for s in GAUSS_1D for r in GAUSS_1D)
NODE_SIGNS = np.asarray(
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
LENGTH_SCALE_KEYS = (
    "H",
    "length_scale",
    "length_scale_H",
    "scale_H",
    "element_H",
    "element_length",
    "macro_length_scale",
)


@dataclass(frozen=True)
class True176Arrays:
    compact_paths: list[str]
    shape4: np.ndarray
    q48_raw: np.ndarray
    le: np.ndarray
    b: np.ndarray
    length_scale: np.ndarray
    length_scale_source: str
    keep_node_coords: np.ndarray | None
    macro_nodes: np.ndarray | None
    sample_index: np.ndarray
    frame_number: np.ndarray
    case_id: np.ndarray
    source_index: np.ndarray
    source_row: np.ndarray
    shape_index: np.ndarray


def stats(arr: np.ndarray, axis: tuple[int, ...] | int, floor: float = 1.0e-8) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(arr, dtype=np.float32)
    mean = vals.mean(axis=axis, keepdims=True).astype(np.float32)
    std = vals.std(axis=axis, keepdims=True).astype(np.float32)
    return mean, np.maximum(std, np.asarray(float(floor), dtype=np.float32)).astype(np.float32)


def canonical_scale_mode(mode: str) -> str:
    key = str(mode).strip().lower().replace("_", "-")
    if key in {"normalized", "dimensionless", "hat"}:
        return "normalized"
    if key in {"physical", "dimensional"}:
        return "physical"
    raise ValueError("scale_mode must be normalized or physical")


def length_scale_column(length_scale: np.ndarray | None, n: int) -> np.ndarray:
    if length_scale is None:
        out = np.ones((int(n), 1), dtype=np.float32)
    else:
        vals = np.asarray(length_scale, dtype=np.float32)
        if vals.ndim == 0:
            out = np.full((int(n), 1), float(vals), dtype=np.float32)
        elif vals.shape == (1,):
            out = np.full((int(n), 1), float(vals.reshape(-1)[0]), dtype=np.float32)
        elif vals.shape == (int(n),):
            out = vals.reshape(int(n), 1).astype(np.float32)
        elif vals.shape == (int(n), 1):
            out = vals.astype(np.float32)
        else:
            raise ValueError(f"length_scale must be scalar, [N], or [N,1]; got {vals.shape} for N={n}")
    if np.any(~np.isfinite(out)) or np.any(out <= 0.0):
        raise ValueError("length_scale/H must be finite and positive")
    return out


def transform_b_target_for_q_coordinate(
    b_raw: np.ndarray,
    length_scale: np.ndarray,
    *,
    scale_mode: str,
    b_label_coordinate: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return B target in the same coordinate used by the branch q slice.

    In physical scale mode the branch uses q_hat=q_phys/H, so a physical label
    dLE/dq_phys must be multiplied by H to become dLE/dq_hat.
    """

    mode = canonical_scale_mode(scale_mode)
    b = np.asarray(b_raw, dtype=np.float32)
    h = length_scale_column(length_scale, b.shape[0]).reshape(b.shape[0], 1, 1, 1)
    coord = str(b_label_coordinate).strip().lower().replace("_", "-")
    if coord == "auto":
        coord = "physical" if mode == "physical" else "dimensionless"
    if coord in {"normalized", "dimensionless", "hat", "q-hat"}:
        target = b
        transform = "identity"
        target_coordinate = "dLE/dq_hat" if mode == "physical" else "dLE/dq48_raw_normalized"
    elif coord in {"physical", "dimensional", "q-phys"}:
        if mode == "physical":
            target = b * h
            transform = "H * B_phys"
            target_coordinate = "dLE/dq_hat"
        else:
            target = b
            transform = "identity_physical_equals_normalized_H1"
            target_coordinate = "dLE/dq48_raw_normalized"
    else:
        raise ValueError("b_label_coordinate must be auto, physical, or dimensionless")
    return target.astype(np.float32), {
        "scale_mode": mode,
        "input_b_label_coordinate": coord,
        "training_b_coordinate": target_coordinate,
        "b_target_transform": transform,
        "H_min": float(np.min(h)),
        "H_max": float(np.max(h)),
    }


def parse_int_list(text: str, *, default: list[int] | None = None) -> list[int]:
    raw = str(text).strip()
    if not raw:
        return [] if default is None else list(default)
    if raw.lower() == "all":
        return list(range(48))
    return [int(x) for x in raw.replace(";", ",").split(",") if x.strip()]


def parse_target_ips(text: str) -> list[int]:
    ips = [int(x) for x in str(text).replace(";", ",").split(",") if x.strip()]
    if not ips:
        raise ValueError("target_ips must not be empty")
    if len(set(ips)) != len(ips):
        raise ValueError(f"duplicate target ips: {ips}")
    for ip in ips:
        if ip < 0 or ip >= 128:
            raise ValueError(f"target ip out of range [0,127]: {ip}")
    return ips


def case_id_from_path(path: str) -> int:
    match = re.search(r"case(\d+)", str(path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def sample_meta(n: int, sample_paths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = np.asarray(sample_paths).reshape(-1)
    sample_count = int(paths.size)
    if sample_count > 0 and n % sample_count == 0:
        frames = n // sample_count
        sample_index = np.repeat(np.arange(sample_count, dtype=np.int64), frames)
        frame_number = np.tile(np.arange(1, frames + 1, dtype=np.int64), sample_count)
        case_by_sample = np.asarray([case_id_from_path(str(p)) for p in paths], dtype=np.int64)
        case_id = case_by_sample[sample_index]
    else:
        sample_index = np.zeros(n, dtype=np.int64)
        frame_number = np.arange(1, n + 1, dtype=np.int64)
        case_id = np.full(n, -1, dtype=np.int64)
    return sample_index, frame_number, case_id


def load_one_compact(path: str | Path, *, frame_stride: int = 1, max_frames: int = 0) -> dict[str, Any]:
    compact = Path(path).resolve()
    if not compact.exists():
        raise FileNotFoundError(compact)

    def require_shape(z: np.lib.npyio.NpzFile, key: str, tail: tuple[int, ...], n_total: int) -> np.ndarray:
        arr = np.asarray(z[key], dtype=np.float32)
        expected = (int(n_total),) + tuple(tail)
        if tuple(arr.shape) != expected:
            raise ValueError(
                f"{compact}: expected {key} shape {expected}, got {tuple(arr.shape)}. "
                "This TRUE176 DeepONet trainer requires full q48/B48 compact data."
            )
        return arr

    with np.load(str(compact), allow_pickle=True) as z:
        n_total = int(z["shape4"].shape[0])
        stride = max(1, int(frame_stride))
        idx = np.arange(0, n_total, stride, dtype=np.int64)
        if int(max_frames) > 0:
            idx = idx[: int(max_frames)]
        sample_paths = np.asarray(z["sample_paths"]).astype(str) if "sample_paths" in z.files else np.asarray([], dtype=str)
        sample_index, frame_number, case_id = sample_meta(n_total, sample_paths)
        le_key = "LE128_base" if "LE128_base" in z.files else "le"
        b_key = "B_LE128_forward" if "B_LE128_forward" in z.files else "b"
        length_scale = np.ones((idx.size, 1), dtype=np.float32)
        length_scale_source = "implicit_H_1"
        for h_key in LENGTH_SCALE_KEYS:
            if h_key in z.files:
                length_scale = length_scale_column(np.asarray(z[h_key], dtype=np.float32), n_total)[idx]
                length_scale_source = h_key
                break
        keep_node_coords = None
        for keep_key in (
            "X_keep",
            "X_keep_ref",
            "x_keep",
            "xkeep",
            "keep_node_coords",
            "coords48",
            "coords48_for_321",
            "control_node_coords",
            "control_nodes16",
        ):
            if keep_key in z.files:
                raw_keep = np.asarray(z[keep_key], dtype=np.float32)
                if raw_keep.shape == (16, 3):
                    keep_node_coords = np.broadcast_to(raw_keep.reshape(1, 16, 3), (n_total, 16, 3)).copy()[idx]
                elif raw_keep.shape == (n_total, 16, 3):
                    keep_node_coords = raw_keep[idx]
                else:
                    raise ValueError(f"{compact}: expected {keep_key} shape [16,3] or [{n_total},16,3], got {raw_keep.shape}")
                break
        macro_nodes = None
        for node_key in ("X_macro", "macro_node_coords", "x_macro", "macro_nodes", "nodes50"):
            if node_key in z.files:
                raw_nodes = np.asarray(z[node_key], dtype=np.float32)
                if raw_nodes.shape == (50, 3):
                    macro_nodes = np.broadcast_to(raw_nodes.reshape(1, 50, 3), (n_total, 50, 3)).copy()[idx]
                elif raw_nodes.shape == (n_total, 50, 3):
                    macro_nodes = raw_nodes[idx]
                else:
                    raise ValueError(f"{compact}: expected {node_key} shape [50,3] or [{n_total},50,3], got {raw_nodes.shape}")
                break
        return {
            "compact_path": str(compact),
            "shape4": require_shape(z, "shape4", (4,), n_total)[idx],
            "q48_raw": require_shape(z, "q48_raw", (48,), n_total)[idx],
            "le": require_shape(z, le_key, (128, 6), n_total)[idx],
            "b": require_shape(z, b_key, (128, 6, 48), n_total)[idx],
            "length_scale": length_scale,
            "length_scale_source": length_scale_source,
            "keep_node_coords": keep_node_coords,
            "macro_nodes": macro_nodes,
            "sample_index": sample_index[idx],
            "frame_number": frame_number[idx],
            "case_id": case_id[idx],
            "source_index": np.full(idx.size, -1, dtype=np.int64),
            "source_row": idx.astype(np.int64),
        }


def load_compacts(paths: Iterable[str], *, frame_stride: int = 1, max_frames_per_compact: int = 0) -> True176Arrays:
    chunks: list[dict[str, Any]] = []
    for source_id, text in enumerate(paths):
        chunk = load_one_compact(text, frame_stride=frame_stride, max_frames=max_frames_per_compact)
        chunk["source_index"][:] = int(source_id)
        chunks.append(chunk)
    if not chunks:
        raise ValueError("at least one compact file is required")
    shape4 = np.concatenate([c["shape4"] for c in chunks], axis=0)
    if all(c["macro_nodes"] is not None for c in chunks):
        macro_nodes = np.concatenate([np.asarray(c["macro_nodes"], dtype=np.float32) for c in chunks], axis=0)
    elif all(c["macro_nodes"] is None for c in chunks):
        macro_nodes = None
    else:
        raise ValueError("Either every compact must provide X_macro/macro_node_coords/nodes50, or none of them should.")
    if all(c["keep_node_coords"] is not None for c in chunks):
        keep_node_coords = np.concatenate([np.asarray(c["keep_node_coords"], dtype=np.float32) for c in chunks], axis=0)
    elif all(c["keep_node_coords"] is None for c in chunks):
        keep_node_coords = None
    else:
        raise ValueError("Either every compact must provide X_keep/coords48/keep_node_coords, or none of them should.")
    length_scale_sources = sorted({str(c["length_scale_source"]) for c in chunks})
    _, shape_index = np.unique(shape4, axis=0, return_inverse=True)
    return True176Arrays(
        compact_paths=[str(c["compact_path"]) for c in chunks],
        shape4=shape4,
        q48_raw=np.concatenate([c["q48_raw"] for c in chunks], axis=0),
        le=np.concatenate([c["le"] for c in chunks], axis=0),
        b=np.concatenate([c["b"] for c in chunks], axis=0),
        length_scale=length_scale_column(np.concatenate([c["length_scale"] for c in chunks], axis=0), shape4.shape[0]),
        length_scale_source="+".join(length_scale_sources),
        keep_node_coords=keep_node_coords,
        macro_nodes=macro_nodes,
        sample_index=np.concatenate([c["sample_index"] for c in chunks], axis=0),
        frame_number=np.concatenate([c["frame_number"] for c in chunks], axis=0),
        case_id=np.concatenate([c["case_id"] for c in chunks], axis=0),
        source_index=np.concatenate([c["source_index"] for c in chunks], axis=0),
        source_row=np.concatenate([c["source_row"] for c in chunks], axis=0),
        shape_index=shape_index.astype(np.int64),
    )


def fine_axis_coords(n_sub: int = 4) -> np.ndarray:
    out: list[float] = []
    h = 2.0 / float(n_sub)
    for i in range(n_sub):
        center = -1.0 + (float(i) + 0.5) * h
        for g in GAUSS_1D:
            out.append(center + 0.5 * h * float(g))
    return np.asarray(out, dtype=np.float64)


def standard_ip_keys(nx: int = 4, ny: int = 4) -> np.ndarray:
    rows: list[list[int]] = []
    for elem in range(1, int(nx) * int(ny) + 1):
        for ip in range(1, 9):
            rows.append([elem, ip, 0])
    return np.asarray(rows, dtype=np.int64)


def standard_css8_row_map(nx: int = 4, ny: int = 4) -> np.ndarray:
    fine_x = fine_axis_coords(nx)
    fine_y = fine_axis_coords(ny)
    rows: list[list[float]] = []
    data_row = 0
    for elem in range(1, int(nx) * int(ny) + 1):
        ex = (elem - 1) % int(nx)
        ey = (elem - 1) // int(nx)
        for ip in range(1, 9):
            gp = ip - 1
            lr = gp % 2
            ls = (gp // 2) % 2
            iz = gp // 4
            ix = ex * 2 + lr
            iy = ey * 2 + ls
            rows.append([data_row, elem, ip, gp, ex, ey, ix, iy, iz, fine_x[ix], fine_y[iy], GAUSS_1D[iz]])
            data_row += 1
    out = np.asarray(rows, dtype=np.float64)
    if out.shape != (128, 12):
        raise RuntimeError(f"bad CSS8 row map shape {out.shape}")
    return out


def node_id(i: int, j: int, k: int) -> int:
    return 1 + int(k) * (NX + 1) * (NY + 1) + int(j) * (NX + 1) + int(i)


def css8_elements() -> dict[int, list[int]]:
    elements: dict[int, list[int]] = {}
    eid = 1
    for j in range(NY):
        for i in range(NX):
            elements[eid] = [
                node_id(i, j, 0),
                node_id(i + 1, j, 0),
                node_id(i + 1, j + 1, 0),
                node_id(i, j + 1, 0),
                node_id(i, j, 1),
                node_id(i + 1, j, 1),
                node_id(i + 1, j + 1, 1),
                node_id(i, j + 1, 1),
            ]
            eid += 1
    return elements


def css8_shape(r: float, s: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    n = np.empty(8, dtype=np.float64)
    dndr = np.empty((8, 3), dtype=np.float64)
    for a, (ra, sa, ta) in enumerate(NODE_SIGNS):
        n[a] = 0.125 * (1.0 + ra * r) * (1.0 + sa * s) * (1.0 + ta * t)
        dndr[a, 0] = 0.125 * ra * (1.0 + sa * s) * (1.0 + ta * t)
        dndr[a, 1] = 0.125 * sa * (1.0 + ra * r) * (1.0 + ta * t)
        dndr[a, 2] = 0.125 * ta * (1.0 + ra * r) * (1.0 + sa * s)
    return n, dndr


def _derive_shape4(shape4: np.ndarray) -> dict[str, float | str | None]:
    lam, tau, chi, mu = [float(v) for v in np.asarray(shape4, dtype=np.float64).reshape(4)]
    if abs(chi) <= 1.0e-12:
        return {"shape_type": "flat", "lambda": lam, "tau": tau, "chi": 0.0, "mu": mu, "root": 1.0, "R_mid": None, "theta": 0.0}
    root = math.sqrt(1.0 + mu * mu)
    r_mid = 1.0 / (chi * root)
    theta = lam * chi * root
    return {"shape_type": "cylinder" if abs(mu) <= 1.0e-12 else "cone", "lambda": lam, "tau": tau, "chi": chi, "mu": mu, "root": root, "R_mid": r_mid, "theta": theta}


def frame_at_params(shape4: np.ndarray, u: np.ndarray | float, v: np.ndarray | float) -> np.ndarray:
    d = _derive_shape4(shape4)
    uu, vv = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
    if d["shape_type"] == "flat":
        out = np.zeros(uu.shape + (3, 3), dtype=np.float64)
        out[..., 0, 0] = 1.0
        out[..., 1, 1] = 1.0
        out[..., 2, 2] = 1.0
        return out
    theta = float(d["theta"])
    mu = float(d["mu"])
    root = float(d["root"])
    phi = -vv * theta
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    e_width = np.stack([sin_p, np.zeros_like(phi), -cos_p], axis=-1)
    e_axial = np.stack([mu * cos_p / root, np.ones_like(phi) / root, mu * sin_p / root], axis=-1)
    e_normal = np.stack([cos_p / root, -mu * np.ones_like(phi) / root, sin_p / root], axis=-1)
    return np.stack([e_width, e_axial, e_normal], axis=-2)


def point_at_params(shape4: np.ndarray, u: np.ndarray | float, v: np.ndarray | float, w: np.ndarray | float) -> np.ndarray:
    d = _derive_shape4(shape4)
    uu, vv, ww = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64), np.asarray(w, dtype=np.float64))
    lam = float(d["lambda"])
    tau = float(d["tau"])
    if d["shape_type"] == "flat":
        return np.stack([lam * vv, uu, tau * ww], axis=-1)
    theta = float(d["theta"])
    mu = float(d["mu"])
    r_mid = float(d["R_mid"])
    phi = -vv * theta
    radius = r_mid + mu * uu
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    normal = frame_at_params(shape4, uu, vv)[..., 2, :]
    mid = np.stack([radius * cos_p, uu, radius * sin_p], axis=-1)
    return mid + (ww * tau)[..., None] * normal


def build_shape4_nodes(shape4: np.ndarray) -> np.ndarray:
    nodes = np.empty(((NX + 1) * (NY + 1) * 2, 3), dtype=np.float64)
    for k in range(2):
        w = -0.5 + float(k)
        for j in range(NY + 1):
            u = -0.5 + float(j) / float(NY)
            for i in range(NX + 1):
                v = -0.5 + float(i) / float(NX)
                nodes[node_id(i, j, k) - 1] = point_at_params(shape4, u, v, w)
    return nodes


def build_macro_node_coords_unique(shape4: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Return X_macro node coordinates for each frame as [N,50,3]."""

    shapes = np.asarray(shape4, dtype=np.float32).reshape(-1, 4)
    if shapes.shape[0] == 0:
        return np.empty((0, (NX + 1) * (NY + 1) * 2, 3), dtype=np.float32), {
            "macro_node_count": int((NX + 1) * (NY + 1) * 2),
            "unique_shape4_count": 0,
            "macro_geometry_source": "empty",
        }
    unique_shape4, inverse = np.unique(shapes, axis=0, return_inverse=True)
    nodes_unique = np.asarray([build_shape4_nodes(shape) for shape in unique_shape4], dtype=np.float32)
    nodes = nodes_unique[np.asarray(inverse, dtype=np.int64)]
    return nodes.astype(np.float32), {
        "macro_node_count": int(nodes.shape[1]),
        "unique_shape4_count": int(unique_shape4.shape[0]),
        "macro_geometry_source": "shape4_reconstructed_nodes",
        "node_order": "CSS8 node_id order over 5x5x2 grid",
    }


def keep_node_ids() -> np.ndarray:
    keep: list[int] = []
    for k in range(2):
        for j in range(NY + 1):
            for i in range(NX + 1):
                if (int(i), int(j)) in {
                    (0, 0),
                    (NX // 2, 0),
                    (NX, 0),
                    (NX, NY // 2),
                    (NX, NY),
                    (NX // 2, NY),
                    (0, NY),
                    (0, NY // 2),
                }:
                    keep.append(node_id(i, j, k))
    return np.asarray(keep, dtype=np.int64)


def build_keep_node_coords_unique(shape4: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Return coordinates of the 16 q48 control nodes as [N,16,3]."""

    macro_nodes, meta = build_macro_node_coords_unique(shape4)
    ids = keep_node_ids()
    keep = macro_nodes[:, ids - 1, :]
    source = str(meta.get("macro_geometry_source", "shape4_reconstructed_nodes"))
    return keep.astype(np.float32), {
        "unique_shape4_count": int(meta.get("unique_shape4_count", 0)),
        "macro_geometry_source": source.replace("nodes", "keep_nodes") if "nodes" in source else source,
        "reconstruction_macro_node_count": int(meta.get("macro_node_count", macro_nodes.shape[1])),
        "keep_node_count": int(keep.shape[1]),
        "keep_node_ids": ids.astype(np.int64),
        "geometry_input": "X_keep[16,3]",
        "node_order": "keep_nodes/q48 order",
    }


def build_branch_features(
    *,
    shape4: np.ndarray,
    q48_raw: np.ndarray,
    mode: str,
    keep_node_coords: np.ndarray | None = None,
    macro_nodes: np.ndarray | None = None,
    length_scale: np.ndarray | None = None,
    scale_mode: str = "normalized",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build branch features and record where the q48 slice lives."""

    key = str(mode).strip().lower().replace("_", "-")
    scale_key = canonical_scale_mode(scale_mode)
    shape = np.asarray(shape4, dtype=np.float32).reshape(-1, 4)
    q_raw = np.asarray(q48_raw, dtype=np.float32).reshape(-1, 48)
    if shape.shape[0] != q_raw.shape[0]:
        raise ValueError(f"shape4 frame count {shape.shape[0]} does not match q48 frame count {q_raw.shape[0]}")
    h_col = length_scale_column(length_scale, q_raw.shape[0])
    h3 = h_col.reshape(-1, 1, 1)
    q = q_raw / h_col if scale_key == "physical" else q_raw
    scale_meta = {
        "scale_mode": scale_key,
        "length_scale_mode": "explicit_H_physical_to_dimensionless" if scale_key == "physical" else "implicit_or_already_dimensionless_H1",
        "H_min": float(np.min(h_col)),
        "H_max": float(np.max(h_col)),
        "q_coordinate": "q_hat=q48_raw/H" if scale_key == "physical" else "q48_raw_in_normalized_length",
        "geometry_units": "dimensionless_hat" if scale_key == "physical" else "already_dimensionless_or_dataset_units",
    }
    if key == "shape4-qraw":
        x = np.concatenate([shape, q], axis=1).astype(np.float32)
        return x, {
            **scale_meta,
            "branch_mode": "shape4-qraw",
            "branch_dim": int(x.shape[1]),
            "q_start": 4,
            "q_dim": 48,
            "geometry_input": "shape4[4]",
        }
    if key == "xkeep-qraw":
        if keep_node_coords is not None:
            keep = np.asarray(keep_node_coords, dtype=np.float32).reshape(-1, 16, 3)
            if keep.shape[0] != q_raw.shape[0]:
                raise ValueError(f"X_keep frame count {keep.shape[0]} does not match q48 frame count {q_raw.shape[0]}")
            if scale_key == "physical":
                keep = keep / h3
            meta = {
                "keep_node_count": 16,
                "keep_node_ids": keep_node_ids().astype(np.int64),
                "macro_geometry_source": "compact_X_keep",
                "geometry_input": "X_keep[16,3]",
                "node_order": "keep_nodes/q48 order",
            }
        elif macro_nodes is not None:
            nodes = np.asarray(macro_nodes, dtype=np.float32).reshape(-1, 50, 3)
            if nodes.shape[0] != q_raw.shape[0]:
                raise ValueError(f"X_macro frame count {nodes.shape[0]} does not match q48 frame count {q_raw.shape[0]}")
            if scale_key == "physical":
                nodes = nodes / h3
            ids = keep_node_ids()
            keep = nodes[:, ids - 1, :]
            meta = {
                "keep_node_count": 16,
                "keep_node_ids": ids.astype(np.int64),
                "reconstruction_macro_node_count": 50,
                "macro_geometry_source": "compact_X_macro_subset_keep_nodes",
                "geometry_input": "X_keep[16,3]",
                "node_order": "keep_nodes/q48 order",
            }
        else:
            keep, meta = build_keep_node_coords_unique(shape)
            if scale_key == "physical":
                keep = keep / h3
        x_keep_flat = keep.reshape(keep.shape[0], -1).astype(np.float32)
        x = np.concatenate([q, x_keep_flat], axis=1).astype(np.float32)
        return x, {
            **meta,
            **scale_meta,
            "branch_mode": "xkeep-qraw",
            "branch_dim": int(x.shape[1]),
            "q_start": 0,
            "q_dim": 48,
            "geometry_input": "X_keep[16,3]",
            "contract": "Branch=[q48_raw, X_keep_nodes]; Trunk=[xi, x(xi), J, invJ, detJ, ...]",
            "visibility": "macro-element branch sees only q48 control-node coordinates",
        }
    if key == "xnodes-qraw":
        if macro_nodes is None:
            nodes, meta = build_macro_node_coords_unique(shape)
        else:
            nodes = np.asarray(macro_nodes, dtype=np.float32).reshape(-1, 50, 3)
            if nodes.shape[0] != q_raw.shape[0]:
                raise ValueError(f"X_macro frame count {nodes.shape[0]} does not match q48 frame count {q_raw.shape[0]}")
            meta = {
                "macro_node_count": 50,
                "unique_shape4_count": int(np.unique(shape, axis=0).shape[0]) if shape.shape[0] else 0,
                "macro_geometry_source": "compact_X_macro",
                "node_order": "provided by compact; must match the element connectivity contract",
            }
        if scale_key == "physical":
            nodes = nodes / h3
        x_nodes_flat = nodes.reshape(nodes.shape[0], -1).astype(np.float32)
        x = np.concatenate([q, x_nodes_flat], axis=1).astype(np.float32)
        return x, {
            **meta,
            **scale_meta,
            "branch_mode": "xnodes-qraw",
            "branch_dim": int(x.shape[1]),
            "q_start": 0,
            "q_dim": 48,
            "geometry_input": "X_macro[50,3]",
            "contract": "Branch=[q48_raw, X_macro_nodes]; Trunk=[xi, x(xi), J, invJ, detJ, ...]",
        }
    raise ValueError("branch feature mode must be one of: shape4-qraw, xkeep-qraw, xnodes-qraw")


def build_shape4_ip_geometry(shape4: np.ndarray) -> dict[str, np.ndarray]:
    """Reconstruct the CSS8 reference integration-point geometry from shape4."""

    shapes = np.asarray(shape4, dtype=np.float64).reshape(-1, 4)
    row = standard_css8_row_map()
    elements = css8_elements()
    n_shape = int(shapes.shape[0])
    xyz = np.empty((n_shape, 128, 3), dtype=np.float64)
    jmat = np.empty((n_shape, 128, 3, 3), dtype=np.float64)
    invj = np.empty((n_shape, 128, 3, 3), dtype=np.float64)
    detj = np.empty((n_shape, 128), dtype=np.float64)
    for si, shape in enumerate(shapes):
        nodes = build_shape4_nodes(shape)
        for ri, rr in enumerate(row):
            elem = int(rr[1])
            gp = int(rr[3])
            nshape, dndr = css8_shape(*GAUSS_POINTS[gp])
            x = nodes[np.asarray(elements[elem], dtype=np.int64) - 1]
            xyz[si, ri] = nshape.dot(x)
            jj = dndr.T.dot(x)
            jmat[si, ri] = jj
            detj[si, ri] = float(np.linalg.det(jj))
            invj[si, ri] = np.linalg.inv(jj)
    gp = row[:, 3].astype(np.int64).reshape(128)
    local = np.asarray([GAUSS_POINTS[int(v)] for v in gp], dtype=np.float64).reshape(128, 3)
    frame = np.asarray([frame_at_params(shape, 0.5 * row[:, 10], 0.5 * row[:, 9]) for shape in shapes], dtype=np.float64)
    return {
        "row_map": row.astype(np.float64),
        "ip_keys": standard_ip_keys().astype(np.int64),
        "ip_xi": row[:, 9:12].astype(np.float64),
        "ip_local": local,
        "ip_xyz": xyz,
        "ip_frame": frame,
        "ip_J": jmat,
        "ip_invJ": invj,
        "ip_detJ": detj,
    }


def build_point_features(shape4: np.ndarray, *, include_id_features: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    shapes = np.asarray(shape4, dtype=np.float64).reshape(-1, 4)
    geom = build_shape4_ip_geometry(shapes)
    row = geom["row_map"]
    xyz = geom["ip_xyz"]
    jmat = geom["ip_J"]
    invj = geom["ip_invJ"]
    detj = geom["ip_detJ"]
    frame = geom["ip_frame"]
    n_shape = int(shapes.shape[0])
    xi = row[:, 9].reshape(1, 128)
    eta = row[:, 10].reshape(1, 128)
    zeta = row[:, 11].reshape(1, 128)
    local = geom["ip_local"]
    local_r = local[:, 0].reshape(1, 128)
    local_s = local[:, 1].reshape(1, 128)
    local_t = local[:, 2].reshape(1, 128)
    point = np.concatenate(
        [
            np.broadcast_to(xi, (n_shape, 128)).reshape(n_shape, 128, 1),
            np.broadcast_to(eta, (n_shape, 128)).reshape(n_shape, 128, 1),
            np.broadcast_to(zeta, (n_shape, 128)).reshape(n_shape, 128, 1),
            np.broadcast_to(xi * xi, (n_shape, 128)).reshape(n_shape, 128, 1),
            np.broadcast_to(eta * eta, (n_shape, 128)).reshape(n_shape, 128, 1),
            np.broadcast_to(zeta * zeta, (n_shape, 128)).reshape(n_shape, 128, 1),
            np.broadcast_to(np.sin(math.pi * xi), (n_shape, 128)).reshape(n_shape, 128, 1),
            np.broadcast_to(np.cos(math.pi * xi), (n_shape, 128)).reshape(n_shape, 128, 1),
            np.broadcast_to(np.sin(math.pi * eta), (n_shape, 128)).reshape(n_shape, 128, 1),
            np.broadcast_to(np.cos(math.pi * eta), (n_shape, 128)).reshape(n_shape, 128, 1),
            np.broadcast_to(local_r, (n_shape, 128)).reshape(n_shape, 128, 1),
            np.broadcast_to(local_s, (n_shape, 128)).reshape(n_shape, 128, 1),
            np.broadcast_to(local_t, (n_shape, 128)).reshape(n_shape, 128, 1),
            xyz,
            (0.5 * np.broadcast_to(eta, (n_shape, 128))).reshape(n_shape, 128, 1),
            (0.5 * np.broadcast_to(xi, (n_shape, 128))).reshape(n_shape, 128, 1),
            (0.5 * np.broadcast_to(zeta, (n_shape, 128))).reshape(n_shape, 128, 1),
            frame.reshape(n_shape, 128, 9),
            jmat.reshape(n_shape, 128, 9),
            invj.reshape(n_shape, 128, 9),
            detj.reshape(n_shape, 128, 1),
            np.log(np.maximum(np.abs(detj), 1.0e-30)).reshape(n_shape, 128, 1),
        ],
        axis=2,
    )
    names = [
        "xi_fine", "eta_fine", "zeta_fine", "xi_fine_sq", "eta_fine_sq", "zeta_fine_sq",
        "sin_pi_xi", "cos_pi_xi", "sin_pi_eta", "cos_pi_eta", "local_r", "local_s", "local_t",
        "X_hat_exact", "Y_hat_exact", "Z_hat_exact", "u_axial_hat", "v_width_hat", "w_thickness_hat",
    ]
    names += [f"frame_hat_{a}{b}" for a in range(3) for b in range(3)]
    names += [f"J_hat_{a}{b}" for a in range(3) for b in range(3)]
    names += [f"invJ_hat_{a}{b}" for a in range(3) for b in range(3)]
    names += ["detJ_hat", "log_abs_detJ_hat"]
    if include_id_features:
        elem = row[:, 1].reshape(1, 128)
        ip = row[:, 2].reshape(1, 128)
        ex = row[:, 4].reshape(1, 128)
        ey = row[:, 5].reshape(1, 128)
        elem_x = -1.0 + 2.0 * (ex + 0.5) / float(NX)
        elem_y = -1.0 + 2.0 * (ey + 0.5) / float(NY)
        id_feats = np.stack(
            [
                np.broadcast_to((elem - 8.5) / 7.5, (n_shape, 128)),
                np.broadcast_to((ip - 4.5) / 3.5, (n_shape, 128)),
                np.broadcast_to(elem_x, (n_shape, 128)),
                np.broadcast_to(elem_y, (n_shape, 128)),
            ],
            axis=2,
        )
        point = np.concatenate([point, id_feats], axis=2)
        names += ["elem_index_norm", "ip_index_norm", "elem_x_center_id", "elem_y_center_id"]
    meta = {
        "feature_names": names,
        "feature_dim": int(point.shape[2]),
        "point_count": 128,
        "include_id_features": bool(include_id_features),
        "row_map": row.astype(np.float32),
        "detJ_min": float(np.min(detj)),
        "detJ_max": float(np.max(detj)),
        "detJ_abs_min": float(np.min(np.abs(detj))),
    }
    return point.astype(np.float32), meta


def build_point_features_unique(shape4: np.ndarray, *, include_id_features: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """Build shape4 point features once per unique geometry and expand back."""

    shapes = np.asarray(shape4, dtype=np.float32).reshape(-1, 4)
    if shapes.shape[0] == 0:
        dummy = np.asarray([[1.0, 0.01, 0.0, 0.0]], dtype=np.float32)
        point, meta = build_point_features(dummy, include_id_features=include_id_features)
        return point[:0].astype(np.float32), {**meta, "unique_shape4_count": 0, "computed_unique_shape4": True}
    unique_shape4, inverse = np.unique(shapes, axis=0, return_inverse=True)
    point_unique, meta = build_point_features(unique_shape4, include_id_features=include_id_features)
    point = point_unique[np.asarray(inverse, dtype=np.int64)]
    return point.astype(np.float32), {
        **meta,
        "unique_shape4_count": int(unique_shape4.shape[0]),
        "computed_unique_shape4": True,
    }


class SobolevArrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, x_norm: np.ndarray, point_norm: np.ndarray, le_norm: np.ndarray, j_norm: np.ndarray, indices: np.ndarray) -> None:
        self.x_norm = np.asarray(x_norm, dtype=np.float32)
        self.point_norm = np.asarray(point_norm, dtype=np.float32)
        self.le_norm = np.asarray(le_norm, dtype=np.float32)
        self.j_norm = np.asarray(j_norm, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        return (
            torch.from_numpy(self.x_norm[idx]),
            torch.from_numpy(self.point_norm[idx]),
            torch.from_numpy(self.le_norm[idx]),
            torch.from_numpy(self.j_norm[idx]),
        )


def split_indices(data: True176Arrays, val_fraction: float, seed: int, val_cases: str = "") -> tuple[np.ndarray, np.ndarray]:
    n = int(data.shape4.shape[0])
    all_idx = np.arange(n, dtype=np.int64)
    text = str(val_cases).strip()
    if text:
        wanted = {int(x) for x in re.split(r"[,;\s]+", text) if x.strip()}
        val = all_idx[np.isin(data.case_id, np.asarray(sorted(wanted), dtype=np.int64))]
        train = all_idx[~np.isin(data.case_id, np.asarray(sorted(wanted), dtype=np.int64))]
        if val.size and train.size:
            return train, val
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_val = int(round(max(0.0, min(float(val_fraction), 0.9)) * n))
    if n_val <= 0:
        return perm, perm[: min(n, max(1, n // 5))]
    return perm[n_val:], perm[:n_val]
