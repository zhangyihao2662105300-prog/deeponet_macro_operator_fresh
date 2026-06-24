#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the v3 CSS8 standard-operator compact contract."""

from __future__ import annotations

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
