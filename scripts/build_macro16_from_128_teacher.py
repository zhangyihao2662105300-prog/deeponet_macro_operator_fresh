#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Macro16 v4 teacher compacts from audited 128-IP TRUE176/CSS8 labels.

The 128-IP compact is used only as a teacher field.  The written v4 compact
contains the Macro16 interface: X16, q48, 18 standard Macro16 labels and
integration weights.  It deliberately does not write X_macro or CSS8 row data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_v2_local_strain_pilot_compact import strain_transform_matrices  # noqa: E402
from macro_deeponet.macro16_geometry import (  # noqa: E402
    MACRO16_CONTRACT_VERSION,
    Macro16GeometryMap,
    Macro16PointTable,
    macro16_standard_point_table,
)
from macro_deeponet.true176_data import (  # noqa: E402
    build_keep_node_coords_unique,
    fine_axis_coords,
    standard_css8_row_map,
    standard_ip_keys,
)

TEACHER_CONTRACT_VERSION = "macro16-from-128-teacher-parent-linear-001"
TRUE176_SURFACE_MACRO_TO_KEEP = np.asarray([0, 1, 2, 4, 7, 6, 5, 3], dtype=np.int64)
TRUE176_MACRO_TO_KEEP_NODE = np.concatenate([TRUE176_SURFACE_MACRO_TO_KEEP, TRUE176_SURFACE_MACRO_TO_KEEP + 8])
TRUE176_KEEP_TO_MACRO_NODE = np.argsort(TRUE176_MACRO_TO_KEEP_NODE)
TRUE176_MACRO_TO_KEEP_FLAT = np.asarray(
    [int(node) * 3 + axis for node in TRUE176_MACRO_TO_KEEP_NODE for axis in range(3)],
    dtype=np.int64,
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def read_path_list(path: Path) -> list[Path]:
    return [
        Path(line.strip()).resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def collect_paths(compacts: Iterable[Path], compact_lists: Iterable[Path], *, case_limit: int | None = None) -> list[Path]:
    paths = [Path(p).resolve() for p in compacts]
    for list_path in compact_lists:
        paths.extend(read_path_list(Path(list_path)))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if case_limit is not None and int(case_limit) > 0:
        unique = unique[: int(case_limit)]
    return unique


def scalar_text(z: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    if arr.size == 0:
        return default
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def scalar_int(z: np.lib.npyio.NpzFile, key: str, default: int | None = None) -> int | None:
    if key not in z.files:
        return default
    try:
        arr = np.asarray(z[key])
        if arr.size == 0:
            return default
        return int(arr.reshape(-1)[0])
    except Exception:
        return default


def first_key(files: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    present = set(files)
    for key in candidates:
        if key in present:
            return key
    return None


def infer_n(z: np.lib.npyio.NpzFile, path: Path) -> int:
    for key in ("q48_raw", "q48_hat", "q_boundary", "LE128_base", "LE_abq", "B_LE128_forward", "LE_macro", "B_macro"):
        if key in z.files:
            arr = np.asarray(z[key])
            if arr.ndim >= 1:
                return int(arr.shape[0])
    raise KeyError(f"{path}: cannot infer frame count")


def frame_rows(n_total: int, *, frame_stride: int, max_frames: int) -> np.ndarray:
    rows = np.arange(0, int(n_total), max(1, int(frame_stride)), dtype=np.int64)
    if int(max_frames) > 0:
        rows = rows[: int(max_frames)]
    return rows


def parse_case_id(path: Path, z: np.lib.npyio.NpzFile) -> int:
    for key in ("v2b_pilot_case_id", "v2a_pilot_case_id", "case_id"):
        value = scalar_int(z, key)
        if value is not None and int(value) >= 0:
            return int(value)
    match = re.search(r"case[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def broadcast_or_rows(arr: np.ndarray, rows: np.ndarray, n_total: int, tail: tuple[int, ...], key: str, path: Path) -> np.ndarray:
    vals = np.asarray(arr)
    expected_frame = (int(n_total),) + tuple(tail)
    expected_fixed = tuple(tail)
    if vals.shape == expected_frame:
        return vals[rows]
    if vals.shape == expected_fixed:
        return np.broadcast_to(vals.reshape((1,) + expected_fixed), (rows.size,) + expected_fixed).copy()
    raise ValueError(f"{path}: {key} must have shape {expected_fixed} or {expected_frame}, got {vals.shape}")


def load_q48(z: np.lib.npyio.NpzFile, path: Path, n_total: int, rows: np.ndarray) -> tuple[np.ndarray, str]:
    key = first_key(z.files, ("q48_raw", "q48_hat", "q_boundary"))
    if key is None:
        raise KeyError(f"{path}: missing q48_raw q48_hat or q_boundary")
    vals = np.asarray(z[key], dtype=np.float64)
    if vals.shape == (n_total, 48):
        return vals[rows].astype(np.float64), key
    if vals.shape == (n_total, 16, 3):
        return vals[rows].reshape(rows.size, 48).astype(np.float64), key
    raise ValueError(f"{path}: {key} must be [{n_total},48] or [{n_total},16,3], got {vals.shape}")


def load_x16(z: np.lib.npyio.NpzFile, path: Path, n_total: int, rows: np.ndarray) -> tuple[np.ndarray, str]:
    for key in ("X16", "X16_ref", "X_keep", "X_keep_ref", "control_node_coords", "keep_node_coords"):
        if key in z.files:
            vals = broadcast_or_rows(np.asarray(z[key], dtype=np.float64), rows, n_total, (16, 3), key, path)
            return vals.astype(np.float64), key
    if "shape4" in z.files:
        shape = np.asarray(z["shape4"], dtype=np.float32)
        if shape.shape == (4,):
            shape_rows = np.broadcast_to(shape.reshape(1, 4), (rows.size, 4)).copy()
        elif shape.shape == (1, 4):
            shape_rows = np.broadcast_to(shape.reshape(1, 4), (rows.size, 4)).copy()
        elif shape.shape == (n_total, 4):
            shape_rows = shape[rows]
        else:
            raise ValueError(f"{path}: shape4 must be [4], [1,4], or [{n_total},4], got {shape.shape}")
        x_keep, _meta = build_keep_node_coords_unique(shape_rows)
        return np.asarray(x_keep, dtype=np.float64), "shape4_reconstructed_X_keep"
    raise KeyError(f"{path}: missing X16 or X_keep; Macro16 export must not use X_macro as visible geometry")


def resolve_source_node_order(requested: str, x16_source: str) -> str:
    key = str(requested).strip().lower().replace("_", "-")
    if key == "auto":
        if x16_source in {"X16", "X16_ref"}:
            return "macro16"
        return "true176-keep"
    if key in {"macro16", "macro-16"}:
        return "macro16"
    if key in {"true176-keep", "true176", "xkeep", "x-keep", "keep"}:
        return "true176-keep"
    raise ValueError("source_node_order must be auto, macro16, or true176-keep")


def reorder_nodes_q_b_to_macro16(
    *,
    x16: np.ndarray,
    q48: np.ndarray,
    b128: np.ndarray,
    source_node_order: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    order = str(source_node_order).strip().lower().replace("_", "-")
    if order == "macro16":
        return x16, q48, b128, {
            "source_node_order": "macro16",
            "node_order_transform": "identity",
            "q48_output_order": "macro16",
            "B_output_q_column_order": "macro16",
        }
    if order != "true176-keep":
        raise ValueError(f"unsupported source node order {source_node_order!r}")
    x_macro = np.asarray(x16, dtype=np.float64)[:, TRUE176_MACRO_TO_KEEP_NODE, :]
    q_macro = np.asarray(q48, dtype=np.float64)[:, TRUE176_MACRO_TO_KEEP_FLAT]
    b_macro_cols = np.asarray(b128, dtype=np.float64)[..., TRUE176_MACRO_TO_KEEP_FLAT]
    return x_macro, q_macro, b_macro_cols, {
        "source_node_order": "true176-keep",
        "node_order_transform": "true176_keep_to_macro16_surface8",
        "macro16_node_to_true176_keep_node_index": TRUE176_MACRO_TO_KEEP_NODE.astype(int).tolist(),
        "true176_keep_node_to_macro16_node_index": TRUE176_KEEP_TO_MACRO_NODE.astype(int).tolist(),
        "q48_output_order": "macro16",
        "B_output_q_column_order": "macro16",
    }


def load_case_ids(z: np.lib.npyio.NpzFile, path: Path, n_total: int, rows: np.ndarray) -> np.ndarray:
    if "case_id" in z.files:
        vals = np.asarray(z["case_id"], dtype=np.int64)
        if vals.ndim == 0 or vals.shape == (1,):
            return np.full(rows.size, int(vals.reshape(-1)[0]), dtype=np.int64)
        if vals.shape == (n_total,):
            return vals[rows].astype(np.int64)
    return np.full(rows.size, parse_case_id(path, z), dtype=np.int64)


def load_le_b_128(z: np.lib.npyio.NpzFile, path: Path, n_total: int, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, str]:
    le_key = first_key(z.files, ("LE128_base", "LE_abq", "le"))
    if le_key is None:
        raise KeyError(f"{path}: missing LE128_base or compatible 128-IP LE label")
    b_key = first_key(z.files, ("B_LE128_forward", "B_LE128_raw", "b"))
    if b_key is None:
        raise KeyError(f"{path}: missing B_LE128_forward or compatible 128-IP B label")
    le = np.asarray(z[le_key], dtype=np.float64)
    b = np.asarray(z[b_key], dtype=np.float64)
    if le.shape != (n_total, 128, 6):
        raise ValueError(f"{path}: {le_key} must have shape [{n_total},128,6], got {le.shape}")
    if b.shape != (n_total, 128, 6, 48):
        raise ValueError(f"{path}: {b_key} must have shape [{n_total},128,6,48], got {b.shape}")
    return le[rows].astype(np.float64), b[rows].astype(np.float64), le_key, b_key


def load_ip_xyz(z: np.lib.npyio.NpzFile, n_total: int, rows: np.ndarray) -> np.ndarray | None:
    key = first_key(z.files, ("ip_xyz", "ip_coords", "ip_coordinates", "integration_point_xyz", "gauss_xyz"))
    if key is None:
        return None
    vals = np.asarray(z[key], dtype=np.float64)
    if vals.shape == (128, 3):
        return np.broadcast_to(vals.reshape(1, 128, 3), (rows.size, 128, 3)).copy()
    if vals.shape == (n_total, 128, 3):
        return vals[rows].astype(np.float64)
    return None


def validate_ip_keys(z: np.lib.npyio.NpzFile, path: Path, n_total: int, rows: np.ndarray, *, allow_missing: bool) -> dict[str, Any]:
    std = standard_ip_keys().astype(np.int64)
    key = first_key(z.files, ("ip_keys", "integration_point_keys", "point_keys", "abaqus_ip_keys"))
    if key is None:
        if allow_missing:
            return {"ip_keys_source": "missing_allowed", "standard_ip_key_order_verified": False}
        raise KeyError(f"{path}: missing ip_keys. Provide --allow-missing-ip-keys only for audited standard 128-row data.")
    vals = np.asarray(z[key], dtype=np.int64)
    if vals.shape == std.shape:
        ok = bool(np.array_equal(vals[:, :3], std))
    elif vals.shape == (n_total, 128, std.shape[1]):
        ok = bool(np.all(vals[rows, :, :3] == std.reshape(1, 128, 3)))
    else:
        raise ValueError(f"{path}: {key} must have shape [128,3] or [{n_total},128,3], got {vals.shape}")
    if not ok:
        raise ValueError(f"{path}: {key} is not the standard CSS8 row order elem1 IP1..IP8 through elem16 IP8")
    return {"ip_keys_source": key, "standard_ip_key_order_verified": True}


def interp1d_weights(axis: np.ndarray, x: float) -> tuple[int, int, float, float]:
    vals = np.asarray(axis, dtype=np.float64).reshape(-1)
    xx = float(x)
    if xx <= float(vals[0]):
        return 0, 0, 1.0, 0.0
    if xx >= float(vals[-1]):
        return vals.size - 1, vals.size - 1, 1.0, 0.0
    hi = int(np.searchsorted(vals, xx, side="right"))
    lo = hi - 1
    den = float(vals[hi] - vals[lo])
    if den <= 0.0:
        raise ValueError("interpolation axis must be strictly increasing")
    wh = (xx - float(vals[lo])) / den
    wl = 1.0 - wh
    return lo, hi, float(wl), float(wh)


def macro16_from_128_interpolation_matrix(point_table: Macro16PointTable) -> np.ndarray:
    """Return W[P,128] from standard 128 CSS8 parent rows to Macro16 points."""

    row = standard_css8_row_map()
    fine_r = fine_axis_coords(4)
    fine_s = fine_axis_coords(4)
    fine_t = np.unique(row[:, 11])
    lookup: dict[tuple[int, int, int], int] = {}
    for source_row in row:
        lookup[(int(source_row[6]), int(source_row[7]), int(source_row[8]))] = int(source_row[0])
    w = np.zeros((point_table.xi.shape[0], 128), dtype=np.float64)
    for p, (r, s, t) in enumerate(np.asarray(point_table.xi, dtype=np.float64)):
        r0, r1, wr0, wr1 = interp1d_weights(fine_r, float(r))
        s0, s1, ws0, ws1 = interp1d_weights(fine_s, float(s))
        t0, t1, wt0, wt1 = interp1d_weights(fine_t, float(t))
        for ix, wr in ((r0, wr0), (r1, wr1)):
            for iy, ws in ((s0, ws0), (s1, ws1)):
                for iz, wt in ((t0, wt0), (t1, wt1)):
                    if wr == 0.0 or ws == 0.0 or wt == 0.0:
                        continue
                    w[p, lookup[(ix, iy, iz)]] += float(wr * ws * wt)
    if not np.allclose(np.sum(w, axis=1), 1.0, atol=1.0e-12):
        raise RuntimeError("bad Macro16-from-128 interpolation weights")
    return w


def macro16_geometry_fields(x16: np.ndarray, point_table: Macro16PointTable) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    nodes = np.asarray(x16, dtype=np.float64).reshape(-1, 16, 3)
    q_frames = np.empty((nodes.shape[0], point_table.xi.shape[0], 3, 3), dtype=np.float64)
    weights = np.empty((nodes.shape[0], point_table.xi.shape[0]), dtype=np.float64)
    xyz = np.empty((nodes.shape[0], point_table.xi.shape[0], 3), dtype=np.float64)
    det_min = math.inf
    det_max = -math.inf
    for i, frame_nodes in enumerate(nodes):
        geom = Macro16GeometryMap(frame_nodes)
        fields = geom.eval_points(point_table)
        q_frames[i] = np.asarray(fields["Q"], dtype=np.float64)
        weights[i] = np.asarray(fields["integration_weight_hat"], dtype=np.float64)
        xyz[i] = np.asarray(fields["x_hat"], dtype=np.float64) * float(geom.l_ref) + geom.center.reshape(1, 3)
        det = np.asarray(fields["detJ_hat"], dtype=np.float64)
        det_min = min(det_min, float(np.min(det)))
        det_max = max(det_max, float(np.max(det)))
    return q_frames, weights, xyz, {"macro16_detJ_hat_min": det_min, "macro16_detJ_hat_max": det_max}


def rel_norm(diff: np.ndarray, ref: np.ndarray) -> float:
    den = max(float(np.linalg.norm(np.asarray(ref, dtype=np.float64).reshape(-1))), 1.0e-30)
    return float(np.linalg.norm(np.asarray(diff, dtype=np.float64).reshape(-1)) / den)


def max_abs(arr: np.ndarray) -> float:
    vals = np.asarray(arr, dtype=np.float64)
    return float(np.max(np.abs(vals))) if vals.size else 0.0


def build_one(
    path: Path,
    out_root: Path,
    *,
    source_index: int,
    point_table: Macro16PointTable,
    interpolation_matrix: np.ndarray,
    frame_stride: int,
    max_frames_per_compact: int,
    allow_missing_ip_keys: bool,
    source_geometry_tol: float,
    source_node_order: str,
) -> dict[str, Any]:
    path = Path(path).resolve()
    with np.load(str(path), allow_pickle=True) as z:
        n_total = infer_n(z, path)
        rows = frame_rows(n_total, frame_stride=frame_stride, max_frames=max_frames_per_compact)
        if rows.size == 0:
            raise ValueError(f"{path}: no frames selected")
        ip_key_meta = validate_ip_keys(z, path, n_total, rows, allow_missing=allow_missing_ip_keys)
        q48, q_key = load_q48(z, path, n_total, rows)
        x16, x16_source = load_x16(z, path, n_total, rows)
        case_id = load_case_ids(z, path, n_total, rows)
        le128, b128, le_key, b_key = load_le_b_128(z, path, n_total, rows)
        source_xyz128 = load_ip_xyz(z, n_total, rows)
        strain_field = scalar_text(z, "strain_field", scalar_text(z, "strain_label_key", "LE"))
        b_strain_field = scalar_text(z, "B_label_strain_field", strain_field)
    resolved_node_order = resolve_source_node_order(source_node_order, x16_source)
    x16, q48, b128, node_order_meta = reorder_nodes_q_b_to_macro16(
        x16=x16,
        q48=q48,
        b128=b128,
        source_node_order=resolved_node_order,
    )

    w = np.asarray(interpolation_matrix, dtype=np.float64)
    le_global = np.einsum("pr,nra->npa", w, le128)
    b_global = np.einsum("pr,nraj->npaj", w, b128)
    q_frames, weights, target_xyz, geom_meta = macro16_geometry_fields(x16, point_table)
    t_from, _t_to = strain_transform_matrices(q_frames)
    le_macro = np.einsum("npab,npb->npa", t_from, le_global)
    b_macro = np.einsum("npab,npbj->npaj", t_from, b_global)

    geometry_audit: dict[str, Any] = {"source_ip_xyz_available": source_xyz128 is not None}
    if source_xyz128 is not None:
        source_xyz_macro = np.einsum("pr,nri->npi", w, source_xyz128)
        diff = target_xyz - source_xyz_macro
        geometry_audit.update(
            {
                "source_parent_interp_xyz_vs_macro16_xyz_rel": rel_norm(diff, target_xyz),
                "source_parent_interp_xyz_vs_macro16_xyz_max_abs": max_abs(diff),
            }
        )
        if float(source_geometry_tol) > 0.0 and float(geometry_audit["source_parent_interp_xyz_vs_macro16_xyz_rel"]) > float(source_geometry_tol):
            raise ValueError(
                f"{path}: source parent-interpolated xyz differs from Macro16 xyz with rel="
                f"{geometry_audit['source_parent_interp_xyz_vs_macro16_xyz_rel']:.6g}, tol={float(source_geometry_tol):.6g}"
            )

    if not np.all(np.isfinite(le_macro)) or not np.all(np.isfinite(b_macro)):
        raise ValueError(f"{path}: non-finite Macro16 teacher labels")

    first_case = int(case_id[0]) if case_id.size else -1
    case_name = f"case{first_case:03d}" if first_case >= 0 else "case_unknown"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{int(source_index):04d}_{case_name}_macro16_from_128_teacher.npz"
    np.savez_compressed(
        out_path,
        standard_operator_contract_version=np.asarray(MACRO16_CONTRACT_VERSION, dtype=object),
        macro16_teacher_contract_version=np.asarray(TEACHER_CONTRACT_VERSION, dtype=object),
        q48_raw=q48.astype(np.float32),
        X16=x16.astype(np.float32),
        LE_macro=le_macro.astype(np.float32),
        B_macro=b_macro.astype(np.float32),
        integration_weight_hat=weights.astype(np.float32),
        case_id=case_id.astype(np.int64),
        source_compact=np.asarray(str(path), dtype=object),
        source_index=np.full(rows.size, int(source_index), dtype=np.int64),
        source_row=rows.astype(np.int64),
        source_q_key=np.asarray(q_key, dtype=object),
        source_x16_key=np.asarray(x16_source, dtype=object),
        source_node_order=np.asarray(node_order_meta["source_node_order"], dtype=object),
        node_order_transform=np.asarray(node_order_meta["node_order_transform"], dtype=object),
        source_le128_key=np.asarray(le_key, dtype=object),
        source_b128_key=np.asarray(b_key, dtype=object),
        macro16_point_xi=point_table.xi.astype(np.float32),
        macro16_parent_interpolation_from_128=np.asarray("standard CSS8 8x8x2 parent-coordinate linear interpolation", dtype=object),
        strain_output_coordinate=np.asarray("macro16_local_frame", dtype=object),
        strain_field=np.asarray(strain_field, dtype=object),
        B_label_strain_field=np.asarray(b_strain_field, dtype=object),
        B_label_q_coordinate=np.asarray("q48_raw", dtype=object),
        fine_grid_geometry_visible_to_model=np.asarray(False),
        prepared_from=np.asarray("scripts/build_macro16_from_128_teacher.py", dtype=object),
    )

    summary = {
        "source_compact": str(path),
        "macro16_compact": str(out_path),
        "standard_operator_contract_version": MACRO16_CONTRACT_VERSION,
        "macro16_teacher_contract_version": TEACHER_CONTRACT_VERSION,
        "frame_count": int(rows.size),
        "source_rows_first_last": [int(rows[0]), int(rows[-1])],
        "point_count": int(point_table.xi.shape[0]),
        "q_key": q_key,
        "x16_source": x16_source,
        **node_order_meta,
        "source_le128_key": le_key,
        "source_b128_key": b_key,
        "strain_output_coordinate": "macro16_local_frame",
        "label_source": "128-IP TRUE176/CSS8 teacher labels interpolated in macro parent coordinates",
        "model_visible_arrays": ["q48_raw", "X16", "macro16_point_features_generated_by_v4_loader"],
        "fine_grid_geometry_visible_to_model": False,
        "writes_X_macro": False,
        "writes_css8_labels": False,
        "case_ids": sorted(np.unique(case_id).astype(np.int64).tolist()),
        "q48_shape": list(q48.shape),
        "X16_shape": list(x16.shape),
        "LE_macro_shape": list(le_macro.shape),
        "B_macro_shape": list(b_macro.shape),
        **ip_key_meta,
        **geom_meta,
        **geometry_audit,
    }
    summary_path = out_path.with_suffix(".summary.json")
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def build_all(args: argparse.Namespace) -> dict[str, Any]:
    paths = collect_paths(args.compact, args.compact_list, case_limit=args.case_limit)
    if not paths:
        raise ValueError("no compact inputs provided")
    out_root = Path(args.out_root).resolve()
    point_table = macro16_standard_point_table(
        plane_order=int(args.plane_gauss_order),
        thickness_order=int(args.thickness_gauss_order),
    )
    w = macro16_from_128_interpolation_matrix(point_table)
    rows = [
        build_one(
            path,
            out_root,
            source_index=i,
            point_table=point_table,
            interpolation_matrix=w,
            frame_stride=int(args.frame_stride),
            max_frames_per_compact=int(args.max_frames_per_compact),
            allow_missing_ip_keys=bool(args.allow_missing_ip_keys),
            source_geometry_tol=float(args.source_geometry_tol),
            source_node_order=str(args.source_node_order),
        )
        for i, path in enumerate(paths)
    ]
    list_path = out_root / "macro16_from_128_teacher_compact_list.txt"
    list_path.write_text("\n".join(str(row["macro16_compact"]) for row in rows) + "\n", encoding="utf-8")
    summary = {
        "script": "build_macro16_from_128_teacher",
        "standard_operator_contract_version": MACRO16_CONTRACT_VERSION,
        "macro16_teacher_contract_version": TEACHER_CONTRACT_VERSION,
        "source_compact_count": int(len(paths)),
        "macro16_compact_count": int(len(rows)),
        "total_frame_count": int(sum(int(row["frame_count"]) for row in rows)),
        "point_count": int(point_table.xi.shape[0]),
        "compact_list": str(list_path),
        "label_source": "128-IP TRUE176/CSS8 teacher labels interpolated to standard Macro16 integration points",
        "fine_grid_geometry_visible_to_model": False,
        "writes_X_macro": False,
        "cases": rows,
    }
    summary_path = out_root / "macro16_from_128_teacher_summary.json"
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "compact_list": str(list_path),
                "macro16_compact_count": int(len(rows)),
                "total_frame_count": int(summary["total_frame_count"]),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=json_default,
        )
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="append", type=Path, default=[])
    parser.add_argument("--compact-list", action="append", type=Path, default=[])
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-compact", type=int, default=0)
    parser.add_argument("--plane-gauss-order", type=int, default=3)
    parser.add_argument("--thickness-gauss-order", type=int, default=2)
    parser.add_argument("--allow-missing-ip-keys", action="store_true")
    parser.add_argument(
        "--source-node-order",
        default="auto",
        choices=("auto", "macro16", "true176-keep"),
        help="Order of X16/X_keep/q48/B columns in the source compact. auto treats X16 as Macro16 and X_keep/shape4 as TRUE176 keep order.",
    )
    parser.add_argument(
        "--source-geometry-tol",
        type=float,
        default=0.0,
        help="Optional relative tolerance for source parent-interpolated xyz versus Macro16 xyz. 0 disables the check.",
    )
    return parser.parse_args()


def main() -> None:
    build_all(parse_args())


if __name__ == "__main__":
    main()
