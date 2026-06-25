#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit Macro16 teacher compacts before interpreting training results.

The audit checks whether the exported v4 Macro16 labels are geometrically
aligned and whether B_macro is consistent with finite-difference LE responses
from the source 128-IP teacher compact when those responses are available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
from build_macro16_from_128_teacher import (  # noqa: E402
    TEACHER_CONTRACT_VERSION,
    TRUE176_MACRO_TO_KEEP_FLAT,
    collect_paths,
    infer_n,
    load_ip_xyz,
    macro16_from_128_interpolation_matrix,
    macro16_geometry_fields,
    rel_norm,
    scalar_text,
)
from macro_deeponet.macro16_geometry import (  # noqa: E402
    MACRO16_CONTRACT_VERSION,
    Macro16PointTable,
    macro16_standard_point_table,
)

LENGTH_SCALE_KEYS = ("H", "length_scale", "length_scale_H", "macro_length_scale")
SOURCE_COORD_KEYS = (
    "strain_field",
    "strain_label_key",
    "strain_output_coordinate",
    "B_label_strain_field",
    "B_label_output_coordinate",
    "B_label_q_coordinate",
    "q_useful_coordinate",
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


def compact_paths_from_args(args: argparse.Namespace) -> list[Path]:
    return collect_paths(args.compact, args.compact_list, case_limit=args.case_limit)


def scalar_from_npz(z: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    if arr.size == 0:
        return default
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def maybe_array(z: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    return np.asarray(z[key]) if key in z.files else None


def finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    val = float(value)
    return val if math.isfinite(val) else None


def max_abs(arr: np.ndarray) -> float:
    vals = np.asarray(arr, dtype=np.float64)
    return float(np.max(np.abs(vals))) if vals.size else 0.0


def rms(arr: np.ndarray) -> float:
    vals = np.asarray(arr, dtype=np.float64)
    return float(np.sqrt(np.mean(vals * vals))) if vals.size else 0.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    den = max(float(np.linalg.norm(aa) * np.linalg.norm(bb)), 1.0e-30)
    return float(np.dot(aa, bb) / den)


def rel_metric(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    return {
        "rel": rel_norm(diff, ref),
        "cos": cosine(pred, ref),
        "diff_rms": rms(diff),
        "ref_rms": rms(ref),
        "pred_rms": rms(pred),
        "diff_max_abs": max_abs(diff),
    }


def source_dirs_to_macro_dirs(source_dirs: np.ndarray, source_node_order: str) -> np.ndarray:
    dirs = np.asarray(source_dirs, dtype=np.int64).reshape(-1)
    if dirs.size == 0:
        return dirs
    if np.min(dirs) < 0 or np.max(dirs) >= 48:
        raise ValueError(f"perturb directions must be in [0,47], got {dirs.tolist()}")
    order = str(source_node_order).strip().lower().replace("_", "-")
    if order == "true176-keep":
        inv = np.empty(48, dtype=np.int64)
        inv[np.asarray(TRUE176_MACRO_TO_KEEP_FLAT, dtype=np.int64)] = np.arange(48, dtype=np.int64)
        return inv[dirs]
    return dirs


def load_length_scale(z: np.lib.npyio.NpzFile, rows: np.ndarray, n_total: int) -> tuple[np.ndarray, str]:
    for key in LENGTH_SCALE_KEYS:
        if key not in z.files:
            continue
        vals = np.asarray(z[key], dtype=np.float64)
        if vals.ndim == 0 or vals.shape == (1,):
            return np.full(rows.size, float(vals.reshape(-1)[0]), dtype=np.float64), key
        if vals.shape == (n_total,):
            return vals[rows].astype(np.float64), key
        if vals.shape == (n_total, 1):
            return vals[rows, 0].astype(np.float64), key
    return np.ones(rows.size, dtype=np.float64), "implicit_H1"


def source_metadata(z: np.lib.npyio.NpzFile) -> dict[str, Any]:
    return {key: scalar_text(z, key, "") for key in SOURCE_COORD_KEYS if key in z.files}


def coordinate_warnings(*, macro: dict[str, Any], source_meta: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    le_key = str(macro.get("source_le128_key", ""))
    b_key = str(macro.get("source_b128_key", ""))
    strain_coord = str(source_meta.get("strain_output_coordinate", "")).strip().lower()
    b_coord = str(source_meta.get("B_label_output_coordinate", "")).strip().lower()
    if le_key in {"LE128_base", "LE_abq"} and strain_coord and "local" in strain_coord:
        warnings.append(
            f"source {le_key} was used but source strain_output_coordinate={source_meta.get('strain_output_coordinate')}."
        )
    if b_key in {"B_LE128_forward", "B_LE128_raw"} and b_coord and "local" in b_coord:
        warnings.append(
            f"source {b_key} was used but source B_label_output_coordinate={source_meta.get('B_label_output_coordinate')}."
        )
    return warnings


def load_macro_compact(path: Path) -> dict[str, Any]:
    with np.load(str(path), allow_pickle=True) as z:
        required = {"q48_raw", "X16", "LE_macro", "B_macro"}
        missing = sorted(required.difference(z.files))
        if missing:
            raise KeyError(f"{path}: missing Macro16 fields {missing}")
        q = np.asarray(z["q48_raw"], dtype=np.float64)
        x16 = np.asarray(z["X16"], dtype=np.float64)
        le = np.asarray(z["LE_macro"], dtype=np.float64)
        b = np.asarray(z["B_macro"], dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 48:
            raise ValueError(f"{path}: q48_raw must be [N,48], got {q.shape}")
        if x16.shape == (16, 3):
            x16 = np.broadcast_to(x16.reshape(1, 16, 3), (q.shape[0], 16, 3)).copy()
        if x16.shape != (q.shape[0], 16, 3):
            raise ValueError(f"{path}: X16 must be [16,3] or [N,16,3], got {x16.shape}")
        if le.ndim != 3 or le.shape[0] != q.shape[0] or le.shape[2] != 6:
            raise ValueError(f"{path}: LE_macro must be [N,P,6], got {le.shape}")
        if b.shape != (q.shape[0], le.shape[1], 6, 48):
            raise ValueError(f"{path}: B_macro must be [N,{le.shape[1]},6,48], got {b.shape}")
        source_rows = np.asarray(z["source_row"], dtype=np.int64) if "source_row" in z.files else np.arange(q.shape[0], dtype=np.int64)
        if source_rows.shape != (q.shape[0],):
            raise ValueError(f"{path}: source_row must be [N], got {source_rows.shape}")
        source_compact = scalar_from_npz(z, "source_compact", "")
        return {
            "path": str(path),
            "q": q,
            "x16": x16,
            "le": le,
            "b": b,
            "source_row": source_rows,
            "source_compact": source_compact,
            "source_node_order": scalar_from_npz(z, "source_node_order", "macro16"),
            "node_order_transform": scalar_from_npz(z, "node_order_transform", "unknown"),
            "source_le128_key": scalar_from_npz(z, "source_le128_key", ""),
            "source_b128_key": scalar_from_npz(z, "source_b128_key", ""),
            "source_strain_coordinate_mode": scalar_from_npz(z, "source_strain_coordinate_mode", "global-to-macro-local"),
            "standard_operator_contract_version": scalar_from_npz(z, "standard_operator_contract_version", ""),
            "macro16_teacher_contract_version": scalar_from_npz(z, "macro16_teacher_contract_version", ""),
            "B_label_q_coordinate": scalar_from_npz(z, "B_label_q_coordinate", ""),
            "strain_output_coordinate": scalar_from_npz(z, "strain_output_coordinate", ""),
        }


def audit_geometry_against_source(
    macro: dict[str, Any],
    *,
    point_table: Macro16PointTable,
    interpolation_matrix: np.ndarray,
) -> dict[str, Any]:
    source = str(macro["source_compact"]).strip()
    if not source:
        return {"available": False, "reason": "macro compact has no source_compact metadata"}
    source_path = Path(source)
    if not source_path.exists():
        return {"available": False, "reason": f"source compact not found: {source}"}
    rows = np.asarray(macro["source_row"], dtype=np.int64)
    with np.load(str(source_path), allow_pickle=True) as z:
        n_total = infer_n(z, source_path)
        source_xyz = load_ip_xyz(z, n_total, rows)
    if source_xyz is None:
        return {"available": False, "reason": "source compact has no ip_xyz field"}
    _q_frames, _weights, target_xyz, geom_meta = macro16_geometry_fields(np.asarray(macro["x16"], dtype=np.float64), point_table)
    source_xyz_macro = np.einsum("pr,nri->npi", interpolation_matrix, source_xyz)
    diff = target_xyz - source_xyz_macro
    return {
        "available": True,
        "rel": rel_norm(diff, target_xyz),
        "max_abs": max_abs(diff),
        "target_xyz_rms": rms(target_xyz),
        "source_interp_xyz_rms": rms(source_xyz_macro),
        **geom_meta,
    }


def audit_plus_fd(
    macro: dict[str, Any],
    *,
    point_table: Macro16PointTable,
    interpolation_matrix: np.ndarray,
) -> dict[str, Any]:
    source = str(macro["source_compact"]).strip()
    if not source:
        return {"available": False, "reason": "macro compact has no source_compact metadata"}
    source_path = Path(source)
    if not source_path.exists():
        return {"available": False, "reason": f"source compact not found: {source}"}
    rows = np.asarray(macro["source_row"], dtype=np.int64)
    with np.load(str(source_path), allow_pickle=True) as z:
        missing = [key for key in ("LE128_plus", "perturb_directions", "delta") if key not in z.files]
        if missing:
            return {
                "available": False,
                "reason": f"source compact lacks finite-difference plus fields: {missing}",
                "source_metadata": source_metadata(z),
            }
        n_total = infer_n(z, source_path)
        le_plus_all = np.asarray(z["LE128_plus"], dtype=np.float64)
        if le_plus_all.ndim != 4 or le_plus_all.shape[0] != n_total or le_plus_all.shape[2:] != (128, 6):
            raise ValueError(f"{source_path}: LE128_plus must be [N,D,128,6], got {le_plus_all.shape}")
        le_plus = le_plus_all[rows]
        directions_source = np.asarray(z["perturb_directions"], dtype=np.int64).reshape(-1)
        if directions_source.shape[0] != le_plus.shape[1]:
            raise ValueError(f"{source_path}: perturb_directions length does not match LE128_plus D")
        delta = float(np.asarray(z["delta"], dtype=np.float64).reshape(-1)[0])
        if not math.isfinite(delta) or delta == 0.0:
            raise ValueError(f"{source_path}: bad finite difference delta {delta:g}")
        h, h_source = load_length_scale(z, rows, n_total)
        meta = source_metadata(z)
        t_eps_from = np.asarray(z["T_eps_from_abq"], dtype=np.float64) if "T_eps_from_abq" in z.files else None

    le_base = np.asarray(macro["le"], dtype=np.float64)
    mode = str(macro.get("source_strain_coordinate_mode", "global-to-macro-local")).strip().lower().replace("_", "-")
    if mode == "source-local":
        if t_eps_from is None:
            raise KeyError(f"{source_path}: source-local audit requires T_eps_from_abq")
        if t_eps_from.shape == (128, 6, 6):
            t_rows = np.broadcast_to(t_eps_from.reshape(1, 128, 6, 6), (rows.size, 128, 6, 6)).copy()
        elif t_eps_from.shape == (n_total, 128, 6, 6):
            t_rows = t_eps_from[rows]
        else:
            raise ValueError(f"{source_path}: T_eps_from_abq must be [128,6,6] or [N,128,6,6], got {t_eps_from.shape}")
        le_plus_local = np.einsum("nrab,ndrb->ndra", t_rows, le_plus)
        le_plus_macro = np.einsum("pr,ndra->ndpa", interpolation_matrix, le_plus_local)
    else:
        q_frames, _weights, _target_xyz, _geom_meta = macro16_geometry_fields(np.asarray(macro["x16"], dtype=np.float64), point_table)
        t_from, _t_to = strain_transform_matrices(q_frames)
        le_plus_global = np.einsum("pr,ndra->ndpa", interpolation_matrix, le_plus)
        le_plus_macro = np.einsum("npab,ndpb->ndpa", t_from, le_plus_global)
    fd_macro = (le_plus_macro - le_base[:, None, :, :]) / delta
    directions_macro = source_dirs_to_macro_dirs(directions_source, str(macro["source_node_order"]))
    b_dirs = np.asarray(macro["b"], dtype=np.float64)[:, :, :, directions_macro]
    b_dirs = np.moveaxis(b_dirs, -1, 1)
    raw = rel_metric(b_dirs, fd_macro)
    h4 = h.reshape(h.shape[0], 1, 1, 1)
    qhat = rel_metric(h4 * b_dirs, h4 * fd_macro)
    per_direction: list[dict[str, Any]] = []
    for pos, (src_dir, macro_dir) in enumerate(zip(directions_source.tolist(), directions_macro.tolist())):
        item = rel_metric(b_dirs[:, pos], fd_macro[:, pos])
        per_direction.append(
            {
                "source_direction": int(src_dir),
                "macro_direction": int(macro_dir),
                "rel": item["rel"],
                "cos": item["cos"],
                "diff_max_abs": item["diff_max_abs"],
            }
        )
    return {
        "available": True,
        "delta_raw": float(delta),
        "direction_count": int(directions_source.shape[0]),
        "raw_coordinate": raw,
        "qhat_coordinate_scaled_consistently": qhat,
        "source_strain_coordinate_mode": mode,
        "length_scale_source": h_source,
        "H_min": float(np.min(h)),
        "H_max": float(np.max(h)),
        "worst_direction_rel": float(max((row["rel"] for row in per_direction), default=0.0)),
        "worst_direction_max_abs": float(max((row["diff_max_abs"] for row in per_direction), default=0.0)),
        "per_direction_first16": per_direction[:16],
        "source_metadata": meta,
        "coordinate_warnings": coordinate_warnings(macro=macro, source_meta=meta),
    }


def audit_bq(macro: dict[str, Any]) -> dict[str, Any]:
    q = np.asarray(macro["q"], dtype=np.float64)
    le = np.asarray(macro["le"], dtype=np.float64)
    b = np.asarray(macro["b"], dtype=np.float64)
    bq = np.einsum("npaj,nj->npa", b, q)
    metric = rel_metric(bq, le)
    le0 = le - bq
    le0_norm = np.linalg.norm(le0.reshape(le0.shape[0], -1), axis=1)
    le_norm = np.linalg.norm(le.reshape(le.shape[0], -1), axis=1)
    le0_mean = np.mean(le0, axis=0, keepdims=True)
    le0_centered = le0 - le0_mean
    q_norm = np.linalg.norm(q, axis=1)
    return {
        **metric,
        "q_norm_min": float(np.min(q_norm)),
        "q_norm_max": float(np.max(q_norm)),
        "q_norm_mean": float(np.mean(q_norm)),
        "LE0_star_definition": "LE0_star = LE_macro - B_macro(q) @ q48",
        "LE0_star_rms": rms(le0),
        "LE0_star_norm_min": float(np.min(le0_norm)),
        "LE0_star_norm_max": float(np.max(le0_norm)),
        "LE0_star_norm_mean": float(np.mean(le0_norm)),
        "LE0_star_rel_to_LE": rel_norm(le0, le),
        "LE0_star_centered_rel_to_LE0": rel_norm(le0_centered, le0),
        "LE0_star_centered_rms": rms(le0_centered),
        "LE_norm_min": float(np.min(le_norm)),
        "LE_norm_max": float(np.max(le_norm)),
        "LE_norm_mean": float(np.mean(le_norm)),
    }


def audit_frame_fd(macro: dict[str, Any]) -> dict[str, Any]:
    q = np.asarray(macro["q"], dtype=np.float64)
    x16 = np.asarray(macro["x16"], dtype=np.float64)
    le = np.asarray(macro["le"], dtype=np.float64)
    b = np.asarray(macro["b"], dtype=np.float64)
    source_rows = np.asarray(macro["source_row"], dtype=np.int64)
    groups: dict[tuple[float, ...], list[int]] = {}
    for i, row in enumerate(np.round(x16.reshape(x16.shape[0], -1), 10)):
        groups.setdefault(tuple(float(v) for v in row.tolist()), []).append(i)
    pred_i: list[np.ndarray] = []
    pred_avg: list[np.ndarray] = []
    refs: list[np.ndarray] = []
    dq_norms: list[float] = []
    for idxs in groups.values():
        idxs = sorted(idxs, key=lambda i: int(source_rows[i]))
        for a, bidx in zip(idxs[:-1], idxs[1:]):
            dq = q[bidx] - q[a]
            dle = le[bidx] - le[a]
            pred_i.append(np.einsum("paj,j->pa", b[a], dq))
            pred_avg.append(np.einsum("paj,j->pa", 0.5 * (b[a] + b[bidx]), dq))
            refs.append(dle)
            dq_norms.append(float(np.linalg.norm(dq)))
    if not refs:
        return {"available": False, "reason": "fewer than two frames per geometry group"}
    ref = np.stack(refs, axis=0)
    pi = np.stack(pred_i, axis=0)
    pa = np.stack(pred_avg, axis=0)
    return {
        "available": True,
        "pair_count": int(ref.shape[0]),
        "forward_B_at_left": rel_metric(pi, ref),
        "trapezoid_average_B": rel_metric(pa, ref),
        "dq_norm_min": float(np.min(dq_norms)),
        "dq_norm_max": float(np.max(dq_norms)),
        "dq_norm_mean": float(np.mean(dq_norms)),
    }


def audit_scale_metadata(macro: dict[str, Any], plus: dict[str, Any]) -> dict[str, Any]:
    h_min = plus.get("H_min") if plus.get("available") else None
    h_max = plus.get("H_max") if plus.get("available") else None
    return {
        "macro_B_label_q_coordinate": macro.get("B_label_q_coordinate", ""),
        "macro_q_key": "q48_raw",
        "source_length_scale_source": plus.get("length_scale_source") if plus.get("available") else None,
        "H_min": finite_float(h_min),
        "H_max": finite_float(h_max),
        "note": "Finite-difference LE_plus checks dLE/dq_raw. If using q_hat=q_raw/H in training, scale both FD and B by H; relative error is unchanged when H is consistent.",
    }


def audit_one(path: Path, point_table: Macro16PointTable, interpolation_matrix: np.ndarray) -> dict[str, Any]:
    macro = load_macro_compact(path)
    geometry = audit_geometry_against_source(macro, point_table=point_table, interpolation_matrix=interpolation_matrix)
    plus = audit_plus_fd(macro, point_table=point_table, interpolation_matrix=interpolation_matrix)
    bq = audit_bq(macro)
    frame_fd = audit_frame_fd(macro)
    scale = audit_scale_metadata(macro, plus)
    return {
        "compact": str(path),
        "standard_operator_contract_version": macro["standard_operator_contract_version"],
        "macro16_teacher_contract_version": macro["macro16_teacher_contract_version"],
        "frame_count": int(macro["q"].shape[0]),
        "point_count": int(macro["le"].shape[1]),
        "source_compact": macro["source_compact"],
        "source_node_order": macro["source_node_order"],
        "node_order_transform": macro["node_order_transform"],
        "source_le128_key": macro["source_le128_key"],
        "source_b128_key": macro["source_b128_key"],
        "source_strain_coordinate_mode": macro["source_strain_coordinate_mode"],
        "strain_output_coordinate": macro["strain_output_coordinate"],
        "geometry_position": geometry,
        "plus_fd_B_consistency": plus,
        "LE_vs_Bq_at_frames": bq,
        "frame_to_frame_fd": frame_fd,
        "scale_metadata": scale,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(path: tuple[str, ...]) -> list[float]:
        out: list[float] = []
        for row in rows:
            cur: Any = row
            for key in path:
                if not isinstance(cur, dict) or key not in cur:
                    cur = None
                    break
                cur = cur[key]
            if cur is not None:
                val = float(cur)
                if math.isfinite(val):
                    out.append(val)
        return out

    def stats(name: str, path: tuple[str, ...]) -> dict[str, Any]:
        data = vals(path)
        return {
            f"{name}_count": int(len(data)),
            f"{name}_max": float(max(data)) if data else None,
            f"{name}_mean": float(np.mean(data)) if data else None,
        }

    out: dict[str, Any] = {"compact_count": int(len(rows))}
    out.update(stats("geometry_rel", ("geometry_position", "rel")))
    out.update(stats("plus_fd_rel", ("plus_fd_B_consistency", "raw_coordinate", "rel")))
    out.update(stats("bq_rel", ("LE_vs_Bq_at_frames", "rel")))
    out.update(stats("LE0_star_rel", ("LE_vs_Bq_at_frames", "LE0_star_rel_to_LE")))
    out.update(stats("LE0_star_centered_rel", ("LE_vs_Bq_at_frames", "LE0_star_centered_rel_to_LE0")))
    out.update(stats("frame_fd_rel", ("frame_to_frame_fd", "trapezoid_average_B", "rel")))
    warnings = []
    for row in rows:
        warnings.extend(row.get("plus_fd_B_consistency", {}).get("coordinate_warnings", []))
    out["coordinate_warning_count"] = int(len(warnings))
    out["coordinate_warnings_first10"] = warnings[:10]
    return out


def apply_thresholds(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    threshold_specs = [
        ("geometry_rel", "geometry_rel_max", float(args.max_geometry_rel)),
        ("plus_fd_rel", "plus_fd_rel_max", float(args.max_plus_fd_rel)),
        ("bq_rel", "bq_rel_max", float(args.max_bq_rel)),
        ("frame_fd_rel", "frame_fd_rel_max", float(args.max_frame_fd_rel)),
    ]
    agg = summary["aggregate"]
    for name, key, limit in threshold_specs:
        value = agg.get(key)
        enforced = limit > 0.0 and value is not None
        passed = (not enforced) or float(value) <= float(limit)
        checks.append({"name": name, "value": value, "limit": limit, "enforced": enforced, "passed": bool(passed)})
    summary["threshold_checks"] = checks
    summary["strict_pass"] = bool(all(row["passed"] for row in checks))
    return summary


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    paths = compact_paths_from_args(args)
    if not paths:
        raise ValueError("provide --compact or --compact-list")
    point_table = macro16_standard_point_table(
        plane_order=int(args.plane_gauss_order),
        thickness_order=int(args.thickness_gauss_order),
    )
    w = macro16_from_128_interpolation_matrix(point_table)
    rows = [audit_one(path, point_table, w) for path in paths]
    summary = {
        "script": "audit_macro16_teacher_labels",
        "standard_operator_contract_version": MACRO16_CONTRACT_VERSION,
        "macro16_teacher_contract_version": TEACHER_CONTRACT_VERSION,
        "compact_paths": [str(path) for path in paths],
        "aggregate": aggregate(rows),
        "compacts": rows,
    }
    summary = apply_thresholds(summary, args)
    out = Path(args.out).resolve()
    write_json(out, summary)
    print(
        json.dumps(
            {
                "summary": str(out),
                "strict_pass": summary["strict_pass"],
                **summary["aggregate"],
            },
            ensure_ascii=False,
            sort_keys=True,
            default=json_default,
        )
    )
    if bool(args.strict) and not bool(summary["strict_pass"]):
        raise SystemExit(1)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="append", type=Path, default=[])
    parser.add_argument("--compact-list", action="append", type=Path, default=[])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--plane-gauss-order", type=int, default=3)
    parser.add_argument("--thickness-gauss-order", type=int, default=2)
    parser.add_argument("--max-geometry-rel", type=float, default=0.0)
    parser.add_argument("--max-plus-fd-rel", type=float, default=0.0)
    parser.add_argument("--max-bq-rel", type=float, default=0.0)
    parser.add_argument("--max-frame-fd-rel", type=float, default=0.0)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_audit(parse_args())


if __name__ == "__main__":
    main()
