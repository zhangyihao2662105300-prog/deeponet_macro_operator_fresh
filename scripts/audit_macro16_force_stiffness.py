#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit Macro16 teacher force and stiffness closure.

This script does not train a network.  It checks whether the exported
Macro16 teacher fields can assemble boundary force and material stiffness
that agree with the fine-grid projected boundary responses.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

TRUE176_SURFACE_MACRO_TO_KEEP = np.asarray([0, 1, 2, 4, 7, 6, 5, 3], dtype=np.int64)
TRUE176_MACRO_TO_KEEP_NODE = np.concatenate([TRUE176_SURFACE_MACRO_TO_KEEP, TRUE176_SURFACE_MACRO_TO_KEEP + 8])
TRUE176_MACRO_TO_KEEP_FLAT = np.asarray(
    [int(node) * 3 + axis for node in TRUE176_MACRO_TO_KEEP_NODE for axis in range(3)],
    dtype=np.int64,
)

WEIGHT_MODES = ("lref3", "as-stored", "source-volume-sum", "source-inferred-volume-sum")


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


def collect_paths(compacts: Iterable[Path], compact_lists: Iterable[Path], compact_globs: Iterable[str], *, case_limit: int) -> list[Path]:
    paths = [Path(p).resolve() for p in compacts]
    for list_path in compact_lists:
        paths.extend(read_path_list(Path(list_path)))
    for pattern in compact_globs:
        paths.extend(Path(hit).resolve() for hit in sorted(glob.glob(str(pattern))))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if int(case_limit) > 0:
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


def max_abs(arr: np.ndarray) -> float:
    vals = np.asarray(arr, dtype=np.float64)
    return float(np.max(np.abs(vals))) if vals.size else 0.0


def rms(arr: np.ndarray) -> float:
    vals = np.asarray(arr, dtype=np.float64)
    return float(np.sqrt(np.mean(vals * vals))) if vals.size else 0.0


def rel_norm(diff: np.ndarray, ref: np.ndarray) -> float:
    den = max(float(np.linalg.norm(np.asarray(ref, dtype=np.float64).reshape(-1))), 1.0e-30)
    return float(np.linalg.norm(np.asarray(diff, dtype=np.float64).reshape(-1)) / den)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    den = max(float(np.linalg.norm(aa) * np.linalg.norm(bb)), 1.0e-30)
    return float(np.dot(aa, bb) / den)


def metric(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    return {
        "rel": rel_norm(diff, ref),
        "cos": cosine(pred, ref),
        "pred_rms": rms(pred),
        "ref_rms": rms(ref),
        "diff_rms": rms(diff),
        "diff_max_abs": max_abs(diff),
    }


def characteristic_length(x16: np.ndarray) -> float:
    nodes = np.asarray(x16, dtype=np.float64).reshape(16, 3)
    span = np.max(nodes, axis=0) - np.min(nodes, axis=0)
    l_ref = float(np.max(span))
    if not math.isfinite(l_ref) or l_ref <= 1.0e-14:
        center = np.mean(nodes, axis=0)
        l_ref = float(np.sqrt(np.mean(np.sum((nodes - center.reshape(1, 3)) ** 2, axis=1))))
    if not math.isfinite(l_ref) or l_ref <= 1.0e-14:
        raise ValueError("cannot compute nondegenerate L_ref from X16")
    return l_ref


def macro_to_source_order(source_node_order: str) -> np.ndarray:
    order = str(source_node_order).strip().lower().replace("_", "-")
    if order == "true176-keep":
        return TRUE176_MACRO_TO_KEEP_FLAT.copy()
    return np.arange(48, dtype=np.int64)


def source_to_macro_order(source_node_order: str) -> np.ndarray:
    macro_to_source = macro_to_source_order(source_node_order)
    inv = np.empty(48, dtype=np.int64)
    inv[macro_to_source] = np.arange(48, dtype=np.int64)
    return inv


def source_vectors_to_macro(vals: np.ndarray, source_node_order: str) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.shape[-1] != 48:
        raise ValueError(f"source vector last dimension must be 48, got {arr.shape}")
    return arr[..., macro_to_source_order(source_node_order)]


def source_stiffness_to_macro(vals: np.ndarray, source_node_order: str) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (48, 48):
        raise ValueError(f"source stiffness must be [N,48,48], got {arr.shape}")
    order = macro_to_source_order(source_node_order)
    return arr[:, order, :][:, :, order]


def source_stiffness_to_macro_subset(
    rf_base_source: np.ndarray,
    rf_plus_source: np.ndarray,
    directions_source: np.ndarray,
    delta: float,
    source_node_order: str,
) -> tuple[np.ndarray, np.ndarray]:
    base_macro = source_vectors_to_macro(rf_base_source, source_node_order)
    plus_macro = source_vectors_to_macro(rf_plus_source, source_node_order)
    dirs_source = np.asarray(directions_source, dtype=np.int64).reshape(-1)
    if plus_macro.ndim != 3 or plus_macro.shape[1] != dirs_source.size or plus_macro.shape[2] != 48:
        raise ValueError(
            f"RF_projected_plus must have shape [N,{dirs_source.size},48] after row selection, got {plus_macro.shape}"
        )
    if not math.isfinite(float(delta)) or float(delta) == 0.0:
        raise ValueError(f"delta must be finite and nonzero, got {delta:g}")
    source_to_macro = source_to_macro_order(source_node_order)
    dirs_macro = source_to_macro[dirs_source]
    kfd_subset = np.empty((base_macro.shape[0], 48, dirs_macro.size), dtype=np.float64)
    for pos in range(dirs_macro.size):
        kfd_subset[:, :, pos] = (plus_macro[:, pos, :] - base_macro) / float(delta)
    return kfd_subset, dirs_macro.astype(np.int64)


def row_select_or_broadcast(vals: np.ndarray, rows: np.ndarray, tail: tuple[int, ...], key: str) -> np.ndarray:
    arr = np.asarray(vals)
    if arr.shape == tuple(tail):
        return np.broadcast_to(arr.reshape((1,) + tuple(tail)), (rows.size,) + tuple(tail)).copy()
    if arr.shape == (1,) + tuple(tail):
        return np.broadcast_to(arr.reshape((1,) + tuple(tail)), (rows.size,) + tuple(tail)).copy()
    if arr.ndim == len(tail) + 1 and arr.shape[1:] == tuple(tail):
        return arr[rows]
    raise ValueError(f"{key} must have shape {tail} or [N,{','.join(str(v) for v in tail)}], got {arr.shape}")


def load_macro(path: Path) -> dict[str, Any]:
    with np.load(str(path), allow_pickle=True) as z:
        required = {"q48_raw", "X16", "LE_macro", "B_macro", "integration_weight_hat", "source_compact", "source_row"}
        missing = sorted(required.difference(z.files))
        if missing:
            raise KeyError(f"{path}: missing Macro16 fields {missing}")
        q = np.asarray(z["q48_raw"], dtype=np.float64)
        x16 = np.asarray(z["X16"], dtype=np.float64)
        le = np.asarray(z["LE_macro"], dtype=np.float64)
        b = np.asarray(z["B_macro"], dtype=np.float64)
        weights = np.asarray(z["integration_weight_hat"], dtype=np.float64)
        rows = np.asarray(z["source_row"], dtype=np.int64).reshape(-1)
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
        if weights.shape != (q.shape[0], le.shape[1]):
            raise ValueError(f"{path}: integration_weight_hat must be [N,{le.shape[1]}], got {weights.shape}")
        if rows.shape != (q.shape[0],):
            raise ValueError(f"{path}: source_row must be [N], got {rows.shape}")
        return {
            "path": str(path),
            "q": q,
            "x16": x16,
            "le": le,
            "b": b,
            "weights_hat": weights,
            "source_rows": rows,
            "source_compact": scalar_text(z, "source_compact"),
            "source_node_order": scalar_text(z, "source_node_order", "macro16"),
            "strain_output_coordinate": scalar_text(z, "strain_output_coordinate", ""),
            "B_label_q_coordinate": scalar_text(z, "B_label_q_coordinate", ""),
        }


def load_source(path: Path, rows: np.ndarray) -> dict[str, Any]:
    with np.load(str(path), allow_pickle=True) as z:
        required = {"RF_projected", "elastic_D"}
        missing = sorted(required.difference(z.files))
        if missing:
            raise KeyError(f"{path}: missing source fields {missing}")
        rf = row_select_or_broadcast(np.asarray(z["RF_projected"], dtype=np.float64), rows, (48,), "RF_projected")
        elastic_d = np.asarray(z["elastic_D"], dtype=np.float64)
        if elastic_d.shape != (6, 6):
            raise ValueError(f"{path}: elastic_D must be [6,6], got {elastic_d.shape}")
        out: dict[str, Any] = {"path": str(path), "rf": rf, "elastic_D": elastic_d}
        if "RF_projected_plus" in z.files:
            rf_plus = np.asarray(z["RF_projected_plus"], dtype=np.float64)
            if rf_plus.ndim != 3 or rf_plus.shape[2] != 48:
                raise ValueError(f"{path}: RF_projected_plus must be [N,D,48], got {rf_plus.shape}")
            out["rf_plus"] = rf_plus[rows]
            out["delta"] = float(np.asarray(z["delta"], dtype=np.float64).reshape(-1)[0]) if "delta" in z.files else None
            if "perturb_directions" in z.files:
                out["perturb_directions"] = np.asarray(z["perturb_directions"], dtype=np.int64).reshape(-1)
            else:
                out["perturb_directions"] = np.arange(rf_plus.shape[1], dtype=np.int64)
        if "F_DLE_IVOL" in z.files:
            out["F_DLE_IVOL"] = row_select_or_broadcast(np.asarray(z["F_DLE_IVOL"], dtype=np.float64), rows, (48,), "F_DLE_IVOL")
        if "LE128_base" in z.files:
            le128 = row_select_or_broadcast(np.asarray(z["LE128_base"], dtype=np.float64), rows, (128, 6), "LE128_base")
            out["LE128_base"] = le128
        if "B_LE128_forward" in z.files:
            b128 = row_select_or_broadcast(np.asarray(z["B_LE128_forward"], dtype=np.float64), rows, (128, 6, 48), "B_LE128_forward")
            out["B_LE128_forward"] = b128
        if {"LE128_base", "B_LE128_forward", "ip_IVOL_abaqus"}.issubset(z.files):
            ivol = row_select_or_broadcast(np.asarray(z["ip_IVOL_abaqus"], dtype=np.float64), rows, (128,), "ip_IVOL_abaqus")
            stress128 = np.einsum("ab,nrb->nra", elastic_d, le128)
            out["F128_D_IVOL"] = np.einsum("nraj,nra,nr->nj", b128, stress128, ivol)
        if "IVOL128_inferred_from_DLE" in z.files:
            out["source_inferred_volume"] = row_select_or_broadcast(
                np.asarray(z["IVOL128_inferred_from_DLE"], dtype=np.float64),
                rows,
                (128,),
                "IVOL128_inferred_from_DLE",
            )
        if "ip_IVOL_abaqus" in z.files:
            out["source_ip_volume"] = row_select_or_broadcast(
                np.asarray(z["ip_IVOL_abaqus"], dtype=np.float64),
                rows,
                (128,),
                "ip_IVOL_abaqus",
            )
        return out


def physical_weights(macro: dict[str, Any], source: dict[str, Any], weight_mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    weights = np.asarray(macro["weights_hat"], dtype=np.float64).copy()
    mode = str(weight_mode).strip().lower().replace("_", "-")
    if mode not in WEIGHT_MODES:
        raise ValueError(f"weight_mode must be one of {WEIGHT_MODES}, got {weight_mode!r}")
    l_ref = np.asarray([characteristic_length(x) for x in np.asarray(macro["x16"], dtype=np.float64)], dtype=np.float64)
    if mode == "lref3":
        weights *= (l_ref ** 3).reshape(-1, 1)
    elif mode == "source-volume-sum":
        if "source_ip_volume" not in source:
            raise KeyError("weight-mode source-volume-sum requires ip_IVOL_abaqus in source compact")
        scale = np.sum(np.asarray(source["source_ip_volume"], dtype=np.float64), axis=1) / np.sum(weights, axis=1)
        weights *= scale.reshape(-1, 1)
    elif mode == "source-inferred-volume-sum":
        if "source_inferred_volume" not in source:
            raise KeyError("weight-mode source-inferred-volume-sum requires IVOL128_inferred_from_DLE in source compact")
        scale = np.sum(np.asarray(source["source_inferred_volume"], dtype=np.float64), axis=1) / np.sum(weights, axis=1)
        weights *= scale.reshape(-1, 1)
    return weights, {
        "weight_mode": mode,
        "L_ref_min": float(np.min(l_ref)),
        "L_ref_max": float(np.max(l_ref)),
        "macro_weight_sum_min": float(np.min(np.sum(weights, axis=1))),
        "macro_weight_sum_max": float(np.max(np.sum(weights, axis=1))),
    }


def assemble_force_stiffness(le: np.ndarray, b: np.ndarray, weights: np.ndarray, elastic_d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stress = np.einsum("ab,npb->npa", elastic_d, le)
    force = np.einsum("npaj,npa,np->nj", b, stress, weights)
    stiffness = np.einsum("npaj,ab,npbk,np->njk", b, elastic_d, b, weights)
    energy_density_twice = np.einsum("npa,npa,np->n", le, stress, weights)
    return force, stiffness, energy_density_twice


def candidate_report(
    *,
    name: str,
    point_count: int,
    force: np.ndarray,
    stiffness: np.ndarray,
    energy_twice: np.ndarray,
    source_rf_macro: np.ndarray,
    macro: dict[str, Any],
    kfd_subset: np.ndarray | None,
    dirs_macro: np.ndarray | None,
) -> dict[str, Any]:
    force_metric = metric(force, source_rf_macro)
    stiffness_report: dict[str, Any] = {"available": False, "reason": "source compact lacks RF_projected_plus"}
    plus_kdq_report: dict[str, Any] = {"available": False, "reason": "source compact lacks RF_projected_plus"}
    if kfd_subset is not None and dirs_macro is not None:
        subset = stiffness[:, :, dirs_macro]
        stiffness_report = {
            "available": True,
            "direction_count": int(dirs_macro.size),
            "macro_directions_first16": dirs_macro[:16].astype(int).tolist(),
            **metric(subset, kfd_subset),
        }
        plus_kdq_report = {
            "available": True,
            "definition": "K_candidate[:, dir] * delta versus RF_projected_plus(dir) - RF_projected",
            **metric(subset, kfd_subset),
        }
    return {
        "name": name,
        "point_count": int(point_count),
        "force": force_metric,
        "force_dof_max_abs_error": float(np.max(np.abs(force - source_rf_macro), axis=None)),
        "stiffness": stiffness_report,
        "stiffness_symmetry": stiffness_symmetry(stiffness),
        "plus_Kdq_vs_dF": plus_kdq_report,
        "frame_to_frame_Kdq_vs_dF_source_RF": frame_to_frame_kdq(macro, source_rf_macro, stiffness),
        "energy": energy_checks(np.asarray(macro["q"], dtype=np.float64), force, source_rf_macro, energy_twice),
    }


def source128_candidate(
    *,
    name: str,
    source: dict[str, Any],
    source_order: str,
    weights_key: str,
    source_rf_macro: np.ndarray,
    macro: dict[str, Any],
    kfd_subset: np.ndarray | None,
    dirs_macro: np.ndarray | None,
) -> dict[str, Any] | None:
    required = {"LE128_base", "B_LE128_forward", weights_key}
    missing = sorted(key for key in required if key not in source)
    if missing:
        return {"name": name, "available": False, "reason": f"missing source fields {missing}"}
    le128 = np.asarray(source["LE128_base"], dtype=np.float64)
    b128 = np.asarray(source["B_LE128_forward"], dtype=np.float64)
    weights = np.asarray(source[weights_key], dtype=np.float64)
    force_src, stiffness_src, energy_twice = assemble_force_stiffness(
        le128,
        b128,
        weights,
        np.asarray(source["elastic_D"], dtype=np.float64),
    )
    force = source_vectors_to_macro(force_src, source_order)
    stiffness = source_stiffness_to_macro(stiffness_src, source_order)
    report = candidate_report(
        name=name,
        point_count=128,
        force=force,
        stiffness=stiffness,
        energy_twice=energy_twice,
        source_rf_macro=source_rf_macro,
        macro=macro,
        kfd_subset=kfd_subset,
        dirs_macro=dirs_macro,
    )
    report["available"] = True
    report["weights_key"] = weights_key
    report["weight_sum_min"] = float(np.min(np.sum(weights, axis=1)))
    report["weight_sum_max"] = float(np.max(np.sum(weights, axis=1)))
    return report


def stiffness_symmetry(stiffness: np.ndarray) -> dict[str, float]:
    skew = stiffness - np.swapaxes(stiffness, 1, 2)
    return {
        "rel": rel_norm(skew, stiffness),
        "max_abs": max_abs(skew),
    }


def frame_to_frame_kdq(macro: dict[str, Any], force_ref: np.ndarray, stiffness: np.ndarray) -> dict[str, Any]:
    q = np.asarray(macro["q"], dtype=np.float64)
    x16 = np.asarray(macro["x16"], dtype=np.float64)
    groups: dict[tuple[float, ...], list[int]] = {}
    for i, flat in enumerate(np.round(x16.reshape(x16.shape[0], -1), 10)):
        groups.setdefault(tuple(float(v) for v in flat.tolist()), []).append(i)
    preds: list[np.ndarray] = []
    refs: list[np.ndarray] = []
    dq_norms: list[float] = []
    for idxs in groups.values():
        idxs = sorted(idxs)
        for left, right in zip(idxs[:-1], idxs[1:]):
            dq = q[right] - q[left]
            refs.append(force_ref[right] - force_ref[left])
            preds.append(stiffness[left] @ dq)
            dq_norms.append(float(np.linalg.norm(dq)))
    if not refs:
        return {"available": False, "reason": "fewer than two frames per geometry group"}
    pred = np.stack(preds, axis=0)
    ref = np.stack(refs, axis=0)
    return {
        "available": True,
        "pair_count": int(ref.shape[0]),
        "dq_norm_min": float(np.min(dq_norms)),
        "dq_norm_max": float(np.max(dq_norms)),
        "dq_norm_mean": float(np.mean(dq_norms)),
        **metric(pred, ref),
    }


def energy_checks(q: np.ndarray, force_macro: np.ndarray, force_ref: np.ndarray, strain_energy_twice: np.ndarray) -> dict[str, Any]:
    q_dot_macro = np.einsum("nj,nj->n", q, force_macro)
    q_dot_ref = np.einsum("nj,nj->n", q, force_ref)
    return {
        "macro_q_dot_F_vs_integral_LE_sigma": metric(q_dot_macro, strain_energy_twice),
        "source_q_dot_RF_vs_macro_q_dot_F": metric(q_dot_macro, q_dot_ref),
        "note": "For a zero-initial linear path, q dot F should match integral LE dot sigma dV. Nonzero LE0 or nonlinear paths can break this secant check.",
    }


def audit_one(path: Path, weight_mode: str) -> dict[str, Any]:
    macro = load_macro(path)
    source_path = Path(str(macro["source_compact"]))
    if not source_path.exists():
        raise FileNotFoundError(f"{path}: source compact not found: {source_path}")
    source = load_source(source_path, np.asarray(macro["source_rows"], dtype=np.int64))
    source_order = str(macro["source_node_order"])
    source_rf_macro = source_vectors_to_macro(np.asarray(source["rf"], dtype=np.float64), source_order)
    weights, weight_meta = physical_weights(macro, source, weight_mode)
    force_macro, stiffness_macro, energy_twice = assemble_force_stiffness(
        np.asarray(macro["le"], dtype=np.float64),
        np.asarray(macro["b"], dtype=np.float64),
        weights,
        np.asarray(source["elastic_D"], dtype=np.float64),
    )
    kfd_subset: np.ndarray | None = None
    dirs_macro: np.ndarray | None = None
    source_kfd_symmetry: dict[str, Any] | None = None
    if "rf_plus" in source:
        if source.get("delta") is None:
            raise KeyError(f"{source_path}: RF_projected_plus requires delta")
        kfd_subset, dirs_macro = source_stiffness_to_macro_subset(
            np.asarray(source["rf"], dtype=np.float64),
            np.asarray(source["rf_plus"], dtype=np.float64),
            np.asarray(source["perturb_directions"], dtype=np.int64),
            float(source["delta"]),
            source_order,
        )
        if dirs_macro.size == 48 and sorted(dirs_macro.astype(int).tolist()) == list(range(48)):
            full_kfd = np.empty_like(stiffness_macro)
            for pos, macro_dir in enumerate(dirs_macro.tolist()):
                full_kfd[:, :, int(macro_dir)] = kfd_subset[:, :, pos]
            source_kfd_symmetry = stiffness_symmetry(full_kfd)
    macro18 = candidate_report(
        name="macro16_18pt",
        point_count=int(np.asarray(macro["le"]).shape[1]),
        force=force_macro,
        stiffness=stiffness_macro,
        energy_twice=energy_twice,
        source_rf_macro=source_rf_macro,
        macro=macro,
        kfd_subset=kfd_subset,
        dirs_macro=dirs_macro,
    )
    source128_ip_volume = source128_candidate(
        name="source128_ip_volume",
        source=source,
        source_order=source_order,
        weights_key="source_ip_volume",
        source_rf_macro=source_rf_macro,
        macro=macro,
        kfd_subset=kfd_subset,
        dirs_macro=dirs_macro,
    )
    source128_inferred_volume = source128_candidate(
        name="source128_inferred_volume",
        source=source,
        source_order=source_order,
        weights_key="source_inferred_volume",
        source_rf_macro=source_rf_macro,
        macro=macro,
        kfd_subset=kfd_subset,
        dirs_macro=dirs_macro,
    )
    source_force_checks: dict[str, Any] = {}
    if "F_DLE_IVOL" in source:
        source_force_checks["F_DLE_IVOL_vs_RF_projected"] = metric(
            source_vectors_to_macro(np.asarray(source["F_DLE_IVOL"], dtype=np.float64), source_order),
            source_rf_macro,
        )
    if "F128_D_IVOL" in source:
        source_force_checks["F128_D_IVOL_vs_RF_projected"] = metric(
            source_vectors_to_macro(np.asarray(source["F128_D_IVOL"], dtype=np.float64), source_order),
            source_rf_macro,
        )
    return {
        "compact": str(path),
        "source_compact": str(source_path),
        "frame_count": int(np.asarray(macro["q"]).shape[0]),
        "point_count": int(np.asarray(macro["le"]).shape[1]),
        "source_node_order": source_order,
        "strain_output_coordinate": macro["strain_output_coordinate"],
        "B_label_q_coordinate": macro["B_label_q_coordinate"],
        "weight": weight_meta,
        "force": macro18["force"],
        "force_dof_max_abs_error": macro18["force_dof_max_abs_error"],
        "stiffness": macro18["stiffness"],
        "stiffness_symmetry_macro": macro18["stiffness_symmetry"],
        "stiffness_symmetry_source_fd": source_kfd_symmetry,
        "plus_Kdq_vs_dF": macro18["plus_Kdq_vs_dF"],
        "frame_to_frame_Kdq_vs_dF_source_RF": macro18["frame_to_frame_Kdq_vs_dF_source_RF"],
        "energy": macro18["energy"],
        "candidates": {
            "macro16_18pt": macro18,
            "source128_ip_volume": source128_ip_volume,
            "source128_inferred_volume": source128_inferred_volume,
        },
        "source_force_checks": source_force_checks,
        "interpretation_note": (
            "Large Macro16 force/stiffness errors here are teacher-label assembly errors, not network training errors. "
            "Compare candidates.macro16_18pt against candidates.source128_ip_volume to separate 18-point reduction error from source 128-point closure."
        ),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(path: tuple[str, ...]) -> list[float]:
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
        data = values(path)
        return {
            f"{name}_count": int(len(data)),
            f"{name}_mean": float(np.mean(data)) if data else None,
            f"{name}_max": float(max(data)) if data else None,
        }

    out: dict[str, Any] = {"compact_count": int(len(rows))}
    out.update(stats("force_rel", ("force", "rel")))
    out.update(stats("stiffness_rel", ("stiffness", "rel")))
    out.update(stats("macro_stiffness_symmetry_rel", ("stiffness_symmetry_macro", "rel")))
    out.update(stats("source_fd_stiffness_symmetry_rel", ("stiffness_symmetry_source_fd", "rel")))
    out.update(stats("plus_Kdq_vs_dF_rel", ("plus_Kdq_vs_dF", "rel")))
    out.update(stats("frame_Kdq_vs_dF_rel", ("frame_to_frame_Kdq_vs_dF_source_RF", "rel")))
    out.update(stats("energy_qF_vs_integral_rel", ("energy", "macro_q_dot_F_vs_integral_LE_sigma", "rel")))
    out.update(stats("source_F_DLE_vs_RF_rel", ("source_force_checks", "F_DLE_IVOL_vs_RF_projected", "rel")))
    out.update(stats("source_F128_D_IVOL_vs_RF_rel", ("source_force_checks", "F128_D_IVOL_vs_RF_projected", "rel")))
    return out


def apply_thresholds(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    specs = [
        ("force_rel", "force_rel_max", float(args.max_force_rel)),
        ("stiffness_rel", "stiffness_rel_max", float(args.max_stiffness_rel)),
        ("macro_stiffness_symmetry_rel", "macro_stiffness_symmetry_rel_max", float(args.max_macro_stiffness_symmetry_rel)),
        ("plus_Kdq_vs_dF_rel", "plus_Kdq_vs_dF_rel_max", float(args.max_plus_kdq_rel)),
        ("source_F_DLE_vs_RF_rel", "source_F_DLE_vs_RF_rel_max", float(args.max_source_force_rel)),
        ("source_F128_D_IVOL_vs_RF_rel", "source_F128_D_IVOL_vs_RF_rel_max", float(args.max_source_128_force_rel)),
    ]
    checks: list[dict[str, Any]] = []
    agg = summary["aggregate"]
    for name, key, limit in specs:
        value = agg.get(key)
        enforced = limit > 0.0 and value is not None
        passed = (not enforced) or float(value) <= limit
        checks.append({"name": name, "value": value, "limit": limit, "enforced": enforced, "passed": bool(passed)})
    summary["threshold_checks"] = checks
    summary["strict_pass"] = bool(all(row["passed"] for row in checks))
    return summary


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    paths = collect_paths(args.compact, args.compact_list, args.compact_glob, case_limit=int(args.case_limit))
    if not paths:
        raise ValueError("provide --compact, --compact-list, or --compact-glob")
    rows = [audit_one(path, str(args.weight_mode)) for path in paths]
    summary = {
        "script": "audit_macro16_force_stiffness",
        "compact_paths": [str(path) for path in paths],
        "weight_mode": str(args.weight_mode),
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
    parser.add_argument("--compact-glob", action="append", default=[])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--weight-mode", choices=WEIGHT_MODES, default="lref3")
    parser.add_argument("--max-force-rel", type=float, default=0.0)
    parser.add_argument("--max-stiffness-rel", type=float, default=0.0)
    parser.add_argument("--max-macro-stiffness-symmetry-rel", type=float, default=0.0)
    parser.add_argument("--max-plus-kdq-rel", type=float, default=0.0)
    parser.add_argument("--max-source-force-rel", type=float, default=0.0)
    parser.add_argument("--max-source-128-force-rel", type=float, default=0.0)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_audit(parse_args())


if __name__ == "__main__":
    main()
