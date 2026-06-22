#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a v1.2 query-point data-coverage audit ledger.

This script does not train a model.  It records whether a set of real
training-ready complete compacts covers enough independent q/load directions to
justify the next formal case-level training audit.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


def case_id_from_path(path: str) -> int:
    match = re.search(r"case(\d+)", str(path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def canonical_strain_field(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "unknown"
    key = text.upper()
    if key in {"UNKNOWN", "NONE", "NULL", "NA", "N/A"}:
        return "unknown"
    return key


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


def read_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def scalar_text(z: np.lib.npyio.NpzFile, key: str, default: str = "unknown") -> str:
    if key not in z.files:
        return str(default)
    arr = np.asarray(z[key])
    if arr.size == 0:
        return str(default)
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def scalar_float(z: np.lib.npyio.NpzFile, key: str, default: float = math.nan) -> float:
    if key not in z.files:
        return float(default)
    arr = np.asarray(z[key])
    if arr.size == 0:
        return float(default)
    try:
        return float(arr.reshape(-1)[0])
    except Exception:
        return float(default)


def scalar_bool(z: np.lib.npyio.NpzFile, key: str, default: bool = False) -> bool:
    if key not in z.files:
        return bool(default)
    arr = np.asarray(z[key])
    if arr.size == 0:
        return bool(default)
    item = arr.reshape(-1)[0]
    if isinstance(item, (np.bool_, bool)):
        return bool(item)
    text = str(item).strip().lower()
    return text in {"1", "true", "yes", "y"}


def finite_or_none(value: float) -> float | None:
    val = float(value)
    return val if math.isfinite(val) else None


def rms_per_frame(arr: np.ndarray) -> np.ndarray:
    vals = np.asarray(arr, dtype=np.float64)
    if vals.ndim < 2:
        raise ValueError(f"expected frame-major array, got shape {vals.shape}")
    return np.sqrt(np.mean(vals * vals, axis=tuple(range(1, vals.ndim))))


def normalize(vec: np.ndarray, eps: float = 1.0e-12) -> tuple[np.ndarray, bool]:
    vals = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vals))
    if not math.isfinite(norm) or norm <= eps:
        return np.zeros_like(vals, dtype=np.float64), False
    return vals / norm, True


def representative_q_direction(q48: np.ndarray, mode: str) -> tuple[np.ndarray, bool, dict[str, float | None]]:
    q = np.asarray(q48, dtype=np.float64).reshape(int(q48.shape[0]), 48)
    norms = np.linalg.norm(q, axis=1)
    valid = norms > 1.0e-12
    if not np.any(valid):
        return np.zeros(48, dtype=np.float64), False, {
            "case_internal_q_direction_cos_min": None,
            "case_internal_q_direction_cos_mean": None,
            "case_internal_q_direction_abs_cos_min": None,
            "case_internal_q_direction_abs_cos_mean": None,
        }
    dirs = q[valid] / norms[valid, None]
    if str(mode).lower() == "max-norm":
        rep = dirs[int(np.argmax(norms[valid]))]
    else:
        rep, ok = normalize(np.mean(dirs, axis=0))
        if not ok:
            rep = dirs[int(np.argmax(norms[valid]))]
    cos = dirs @ dirs.T
    tri = cos[np.triu_indices_from(cos, k=1)]
    if tri.size == 0:
        signed_min = signed_mean = abs_min = abs_mean = 1.0
    else:
        signed_min = float(np.min(tri))
        signed_mean = float(np.mean(tri))
        abs_tri = np.abs(tri)
        abs_min = float(np.min(abs_tri))
        abs_mean = float(np.mean(abs_tri))
    return rep.astype(np.float64), True, {
        "case_internal_q_direction_cos_min": signed_min,
        "case_internal_q_direction_cos_mean": signed_mean,
        "case_internal_q_direction_abs_cos_min": abs_min,
        "case_internal_q_direction_abs_cos_mean": abs_mean,
    }


def case_ids_from_compact(z: np.lib.npyio.NpzFile, path: Path) -> list[int]:
    ids: list[int] = []
    if "case_id" in z.files:
        for item in np.asarray(z["case_id"]).reshape(-1):
            try:
                val = int(item)
            except Exception:
                continue
            if val >= 0:
                ids.append(val)
    if "sample_paths" in z.files:
        for item in np.asarray(z["sample_paths"]).reshape(-1):
            text = item.decode("utf-8") if isinstance(item, bytes) else str(item)
            val = case_id_from_path(text)
            if val >= 0:
                ids.append(val)
    val = case_id_from_path(str(path))
    if val >= 0:
        ids.append(val)
    return sorted(set(ids))


def compact_row(path: Path, compact_index: int, *, q_direction_mode: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with np.load(str(path), allow_pickle=True) as z:
        if "q48_raw" not in z.files:
            raise KeyError(f"{path}: missing q48_raw")
        le_key = "LE128_base" if "LE128_base" in z.files else "le"
        b_key = "B_LE128_forward" if "B_LE128_forward" in z.files else "b"
        if le_key not in z.files:
            raise KeyError(f"{path}: missing LE128_base/le")
        if b_key not in z.files:
            raise KeyError(f"{path}: missing B_LE128_forward/b")

        q48 = np.asarray(z["q48_raw"], dtype=np.float64)
        le = np.asarray(z[le_key], dtype=np.float64)
        b = np.asarray(z[b_key], dtype=np.float64)
        if q48.ndim != 2 or q48.shape[1] != 48:
            raise ValueError(f"{path}: expected q48_raw [N,48], got {q48.shape}")
        if le.shape[0] != q48.shape[0] or b.shape[0] != q48.shape[0]:
            raise ValueError(f"{path}: q/LE/B frame counts do not match")

        q_norm = np.linalg.norm(q48, axis=1)
        le_rms = rms_per_frame(le)
        b_rms = rms_per_frame(b)
        q_dir, q_dir_valid, internal = representative_q_direction(q48, q_direction_mode)
        case_ids = case_ids_from_compact(z, path)
        case_id = int(case_ids[0]) if len(case_ids) == 1 else -1
        strain_field = canonical_strain_field(scalar_text(z, "strain_field", "unknown"))
        b_label_strain_field = canonical_strain_field(scalar_text(z, "B_label_strain_field", strain_field))

        row: dict[str, Any] = {
            "compact_index": int(compact_index),
            "case_id": case_id,
            "case_ids": case_ids,
            "compact_path": str(path),
            "frame_count": int(q48.shape[0]),
            "strain_field": strain_field,
            "B_label_strain_field": b_label_strain_field,
            "strain_label_key": scalar_text(z, "strain_label_key", le_key),
            "B_label_key": scalar_text(z, "B_label_key", b_key),
            "training_ready_sobolev": scalar_bool(z, "training_ready_sobolev", b_key in z.files),
            "q_norm_min": float(np.min(q_norm)),
            "q_norm_max": float(np.max(q_norm)),
            "q_norm_mean": float(np.mean(q_norm)),
            "q_direction_valid": bool(q_dir_valid),
            "q_direction_representative": q_dir.tolist(),
            "q_direction_json": json.dumps(q_dir.tolist(), separators=(",", ":")),
            "LE_rms_min": float(np.min(le_rms)),
            "LE_rms_max": float(np.max(le_rms)),
            "LE_rms_mean": float(np.mean(le_rms)),
            "B_rms_mean": float(np.mean(b_rms)),
            "merge_q48_max_abs_diff": finite_or_none(scalar_float(z, "merge_q48_max_abs_diff")),
            "merge_LE128_base_max_abs_diff": finite_or_none(scalar_float(z, "merge_LE128_base_max_abs_diff")),
            "merge_ip_keys_match": scalar_bool(z, "merge_ip_keys_match", False),
            "merge_strain_field_match": scalar_bool(z, "merge_strain_field_match", False),
            "audit_ref_ip_xyz_vs_abaqus_coord_max_abs": finite_or_none(
                scalar_float(z, "audit_ref_ip_xyz_vs_abaqus_coord_max_abs")
            ),
            "audit_detJ_vs_IVOL_max_abs": finite_or_none(scalar_float(z, "audit_detJ_vs_IVOL_max_abs")),
        }
        row.update(internal)

    frames = [
        {
            "compact_index": int(compact_index),
            "case_id": case_id,
            "frame_index": int(i),
            "q_norm": float(q_norm[i]),
            "LE_rms": float(le_rms[i]),
            "B_rms": float(b_rms[i]),
        }
        for i in range(q48.shape[0])
    ]
    return row, frames


def cosine_matrix(rows: list[dict[str, Any]], *, absolute: bool = False) -> np.ndarray:
    n = len(rows)
    out = np.full((n, n), np.nan, dtype=np.float64)
    for i, lhs in enumerate(rows):
        if not lhs.get("q_direction_valid", False):
            continue
        vi = np.asarray(lhs["q_direction_representative"], dtype=np.float64)
        for j, rhs in enumerate(rows):
            if not rhs.get("q_direction_valid", False):
                continue
            vj = np.asarray(rhs["q_direction_representative"], dtype=np.float64)
            val = float(np.dot(vi, vj))
            out[i, j] = abs(val) if absolute else val
    return out


def assign_direction_clusters(rows: list[dict[str, Any]], threshold: float) -> None:
    reps: list[np.ndarray] = []
    clusters: list[list[int]] = []
    for idx, row in enumerate(rows):
        if not row.get("q_direction_valid", False):
            row["q_direction_cluster_id"] = -1
            continue
        vec = np.asarray(row["q_direction_representative"], dtype=np.float64)
        if not reps:
            reps.append(vec.copy())
            clusters.append([idx])
            row["q_direction_cluster_id"] = 0
            continue
        sims = [abs(float(np.dot(vec, rep))) for rep in reps]
        best = int(np.argmax(sims))
        if sims[best] >= float(threshold):
            row["q_direction_cluster_id"] = best
            clusters[best].append(idx)
            aligned = vec if float(np.dot(vec, reps[best])) >= 0.0 else -vec
            new_rep, ok = normalize(reps[best] + aligned)
            if ok:
                reps[best] = new_rep
        else:
            reps.append(vec.copy())
            clusters.append([idx])
            row["q_direction_cluster_id"] = len(reps) - 1


def add_existing_direction_fields(rows: list[dict[str, Any]]) -> None:
    for i, row in enumerate(rows):
        row["q_direction_cosine_to_existing_max_abs"] = None
        row["q_direction_nearest_existing_case_id"] = None
        if i == 0 or not row.get("q_direction_valid", False):
            continue
        vec = np.asarray(row["q_direction_representative"], dtype=np.float64)
        best_val = -1.0
        best_case: int | None = None
        for prev in rows[:i]:
            if not prev.get("q_direction_valid", False):
                continue
            val = abs(float(np.dot(vec, np.asarray(prev["q_direction_representative"], dtype=np.float64))))
            if val > best_val:
                best_val = val
                best_case = int(prev["case_id"])
        if best_case is not None:
            row["q_direction_cosine_to_existing_max_abs"] = best_val
            row["q_direction_nearest_existing_case_id"] = best_case


def strict_failures(
    rows: list[dict[str, Any]],
    *,
    audit_ref_coord_tol: float,
    audit_detj_ivol_tol: float,
    merge_q_tol: float,
    merge_le_tol: float,
) -> list[str]:
    failures: list[str] = []
    for row in rows:
        label = f"case {row['case_id']} compact {row['compact_index']}"
        if not bool(row["training_ready_sobolev"]):
            failures.append(f"{label}: training_ready_sobolev is false")
        if canonical_strain_field(row["strain_field"]) == "unknown":
            failures.append(f"{label}: strain_field is unknown")
        if canonical_strain_field(row["strain_field"]) != canonical_strain_field(row["B_label_strain_field"]):
            failures.append(f"{label}: strain_field and B_label_strain_field differ")
        if row["merge_q48_max_abs_diff"] is None or float(row["merge_q48_max_abs_diff"]) > float(merge_q_tol):
            failures.append(f"{label}: merge_q48_max_abs_diff exceeds tolerance or is missing")
        if row["merge_LE128_base_max_abs_diff"] is None or float(row["merge_LE128_base_max_abs_diff"]) > float(merge_le_tol):
            failures.append(f"{label}: merge_LE128_base_max_abs_diff exceeds tolerance or is missing")
        if not bool(row["merge_ip_keys_match"]):
            failures.append(f"{label}: merge_ip_keys_match is false")
        if not bool(row["merge_strain_field_match"]):
            failures.append(f"{label}: merge_strain_field_match is false")
        if row["audit_ref_ip_xyz_vs_abaqus_coord_max_abs"] is None or float(row["audit_ref_ip_xyz_vs_abaqus_coord_max_abs"]) > float(audit_ref_coord_tol):
            failures.append(f"{label}: reference COORD audit exceeds tolerance or is missing")
        if row["audit_detJ_vs_IVOL_max_abs"] is None or float(row["audit_detJ_vs_IVOL_max_abs"]) > float(audit_detj_ivol_tol):
            failures.append(f"{label}: detJ vs IVOL audit exceeds tolerance or is missing")
    return failures


def pairwise_abs_stats(abs_cos: np.ndarray) -> dict[str, float | None]:
    if abs_cos.shape[0] <= 1:
        return {"pairwise_abs_cos_min": None, "pairwise_abs_cos_median": None, "pairwise_abs_cos_max": None}
    tri = abs_cos[np.triu_indices(abs_cos.shape[0], k=1)]
    tri = tri[np.isfinite(tri)]
    if tri.size == 0:
        return {"pairwise_abs_cos_min": None, "pairwise_abs_cos_median": None, "pairwise_abs_cos_max": None}
    return {
        "pairwise_abs_cos_min": float(np.min(tri)),
        "pairwise_abs_cos_median": float(np.median(tri)),
        "pairwise_abs_cos_max": float(np.max(tri)),
    }


def case_label(row: dict[str, Any]) -> str:
    cid = int(row["case_id"])
    return f"case{cid:03d}" if cid >= 0 else f"compact{int(row['compact_index']):03d}"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "compact_index",
        "case_id",
        "case_ids",
        "compact_path",
        "frame_count",
        "strain_field",
        "B_label_strain_field",
        "strain_label_key",
        "B_label_key",
        "training_ready_sobolev",
        "q_norm_min",
        "q_norm_max",
        "q_norm_mean",
        "q_direction_valid",
        "q_direction_json",
        "q_direction_cosine_to_existing_max_abs",
        "q_direction_nearest_existing_case_id",
        "q_direction_cluster_id",
        "case_internal_q_direction_cos_min",
        "case_internal_q_direction_cos_mean",
        "case_internal_q_direction_abs_cos_min",
        "case_internal_q_direction_abs_cos_mean",
        "LE_rms_min",
        "LE_rms_max",
        "LE_rms_mean",
        "B_rms_mean",
        "merge_q48_max_abs_diff",
        "merge_LE128_base_max_abs_diff",
        "merge_ip_keys_match",
        "merge_strain_field_match",
        "audit_ref_ip_xyz_vs_abaqus_coord_max_abs",
        "audit_detJ_vs_IVOL_max_abs",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_frame_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["compact_index", "case_id", "frame_index", "q_norm", "LE_rms", "B_rms"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(path: Path, rows: list[dict[str, Any]], matrix: np.ndarray) -> None:
    labels = [case_label(row) for row in rows]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id"] + labels)
        for label, vals in zip(labels, matrix):
            writer.writerow([label] + ["" if not math.isfinite(float(v)) else f"{float(v):.10g}" for v in vals])


def coverage_to_train(row: dict[str, Any], train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not row.get("q_direction_valid", False):
        return {"q_direction_cosine_to_train_max_abs": None, "q_direction_cosine_to_train_min_abs": None}
    vec = np.asarray(row["q_direction_representative"], dtype=np.float64)
    vals = [
        abs(float(np.dot(vec, np.asarray(train["q_direction_representative"], dtype=np.float64))))
        for train in train_rows
        if train.get("q_direction_valid", False)
    ]
    if not vals:
        return {"q_direction_cosine_to_train_max_abs": None, "q_direction_cosine_to_train_min_abs": None}
    return {
        "q_direction_cosine_to_train_max_abs": float(np.max(vals)),
        "q_direction_cosine_to_train_min_abs": float(np.min(vals)),
    }


def build_train_val_plan(rows: list[dict[str, Any]], val_fraction: float, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    n = len(rows)
    if n < 2:
        train_idx = list(range(n))
        val_idx: list[int] = []
    else:
        order = rng.permutation(n)
        val_count = max(1, int(math.ceil(float(val_fraction) * n)))
        val_count = min(val_count, n - 1)
        val_idx = sorted(int(i) for i in order[:val_count])
        train_idx = sorted(int(i) for i in order[val_count:])
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    train_le_min = min((float(r["LE_rms_min"]) for r in train_rows), default=math.nan)
    train_le_max = max((float(r["LE_rms_max"]) for r in train_rows), default=math.nan)
    val_coverage = []
    for row in val_rows:
        item = {
            "val_case_id": int(row["case_id"]),
            "val_compact_index": int(row["compact_index"]),
            "val_LE_rms_mean": float(row["LE_rms_mean"]),
            "val_LE_rms_inside_train_range": bool(train_le_min <= float(row["LE_rms_mean"]) <= train_le_max)
            if math.isfinite(train_le_min) and math.isfinite(train_le_max)
            else None,
        }
        item.update(coverage_to_train(row, train_rows))
        val_coverage.append(item)
    return {
        "training_status": "not_run_by_coverage_audit",
        "split_mode": "case",
        "allow_overlap_val": False,
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "train_case_count": len(train_rows),
        "val_case_count": len(val_rows),
        "train_cases": [int(r["case_id"]) for r in train_rows],
        "val_cases": [int(r["case_id"]) for r in val_rows],
        "train_LE_rms_min": finite_or_none(train_le_min),
        "train_LE_rms_max": finite_or_none(train_le_max),
        "val_coverage": val_coverage,
        "metrics": {
            "val_AD_B_rel": None,
            "val_AD_B_cos": None,
            "val_phys_rand_dir_B_rel": None,
            "val_LE_rel": None,
        },
    }


def build_loo_plan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases = []
    for i, row in enumerate(rows):
        train_rows = [r for j, r in enumerate(rows) if j != i]
        train_le_min = min((float(r["LE_rms_min"]) for r in train_rows), default=math.nan)
        train_le_max = max((float(r["LE_rms_max"]) for r in train_rows), default=math.nan)
        item = {
            "training_status": "not_run_by_coverage_audit",
            "val_case_id": int(row["case_id"]),
            "val_compact_index": int(row["compact_index"]),
            "train_cases": [int(r["case_id"]) for r in train_rows],
            "val_LE_rms_mean": float(row["LE_rms_mean"]),
            "train_LE_rms_min": finite_or_none(train_le_min),
            "train_LE_rms_max": finite_or_none(train_le_max),
            "val_LE_rms_inside_train_range": bool(train_le_min <= float(row["LE_rms_mean"]) <= train_le_max)
            if math.isfinite(train_le_min) and math.isfinite(train_le_max)
            else None,
            "metrics": {
                "val_AD_B_rel": None,
                "val_AD_B_cos": None,
                "val_phys_rand_dir_B_rel": None,
                "val_LE_rel": None,
            },
        }
        item.update(coverage_to_train(row, train_rows))
        cases.append(item)
    return {
        "training_status": "not_run_by_coverage_audit",
        "split_mode": "leave-one-case-out",
        "case_count": len(rows),
        "cases": cases,
    }


def run_coverage_audit(
    compact_paths: list[str | Path],
    out_root: str | Path,
    *,
    q_direction_mode: str = "mean",
    cluster_abs_cos_threshold: float = 0.95,
    val_fraction: float = 0.2,
    seed: int = 123,
    audit_ref_coord_tol: float = 1.0e-6,
    audit_detj_ivol_tol: float = 1.0e-8,
    merge_q_tol: float = 1.0e-8,
    merge_le_tol: float = 1.0e-8,
    strict_v1_2: bool = False,
) -> dict[str, Any]:
    paths = [Path(p).resolve() for p in compact_paths]
    if not paths:
        raise ValueError("at least one compact is required")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    out = Path(out_root).resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for i, path in enumerate(paths):
        row, frames = compact_row(path, i, q_direction_mode=q_direction_mode)
        rows.append(row)
        frame_rows.extend(frames)

    assign_direction_clusters(rows, float(cluster_abs_cos_threshold))
    add_existing_direction_fields(rows)
    cos = cosine_matrix(rows, absolute=False)
    abs_cos = cosine_matrix(rows, absolute=True)
    failures = strict_failures(
        rows,
        audit_ref_coord_tol=audit_ref_coord_tol,
        audit_detj_ivol_tol=audit_detj_ivol_tol,
        merge_q_tol=merge_q_tol,
        merge_le_tol=merge_le_tol,
    )
    cluster_ids = sorted({int(row["q_direction_cluster_id"]) for row in rows if int(row["q_direction_cluster_id"]) >= 0})
    summary = {
        "audit_name": "query_point_v1_2_data_coverage_audit",
        "compact_count": len(rows),
        "case_count": len({int(row["case_id"]) for row in rows if int(row["case_id"]) >= 0}),
        "frame_count": int(sum(int(row["frame_count"]) for row in rows)),
        "q_direction_mode": q_direction_mode,
        "cluster_abs_cos_threshold": float(cluster_abs_cos_threshold),
        "q_direction_cluster_count": len(cluster_ids),
        "q_direction_cluster_ids": cluster_ids,
        "strict_v1_2_pass": len(failures) == 0,
        "strict_v1_2_failures": failures,
        "target_outputs": [
            "compact_manifest.csv",
            "compact_manifest.json",
            "q_direction_cosine_matrix.csv",
            "q_direction_abs_cosine_matrix.csv",
            "le_rms_distribution.csv",
            "audit_summary.json",
            "train_80_20_summary.json",
            "loo_summary.json",
        ],
    }
    summary.update(pairwise_abs_stats(abs_cos))

    write_manifest_csv(out / "compact_manifest.csv", rows)
    write_json(out / "compact_manifest.json", {"compacts": rows})
    write_matrix_csv(out / "q_direction_cosine_matrix.csv", rows, cos)
    write_matrix_csv(out / "q_direction_abs_cosine_matrix.csv", rows, abs_cos)
    write_frame_csv(out / "le_rms_distribution.csv", frame_rows)
    train_plan = build_train_val_plan(rows, val_fraction=val_fraction, seed=seed)
    loo_plan = build_loo_plan(rows)
    write_json(out / "train_80_20_summary.json", train_plan)
    write_json(out / "loo_summary.json", loo_plan)
    write_json(out / "audit_summary.json", summary)

    result = {
        "out_root": str(out),
        "audit_summary": summary,
        "train_80_20_summary": train_plan,
        "loo_summary": loo_plan,
    }
    if strict_v1_2 and failures:
        raise RuntimeError("strict v1.2 data coverage audit failed: " + "; ".join(failures))
    return result


def collect_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for item in args.compact or []:
        paths.append(Path(item))
    if args.compact_list:
        paths.extend(Path(item) for item in read_list(Path(args.compact_list)))
    for pattern in args.compact_glob or []:
        matches = sorted(Path(item) for item in glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"compact glob matched nothing: {pattern}")
        paths.extend(matches)
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path.resolve())
    return unique


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compact", action="append", default=[], help="Input complete compact NPZ. May be repeated.")
    p.add_argument("--compact-list", default="", help="Text file containing one compact path per line.")
    p.add_argument("--compact-glob", action="append", default=[], help="Glob for compact NPZ files. May be repeated.")
    p.add_argument("--out-root", required=True)
    p.add_argument("--q-direction-mode", choices=("mean", "max-norm"), default="mean")
    p.add_argument("--cluster-abs-cos-threshold", type=float, default=0.95)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--audit-ref-coord-tol", type=float, default=1.0e-6)
    p.add_argument("--audit-detj-ivol-tol", type=float, default=1.0e-8)
    p.add_argument("--merge-q-tol", type=float, default=1.0e-8)
    p.add_argument("--merge-le-tol", type=float, default=1.0e-8)
    p.add_argument("--strict-v1-2", action="store_true")
    args = p.parse_args()

    result = run_coverage_audit(
        collect_paths(args),
        args.out_root,
        q_direction_mode=args.q_direction_mode,
        cluster_abs_cos_threshold=args.cluster_abs_cos_threshold,
        val_fraction=args.val_fraction,
        seed=args.seed,
        audit_ref_coord_tol=args.audit_ref_coord_tol,
        audit_detj_ivol_tol=args.audit_detj_ivol_tol,
        merge_q_tol=args.merge_q_tol,
        merge_le_tol=args.merge_le_tol,
        strict_v1_2=bool(args.strict_v1_2),
    )
    print(json.dumps(result["audit_summary"], indent=2, ensure_ascii=False, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
