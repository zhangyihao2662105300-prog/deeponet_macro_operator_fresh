#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select legacy TRUE176 load directions worth regenerating for v1.2.

Legacy TRUE176 compacts are not v1.2 training-ready data.  This script only
uses their q/LE/sample-path metadata as a candidate load-direction library and
does not train a model or add them to the formal query-point pool.  Selected
q directions should be reapplied as fresh boundary-displacement controls and
then exported through the current strict complete-compact route.
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


def case_id_from_path(path: str) -> int:
    match = re.search(r"case(\d+)", str(path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def normalize(vec: np.ndarray, eps: float = 1.0e-12) -> tuple[np.ndarray, bool]:
    vals = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vals))
    if not math.isfinite(norm) or norm <= eps:
        return np.zeros_like(vals, dtype=np.float64), False
    return vals / norm, True


def rms_per_frame(arr: np.ndarray) -> np.ndarray:
    vals = np.asarray(arr)
    flat = vals.reshape(int(vals.shape[0]), -1)
    sq_sum = np.einsum("ij,ij->i", flat, flat, dtype=np.float64)
    return np.sqrt(sq_sum / float(flat.shape[1]))


def representative_q_direction(q48: np.ndarray, mode: str) -> tuple[np.ndarray, bool, dict[str, float | None]]:
    q = np.asarray(q48, dtype=np.float64).reshape(int(q48.shape[0]), 48)
    norms = np.linalg.norm(q, axis=1)
    valid = norms > 1.0e-12
    if not np.any(valid):
        return np.zeros(48, dtype=np.float64), False, {
            "legacy_internal_q_direction_abs_cos_min": None,
            "legacy_internal_q_direction_abs_cos_mean": None,
        }
    dirs = q[valid] / norms[valid, None]
    if str(mode).lower() == "max-norm":
        rep = dirs[int(np.argmax(norms[valid]))]
    else:
        rep, ok = normalize(np.mean(dirs, axis=0))
        if not ok:
            rep = dirs[int(np.argmax(norms[valid]))]
    cos = dirs @ dirs.T
    tri = np.abs(cos[np.triu_indices_from(cos, k=1)])
    if tri.size == 0:
        abs_min = abs_mean = 1.0
    else:
        abs_min = float(np.min(tri))
        abs_mean = float(np.mean(tri))
    return rep.astype(np.float64), True, {
        "legacy_internal_q_direction_abs_cos_min": abs_min,
        "legacy_internal_q_direction_abs_cos_mean": abs_mean,
    }


def load_current_cases(current_manifest: Path) -> list[dict[str, Any]]:
    payload = json.loads(current_manifest.read_text(encoding="utf-8"))
    rows = payload.get("compacts", payload if isinstance(payload, list) else [])
    out: list[dict[str, Any]] = []
    for row in rows:
        if not bool(row.get("q_direction_valid", True)):
            continue
        vec = np.asarray(row.get("q_direction_representative"), dtype=np.float64)
        if vec.shape != (48,):
            text = row.get("q_direction_json", "")
            vec = np.asarray(json.loads(text), dtype=np.float64) if text else np.zeros(48, dtype=np.float64)
        vec, ok = normalize(vec)
        if not ok:
            continue
        out.append(
            {
                "case_id": int(row.get("case_id", -1)),
                "q_direction": vec,
                "LE_rms_mean": float(row.get("LE_rms_mean", math.nan)),
                "cluster_id": int(row.get("q_direction_cluster_id", -1)),
            }
        )
    if not out:
        raise ValueError(f"{current_manifest}: no usable current q directions")
    return out


def infer_direction_type(text: str) -> str:
    key = str(text).lower()
    labels: list[str] = []
    if "axial" in key:
        labels.append("axial")
    if "shear" in key:
        labels.append("shear")
    if "torque" in key or "torsion" in key:
        labels.append("torsion")
    if "moment" in key or "bend" in key or "bending" in key:
        labels.append("bending")
    if len(labels) > 1:
        return "mixed:" + "+".join(labels)
    if labels:
        return labels[0]
    return "unknown"


def sample_ranges(n_frames: int, sample_paths: np.ndarray | None) -> list[tuple[int, int, int, str]]:
    if sample_paths is None or int(sample_paths.size) == 0:
        return [(0, 0, int(n_frames), "")]
    paths = np.asarray(sample_paths).reshape(-1)
    sample_count = int(paths.size)
    if sample_count > 0 and int(n_frames) % sample_count == 0:
        frames_per_sample = int(n_frames) // sample_count
        out = []
        for i, item in enumerate(paths):
            text = item.decode("utf-8") if isinstance(item, bytes) else str(item)
            start = i * frames_per_sample
            out.append((i, start, start + frames_per_sample, text))
        return out
    return [(0, 0, int(n_frames), "")]


def finite_or_none(value: float) -> float | None:
    val = float(value)
    return val if math.isfinite(val) else None


def collect_legacy_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for item in args.legacy_compact or []:
        paths.append(Path(item))
    if args.legacy_compact_list:
        paths.extend(Path(item) for item in read_list(Path(args.legacy_compact_list)))
    for pattern in args.legacy_compact_glob or []:
        matches = sorted(Path(item) for item in glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"legacy compact glob matched nothing: {pattern}")
        paths.extend(matches)
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path.resolve())
    return unique


def nearest_odb_hint(sample_path: str) -> dict[str, Any]:
    text = str(sample_path)
    path = Path(text) if text else Path()
    sample_root = path
    if path.suffix.lower() in {".npz", ".json", ".odb"}:
        sample_root = path.parent
    hints = {
        "sample_path_exists": bool(text) and path.exists(),
        "sample_root_hint": str(sample_root) if text else "",
        "sample_root_exists": bool(text) and sample_root.exists(),
        "base_odb_hint": "",
        "base_odb_exists": False,
        "perturb_dir_hint": "",
        "perturb_dir_exists": False,
    }
    if not text:
        return hints
    if path.suffix.lower() == ".odb":
        hints["base_odb_hint"] = str(path)
        hints["base_odb_exists"] = path.exists()
        sample_root = path.parent.parent if path.parent.name.lower() == "base" else path.parent
        hints["sample_root_hint"] = str(sample_root)
        hints["sample_root_exists"] = sample_root.exists()
    else:
        base = sample_root / "base"
        odbs = sorted(base.glob("*.odb")) if base.exists() else []
        if odbs:
            hints["base_odb_hint"] = str(odbs[0])
            hints["base_odb_exists"] = True
    perturb = sample_root / "perturb"
    hints["perturb_dir_hint"] = str(perturb)
    hints["perturb_dir_exists"] = perturb.exists()
    return hints


def analyze_legacy_compact(
    path: Path,
    source_index: int,
    current_cases: list[dict[str, Any]],
    *,
    q_direction_mode: str,
    compute_b_rms: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with np.load(str(path), allow_pickle=True) as z:
        if "q48_raw" not in z.files:
            return rows
        le_key = "LE128_base" if "LE128_base" in z.files else "le"
        if le_key not in z.files:
            return rows
        q48 = np.asarray(z["q48_raw"], dtype=np.float64)
        le = np.asarray(z[le_key], dtype=np.float64)
        if q48.ndim != 2 or q48.shape[1] != 48 or le.shape[0] != q48.shape[0]:
            return rows
        b_rms_all = None
        b_key = "B_LE128_forward" if "B_LE128_forward" in z.files else ("b" if "b" in z.files else "")
        if compute_b_rms and b_key:
            b_arr = np.asarray(z[b_key], dtype=np.float64)
            if b_arr.shape[0] == q48.shape[0]:
                b_rms_all = rms_per_frame(b_arr)
        le_rms_all = rms_per_frame(le)
        samples = sample_ranges(q48.shape[0], np.asarray(z["sample_paths"]) if "sample_paths" in z.files else None)
        case_arr = np.asarray(z["case_id"]).reshape(-1) if "case_id" in z.files else None
        for sample_index, start, end, sample_path in samples:
            q_chunk = q48[start:end]
            le_chunk = le_rms_all[start:end]
            if q_chunk.size == 0:
                continue
            q_dir, valid, internal = representative_q_direction(q_chunk, q_direction_mode)
            if not valid:
                continue
            cos_vals = [abs(float(np.dot(q_dir, cur["q_direction"]))) for cur in current_cases]
            nearest_i = int(np.argmax(cos_vals)) if cos_vals else -1
            inferred_case = case_id_from_path(sample_path)
            if inferred_case < 0 and case_arr is not None and case_arr.size == q48.shape[0]:
                vals = sorted({int(v) for v in case_arr[start:end] if int(v) >= 0})
                inferred_case = vals[0] if len(vals) == 1 else -1
            q_norm = np.linalg.norm(q_chunk, axis=1)
            b_rms_mean = None
            if b_rms_all is not None:
                b_rms_mean = float(np.mean(b_rms_all[start:end]))
            row = {
                "legacy_index": len(rows),
                "source_index": int(source_index),
                "source_compact_path": str(path),
                "sample_index": int(sample_index),
                "sample_path": sample_path,
                "inferred_case_id": int(inferred_case),
                "sample_id": Path(sample_path).name if sample_path else f"source{source_index}_sample{sample_index}",
                "frame_start": int(start),
                "frame_end": int(end),
                "frame_count": int(end - start),
                "q_norm_min": float(np.min(q_norm)),
                "q_norm_max": float(np.max(q_norm)),
                "q_norm_mean": float(np.mean(q_norm)),
                "q_direction_valid": bool(valid),
                "q_direction_representative": q_dir.tolist(),
                "q_direction_json": json.dumps(q_dir.tolist(), separators=(",", ":")),
                "LE_rms_min": float(np.min(le_chunk)),
                "LE_rms_max": float(np.max(le_chunk)),
                "LE_rms_mean": float(np.mean(le_chunk)),
                "B_rms_mean": b_rms_mean,
                "B_rms_status": "computed" if b_rms_mean is not None else ("skipped_to_avoid_large_B_load" if b_key else "missing_B"),
                "max_abs_cos_to_current": float(np.max(cos_vals)) if cos_vals else None,
                "nearest_existing_case": int(current_cases[nearest_i]["case_id"]) if nearest_i >= 0 else None,
                "nearest_existing_cluster": int(current_cases[nearest_i]["cluster_id"]) if nearest_i >= 0 else None,
                "direction_type": infer_direction_type(sample_path),
            }
            row.update(internal)
            row.update(nearest_odb_hint(sample_path))
            rows.append(row)
    return rows


def assign_clusters(rows: list[dict[str, Any]], threshold: float) -> None:
    reps: list[np.ndarray] = []
    for row in rows:
        vec = np.asarray(row["q_direction_representative"], dtype=np.float64)
        if not reps:
            reps.append(vec.copy())
            row["legacy_q_direction_cluster_id"] = 0
            row["legacy_cluster_nearest_abs_cos"] = None
            continue
        sims = [abs(float(np.dot(vec, rep))) for rep in reps]
        best = int(np.argmax(sims))
        row["legacy_cluster_nearest_abs_cos"] = float(sims[best])
        if sims[best] >= float(threshold):
            row["legacy_q_direction_cluster_id"] = best
            aligned = vec if float(np.dot(vec, reps[best])) >= 0.0 else -vec
            new_rep, ok = normalize(reps[best] + aligned)
            if ok:
                reps[best] = new_rep
        else:
            reps.append(vec.copy())
            row["legacy_q_direction_cluster_id"] = len(reps) - 1


def mark_recommendations(rows: list[dict[str, Any]], prefer_threshold: float, strong_threshold: float) -> None:
    le_vals = np.asarray([float(r["LE_rms_mean"]) for r in rows], dtype=np.float64)
    q1, q2 = np.quantile(le_vals, [1.0 / 3.0, 2.0 / 3.0]) if le_vals.size >= 3 else (math.nan, math.nan)
    for row in rows:
        le = float(row["LE_rms_mean"])
        if math.isfinite(q1) and le <= q1:
            row["LE_rms_bin"] = "low"
        elif math.isfinite(q2) and le >= q2:
            row["LE_rms_bin"] = "high"
        else:
            row["LE_rms_bin"] = "medium"
        cos = row.get("max_abs_cos_to_current")
        if cos is None:
            row["recommended_for_reexport"] = False
            row["recommendation_level"] = "skip"
            row["recommendation_reason"] = "missing current-cosine comparison"
        elif float(cos) < float(strong_threshold):
            row["recommended_for_reexport"] = True
            row["recommendation_level"] = "strong"
            row["recommendation_reason"] = f"newer direction: max_abs_cos_to_current={float(cos):.4f} < {strong_threshold}"
        elif float(cos) < float(prefer_threshold):
            row["recommended_for_reexport"] = True
            row["recommendation_level"] = "candidate"
            row["recommendation_reason"] = f"useful direction: max_abs_cos_to_current={float(cos):.4f} < {prefer_threshold}"
        else:
            row["recommended_for_reexport"] = False
            row["recommendation_level"] = "repeat"
            row["recommendation_reason"] = f"near existing direction: max_abs_cos_to_current={float(cos):.4f}"


def select_top(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    recommended = [r for r in rows if bool(r.get("recommended_for_reexport"))]
    bins_order = {"medium": 0, "low": 1, "high": 2}
    recommended.sort(
        key=lambda r: (
            float(r["max_abs_cos_to_current"]),
            bins_order.get(str(r.get("LE_rms_bin")), 9),
            int(r.get("legacy_q_direction_cluster_id", 999999)),
        )
    )
    selected: list[dict[str, Any]] = []
    used_clusters: set[int] = set()
    for row in recommended:
        cluster = int(row.get("legacy_q_direction_cluster_id", -1))
        if cluster in used_clusters:
            continue
        selected.append(row)
        used_clusters.add(cluster)
        if len(selected) >= int(count):
            return selected
    for row in recommended:
        if row not in selected:
            selected.append(row)
        if len(selected) >= int(count):
            break
    return selected


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "legacy_index",
        "recommended_new_case_id",
        "recommended_for_reexport",
        "recommendation_level",
        "recommendation_reason",
        "source_index",
        "source_compact_path",
        "sample_index",
        "sample_id",
        "sample_path",
        "inferred_case_id",
        "direction_type",
        "frame_start",
        "frame_end",
        "frame_count",
        "q_norm_min",
        "q_norm_max",
        "q_norm_mean",
        "LE_rms_min",
        "LE_rms_max",
        "LE_rms_mean",
        "LE_rms_bin",
        "B_rms_mean",
        "B_rms_status",
        "max_abs_cos_to_current",
        "nearest_existing_case",
        "nearest_existing_cluster",
        "legacy_q_direction_cluster_id",
        "legacy_cluster_nearest_abs_cos",
        "legacy_internal_q_direction_abs_cos_min",
        "legacy_internal_q_direction_abs_cos_mean",
        "base_odb_hint",
        "base_odb_exists",
        "perturb_dir_hint",
        "perturb_dir_exists",
        "sample_root_hint",
        "sample_root_exists",
        "q_direction_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_cosine_csv(path: Path, rows: list[dict[str, Any]], current_cases: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["legacy_index", "sample_id", "inferred_case_id"] + [f"case{c['case_id']:03d}" for c in current_cases])
        for row in rows:
            vec = np.asarray(row["q_direction_representative"], dtype=np.float64)
            vals = [abs(float(np.dot(vec, cur["q_direction"]))) for cur in current_cases]
            writer.writerow([row["legacy_index"], row["sample_id"], row["inferred_case_id"]] + [f"{v:.10g}" for v in vals])


def write_reexport_plan(path: Path, selected: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# v1.2 Legacy TRUE176 Re-Export Plan",
        "",
        "旧 176 compact 不能直接进入 v1.2 query-point/Abaqus 正式训练池；这里仅把它们作为候选 q/load direction 来源。",
        "",
        "## Summary",
        "",
        f"- legacy candidates scanned: `{summary['legacy_candidate_count']}`",
        f"- recommended candidates: `{summary['recommended_candidate_count']}`",
        f"- legacy direction clusters: `{summary['legacy_q_direction_cluster_count']}`",
        f"- strong threshold: `max_abs_cos_to_current < {summary['strong_cos_threshold']}`",
        f"- candidate threshold: `max_abs_cos_to_current < {summary['prefer_cos_threshold']}`",
        "",
        "## Recommended Candidates",
        "",
    ]
    if not selected:
        lines.append("No candidates met the current cosine thresholds.")
    for row in selected:
        lines.extend(
            [
                f"### {row['recommended_new_case_id']} - legacy sample {row['legacy_index']}",
                "",
                f"- legacy source compact: `{row['source_compact_path']}`",
                f"- legacy sample_path: `{row['sample_path']}`",
                f"- inferred_case_id: `{row['inferred_case_id']}`",
                f"- q_norm_mean: `{float(row['q_norm_mean']):.6g}`",
                f"- LE_rms_mean: `{float(row['LE_rms_mean']):.6g}` ({row['LE_rms_bin']})",
                f"- B_rms_mean: `{row['B_rms_mean']}` ({row['B_rms_status']})",
                f"- max_abs_cos_to_current: `{float(row['max_abs_cos_to_current']):.6g}`",
                f"- nearest_existing_case: `{row['nearest_existing_case']}`",
                f"- legacy_q_direction_cluster_id: `{row['legacy_q_direction_cluster_id']}`",
                f"- predicted direction type: `{row['direction_type']}`",
                f"- base ODB hint: `{row['base_odb_hint']}` exists=`{row['base_odb_exists']}`",
                f"- perturb dir hint: `{row['perturb_dir_hint']}` exists=`{row['perturb_dir_exists']}`",
                f"- reason: {row['recommendation_reason']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundary-Control Regeneration Requirement",
            "",
            "Use the selected legacy direction only as:",
            "",
            "```text",
            "q_dir = q48_raw / ||q48_raw||",
            "q_new = alpha * q_dir",
            "```",
            "",
            "The old LE/B labels are scale hints only, not training labels.  A candidate is not allowed into the formal v1.2 pool until a fresh displacement-controlled Abaqus run is exported as:",
            "",
            "```text",
            "complete_case*_training_ready.npz",
            "strict_v1_2_pass = true",
            "strain_field = LE",
            "B_label_strain_field = LE",
            "```",
            "",
            "Use the current complete compact exporter with an Abaqus ODB and a matching Sobolev B compact; do not train directly on legacy compact files.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_boundary_control_plan(path: Path, selected: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    cases: list[dict[str, Any]] = []
    for row in selected:
        cases.append(
            {
                "recommended_new_case_id": row.get("recommended_new_case_id", ""),
                "legacy_index": int(row["legacy_index"]),
                "legacy_source_compact_path": row["source_compact_path"],
                "legacy_sample_path": row["sample_path"],
                "inferred_case_id": int(row["inferred_case_id"]),
                "direction_type": row["direction_type"],
                "q_direction_48": row["q_direction_representative"],
                "q_norm_seed_mean": float(row["q_norm_mean"]),
                "q_norm_seed_min": float(row["q_norm_min"]),
                "q_norm_seed_max": float(row["q_norm_max"]),
                "LE_rms_hint_mean": float(row["LE_rms_mean"]),
                "B_rms_hint_mean": finite_or_none(row["B_rms_mean"]) if row["B_rms_mean"] is not None else None,
                "max_abs_cos_to_current": float(row["max_abs_cos_to_current"]),
                "nearest_existing_case": row["nearest_existing_case"],
                "boundary_control_rule": "use q_new = alpha * q_direction_48 in a fresh Abaqus displacement-control run; legacy LE/B are not training labels",
                "formal_pool_requirement": summary["formal_pool_rule"],
            }
        )
    payload = {
        "summary": {
            **summary,
            "boundary_control_note": "selected legacy q directions are displacement-control seeds for new Abaqus runs",
        },
        "cases": cases,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")


def run_selection(
    legacy_paths: list[Path],
    current_manifest: Path,
    out_root: Path,
    *,
    q_direction_mode: str,
    cluster_abs_cos_threshold: float,
    prefer_cos_threshold: float,
    strong_cos_threshold: float,
    top_count: int,
    new_case_start: int,
    compute_b_rms: bool,
) -> dict[str, Any]:
    current_cases = load_current_cases(current_manifest)
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for source_index, path in enumerate(legacy_paths):
        try:
            cur_rows = analyze_legacy_compact(
                path,
                source_index,
                current_cases,
                q_direction_mode=q_direction_mode,
                compute_b_rms=compute_b_rms,
            )
            if not cur_rows:
                skipped.append({"path": str(path), "reason": "missing usable q48_raw/LE data"})
            rows.extend(cur_rows)
        except Exception as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
    if not rows:
        raise ValueError("no legacy candidates found")
    for i, row in enumerate(rows):
        row["legacy_index"] = int(i)
    assign_clusters(rows, cluster_abs_cos_threshold)
    mark_recommendations(rows, prefer_cos_threshold, strong_cos_threshold)
    selected = select_top(rows, top_count)
    selected_ids = {int(row["legacy_index"]) for row in selected}
    for i, row in enumerate(selected):
        row["recommended_new_case_id"] = f"case{int(new_case_start) + i:03d}"
    for row in rows:
        if int(row["legacy_index"]) not in selected_ids:
            row["recommended_new_case_id"] = ""
    # Ensure selected rows in the manifest carry the assigned ids.
    id_by_idx = {int(row["legacy_index"]): row["recommended_new_case_id"] for row in selected}
    for row in rows:
        if int(row["legacy_index"]) in id_by_idx:
            row["recommended_new_case_id"] = id_by_idx[int(row["legacy_index"])]
    clusters = sorted({int(row["legacy_q_direction_cluster_id"]) for row in rows})
    summary = {
        "audit_name": "legacy176_reexport_candidate_audit",
        "legacy_compact_count": len(legacy_paths),
        "legacy_candidate_count": len(rows),
        "skipped_compacts": skipped,
        "current_cases": [int(c["case_id"]) for c in current_cases],
        "recommended_candidate_count": len(selected),
        "legacy_q_direction_cluster_count": len(clusters),
        "legacy_q_direction_cluster_ids": clusters,
        "cluster_abs_cos_threshold": float(cluster_abs_cos_threshold),
        "prefer_cos_threshold": float(prefer_cos_threshold),
        "strong_cos_threshold": float(strong_cos_threshold),
        "top_count": int(top_count),
        "compute_b_rms": bool(compute_b_rms),
        "formal_pool_rule": "legacy compacts are boundary-displacement direction sources only; run fresh Abaqus displacement-control cases, export complete_case*_training_ready.npz, and pass strict_v1_2 before training",
    }
    (out_root / "legacy176_candidate_manifest.json").write_text(
        json.dumps({"summary": summary, "candidates": rows}, indent=2, ensure_ascii=False, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    write_manifest_csv(out_root / "legacy176_candidate_manifest.csv", rows)
    write_cosine_csv(out_root / "legacy176_to_current_abs_cosine.csv", rows, current_cases)
    cluster_summary = {
        "summary": summary,
        "clusters": [
            {
                "legacy_q_direction_cluster_id": cluster,
                "candidate_count": int(sum(int(r["legacy_q_direction_cluster_id"]) == cluster for r in rows)),
                "recommended_count": int(
                    sum(int(r["legacy_q_direction_cluster_id"]) == cluster and bool(r["recommended_for_reexport"]) for r in rows)
                ),
                "best_max_abs_cos_to_current": float(
                    min(float(r["max_abs_cos_to_current"]) for r in rows if int(r["legacy_q_direction_cluster_id"]) == cluster)
                ),
            }
            for cluster in clusters
        ],
    }
    (out_root / "legacy176_candidate_cluster_summary.json").write_text(
        json.dumps(cluster_summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    write_reexport_plan(out_root / "legacy176_reexport_plan.md", selected, summary)
    write_boundary_control_plan(out_root / "legacy176_boundary_control_plan.json", selected, summary)
    return {"summary": summary, "selected": selected, "out_root": str(out_root)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-compact", action="append", default=[])
    p.add_argument("--legacy-compact-list", default="")
    p.add_argument("--legacy-compact-glob", action="append", default=[])
    p.add_argument("--current-manifest", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--q-direction-mode", choices=("mean", "max-norm"), default="mean")
    p.add_argument("--cluster-abs-cos-threshold", type=float, default=0.95)
    p.add_argument("--prefer-cos-threshold", type=float, default=0.95)
    p.add_argument("--strong-cos-threshold", type=float, default=0.90)
    p.add_argument("--top-count", type=int, default=12)
    p.add_argument("--new-case-start", type=int, default=40)
    p.add_argument("--compute-b-rms", action="store_true")
    args = p.parse_args()

    result = run_selection(
        collect_legacy_paths(args),
        Path(args.current_manifest).resolve(),
        Path(args.out_root).resolve(),
        q_direction_mode=args.q_direction_mode,
        cluster_abs_cos_threshold=float(args.cluster_abs_cos_threshold),
        prefer_cos_threshold=float(args.prefer_cos_threshold),
        strong_cos_threshold=float(args.strong_cos_threshold),
        top_count=int(args.top_count),
        new_case_start=int(args.new_case_start),
        compute_b_rms=bool(args.compute_b_rms),
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
