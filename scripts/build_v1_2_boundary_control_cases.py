#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build fresh v1.2 q48 boundary-control case plans.

This script does not run Abaqus and does not train a model.  It converts
selected legacy TRUE176 q directions into fresh boundary-displacement control
frames:

    q_new = alpha * q_direction_48

Legacy LE/B values are kept only as scale hints in metadata; they are not labels
for v1.2 training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


KEEP_NODE_IDS = [1, 3, 5, 11, 15, 21, 23, 25, 26, 28, 30, 36, 40, 46, 48, 50]


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def parse_alpha_factors(text: str) -> list[float]:
    vals = [float(item.strip()) for item in str(text).replace(";", ",").split(",") if item.strip()]
    if not vals:
        raise ValueError("alpha factor list is empty")
    if any((not math.isfinite(v)) or v <= 0.0 for v in vals):
        raise ValueError(f"alpha factors must be positive finite values, got {vals}")
    return vals


def case_number(case_id: str) -> int:
    text = str(case_id).strip().lower()
    if text.startswith("case"):
        text = text[4:]
    return int(text)


def case_label(num: int) -> str:
    return f"case{int(num):03d}"


def vector_norm(vals: list[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in vals))


def normalize(vals: list[float], eps: float = 1.0e-12) -> tuple[list[float], float]:
    norm = vector_norm(vals)
    if not math.isfinite(norm) or norm <= eps:
        raise ValueError("q_direction_48 is zero or invalid")
    return [float(v) / norm for v in vals], norm


def dot(a: list[float], b: list[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def finite_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def load_cases(path: Path, case_start: int, case_end: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cases", [])
    out = []
    for row in rows:
        num = case_number(row["recommended_new_case_id"])
        if int(case_start) <= num <= int(case_end):
            out.append(row)
    out.sort(key=lambda r: case_number(r["recommended_new_case_id"]))
    if not out:
        raise ValueError(f"no cases in requested range case{case_start:03d}-case{case_end:03d}")
    return out


def build_case(
    row: dict[str, Any],
    *,
    alpha_factors: list[float],
    alpha_scale: float,
    min_alpha_mid: float,
    fallback_alpha_mid: float,
) -> dict[str, Any]:
    case_id = str(row["recommended_new_case_id"])
    q_dir, input_norm = normalize([float(v) for v in row["q_direction_48"]])
    if len(q_dir) != 48:
        raise ValueError(f"{case_id}: q_direction_48 must have length 48, got {len(q_dir)}")

    q_norm_seed_mean = finite_float(row.get("q_norm_seed_mean"))
    if q_norm_seed_mean is None or q_norm_seed_mean <= float(min_alpha_mid):
        alpha_mid = float(fallback_alpha_mid)
        alpha_source = "fallback_alpha_mid"
    else:
        alpha_mid = q_norm_seed_mean
        alpha_source = "q_norm_seed_mean"

    frames = []
    for i, factor in enumerate(alpha_factors, start=1):
        alpha = float(factor) * float(alpha_scale) * float(alpha_mid)
        q48 = [alpha * v for v in q_dir]
        q_norm = vector_norm(q48)
        q_dir_frame = [v / q_norm for v in q48] if q_norm > 0.0 else [0.0] * 48
        cos = dot(q_dir, q_dir_frame)
        frames.append(
            {
                "case_id": case_id,
                "frame_index": int(i),
                "alpha_factor": float(factor),
                "alpha": alpha,
                "q_norm": q_norm,
                "direction_cos_to_seed": cos,
                "q48": q48,
            }
        )

    q_norms = [float(frame["q_norm"]) for frame in frames]
    return {
        "case_id": case_id,
        "source_legacy_case": row.get("inferred_case_id"),
        "source_legacy_index": row.get("legacy_index"),
        "legacy_source_compact_path": row.get("legacy_source_compact_path", ""),
        "legacy_sample_path": row.get("legacy_sample_path", ""),
        "direction_type": row.get("direction_type", "unknown"),
        "max_abs_cos_to_current": row.get("max_abs_cos_to_current"),
        "nearest_existing_case": row.get("nearest_existing_case"),
        "q_direction_48": q_dir,
        "q_direction_input_norm": input_norm,
        "q_norm_seed_mean": q_norm_seed_mean,
        "alpha_mid": alpha_mid,
        "alpha_mid_source": alpha_source,
        "alpha_scale": float(alpha_scale),
        "alpha_factors": [float(v) for v in alpha_factors],
        "q_norm_min": min(q_norms),
        "q_norm_max": max(q_norms),
        "q48_frames": frames,
        "keep_node_ids": KEEP_NODE_IDS,
        "q48_node_dof_order": "q[3*i + dof] maps to KEEP_NODE_IDS[i], dof=1..3",
        "expected_LE_rms_hint": row.get("LE_rms_hint_mean"),
        "B_rms_hint": row.get("B_rms_hint_mean"),
        "notes": (
            "Fresh v1.2 boundary-control plan. Legacy LE/B values are scale hints only; "
            "new Abaqus ODB, Sobolev B compact, and strict_v1_2 complete compact export are required."
        ),
    }


def validate_case(case: dict[str, Any], tol: float) -> list[str]:
    errors: list[str] = []
    q_dir = [float(v) for v in case["q_direction_48"]]
    for frame in case["q48_frames"]:
        q = [float(v) for v in frame["q48"]]
        alpha = float(frame["alpha"])
        q_norm = vector_norm(q)
        if abs(q_norm - alpha) > float(tol) * max(1.0, abs(alpha)):
            errors.append(
                f"{case['case_id']} frame {frame['frame_index']}: q_norm {q_norm:.16g} != alpha {alpha:.16g}"
            )
        if q_norm > 0.0:
            q_dir_frame = [v / q_norm for v in q]
            cos = dot(q_dir, q_dir_frame)
            if abs(cos - 1.0) > 1.0e-10:
                errors.append(f"{case['case_id']} frame {frame['frame_index']}: direction cosine {cos:.16g} != 1")
    return errors


def write_case_csv(path: Path, case: dict[str, Any]) -> None:
    fields = ["case_id", "frame_index", "alpha_factor", "alpha", "q_norm"] + [f"q{i}" for i in range(1, 49)]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for frame in case["q48_frames"]:
            row = {
                "case_id": frame["case_id"],
                "frame_index": frame["frame_index"],
                "alpha_factor": frame["alpha_factor"],
                "alpha": frame["alpha"],
                "q_norm": frame["q_norm"],
            }
            row.update({f"q{i + 1}": frame["q48"][i] for i in range(48)})
            writer.writerow(row)


def write_summary_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "source_legacy_case",
        "source_legacy_index",
        "direction_type",
        "max_abs_cos_to_current",
        "nearest_existing_case",
        "q_norm_seed_mean",
        "alpha_mid",
        "alpha_mid_source",
        "q_norm_min",
        "q_norm_max",
        "expected_LE_rms_hint",
        "B_rms_hint",
        "frame_count",
        "legacy_sample_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "source_legacy_case": case["source_legacy_case"],
                    "source_legacy_index": case["source_legacy_index"],
                    "direction_type": case["direction_type"],
                    "max_abs_cos_to_current": case["max_abs_cos_to_current"],
                    "nearest_existing_case": case["nearest_existing_case"],
                    "q_norm_seed_mean": case["q_norm_seed_mean"],
                    "alpha_mid": case["alpha_mid"],
                    "alpha_mid_source": case["alpha_mid_source"],
                    "q_norm_min": case["q_norm_min"],
                    "q_norm_max": case["q_norm_max"],
                    "expected_LE_rms_hint": case["expected_LE_rms_hint"],
                    "B_rms_hint": case["B_rms_hint"],
                    "frame_count": len(case["q48_frames"]),
                    "legacy_sample_path": case["legacy_sample_path"],
                }
            )


def write_command_template(path: Path, cases: list[dict[str, Any]], out_root: Path) -> None:
    lines = [
        "# v1.2 Fresh Boundary-Control Command Templates",
        "",
        "These commands are templates only.  This repository currently has the complete",
        "compact exporter, but not a confirmed standalone script that turns a q48 frame",
        "plan into a fresh Abaqus ODB.  Do not treat these plans as training data.",
        "",
        "Core rule:",
        "",
        "```text",
        "q_new = alpha * q_direction_48",
        "legacy LE/B are not v1.2 labels",
        "fresh Abaqus run + current exporter + strict_v1_2_pass=true are required",
        "```",
        "",
        "q48 order:",
        "",
        "```text",
        "keep_node_ids = [1, 3, 5, 11, 15, 21, 23, 25, 26, 28, 30, 36, 40, 46, 48, 50]",
        "q[3*i + 0] -> KEEP_NODE_IDS[i], dof 1",
        "q[3*i + 1] -> KEEP_NODE_IDS[i], dof 2",
        "q[3*i + 2] -> KEEP_NODE_IDS[i], dof 3",
        "```",
        "",
        "Important Abaqus note:",
        "",
        "Existing generated `.inp` examples often apply displacements on 32 boundary",
        "nodes / 96 DOF after a boundary-contract bridge.  If your Abaqus generator",
        "expects 96 DOF, first apply the existing q48 -> q_boundary[32,3] bridge for",
        "the same shape/geometry contract.  Do not silently reinterpret q48 as 96 DOF.",
        "",
        "Template flow for each case:",
        "",
        "```powershell",
        "# 1. Generate an Abaqus input/ODB from the q48 frame CSV.",
        "# Replace this with the project-specific q48 boundary-control generator.",
        "py -3 <your_q48_to_abaqus_boundary_control_generator.py> `",
        "  --q48-frames <caseXXX_q48_frames.csv> `",
        "  --shape4-json <shape4-json-or-values> `",
        "  --out-root <fresh_case_run_root>",
        "",
        "# 2. Run Abaqus.",
        "abaqus job=<fresh_case_job> interactive",
        "",
        "# 3. Generate or locate the matching Sobolev B compact for the same q48 frames.",
        "",
        "# 4. Export the complete compact with hard guards.",
        "abaqus python scripts/export_abaqus_true176_complete_compact.py `",
        "  --odb <fresh_case_job>.odb `",
        "  --out <complete_caseXXX_training_ready.npz> `",
        "  --merge-compact <matching_LE_B_compact.npz> `",
        "  --require-b `",
        "  --require-ip-audit `",
        "  --require-merge-ip-keys `",
        "  --strain-field LE `",
        "  --merge-q-tol 1e-8 `",
        "  --merge-le-tol 1e-8",
        "```",
        "",
        "Case inputs:",
        "",
    ]
    for case in cases:
        lines.extend(
            [
                f"## {case['case_id']}",
                "",
                f"- q48 frame CSV: `{out_root / (case['case_id'] + '_q48_frames.csv')}`",
                f"- alpha_mid: `{case['alpha_mid']:.12g}`",
                f"- q_norm range: `{case['q_norm_min']:.12g}` to `{case['q_norm_max']:.12g}`",
                f"- direction type: `{case['direction_type']}`",
                f"- legacy source case: `{case['source_legacy_case']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.boundary_control_plan).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    alpha_factors = parse_alpha_factors(args.alpha_factors)

    raw_cases = load_cases(input_path, int(args.case_start), int(args.case_end))
    cases = [
        build_case(
            row,
            alpha_factors=alpha_factors,
            alpha_scale=float(args.alpha_scale),
            min_alpha_mid=float(args.min_alpha_mid),
            fallback_alpha_mid=float(args.fallback_alpha_mid),
        )
        for row in raw_cases
    ]

    validation_errors: list[str] = []
    for case in cases:
        validation_errors.extend(validate_case(case, float(args.validation_tol)))
        write_case_csv(out_root / f"{case['case_id']}_q48_frames.csv", case)

    summary = {
        "audit_name": "v1_2_fresh_boundary_control_case_plan",
        "source_boundary_control_plan": str(input_path),
        "case_count": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "frame_count_per_case": len(alpha_factors),
        "alpha_factors": alpha_factors,
        "alpha_scale": float(args.alpha_scale),
        "keep_node_ids": KEEP_NODE_IDS,
        "q48_node_dof_order": "q[3*i + dof] maps to KEEP_NODE_IDS[i], dof=1..3",
        "uses_legacy_LE_B_as_labels": False,
        "legacy_LE_B_usage": "scale hints only",
        "requires_fresh_abaqus_run": True,
        "requires_strict_v1_2_pass": True,
        "validation_pass": not validation_errors,
        "validation_errors": validation_errors,
        "target_outputs": [
            "fresh_boundary_control_plan.json",
            "fresh_boundary_control_plan.csv",
            "case040_q48_frames.csv ... case051_q48_frames.csv",
            "abaqus_boundary_control_command_templates.md",
        ],
    }
    payload = {"summary": summary, "cases": cases}
    (out_root / "fresh_boundary_control_plan.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    write_summary_csv(out_root / "fresh_boundary_control_plan.csv", cases)
    write_command_template(out_root / "abaqus_boundary_control_command_templates.md", cases, out_root)

    if validation_errors:
        raise ValueError("boundary control plan validation failed:\n" + "\n".join(validation_errors))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-control-plan", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--case-start", type=int, default=40)
    parser.add_argument("--case-end", type=int, default=51)
    parser.add_argument(
        "--alpha-factors",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated positive factors multiplied by alpha_mid.",
    )
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    parser.add_argument("--min-alpha-mid", type=float, default=1.0e-12)
    parser.add_argument("--fallback-alpha-mid", type=float, default=1.0e-3)
    parser.add_argument("--validation-tol", type=float, default=1.0e-10)
    args = parser.parse_args()

    summary = build_plan(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
