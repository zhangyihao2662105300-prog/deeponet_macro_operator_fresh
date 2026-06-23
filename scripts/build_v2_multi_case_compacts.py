#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build multi-case v2b local-strain compacts from strict fresh v1.x compacts.

This is a data-contract batch wrapper, not a training script.  It reuses the
existing one-case v2a/v2b builders:

    v2a: q48_raw -> q_useful, B_raw -> B_useful_abq
    v2b: Abaqus global LE/B -> local_jacobian_frame LE/B

The wrapper requires `X_keep_ref` so the q48 control-node order is explicit.  It
does not use old TRUE176 LE/B labels as v2 labels.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

import audit_v2_coordinate_consistent_contract as v2_audit
import build_v2_local_strain_pilot_compact as v2b_builder
import build_v2_q_useful_pilot_compact as v2a_builder


REQUIRED_SOURCE_FIELDS = (
    "q48_raw",
    "LE128_base",
    "B_LE128_forward",
    "ip_J",
    "ip_xi",
    "ip_invJ",
    "ip_detJ",
    "X_keep_ref",
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


def scalar_text(value: Any, default: str = "unknown") -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return str(default)
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def read_compact_list(path: Path) -> list[Path]:
    rows: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        rows.append(Path(item).resolve())
    return rows


def parse_case_id(path: Path) -> int:
    match = re.search(r"case[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot parse case id from path: {path}")
    return int(match.group(1))


def case_name(case_id: int) -> str:
    return f"case{int(case_id):03d}"


def shape_list(value: np.ndarray) -> list[int]:
    return [int(v) for v in np.asarray(value).shape]


def bool_scalar(value: Any, default: bool = False) -> bool:
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return bool(default)
        return bool(arr.reshape(-1)[0])
    except Exception:
        return bool(default)


def check_source_compact(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_compact": str(path),
        "source_exists": path.exists(),
        "source_check_pass": False,
        "missing_fields": [],
        "source_failures": [],
    }
    if not path.exists():
        row["source_failures"] = ["source_compact_missing"]
        return row

    with np.load(str(path), allow_pickle=True) as z:
        missing = [key for key in REQUIRED_SOURCE_FIELDS if key not in z.files]
        row["missing_fields"] = missing
        if missing:
            row["source_failures"] = [f"missing_{key}" for key in missing]
            return row

        q48 = np.asarray(z["q48_raw"])
        le = np.asarray(z["LE128_base"])
        b = np.asarray(z["B_LE128_forward"])
        x_ref = np.asarray(z["X_keep_ref"])
        ip_j = np.asarray(z["ip_J"])
        ip_xi = np.asarray(z["ip_xi"])
        ip_invj = np.asarray(z["ip_invJ"])
        ip_detj = np.asarray(z["ip_detJ"])

        failures: list[str] = []
        if q48.ndim != 2 or q48.shape[1] != 48:
            failures.append(f"bad_q48_raw_shape_{q48.shape}")
        if le.ndim != 3 or le.shape[1:] != (128, 6):
            failures.append(f"bad_LE128_base_shape_{le.shape}")
        if b.ndim != 4 or b.shape[1:] != (128, 6, 48):
            failures.append(f"bad_B_LE128_forward_shape_{b.shape}")
        if q48.ndim == 2 and le.ndim == 3 and q48.shape[0] != le.shape[0]:
            failures.append("q48_LE_frame_count_mismatch")
        if q48.ndim == 2 and b.ndim == 4 and q48.shape[0] != b.shape[0]:
            failures.append("q48_B_frame_count_mismatch")
        if x_ref.shape != (16, 3):
            failures.append(f"bad_X_keep_ref_shape_{x_ref.shape}")
        if ip_j.shape not in {(128, 3, 3), (q48.shape[0], 128, 3, 3)}:
            failures.append(f"bad_ip_J_shape_{ip_j.shape}")
        if ip_xi.shape not in {(128, 3), (q48.shape[0], 128, 3)}:
            failures.append(f"bad_ip_xi_shape_{ip_xi.shape}")
        if ip_invj.shape not in {(128, 3, 3), (q48.shape[0], 128, 3, 3)}:
            failures.append(f"bad_ip_invJ_shape_{ip_invj.shape}")
        if ip_detj.shape not in {(128,), (1, 128), (q48.shape[0], 128)}:
            failures.append(f"bad_ip_detJ_shape_{ip_detj.shape}")

        training_ready = bool_scalar(z["training_ready_sobolev"]) if "training_ready_sobolev" in z.files else False
        if not training_ready:
            failures.append("training_ready_sobolev_not_true")
        strain_field = scalar_text(z["strain_field"]) if "strain_field" in z.files else "unknown"
        b_strain_field = scalar_text(z["B_label_strain_field"]) if "B_label_strain_field" in z.files else "unknown"
        if strain_field != "LE" or b_strain_field != "LE":
            failures.append(f"strain_field_not_LE_{strain_field}_{b_strain_field}")

        row.update(
            {
                "q48_shape": shape_list(q48),
                "LE128_base_shape": shape_list(le),
                "B_LE128_forward_shape": shape_list(b),
                "X_keep_ref_shape": shape_list(x_ref),
                "ip_J_shape": shape_list(ip_j),
                "ip_xi_shape": shape_list(ip_xi),
                "ip_invJ_shape": shape_list(ip_invj),
                "ip_detJ_shape": shape_list(ip_detj),
                "frame_count": int(q48.shape[0]) if q48.ndim == 2 else None,
                "training_ready_sobolev": training_ready,
                "strain_field": strain_field,
                "B_label_strain_field": b_strain_field,
                "source_failures": failures,
                "source_check_pass": bool(not failures),
            }
        )
    return row


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")


def build_case(
    source: Path,
    *,
    case_id: int,
    out_root: Path,
    audit_root: Path,
    strict: bool,
    dry_run: bool,
) -> dict[str, Any]:
    cname = case_name(case_id)
    case_dir = out_root / cname
    audit_case_dir = audit_root / cname
    v2a_path = case_dir / f"complete_{cname}_v2a_q_useful.npz"
    v2b_path = case_dir / f"complete_{cname}_v2b_local_strain.npz"
    row: dict[str, Any] = {
        "case_id": case_id,
        "case_name": cname,
        "source_compact": str(source),
        "v2a_compact": str(v2a_path),
        "output_compact": str(v2b_path),
        "audit_root": str(audit_case_dir),
        "dry_run": bool(dry_run),
        "strict_v2_coordinate_pass": False,
        "failure": None,
    }

    source_check = check_source_compact(source)
    row.update({f"source_{key}": value for key, value in source_check.items() if key != "source_compact"})
    if not source_check.get("source_check_pass"):
        row["failure"] = ";".join(source_check.get("source_failures", [])) or "source_check_failed"
        return row
    if dry_run:
        row["strict_v2_coordinate_pass"] = None
        row["planned_output_compact"] = str(v2b_path)
        row["failure"] = None
        return row

    try:
        case_dir.mkdir(parents=True, exist_ok=True)
        v2a_args = argparse.Namespace(
            compact=str(source),
            out=str(v2a_path),
            case_id=int(case_id),
            q48_node_coordinate_source="X_keep_ref",
            strict=bool(strict),
        )
        v2a_payload, v2a_summary = v2a_builder.build_payload(v2a_args)
        if strict:
            if int(v2a_summary["rigid_mode_rank"]) != 6:
                raise ValueError("v2a rigid_mode_rank is not 6")
            if float(v2a_summary["rigid_annihilation_max"]) > 1.0e-10:
                raise ValueError(f"v2a rigid_annihilation_max too large: {v2a_summary['rigid_annihilation_max']}")
            if float(v2a_summary["B_chain_rule_projected_rel"]) > 1.0e-6:
                raise ValueError(f"v2a B_chain_rule_projected_rel too large: {v2a_summary['B_chain_rule_projected_rel']}")
        v2a_builder.write_npz(v2a_path, v2a_payload)
        write_json(v2a_path.with_suffix(".summary.json"), v2a_summary)

        v2b_args = argparse.Namespace(
            compact=str(v2a_path),
            out=str(v2b_path),
            case_id=int(case_id),
            ip_j_axis_convention="auto",
            degenerate_tol=1.0e-12,
            strict=bool(strict),
        )
        v2b_payload, v2b_summary = v2b_builder.build_payload(v2b_args)
        if strict:
            checks = {
                "local_frame_orthonormal_max": 1.0e-10,
                "T_eps_roundtrip_local_rel": 1.0e-10,
                "T_eps_roundtrip_abq_rel": 1.0e-10,
                "LE_local_roundtrip_rel": 1.0e-10,
                "B_local_chain_rule_projected_rel": 1.0e-8,
            }
            for key, tol in checks.items():
                val = float(v2b_summary[key])
                if not math.isfinite(val) or val > float(tol):
                    raise ValueError(f"v2b {key}={val:g} exceeds strict tolerance {tol:g}")
            if float(v2b_summary["local_frame_det_min"]) < 1.0 - 1.0e-10:
                raise ValueError(f"v2b local_frame_det_min={v2b_summary['local_frame_det_min']} is not right-handed")
        v2b_builder.write_npz(v2b_path, v2b_payload)
        write_json(v2b_path.with_suffix(".summary.json"), v2b_summary)

        audit_row = v2_audit.compact_report(v2b_path, tol=1.0e-8)
        audit_summary = v2_audit.write_outputs(audit_case_dir, [audit_row], argparse.Namespace(tol=1.0e-8))
        strict_pass = bool(audit_row.get("strict_v2_coordinate_pass"))
        if strict and not strict_pass:
            row["failure"] = "strict_v2_audit_failed:" + ",".join(audit_row.get("strict_failures", []))

        row.update(
            {
                "strict_v2_coordinate_pass": strict_pass,
                "strict_failures": audit_row.get("strict_failures", []),
                "q48_shape": v2a_summary.get("q48_shape"),
                "q_useful_shape": v2a_summary.get("q_useful_shape"),
                "T_q_shape": v2a_summary.get("T_q_raw_to_useful_shape"),
                "rigid_mode_rank": v2a_summary.get("rigid_mode_rank"),
                "rigid_annihilation_max": v2a_summary.get("rigid_annihilation_max"),
                "q_useful_reconstruction_rel": v2a_summary.get("q_useful_reconstruction_rel"),
                "q_useful_removed_rigid_rel": v2a_summary.get("q_useful_removed_rigid_rel"),
                "B_chain_rule_projected_rel": v2a_summary.get("B_chain_rule_projected_rel"),
                "B_chain_rule_raw_rel": v2a_summary.get("B_chain_rule_raw_rel"),
                "B_rigid_residual_rel": v2a_summary.get("B_rigid_residual_rel"),
                "local_frame_orthonormal_max": v2b_summary.get("local_frame_orthonormal_max"),
                "local_frame_det_min": v2b_summary.get("local_frame_det_min"),
                "local_frame_det_max": v2b_summary.get("local_frame_det_max"),
                "T_eps_roundtrip_local_rel": v2b_summary.get("T_eps_roundtrip_local_rel"),
                "T_eps_roundtrip_abq_rel": v2b_summary.get("T_eps_roundtrip_abq_rel"),
                "LE_local_roundtrip_rel": v2b_summary.get("LE_local_roundtrip_rel"),
                "B_local_chain_rule_projected_rel": v2b_summary.get("B_local_chain_rule_projected_rel"),
                "B_local_chain_rule_raw_rel": v2b_summary.get("B_local_chain_rule_raw_rel"),
                "audit_summary": audit_summary,
            }
        )
    except Exception as exc:  # Keep the batch ledger even when one case fails.
        row["failure"] = f"{type(exc).__name__}: {exc}"
        row["strict_v2_coordinate_pass"] = False
    return row


def finite_range(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    vals = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            vals.append(val)
    if not vals:
        return {"min": None, "max": None}
    return {"min": float(min(vals)), "max": float(max(vals))}


def write_manifest(out_root: Path, rows: list[dict[str, Any]], *, dry_run: bool) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    compact_list_path = out_root / "v2b_compact_list.txt"
    compact_paths = [str(row["output_compact"]) for row in rows if row.get("failure") is None]
    compact_list_path.write_text("\n".join(compact_paths) + ("\n" if compact_paths else ""), encoding="utf-8")

    pass_count = sum(1 for row in rows if row.get("strict_v2_coordinate_pass") is True)
    failed_cases = [
        {"case_id": row.get("case_id"), "failure": row.get("failure"), "strict_failures": row.get("strict_failures", [])}
        for row in rows
        if row.get("failure") is not None or row.get("strict_v2_coordinate_pass") is False
    ]
    summary = {
        "audit_name": "v2d_multi_case_v2b_compact_generation",
        "dry_run": bool(dry_run),
        "compact_count": len(rows),
        "pass_count": int(pass_count),
        "fail_count": int(len(failed_cases)),
        "failed_cases": failed_cases,
        "v2b_compact_list": str(compact_list_path),
        "q_removed_rel_range": finite_range(rows, "q_useful_removed_rigid_rel"),
        "B_rigid_residual_rel_range": finite_range(rows, "B_rigid_residual_rel"),
        "B_local_chain_rule_projected_rel_range": finite_range(rows, "B_local_chain_rule_projected_rel"),
        "LE_local_roundtrip_rel_range": finite_range(rows, "LE_local_roundtrip_rel"),
        "model_training_performed": False,
        "uses_old_true176_labels_as_v2_labels": False,
        "case_summaries": rows,
    }
    summary_path = out_root / "v2b_multi_case_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({**summary, "summary_path": str(summary_path)}, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", required=True, help="Text file containing source v1.x compact paths")
    parser.add_argument("--out-root", required=True, help="Output root for v2a/v2b generated compacts and summary")
    parser.add_argument("--audit-root", required=True, help="Output root for strict v2 audit ledgers")
    parser.add_argument("--strict", action="store_true", help="Apply strict per-case v2a/v2b checks")
    parser.add_argument("--dry-run", action="store_true", help="Only validate inputs and planned outputs; do not write NPZ files")
    args = parser.parse_args()

    compact_list = Path(args.compact_list).resolve()
    if not compact_list.exists():
        raise FileNotFoundError(compact_list)
    sources = read_compact_list(compact_list)
    if not sources:
        raise SystemExit("compact list is empty")

    out_root = Path(args.out_root).resolve()
    audit_root = Path(args.audit_root).resolve()
    rows = []
    for source in sources:
        case_id = parse_case_id(source)
        rows.append(
            build_case(
                source,
                case_id=case_id,
                out_root=out_root,
                audit_root=audit_root,
                strict=bool(args.strict),
                dry_run=bool(args.dry_run),
            )
        )

    summary = write_manifest(out_root, rows, dry_run=bool(args.dry_run))
    if args.strict and summary["fail_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
