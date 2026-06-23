#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit the proposed v2 coordinate-consistent query-point compact contract.

This script is intentionally read-only.  It does not train a model, does not
transform labels, and does not treat old TRUE176 LE/B values as v2 labels.

The v2 contract requires enough metadata to make the chain rule explicit:

    q_useful = T_q_raw_to_useful @ q48_raw
    B_raw_hat = T_eps_to_abq @ d(strain_std)/d(q_useful) @ T_q_raw_to_useful

Current v1.x compacts are expected to fail this strict v2 audit because they do
not yet store q_useful/T_q and strain-coordinate transforms.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


Q_USEFUL_KEYS = (
    "q_useful",
    "q48_useful",
    "q_effective",
    "q48_effective",
    "q_branch_useful",
    "q48_rigid_projected",
)
T_Q_KEYS = (
    "T_q_raw_to_useful",
    "T_q",
    "q_raw_to_useful",
    "q48_raw_to_useful",
    "q_useful_transform",
)
T_EPS_KEYS = (
    "T_eps_to_abq",
    "T_epsilon_to_abq",
    "T_epsilon_std_to_abq",
    "T_eps_std_to_abq",
    "strain_std_to_abq",
    "T_strain_to_abq",
    "T_LE_local_to_global",
)
B_USEFUL_KEYS = (
    "B_standard_useful",
    "B_std_useful",
    "B_LE_useful",
    "B_LE128_useful",
    "B_label_useful",
    "B_useful",
)
IP_XI_KEYS = ("ip_xi", "xi128", "ip_natural_coords", "natural_coords")
IP_J_KEYS = ("ip_J", "ip_jacobian", "J128", "jmat", "ip_jmat")
IP_INVJ_KEYS = ("ip_invJ", "ip_inverse_jacobian", "invJ128", "invj", "ip_invj")
IP_DETJ_KEYS = ("ip_detJ", "detJ128", "detj", "ip_detj")


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
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def compact_paths_from_args(args: argparse.Namespace) -> list[Path]:
    paths: list[str] = []
    for item in args.compact:
        paths.append(str(item))
    for pattern in args.compact_glob:
        paths.extend(glob.glob(str(pattern)))
    for list_path in args.compact_list:
        paths.extend(read_list(Path(list_path)))
    unique = []
    seen = set()
    for item in paths:
        resolved = str(Path(item).resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(Path(resolved))
    return unique


def find_key(z: np.lib.npyio.NpzFile, keys: Iterable[str]) -> str | None:
    files = set(z.files)
    for key in keys:
        if key in files:
            return key
    return None


def scalar_text(z: np.lib.npyio.NpzFile, keys: Iterable[str] | str, default: str = "unknown") -> str:
    key_list = [keys] if isinstance(keys, str) else list(keys)
    key = find_key(z, key_list)
    if key is None:
        return str(default)
    arr = np.asarray(z[key])
    if arr.size == 0:
        return str(default)
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def shape_or_none(z: np.lib.npyio.NpzFile, key: str | None) -> list[int] | None:
    if key is None:
        return None
    return [int(v) for v in np.asarray(z[key]).shape]


def array_shape_ok(arr: np.ndarray, n_frames: int, point_count: int, tail: tuple[int, ...] | None = None) -> bool:
    vals = np.asarray(arr)
    if vals.ndim < 1:
        return False
    if vals.shape[0] == int(n_frames):
        rest = vals.shape[1:]
    elif vals.shape[0] == int(point_count):
        rest = vals.shape
    else:
        return False
    if tail is None:
        return True
    return tuple(rest[-len(tail) :]) == tuple(tail) if len(rest) >= len(tail) else False


def check_t_q(q_raw: np.ndarray | None, q_useful: np.ndarray | None, t_q: np.ndarray | None, tol: float) -> dict[str, Any]:
    report: dict[str, Any] = {
        "T_q_shape_ok": False,
        "q_useful_matches_T_q_q_raw": None,
        "q_useful_reconstruction_max_abs": None,
        "q_useful_reconstruction_rel": None,
        "q_useful_dim": None,
        "T_q_rank_min": None,
        "T_q_rank_max": None,
    }
    if q_raw is None or t_q is None:
        return report
    q = np.asarray(q_raw, dtype=np.float64)
    t = np.asarray(t_q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 48:
        return report
    if t.ndim == 2 and t.shape[1] == 48:
        pred = q @ t.T
        ranks = [int(np.linalg.matrix_rank(t))]
        report["T_q_shape_ok"] = True
    elif t.ndim == 3 and t.shape[0] == q.shape[0] and t.shape[2] == 48:
        pred = np.einsum("nki,ni->nk", t, q)
        ranks = [int(np.linalg.matrix_rank(ti)) for ti in t]
        report["T_q_shape_ok"] = True
    else:
        return report
    report["q_useful_dim"] = int(pred.shape[1])
    report["T_q_rank_min"] = int(min(ranks))
    report["T_q_rank_max"] = int(max(ranks))
    if q_useful is None:
        return report
    useful = np.asarray(q_useful, dtype=np.float64)
    if useful.shape != pred.shape:
        report["q_useful_matches_T_q_q_raw"] = False
        report["q_useful_shape_mismatch"] = {"expected": list(pred.shape), "got": list(useful.shape)}
        return report
    diff = pred - useful
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    denom = max(float(np.linalg.norm(useful)), 1.0e-30)
    rel = float(np.linalg.norm(diff) / denom)
    report["q_useful_reconstruction_max_abs"] = max_abs
    report["q_useful_reconstruction_rel"] = rel
    report["q_useful_matches_T_q_q_raw"] = bool(max_abs <= float(tol))
    return report


def compact_report(path: Path, *, tol: float) -> dict[str, Any]:
    row: dict[str, Any] = {"compact_path": str(path), "exists": path.exists()}
    if not path.exists():
        row["strict_v2_coordinate_pass"] = False
        row["strict_failures"] = ["compact_missing"]
        return row

    with np.load(str(path), allow_pickle=True) as z:
        q_key = "q48_raw" if "q48_raw" in z.files else None
        q_useful_key = find_key(z, Q_USEFUL_KEYS)
        t_q_key = find_key(z, T_Q_KEYS)
        le_key = "LE128_base" if "LE128_base" in z.files else ("le" if "le" in z.files else None)
        b_key = "B_LE128_forward" if "B_LE128_forward" in z.files else ("b" if "b" in z.files else None)
        b_useful_key = find_key(z, B_USEFUL_KEYS)
        t_eps_key = find_key(z, T_EPS_KEYS)
        ip_xi_key = find_key(z, IP_XI_KEYS)
        ip_j_key = find_key(z, IP_J_KEYS)
        ip_invj_key = find_key(z, IP_INVJ_KEYS)
        ip_detj_key = find_key(z, IP_DETJ_KEYS)

        q_raw = np.asarray(z[q_key], dtype=np.float64) if q_key else None
        q_useful = np.asarray(z[q_useful_key], dtype=np.float64) if q_useful_key else None
        t_q = np.asarray(z[t_q_key], dtype=np.float64) if t_q_key else None

        frame_count = int(q_raw.shape[0]) if q_raw is not None and q_raw.ndim >= 1 else None
        point_count = 128
        le_shape = shape_or_none(z, le_key)
        b_shape = shape_or_none(z, b_key)
        if le_shape and len(le_shape) >= 2:
            point_count = int(le_shape[1])
        elif b_shape and len(b_shape) >= 2:
            point_count = int(b_shape[1])

        t_q_report = check_t_q(q_raw, q_useful, t_q, tol)
        strain_output_coordinate = scalar_text(
            z,
            (
                "strain_output_coordinate",
                "strain_coordinate_system",
                "output_strain_coordinate",
                "LE_coordinate_system",
            ),
            "unknown",
        )
        q_useful_coordinate = scalar_text(
            z,
            ("q_useful_coordinate", "q_branch_coordinate", "branch_q_coordinate", "q_coordinate"),
            "unknown",
        )
        b_label_q_coordinate = scalar_text(
            z,
            ("B_label_q_coordinate", "B_q_coordinate", "b_label_coordinate", "B_label_coordinate"),
            "unknown",
        )
        b_label_output_coordinate = scalar_text(
            z,
            ("B_label_output_coordinate", "B_output_coordinate", "B_label_strain_coordinate"),
            "unknown",
        )
        b_chain_rule = scalar_text(z, ("B_chain_rule", "B_chain_rule_contract", "B_transform_contract"), "unknown")

        ip_xi_ok = bool(ip_xi_key and frame_count is not None and array_shape_ok(z[ip_xi_key], frame_count, point_count, (3,)))
        ip_j_ok = bool(ip_j_key and frame_count is not None and array_shape_ok(z[ip_j_key], frame_count, point_count, (3, 3)))
        ip_invj_ok = bool(ip_invj_key and frame_count is not None and array_shape_ok(z[ip_invj_key], frame_count, point_count, (3, 3)))
        ip_detj_ok = bool(ip_detj_key and frame_count is not None and array_shape_ok(z[ip_detj_key], frame_count, point_count))

        failures: list[str] = []
        if q_raw is None or q_raw.ndim != 2 or q_raw.shape[1] != 48:
            failures.append("missing_or_bad_q48_raw")
        if q_useful_key is None:
            failures.append("missing_q_useful")
        if t_q_key is None:
            failures.append("missing_T_q_raw_to_useful")
        if not t_q_report["T_q_shape_ok"]:
            failures.append("bad_T_q_shape")
        if t_q_report["q_useful_matches_T_q_q_raw"] is not True:
            failures.append("q_useful_not_verified_from_T_q")
        if not ip_xi_ok:
            failures.append("missing_or_bad_ip_xi")
        if not ip_j_ok:
            failures.append("missing_or_bad_ip_J")
        if not ip_invj_ok:
            failures.append("missing_or_bad_ip_invJ")
        if not ip_detj_ok:
            failures.append("missing_or_bad_ip_detJ")
        if strain_output_coordinate.strip().lower() in {"", "unknown", "none", "null"}:
            failures.append("missing_strain_output_coordinate")
        if q_useful_coordinate.strip().lower() in {"", "unknown", "none", "null"}:
            failures.append("missing_q_useful_coordinate")
        if b_label_q_coordinate.strip().lower() in {"", "unknown", "none", "null"}:
            failures.append("missing_B_label_q_coordinate")
        if b_label_output_coordinate.strip().lower() in {"", "unknown", "none", "null"}:
            failures.append("missing_B_label_output_coordinate")
        if t_eps_key is None and b_useful_key is None:
            failures.append("missing_T_eps_to_abq_or_B_standard_useful")
        if b_chain_rule.strip().lower() in {"", "unknown", "none", "null"}:
            failures.append("missing_B_chain_rule_metadata")

        row.update(
            {
                "npz_keys": list(z.files),
                "frame_count": frame_count,
                "point_count": point_count,
                "q48_raw_key": q_key,
                "q48_raw_shape": shape_or_none(z, q_key),
                "q_useful_key": q_useful_key,
                "q_useful_shape": shape_or_none(z, q_useful_key),
                "T_q_key": t_q_key,
                "T_q_shape": shape_or_none(z, t_q_key),
                "LE_label_key": le_key,
                "LE_label_shape": le_shape,
                "B_label_key": b_key,
                "B_label_shape": b_shape,
                "B_useful_key": b_useful_key,
                "B_useful_shape": shape_or_none(z, b_useful_key),
                "T_eps_to_abq_key": t_eps_key,
                "T_eps_to_abq_shape": shape_or_none(z, t_eps_key),
                "ip_xi_key": ip_xi_key,
                "ip_xi_shape": shape_or_none(z, ip_xi_key),
                "ip_J_key": ip_j_key,
                "ip_J_shape": shape_or_none(z, ip_j_key),
                "ip_invJ_key": ip_invj_key,
                "ip_invJ_shape": shape_or_none(z, ip_invj_key),
                "ip_detJ_key": ip_detj_key,
                "ip_detJ_shape": shape_or_none(z, ip_detj_key),
                "ip_xi_ok": ip_xi_ok,
                "ip_J_ok": ip_j_ok,
                "ip_invJ_ok": ip_invj_ok,
                "ip_detJ_ok": ip_detj_ok,
                "strain_field": scalar_text(z, "strain_field", "unknown"),
                "B_label_strain_field": scalar_text(z, "B_label_strain_field", "unknown"),
                "strain_output_coordinate": strain_output_coordinate,
                "q_useful_coordinate": q_useful_coordinate,
                "B_label_q_coordinate": b_label_q_coordinate,
                "B_label_output_coordinate": b_label_output_coordinate,
                "B_chain_rule": b_chain_rule,
                **t_q_report,
                "strict_failures": failures,
                "strict_v2_coordinate_pass": bool(not failures),
            }
        )
    return row


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key == "npz_keys":
            continue
        if isinstance(value, (list, dict, tuple)):
            out[key] = json.dumps(value, ensure_ascii=False, sort_keys=True, default=json_default)
        else:
            out[key] = value
    return out


def write_outputs(out_root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    pass_count = sum(1 for row in rows if row.get("strict_v2_coordinate_pass"))
    failure_counts: dict[str, int] = {}
    for row in rows:
        for failure in row.get("strict_failures", []):
            failure_counts[str(failure)] = failure_counts.get(str(failure), 0) + 1
    summary = {
        "audit_name": "v2_coordinate_consistent_contract",
        "compact_count": len(rows),
        "strict_v2_coordinate_pass_count": pass_count,
        "strict_v2_coordinate_pass": bool(rows and pass_count == len(rows)),
        "failure_counts": dict(sorted(failure_counts.items())),
        "tolerance": float(args.tol),
        "note": (
            "This audit is read-only. Existing v1.x compacts are expected to fail "
            "until q_useful/T_q and strain/B coordinate metadata are added."
        ),
    }
    (out_root / "v2_coordinate_contract_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    (out_root / "v2_coordinate_contract_manifest.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    csv_rows = [flatten_for_csv(row) for row in rows]
    if csv_rows:
        fields = sorted({key for row in csv_rows for key in row.keys()})
        with (out_root / "v2_coordinate_contract_manifest.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="append", default=[], help="Path to a complete compact .npz")
    parser.add_argument("--compact-glob", action="append", default=[], help="Glob for complete compact .npz files")
    parser.add_argument("--compact-list", action="append", default=[], help="Text file containing compact paths")
    parser.add_argument("--out-root", required=True, help="Output directory for v2 contract audit ledgers")
    parser.add_argument("--tol", type=float, default=1.0e-8, help="Absolute tolerance for q_useful = T_q @ q48_raw")
    parser.add_argument("--strict-v2", action="store_true", help="Exit nonzero unless every compact passes strict v2")
    args = parser.parse_args()

    paths = compact_paths_from_args(args)
    if not paths:
        raise SystemExit("no compact inputs provided")
    rows = [compact_report(path, tol=float(args.tol)) for path in paths]
    summary = write_outputs(Path(args.out_root).resolve(), rows, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))
    if args.strict_v2 and not summary["strict_v2_coordinate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
