#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose which Macro16 branch columns create out-of-distribution inputs.

This is a read-only Gate 33 helper. It does not train a network, change model
structure, change q48/LE ordering, or change the 128-point rule.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_macro16_force_stiffness import json_default, read_path_list, write_json  # noqa: E402
from macro_deeponet.macro16_geometry import macro16_standard_point_table  # noqa: E402
from macro_deeponet.train_macro16_boundary_sobolev import load_macro16_compacts, parse_int_list  # noqa: E402


def branch_names() -> list[str]:
    names: list[str] = []
    for i in range(48):
        names.append(f"q{i:02d}")
    for node in range(16):
        for axis in ("x", "y", "z"):
            names.append(f"X{node:02d}_{axis}")
    names.append("L_ref")
    return names


def group_name(col: int) -> str:
    if col < 48:
        return "q48_def_hat"
    if col < 96:
        axis = ("x", "y", "z")[(col - 48) % 3]
        return f"X16_hat_{axis}"
    return "L_ref"


def load_checkpoint_norms(path: Path) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    norms = checkpoint.get("norms", {})
    if "branch_mean" not in norms or "branch_std" not in norms:
        raise KeyError(f"{path}: missing branch_mean or branch_std")
    return np.asarray(norms["branch_mean"], dtype=np.float64).reshape(-1), np.asarray(norms["branch_std"], dtype=np.float64).reshape(-1)


def select_cases(case_ids: np.ndarray, text: str) -> np.ndarray:
    wanted = set(int(v) for v in parse_int_list(str(text))) if str(text).strip() else set()
    idx = np.arange(case_ids.shape[0], dtype=np.int64)
    if wanted:
        idx = idx[np.isin(case_ids, np.asarray(sorted(wanted), dtype=np.int64))]
    return idx


def top_columns(
    data: Any,
    branch_raw: np.ndarray,
    branch_norm: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    idx: np.ndarray,
    *,
    count: int,
) -> list[dict[str, Any]]:
    names = branch_names()
    vals = np.abs(branch_norm[idx])
    rows: list[dict[str, Any]] = []
    for col in range(vals.shape[1]):
        local_frame = int(np.argmax(vals[:, col]))
        frame = int(idx[local_frame])
        rows.append(
            {
                "column": int(col),
                "name": names[col],
                "group": group_name(col),
                "case": int(data.case_id[frame]),
                "frame": frame,
                "frame_local_in_case": int(np.sum(data.case_id[: frame + 1] == data.case_id[frame]) - 1),
                "raw": float(branch_raw[frame, col]),
                "mean": float(mean[col]),
                "std": float(std[col]),
                "z": float(branch_norm[frame, col]),
                "abs_z": float(vals[local_frame, col]),
            }
        )
    return sorted(rows, key=lambda r: -float(r["abs_z"]))[: int(count)]


def top_case_rows(
    data: Any,
    branch_raw: np.ndarray,
    branch_norm: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    idx: np.ndarray,
) -> list[dict[str, Any]]:
    names = branch_names()
    rows: list[dict[str, Any]] = []
    for case in sorted(np.unique(data.case_id[idx]).astype(int).tolist()):
        case_idx = idx[data.case_id[idx] == case]
        vals = np.abs(branch_norm[case_idx])
        flat = int(np.argmax(vals))
        local = flat // vals.shape[1]
        col = flat % vals.shape[1]
        frame = int(case_idx[local])
        rows.append(
            {
                "case": int(case),
                "frame_count": int(case_idx.size),
                "column": int(col),
                "name": names[col],
                "group": group_name(col),
                "frame": frame,
                "frame_local_in_case": int(local),
                "raw": float(branch_raw[frame, col]),
                "mean": float(mean[col]),
                "std": float(std[col]),
                "z": float(branch_norm[frame, col]),
                "abs_z": float(vals.reshape(-1)[flat]),
                "branch_norm_max": float(np.max(np.linalg.norm(branch_norm[case_idx], axis=1))),
                "q_norm_p50": float(np.percentile(np.linalg.norm(branch_raw[case_idx, :48], axis=1), 50)),
                "x_norm_p50": float(np.percentile(np.linalg.norm(branch_raw[case_idx, 48:96], axis=1), 50)),
                "l_ref": [float(v) for v in np.unique(branch_raw[case_idx, 96]).tolist()[:5]],
            }
        )
    return rows


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in rows:
        key = str(row["group"])
        item = out.setdefault(key, {"count": 0, "max_abs_z": 0.0, "top": None})
        item["count"] = int(item["count"]) + 1
        if float(row["abs_z"]) > float(item["max_abs_z"]):
            item["max_abs_z"] = float(row["abs_z"])
            item["top"] = row
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = [str(p) for p in read_path_list(Path(args.compact_list))]
    data = load_macro16_compacts(
        paths,
        point_table=macro16_standard_point_table(
            plane_order=int(args.plane_gauss_order),
            thickness_order=int(args.thickness_gauss_order),
        ),
        frame_stride=1,
        max_frames_per_compact=0,
        scale_mode="normalized",
        b_label_coordinate="auto",
    )
    mean, std_raw = load_checkpoint_norms(Path(args.checkpoint).resolve())
    std = np.maximum(std_raw, 1.0e-12)
    branch_raw = np.concatenate(
        [data.q48_hat, data.x16_hat.reshape(data.x16_hat.shape[0], -1), data.length_scale],
        axis=1,
    ).astype(np.float64)
    branch_norm = (branch_raw - mean.reshape(1, -1)) / std.reshape(1, -1)
    idx = select_cases(data.case_id, args.case_list)
    if idx.size == 0:
        raise ValueError("no frames selected")
    case_rows = top_case_rows(data, branch_raw, branch_norm, mean, std, idx)
    top_rows = top_columns(data, branch_raw, branch_norm, mean, std, idx, count=int(args.top_count))
    summary = {
        "script": "diagnose_macro16_branch_outlier_columns.py",
        "compact_list": str(Path(args.compact_list).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "case_list": sorted(np.unique(data.case_id[idx]).astype(int).tolist()),
        "frame_count": int(idx.size),
        "top_case_rows": case_rows,
        "top_columns": top_rows,
        "group_summary": group_summary(top_rows),
        "interpretation": {
            "largest_outlier_group": top_rows[0]["group"] if top_rows else "",
            "largest_outlier_column": top_rows[0]["name"] if top_rows else "",
            "largest_outlier_abs_z": float(top_rows[0]["abs_z"]) if top_rows else None,
            "branch_contract": "branch = q48_def_hat[48] + X16_hat[48] + L_ref[1]",
        },
    }
    if str(args.out).strip():
        write_json(Path(args.out).resolve(), summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--case-list", default="19,25,31,41,43,44,45,46,49,50,60,61,70,71,72,73")
    parser.add_argument("--top-count", type=int, default=40)
    parser.add_argument("--out", default="")
    parser.add_argument("--plane-gauss-order", type=int, default=3)
    parser.add_argument("--thickness-gauss-order", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
