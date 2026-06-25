#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay Macro16 branch normalization with candidate geometry std floors.

This Gate 34 helper is read-only. It does not train a network, change model
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


def parse_float_list(text: str) -> list[float]:
    return [float(v.strip()) for v in str(text).split(",") if v.strip()]


def select_cases(case_ids: np.ndarray, text: str) -> np.ndarray:
    wanted = set(int(v) for v in parse_int_list(str(text))) if str(text).strip() else set()
    idx = np.arange(case_ids.shape[0], dtype=np.int64)
    if wanted:
        idx = idx[np.isin(case_ids, np.asarray(sorted(wanted), dtype=np.int64))]
    return idx


def load_checkpoint_norms(path: Path) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    norms = checkpoint.get("norms", {})
    if "branch_mean" not in norms or "branch_std" not in norms:
        raise KeyError(f"{path}: missing branch_mean or branch_std")
    return np.asarray(norms["branch_mean"], dtype=np.float64).reshape(-1), np.asarray(norms["branch_std"], dtype=np.float64).reshape(-1)


def finite_stats(vals: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(vals, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": int(arr.size), "finite_count": 0}
    return {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def group_norms(branch_norm: np.ndarray, idx: np.ndarray) -> dict[str, dict[str, Any]]:
    vals = np.asarray(branch_norm, dtype=np.float64)
    return {
        "all": finite_stats(np.linalg.norm(vals[idx], axis=1)),
        "q48_def_hat": finite_stats(np.linalg.norm(vals[idx, :48], axis=1)),
        "X16_hat": finite_stats(np.linalg.norm(vals[idx, 48:96], axis=1)),
        "L_ref": finite_stats(np.abs(vals[idx, 96])),
        "max_abs_column": finite_stats(np.max(np.abs(vals[idx]), axis=1)),
    }


def apply_floor(std: np.ndarray, *, x_floor: float, q_floor: float, l_ref_floor: float) -> np.ndarray:
    out = np.maximum(np.asarray(std, dtype=np.float64).reshape(-1), 1.0e-12).copy()
    if float(q_floor) > 0.0:
        out[:48] = np.maximum(out[:48], float(q_floor))
    if float(x_floor) > 0.0:
        out[48:96] = np.maximum(out[48:96], float(x_floor))
    if float(l_ref_floor) > 0.0:
        out[96] = max(float(out[96]), float(l_ref_floor))
    return out


def top_abs(branch_norm: np.ndarray, branch_raw: np.ndarray, mean: np.ndarray, std: np.ndarray, case_id: np.ndarray, idx: np.ndarray) -> dict[str, Any]:
    names: list[str] = []
    for i in range(48):
        names.append(f"q{i:02d}")
    for node in range(16):
        for axis in ("x", "y", "z"):
            names.append(f"X{node:02d}_{axis}")
    names.append("L_ref")
    vals = np.abs(branch_norm[idx])
    flat = int(np.argmax(vals))
    local = flat // vals.shape[1]
    col = flat % vals.shape[1]
    frame = int(idx[local])
    return {
        "case": int(case_id[frame]),
        "frame": frame,
        "column": int(col),
        "name": names[col],
        "raw": float(branch_raw[frame, col]),
        "mean": float(mean[col]),
        "std": float(std[col]),
        "z": float(branch_norm[frame, col]),
        "abs_z": float(vals.reshape(-1)[flat]),
    }


def replay_one(
    branch_raw: np.ndarray,
    case_id: np.ndarray,
    mean: np.ndarray,
    std_base: np.ndarray,
    train_idx: np.ndarray,
    wind_idx: np.ndarray,
    *,
    x_floor: float,
    q_floor: float,
    l_ref_floor: float,
) -> dict[str, Any]:
    std = apply_floor(std_base, x_floor=x_floor, q_floor=q_floor, l_ref_floor=l_ref_floor)
    normed = (branch_raw - mean.reshape(1, -1)) / std.reshape(1, -1)
    train_stats = group_norms(normed, train_idx)
    wind_stats = group_norms(normed, wind_idx)
    ratio = float(wind_stats["all"]["max"] / max(float(train_stats["all"]["p99"]), 1.0e-300))
    return {
        "x16_std_floor": float(x_floor),
        "q_std_floor": float(q_floor),
        "l_ref_std_floor": float(l_ref_floor),
        "train": train_stats,
        "wind_shell": wind_stats,
        "wind_max_over_train_p99": ratio,
        "wind_top_abs": top_abs(normed, branch_raw, mean, std, case_id, wind_idx),
        "train_top_abs": top_abs(normed, branch_raw, mean, std, case_id, train_idx),
        "pass_branch_range_100": bool(float(wind_stats["all"]["max"]) <= 100.0),
        "pass_ratio_10": bool(ratio <= 10.0),
    }


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
    mean, std_base = load_checkpoint_norms(Path(args.checkpoint).resolve())
    branch_raw = np.concatenate(
        [data.q48_hat, data.x16_hat.reshape(data.x16_hat.shape[0], -1), data.length_scale],
        axis=1,
    ).astype(np.float64)
    train_idx = select_cases(data.case_id, args.train_cases)
    wind_idx = select_cases(data.case_id, args.wind_cases)
    if train_idx.size == 0:
        raise ValueError("no train frames selected")
    if wind_idx.size == 0:
        raise ValueError("no wind-shell frames selected")
    candidates = [
        replay_one(
            branch_raw,
            data.case_id,
            mean,
            std_base,
            train_idx,
            wind_idx,
            x_floor=floor,
            q_floor=float(args.q_std_floor),
            l_ref_floor=float(args.l_ref_std_floor),
        )
        for floor in parse_float_list(args.x16_std_floors)
    ]
    recommended = None
    passing = [row for row in candidates if row["pass_branch_range_100"] and row["pass_ratio_10"]]
    if passing:
        recommended = min(passing, key=lambda r: float(r["x16_std_floor"]))
    summary = {
        "script": "replay_macro16_branch_normalization.py",
        "compact_list": str(Path(args.compact_list).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "train_cases": sorted(np.unique(data.case_id[train_idx]).astype(int).tolist()),
        "wind_cases": sorted(np.unique(data.case_id[wind_idx]).astype(int).tolist()),
        "branch_contract": "branch = q48_def_hat[48] + X16_hat[48] + L_ref[1]",
        "candidates": candidates,
        "recommended": recommended,
        "decision": {
            "normalization_replay_pass": bool(recommended is not None),
            "next_step": "implement guarded train-time geometry std floor only if this replay is accepted",
        },
    }
    if str(args.out).strip():
        write_json(Path(args.out).resolve(), summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", default="")
    parser.add_argument("--train-cases", default="19,25,31,41,43,44,45,46,49,50,60,61")
    parser.add_argument("--wind-cases", default="70,71,72,73")
    parser.add_argument("--x16-std-floors", default="0,0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--q-std-floor", type=float, default=0.0)
    parser.add_argument("--l-ref-std-floor", type=float, default=0.0)
    parser.add_argument("--plane-gauss-order", type=int, default=3)
    parser.add_argument("--thickness-gauss-order", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
