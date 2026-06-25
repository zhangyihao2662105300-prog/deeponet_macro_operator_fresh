#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose Macro16 AD-B loss scale.

This Gate 09 diagnostic is read-only with respect to data and model code.  It
reproduces the current trainer normalization for B targets and compares that
target with the physical qdef-coordinate B label.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macro_deeponet.macro16_geometry import macro16_source128_point_table  # noqa: E402
from macro_deeponet.train_macro16_boundary_sobolev import (  # noqa: E402
    _le_std_scale,
    load_macro16_compacts,
    split_indices,
)
from macro_deeponet.true176_data import stats  # noqa: E402


COMPONENT_NAMES = [f"component_{i}" for i in range(6)]


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
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def read_path_list(path: Path) -> list[Path]:
    return [
        Path(line.strip()).resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def compact_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for item in getattr(args, "compact", []) or []:
        paths.append(Path(item).resolve())
    for item in getattr(args, "compact_list", []) or []:
        paths.extend(read_path_list(Path(item).resolve()))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path.resolve())
    if not unique:
        raise ValueError("provide --compact or --compact-list")
    return unique


def scalar_summary(values: np.ndarray) -> dict[str, Any]:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        return {"count": 0}
    return {
        "count": int(vals.size),
        "min": float(np.min(vals)),
        "p01": float(np.percentile(vals, 1.0)),
        "p05": float(np.percentile(vals, 5.0)),
        "p10": float(np.percentile(vals, 10.0)),
        "median": float(np.percentile(vals, 50.0)),
        "p90": float(np.percentile(vals, 90.0)),
        "p95": float(np.percentile(vals, 95.0)),
        "p99": float(np.percentile(vals, 99.0)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "rms": float(np.sqrt(np.mean(vals * vals))),
    }


def magnitude_summary(values: np.ndarray) -> dict[str, Any]:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    abs_vals = np.abs(vals)
    out = scalar_summary(abs_vals)
    if vals.size:
        out.update(
            {
                "signed_min": float(np.min(vals)),
                "signed_max": float(np.max(vals)),
                "signed_mean": float(np.mean(vals)),
                "rms": float(np.sqrt(np.mean(vals * vals))),
            }
        )
    return out


def safe_share(energy: np.ndarray) -> np.ndarray:
    vals = np.asarray(energy, dtype=np.float64)
    total = float(np.sum(vals))
    if total <= 0.0:
        return np.zeros_like(vals, dtype=np.float64)
    return vals / total


def energy_concentration(values: np.ndarray) -> dict[str, float]:
    sq = np.asarray(values, dtype=np.float64).reshape(-1) ** 2
    if sq.size == 0:
        return {}
    order = np.sort(sq)[::-1]
    total = float(np.sum(order))
    if total <= 0.0:
        return {
            "top_0p1_percent_share": 0.0,
            "top_1_percent_share": 0.0,
            "top_5_percent_share": 0.0,
            "top_10_percent_share": 0.0,
            "top_100_values_share": 0.0,
        }

    def top_percent(frac: float) -> float:
        count = max(1, int(np.ceil(float(frac) * order.size)))
        return float(np.sum(order[:count]) / total)

    return {
        "top_0p1_percent_share": top_percent(0.001),
        "top_1_percent_share": top_percent(0.01),
        "top_5_percent_share": top_percent(0.05),
        "top_10_percent_share": top_percent(0.10),
        "top_100_values_share": float(np.sum(order[: min(100, order.size)]) / total),
    }


def top_indices(values: np.ndarray, limit: int) -> list[int]:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        return []
    order = np.argsort(vals)[::-1]
    return [int(v) for v in order[: max(0, min(int(limit), vals.size))]]


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(aa - bb) / max(float(np.linalg.norm(bb)), 1.0e-300))


def group_stats(
    b: np.ndarray,
    j: np.ndarray,
    q_std: np.ndarray,
    le_std: np.ndarray,
    xi: np.ndarray,
    *,
    top_n: int,
) -> dict[str, Any]:
    b64 = np.asarray(b, dtype=np.float64)
    j64 = np.asarray(j, dtype=np.float64)
    b_sq = b64 * b64
    j_sq = j64 * j64
    b_total = float(np.sum(b_sq))
    j_total = float(np.sum(j_sq))

    b_col = np.sum(b_sq, axis=(0, 1, 2))
    j_col = np.sum(j_sq, axis=(0, 1, 2))
    b_comp = np.sum(b_sq, axis=(0, 1, 3))
    j_comp = np.sum(j_sq, axis=(0, 1, 3))
    b_ip = np.sum(b_sq, axis=(0, 2, 3))
    j_ip = np.sum(j_sq, axis=(0, 2, 3))
    b_col_share = safe_share(b_col)
    j_col_share = safe_share(j_col)
    b_comp_share = safe_share(b_comp)
    j_comp_share = safe_share(j_comp)
    b_ip_share = safe_share(b_ip)
    j_ip_share = safe_share(j_ip)

    b_col_rms = np.sqrt(np.mean(b_sq, axis=(0, 1, 2)))
    j_col_rms = np.sqrt(np.mean(j_sq, axis=(0, 1, 2)))
    b_col_max = np.max(np.abs(b64), axis=(0, 1, 2))
    j_col_max = np.max(np.abs(j64), axis=(0, 1, 2))

    columns = []
    for col in top_indices(j_col_share, top_n):
        b_share = float(b_col_share[col])
        j_share = float(j_col_share[col])
        columns.append(
            {
                "column": int(col),
                "q_std": float(q_std[col]),
                "B_rms": float(b_col_rms[col]),
                "B_abs_max": float(b_col_max[col]),
                "B_energy_share": b_share,
                "j_norm_rms": float(j_col_rms[col]),
                "j_norm_abs_max": float(j_col_max[col]),
                "j_norm_energy_share": j_share,
                "j_to_B_energy_share_ratio": float(j_share / max(b_share, 1.0e-300)),
            }
        )

    b_comp_rms = np.sqrt(np.mean(b_sq, axis=(0, 1, 3)))
    j_comp_rms = np.sqrt(np.mean(j_sq, axis=(0, 1, 3)))
    b_comp_max = np.max(np.abs(b64), axis=(0, 1, 3))
    j_comp_max = np.max(np.abs(j64), axis=(0, 1, 3))
    le_std_by_comp = np.asarray(le_std, dtype=np.float64).reshape(le_std.shape[1], le_std.shape[2])
    components = []
    for comp in top_indices(j_comp_share, top_n):
        b_share = float(b_comp_share[comp])
        j_share = float(j_comp_share[comp])
        components.append(
            {
                "component": int(comp),
                "component_name": COMPONENT_NAMES[comp],
                "LE_std_min": float(np.min(le_std_by_comp[:, comp])),
                "LE_std_median": float(np.median(le_std_by_comp[:, comp])),
                "B_rms": float(b_comp_rms[comp]),
                "B_abs_max": float(b_comp_max[comp]),
                "B_energy_share": b_share,
                "j_norm_rms": float(j_comp_rms[comp]),
                "j_norm_abs_max": float(j_comp_max[comp]),
                "j_norm_energy_share": j_share,
                "j_to_B_energy_share_ratio": float(j_share / max(b_share, 1.0e-300)),
            }
        )

    b_ip_rms = np.sqrt(np.mean(b_sq, axis=(0, 2, 3)))
    j_ip_rms = np.sqrt(np.mean(j_sq, axis=(0, 2, 3)))
    b_ip_max = np.max(np.abs(b64), axis=(0, 2, 3))
    j_ip_max = np.max(np.abs(j64), axis=(0, 2, 3))
    le_std_by_ip = np.asarray(le_std, dtype=np.float64).reshape(le_std.shape[1], le_std.shape[2])
    ips = []
    for ip in top_indices(j_ip_share, top_n):
        b_share = float(b_ip_share[ip])
        j_share = float(j_ip_share[ip])
        ips.append(
            {
                "ip": int(ip),
                "xi": np.asarray(xi[ip], dtype=np.float64).tolist() if ip < xi.shape[0] else [],
                "LE_std_min": float(np.min(le_std_by_ip[ip])),
                "LE_std_median": float(np.median(le_std_by_ip[ip])),
                "B_rms": float(b_ip_rms[ip]),
                "B_abs_max": float(b_ip_max[ip]),
                "B_energy_share": b_share,
                "j_norm_rms": float(j_ip_rms[ip]),
                "j_norm_abs_max": float(j_ip_max[ip]),
                "j_norm_energy_share": j_share,
                "j_to_B_energy_share_ratio": float(j_share / max(b_share, 1.0e-300)),
            }
        )

    return {
        "B_energy_total": b_total,
        "j_norm_energy_total": j_total,
        "column_share_l1_distance_half": float(0.5 * np.sum(np.abs(j_col_share - b_col_share))),
        "component_share_l1_distance_half": float(0.5 * np.sum(np.abs(j_comp_share - b_comp_share))),
        "ip_share_l1_distance_half": float(0.5 * np.sum(np.abs(j_ip_share - b_ip_share))),
        "top_columns_by_j_norm_energy": columns,
        "top_components_by_j_norm_energy": components,
        "top_ips_by_j_norm_energy": ips,
    }


def top_elements(
    b: np.ndarray,
    j: np.ndarray,
    scale_factor: np.ndarray,
    q_std: np.ndarray,
    le_std: np.ndarray,
    case_id: np.ndarray,
    source_row: np.ndarray,
    xi: np.ndarray,
    *,
    top_n: int,
) -> dict[str, Any]:
    b_abs = np.abs(np.asarray(b, dtype=np.float64))
    j_abs = np.abs(np.asarray(j, dtype=np.float64))
    scale_abs = np.abs(np.asarray(scale_factor, dtype=np.float64))
    scale_full = np.broadcast_to(scale_abs, b_abs.shape)
    le_std_full = np.broadcast_to(np.asarray(le_std, dtype=np.float64).reshape(1, le_std.shape[1], le_std.shape[2], 1), b_abs.shape)

    def rows_for(metric: np.ndarray) -> list[dict[str, Any]]:
        flat_order = np.argsort(metric.reshape(-1))[::-1][:top_n]
        out = []
        for flat in flat_order:
            n, ip, comp, col = np.unravel_index(int(flat), metric.shape)
            out.append(
                {
                    "frame_index": int(n),
                    "case_id": int(case_id[n]),
                    "source_row": int(source_row[n]),
                    "ip": int(ip),
                    "xi": np.asarray(xi[ip], dtype=np.float64).tolist() if ip < xi.shape[0] else [],
                    "component": int(comp),
                    "column": int(col),
                    "B_value": float(b[n, ip, comp, col]),
                    "B_abs": float(b_abs[n, ip, comp, col]),
                    "j_norm_value": float(j[n, ip, comp, col]),
                    "j_norm_abs": float(j_abs[n, ip, comp, col]),
                    "scale_factor_qstd_over_LEstd": float(scale_full[n, ip, comp, col]),
                    "q_std": float(q_std[col]),
                    "LE_std": float(le_std_full[n, ip, comp, col]),
                }
            )
        return out

    return {
        "top_by_abs_j_norm_target": rows_for(j_abs),
        "top_by_abs_B_macro_qdef": rows_for(b_abs),
        "top_by_scale_factor": rows_for(scale_full),
    }


def low_scale_entries(q_std: np.ndarray, le_std: np.ndarray, xi: np.ndarray, *, top_n: int) -> dict[str, Any]:
    q_order = np.argsort(np.asarray(q_std, dtype=np.float64))
    le_vals = np.asarray(le_std, dtype=np.float64).reshape(le_std.shape[1], le_std.shape[2])
    le_order = np.argsort(le_vals.reshape(-1))
    q_rows = [
        {"column": int(col), "q_std": float(q_std[col])}
        for col in q_order[: max(0, min(top_n, q_order.size))]
    ]
    le_rows = []
    for flat in le_order[: max(0, min(top_n, le_order.size))]:
        ip, comp = np.unravel_index(int(flat), le_vals.shape)
        le_rows.append(
            {
                "ip": int(ip),
                "xi": np.asarray(xi[ip], dtype=np.float64).tolist() if ip < xi.shape[0] else [],
                "component": int(comp),
                "component_name": COMPONENT_NAMES[comp],
                "LE_std": float(le_vals[ip, comp]),
            }
        )
    return {"lowest_q_std_columns": q_rows, "lowest_LE_std_ip_components": le_rows}


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    paths = compact_paths(args)
    point_table = macro16_source128_point_table()
    data = load_macro16_compacts(
        [str(path) for path in paths],
        point_table=point_table,
        frame_stride=int(args.frame_stride),
        max_frames_per_compact=int(args.max_frames_per_compact),
        scale_mode="normalized",
        b_label_coordinate="auto",
    )
    train_idx, val_idx, split_meta = split_indices(
        data.case_id,
        val_fraction=float(args.val_fraction),
        val_cases=str(args.val_cases),
        seed=int(args.seed),
    )
    branch_raw = np.concatenate(
        [data.q48_hat, data.x16_hat.reshape(data.x16_hat.shape[0], -1), data.length_scale],
        axis=1,
    )
    branch_mean, branch_std = stats(branch_raw[train_idx], axis=0)
    le_mean, le_std = stats(data.le[train_idx], axis=0)
    q_std = branch_std.reshape(-1)[:48].astype(np.float64)
    le_scale = _le_std_scale(le_std, data.le.shape[1]).astype(np.float64)
    j_norm_target = (np.asarray(data.b, dtype=np.float64) * q_std.reshape(1, 1, 1, 48) / le_scale).astype(np.float64)
    b_recovered = j_norm_target * le_scale / q_std.reshape(1, 1, 1, 48)
    scale_factor = q_std.reshape(1, 1, 1, 48) / le_scale

    masks = {
        "train": train_idx,
        "val": val_idx,
        "all": np.arange(data.b.shape[0], dtype=np.int64),
    }
    subsets: dict[str, Any] = {}
    for name, idx in masks.items():
        subsets[name] = {
            "frame_count": int(idx.size),
            "case_ids": sorted(np.unique(data.case_id[idx]).astype(int).tolist()),
            "B_macro_qdef": magnitude_summary(data.b[idx]),
            "j_norm_target": magnitude_summary(j_norm_target[idx]),
            "B_energy_concentration": energy_concentration(data.b[idx]),
            "j_norm_energy_concentration": energy_concentration(j_norm_target[idx]),
            "group_energy": group_stats(
                data.b[idx],
                j_norm_target[idx],
                q_std,
                le_std,
                point_table.xi,
                top_n=int(args.top_n),
            ),
        }

    q_std_ratio = float(np.max(q_std) / max(float(np.min(q_std)), 1.0e-300))
    le_std_vals = np.asarray(le_std, dtype=np.float64).reshape(-1)
    le_std_ratio = float(np.max(le_std_vals) / max(float(np.min(le_std_vals)), 1.0e-300))
    scale_vals = np.asarray(scale_factor, dtype=np.float64).reshape(-1)
    scale_ratio = float(np.max(scale_vals) / max(float(np.min(scale_vals)), 1.0e-300))
    train_j = j_norm_target[train_idx]
    train_b = np.asarray(data.b[train_idx], dtype=np.float64)
    top1_share = float(energy_concentration(train_j).get("top_1_percent_share", 0.0))
    top01_share = float(energy_concentration(train_j).get("top_0p1_percent_share", 0.0))
    group = subsets["train"]["group_energy"]
    dominated = bool(
        top1_share > float(args.top1_dominance_threshold)
        or top01_share > float(args.top01_dominance_threshold)
        or q_std_ratio > float(args.scale_ratio_threshold)
        or le_std_ratio > float(args.scale_ratio_threshold)
        or float(group["column_share_l1_distance_half"]) > 0.25
        or float(group["component_share_l1_distance_half"]) > 0.25
        or float(group["ip_share_l1_distance_half"]) > 0.25
    )

    return {
        "status": "done",
        "diagnosis": {
            "current_B_loss_is_extreme_scale_dominated": dominated,
            "reason_flags": {
                "top_1_percent_j_norm_energy_share": top1_share,
                "top_0p1_percent_j_norm_energy_share": top01_share,
                "q_std_ratio_max_over_min": q_std_ratio,
                "LE_std_ratio_max_over_min": le_std_ratio,
                "scale_factor_ratio_max_over_min": scale_ratio,
                "column_share_l1_distance_half": float(group["column_share_l1_distance_half"]),
                "component_share_l1_distance_half": float(group["component_share_l1_distance_half"]),
                "ip_share_l1_distance_half": float(group["ip_share_l1_distance_half"]),
            },
        },
        "data": {
            "compact_paths": [str(path) for path in paths],
            "frame_count": int(data.b.shape[0]),
            "point_count": int(data.b.shape[1]),
            "case_ids": sorted(np.unique(data.case_id).astype(int).tolist()),
            "model_visible_q": data.point_meta.get("model_visible_q"),
            "model_visible_B": data.point_meta.get("model_visible_B"),
            "B_macro_qhat_source": data.point_meta.get("B_macro_qhat_source"),
            "q48_hat_source": data.point_meta.get("q48_hat_source"),
            "scale_consistency": data.point_meta.get("scale_consistency"),
        },
        "split": {
            **split_meta,
            "train_frame_count": int(train_idx.size),
            "val_frame_count": int(val_idx.size),
            "train_case_ids": sorted(np.unique(data.case_id[train_idx]).astype(int).tolist()),
            "val_case_ids": sorted(np.unique(data.case_id[val_idx]).astype(int).tolist()),
        },
        "normalization_formula": {
            "j_norm_target": "B_macro_qdef * q_std / LE_std",
            "B_recovered": "j_norm_target * LE_std / q_std",
            "B_recovery_rel_error": rel_l2(b_recovered, data.b),
            "B_recovery_max_abs_error": float(np.max(np.abs(b_recovered - np.asarray(data.b, dtype=np.float64)))),
            "branch_raw": "[q48_def_hat, X16_hat.flatten, L_ref]",
        },
        "scale": {
            "q48_def_hat_train_std": scalar_summary(q_std),
            "LE_macro_train_std": scalar_summary(le_std),
            "qstd_over_LEstd_scale_factor": scalar_summary(scale_factor),
            "lowest_scales": low_scale_entries(q_std, le_std, point_table.xi, top_n=int(args.top_n)),
            "q_std_ratio_max_over_min": q_std_ratio,
            "LE_std_ratio_max_over_min": le_std_ratio,
            "scale_factor_ratio_max_over_min": scale_ratio,
        },
        "subsets": subsets,
        "top_elements_train": top_elements(
            train_b,
            train_j,
            scale_factor,
            q_std,
            le_std,
            data.case_id[train_idx],
            data.source_row[train_idx],
            point_table.xi,
            top_n=int(args.top_n),
        ),
        "loss_scheme_recommendation": {
            "primary": "Use physical-coordinate B loss after converting AD J back to B_macro_qdef coordinate.",
            "formula": "B_pred = J_pred_norm * LE_std_eff / q_std_eff; loss = mean_balanced((B_pred - B_macro_qdef)^2 / B_scale^2)",
            "B_scale": "robust RMS of B_macro_qdef by column and component, with a global floor",
            "balance": [
                "average loss per q column, not per raw tensor element only",
                "average loss per LE component",
                "optionally average per integration point group when point dominance remains high",
            ],
            "avoid": [
                "do not select models by current J_norm_mse alone",
                "do not let tiny LE_std or near-single-direction q_std define the only B scale",
            ],
            "diagnostics_to_keep": [
                "physical AD_B_rel",
                "column-balanced B_rel",
                "component-balanced B_rel",
                "selected-frame force closure after training",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-list", action="append", default=[])
    parser.add_argument("--compact", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-cases", default="")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-compact", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--top1-dominance-threshold", type=float, default=0.50)
    parser.add_argument("--top01-dominance-threshold", type=float, default=0.20)
    parser.add_argument("--scale-ratio-threshold", type=float, default=100.0)
    args = parser.parse_args()
    payload = diagnose(args)
    write_json(Path(args.out), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "dominated": payload["diagnosis"]["current_B_loss_is_extreme_scale_dominated"],
                "train_B_rms": payload["subsets"]["train"]["B_macro_qdef"]["rms"],
                "train_j_norm_rms": payload["subsets"]["train"]["j_norm_target"]["rms"],
                "top_1_percent_j_share": payload["diagnosis"]["reason_flags"]["top_1_percent_j_norm_energy_share"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
