#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a small Macro16 real-data training smoke on two geometries.

The script selects one near-regular geometry and one lightly distorted geometry
from Macro16-compatible compacts, writes a tiny normalized compact, then calls
the v4 Macro16 Sobolev trainer.  It is meant as the step before any large run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macro_deeponet.macro16_geometry import (  # noqa: E402
    MACRO16_CONTRACT_VERSION,
    Macro16GeometryMap,
    macro16_standard_point_table,
)
from macro_deeponet.train_macro16_boundary_sobolev import (  # noqa: E402
    Macro16Arrays,
    load_macro16_compacts,
    train as train_macro16_boundary,
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


def read_path_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def compact_paths_from_args(args: argparse.Namespace | SimpleNamespace) -> list[str]:
    paths = [str(v) for v in getattr(args, "compact", [])]
    if str(getattr(args, "compact_list", "")).strip():
        paths.extend(read_path_list(Path(getattr(args, "compact_list"))))
    unique: list[str] = []
    seen: set[str] = set()
    for text in paths:
        key = str(Path(text).resolve())
        if key not in seen:
            seen.add(key)
            unique.append(key)
    if not unique:
        raise ValueError("provide --compact or --compact-list")
    return unique


def geometry_distortion_score(x16_hat: np.ndarray) -> dict[str, float]:
    geom = Macro16GeometryMap(np.asarray(x16_hat, dtype=np.float64).reshape(16, 3))
    table = macro16_standard_point_table(plane_order=3, thickness_order=2)
    fields = geom.eval_points(table)
    det = np.asarray(fields["detJ_hat"], dtype=np.float64)
    thickness = np.asarray(fields["thickness_hat"], dtype=np.float64)
    metric = np.asarray(fields["metric_hat"], dtype=np.float64)
    det_ratio = float(np.max(det) / max(float(np.min(det)), 1.0e-30))
    thickness_ratio = float(np.max(thickness) / max(float(np.min(thickness)), 1.0e-30))
    diag = np.sqrt(np.maximum(np.einsum("pii->pi", metric), 1.0e-30))
    skew01 = np.abs(metric[:, 0, 1]) / np.maximum(diag[:, 0] * diag[:, 1], 1.0e-30)
    skew02 = np.abs(metric[:, 0, 2]) / np.maximum(diag[:, 0] * diag[:, 2], 1.0e-30)
    skew12 = np.abs(metric[:, 1, 2]) / np.maximum(diag[:, 1] * diag[:, 2], 1.0e-30)
    skew_max = float(np.max(np.stack([skew01, skew02, skew12], axis=1)))
    score = math.log(max(det_ratio, 1.0)) + 0.25 * math.log(max(thickness_ratio, 1.0)) + skew_max
    return {
        "distortion_score": float(score),
        "detJ_min": float(np.min(det)),
        "detJ_max": float(np.max(det)),
        "detJ_ratio": det_ratio,
        "thickness_ratio": thickness_ratio,
        "metric_skew_max": skew_max,
    }


def geometry_groups(data: Macro16Arrays, *, decimals: int) -> list[dict[str, Any]]:
    groups: dict[tuple[float, ...], list[int]] = {}
    flat = data.x16_hat.reshape(data.x16_hat.shape[0], -1)
    for idx, row in enumerate(np.round(flat, int(decimals))):
        groups.setdefault(tuple(float(v) for v in row.tolist()), []).append(int(idx))
    rows: list[dict[str, Any]] = []
    for group_id, (_key, indices) in enumerate(groups.items()):
        x16 = np.mean(data.x16_hat[np.asarray(indices, dtype=np.int64)], axis=0)
        metrics = geometry_distortion_score(x16)
        rows.append(
            {
                "group_id": int(group_id),
                "frame_indices": indices,
                "frame_count": int(len(indices)),
                **metrics,
            }
        )
    return sorted(rows, key=lambda item: (float(item["distortion_score"]), int(item["group_id"])))


def select_two_geometry_groups(
    data: Macro16Arrays,
    *,
    frames_per_geometry: int,
    decimals: int,
    slight_delta: float,
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    groups = geometry_groups(data, decimals=decimals)
    if len(groups) < 2:
        raise ValueError(f"Need at least two distinct Macro16 geometries, found {len(groups)}")
    regular = groups[0]
    distorted = next(
        (row for row in groups[1:] if float(row["distortion_score"]) >= float(regular["distortion_score"]) + float(slight_delta)),
        groups[1],
    )
    selected_rows: list[int] = []
    selected_groups: list[dict[str, Any]] = []
    for role, group in (("regular", regular), ("lightly_distorted", distorted)):
        indices = list(group["frame_indices"])[: max(1, int(frames_per_geometry))]
        selected_rows.extend(indices)
        selected_groups.append(
            {
                **{k: v for k, v in group.items() if k != "frame_indices"},
                "role": role,
                "selected_frame_indices": indices,
                "selected_frame_count": int(len(indices)),
            }
        )
    return selected_rows, selected_groups, groups


def write_subset_compact(data: Macro16Arrays, selected_groups: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    selected_rows: list[int] = []
    geometry_ids: list[int] = []
    case_ids: list[int] = []
    for gid, group in enumerate(selected_groups, start=1):
        rows = [int(v) for v in group["selected_frame_indices"]]
        selected_rows.extend(rows)
        geometry_ids.extend([gid - 1] * len(rows))
        case_ids.extend([gid] * len(rows))
    idx = np.asarray(selected_rows, dtype=np.int64)
    subset_dir = out_dir / "prepared_subset"
    subset_dir.mkdir(parents=True, exist_ok=True)
    subset_path = subset_dir / "macro16_two_geometry_subset.npz"
    list_path = subset_dir / "macro16_two_geometry_subset_list.txt"
    np.savez_compressed(
        subset_path,
        standard_operator_contract_version=np.asarray(MACRO16_CONTRACT_VERSION, dtype=object),
        q48_raw=data.q48_hat[idx].astype(np.float32),
        X16=data.x16_hat[idx].astype(np.float32),
        LE_macro=data.le[idx].astype(np.float32),
        B_macro=data.b[idx].astype(np.float32),
        integration_weight_hat=data.weights[idx].astype(np.float32),
        case_id=np.asarray(case_ids, dtype=np.int64),
        geometry_id=np.asarray(geometry_ids, dtype=np.int64),
        geometry_role=np.asarray(
            [str(group["role"]) for group in selected_groups for _ in group["selected_frame_indices"]],
            dtype=object,
        ),
        source_compact_paths=np.asarray(data.compact_paths, dtype=object),
        source_index=data.source_index[idx].astype(np.int64),
        source_row=data.source_row[idx].astype(np.int64),
        prepared_from=np.asarray("scripts/run_macro16_two_geometry_smoke.py", dtype=object),
        prepared_scale_mode=np.asarray("normalized", dtype=object),
    )
    list_path.write_text(str(subset_path) + "\n", encoding="utf-8")
    return subset_path, list_path


def make_train_args(args: argparse.Namespace | SimpleNamespace, subset_list: Path, train_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        seed=int(getattr(args, "seed", 20260625)),
        out_dir=train_dir,
        compact=[],
        compact_list=str(subset_list),
        plane_gauss_order=int(getattr(args, "plane_gauss_order", 3)),
        thickness_gauss_order=int(getattr(args, "thickness_gauss_order", 2)),
        scale_mode="normalized",
        b_label_coordinate="dimensionless",
        epochs=int(getattr(args, "epochs", 20)),
        batch_size=int(getattr(args, "batch_size", 2)),
        eval_batch_size=int(getattr(args, "eval_batch_size", 2)),
        frame_stride=1,
        max_frames_per_compact=0,
        max_eval_frames=int(getattr(args, "max_eval_frames", 64)),
        basis_dim=int(getattr(args, "basis_dim", 32)),
        hidden_dim=int(getattr(args, "hidden_dim", 96)),
        branch_depth=int(getattr(args, "branch_depth", 3)),
        trunk_depth=int(getattr(args, "trunk_depth", 3)),
        activation=str(getattr(args, "activation", "tanh")),
        residual_scale=float(getattr(args, "residual_scale", 1.0)),
        fe_baseline_scale=float(getattr(args, "fe_baseline_scale", 1.0)),
        freeze_fe_point_baseline=bool(getattr(args, "freeze_fe_point_baseline", False)),
        freeze_skip=bool(getattr(args, "freeze_skip", False)),
        global_b_prior=bool(getattr(args, "global_b_prior", True)),
        anchored_residual_gate_q0=float(getattr(args, "anchored_residual_gate_q0", 0.0)),
        le_loss_weight=float(getattr(args, "le_loss_weight", 1.0)),
        jacobian_loss_weight=float(getattr(args, "jacobian_loss_weight", 1.0)),
        rigid_loss_weight=float(getattr(args, "rigid_loss_weight", 0.1)),
        rigid_mode_scale=float(getattr(args, "rigid_mode_scale", 0.1)),
        jacobian_columns=str(getattr(args, "jacobian_columns", "0,1,2,3,4,5,6,7,8,9,10,11")),
        jacobian_columns_per_batch=int(getattr(args, "jacobian_columns_per_batch", 4)),
        eval_columns=str(getattr(args, "eval_columns", "0,1,2,3,4,5,6,7,8,9,10,11")),
        lr=float(getattr(args, "lr", 8.0e-5)),
        lr_decay=float(getattr(args, "lr_decay", 0.9995)),
        weight_decay=float(getattr(args, "weight_decay", 1.0e-5)),
        grad_clip=float(getattr(args, "grad_clip", 10.0)),
        val_fraction=float(getattr(args, "val_fraction", 0.25)),
        val_cases=str(getattr(args, "val_cases", "2")),
        eval_every=int(getattr(args, "eval_every", 1)),
        cuda=bool(getattr(args, "cuda", False)),
    )


def convergence_report(training_summary: dict[str, Any], args: argparse.Namespace | SimpleNamespace) -> dict[str, Any]:
    best = training_summary.get("best_report") or {}
    latest = training_summary.get("latest_report") or {}
    max_le = float(getattr(args, "max_best_le_rel", 0.0))
    max_b = float(getattr(args, "max_best_b_rel", 0.0))
    max_val_le = float(getattr(args, "max_best_val_le_rel", 0.0))
    max_val_b = float(getattr(args, "max_best_val_b_rel", 0.0))
    checks: dict[str, Any] = {
        "best_score": training_summary.get("best_score"),
        "best_train_LE_rel": best.get("train_LE_rel"),
        "best_train_AD_B_rel": best.get("train_AD_B_rel"),
        "best_val_LE_rel": best.get("val_LE_rel"),
        "best_val_AD_B_rel": best.get("val_AD_B_rel"),
        "latest_loss": latest.get("loss"),
        "thresholds_enforced": bool(getattr(args, "require_convergence", False)),
        "max_best_le_rel": max_le,
        "max_best_b_rel": max_b,
        "max_best_val_le_rel": max_val_le,
        "max_best_val_b_rel": max_val_b,
    }
    pass_train_le = max_le <= 0.0 or float(best.get("train_LE_rel", float("inf"))) <= max_le
    pass_train_b = max_b <= 0.0 or float(best.get("train_AD_B_rel", float("inf"))) <= max_b
    pass_val_le = max_val_le <= 0.0 or float(best.get("val_LE_rel", float("inf"))) <= max_val_le
    pass_val_b = max_val_b <= 0.0 or float(best.get("val_AD_B_rel", float("inf"))) <= max_val_b
    checks["stable_convergence_pass"] = bool(pass_train_le and pass_train_b and pass_val_le and pass_val_b)
    return checks


def run_two_geometry_smoke(args: argparse.Namespace | SimpleNamespace) -> dict[str, Any]:
    out_root = Path(getattr(args, "out_root")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    point_table = macro16_standard_point_table(
        plane_order=int(getattr(args, "plane_gauss_order", 3)),
        thickness_order=int(getattr(args, "thickness_gauss_order", 2)),
    )
    data = load_macro16_compacts(
        compact_paths_from_args(args),
        point_table=point_table,
        frame_stride=int(getattr(args, "scan_frame_stride", 1)),
        max_frames_per_compact=int(getattr(args, "scan_max_frames_per_compact", 0)),
        scale_mode=str(getattr(args, "scale_mode", "normalized")),
        b_label_coordinate=str(getattr(args, "b_label_coordinate", "auto")),
    )
    selected_rows, selected_groups, all_groups = select_two_geometry_groups(
        data,
        frames_per_geometry=int(getattr(args, "frames_per_geometry", 16)),
        decimals=int(getattr(args, "geometry_round_decimals", 10)),
        slight_delta=float(getattr(args, "slight_distortion_min_delta", 1.0e-4)),
    )
    subset_path, subset_list = write_subset_compact(data, selected_groups, out_root)
    train_dir = out_root / "training"
    train_args = make_train_args(args, subset_list, train_dir)
    train_macro16_boundary(train_args)
    training_summary_path = train_dir / "training_summary.json"
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    training_config_path = train_dir / "config.json"
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    convergence = convergence_report(training_summary, args)
    summary = {
        "script": "run_macro16_two_geometry_smoke",
        "standard_operator_contract_version": MACRO16_CONTRACT_VERSION,
        "out_root": str(out_root),
        "source_compacts": data.compact_paths,
        "loaded_frames": int(data.q48_hat.shape[0]),
        "distinct_geometry_count": int(len(all_groups)),
        "selected_geometry_count": 2,
        "selected_frame_count": int(len(selected_rows)),
        "selected_groups": selected_groups,
        "all_geometry_scores": [{k: v for k, v in row.items() if k != "frame_indices"} for row in all_groups],
        "subset_compact": str(subset_path),
        "subset_compact_list": str(subset_list),
        "training_out_dir": str(train_dir),
        "training_summary": str(training_summary_path),
        "training_config": str(training_config_path),
        "training_args": vars(train_args),
        "validation_split": training_config.get("validation_split", {}),
        "validation_design": "case_id=1 regular train, case_id=2 lightly_distorted validation by default",
        "convergence": convergence,
    }
    summary_path = out_root / "macro16_two_geometry_smoke_summary.json"
    write_json(summary_path, summary)
    if bool(getattr(args, "require_convergence", False)) and not bool(convergence["stable_convergence_pass"]):
        raise RuntimeError(f"Macro16 two-geometry convergence check failed; see {summary_path}")
    print(json.dumps({"summary": str(summary_path), **convergence}, ensure_ascii=False, sort_keys=True, default=json_default))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compact", action="append", default=[])
    p.add_argument("--compact-list", default="")
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--frames-per-geometry", type=int, default=16)
    p.add_argument("--geometry-round-decimals", type=int, default=10)
    p.add_argument("--slight-distortion-min-delta", type=float, default=1.0e-4)
    p.add_argument("--scan-frame-stride", type=int, default=1)
    p.add_argument("--scan-max-frames-per-compact", type=int, default=0)
    p.add_argument("--plane-gauss-order", type=int, default=3)
    p.add_argument("--thickness-gauss-order", type=int, default=2)
    p.add_argument("--scale-mode", default="normalized", choices=["normalized", "physical"])
    p.add_argument("--b-label-coordinate", default="auto", choices=["auto", "physical", "dimensionless"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--max-eval-frames", type=int, default=64)
    p.add_argument("--basis-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--branch-depth", type=int, default=3)
    p.add_argument("--trunk-depth", type=int, default=3)
    p.add_argument("--activation", default="tanh")
    p.add_argument("--residual-scale", type=float, default=1.0)
    p.add_argument("--fe-baseline-scale", type=float, default=1.0)
    p.add_argument("--freeze-fe-point-baseline", action="store_true")
    p.add_argument("--freeze-skip", action="store_true")
    p.add_argument("--global-b-prior", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--anchored-residual-gate-q0", type=float, default=0.0)
    p.add_argument("--le-loss-weight", type=float, default=1.0)
    p.add_argument("--jacobian-loss-weight", type=float, default=1.0)
    p.add_argument("--rigid-loss-weight", type=float, default=0.1)
    p.add_argument("--rigid-mode-scale", type=float, default=0.1)
    p.add_argument("--jacobian-columns", default="0,1,2,3,4,5,6,7,8,9,10,11")
    p.add_argument("--jacobian-columns-per-batch", type=int, default=4)
    p.add_argument("--eval-columns", default="0,1,2,3,4,5,6,7,8,9,10,11")
    p.add_argument("--lr", type=float, default=8.0e-5)
    p.add_argument("--lr-decay", type=float, default=0.9995)
    p.add_argument("--weight-decay", type=float, default=1.0e-5)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260625)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--require-convergence", action="store_true")
    p.add_argument("--max-best-le-rel", type=float, default=0.0)
    p.add_argument("--max-best-b-rel", type=float, default=0.0)
    p.add_argument("--max-best-val-le-rel", type=float, default=0.0)
    p.add_argument("--max-best-val-b-rel", type=float, default=0.0)
    p.add_argument("--val-cases", default="2")
    return p.parse_args()


def main() -> None:
    run_two_geometry_smoke(parse_args())


if __name__ == "__main__":
    main()
