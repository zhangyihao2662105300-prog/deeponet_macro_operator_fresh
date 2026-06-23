#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit v2j state/regime-dependent B-prior anchors without training.

This diagnostic compares point-only global B prior with simple train-only
amplitude-regime priors.  It does not train a model, does not use old TRUE176
labels, and does not write checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import train_v2_formal_prototype as v2g


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "mode",
        "detach_regime_weight",
        "split",
        "case_id",
        "frame_count",
        "LE_local_rel",
        "LE_local_rmse",
        "LE_target_rms",
        "AD_B_local_rel",
        "AD_B_local_cos",
        "B_raw_projected_rel",
        "B_raw_rel",
        "q_amp_mean",
    ]
    ordered = [key for key in preferred if key in fields]
    ordered.extend(key for key in fields if key not in ordered)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def load_split(compact_list: Path, train_cases: list[int], val_cases: list[int]) -> tuple[list[v2g.CaseData], dict[str, np.ndarray], dict[str, np.ndarray]]:
    cases = [v2g.load_case(path) for path in v2g.read_compact_list(compact_list)]
    case_ids = {c.case_id for c in cases}
    missing = sorted(set(train_cases + val_cases).difference(case_ids))
    if missing:
        raise SystemExit(f"cases not found: {missing}")
    train_np = v2g.concat_cases([c for c in cases if c.case_id in set(train_cases)])
    val_np = v2g.concat_cases([c for c in cases if c.case_id in set(val_cases)])
    return cases, train_np, val_np


def make_model(
    *,
    train_np: dict[str, np.ndarray],
    norm: dict[str, Any],
    mode: str,
    detach: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> v2g.FormalV2Prototype:
    state = v2g.build_b_prior_state(train_np, mode)
    ip_xi = torch.as_tensor(train_np["ip_xi"], dtype=dtype, device=device)
    q_std = torch.as_tensor(norm["q_std_floor"], dtype=dtype, device=device)
    return v2g.FormalV2Prototype(
        ip_xi,
        state,
        q_std,
        q_amp_ref=float(norm["q_amp_ref"]),
        hidden=8,
        use_q_amp=False,
        use_q_dir=False,
        use_amp_regime_descriptor=False,
        b_prior_mode=mode,
        detach_b_prior_regime_weight=detach,
        gate_c=0.1,
    ).to(device=device, dtype=dtype)


def per_case_anchor_metrics(
    model: v2g.FormalV2Prototype,
    data: dict[str, torch.Tensor],
    *,
    split: str,
    q_std: torch.Tensor,
    le_std: torch.Tensor,
    frame_chunk: int,
    point_chunk: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_ids = sorted(set(int(v) for v in data["case_ids"].detach().cpu().numpy().tolist()))
    for case_id in case_ids:
        idx = torch.nonzero(data["case_ids"] == int(case_id), as_tuple=False).reshape(-1)
        subset = {key: value.index_select(0, idx) for key, value in data.items() if key != "case_ids"}
        subset["case_ids"] = data["case_ids"].index_select(0, idx)
        metrics = v2g.eval_split(
            model,
            subset,
            prefix=split,
            q_std=q_std,
            le_std=le_std,
            frame_chunk=frame_chunk,
            point_chunk=point_chunk,
            anchor_only=True,
        )
        q_amp = torch.linalg.vector_norm(subset["q"], dim=1)
        rows.append(
            {
                "split": split,
                "case_id": int(case_id),
                "frame_count": int(idx.numel()),
                "LE_local_rel": metrics[f"{split}_LE_local_rel"],
                "LE_local_rmse": metrics[f"{split}_LE_local_rmse"],
                "LE_target_rms": metrics[f"{split}_LE_target_rms"],
                "AD_B_local_rel": metrics[f"{split}_AD_B_local_rel"],
                "AD_B_local_cos": metrics[f"{split}_AD_B_local_cos"],
                "B_raw_projected_rel": metrics[f"{split}_B_model_raw_projected_rel"],
                "B_raw_rel": metrics[f"{split}_B_model_raw_rel"],
                "zero_q_LE_local_rms": metrics[f"{split}_zero_q_LE_local_rms"],
                "q_amp_mean": v2g.scalar_float(torch.mean(q_amp)),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    compact_list = Path(args.compact_list).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    train_cases = v2g.parse_cases(args.train_cases)
    val_cases = v2g.parse_cases(args.val_cases)
    cases, train_raw, val_raw = load_split(compact_list, train_cases, val_cases)
    norm = v2g.compute_normalization(train_raw)
    train_np = v2g.normalize_dataset(train_raw, norm)
    val_np = v2g.normalize_dataset(val_raw, norm)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32
    train_data = v2g.make_tensors(train_np, device=device, dtype=dtype)
    val_data = v2g.make_tensors(val_np, device=device, dtype=dtype)
    q_std = torch.as_tensor(norm["q_std_floor"], dtype=dtype, device=device)
    le_std = torch.as_tensor(norm["le_std_floor"], dtype=dtype, device=device)

    mode_specs = [
        ("global_mean", True, "global_mean"),
        ("amp_median_cluster", True, "amp_median_cluster"),
        ("amp_p75_cluster", True, "amp_p75_cluster"),
        ("amp_linear_interp_detached", True, "amp_linear_interp"),
        ("amp_linear_interp_differentiable", False, "amp_linear_interp"),
    ]
    summary_rows: list[dict[str, Any]] = []
    per_case_rows: list[dict[str, Any]] = []
    mode_summaries: dict[str, Any] = {}
    for label, detach, mode in mode_specs:
        model = make_model(train_np=train_np, norm=norm, mode=mode, detach=detach, device=device, dtype=dtype)
        state = v2g.build_b_prior_state(train_np, mode)
        train_metrics = v2g.eval_split(
            model,
            train_data,
            prefix="train",
            q_std=q_std,
            le_std=le_std,
            frame_chunk=int(args.eval_frame_batch),
            point_chunk=int(args.eval_point_batch),
            anchor_only=True,
        )
        val_metrics = v2g.eval_split(
            model,
            val_data,
            prefix="val",
            q_std=q_std,
            le_std=le_std,
            frame_chunk=int(args.eval_frame_batch),
            point_chunk=int(args.eval_point_batch),
            anchor_only=True,
        )
        mode_meta = {
            "mode": label,
            "b_prior_mode": mode,
            "detach_regime_weight": bool(detach),
            "cluster_threshold": state["cluster_threshold"],
            "cluster_threshold_kind": state["cluster_threshold_kind"],
            "cluster_low_frame_count": state["cluster_low_frame_count"],
            "cluster_high_frame_count": state["cluster_high_frame_count"],
            "amp_low_ref": state["amp_low_ref"],
            "amp_high_ref": state["amp_high_ref"],
        }
        for split, metrics in (("train", train_metrics), ("val", val_metrics)):
            summary_rows.append(
                {
                    **mode_meta,
                    "split": split,
                    "LE_local_rel": metrics[f"{split}_LE_local_rel"],
                    "LE_local_rmse": metrics[f"{split}_LE_local_rmse"],
                    "LE_target_rms": metrics[f"{split}_LE_target_rms"],
                    "AD_B_local_rel": metrics[f"{split}_AD_B_local_rel"],
                    "AD_B_local_cos": metrics[f"{split}_AD_B_local_cos"],
                    "B_raw_projected_rel": metrics[f"{split}_B_model_raw_projected_rel"],
                    "B_raw_rel": metrics[f"{split}_B_model_raw_rel"],
                    "zero_q_LE_local_rms": metrics[f"{split}_zero_q_LE_local_rms"],
                }
            )
        for row in per_case_anchor_metrics(
            model,
            train_data,
            split="train",
            q_std=q_std,
            le_std=le_std,
            frame_chunk=int(args.eval_frame_batch),
            point_chunk=int(args.eval_point_batch),
        ) + per_case_anchor_metrics(
            model,
            val_data,
            split="val",
            q_std=q_std,
            le_std=le_std,
            frame_chunk=int(args.eval_frame_batch),
            point_chunk=int(args.eval_point_batch),
        ):
            per_case_rows.append({**mode_meta, **row})
        mode_summaries[label] = {
            **mode_meta,
            "train": train_metrics,
            "val": val_metrics,
        }

    write_csv(out_root / "state_b_prior_anchor_summary.csv", summary_rows)
    write_csv(out_root / "state_b_prior_per_case_attribution.csv", per_case_rows)
    summary = {
        "audit_name": "v2j_state_dependent_b_prior_anchor_audit",
        "compact_list": str(compact_list),
        "out_root": str(out_root),
        "case_ids": sorted(c.case_id for c in cases),
        "train_cases": train_cases,
        "val_cases": val_cases,
        "mode_summaries": mode_summaries,
        "uses_old_true176_labels_as_v2_labels": False,
        "formal_training_result": False,
        "checkpoint_written": False,
        "output_files": {
            "summary": str(out_root / "state_b_prior_anchor_summary.json"),
            "summary_csv": str(out_root / "state_b_prior_anchor_summary.csv"),
            "per_case_csv": str(out_root / "state_b_prior_per_case_attribution.csv"),
        },
    }
    (out_root / "state_b_prior_anchor_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--train-cases", default="19,25,31,41,43,45,49,50")
    parser.add_argument("--val-cases", default="44,46")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-frame-batch", type=int, default=16)
    parser.add_argument("--eval-point-batch", type=int, default=16)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
