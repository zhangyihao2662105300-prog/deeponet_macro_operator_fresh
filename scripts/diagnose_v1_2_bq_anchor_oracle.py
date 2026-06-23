"""Diagnose whether B@q anchors LE better than the learned LE head.

This is an evaluation-only v1.2 diagnostic.  It compares true B@q, query-B-prior
B@q, model LE prediction, q=0 model output, and residual/value offsets for
selected held-out cases.  It does not train or update model weights.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from macro_deeponet.point_features import load_point_features_from_compacts, transform_point_features_for_scale
from macro_deeponet.train_true176_deeponet_sobolev import cos_np, rel_np
from macro_deeponet.train_true176_generic_sobolev import _le_scale_np, build_model, validate_point_feature_source_for_scale
from macro_deeponet.true176_data import (
    build_branch_features,
    canonical_scale_mode,
    load_compacts,
    parse_int_list,
    transform_b_target_for_q_coordinate,
)


def read_compact_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]


def rms(arr: np.ndarray) -> float:
    vals = np.asarray(arr, dtype=np.float64)
    return float(np.sqrt(np.mean(vals * vals))) if vals.size else 0.0


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return rms(np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def predict_le(model: torch.nn.Module, x_norm: np.ndarray, point_norm: np.ndarray, norms: dict[str, np.ndarray], device: torch.device, batch_size: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    model.eval()
    for start in range(0, int(x_norm.shape[0]), int(batch_size)):
        xb = torch.as_tensor(x_norm[start : start + batch_size], dtype=torch.float32, device=device)
        pb = torch.as_tensor(point_norm[start : start + batch_size], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_norm = model(xb, pb).detach().cpu().numpy()
        rows.append(pred_norm * norms["le_std"] + norms["le_mean"])
    return np.concatenate(rows, axis=0).astype(np.float64)


def predict_b_prior_raw(model: torch.nn.Module, point_norm: np.ndarray, le_std: np.ndarray, q_std: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    if not hasattr(model, "_linear_b_norm"):
        raise TypeError("B-prior oracle requires a model with _linear_b_norm")
    rows: list[np.ndarray] = []
    model.eval()
    for start in range(0, int(point_norm.shape[0]), int(batch_size)):
        pb = torch.as_tensor(point_norm[start : start + batch_size], dtype=torch.float32, device=device)
        with torch.no_grad():
            b_norm = model._linear_b_norm(pb).detach().cpu().numpy().astype(np.float64)
        rows.append(b_norm)
    b_norm_all = np.concatenate(rows, axis=0)
    le_scale = _le_scale_np(np.asarray(le_std, dtype=np.float32), int(b_norm_all.shape[1])).astype(np.float64)
    q_scale = np.maximum(np.asarray(q_std, dtype=np.float64).reshape(1, 1, 1, -1), 1.0e-12)
    return b_norm_all * le_scale / q_scale


def q_zero_branch(x_norm: np.ndarray, norms: dict[str, np.ndarray]) -> np.ndarray:
    out = np.array(x_norm, copy=True)
    q_start = int(np.asarray(norms.get("q_start", 4)).reshape(-1)[0])
    q_dim = int(np.asarray(norms.get("q_dim", 48)).reshape(-1)[0])
    x_mean = np.asarray(norms["x_mean"], dtype=np.float64).reshape(-1)
    x_std = np.asarray(norms["x_std"], dtype=np.float64).reshape(-1)
    # Physical/raw q=0 corresponds to normalized (0 - mean) / std.
    out[:, q_start : q_start + q_dim] = ((0.0 - x_mean[q_start : q_start + q_dim]) / x_std[q_start : q_start + q_dim]).astype(np.float32)
    return out.astype(np.float32)


def frame_alpha(q48: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_norm = np.linalg.norm(np.asarray(q48, dtype=np.float64), axis=1)
    alpha = q_norm / max(float(np.max(q_norm)) if q_norm.size else 0.0, 1.0e-30)
    return q_norm, alpha


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).resolve()
    compact_list = Path(args.compact_list).resolve()
    out_root = Path(args.out_root).resolve() / str(args.run_label)
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    ns = SimpleNamespace(**dict(ckpt.get("args", {})))
    norms = {str(k): np.asarray(v) for k, v in ckpt["norms"].items()}
    compact_paths = read_compact_list(compact_list)
    target_ips = [int(v) for v in ckpt.get("target_ips", parse_int_list(str(getattr(ns, "target_ips", "")), default=list(range(128))))]
    scale_mode = canonical_scale_mode(str(getattr(ns, "scale_mode", "normalized")))

    data = load_compacts(
        compact_paths,
        frame_stride=int(getattr(ns, "frame_stride", 1)),
        max_frames_per_compact=int(getattr(ns, "max_frames_per_compact", 0)),
        target_ips=target_ips,
    )
    x_raw, _branch_meta = build_branch_features(
        shape4=data.shape4,
        q48_raw=data.q48_raw,
        mode=str(getattr(ns, "branch_feature_mode", "xkeep-qraw")),
        keep_node_coords=data.keep_node_coords,
        macro_nodes=data.macro_nodes,
        length_scale=data.length_scale,
        scale_mode=scale_mode,
    )
    b_train, _b_scale_meta = transform_b_target_for_q_coordinate(
        data.b.astype(np.float32, copy=False),
        data.length_scale,
        scale_mode=scale_mode,
        b_label_coordinate=str(getattr(ns, "b_label_coordinate", "auto")),
    )
    point_raw, point_meta = load_point_features_from_compacts(
        compact_paths=data.compact_paths,
        source_index=data.source_index,
        source_row=data.source_row,
        shape4=data.shape4,
        target_ips=target_ips,
        source=str(getattr(ns, "point_feature_source", "data")),
        include_id_features=bool(getattr(ns, "include_id_features", False)),
        allow_shape4_fallback=bool(getattr(ns, "allow_shape4_point_feature_fallback", False)),
    )
    validate_point_feature_source_for_scale(
        scale_mode=scale_mode,
        requested_source=str(getattr(ns, "point_feature_source", "data")),
        point_meta=point_meta,
        allow_physical_shape4_trunk=bool(getattr(ns, "allow_physical_shape4_trunk", False)),
    )
    point_feature_names = list(point_meta.get("feature_names", [f"point_feature_{i}" for i in range(point_raw.shape[-1])]))
    point_raw, _point_scale_meta = transform_point_features_for_scale(
        point_raw,
        point_feature_names,
        data.length_scale,
        scale_mode=scale_mode,
        detj_scale_dim=int(getattr(ns, "detj_scale_dim", 3)),
    )

    q_start = int(np.asarray(norms.get("q_start", 4)).reshape(-1)[0])
    q_dim = int(np.asarray(norms.get("q_dim", 48)).reshape(-1)[0])
    q_coord = x_raw[:, q_start : q_start + q_dim].astype(np.float64)
    x_norm = ((x_raw - norms["x_mean"]) / norms["x_std"]).astype(np.float32)
    point_norm = ((point_raw - norms["point_mean"].reshape(1, 1, -1)) / norms["point_std"].reshape(1, 1, -1)).astype(np.float32)
    q_std = norms["x_std"].reshape(-1)[q_start : q_start + q_dim].astype(np.float32)

    device = torch.device("cuda" if bool(args.cuda) and torch.cuda.is_available() else "cpu")
    model = build_model(
        ns,
        input_dim=int(x_norm.shape[-1]),
        point_dim=int(point_norm.shape[-1]),
        ip_count=len(target_ips),
        q_start=q_start,
        q_dim=q_dim,
        skip_init=np.zeros((len(target_ips), 6, 48), dtype=np.float32),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    batch_size = max(1, int(getattr(ns, "eval_batch_size", 1)))
    le_pred_all = predict_le(model, x_norm, point_norm, norms, device, batch_size)
    le_zero_q_all = predict_le(model, q_zero_branch(x_norm, norms), point_norm, norms, device, batch_size)
    b_prior_raw_all = predict_b_prior_raw(model, point_norm, norms["le_std"].astype(np.float32), q_std, device, batch_size)

    cases = parse_int_list(str(args.cases), default=[])
    if not cases:
        cases = [int(v) for v in ckpt.get("validation_split", {}).get("val_cases", [])]
    if not cases:
        raise ValueError("provide --cases or use a checkpoint with validation_split.val_cases")

    case_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    zero_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    for case_id in cases:
        indices = np.flatnonzero(data.case_id.astype(np.int64) == int(case_id)).astype(np.int64)
        if indices.size == 0:
            raise ValueError(f"case {case_id} selected no frames")
        le_true = data.le[indices].astype(np.float64)
        q_case = q_coord[indices].astype(np.float64)
        b_true = b_train[indices].astype(np.float64)
        b_prior = b_prior_raw_all[indices].astype(np.float64)
        le_model = le_pred_all[indices].astype(np.float64)
        le_zero_q = le_zero_q_all[indices].astype(np.float64)
        le_btrue = np.einsum("npck,nk->npc", b_true, q_case)
        le_bprior = np.einsum("npck,nk->npc", b_prior, q_case)
        current_offset = le_model - le_bprior
        true_offset = le_true - le_btrue
        bprior_offset = le_true - le_bprior
        target_rms = rms(le_true)
        model_rel = rel_np(le_model, le_true)
        bprior_rel = rel_np(le_bprior, le_true)
        improve = float(model_rel / max(bprior_rel, 1.0e-30))
        row = {
            "split_id": str(args.run_label),
            "case_id": int(case_id),
            "frame_count": int(indices.size),
            "target_rms": target_rms,
            "model_pred_rms": rms(le_model),
            "model_LE_rel": model_rel,
            "model_abs_rmse": rmse(le_model, le_true),
            "model_LE_cosine": cos_np(le_model, le_true),
            "zero_pred_LE_rel": rel_np(np.zeros_like(le_true), le_true),
            "Btrue_q_pred_rms": rms(le_btrue),
            "Btrue_q_LE_rel": rel_np(le_btrue, le_true),
            "Btrue_q_abs_rmse": rmse(le_btrue, le_true),
            "Btrue_q_cosine": cos_np(le_btrue, le_true),
            "Bprior_q_pred_rms": rms(le_bprior),
            "Bprior_q_LE_rel": bprior_rel,
            "Bprior_q_abs_rmse": rmse(le_bprior, le_true),
            "Bprior_q_cosine": cos_np(le_bprior, le_true),
            "true_offset_rms": rms(true_offset),
            "Bprior_offset_rms": rms(bprior_offset),
            "current_offset_rms": rms(current_offset),
            "current_offset_mean": float(np.mean(current_offset)),
            "current_offset_std": float(np.std(current_offset)),
            "current_offset_cosine_with_LE_true": cos_np(current_offset, le_true),
            "current_offset_cosine_with_LE_Bprior": cos_np(current_offset, le_bprior),
            "zero_q_pred_rms": rms(le_zero_q),
            "zero_q_pred_max_abs": float(np.max(np.abs(le_zero_q))) if le_zero_q.size else 0.0,
            "model_vs_Bprior_improvement_factor": improve,
            "Bprior_better_than_model": bool(bprior_rel < model_rel),
        }
        case_rows.append(row)
        comp_rms = np.sqrt(np.mean(le_zero_q * le_zero_q, axis=(0, 1)))
        for comp_idx, val in enumerate(comp_rms):
            zero_rows.append(
                {
                    "split_id": str(args.run_label),
                    "case_id": int(case_id),
                    "component_index": int(comp_idx),
                    "zero_q_component_rms": float(val),
                    "zero_q_pred_rms": row["zero_q_pred_rms"],
                    "zero_q_pred_max_abs": row["zero_q_pred_max_abs"],
                }
            )
        q_norm, alpha = frame_alpha(q_case)
        frame_offset_rms = np.sqrt(np.mean(current_offset * current_offset, axis=(1, 2)))
        for local, idx in enumerate(indices):
            frame_rows.append(
                {
                    "split_id": str(args.run_label),
                    "case_id": int(case_id),
                    "frame_index": int(local),
                    "global_frame_index": int(idx),
                    "alpha": float(alpha[local]),
                    "q_norm": float(q_norm[local]),
                    "target_rms": rms(le_true[local]),
                    "model_pred_rms": rms(le_model[local]),
                    "Btrue_q_pred_rms": rms(le_btrue[local]),
                    "Bprior_q_pred_rms": rms(le_bprior[local]),
                    "model_LE_rel": rel_np(le_model[local], le_true[local]),
                    "Btrue_q_LE_rel": rel_np(le_btrue[local], le_true[local]),
                    "Bprior_q_LE_rel": rel_np(le_bprior[local], le_true[local]),
                    "current_offset_rms": float(frame_offset_rms[local]),
                    "zero_q_pred_rms": rms(le_zero_q[local]),
                }
            )
        residual_rows.append(
            {
                "split_id": str(args.run_label),
                "case_id": int(case_id),
                "current_offset_rms": row["current_offset_rms"],
                "current_offset_mean": row["current_offset_mean"],
                "current_offset_std": row["current_offset_std"],
                "current_offset_cosine_with_LE_true": row["current_offset_cosine_with_LE_true"],
                "current_offset_cosine_with_LE_Bprior": row["current_offset_cosine_with_LE_Bprior"],
                "current_offset_alpha_corr": float(np.corrcoef(alpha, frame_offset_rms)[0, 1])
                if alpha.size > 1 and np.std(alpha) > 0 and np.std(frame_offset_rms) > 0
                else 0.0,
            }
        )

    write_csv(out_root / "bq_anchor_case_summary.csv", case_rows)
    write_csv(out_root / "bq_anchor_frame_summary.csv", frame_rows)
    write_csv(out_root / "zero_q_anchor_summary.csv", zero_rows)
    write_csv(out_root / "residual_offset_summary.csv", residual_rows)
    report = {
        "audit_name": "v1_2_bq_anchor_oracle_diagnostic",
        "run_label": str(args.run_label),
        "checkpoint": str(checkpoint_path),
        "compact_list": str(compact_list),
        "cases": cases,
        "validation_split": ckpt.get("validation_split"),
        "case_summary": case_rows,
        "outputs": [
            "bq_anchor_case_summary.csv",
            "bq_anchor_frame_summary.csv",
            "zero_q_anchor_summary.csv",
            "residual_offset_summary.csv",
        ],
        "no_training_performed": True,
    }
    (out_root / "bq_anchor_diagnostic_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"run_label": str(args.run_label), "cases": cases, "no_training_performed": True}, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--compact-list", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--run-label", required=True)
    p.add_argument("--cases", default="")
    p.add_argument("--cuda", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
