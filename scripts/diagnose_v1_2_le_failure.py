"""Diagnose v1.2 LE value-field failures without training.

The script reconstructs the generic query-point preprocessing from a saved
checkpoint, runs forward prediction for selected cases, and writes LE
decomposition reports.  It also computes train-only affine LE oracle and
B-prior offset diagnostics.  No model weights are updated.
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
from macro_deeponet.train_true176_generic_sobolev import (
    _le_scale_np,
    build_model,
    validate_point_feature_source_for_scale,
)
from macro_deeponet.true176_data import (
    build_branch_features,
    canonical_scale_mode,
    load_compacts,
    parse_int_list,
    transform_b_target_for_q_coordinate,
)

COMPONENT_NAMES = ["LE11", "LE22", "LE33", "LE12", "LE13", "LE23"]


def read_compact_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]


def rms(arr: np.ndarray) -> float:
    vals = np.asarray(arr, dtype=np.float64)
    return float(np.sqrt(np.mean(vals * vals))) if vals.size else 0.0


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return rms(np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64))


def safe_rel(pred: np.ndarray, true: np.ndarray) -> float:
    return rel_np(np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64))


def safe_cos(pred: np.ndarray, true: np.ndarray) -> float:
    return cos_np(np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64))


def scalar_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


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
        raise TypeError("B-prior diagnostics require a model with _linear_b_norm")
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


def fit_affine_oracle(q48: np.ndarray, le: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x = np.concatenate([np.ones((train_idx.size, 1), dtype=np.float64), q48[train_idx].astype(np.float64)], axis=1)
    y = le[train_idx].reshape(train_idx.size, -1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred_all = np.concatenate([np.ones((q48.shape[0], 1), dtype=np.float64), q48.astype(np.float64)], axis=1) @ coef
    svals = np.linalg.svd(x, compute_uv=False)
    meta = {
        "affine_design_rows": int(x.shape[0]),
        "affine_design_cols": int(x.shape[1]),
        "affine_design_rank": int(np.linalg.matrix_rank(x)),
        "affine_design_singular_min": float(np.min(svals)) if svals.size else 0.0,
        "affine_design_singular_max": float(np.max(svals)) if svals.size else 0.0,
        "affine_design_condition": float(np.max(svals) / max(np.min(svals), 1.0e-30)) if svals.size else 0.0,
    }
    return coef, pred_all.reshape(le.shape).astype(np.float64), meta


def frame_alpha(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_norm = np.linalg.norm(np.asarray(q, dtype=np.float64), axis=1)
    scale = float(np.max(q_norm)) if q_norm.size else 0.0
    alpha = q_norm / max(scale, 1.0e-30)
    return q_norm, alpha


def component_rows(case_id: int, le_pred: np.ndarray, le_true: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comp in range(le_true.shape[-1]):
        pred_c = le_pred[:, :, comp]
        true_c = le_true[:, :, comp]
        rows.append(
            {
                "case_id": int(case_id),
                "component_index": int(comp),
                "component_name": COMPONENT_NAMES[comp] if comp < len(COMPONENT_NAMES) else f"LE{comp}",
                "target_rms": rms(true_c),
                "pred_rms": rms(pred_c),
                "abs_rmse": rmse(pred_c, true_c),
                "rel": safe_rel(pred_c, true_c),
                "cosine": safe_cos(pred_c, true_c),
                "mean_error": float(np.mean(pred_c - true_c)),
            }
        )
    return rows


def ip_rows(case_id: int, le_pred: np.ndarray, le_true: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ip in range(le_true.shape[1]):
        pred_ip = le_pred[:, ip, :]
        true_ip = le_true[:, ip, :]
        comp_rel = [safe_rel(pred_ip[:, c], true_ip[:, c]) for c in range(le_true.shape[-1])]
        worst_comp = int(np.argmax(np.asarray(comp_rel, dtype=np.float64)))
        rows.append(
            {
                "case_id": int(case_id),
                "ip_local": int(ip),
                "IP_LE_rel": safe_rel(pred_ip, true_ip),
                "IP_abs_rmse": rmse(pred_ip, true_ip),
                "worst_component_index": worst_comp,
                "worst_component_name": COMPONENT_NAMES[worst_comp] if worst_comp < len(COMPONENT_NAMES) else f"LE{worst_comp}",
                "worst_component_rel": float(comp_rel[worst_comp]),
            }
        )
    rows.sort(key=lambda row: float(row["IP_LE_rel"]), reverse=True)
    return rows


def frame_rows(case_id: int, indices: np.ndarray, le_pred: np.ndarray, le_true: np.ndarray, q48: np.ndarray) -> list[dict[str, Any]]:
    q_norm, alpha = frame_alpha(q48)
    rows: list[dict[str, Any]] = []
    for local, idx in enumerate(indices):
        rows.append(
            {
                "case_id": int(case_id),
                "frame_index": int(local),
                "global_frame_index": int(idx),
                "alpha": float(alpha[local]),
                "q_norm": float(q_norm[local]),
                "target_LE_rms": rms(le_true[local]),
                "pred_LE_rms": rms(le_pred[local]),
                "abs_rmse": rmse(le_pred[local], le_true[local]),
                "LE_rel": safe_rel(le_pred[local], le_true[local]),
                "LE_cosine": safe_cos(le_pred[local], le_true[local]),
                "mean_error": float(np.mean(le_pred[local] - le_true[local])),
            }
        )
    return rows


def case_summary(
    *,
    case_id: int,
    indices: np.ndarray,
    le_pred: np.ndarray,
    le_true: np.ndarray,
    train_mean: np.ndarray,
    affine_pred: np.ndarray,
    q48: np.ndarray,
    b_true: np.ndarray,
    b_prior_raw: np.ndarray,
    train_offset_mean: np.ndarray,
) -> dict[str, Any]:
    err = le_pred - le_true
    zero = np.zeros_like(le_true)
    mean_pred = np.broadcast_to(train_mean, le_true.shape)
    pred_flat = le_pred.reshape(-1)
    true_flat = le_true.reshape(-1)
    denom = float(np.dot(pred_flat, pred_flat))
    optimal_scale = float(np.dot(pred_flat, true_flat) / denom) if denom > 1.0e-30 else 0.0
    scaled = optimal_scale * le_pred
    comp_bias = np.mean(err, axis=(0, 1), keepdims=True)
    bias_corrected = le_pred - comp_bias
    x = np.stack([pred_flat, np.ones_like(pred_flat)], axis=1)
    ab, *_ = np.linalg.lstsq(x, true_flat, rcond=None)
    scale_bias = float(ab[0]) * le_pred + float(ab[1])

    true_bq = np.einsum("npck,nk->npc", b_true, q48)
    offset_true = le_true - true_bq
    prior_bq = np.einsum("npck,nk->npc", b_prior_raw, q48)
    offset_pred = le_pred - prior_bq
    offset_delta = offset_pred - offset_true
    offset_center_delta = offset_true.reshape(offset_true.shape[0], -1) - train_offset_mean.reshape(1, -1)
    offset_nearest_dist = float(np.min(np.linalg.norm(offset_center_delta, axis=1))) if offset_center_delta.size else 0.0

    frame_rel = np.asarray([safe_rel(le_pred[i], le_true[i]) for i in range(le_true.shape[0])], dtype=np.float64)
    q_norm, alpha = frame_alpha(q48)
    alpha_corr = float(np.corrcoef(alpha, frame_rel)[0, 1]) if alpha.size > 1 and np.std(alpha) > 0 and np.std(frame_rel) > 0 else 0.0
    low_mask = alpha <= 0.3
    high_mask = alpha >= 0.8

    return {
        "case_id": int(case_id),
        "frame_count": int(indices.size),
        "LE_target_rms": rms(le_true),
        "LE_pred_rms": rms(le_pred),
        "LE_abs_rmse": rmse(le_pred, le_true),
        "LE_rel": safe_rel(le_pred, le_true),
        "LE_cosine": safe_cos(le_pred, le_true),
        "LE_mean_target": float(np.mean(le_true)),
        "LE_mean_pred": float(np.mean(le_pred)),
        "LE_mean_error": float(np.mean(err)),
        "LE_std_target": float(np.std(le_true)),
        "LE_std_pred": float(np.std(le_pred)),
        "zero_pred_LE_rel": safe_rel(zero, le_true),
        "train_mean_LE_rel": safe_rel(mean_pred, le_true),
        "train_mean_LE_abs_rmse": rmse(mean_pred, le_true),
        "optimal_scale_a": optimal_scale,
        "scaled_LE_rel": safe_rel(scaled, le_true),
        "bias_component_mean_abs": float(np.mean(np.abs(comp_bias))),
        "bias_corrected_LE_rel": safe_rel(bias_corrected, le_true),
        "scale_bias_a": float(ab[0]),
        "scale_bias_b": float(ab[1]),
        "scale_bias_corrected_LE_rel": safe_rel(scale_bias, le_true),
        "affine_oracle_LE_rel": safe_rel(affine_pred, le_true),
        "affine_oracle_abs_rmse": rmse(affine_pred, le_true),
        "offset_true_rms": rms(offset_true),
        "offset_true_mean": float(np.mean(offset_true)),
        "offset_true_std": float(np.std(offset_true)),
        "offset_pred_rms": rms(offset_pred),
        "offset_pred_mean": float(np.mean(offset_pred)),
        "offset_pred_std": float(np.std(offset_pred)),
        "pred_offset_vs_true_offset_rel": safe_rel(offset_pred, offset_true),
        "offset_nearest_train_mean_l2": offset_nearest_dist,
        "alpha_min": float(np.min(alpha)) if alpha.size else 0.0,
        "alpha_max": float(np.max(alpha)) if alpha.size else 0.0,
        "LE_rel_alpha_correlation": alpha_corr,
        "LE_rel_low_alpha_mean": float(np.mean(frame_rel[low_mask])) if np.any(low_mask) else None,
        "LE_rel_high_alpha_mean": float(np.mean(frame_rel[high_mask])) if np.any(high_mask) else None,
        "model_worse_than_zero": bool(safe_rel(le_pred, le_true) > safe_rel(zero, le_true)),
        "model_worse_than_train_mean": bool(safe_rel(le_pred, le_true) > safe_rel(mean_pred, le_true)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: scalar_json(row.get(key)) for key in fields})


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).resolve()
    compact_list = Path(args.compact_list).resolve()
    out_root = Path(args.out_root).resolve() / str(args.run_label)
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    train_args = dict(ckpt.get("args", {}))
    ns = SimpleNamespace(**train_args)
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

    x_norm = ((x_raw - norms["x_mean"]) / norms["x_std"]).astype(np.float32)
    point_norm = ((point_raw - norms["point_mean"].reshape(1, 1, -1)) / norms["point_std"].reshape(1, 1, -1)).astype(np.float32)
    q_start = int(np.asarray(norms.get("q_start", 4)).reshape(-1)[0])
    q_dim = int(np.asarray(norms.get("q_dim", 48)).reshape(-1)[0])
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
    b_prior_raw_all = predict_b_prior_raw(model, point_norm, norms["le_std"].astype(np.float32), q_std, device, batch_size)

    split = ckpt.get("validation_split", {})
    train_cases = [int(v) for v in split.get("train_cases", [])]
    if not train_cases:
        val_cases_ckpt = {int(v) for v in split.get("val_cases", [])}
        train_cases = [int(v) for v in np.unique(data.case_id).tolist() if int(v) not in val_cases_ckpt]
    train_idx = np.flatnonzero(np.isin(data.case_id, np.asarray(train_cases, dtype=np.int64))).astype(np.int64)
    train_mean = np.mean(data.le[train_idx].astype(np.float64), axis=0, keepdims=True)
    _coef, affine_pred_all, affine_meta = fit_affine_oracle(data.q48_raw.astype(np.float64), data.le.astype(np.float64), train_idx)
    train_offset = data.le[train_idx].astype(np.float64) - np.einsum("npck,nk->npc", b_train[train_idx].astype(np.float64), data.q48_raw[train_idx].astype(np.float64))
    train_offset_mean = np.mean(train_offset.reshape(train_offset.shape[0], -1), axis=0)

    cases = parse_int_list(str(args.cases), default=[])
    if not cases:
        cases = [int(v) for v in split.get("val_cases", [])]
    if not cases:
        raise ValueError("provide --cases or use a checkpoint with validation_split.val_cases")

    offset_by_case: list[dict[str, Any]] = []
    val_case_set = {int(v) for v in cases}
    train_case_set = {int(v) for v in train_cases}
    for cid in sorted(int(v) for v in np.unique(data.case_id).tolist()):
        idx = np.flatnonzero(data.case_id.astype(np.int64) == int(cid)).astype(np.int64)
        q_case_all = data.q48_raw[idx].astype(np.float64)
        le_case_all = data.le[idx].astype(np.float64)
        true_bq_all = np.einsum("npck,nk->npc", b_train[idx].astype(np.float64), q_case_all)
        offset_all = le_case_all - true_bq_all
        q_norm = np.linalg.norm(q_case_all, axis=1)
        role = "val_selected" if cid in val_case_set else ("train" if cid in train_case_set else "other")
        offset_by_case.append(
            {
                "case_id": int(cid),
                "role": role,
                "frame_count": int(idx.size),
                "q_norm_mean": float(np.mean(q_norm)) if q_norm.size else 0.0,
                "q_norm_max": float(np.max(q_norm)) if q_norm.size else 0.0,
                "LE_rms": rms(le_case_all),
                "offset_true_rms": rms(offset_all),
                "offset_true_mean": float(np.mean(offset_all)),
                "offset_true_std": float(np.std(offset_all)),
                "offset_true_max_abs": float(np.max(np.abs(offset_all))) if offset_all.size else 0.0,
            }
        )

    case_rows: list[dict[str, Any]] = []
    component_all: list[dict[str, Any]] = []
    worst_ip_all: list[dict[str, Any]] = []
    frame_all: list[dict[str, Any]] = []
    offset_rows: list[dict[str, Any]] = []
    for case_id in cases:
        indices = np.flatnonzero(data.case_id.astype(np.int64) == int(case_id)).astype(np.int64)
        if indices.size == 0:
            raise ValueError(f"case {case_id} selected no frames")
        le_true = data.le[indices].astype(np.float64)
        le_pred = le_pred_all[indices].astype(np.float64)
        q_case = data.q48_raw[indices].astype(np.float64)
        b_case = b_train[indices].astype(np.float64)
        b_prior_case = b_prior_raw_all[indices].astype(np.float64)
        affine_case = affine_pred_all[indices].astype(np.float64)

        row = case_summary(
            case_id=int(case_id),
            indices=indices,
            le_pred=le_pred,
            le_true=le_true,
            train_mean=train_mean,
            affine_pred=affine_case,
            q48=q_case,
            b_true=b_case,
            b_prior_raw=b_prior_case,
            train_offset_mean=train_offset_mean,
        )
        case_rows.append(row)
        component_all.extend(component_rows(int(case_id), le_pred, le_true))
        worst_ip_all.extend(ip_rows(int(case_id), le_pred, le_true)[:10])
        frame_case = frame_rows(int(case_id), indices, le_pred, le_true, q_case)
        frame_all.extend(frame_case)
        write_csv(out_root / f"le_frame_trend_case{int(case_id):03d}.csv", frame_case)
        offset_rows.append(
            {
                "case_id": int(case_id),
                "offset_true_rms": row["offset_true_rms"],
                "offset_true_mean": row["offset_true_mean"],
                "offset_true_std": row["offset_true_std"],
                "offset_pred_rms": row["offset_pred_rms"],
                "offset_pred_mean": row["offset_pred_mean"],
                "offset_pred_std": row["offset_pred_std"],
                "pred_offset_vs_true_offset_rel": row["pred_offset_vs_true_offset_rel"],
                "offset_nearest_train_mean_l2": row["offset_nearest_train_mean_l2"],
            }
        )

    write_csv(out_root / "le_case_summary.csv", case_rows)
    write_csv(out_root / "le_component_summary.csv", component_all)
    write_csv(out_root / "le_worst_ip_summary.csv", worst_ip_all)
    write_csv(out_root / "le_frame_trend_all_cases.csv", frame_all)
    write_csv(out_root / "le_offset_summary.csv", offset_rows)
    write_csv(out_root / "le_offset_by_case.csv", offset_by_case)
    report = {
        "audit_name": "v1_2_le_failure_diagnostic",
        "run_label": str(args.run_label),
        "checkpoint": str(checkpoint_path),
        "compact_list": str(compact_list),
        "cases": cases,
        "train_cases": train_cases,
        "validation_split": split,
        "case_summary": case_rows,
        "affine_meta": affine_meta,
        "outputs": [
            "le_case_summary.csv",
            "le_component_summary.csv",
            "le_worst_ip_summary.csv",
            "le_frame_trend_all_cases.csv",
            "le_offset_summary.csv",
            "le_offset_by_case.csv",
        ],
        "no_training_performed": True,
    }
    (out_root / "le_failure_diagnostic_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
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
