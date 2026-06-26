#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose the Macro16 network input and output contract.

This is a read-only diagnostic. It does not train a network, change the model,
change q48/LE ordering, or change the 128-point rule.
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

from audit_macro16_trained_force_closure import make_model, torch_load_cross_platform  # noqa: E402
from macro_deeponet.macro16_geometry import macro16_source128_point_table, macro16_standard_point_table  # noqa: E402
from macro_deeponet.train_macro16_boundary_sobolev import (  # noqa: E402
    _le_std_scale,
    ad_jacobian,
    load_macro16_compacts,
    parse_int_list,
    read_path_list,
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
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def rel_np(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(aa - bb) / max(float(np.linalg.norm(bb)), 1.0e-300))


def max_abs_np(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(aa - bb))) if aa.size else 0.0


def metric(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    return {"rel": rel_np(a, b), "max_abs": max_abs_np(a, b)}


def select_indices(case_id: np.ndarray, case_list: str, max_frames: int) -> np.ndarray:
    idx = np.arange(case_id.shape[0], dtype=np.int64)
    if str(case_list).strip():
        cases = np.asarray(parse_int_list(str(case_list)), dtype=np.int64)
        idx = idx[np.isin(case_id[idx], cases)]
    if int(max_frames) > 0:
        idx = idx[: int(max_frames)]
    if idx.size == 0:
        raise ValueError("no frames selected")
    return idx


def compact_field_checks(data: Any, indices: np.ndarray) -> dict[str, Any]:
    source128 = macro16_source128_point_table()
    q_rows: list[np.ndarray] = []
    q_loader: list[np.ndarray] = []
    le_rows: list[np.ndarray] = []
    le_loader: list[np.ndarray] = []
    b_rows: list[np.ndarray] = []
    b_loader: list[np.ndarray] = []
    x_rows: list[np.ndarray] = []
    x_loader: list[np.ndarray] = []
    xi_checks: list[dict[str, Any]] = []

    for source_id in sorted(np.unique(data.source_index[indices]).astype(int).tolist()):
        local = indices[np.where(data.source_index[indices] == source_id)[0]]
        rows = data.source_row[local]
        path = Path(data.compact_paths[source_id])
        with np.load(str(path), allow_pickle=True) as z:
            q_rows.append(np.asarray(z["q48_def_hat"], dtype=np.float64)[rows].reshape(rows.size, 48))
            q_loader.append(np.asarray(data.q48_hat[local], dtype=np.float64).reshape(rows.size, 48))
            le_rows.append(np.asarray(z["LE_macro"], dtype=np.float64)[rows])
            le_loader.append(np.asarray(data.le[local], dtype=np.float64))
            b_rows.append(np.asarray(z["B_macro_qdef"], dtype=np.float64)[rows])
            b_loader.append(np.asarray(data.b[local], dtype=np.float64))
            if "X16_hat" in z.files:
                x_rows.append(np.asarray(z["X16_hat"], dtype=np.float64)[rows])
                x_loader.append(np.asarray(data.x16_hat[local], dtype=np.float64))
            xi = np.asarray(z["macro16_point_xi"], dtype=np.float64) if "macro16_point_xi" in z.files else np.empty((0, 3))
            xi_checks.append(
                {
                    "compact": str(path),
                    "macro16_point_xi_shape": list(xi.shape),
                    "matches_source128": bool(xi.shape == source128.xi.shape and np.allclose(xi, source128.xi, rtol=1.0e-7, atol=1.0e-7)),
                    "max_abs_vs_source128": float(np.max(np.abs(xi - source128.xi))) if xi.shape == source128.xi.shape else None,
                }
            )

    out: dict[str, Any] = {
        "loader_q_vs_compact_q48_def_hat": metric(np.concatenate(q_loader, axis=0), np.concatenate(q_rows, axis=0)),
        "loader_LE_vs_compact_LE_macro": metric(np.concatenate(le_loader, axis=0), np.concatenate(le_rows, axis=0)),
        "loader_B_vs_compact_B_macro_qdef": metric(np.concatenate(b_loader, axis=0), np.concatenate(b_rows, axis=0)),
        "all_macro16_point_xi_match_source128": bool(all(row["matches_source128"] for row in xi_checks)),
        "macro16_point_xi_checks": xi_checks,
    }
    if x_rows:
        out["loader_X16_hat_vs_compact_X16_hat"] = metric(np.concatenate(x_loader, axis=0), np.concatenate(x_rows, axis=0))
    return out


def top_feature_rows(
    values: np.ndarray,
    raw: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    names: list[str],
    case_id: np.ndarray,
    indices: np.ndarray,
    *,
    count: int = 12,
) -> list[dict[str, Any]]:
    vals = np.asarray(values, dtype=np.float64)
    raw_vals = np.asarray(raw, dtype=np.float64)
    flat = np.abs(vals[indices]).reshape(-1)
    if flat.size == 0:
        return []
    take = min(int(count), int(flat.size))
    picked = np.argpartition(-flat, np.arange(take))[:take]
    picked = picked[np.argsort(-flat[picked])]
    rows: list[dict[str, Any]] = []
    point_count = vals.shape[1] if vals.ndim == 3 else 1
    feature_count = vals.shape[-1]
    for flat_idx in picked.tolist():
        local_frame = flat_idx // (point_count * feature_count)
        rem = flat_idx % (point_count * feature_count)
        point = rem // feature_count
        feature = rem % feature_count
        frame = int(indices[local_frame])
        rows.append(
            {
                "frame_index": frame,
                "case_id": int(case_id[frame]),
                "point": int(point),
                "feature": int(feature),
                "name": names[feature] if feature < len(names) else f"feature_{feature}",
                "z": float(vals[frame, point, feature]) if vals.ndim == 3 else float(vals[frame, feature]),
                "raw": float(raw_vals[frame, point, feature]) if raw_vals.ndim == 3 else float(raw_vals[frame, feature]),
                "mean": float(np.asarray(mean).reshape(-1)[feature]),
                "std": float(np.asarray(std).reshape(-1)[feature]),
            }
        )
    return rows


def top_branch_rows(
    values: np.ndarray,
    raw: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    case_id: np.ndarray,
    indices: np.ndarray,
    *,
    count: int = 12,
) -> list[dict[str, Any]]:
    names = [f"q48_def_hat_{i}" for i in range(48)]
    names.extend([f"X16_hat_{node}_{axis}" for node in range(16) for axis in ("x", "y", "z")])
    names.append("L_ref")
    vals = np.asarray(values, dtype=np.float64)
    flat = np.abs(vals[indices]).reshape(-1)
    if flat.size == 0:
        return []
    take = min(int(count), int(flat.size))
    picked = np.argpartition(-flat, np.arange(take))[:take]
    picked = picked[np.argsort(-flat[picked])]
    rows: list[dict[str, Any]] = []
    feature_count = vals.shape[-1]
    for flat_idx in picked.tolist():
        local_frame = flat_idx // feature_count
        feature = flat_idx % feature_count
        frame = int(indices[local_frame])
        rows.append(
            {
                "frame_index": frame,
                "case_id": int(case_id[frame]),
                "feature": int(feature),
                "name": names[feature] if feature < len(names) else f"branch_{feature}",
                "z": float(vals[frame, feature]),
                "raw": float(raw[frame, feature]),
                "mean": float(mean.reshape(-1)[feature]),
                "std": float(std.reshape(-1)[feature]),
            }
        )
    return rows


def checkpoint_io_checks(
    checkpoint_path: Path,
    data: Any,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    checkpoint = torch_load_cross_platform(checkpoint_path, device)
    args = checkpoint.get("args", {})
    norms = {key: np.asarray(value) for key, value in checkpoint["norms"].items()}
    branch_raw = np.concatenate([data.q48_hat, data.x16_hat.reshape(data.x16_hat.shape[0], -1), data.length_scale], axis=1)
    branch_norm = ((branch_raw - norms["branch_mean"]) / norms["branch_std"]).astype(np.float32)
    point_norm = ((data.point_features_hat - norms["point_mean"].reshape(1, 1, -1)) / norms["point_std"].reshape(1, 1, -1)).astype(np.float32)

    model = make_model(checkpoint, norms, device)
    columns = list(range(48))
    le_rows: list[np.ndarray] = []
    j_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    for start in range(0, indices.size, int(batch_size)):
        sub = indices[start : start + int(batch_size)]
        xb = torch.as_tensor(branch_norm[sub], dtype=torch.float32, device=device)
        pb = torch.as_tensor(point_norm[sub], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_norm = model(xb, pb).detach().cpu().numpy()
        le_rows.append(pred_norm * norms["le_std"] + norms["le_mean"])
        with torch.enable_grad():
            j_norm = ad_jacobian(model, xb, pb, columns, create_graph=False, method="forward")
        j_rows.append(j_norm.detach().cpu().numpy())
        if hasattr(model, "_state_b_norm"):
            with torch.no_grad():
                state = model._state_b_norm(xb, pb).detach().cpu().numpy()  # type: ignore[attr-defined]
            state_rows.append(state)

    le_pred = np.concatenate(le_rows, axis=0)
    j_pred = np.concatenate(j_rows, axis=0)
    le_std_scale = _le_std_scale(norms["le_std"], data.le.shape[1])
    q_std = norms["branch_std"].reshape(-1)[:48]
    b_pred = j_pred * le_std_scale / q_std.reshape(1, 1, 1, 48)
    out: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "model_style": checkpoint.get("model_style"),
        "args_model_style": args.get("model_style"),
        "args_plane_gauss_order": args.get("plane_gauss_order"),
        "args_thickness_gauss_order": args.get("thickness_gauss_order"),
        "model_output_LE_shape": list(le_pred.shape),
        "AD_B_shape": list(b_pred.shape),
        "LE_rel_on_selected_frames": rel_np(le_pred, data.le[indices]),
        "B_qdef_hat_rel_on_selected_frames": rel_np(b_pred, data.b[indices]),
        "branch_norm_finite": bool(np.isfinite(branch_norm[indices]).all()),
        "point_norm_finite": bool(np.isfinite(point_norm[indices]).all()),
        "branch_norm_abs_max_selected": float(np.max(np.abs(branch_norm[indices]))),
        "point_norm_abs_max_selected": float(np.max(np.abs(point_norm[indices]))),
        "branch_std_floor_meta": checkpoint.get("branch_std_floor_meta", {}),
        "state_b_meta": checkpoint.get("state_b", {}),
        "top_branch_norm_abs": top_branch_rows(
            branch_norm,
            branch_raw,
            norms["branch_mean"],
            norms["branch_std"],
            data.case_id,
            indices,
        ),
        "top_point_norm_abs": top_feature_rows(
            point_norm,
            data.point_features_hat,
            norms["point_mean"],
            norms["point_std"],
            data.point_meta.get("point_feature_names", []),
            data.case_id,
            indices,
        ),
    }
    if state_rows:
        state_j = np.concatenate(state_rows, axis=0)
        out["AD_J_norm_vs_explicit_state_B_norm"] = metric(j_pred, state_j)
        state_b = state_j * le_std_scale / q_std.reshape(1, 1, 1, 48)
        out["AD_B_qdef_hat_vs_explicit_state_B_qdef_hat"] = metric(b_pred, state_b)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint: dict[str, Any] | None = None
    ckpt_args: dict[str, Any] = {}
    device = torch.device("cuda" if bool(args.cuda) and torch.cuda.is_available() else "cpu")
    checkpoint_text = str(args.checkpoint).strip()
    has_checkpoint = checkpoint_text not in {"", "."}
    if has_checkpoint:
        checkpoint = torch_load_cross_platform(Path(args.checkpoint), device)
        ckpt_args = dict(checkpoint.get("args", {}))

    plane_order = int(ckpt_args.get("plane_gauss_order", args.plane_gauss_order))
    thickness_order = int(ckpt_args.get("thickness_gauss_order", args.thickness_gauss_order))
    requested_table = macro16_standard_point_table(plane_order=plane_order, thickness_order=thickness_order)
    data = load_macro16_compacts(
        read_path_list(Path(args.compact_list)),
        point_table=requested_table,
        frame_stride=1,
        max_frames_per_compact=0,
        scale_mode=str(ckpt_args.get("scale_mode", args.scale_mode)),
        b_label_coordinate=str(ckpt_args.get("b_label_coordinate", args.b_label_coordinate)),
    )
    indices = select_indices(data.case_id, str(args.case_list), int(args.max_frames))
    branch_raw = np.concatenate([data.q48_hat, data.x16_hat.reshape(data.x16_hat.shape[0], -1), data.length_scale], axis=1)
    source128 = macro16_source128_point_table()
    payload: dict[str, Any] = {
        "script": "diagnose_macro16_network_io_contract.py",
        "status": "diagnosis_complete",
        "device": str(device),
        "selected_frame_count": int(indices.size),
        "selected_cases": sorted(np.unique(data.case_id[indices]).astype(int).tolist()),
        "model_visible_q": data.point_meta.get("model_visible_q"),
        "model_visible_B": data.point_meta.get("model_visible_B"),
        "rigid_motion_removed_by_preprocessing": bool(data.point_meta.get("rigid_motion_removed_by_preprocessing")),
        "actual_branch_contract": "q48_def_hat[48] + X16_hat.flatten[48] + L_ref[1]",
        "actual_branch_dim": int(branch_raw.shape[1]),
        "documented_minimal_branch_without_L_ref_dim": 96,
        "branch_contains_L_ref": bool(branch_raw.shape[1] == 97),
        "branch_first_48": "q48_def_hat",
        "branch_48_to_95": "X16_hat.flatten",
        "branch_96": "L_ref",
        "q_input_dim": 48,
        "x16_input_dim": 48,
        "point_count": int(data.le.shape[1]),
        "point_feature_dim": int(data.point_features_hat.shape[-1]),
        "point_feature_names": data.point_meta.get("point_feature_names", []),
        "LE_shape": list(data.le.shape),
        "B_macro_qdef_shape": list(data.b.shape),
        "requested_standard_point_table_count": int(requested_table.xi.shape[0]),
        "loaded_point_count": int(data.le.shape[1]),
        "source128_point_count": int(source128.xi.shape[0]),
        "loader_switched_to_compact_source128": bool(requested_table.xi.shape[0] != data.le.shape[1] and data.le.shape[1] == source128.xi.shape[0]),
        "L_ref_stats_selected": {
            "min": float(np.min(data.length_scale[indices])),
            "max": float(np.max(data.length_scale[indices])),
            "std": float(np.std(data.length_scale[indices])),
        },
        "compact_field_checks": compact_field_checks(data, indices),
    }
    if checkpoint is not None:
        payload["checkpoint_io_checks"] = checkpoint_io_checks(
            Path(args.checkpoint),
            data,
            indices,
            device=device,
            batch_size=int(args.batch_size),
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", required=True, type=Path)
    parser.add_argument("--checkpoint", default="", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case-list", default="")
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--plane-gauss-order", type=int, default=3)
    parser.add_argument("--thickness-gauss-order", type=int, default=2)
    parser.add_argument("--scale-mode", default="normalized")
    parser.add_argument("--b-label-coordinate", default="auto")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    payload = run(args)
    write_json(Path(args.out), payload)
    print(json.dumps({"status": payload["status"], "selected_cases": payload["selected_cases"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
