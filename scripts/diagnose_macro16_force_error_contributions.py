"""Diagnose trained Macro16 force error contributions.

This script is read-only.  It compares one or more trained checkpoints on the
same Macro16 compact rows and decomposes selected-frame force error by
integration point, strain component, and boundary DOF.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_macro16_force_stiffness import json_default, read_path_list, write_json  # noqa: E402
from audit_macro16_trained_force_closure import (  # noqa: E402
    make_model,
    parse_int_list,
    parse_path_map,
    qdef_projected_rf,
    torch_load_cross_platform,
)
from macro_deeponet.macro16_geometry import MACRO16_CONTRACT_VERSION, macro16_standard_point_table  # noqa: E402
from macro_deeponet.train_macro16_boundary_sobolev import load_macro16_compacts  # noqa: E402
from macro_deeponet.train_true176_deeponet_sobolev import ad_jacobian, rel_np  # noqa: E402


STRAIN_NAMES = ["E11", "E22", "E33", "G12", "G13", "G23"]


def metric(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    pred64 = np.asarray(pred, dtype=np.float64)
    ref64 = np.asarray(ref, dtype=np.float64)
    diff = pred64 - ref64
    denom = max(float(np.linalg.norm(ref64.reshape(-1))), 1.0e-300)
    pred_norm = float(np.linalg.norm(pred64.reshape(-1)))
    ref_norm = float(np.linalg.norm(ref64.reshape(-1)))
    dot = float(np.dot(pred64.reshape(-1), ref64.reshape(-1)))
    cos = dot / max(pred_norm * ref_norm, 1.0e-300)
    return {
        "rel": float(np.linalg.norm(diff.reshape(-1)) / denom),
        "rms": float(np.sqrt(np.mean(diff * diff))),
        "pred_norm": pred_norm,
        "ref_norm": ref_norm,
        "cos": cos,
    }


def top_entries(values: np.ndarray, *, count: int, label: str) -> list[dict[str, Any]]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return []
    order = np.argsort(-arr)[: int(count)]
    total = max(float(np.sum(arr)), 1.0e-300)
    return [
        {
            label: int(i),
            "value": float(arr[int(i)]),
            "fraction": float(arr[int(i)] / total),
        }
        for i in order
    ]


def selected_indices(case_ids: np.ndarray, case_list: str, max_frames: int) -> np.ndarray:
    idx = np.arange(case_ids.shape[0], dtype=np.int64)
    wanted = set(parse_int_list(case_list)) if str(case_list).strip() else set()
    if wanted:
        idx = idx[np.isin(case_ids[idx], np.asarray(sorted(wanted), dtype=np.int64))]
    if int(max_frames) > 0:
        idx = idx[: int(max_frames)]
    if idx.size == 0:
        raise ValueError("no frames selected")
    return idx


def load_prediction(
    checkpoint_path: Path,
    compact_paths: list[str],
    *,
    case_list: str,
    max_frames: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch_load_cross_platform(checkpoint_path, device)
    if str(checkpoint.get("contract_version", "")) != MACRO16_CONTRACT_VERSION:
        raise ValueError(f"{checkpoint_path}: unexpected contract version {checkpoint.get('contract_version')}")
    args = checkpoint.get("args", {})
    point_table = macro16_standard_point_table(
        plane_order=int(args.get("plane_gauss_order", 3)),
        thickness_order=int(args.get("thickness_gauss_order", 2)),
    )
    data = load_macro16_compacts(
        compact_paths,
        point_table=point_table,
        frame_stride=1,
        max_frames_per_compact=0,
        scale_mode=str(args.get("scale_mode", "normalized")),
        b_label_coordinate=str(args.get("b_label_coordinate", "auto")),
    )
    idx = selected_indices(data.case_id, case_list, max_frames)
    norms = {key: np.asarray(value) for key, value in checkpoint["norms"].items()}
    branch_raw = np.concatenate([data.q48_hat, data.x16_hat.reshape(data.x16_hat.shape[0], -1), data.length_scale], axis=1)
    branch_norm = ((branch_raw - norms["branch_mean"]) / norms["branch_std"]).astype(np.float32)
    point_norm = ((data.point_features_hat - norms["point_mean"].reshape(1, 1, -1)) / norms["point_std"].reshape(1, 1, -1)).astype(np.float32)
    model = make_model(checkpoint, norms, device)
    columns = list(range(48))
    le_rows: list[np.ndarray] = []
    b_rows: list[np.ndarray] = []
    for start in range(0, idx.size, int(batch_size)):
        sub = idx[start : start + int(batch_size)]
        xb = torch.as_tensor(branch_norm[sub], dtype=torch.float32, device=device)
        pb = torch.as_tensor(point_norm[sub], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_norm = model(xb, pb).detach().cpu().numpy()
        le_rows.append(pred_norm * norms["le_std"] + norms["le_mean"])
        with torch.enable_grad():
            j_norm = ad_jacobian(model, xb, pb, columns, create_graph=False, method="forward")
        j_np = j_norm.detach().cpu().numpy().astype(np.float64)
        b_hat = j_np * np.asarray(norms["le_std"], dtype=np.float64).reshape(1, data.le.shape[1], 6, 1)
        b_hat = b_hat / np.asarray(norms["branch_std"], dtype=np.float64).reshape(-1)[:48].reshape(1, 1, 1, 48)
        b_rows.append(b_hat)
    le_pred = np.concatenate(le_rows, axis=0).astype(np.float64)
    b_pred_hat = np.concatenate(b_rows, axis=0).astype(np.float64)
    l_ref = np.asarray(data.length_scale[idx], dtype=np.float64).reshape(-1, 1, 1, 1)
    return {
        "checkpoint": str(checkpoint_path),
        "args": args,
        "data": data,
        "indices": idx,
        "le_pred": le_pred,
        "b_pred_hat": b_pred_hat,
        "b_pred_raw": b_pred_hat / np.maximum(l_ref, 1.0e-12),
        "le_true": np.asarray(data.le[idx], dtype=np.float64),
        "b_true_hat": np.asarray(data.b[idx], dtype=np.float64),
        "b_true_raw": np.asarray(data.b[idx], dtype=np.float64) / np.maximum(l_ref, 1.0e-12),
    }


def force_payload(
    pred: dict[str, Any],
    *,
    source_path_map: list[tuple[str, str]],
) -> dict[str, Any]:
    data = pred["data"]
    idx = pred["indices"]
    rf_qdef, weights, elastic_d = qdef_projected_rf(
        data.compact_paths,
        data.source_index[idx],
        data.source_row[idx],
        source_path_map=source_path_map,
    )
    stress_pred = np.einsum("nab,npb->npa", elastic_d, pred["le_pred"])
    stress_true = np.einsum("nab,npb->npa", elastic_d, pred["le_true"])
    force_pred_by_p = np.einsum("npaj,npa,np->npj", pred["b_pred_raw"], stress_pred, weights)
    force_true_by_p = np.einsum("npaj,npa,np->npj", pred["b_true_raw"], stress_true, weights)
    force_pred = np.sum(force_pred_by_p, axis=1)
    force_teacher = np.sum(force_true_by_p, axis=1)
    force_error = force_pred - rf_qdef
    teacher_error = force_teacher - rf_qdef

    point_error = force_pred_by_p - force_true_by_p
    by_point = np.sum(point_error * point_error, axis=(0, 2))
    pred_by_comp = np.einsum("npaj,npa,np->npaj", pred["b_pred_raw"], stress_pred, weights)
    true_by_comp = np.einsum("npaj,npa,np->npaj", pred["b_true_raw"], stress_true, weights)
    comp_error = np.sum(pred_by_comp - true_by_comp, axis=1)
    by_component = np.sum(comp_error * comp_error, axis=(0, 2))
    by_dof = np.sum(force_error * force_error, axis=0)
    by_frame = np.sum(force_error * force_error, axis=1)

    pred_norm = np.linalg.norm(force_pred, axis=1)
    ref_norm = np.linalg.norm(rf_qdef, axis=1)
    cos_frame = np.einsum("nj,nj->n", force_pred, rf_qdef) / np.maximum(pred_norm * ref_norm, 1.0e-300)
    norm_ratio = pred_norm / np.maximum(ref_norm, 1.0e-300)

    comp_total = max(float(np.sum(by_component)), 1.0e-300)
    comp_entries = [
        {
            "component": int(i),
            "name": STRAIN_NAMES[int(i)],
            "value": float(by_component[int(i)]),
            "fraction": float(by_component[int(i)] / comp_total),
        }
        for i in np.argsort(-by_component)
    ]

    return {
        "frame_count": int(idx.size),
        "case_ids": sorted(np.unique(data.case_id[idx]).astype(int).tolist()),
        "point_count": int(pred["le_true"].shape[1]),
        "LE_rel": rel_np(pred["le_pred"], pred["le_true"]),
        "B_qdef_hat_rel": rel_np(pred["b_pred_hat"], pred["b_true_hat"]),
        "force": metric(force_pred, rf_qdef),
        "teacher_force": metric(force_teacher, rf_qdef),
        "teacher_vs_model_force_error_rel": rel_np(force_pred, force_teacher),
        "top_points": top_entries(by_point, count=10, label="point"),
        "top_components": comp_entries[:6],
        "top_dofs": top_entries(by_dof, count=10, label="dof"),
        "top_frames": top_entries(by_frame, count=10, label="frame_local_index"),
        "force_norm_ratio_mean": float(np.mean(norm_ratio)),
        "force_norm_ratio_min": float(np.min(norm_ratio)),
        "force_norm_ratio_max": float(np.max(norm_ratio)),
        "force_cos_mean": float(np.mean(cos_frame)),
        "force_cos_min": float(np.min(cos_frame)),
        "force_cos_max": float(np.max(cos_frame)),
        "force_error_rms_by_frame": [float(v) for v in np.sqrt(np.mean(force_error * force_error, axis=1)).tolist()],
        "force_norm_ratio_by_frame": [float(v) for v in norm_ratio.tolist()],
        "force_cos_by_frame": [float(v) for v in cos_frame.tolist()],
    }


def parse_checkpoint_pair(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"--checkpoint must be NAME=PATH, got {text!r}")
    name, path = text.split("=", 1)
    return name.strip(), Path(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if bool(args.cuda) and torch.cuda.is_available() else "cpu")
    compact_paths = [str(p) for p in read_path_list(Path(args.compact_list))]
    path_map = parse_path_map(list(args.source_path_map))
    outputs: dict[str, Any] = {}
    for item in args.checkpoint:
        name, path = parse_checkpoint_pair(str(item))
        pred = load_prediction(
            Path(path).resolve(),
            compact_paths,
            case_list=str(args.case_list),
            max_frames=int(args.max_frames),
            batch_size=int(args.batch_size),
            device=device,
        )
        outputs[name] = {
            "checkpoint": str(Path(path).resolve()),
            **force_payload(pred, source_path_map=path_map),
        }
    names = list(outputs)
    comparison: dict[str, Any] = {}
    if len(names) >= 2:
        base = outputs[names[0]]
        other = outputs[names[1]]
        comparison = {
            "baseline": names[0],
            "candidate": names[1],
            "force_rel_delta": float(other["force"]["rel"] - base["force"]["rel"]),
            "B_rel_delta": float(other["B_qdef_hat_rel"] - base["B_qdef_hat_rel"]),
            "LE_rel_delta": float(other["LE_rel"] - base["LE_rel"]),
            "candidate_force_worse_despite_lower_B": bool(
                other["force"]["rel"] > base["force"]["rel"] and other["B_qdef_hat_rel"] < base["B_qdef_hat_rel"]
            ),
        }
    return {
        "task": "Gate 25 force error diagnosis",
        "device": str(device),
        "compact_list": str(Path(args.compact_list).resolve()),
        "case_list": str(args.case_list),
        "max_frames": int(args.max_frames),
        "checkpoints": outputs,
        "comparison": comparison,
        "interpretation": {
            "force_weighted_b_loss_recommended": True,
            "reason": "Global B rel does not rank checkpoints by selected-frame force closure.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--compact-list", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case-list", default="")
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--source-path-map", action="append", default=[])
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    write_json(Path(args.out), payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
