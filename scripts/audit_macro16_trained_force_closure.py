#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit selected-frame force closure for a trained Macro16 LE/B checkpoint.

This is a post-training inference audit.  It does not train a network, change
the model architecture, change q48/LE ordering, or change the 128-point rule.
The model predicts LE and B is obtained by autograd with respect to
q48_def_hat.
"""

from __future__ import annotations

import argparse
import json
import math
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

from audit_macro16_force_stiffness import (  # noqa: E402
    assemble_force_stiffness,
    load_source,
    metric,
    source_vectors_to_macro,
)
from macro_deeponet.macro16_geometry import MACRO16_CONTRACT_VERSION, macro16_standard_point_table  # noqa: E402
from macro_deeponet.models import Macro16BoundaryDeepONet, Macro16BoundaryDeepONetWithLE0  # noqa: E402
from macro_deeponet.train_macro16_boundary_sobolev import (  # noqa: E402
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


def rel_np(pred: np.ndarray, ref: np.ndarray) -> float:
    pred_arr = np.asarray(pred, dtype=np.float64).reshape(-1)
    ref_arr = np.asarray(ref, dtype=np.float64).reshape(-1)
    den = max(float(np.linalg.norm(ref_arr)), 1.0e-300)
    return float(np.linalg.norm(pred_arr - ref_arr) / den)


def scalar_text(z: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    if arr.size == 0:
        return default
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def qdef_projected_rf(compact_paths: list[str], source_index: np.ndarray, source_row: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return P^T RF_macro, selected-frame volume, and elastic D for loaded rows."""

    rf_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    d_rows: list[np.ndarray] = []
    for source_id in sorted(np.unique(source_index).astype(int).tolist()):
        idx = np.where(source_index == source_id)[0]
        compact_path = Path(compact_paths[source_id])
        rows = np.asarray(source_row[idx], dtype=np.int64)
        with np.load(str(compact_path), allow_pickle=True) as z:
            if "source_compact" not in z.files:
                raise KeyError(f"{compact_path}: missing source_compact")
            source_path = Path(scalar_text(z, "source_compact"))
            source_order = scalar_text(z, "source_node_order", "macro16")
            n_total = int(np.asarray(z["q48_def_hat"]).shape[0])
            p_rigid_all = np.asarray(z["rigid_projection_P"], dtype=np.float64)
            if p_rigid_all.shape != (n_total, 48, 48):
                raise ValueError(f"{compact_path}: rigid_projection_P must be [N,48,48], got {p_rigid_all.shape}")
            p_rigid = p_rigid_all[rows]
        source = load_source(source_path, rows)
        if "source_selected_frame_volume" not in source:
            raise KeyError(f"{source_path}: missing ip_IVOL_abaqus_selected_frames")
        rf_macro = source_vectors_to_macro(np.asarray(source["rf"], dtype=np.float64), source_order)
        rf_qdef = np.einsum("nkj,nk->nj", p_rigid, rf_macro)
        rf_rows.append(rf_qdef)
        weight_rows.append(np.asarray(source["source_selected_frame_volume"], dtype=np.float64))
        d_rows.append(np.broadcast_to(np.asarray(source["elastic_D"], dtype=np.float64).reshape(1, 6, 6), (idx.size, 6, 6)).copy())
    return np.concatenate(rf_rows, axis=0), np.concatenate(weight_rows, axis=0), np.concatenate(d_rows, axis=0)


def make_model(checkpoint: dict[str, Any], norms: dict[str, np.ndarray], device: torch.device) -> torch.nn.Module:
    args = checkpoint.get("args", {})
    state = checkpoint["model_state"]
    le_mean = np.asarray(norms["le_mean"], dtype=np.float32)
    branch_mean = np.asarray(norms["branch_mean"], dtype=np.float32).reshape(-1)
    branch_std = np.asarray(norms["branch_std"], dtype=np.float32).reshape(-1)
    point_mean = np.asarray(norms["point_mean"], dtype=np.float32).reshape(-1)
    if le_mean.ndim == 3:
        ip_count = int(le_mean.shape[1])
    elif le_mean.ndim == 2:
        ip_count = int(le_mean.shape[0])
    else:
        ip_count = int(state.get("static_le0_norm", torch.zeros(1, 6)).shape[0])
    q_dim = int(np.asarray(norms.get("q_dim", 48)).reshape(-1)[0])
    model_style = str(checkpoint.get("model_style", "")).lower()
    common = dict(
        input_dim=int(branch_mean.size),
        point_dim=int(point_mean.size),
        ip_count=ip_count,
        q_start=int(np.asarray(norms.get("q_start", 0)).reshape(-1)[0]),
        q_dim=q_dim,
        basis_dim=int(args.get("basis_dim", 96)),
        hidden_dim=int(args.get("hidden_dim", 384)),
        branch_depth=int(args.get("branch_depth", 5)),
        trunk_depth=int(args.get("trunk_depth", 5)),
        activation=str(args.get("activation", "tanh")),
        skip_init=torch.zeros((ip_count, 6, q_dim), dtype=torch.float32),
        train_skip=True,
        residual_scale=float(args.get("residual_scale", 1.0)),
        baseline_scale=float(args.get("fe_baseline_scale", 1.0)),
        train_point_baseline=True,
        q_zero_norm=((0.0 - branch_mean[:q_dim]) / np.maximum(branch_std[:q_dim], 1.0e-12)).astype(np.float32),
        q_raw_mean=branch_mean[:q_dim].astype(np.float32),
        q_raw_std=branch_std[:q_dim].astype(np.float32),
        gate_q0=float(args.get("anchored_residual_gate_q0", 0.0)),
    )
    if "with-le0" in model_style or str(args.get("model_style", "le0")).lower() == "le0":
        model = Macro16BoundaryDeepONetWithLE0(
            **common,
            le0_init_norm=torch.zeros((ip_count, 6), dtype=torch.float32),
            le0_scale=float(args.get("le0_scale", 1.0)),
            train_le0_static=True,
            train_le0_point=True,
        )
    else:
        model = Macro16BoundaryDeepONet(
            **common,
            le_zero_norm=((0.0 - le_mean) / np.maximum(np.asarray(norms["le_std"], dtype=np.float32), 1.0e-12)).astype(np.float32),
        )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate_model(
    checkpoint_path: Path,
    compact_paths: list[str],
    *,
    case_ids: set[int],
    max_frames: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
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
    indices = np.arange(data.q48_hat.shape[0], dtype=np.int64)
    if case_ids:
        indices = indices[np.isin(data.case_id[indices], np.asarray(sorted(case_ids), dtype=np.int64))]
    if int(max_frames) > 0:
        indices = indices[: int(max_frames)]
    if indices.size == 0:
        raise ValueError("no frames selected for trained force closure audit")

    norms = {key: np.asarray(value) for key, value in checkpoint["norms"].items()}
    branch_raw = np.concatenate([data.q48_hat, data.x16_hat.reshape(data.x16_hat.shape[0], -1), data.length_scale], axis=1)
    branch_norm = ((branch_raw - norms["branch_mean"]) / norms["branch_std"]).astype(np.float32)
    point_norm = ((data.point_features_hat - norms["point_mean"].reshape(1, 1, -1)) / norms["point_std"].reshape(1, 1, -1)).astype(np.float32)
    model = make_model(checkpoint, norms, device)
    columns = list(range(48))
    le_rows: list[np.ndarray] = []
    b_rows: list[np.ndarray] = []
    for start in range(0, indices.size, int(batch_size)):
        sub = indices[start : start + int(batch_size)]
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
    l_ref = np.asarray(data.length_scale[indices], dtype=np.float64).reshape(-1, 1, 1, 1)
    b_pred_raw = b_pred_hat / np.maximum(l_ref, 1.0e-12)
    b_true_raw = np.asarray(data.b[indices], dtype=np.float64) / np.maximum(l_ref, 1.0e-12)
    le_true = np.asarray(data.le[indices], dtype=np.float64)
    rf_qdef, weights_selected, elastic_d = qdef_projected_rf(
        data.compact_paths,
        data.source_index[indices],
        data.source_row[indices],
    )
    stress_pred = np.einsum("nab,npb->npa", elastic_d, le_pred)
    force_pred = np.einsum("npaj,npa,np->nj", b_pred_raw, stress_pred, weights_selected)
    stress_true = np.einsum("nab,npb->npa", elastic_d, le_true)
    force_teacher = np.einsum("npaj,npa,np->nj", b_true_raw, stress_true, weights_selected)
    return {
        "checkpoint": str(checkpoint_path),
        "frame_count": int(indices.size),
        "case_ids": sorted(np.unique(data.case_id[indices]).astype(int).tolist()),
        "model_visible_q": data.point_meta.get("model_visible_q"),
        "model_visible_B": data.point_meta.get("model_visible_B"),
        "point_count": int(data.le.shape[1]),
        "LE_rel": rel_np(le_pred, le_true),
        "B_qdef_hat_rel": rel_np(b_pred_hat, data.b[indices]),
        "selected_frame_force_qdef": metric(force_pred, rf_qdef),
        "teacher_selected_frame_force_qdef": metric(force_teacher, rf_qdef),
        "force_coordinate": "qdef coordinate, compared against rigid_projection_P^T RF_projected_macro",
        "volume_mode": "selected-frame",
        "volume_source": "source.ip_IVOL_abaqus_selected_frames",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--compact-list", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case-list", default="")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if bool(args.cuda) and torch.cuda.is_available() else "cpu")
    case_ids = set(parse_int_list(str(args.case_list))) if str(args.case_list).strip() else set()
    paths = read_path_list(Path(args.compact_list))
    payload = evaluate_model(
        Path(args.checkpoint).resolve(),
        [str(path) for path in paths],
        case_ids=case_ids,
        max_frames=int(args.max_frames),
        batch_size=int(args.batch_size),
        device=device,
    )
    payload["device"] = str(device)
    write_json(Path(args.out).resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
