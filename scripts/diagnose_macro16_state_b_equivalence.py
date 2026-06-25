#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate 22 Macro16 state-B equivalence diagnostics.

This is a read-only diagnostic.  It compares the Gate 17 direct-B prototype
with the production trainer's le0-state-b path and checks where the two
training objectives stop being equivalent.
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

from audit_macro16_trained_force_closure import torch_load_cross_platform  # noqa: E402
from macro_deeponet.macro16_geometry import macro16_source128_point_table  # noqa: E402
from macro_deeponet.models import Macro16BoundaryDeepONetWithLE0Fixed128StateB, Macro16BoundaryDeepONetWithLE0StateB  # noqa: E402
from macro_deeponet.train_macro16_boundary_sobolev import (  # noqa: E402
    build_physical_b_loss_scale,
    compact_paths_from_args,
    load_macro16_compacts,
    parse_int_list,
    physical_balanced_b_loss_from_j_norm,
    read_path_list,
    split_indices,
    standardize,
    _le_std_scale,
)
from macro_deeponet.train_true176_deeponet_sobolev import ad_jacobian, cos_np, rel_np, write_json  # noqa: E402
from macro_deeponet.true176_data import stats  # noqa: E402
from train_macro16_state_b_baseline_prototype import (  # noqa: E402
    load_prototype_checkpoint,
    prepare_arrays as prepare_prototype_arrays,
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
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return str(obj)


def rms_np(arr: np.ndarray) -> float:
    vals = np.asarray(arr, dtype=np.float64)
    return float(np.sqrt(np.mean(vals * vals))) if vals.size else 0.0


def norm_ratio_np(num: np.ndarray, den: np.ndarray) -> float:
    num_flat = np.asarray(num, dtype=np.float64).reshape(-1)
    den_flat = np.asarray(den, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(num_flat) / max(float(np.linalg.norm(den_flat)), 1.0e-300))


def model_style_key(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


def load_checkpoint_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch_load_cross_platform(path, device)
    args = checkpoint.get("args", {})
    norms = checkpoint.get("norms", {})
    state_b = checkpoint.get("state_b", {})
    model_style = model_style_key(str(checkpoint.get("model_style", args.get("model_style", ""))))
    if "state-b" not in model_style:
        raise ValueError(f"{path}: expected a le0-state-b checkpoint, got {model_style!r}")
    branch_mean = np.asarray(norms["branch_mean"], dtype=np.float32)
    branch_std = np.asarray(norms["branch_std"], dtype=np.float32)
    le_mean = np.asarray(norms["le_mean"], dtype=np.float32)
    le_std = np.asarray(norms["le_std"], dtype=np.float32)
    skip_init = np.zeros((128, 6, 48), dtype=np.float32) if "fixed128" in model_style else np.zeros((6, 48), dtype=np.float32)
    le0_init = np.zeros((128, 6), dtype=np.float32)
    common = dict(
        input_dim=int(branch_mean.reshape(-1).shape[0]),
        point_dim=int(np.asarray(norms["point_mean"]).reshape(-1).shape[0]),
        ip_count=128 if "fixed128" in model_style else 0,
        q_start=0,
        q_dim=48,
        basis_dim=int(args.get("basis_dim", 96)),
        hidden_dim=int(args.get("hidden_dim", 256)),
        branch_depth=int(args.get("branch_depth", 4)),
        trunk_depth=int(args.get("trunk_depth", 4)),
        activation=str(args.get("activation", "tanh")),
        skip_init=torch.as_tensor(skip_init, dtype=torch.float32),
        train_skip=not bool(args.get("freeze_skip", False)),
        residual_scale=float(args.get("residual_scale", 1.0)),
        baseline_scale=float(args.get("fe_baseline_scale", 1.0)),
        train_point_baseline=not bool(args.get("freeze_fe_point_baseline", False)),
        zero_init_residual=not bool(args.get("random_init_residual", False)),
        q_zero_norm=((0.0 - branch_mean.reshape(-1)[:48]) / branch_std.reshape(-1)[:48]).astype(np.float32),
        q_raw_mean=branch_mean.reshape(-1)[:48].astype(np.float32),
        q_raw_std=branch_std.reshape(-1)[:48].astype(np.float32),
        gate_q0=float(args.get("anchored_residual_gate_q0", 0.0)),
        le0_init_norm=torch.as_tensor(le0_init, dtype=torch.float32),
        le0_scale=float(args.get("le0_scale", 1.0)),
        train_le0_static=not bool(args.get("freeze_le0_static", False)),
        train_le0_point=not bool(args.get("freeze_le0_point", False)),
        state_b_rank=int(state_b.get("state_b_rank", args.get("state_b_rank", 8))),
        state_b_scale=float(state_b.get("state_b_scale", args.get("state_b_scale", 1.0))),
        detach_state_b=bool(state_b.get("detach_state_b", args.get("detach_state_b", True))),
        state_b_zero_init=not bool(args.get("state_b_random_init", False)),
    )
    if "fixed128" in model_style:
        model = Macro16BoundaryDeepONetWithLE0Fixed128StateB(**common).to(device)
    else:
        model = Macro16BoundaryDeepONetWithLE0StateB(
            **common,
            state_b_kind=str(state_b.get("state_b_kind", args.get("state_b_kind", "point_q_rank"))),
        ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def prepare_current_arrays(compact_paths: list[str], val_cases: str, seed: int) -> dict[str, Any]:
    data = load_macro16_compacts(
        compact_paths,
        point_table=macro16_source128_point_table(),
        frame_stride=1,
        max_frames_per_compact=0,
        scale_mode="normalized",
        b_label_coordinate="auto",
    )
    train_idx, val_idx, split_meta = split_indices(data.case_id, val_fraction=0.2, val_cases=val_cases, seed=seed)
    branch_raw = np.concatenate([data.q48_hat, data.x16_hat.reshape(data.x16_hat.shape[0], -1), data.length_scale], axis=1)
    branch_mean, branch_std = stats(branch_raw[train_idx], axis=0)
    point_mean, point_std = stats(data.point_features_hat[train_idx].reshape(-1, data.point_features_hat.shape[-1]), axis=0)
    le_mean, le_std = stats(data.le[train_idx], axis=0)
    branch_norm = standardize(branch_raw, branch_mean, branch_std)
    point_norm = standardize(data.point_features_hat, point_mean.reshape(1, 1, -1), point_std.reshape(1, 1, -1))
    le_norm = standardize(data.le, le_mean, le_std)
    q_std = branch_std.reshape(-1)[:48]
    j_norm_target = (data.b * q_std.reshape(1, 1, 1, 48) / _le_std_scale(le_std, data.le.shape[1])).astype(np.float32)
    b_loss_scale = build_physical_b_loss_scale(data.b[train_idx], floor_rel=2.0e-2, floor_abs=1.0e-8)
    return {
        "data": data,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "split_meta": split_meta,
        "branch_raw": branch_raw.astype(np.float32),
        "branch_norm": branch_norm.astype(np.float32),
        "point_norm": point_norm.astype(np.float32),
        "le_norm": le_norm.astype(np.float32),
        "j_norm_target": j_norm_target.astype(np.float32),
        "b_loss_scale": b_loss_scale.astype(np.float32),
        "branch_mean": branch_mean.astype(np.float32),
        "branch_std": branch_std.astype(np.float32),
        "point_mean": point_mean.astype(np.float32),
        "point_std": point_std.astype(np.float32),
        "le_mean": le_mean.astype(np.float32),
        "le_std": le_std.astype(np.float32),
    }


def compare_norms(checkpoint: dict[str, Any], arrays: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    norms = checkpoint.get("norms", {})
    for name in ("branch_mean", "branch_std", "point_mean", "point_std", "le_mean", "le_std"):
        got = np.asarray(norms[name], dtype=np.float32)
        exp = np.asarray(arrays[name], dtype=np.float32)
        diff = got - exp
        out.append(
            {
                "name": name,
                "shape": list(exp.shape),
                "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
                "rel": norm_ratio_np(diff, exp),
                "allclose": bool(np.allclose(got, exp, rtol=1.0e-5, atol=1.0e-7)),
            }
        )
    return out


def evaluate_checkpoint(model: Macro16BoundaryDeepONetWithLE0StateB, arrays: dict[str, Any], indices: np.ndarray, device: torch.device) -> dict[str, Any]:
    data = arrays["data"]
    idx = np.asarray(indices, dtype=np.int64)
    xb = torch.as_tensor(arrays["branch_norm"][idx], dtype=torch.float32, device=device)
    pb = torch.as_tensor(arrays["point_norm"][idx], dtype=torch.float32, device=device)
    columns = list(range(48))
    le_std_scale_np = _le_std_scale(arrays["le_std"], data.le.shape[1]).astype(np.float32)
    q_std = arrays["branch_std"].reshape(-1)[:48].astype(np.float32)
    with torch.no_grad():
        pred_norm = model(xb, pb)
        state_b_norm = model._state_b_norm(xb, pb)
        if hasattr(model, "residual_offset_norm"):
            residual_norm = model.residual_offset_norm(xb, pb)
        else:
            residual_norm = torch.zeros_like(pred_norm)
    with torch.enable_grad():
        j_ad_norm = ad_jacobian(model, xb, pb, columns, create_graph=False, method="forward")
    state_b_np = state_b_norm.detach().cpu().numpy().astype(np.float64)
    j_ad_np = j_ad_norm.detach().cpu().numpy().astype(np.float64)
    residual_np = residual_norm.detach().cpu().numpy().astype(np.float64)
    pred_le = pred_norm.detach().cpu().numpy().astype(np.float64) * arrays["le_std"] + arrays["le_mean"]
    state_b_phys = state_b_np * le_std_scale_np / q_std.reshape(1, 1, 1, 48)
    ad_b_phys = j_ad_np * le_std_scale_np / q_std.reshape(1, 1, 1, 48)
    b_true = data.b[idx].astype(np.float64)
    le_true = data.le[idx].astype(np.float64)
    j_true = arrays["j_norm_target"][idx].astype(np.float64)
    qn = xb[:, :48]
    q0 = model.q_zero_norm.to(dtype=xb.dtype, device=xb.device).view(1, 48)
    q_mean = arrays["branch_mean"].reshape(-1)[:48].astype(np.float32)
    q_raw_from_norm = (
        qn * torch.as_tensor(q_std, dtype=xb.dtype, device=xb.device).view(1, 48)
        + torch.as_tensor(q_mean, dtype=xb.dtype, device=xb.device).view(1, 48)
    ).detach().cpu().numpy()
    q_raw_direct = data.q48_hat[idx].astype(np.float64)
    with torch.no_grad():
        linear_q_minus_q0 = torch.einsum("bpaj,bj->bpa", state_b_norm, qn - q0).detach().cpu().numpy()
        linear_q_norm = torch.einsum("bpaj,bj->bpa", state_b_norm, qn).detach().cpu().numpy()
    j_loss = physical_balanced_b_loss_from_j_norm(
        torch.as_tensor(j_ad_np, dtype=torch.float32),
        torch.as_tensor(j_true, dtype=torch.float32),
        le_std_scale=torch.as_tensor(le_std_scale_np, dtype=torch.float32),
        q_std_cols=torch.as_tensor(q_std, dtype=torch.float32),
        b_scale=torch.as_tensor(arrays["b_loss_scale"], dtype=torch.float32),
    )
    return {
        "frame_count": int(idx.size),
        "LE_rel": rel_np(pred_le, le_true),
        "LE_cos": cos_np(pred_le, le_true),
        "AD_B_rel": rel_np(ad_b_phys, b_true),
        "AD_B_cos": cos_np(ad_b_phys, b_true),
        "explicit_state_B_rel": rel_np(state_b_phys, b_true),
        "explicit_state_B_cos": cos_np(state_b_phys, b_true),
        "AD_B_vs_explicit_state_B_rel": rel_np(ad_b_phys, state_b_phys),
        "AD_B_minus_explicit_state_B_rms": rms_np(ad_b_phys - state_b_phys),
        "AD_B_minus_explicit_state_B_rel_to_true": norm_ratio_np(ad_b_phys - state_b_phys, b_true),
        "residual_norm_rms": rms_np(residual_np),
        "residual_norm_rel_to_pred_norm": norm_ratio_np(residual_np, pred_norm.detach().cpu().numpy()),
        "physical_balanced_b_loss": float(j_loss.detach().cpu()),
        "q_raw_from_norm_vs_direct_rel": rel_np(q_raw_from_norm, q_raw_direct),
        "linear_q_minus_q0_vs_linear_q_norm_rel": rel_np(linear_q_minus_q0, linear_q_norm),
        "linear_q_minus_q0_rms": rms_np(linear_q_minus_q0),
        "linear_q_norm_rms": rms_np(linear_q_norm),
    }


def evaluate_prototype_checkpoint(path: Path, compact_paths: list[str], val_cases: str, seed: int, device: torch.device) -> dict[str, Any]:
    model, checkpoint = load_prototype_checkpoint(path.resolve(), device)
    arrays = prepare_prototype_arrays(compact_paths, val_cases=val_cases, val_fraction=0.2, seed=seed)
    data = arrays["data"]
    all_idx = np.arange(data.q48_hat.shape[0], dtype=np.int64)

    def eval_idx(indices: np.ndarray) -> dict[str, Any]:
        idx = np.asarray(indices, dtype=np.int64)
        q = torch.as_tensor(arrays["q_norm"][idx], dtype=torch.float32, device=device)
        geom = torch.as_tensor(arrays["geom_norm"][idx], dtype=torch.float32, device=device)
        point = torch.as_tensor(arrays["point_norm"][idx], dtype=torch.float32, device=device)
        with torch.no_grad():
            le_pred, b_pred = model(q, geom, point)
        le_np = le_pred.detach().cpu().numpy().astype(np.float64)
        b_np = b_pred.detach().cpu().numpy().astype(np.float64)
        le_true = data.le[idx].astype(np.float64)
        b_true = data.b[idx].astype(np.float64)
        return {
            "frame_count": int(idx.size),
            "LE_rel": rel_np(le_np, le_true),
            "LE_cos": cos_np(le_np, le_true),
            "direct_B_rel": rel_np(b_np, b_true),
            "direct_B_cos": cos_np(b_np, b_true),
        }

    return {
        "checkpoint": str(path),
        "model_kind": checkpoint.get("model_kind"),
        "model_config": checkpoint.get("model_config", {}),
        "best": checkpoint.get("best", {}),
        "latest": checkpoint.get("latest", {}),
        "train": eval_idx(arrays["train_idx"]),
        "val": eval_idx(arrays["val_idx"]),
        "all": eval_idx(all_idx),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--prototype-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-cases", default="3102")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if bool(args.cuda) and torch.cuda.is_available() else "cpu")
    compact_paths = read_path_list(args.compact_list)
    arrays = prepare_current_arrays(compact_paths, val_cases=str(args.val_cases), seed=int(args.seed))
    all_idx = np.arange(arrays["data"].q48_hat.shape[0], dtype=np.int64)
    payload: dict[str, Any] = {
        "diagnostic": "gate22_state_b_equivalence",
        "device": str(device),
        "compact_list": str(args.compact_list),
        "compact_paths": compact_paths,
        "split": arrays["split_meta"],
        "train_frames": int(arrays["train_idx"].size),
        "val_frames": int(arrays["val_idx"].size),
        "checks": [],
        "prototype_checks": [],
        "prototype_reference": {
            "gate17_best_all_B_rel": 0.0473557015,
            "gate17_best_force_rel": 0.0919219356,
        },
    }
    for ckpt_path in args.checkpoint:
        model, checkpoint = load_checkpoint_model(ckpt_path.resolve(), device)
        check = {
            "checkpoint": str(ckpt_path),
            "state_b": checkpoint.get("state_b", {}),
            "best_report": checkpoint.get("best_report", {}),
            "latest_report": checkpoint.get("latest_report", {}),
            "norm_checks": compare_norms(checkpoint, arrays),
            "train": evaluate_checkpoint(model, arrays, arrays["train_idx"], device),
            "val": evaluate_checkpoint(model, arrays, arrays["val_idx"], device),
            "all": evaluate_checkpoint(model, arrays, all_idx, device),
        }
        payload["checks"].append(check)
    for ckpt_path in args.prototype_checkpoint:
        payload["prototype_checks"].append(
            evaluate_prototype_checkpoint(ckpt_path, compact_paths, str(args.val_cases), int(args.seed), device)
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, payload)
    print(json.dumps(payload, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
