#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LE-primary full-direction AD-B smoke trainer for v3 CSS8 compacts.

This is the clean LE-primary gate for the v3 standard-operator route:

    input:  q_useful_hat frame + geometry_global_hat + query-point features
    output: LE_local_stack frame
    AD-B:   dLE_local_stack / dq_useful_hat

The model has no direct B head, no radial loss, and no V + B @ q_perp
decomposition.  It uses a DeepONet-style branch/trunk factorization where the
branch consumes the normalized useful q plus global geometry, and the trunk
consumes normalized point features.

The trainer is a smoke/gate script.  It writes JSON/CSV metrics but no
checkpoint and makes no held-out generalization claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from torch import nn
from torch.func import jacrev


CONTRACT_VERSION = "v3-css8-standard-operator-001"
PRESETS: dict[str, dict[str, int | float]] = {
    "one-frame": {
        "steps": 8000,
        "hidden": 256,
        "basis": 96,
        "lr": 3.0e-4,
        "weight_decay": 0.0,
        "le_weight": 1.0,
        "b_weight": 1.0,
        "sample_batch": 1,
        "le_point_batch": 128,
        "ad_point_batch": 128,
        "eval_point_batch": 128,
        "eval_every": 500,
        "grad_clip": 1.0,
        "le_pretrain_steps": 500,
        "b_ramp_steps": 1500,
    },
    "one-case": {
        "steps": 15000,
        "hidden": 384,
        "basis": 128,
        "lr": 2.0e-4,
        "weight_decay": 1.0e-6,
        "le_weight": 1.0,
        "b_weight": 1.0,
        "sample_batch": 10,
        "le_point_batch": 128,
        "ad_point_batch": 64,
        "eval_point_batch": 128,
        "eval_every": 1000,
        "grad_clip": 1.0,
        "le_pretrain_steps": 1000,
        "b_ramp_steps": 3000,
    },
    "training-pool": {
        "steps": 30000,
        "hidden": 384,
        "basis": 128,
        "lr": 1.0e-4,
        "weight_decay": 1.0e-6,
        "le_weight": 1.0,
        "b_weight": 1.0,
        "sample_batch": 8,
        "le_point_batch": 64,
        "ad_point_batch": 32,
        "eval_point_batch": 64,
        "eval_every": 2000,
        "grad_clip": 1.0,
        "le_pretrain_steps": 2000,
        "b_ramp_steps": 5000,
    },
}
PRESET_KEYS = tuple(PRESETS["one-frame"].keys())


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scalar_text(value: Any, default: str = "unknown") -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return str(default)
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def scalar_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def rel_from_squares(diff_sq: torch.Tensor, den_sq: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(diff_sq) / torch.clamp(torch.sqrt(den_sq), min=1.0e-30)


def standardize_np(
    value: np.ndarray,
    *,
    axis: int | tuple[int, ...],
    eps: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = np.asarray(value, dtype=np.float64)
    mean = np.mean(vals, axis=axis, keepdims=True)
    std = np.std(vals, axis=axis, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (vals - mean) / std, np.squeeze(mean, axis=axis), np.squeeze(std, axis=axis)


def read_path_list(path: Path) -> list[Path]:
    return [
        Path(line.strip()).resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def load_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.compact_list:
        paths.extend(read_path_list(Path(args.compact_list).resolve()))
    paths.extend(Path(item).resolve() for item in args.compact)
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    if not out:
        raise SystemExit("Provide --compact-list or at least one --compact")
    return out


def parse_int_filter(values: list[str]) -> set[int]:
    out: set[int] = set()
    for value in values:
        for item in str(value).split(","):
            text = item.strip()
            if text:
                out.add(int(text))
    return out


def load_one(path: Path) -> dict[str, Any]:
    with np.load(str(path), allow_pickle=True) as z:
        required = {
            "q_useful_hat",
            "geometry_global_hat",
            "trunk_features_hat",
            "LE_local_stack",
            "B_local_useful_stack_hat",
            "T_eps_to_abq_stack",
            "T_q_raw_to_useful_hat",
            "B_LE128_forward",
            "standard_operator_contract_version",
            "case_id",
        }
        missing = sorted(required.difference(z.files))
        if missing:
            raise KeyError(f"{path}: missing required v3 fields {missing}")
        version = scalar_text(z["standard_operator_contract_version"])
        if version != CONTRACT_VERSION:
            raise ValueError(f"{path}: expected {CONTRACT_VERSION}, got {version}")
        old_labels = (
            bool(np.asarray(z["uses_old_true176_labels_as_v3_labels"]).reshape(-1)[0])
            if "uses_old_true176_labels_as_v3_labels" in z.files
            else False
        )
        if old_labels:
            raise ValueError(f"{path}: uses_old_true176_labels_as_v3_labels is true")
        q = np.asarray(z["q_useful_hat"], dtype=np.float64)
        geom = np.asarray(z["geometry_global_hat"], dtype=np.float64).reshape(-1)
        trunk = np.asarray(z["trunk_features_hat"], dtype=np.float64)
        le = np.asarray(z["LE_local_stack"], dtype=np.float64)
        b = np.asarray(z["B_local_useful_stack_hat"], dtype=np.float64)
        t_eps = np.asarray(z["T_eps_to_abq_stack"], dtype=np.float64)
        t_q_hat = np.asarray(z["T_q_raw_to_useful_hat"], dtype=np.float64)
        b_raw = np.asarray(z["B_LE128_forward"], dtype=np.float64)
        case_id = int(np.asarray(z["case_id"]).reshape(-1)[0])
    if q.shape[1] != 42:
        raise ValueError(f"{path}: q_useful_hat must have 42 columns, got {q.shape}")
    if le.shape != (q.shape[0], trunk.shape[0], 6):
        raise ValueError(f"{path}: LE_local_stack shape mismatch, got {le.shape}")
    if b.shape != (q.shape[0], trunk.shape[0], 6, 42):
        raise ValueError(f"{path}: B_local_useful_stack_hat shape mismatch, got {b.shape}")
    return {
        "path": str(path),
        "case_id": case_id,
        "q": q,
        "geom": geom,
        "trunk": trunk,
        "le": le,
        "b": b,
        "t_eps": t_eps,
        "t_q_hat": t_q_hat,
        "b_raw": b_raw,
    }


def load_pool(args: argparse.Namespace) -> dict[str, Any]:
    rows = sorted([load_one(path) for path in load_paths(args)], key=lambda row: int(row["case_id"]))
    filter_cases = parse_int_filter(args.filter_case)
    filter_frames = parse_int_filter(args.filter_frame)
    if filter_cases:
        rows = [row for row in rows if int(row["case_id"]) in filter_cases]
    if not rows:
        raise SystemExit("No compact matched the selected --filter-case values")
    geom_shape = rows[0]["geom"].shape
    trunk_shape = rows[0]["trunk"].shape
    for row in rows:
        if row["geom"].shape != geom_shape:
            raise ValueError("Selected compacts must share geometry_global_hat shape")
        if row["trunk"].shape != trunk_shape:
            raise ValueError("Selected compacts must share trunk_features_hat shape")

    sample_q: list[np.ndarray] = []
    sample_geom: list[np.ndarray] = []
    sample_le: list[np.ndarray] = []
    sample_b: list[np.ndarray] = []
    sample_t_eps: list[np.ndarray] = []
    sample_t_q_hat: list[np.ndarray] = []
    sample_b_raw: list[np.ndarray] = []
    sample_case_ids: list[int] = []
    sample_frame_ids: list[int] = []
    sample_compact_paths: list[str] = []
    for row in rows:
        frames = int(row["q"].shape[0])
        for frame in range(frames):
            if filter_frames and int(frame) not in filter_frames:
                continue
            sample_q.append(row["q"][frame])
            sample_geom.append(row["geom"])
            sample_le.append(row["le"][frame])
            sample_b.append(row["b"][frame])
            sample_t_eps.append(row["t_eps"])
            sample_t_q_hat.append(row["t_q_hat"])
            sample_b_raw.append(row["b_raw"][frame])
            sample_case_ids.append(int(row["case_id"]))
            sample_frame_ids.append(int(frame))
            sample_compact_paths.append(str(row["path"]))
    if not sample_q:
        raise SystemExit("No frames matched the selected --filter-frame values")

    q = np.stack(sample_q, axis=0)
    geom = np.stack(sample_geom, axis=0)
    trunk = np.stack([row["trunk"] for row in rows], axis=0)
    le = np.stack(sample_le, axis=0)
    b = np.stack(sample_b, axis=0)
    t_eps = np.stack(sample_t_eps, axis=0)
    t_q_hat = np.stack(sample_t_q_hat, axis=0)
    b_raw = np.stack(sample_b_raw, axis=0)
    q_normed, q_mean, q_std = standardize_np(q, axis=0)
    geom_norm, geom_mean, geom_std = standardize_np(geom, axis=0)
    trunk_norm, trunk_mean, trunk_std = standardize_np(trunk, axis=(0, 1))
    return {
        "case_ids": [int(row["case_id"]) for row in rows],
        "compact_paths": [str(row["path"]) for row in rows],
        "filter_case": sorted(filter_cases),
        "filter_frame": sorted(filter_frames),
        "sample_case_ids": np.asarray(sample_case_ids, dtype=np.int64),
        "sample_frame_ids": np.asarray(sample_frame_ids, dtype=np.int64),
        "sample_compact_paths": sample_compact_paths,
        "q": q.astype(np.float32),
        "q_norm": q_normed.astype(np.float32),
        "q_mean": q_mean.astype(np.float64),
        "q_std": q_std.astype(np.float64),
        "geom_norm": geom_norm.astype(np.float32),
        "geom_mean": geom_mean.astype(np.float64),
        "geom_std": geom_std.astype(np.float64),
        "trunk_norm_by_case": trunk_norm.astype(np.float32),
        "trunk": trunk.astype(np.float32),
        "trunk_mean": trunk_mean.astype(np.float64),
        "trunk_std": trunk_std.astype(np.float64),
        "le": le.astype(np.float32),
        "b": b.astype(np.float32),
        "t_eps": t_eps.astype(np.float32),
        "t_q_hat": t_q_hat.astype(np.float32),
        "b_raw": b_raw.astype(np.float32),
    }


def make_tensors(data_np: dict[str, Any], *, device: torch.device) -> dict[str, torch.Tensor]:
    dtype = torch.float32
    return {
        "q": torch.as_tensor(data_np["q"], dtype=dtype, device=device),
        "q_norm": torch.as_tensor(data_np["q_norm"], dtype=dtype, device=device),
        "q_mean": torch.as_tensor(data_np["q_mean"], dtype=dtype, device=device),
        "q_std": torch.as_tensor(data_np["q_std"], dtype=dtype, device=device),
        "geom_norm": torch.as_tensor(data_np["geom_norm"], dtype=dtype, device=device),
        "trunk_norm_by_case": torch.as_tensor(data_np["trunk_norm_by_case"], dtype=dtype, device=device),
        "le": torch.as_tensor(data_np["le"], dtype=dtype, device=device),
        "b": torch.as_tensor(data_np["b"], dtype=dtype, device=device),
        "t_eps": torch.as_tensor(data_np["t_eps"], dtype=dtype, device=device),
        "t_q_hat": torch.as_tensor(data_np["t_q_hat"], dtype=dtype, device=device),
        "b_raw": torch.as_tensor(data_np["b_raw"], dtype=dtype, device=device),
        "sample_case_ids": torch.as_tensor(data_np["sample_case_ids"], dtype=torch.long, device=device),
        "sample_frame_ids": torch.as_tensor(data_np["sample_frame_ids"], dtype=torch.long, device=device),
    }


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int) -> None:
        super().__init__()
        if int(depth) < 1:
            self.net = nn.Linear(in_dim, out_dim)
            return
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(int(depth) - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LEPrimaryFullDirectionOperator(nn.Module):
    def __init__(
        self,
        *,
        q_dim: int,
        geom_dim: int,
        trunk_dim: int,
        hidden: int = 256,
        basis: int = 96,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.q_dim = int(q_dim)
        self.hidden = int(hidden)
        self.basis = int(basis)
        self.strain_dim = 6
        out_dim = self.strain_dim * self.basis
        self.branch = MLP(self.q_dim + int(geom_dim), out_dim, int(hidden), int(depth))
        self.trunk = MLP(int(trunk_dim), out_dim, int(hidden), int(depth))
        self.bias = nn.Parameter(torch.zeros(self.strain_dim))

    def forward(self, q_norm: torch.Tensor, geom_norm: torch.Tensor, trunk_norm: torch.Tensor) -> torch.Tensor:
        if q_norm.ndim == 1:
            q_norm = q_norm.unsqueeze(0)
        batch = int(q_norm.shape[0])
        if geom_norm.ndim == 1:
            geom_norm = geom_norm.unsqueeze(0)
        if int(geom_norm.shape[0]) == 1 and batch != 1:
            geom_norm = geom_norm.expand(batch, -1)
        geom_flat = geom_norm.reshape(batch, -1)
        branch_in = torch.cat([q_norm, geom_flat], dim=-1)
        b = self.branch(branch_in).reshape(batch, self.strain_dim, self.basis)
        t = self.trunk(trunk_norm).reshape(int(trunk_norm.shape[0]), self.strain_dim, self.basis)
        le = torch.einsum("nak,pak->npa", b, t) / (self.basis ** 0.5)
        return le + self.bias.view(1, 1, self.strain_dim)


def forward_sample(
    model: LEPrimaryFullDirectionOperator,
    q_norm: torch.Tensor,
    geom_norm: torch.Tensor,
    trunk_norm: torch.Tensor,
    point_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    trunk_use = trunk_norm if point_idx is None else trunk_norm.index_select(0, point_idx)
    return model(q_norm, geom_norm, trunk_use)


def jacobian_sample(
    model: LEPrimaryFullDirectionOperator,
    q_norm: torch.Tensor,
    geom_norm: torch.Tensor,
    trunk_norm: torch.Tensor,
    q_std: torch.Tensor,
    point_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    def one(q_single_norm: torch.Tensor) -> torch.Tensor:
        return forward_sample(model, q_single_norm, geom_norm, trunk_norm, point_idx=point_idx)[0]

    jac_norm = jacrev(one)(q_norm)
    return jac_norm / q_std.reshape(1, 1, -1)


def raw_project_b(ad_b_local_hat: torch.Tensor, t_eps: torch.Tensor, t_q_hat: torch.Tensor) -> torch.Tensor:
    b_abq = torch.einsum("pab,npbk->npak", t_eps, ad_b_local_hat)
    return torch.einsum("npak,kj->npaj", b_abq, t_q_hat)


def add_to_accum(accum: dict[str, torch.Tensor], values: dict[str, torch.Tensor]) -> None:
    for key, value in values.items():
        accum[key] = accum[key] + value.detach()


def radial_transverse_squares(
    pred: torch.Tensor,
    target: torch.Tensor,
    q_raw: torch.Tensor,
) -> dict[str, torch.Tensor]:
    d = q_raw / torch.clamp(torch.linalg.vector_norm(q_raw), min=1.0e-12)
    pred_rad = torch.einsum("pak,k->pa", pred, d)
    target_rad = torch.einsum("pak,k->pa", target, d)
    pred_trans = pred - torch.einsum("pa,k->pak", pred_rad, d)
    target_trans = target - torch.einsum("pa,k->pak", target_rad, d)
    radial_diff = pred_rad - target_rad
    trans_diff = pred_trans - target_trans
    return {
        "ad_radial_diff_sq": torch.sum(radial_diff * radial_diff),
        "ad_radial_den_sq": torch.sum(target_rad * target_rad),
        "ad_trans_diff_sq": torch.sum(trans_diff * trans_diff),
        "ad_trans_den_sq": torch.sum(target_trans * target_trans),
    }


def accum_to_metrics(acc: dict[str, torch.Tensor]) -> dict[str, float]:
    le_rel = rel_from_squares(acc["le_diff_sq"], acc["le_den_sq"])
    ad_rel = rel_from_squares(acc["ad_diff_sq"], acc["ad_den_sq"])
    ad_cos = acc["ad_dot"] / torch.clamp(
        torch.sqrt(acc["ad_pred_sq"]) * torch.sqrt(acc["ad_den_sq"]),
        min=1.0e-30,
    )
    braw_rel = rel_from_squares(acc["braw_diff_sq"], acc["braw_den_sq"])
    radial_rel = rel_from_squares(acc["ad_radial_diff_sq"], acc["ad_radial_den_sq"])
    trans_rel = rel_from_squares(acc["ad_trans_diff_sq"], acc["ad_trans_den_sq"])
    return {
        "train_LE_local_stack_rel": scalar_float(le_rel),
        "train_AD_B_local_useful_hat_rel": scalar_float(ad_rel),
        "train_AD_B_local_useful_hat_cos": scalar_float(ad_cos),
        "train_AD_B_radial_rel": scalar_float(radial_rel),
        "train_AD_B_transverse_rel": scalar_float(trans_rel),
        "B_model_raw_rel": scalar_float(braw_rel),
        "selection_score": scalar_float(le_rel + ad_rel),
    }


def evaluate(
    model: LEPrimaryFullDirectionOperator,
    data: dict[str, torch.Tensor],
    case_ids: list[int],
    *,
    point_chunk: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    device = data["q"].device
    keys = [
        "le_diff_sq",
        "le_den_sq",
        "ad_diff_sq",
        "ad_den_sq",
        "ad_dot",
        "ad_pred_sq",
        "ad_radial_diff_sq",
        "ad_radial_den_sq",
        "ad_trans_diff_sq",
        "ad_trans_den_sq",
        "braw_diff_sq",
        "braw_den_sq",
    ]
    overall = {key: torch.zeros((), device=device) for key in keys}
    case_accum: dict[int, dict[str, torch.Tensor]] = {
        int(case_id): {key: torch.zeros((), device=device) for key in keys}
        for case_id in case_ids
    }
    case_to_index = {int(case_id): idx for idx, case_id in enumerate(case_ids)}
    sample_count = int(data["q"].shape[0])
    point_count = int(data["le"].shape[1])
    for sample_index in range(sample_count):
        case_id = int(data["sample_case_ids"][sample_index].detach().cpu().item())
        q_norm = data["q_norm"][sample_index]
        q_raw = data["q"][sample_index]
        geom = data["geom_norm"][sample_index].reshape(1, -1)
        case_pos = case_to_index[case_id]
        trunk = data["trunk_norm_by_case"][case_pos]
        le = data["le"][sample_index]
        b_target_full = data["b"][sample_index]
        with torch.no_grad():
            pred = forward_sample(model, q_norm, geom, trunk)[0]
            le_diff = pred - le
            le_diff_sq = torch.sum(le_diff * le_diff)
            le_den_sq = torch.sum(le * le)
        values = {
            "le_diff_sq": le_diff_sq,
            "le_den_sq": le_den_sq,
            "ad_diff_sq": torch.zeros((), device=device),
            "ad_den_sq": torch.zeros((), device=device),
            "ad_dot": torch.zeros((), device=device),
            "ad_pred_sq": torch.zeros((), device=device),
            "ad_radial_diff_sq": torch.zeros((), device=device),
            "ad_radial_den_sq": torch.zeros((), device=device),
            "ad_trans_diff_sq": torch.zeros((), device=device),
            "ad_trans_den_sq": torch.zeros((), device=device),
            "braw_diff_sq": torch.zeros((), device=device),
            "braw_den_sq": torch.zeros((), device=device),
        }
        for p0 in range(0, point_count, int(point_chunk)):
            p1 = min(p0 + int(point_chunk), point_count)
            point_idx = torch.arange(p0, p1, device=device)
            ad_b = jacobian_sample(model, q_norm, geom, trunk, data["q_std"], point_idx=point_idx)
            target = b_target_full[p0:p1]
            diff = ad_b - target
            radial_trans = radial_transverse_squares(ad_b.detach(), target, q_raw)
            with torch.no_grad():
                values["ad_diff_sq"] = values["ad_diff_sq"] + torch.sum(diff.detach() * diff.detach())
                values["ad_den_sq"] = values["ad_den_sq"] + torch.sum(target * target)
                values["ad_dot"] = values["ad_dot"] + torch.sum(ad_b.detach() * target)
                values["ad_pred_sq"] = values["ad_pred_sq"] + torch.sum(ad_b.detach() * ad_b.detach())
                for key, value in radial_trans.items():
                    values[key] = values[key] + value
                b_model_raw = raw_project_b(
                    ad_b.detach().unsqueeze(0),
                    data["t_eps"][sample_index, p0:p1],
                    data["t_q_hat"][sample_index],
                )
                b_raw_chunk = data["b_raw"][sample_index, p0:p1].unsqueeze(0)
                braw_diff = b_model_raw - b_raw_chunk
                values["braw_diff_sq"] = values["braw_diff_sq"] + torch.sum(braw_diff * braw_diff)
                values["braw_den_sq"] = values["braw_den_sq"] + torch.sum(b_raw_chunk * b_raw_chunk)
        add_to_accum(overall, values)
        add_to_accum(case_accum[case_id], values)

    case_rows: list[dict[str, Any]] = []
    for idx, case_id in enumerate(case_ids):
        row = accum_to_metrics(case_accum[int(case_id)])
        case_rows.append({
            "case_index": int(idx),
            "case_id": int(case_id),
            **row,
        })
    return accum_to_metrics(overall), case_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    preferred = [
        "step",
        "case_index",
        "case_id",
        "loss",
        "le_loss",
        "b_loss",
        "b_weight_eff",
        "train_LE_local_stack_rel",
        "train_AD_B_local_useful_hat_rel",
        "train_AD_B_local_useful_hat_cos",
        "train_AD_B_radial_rel",
        "train_AD_B_transverse_rel",
        "B_model_raw_rel",
        "selection_score",
    ]
    fields = {key for row in rows for key in row}
    ordered = [key for key in preferred if key in fields]
    ordered.extend(sorted(fields.difference(ordered)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def make_component_scales(
    data: dict[str, torch.Tensor],
    case_ids: list[int],
    *,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = data["q"].device
    le_scales: list[torch.Tensor] = []
    b_scales: list[torch.Tensor] = []
    for case_id in case_ids:
        mask = data["sample_case_ids"] == int(case_id)
        le_scale = torch.sqrt(torch.mean(data["le"][mask] * data["le"][mask], dim=(0, 1)))
        b_scale = torch.sqrt(torch.mean(data["b"][mask] * data["b"][mask], dim=(0, 1)))
        le_scales.append(torch.clamp(le_scale, min=float(eps)).reshape(1, 6))
        b_scales.append(torch.clamp(b_scale, min=float(eps)).reshape(1, 6, 42))
    return torch.stack(le_scales, dim=0).to(device), torch.stack(b_scales, dim=0).to(device)


def b_weight_for_step(step: int, args: argparse.Namespace) -> float:
    if float(args.b_weight) == 0.0:
        return 0.0
    pretrain = int(args.le_pretrain_steps)
    ramp = int(args.b_ramp_steps)
    if step <= pretrain:
        return 0.0
    if ramp <= 0:
        return float(args.b_weight)
    tau = min(1.0, float(step - pretrain) / float(ramp))
    return float(args.b_weight) * tau


def apply_preset_defaults(args: argparse.Namespace) -> argparse.Namespace:
    preset = PRESETS[str(args.preset)]
    for key in PRESET_KEYS:
        if getattr(args, key) is None:
            setattr(args, key, preset[key])
    return args


def train(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))
    data_np = load_pool(args)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    data = make_tensors(data_np, device=device)
    case_ids = list(data_np["case_ids"])
    model = LEPrimaryFullDirectionOperator(
        q_dim=int(data["q"].shape[1]),
        geom_dim=int(data["geom_norm"].shape[1]),
        trunk_dim=int(data["trunk_norm_by_case"].shape[2]),
        hidden=int(args.hidden),
        basis=int(args.basis),
        depth=int(args.depth),
    ).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    sample_count = int(data["q"].shape[0])
    point_count = int(data["le"].shape[1])
    sample_batch = min(int(args.sample_batch), sample_count)
    le_point_batch = min(int(args.le_point_batch), point_count)
    ad_point_batch = min(int(args.ad_point_batch), point_count)
    use_le_loss = float(args.le_weight) != 0.0
    use_ad_loss = float(args.b_weight) != 0.0
    if not (use_le_loss or use_ad_loss):
        raise SystemExit("At least one of --le-weight or --b-weight must be non-zero")
    le_scale_case, b_scale_case = make_component_scales(data, case_ids)
    case_to_index = {int(case_id): idx for idx, case_id in enumerate(case_ids)}
    history: list[dict[str, Any]] = []
    case_history: list[dict[str, Any]] = []
    initial, initial_cases = evaluate(model, data, case_ids, point_chunk=int(args.eval_point_batch))
    best = {"step": 0, "b_weight_eff": 0.0, **initial}
    history.append(best)
    for row in initial_cases:
        case_history.append({"step": 0, **row})
    eval_steps = set(range(0, int(args.steps) + 1, int(args.eval_every)))
    eval_steps.add(int(args.steps))
    for step in range(1, int(args.steps) + 1):
        model.train()
        b_weight_eff = b_weight_for_step(step, args)
        sample_idx = torch.randperm(sample_count, device=device)[:sample_batch]
        losses: list[torch.Tensor] = []
        le_terms: list[torch.Tensor] = []
        b_terms: list[torch.Tensor] = []
        for sample_index in sample_idx.tolist():
            case_id = int(data["sample_case_ids"][sample_index].detach().cpu().item())
            case_pos = int(case_to_index[case_id])
            q_norm = data["q_norm"][sample_index]
            geom = data["geom_norm"][sample_index].reshape(1, -1)
            trunk = data["trunk_norm_by_case"][case_pos]
            sample_loss = torch.zeros((), device=device)
            if use_le_loss:
                le_idx = torch.randperm(point_count, device=device)[:le_point_batch]
                pred = forward_sample(model, q_norm, geom, trunk, point_idx=le_idx)[0]
                le_target = data["le"][sample_index].index_select(0, le_idx)
                le_loss = torch.mean(((pred - le_target) / le_scale_case[case_pos]) ** 2)
                sample_loss = sample_loss + float(args.le_weight) * le_loss
                le_terms.append(le_loss.detach())
            if use_ad_loss and b_weight_eff != 0.0:
                ad_idx = torch.randperm(point_count, device=device)[:ad_point_batch]
                b_target = data["b"][sample_index].index_select(0, ad_idx)
                ad_b = jacobian_sample(model, q_norm, geom, trunk, data["q_std"], point_idx=ad_idx)
                b_loss = torch.mean(((ad_b - b_target) / b_scale_case[case_pos]) ** 2)
                sample_loss = sample_loss + float(b_weight_eff) * b_loss
                b_terms.append(b_loss.detach())
            losses.append(sample_loss)
        loss = torch.stack(losses).mean()
        if not loss.requires_grad:
            raise SystemExit("The effective loss has no gradient; check --le-weight, --b-weight, and pretrain settings")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        opt.step()
        if step in eval_steps:
            metrics, case_rows = evaluate(model, data, case_ids, point_chunk=int(args.eval_point_batch))
            row = {
                "step": int(step),
                "loss": scalar_float(loss.detach()),
                "le_loss": scalar_float(torch.stack(le_terms).mean()) if le_terms else None,
                "b_loss": scalar_float(torch.stack(b_terms).mean()) if b_terms else None,
                "b_weight_eff": float(b_weight_eff),
                **metrics,
            }
            history.append(row)
            for case_row in case_rows:
                case_history.append({"step": int(step), **case_row})
            if float(row["selection_score"]) < float(best["selection_score"]):
                best = dict(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    latest = history[-1]
    latest_cases = [row for row in case_history if int(row["step"]) == int(latest["step"])]
    case031_latest = next((row for row in latest_cases if int(row["case_id"]) == 31), None)
    summary = {
        "audit_name": "v3_le_primary_full_direction_smoke",
        "out_root": str(out_root),
        "device": str(device),
        "seed": int(args.seed),
        "preset": str(args.preset),
        "steps": int(args.steps),
        "case_ids": case_ids,
        "case_count": len(case_ids),
        "sample_count": sample_count,
        "point_count": point_count,
        "q_useful_hat_dim": int(data["q"].shape[1]),
        "geometry_global_hat_dim": int(data["geom_norm"].shape[1]),
        "trunk_features_hat_dim": int(data["trunk_norm_by_case"].shape[2]),
        "model_class": "LEPrimaryFullDirectionOperator",
        "model_inputs": ["q_useful_hat", "geometry_global_hat", "trunk_features_hat"],
        "model_output": "LE_local_stack",
        "ad_target": "dLE_local_stack/dq_useful_hat",
        "operator_form": "DeepONet branch(q_norm,geometry_norm) dot trunk(point_norm)",
        "formal_training": False,
        "checkpoint_written": False,
        "held_out_split": False,
        "uses_case_id_anchor": False,
        "uses_direct_b_head": False,
        "uses_radial_loss": False,
        "uses_old_true176_labels_as_v3_labels": False,
        "loss_weights": {
            "le_weight": float(args.le_weight),
            "b_weight": float(args.b_weight),
        },
        "curriculum": {
            "le_pretrain_steps": int(args.le_pretrain_steps),
            "b_ramp_steps": int(args.b_ramp_steps),
        },
        "normalization": {
            "le_scale_case_shape": [len(case_ids), 1, 6],
            "b_scale_case_shape": [len(case_ids), 1, 6, 42],
            "le_scale": "per case and strain component RMS",
            "b_scale": "per case, strain component, and q direction RMS",
        },
        "active_losses": {
            "le_loss": use_le_loss,
            "ad_b_loss": use_ad_loss,
            "direct_b_head_loss": False,
            "radial_loss": False,
        },
        "filter_case": data_np["filter_case"],
        "filter_frame": data_np["filter_frame"],
        "initial_metrics": initial,
        "best_metrics": best,
        "latest_metrics": latest,
        "latest_case_metrics": latest_cases,
        "case031_latest_metrics": case031_latest,
    }
    summary_path = out_root / "v3_le_primary_full_direction_summary.json"
    history_path = out_root / "v3_le_primary_full_direction_history.csv"
    case_history_path = out_root / "v3_le_primary_full_direction_case_history.csv"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    write_csv(history_path, history)
    write_csv(case_history_path, case_history)
    print(json.dumps(
        {
            **summary,
            "summary_path": str(summary_path),
            "history_path": str(history_path),
            "case_history_path": str(case_history_path),
        },
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
        default=json_default,
    ))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", default="")
    parser.add_argument("--compact", action="append", default=[])
    parser.add_argument("--filter-case", action="append", default=[], help="Optional comma-separated case_id filter.")
    parser.add_argument("--filter-frame", action="append", default=[], help="Optional comma-separated frame index filter.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="one-frame")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--basis", type=int, default=None)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--le-weight", type=float, default=None)
    parser.add_argument("--b-weight", type=float, default=None)
    parser.add_argument("--le-pretrain-steps", type=int, default=None)
    parser.add_argument("--b-ramp-steps", type=int, default=None)
    parser.add_argument("--sample-batch", type=int, default=None)
    parser.add_argument("--le-point-batch", type=int, default=None)
    parser.add_argument("--ad-point-batch", type=int, default=None)
    parser.add_argument("--eval-point-batch", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    args = apply_preset_defaults(parser.parse_args())
    if int(args.steps) < 0:
        raise SystemExit("--steps must be non-negative")
    if int(args.eval_every) < 1:
        raise SystemExit("--eval-every must be positive")
    if int(args.basis) < 1:
        raise SystemExit("--basis must be positive")
    if int(args.le_pretrain_steps) < 0:
        raise SystemExit("--le-pretrain-steps must be non-negative")
    if int(args.b_ramp_steps) < 0:
        raise SystemExit("--b-ramp-steps must be non-negative")
    train(args)


if __name__ == "__main__":
    main()
