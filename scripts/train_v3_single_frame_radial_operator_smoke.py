#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-frame radial-operator training-pool smoke for v3 CSS8 compacts.

This prototype matches the intended inference interface:

    input:  one current q_useful_hat frame + geometry + query point
    output: LE_local_stack frame
    AD:     dLE_local_stack / dq_useful_hat

There is no case-id anchor and no multi-frame path at inference time. Each frame
defines its own radial path:

    s = ||q||
    d = q / (s + eps)
    q_radial = s * stop_gradient(d)
    q_perp = q - q_radial

The model predicts V(s,d,geometry,point) and B(s,d,geometry,point), then uses:

    LE_hat(q,point) = V_hat + B_hat @ q_perp

AD-B supervision is computed by autograd with respect to the normalized q input
and mapped back to q_useful_hat units.

This is a training-pool smoke only. It writes JSON/CSV metrics but no
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
from torch.func import jacrev, vmap


CONTRACT_VERSION = "v3-css8-standard-operator-001"


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


def standardize_np(value: np.ndarray, *, axis: int | tuple[int, ...], eps: float = 1.0e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        old_labels = bool(np.asarray(z["uses_old_true176_labels_as_v3_labels"]).reshape(-1)[0]) if "uses_old_true176_labels_as_v3_labels" in z.files else False
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
    out = {
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
    return out


class SingleFrameRadialOperator(nn.Module):
    def __init__(self, *, q_dim: int, geom_dim: int, trunk_dim: int, hidden: int) -> None:
        super().__init__()
        self.q_dim = int(q_dim)
        self.state_net = nn.Sequential(
            nn.Linear(geom_dim + q_dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.point_net = nn.Sequential(
            nn.Linear(trunk_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.v_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 6))
        self.b_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 6 * q_dim))
        nn.init.zeros_(self.v_head[-1].weight)
        nn.init.zeros_(self.v_head[-1].bias)
        nn.init.zeros_(self.b_head[-1].weight)
        nn.init.zeros_(self.b_head[-1].bias)

    def radial_features(self, q_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if q_raw.ndim == 1:
            q_raw = q_raw.unsqueeze(0)
        s = torch.linalg.vector_norm(q_raw, dim=1, keepdim=True)
        d = q_raw / torch.clamp(s, min=1.0e-12)
        d_detached = d.detach()
        q_radial = s * d_detached
        q_perp = q_raw - q_radial
        return s, d_detached, q_perp, d

    def vb(
        self,
        q_raw: torch.Tensor,
        geom_norm: torch.Tensor,
        trunk_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if q_raw.ndim == 1:
            q_raw = q_raw.unsqueeze(0)
        s, d_detached, q_perp, d = self.radial_features(q_raw)
        batch = int(q_raw.shape[0])
        points = int(trunk_norm.shape[0])
        geom_expand = geom_norm.reshape(batch, -1)
        state = self.state_net(torch.cat([geom_expand, d_detached, s], dim=-1))
        point = self.point_net(trunk_norm)
        state_expand = state[:, None, :].expand(batch, points, state.shape[-1])
        point_expand = point[None, :, :].expand(batch, points, point.shape[-1])
        fused = torch.cat([state_expand, point_expand], dim=-1).reshape(batch * points, -1)
        v = self.v_head(fused).reshape(batch, points, 6)
        b = self.b_head(fused).reshape(batch, points, 6, self.q_dim)
        return v, b, q_perp, d_detached

    def forward(
        self,
        q_raw: torch.Tensor,
        geom_norm: torch.Tensor,
        trunk_norm: torch.Tensor,
    ) -> torch.Tensor:
        v, b, q_perp, _ = self.vb(q_raw, geom_norm, trunk_norm)
        return v + torch.einsum("npak,nk->npa", b, q_perp)


def forward_sample(
    model: SingleFrameRadialOperator,
    q_norm: torch.Tensor,
    geom_norm: torch.Tensor,
    trunk_norm: torch.Tensor,
    q_mean: torch.Tensor,
    q_std: torch.Tensor,
    point_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    q_raw = q_norm * q_std + q_mean
    trunk_use = trunk_norm if point_idx is None else trunk_norm.index_select(0, point_idx)
    return model(q_raw, geom_norm, trunk_use)


def jacobian_sample(
    model: SingleFrameRadialOperator,
    q_norm: torch.Tensor,
    geom_norm: torch.Tensor,
    trunk_norm: torch.Tensor,
    q_mean: torch.Tensor,
    q_std: torch.Tensor,
    point_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    def one(q_single_norm: torch.Tensor) -> torch.Tensor:
        return forward_sample(model, q_single_norm, geom_norm, trunk_norm, q_mean, q_std, point_idx=point_idx)[0]

    jac_norm = jacrev(one)(q_norm)
    return jac_norm / q_std.reshape(1, 1, -1)


def radial_consistency_loss(
    model: SingleFrameRadialOperator,
    q_raw: torch.Tensor,
    geom_norm: torch.Tensor,
    trunk_norm: torch.Tensor,
    point_idx: torch.Tensor,
) -> torch.Tensor:
    q_leaf = q_raw.detach().clone().requires_grad_(True)
    trunk_use = trunk_norm.index_select(0, point_idx)
    v, b_hat, _, d_detached = model.vb(q_leaf, geom_norm, trunk_use)
    batch, points, comps = v.shape
    rows = []
    for point in range(points):
        comp_rows = []
        for comp in range(comps):
            grad = torch.autograd.grad(v[:, point, comp].sum(), q_leaf, create_graph=True, retain_graph=True)[0]
            comp_rows.append(grad)
        rows.append(torch.stack(comp_rows, dim=1))
    dv_dq = torch.stack(rows, dim=1)
    b_radial = torch.einsum("npak,nk->npa", b_hat, d_detached)
    dv_ds = torch.einsum("npak,nk->npa", dv_dq, d_detached)
    return torch.mean((dv_ds - b_radial) ** 2)


def direct_b_head(
    model: SingleFrameRadialOperator,
    q_raw: torch.Tensor,
    geom_norm: torch.Tensor,
    trunk_norm: torch.Tensor,
    point_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    trunk_use = trunk_norm if point_idx is None else trunk_norm.index_select(0, point_idx)
    _, b_hat, _, _ = model.vb(q_raw, geom_norm, trunk_use)
    return b_hat


def raw_project_b(ad_b_local_hat: torch.Tensor, t_eps: torch.Tensor, t_q_hat: torch.Tensor) -> torch.Tensor:
    b_abq = torch.einsum("pab,npbk->npak", t_eps, ad_b_local_hat)
    return torch.einsum("npak,kj->npaj", b_abq, t_q_hat)


def evaluate(
    model: SingleFrameRadialOperator,
    data: dict[str, torch.Tensor],
    case_ids: list[int],
    *,
    point_chunk: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    device = data["q"].device
    overall = {
        "le_diff_sq": torch.zeros((), device=device),
        "le_den_sq": torch.zeros((), device=device),
        "ad_diff_sq": torch.zeros((), device=device),
        "ad_den_sq": torch.zeros((), device=device),
        "ad_dot": torch.zeros((), device=device),
        "ad_pred_sq": torch.zeros((), device=device),
        "braw_diff_sq": torch.zeros((), device=device),
        "braw_den_sq": torch.zeros((), device=device),
        "bhead_diff_sq": torch.zeros((), device=device),
        "bhead_den_sq": torch.zeros((), device=device),
        "bhead_dot": torch.zeros((), device=device),
        "bhead_pred_sq": torch.zeros((), device=device),
    }
    case_accum: dict[int, dict[str, torch.Tensor]] = {
        int(case_id): {key: torch.zeros((), device=device) for key in overall}
        for case_id in case_ids
    }
    sample_count = int(data["q"].shape[0])
    point_count = int(data["le"].shape[1])
    for sample_index in range(sample_count):
        case_id = int(data["sample_case_ids"][sample_index].detach().cpu().item())
        q_norm = data["q_norm"][sample_index]
        q_raw = data["q"][sample_index]
        geom = data["geom_norm"][sample_index].reshape(1, -1)
        case_pos = case_ids.index(case_id)
        trunk = data["trunk_norm_by_case"][case_pos]
        le = data["le"][sample_index]
        b_target_full = data["b"][sample_index]
        with torch.no_grad():
            pred = forward_sample(model, q_norm, geom, trunk, data["q_mean"], data["q_std"])[0]
            le_diff = pred - le
            le_diff_sq = torch.sum(le_diff * le_diff)
            le_den_sq = torch.sum(le * le)
        ad_diff_sq = torch.zeros((), device=device)
        ad_den_sq = torch.zeros((), device=device)
        ad_dot = torch.zeros((), device=device)
        ad_pred_sq = torch.zeros((), device=device)
        braw_diff_sq = torch.zeros((), device=device)
        braw_den_sq = torch.zeros((), device=device)
        bhead_diff_sq = torch.zeros((), device=device)
        bhead_den_sq = torch.zeros((), device=device)
        bhead_dot = torch.zeros((), device=device)
        bhead_pred_sq = torch.zeros((), device=device)
        for p0 in range(0, point_count, int(point_chunk)):
            p1 = min(p0 + int(point_chunk), point_count)
            point_idx = torch.arange(p0, p1, device=device)
            ad_b = jacobian_sample(model, q_norm, geom, trunk, data["q_mean"], data["q_std"], point_idx=point_idx)
            b_hat = direct_b_head(model, q_raw.reshape(1, -1), geom, trunk, point_idx=point_idx)[0]
            target = b_target_full[p0:p1]
            diff = ad_b - target
            bhead_diff = b_hat.detach() - target
            with torch.no_grad():
                ad_diff_sq = ad_diff_sq + torch.sum(diff.detach() * diff.detach())
                ad_den_sq = ad_den_sq + torch.sum(target * target)
                ad_dot = ad_dot + torch.sum(ad_b.detach() * target)
                ad_pred_sq = ad_pred_sq + torch.sum(ad_b.detach() * ad_b.detach())
                bhead_diff_sq = bhead_diff_sq + torch.sum(bhead_diff * bhead_diff)
                bhead_den_sq = bhead_den_sq + torch.sum(target * target)
                bhead_dot = bhead_dot + torch.sum(b_hat.detach() * target)
                bhead_pred_sq = bhead_pred_sq + torch.sum(b_hat.detach() * b_hat.detach())
                b_model_raw = raw_project_b(ad_b.detach().unsqueeze(0), data["t_eps"][sample_index, p0:p1], data["t_q_hat"][sample_index])
                b_raw_chunk = data["b_raw"][sample_index, p0:p1].unsqueeze(0)
                braw_diff = b_model_raw - b_raw_chunk
                braw_diff_sq = braw_diff_sq + torch.sum(braw_diff * braw_diff)
                braw_den_sq = braw_den_sq + torch.sum(b_raw_chunk * b_raw_chunk)
        for key, val in [
            ("le_diff_sq", le_diff_sq),
            ("le_den_sq", le_den_sq),
            ("ad_diff_sq", ad_diff_sq),
            ("ad_den_sq", ad_den_sq),
            ("ad_dot", ad_dot),
            ("ad_pred_sq", ad_pred_sq),
            ("braw_diff_sq", braw_diff_sq),
            ("braw_den_sq", braw_den_sq),
            ("bhead_diff_sq", bhead_diff_sq),
            ("bhead_den_sq", bhead_den_sq),
            ("bhead_dot", bhead_dot),
            ("bhead_pred_sq", bhead_pred_sq),
        ]:
            overall[key] = overall[key] + val.detach()
            case_accum[case_id][key] = case_accum[case_id][key] + val.detach()
    case_rows: list[dict[str, Any]] = []
    for idx, case_id in enumerate(case_ids):
        acc = case_accum[int(case_id)]
        le_rel = rel_from_squares(acc["le_diff_sq"], acc["le_den_sq"])
        ad_rel = rel_from_squares(acc["ad_diff_sq"], acc["ad_den_sq"])
        ad_cos = acc["ad_dot"] / torch.clamp(torch.sqrt(acc["ad_pred_sq"]) * torch.sqrt(acc["ad_den_sq"]), min=1.0e-30)
        braw_rel = rel_from_squares(acc["braw_diff_sq"], acc["braw_den_sq"])
        bhead_rel = rel_from_squares(acc["bhead_diff_sq"], acc["bhead_den_sq"])
        bhead_cos = acc["bhead_dot"] / torch.clamp(torch.sqrt(acc["bhead_pred_sq"]) * torch.sqrt(acc["bhead_den_sq"]), min=1.0e-30)
        case_rows.append({
            "case_index": int(idx),
            "case_id": int(case_id),
            "train_LE_local_stack_rel": scalar_float(le_rel),
            "train_AD_B_local_useful_hat_rel": scalar_float(ad_rel),
            "train_AD_B_local_useful_hat_cos": scalar_float(ad_cos),
            "B_model_raw_rel": scalar_float(braw_rel),
            "B_hat_direct_rel": scalar_float(bhead_rel),
            "B_hat_direct_cos": scalar_float(bhead_cos),
            "selection_score": scalar_float(le_rel + ad_rel),
        })
    le_rel = rel_from_squares(overall["le_diff_sq"], overall["le_den_sq"])
    ad_rel = rel_from_squares(overall["ad_diff_sq"], overall["ad_den_sq"])
    ad_cos = overall["ad_dot"] / torch.clamp(torch.sqrt(overall["ad_pred_sq"]) * torch.sqrt(overall["ad_den_sq"]), min=1.0e-30)
    braw_rel = rel_from_squares(overall["braw_diff_sq"], overall["braw_den_sq"])
    bhead_rel = rel_from_squares(overall["bhead_diff_sq"], overall["bhead_den_sq"])
    bhead_cos = overall["bhead_dot"] / torch.clamp(torch.sqrt(overall["bhead_pred_sq"]) * torch.sqrt(overall["bhead_den_sq"]), min=1.0e-30)
    return {
        "train_LE_local_stack_rel": scalar_float(le_rel),
        "train_AD_B_local_useful_hat_rel": scalar_float(ad_rel),
        "train_AD_B_local_useful_hat_cos": scalar_float(ad_cos),
        "B_model_raw_rel": scalar_float(braw_rel),
        "B_hat_direct_rel": scalar_float(bhead_rel),
        "B_hat_direct_cos": scalar_float(bhead_cos),
        "selection_score": scalar_float(le_rel + ad_rel),
    }, case_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    preferred = [
        "step", "case_index", "case_id", "loss", "le_loss", "b_loss", "direct_b_head_loss", "radial_loss",
        "train_LE_local_stack_rel", "train_AD_B_local_useful_hat_rel",
        "train_AD_B_local_useful_hat_cos", "B_model_raw_rel",
        "B_hat_direct_rel", "B_hat_direct_cos", "selection_score",
    ]
    fields = {key for row in rows for key in row}
    ordered = [key for key in preferred if key in fields]
    ordered.extend(sorted(fields.difference(ordered)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def train(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))
    data_np = load_pool(args)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    data = make_tensors(data_np, device=device)
    case_ids = list(data_np["case_ids"])
    model = SingleFrameRadialOperator(
        q_dim=int(data["q"].shape[1]),
        geom_dim=int(data["geom_norm"].shape[1]),
        trunk_dim=int(data["trunk_norm_by_case"].shape[2]),
        hidden=int(args.hidden),
    ).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    sample_count = int(data["q"].shape[0])
    point_count = int(data["le"].shape[1])
    sample_batch = min(int(args.sample_batch), sample_count)
    le_point_batch = min(int(args.le_point_batch), point_count)
    ad_point_batch = min(int(args.ad_point_batch), point_count)
    use_le_loss = float(args.le_weight) != 0.0
    use_ad_loss = float(args.b_weight) != 0.0
    use_direct_b_head_loss = float(args.direct_b_head_weight) != 0.0
    use_radial_loss = float(args.radial_weight) != 0.0
    if not (use_le_loss or use_ad_loss or use_direct_b_head_loss or use_radial_loss):
        raise SystemExit("At least one loss weight must be non-zero")
    le_scale_case = torch.zeros(len(case_ids), device=device)
    b_scale_case = torch.zeros(len(case_ids), device=device)
    for idx, case_id in enumerate(case_ids):
        mask = data["sample_case_ids"] == int(case_id)
        le_scale_case[idx] = torch.clamp(torch.sqrt(torch.mean(data["le"][mask] * data["le"][mask])), min=1.0e-12)
        b_scale_case[idx] = torch.clamp(torch.sqrt(torch.mean(data["b"][mask] * data["b"][mask])), min=1.0e-12)
    case_to_index = {int(case_id): idx for idx, case_id in enumerate(case_ids)}
    history: list[dict[str, Any]] = []
    case_history: list[dict[str, Any]] = []
    initial, initial_cases = evaluate(model, data, case_ids, point_chunk=int(args.eval_point_batch))
    best = {"step": 0, **initial}
    history.append(best)
    for row in initial_cases:
        case_history.append({"step": 0, **row})
    eval_steps = set(range(0, int(args.steps) + 1, int(args.eval_every)))
    eval_steps.add(int(args.steps))
    for step in range(1, int(args.steps) + 1):
        model.train()
        sample_idx = torch.randperm(sample_count, device=device)[:sample_batch]
        losses: list[torch.Tensor] = []
        le_terms: list[torch.Tensor] = []
        b_terms: list[torch.Tensor] = []
        direct_b_head_terms: list[torch.Tensor] = []
        radial_terms: list[torch.Tensor] = []
        for sample_index in sample_idx.tolist():
            case_id = int(data["sample_case_ids"][sample_index].detach().cpu().item())
            case_pos = int(case_to_index[case_id])
            q_raw = data["q"][sample_index].reshape(1, -1)
            q_norm = data["q_norm"][sample_index]
            geom = data["geom_norm"][sample_index].reshape(1, -1)
            trunk = data["trunk_norm_by_case"][case_pos]
            sample_loss = torch.zeros((), device=device)
            if use_le_loss:
                le_idx = torch.randperm(point_count, device=device)[:le_point_batch]
                pred = forward_sample(model, q_norm, geom, trunk, data["q_mean"], data["q_std"], point_idx=le_idx)[0]
                le_target = data["le"][sample_index].index_select(0, le_idx)
                le_loss = torch.mean(((pred - le_target) / le_scale_case[case_pos]) ** 2)
                sample_loss = sample_loss + float(args.le_weight) * le_loss
                le_terms.append(le_loss.detach())
            if use_ad_loss or use_direct_b_head_loss or use_radial_loss:
                ad_idx = torch.randperm(point_count, device=device)[:ad_point_batch]
                b_target = data["b"][sample_index].index_select(0, ad_idx)
            if use_ad_loss:
                ad_b = jacobian_sample(model, q_norm, geom, trunk, data["q_mean"], data["q_std"], point_idx=ad_idx)
                b_loss = torch.mean(((ad_b - b_target) / b_scale_case[case_pos]) ** 2)
                sample_loss = sample_loss + float(args.b_weight) * b_loss
                b_terms.append(b_loss.detach())
            if use_direct_b_head_loss:
                b_hat_direct = direct_b_head(model, q_raw, geom, trunk, point_idx=ad_idx)[0]
                direct_b_head_loss = torch.mean(((b_hat_direct - b_target) / b_scale_case[case_pos]) ** 2)
                sample_loss = sample_loss + float(args.direct_b_head_weight) * direct_b_head_loss
                direct_b_head_terms.append(direct_b_head_loss.detach())
            if use_radial_loss:
                radial_loss = radial_consistency_loss(model, q_raw, geom, trunk, ad_idx) / (b_scale_case[case_pos] ** 2)
                sample_loss = sample_loss + float(args.radial_weight) * radial_loss
                radial_terms.append(radial_loss.detach())
            losses.append(sample_loss)
        loss = torch.stack(losses).mean()
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
                "direct_b_head_loss": scalar_float(torch.stack(direct_b_head_terms).mean()) if direct_b_head_terms else None,
                "radial_loss": scalar_float(torch.stack(radial_terms).mean()) if radial_terms else None,
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
        "audit_name": "v3_single_frame_radial_operator_smoke",
        "out_root": str(out_root),
        "device": str(device),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "case_ids": case_ids,
        "case_count": len(case_ids),
        "sample_count": sample_count,
        "point_count": point_count,
        "q_useful_hat_dim": int(data["q"].shape[1]),
        "geometry_global_hat_dim": int(data["geom_norm"].shape[1]),
        "trunk_features_hat_dim": int(data["trunk_norm_by_case"].shape[2]),
        "model_inputs": ["single_frame_q_useful_hat", "geometry_global_hat", "trunk_features_hat"],
        "radial_definition": "s=||q||, d=stop_gradient(q/(s+eps)), q_perp=q-s*d",
        "model_output": "LE_local_stack",
        "ad_target": "dLE_local_stack/dq_useful_hat",
        "formal_training": False,
        "checkpoint_written": False,
        "held_out_split": False,
        "uses_case_id_anchor": False,
        "uses_old_true176_labels_as_v3_labels": False,
        "loss_weights": {
            "le_weight": float(args.le_weight),
            "b_weight": float(args.b_weight),
            "direct_b_head_weight": float(args.direct_b_head_weight),
            "radial_weight": float(args.radial_weight),
        },
        "active_losses": {
            "le_loss": use_le_loss,
            "ad_b_loss": use_ad_loss,
            "direct_b_head_loss": use_direct_b_head_loss,
            "radial_loss": use_radial_loss,
        },
        "filter_case": data_np["filter_case"],
        "filter_frame": data_np["filter_frame"],
        "initial_metrics": initial,
        "best_metrics": best,
        "latest_metrics": latest,
        "latest_case_metrics": latest_cases,
        "case031_latest_metrics": case031_latest,
    }
    summary_path = out_root / "v3_single_frame_radial_operator_summary.json"
    history_path = out_root / "v3_single_frame_radial_operator_history.csv"
    case_history_path = out_root / "v3_single_frame_radial_operator_case_history.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    write_csv(history_path, history)
    write_csv(case_history_path, case_history)
    print(json.dumps({**summary, "summary_path": str(summary_path), "history_path": str(history_path), "case_history_path": str(case_history_path)}, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", default="")
    parser.add_argument("--compact", action="append", default=[])
    parser.add_argument("--filter-case", action="append", default=[], help="Optional comma-separated case_id filter for overfit diagnostics.")
    parser.add_argument("--filter-frame", action="append", default=[], help="Optional comma-separated frame index filter for overfit diagnostics.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--le-weight", type=float, default=1.0)
    parser.add_argument("--b-weight", type=float, default=0.1)
    parser.add_argument("--direct-b-head-weight", type=float, default=0.0)
    parser.add_argument("--radial-weight", type=float, default=0.1)
    parser.add_argument("--sample-batch", type=int, default=8)
    parser.add_argument("--le-point-batch", type=int, default=128)
    parser.add_argument("--ad-point-batch", type=int, default=8)
    parser.add_argument("--eval-point-batch", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    args = parser.parse_args()
    if int(args.steps) < 0:
        raise SystemExit("--steps must be non-negative")
    if int(args.eval_every) < 1:
        raise SystemExit("--eval-every must be positive")
    train(args)


if __name__ == "__main__":
    main()
