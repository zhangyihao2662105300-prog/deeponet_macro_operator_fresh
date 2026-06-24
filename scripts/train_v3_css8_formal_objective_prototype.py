#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prototype v3 CSS8 formal objective on the 10-case training pool.

This script is still a prototype, not the production trainer.  It tests the
next objective design before any held-out split:

* train-only multi-case normalization
* per-case/relative LE and B scaling
* an affine or affine-quadratic value anchor:

      LE_anchor = LE_mean(case,point)
                + B_mean(case,point) @ (q_useful_hat - q_mean(case))

      s = <q_useful_hat - q_mean(case), q_dir(case)>
      LE_anchor_quadratic = LE_anchor
                          + C0(case,point)
                          + C1(case,point) * s
                          + C2(case,point) * s^2

  or a derivative-consistent tangent polynomial anchor:

      B_anchor(s) = B0 + B1*s + B2*s^2 + B3*s^3
      dLE_anchor/dq = B_anchor(s) on the scalar path

* residual correction with q=0 anchoring
* AD-B supervision with respect to q_useful_hat

The goal is to verify that the formal objective can stably overfit the existing
10 training cases without sacrificing low-amplitude cases, and to quantify
whether case031 remains a value-anchor hard case.
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
        case_id = int(np.asarray(z["case_id"]).reshape(-1)[0])
        q = np.asarray(z["q_useful_hat"], dtype=np.float64)
        geom = np.asarray(z["geometry_global_hat"], dtype=np.float64).reshape(-1)
        trunk = np.asarray(z["trunk_features_hat"], dtype=np.float64)
        le = np.asarray(z["LE_local_stack"], dtype=np.float64)
        b = np.asarray(z["B_local_useful_stack_hat"], dtype=np.float64)
        t_eps = np.asarray(z["T_eps_to_abq_stack"], dtype=np.float64)
        t_q_hat = np.asarray(z["T_q_raw_to_useful_hat"], dtype=np.float64)
        b_raw = np.asarray(z["B_LE128_forward"], dtype=np.float64)
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
    if len(rows) < 2:
        raise SystemExit("Prototype expects at least two cases")
    q_shape = rows[0]["q"].shape
    geom_shape = rows[0]["geom"].shape
    trunk_shape = rows[0]["trunk"].shape
    for row in rows:
        if row["q"].shape != q_shape:
            raise ValueError("Selected compacts must share q_useful_hat shape")
        if row["geom"].shape != geom_shape:
            raise ValueError("Selected compacts must share geometry_global_hat shape")
        if row["trunk"].shape != trunk_shape:
            raise ValueError("Selected compacts must share trunk_features_hat shape")
    q = np.stack([row["q"] for row in rows], axis=0)
    geom = np.stack([row["geom"] for row in rows], axis=0)
    trunk = np.stack([row["trunk"] for row in rows], axis=0)
    le = np.stack([row["le"] for row in rows], axis=0)
    b = np.stack([row["b"] for row in rows], axis=0)
    t_eps = np.stack([row["t_eps"] for row in rows], axis=0)
    t_q_hat = np.stack([row["t_q_hat"] for row in rows], axis=0)
    b_raw = np.stack([row["b_raw"] for row in rows], axis=0)
    q_norm, q_mean_global, q_std = standardize_np(q, axis=(0, 1))
    geom_norm, geom_mean, geom_std = standardize_np(geom, axis=0)
    trunk_norm, trunk_mean, trunk_std = standardize_np(trunk, axis=(0, 1))
    le_case_mean = np.mean(le, axis=1)
    b_case_mean = np.mean(b, axis=1)
    q_case_mean = np.mean(q, axis=1)
    q_dir_norm = np.linalg.norm(q, axis=2)
    q_frame_dir = q / np.maximum(q_dir_norm[:, :, None], 1.0e-30)
    q_case_dir = np.mean(q_frame_dir, axis=1)
    q_case_dir = q_case_dir / np.maximum(np.linalg.norm(q_case_dir, axis=1, keepdims=True), 1.0e-30)
    quadratic_coeff = np.zeros((len(rows), 3, le.shape[2], le.shape[3]), dtype=np.float64)
    tangent_degree = int(getattr(args, "tangent_anchor_degree", 3))
    if tangent_degree < 0:
        raise ValueError("--tangent-anchor-degree must be non-negative")
    tangent_b_coeff = np.zeros((len(rows), tangent_degree + 1, b.shape[2], b.shape[3], b.shape[4]), dtype=np.float64)
    tangent_le_const = np.zeros((len(rows), le.shape[2], le.shape[3]), dtype=np.float64)
    for case_index in range(len(rows)):
        dq = q[case_index] - q_case_mean[case_index].reshape(1, -1)
        signed_amp = dq @ q_case_dir[case_index]
        affine = le_case_mean[case_index].reshape(1, le.shape[2], le.shape[3]) + np.einsum(
            "pak,nk->npa",
            b_case_mean[case_index],
            dq,
        )
        residual = le[case_index] - affine
        xmat = np.stack([np.ones_like(signed_amp), signed_amp, signed_amp * signed_amp], axis=1)
        coef = np.linalg.lstsq(xmat, residual.reshape(residual.shape[0], -1), rcond=None)[0]
        quadratic_coeff[case_index] = coef.reshape(3, le.shape[2], le.shape[3])
        tmat = np.stack([signed_amp**power for power in range(tangent_degree + 1)], axis=1)
        bcoef = np.linalg.lstsq(tmat, b[case_index].reshape(b.shape[1], -1), rcond=None)[0]
        bcoef = bcoef.reshape(tangent_degree + 1, b.shape[2], b.shape[3], b.shape[4])
        tangent_b_coeff[case_index] = bcoef
        slope_coeff = np.einsum("mpak,k->mpa", bcoef, q_case_dir[case_index])
        integ = np.zeros_like(le[case_index])
        for power in range(tangent_degree + 1):
            integ = integ + (signed_amp ** (power + 1)).reshape(-1, 1, 1) * slope_coeff[power].reshape(1, b.shape[2], b.shape[3]) / float(power + 1)
        tangent_le_const[case_index] = np.mean(le[case_index] - integ, axis=0)
    return {
        "case_ids": [int(row["case_id"]) for row in rows],
        "compact_paths": [row["path"] for row in rows],
        "q": q.astype(np.float32),
        "q_norm": q_norm.astype(np.float32),
        "q_mean_global": q_mean_global.astype(np.float64),
        "q_std": q_std.astype(np.float64),
        "q_case_mean": q_case_mean.astype(np.float32),
        "q_case_dir": q_case_dir.astype(np.float32),
        "geom_norm": geom_norm.astype(np.float32),
        "geom_mean": geom_mean.astype(np.float64),
        "geom_std": geom_std.astype(np.float64),
        "trunk_norm": trunk_norm.astype(np.float32),
        "trunk_mean": trunk_mean.astype(np.float64),
        "trunk_std": trunk_std.astype(np.float64),
        "le": le.astype(np.float32),
        "b": b.astype(np.float32),
        "le_case_mean": le_case_mean.astype(np.float32),
        "b_case_mean": b_case_mean.astype(np.float32),
        "quadratic_case_coeff": quadratic_coeff.astype(np.float32),
        "tangent_b_coeff": tangent_b_coeff.astype(np.float32),
        "tangent_le_const": tangent_le_const.astype(np.float32),
        "tangent_anchor_degree": int(tangent_degree),
        "t_eps": t_eps.astype(np.float32),
        "t_q_hat": t_q_hat.astype(np.float32),
        "b_raw": b_raw.astype(np.float32),
    }


def make_tensors(data_np: dict[str, Any], *, device: torch.device) -> dict[str, torch.Tensor]:
    dtype = torch.float32
    return {
        "q": torch.as_tensor(data_np["q"], dtype=dtype, device=device),
        "q_norm": torch.as_tensor(data_np["q_norm"], dtype=dtype, device=device),
        "q_mean_global": torch.as_tensor(data_np["q_mean_global"], dtype=dtype, device=device),
        "q_std": torch.as_tensor(data_np["q_std"], dtype=dtype, device=device),
        "q_case_mean": torch.as_tensor(data_np["q_case_mean"], dtype=dtype, device=device),
        "q_case_dir": torch.as_tensor(data_np["q_case_dir"], dtype=dtype, device=device),
        "geom_norm": torch.as_tensor(data_np["geom_norm"], dtype=dtype, device=device),
        "trunk_norm": torch.as_tensor(data_np["trunk_norm"], dtype=dtype, device=device),
        "le": torch.as_tensor(data_np["le"], dtype=dtype, device=device),
        "b": torch.as_tensor(data_np["b"], dtype=dtype, device=device),
        "le_case_mean": torch.as_tensor(data_np["le_case_mean"], dtype=dtype, device=device),
        "b_case_mean": torch.as_tensor(data_np["b_case_mean"], dtype=dtype, device=device),
        "quadratic_case_coeff": torch.as_tensor(data_np["quadratic_case_coeff"], dtype=dtype, device=device),
        "tangent_b_coeff": torch.as_tensor(data_np["tangent_b_coeff"], dtype=dtype, device=device),
        "tangent_le_const": torch.as_tensor(data_np["tangent_le_const"], dtype=dtype, device=device),
        "t_eps": torch.as_tensor(data_np["t_eps"], dtype=dtype, device=device),
        "t_q_hat": torch.as_tensor(data_np["t_q_hat"], dtype=dtype, device=device),
        "b_raw": torch.as_tensor(data_np["b_raw"], dtype=dtype, device=device),
    }


class V3FormalObjectivePrototype(nn.Module):
    def __init__(
        self,
        *,
        q_dim: int,
        geom_dim: int,
        trunk_dim: int,
        le_case_mean: torch.Tensor,
        b_case_mean: torch.Tensor,
        q_case_mean: torch.Tensor,
        q_case_dir: torch.Tensor,
        quadratic_case_coeff: torch.Tensor,
        tangent_b_coeff: torch.Tensor,
        tangent_le_const: torch.Tensor,
        hidden: int,
        anchor_mode: str,
    ) -> None:
        super().__init__()
        self.register_buffer("le_case_mean", le_case_mean.clone())
        self.b_case_mean = nn.Parameter(b_case_mean.clone())
        self.register_buffer("q_case_mean", q_case_mean.clone())
        self.register_buffer("q_case_dir", q_case_dir.clone())
        self.register_buffer("quadratic_case_coeff", quadratic_case_coeff.clone())
        self.register_buffer("tangent_b_coeff", tangent_b_coeff.clone())
        self.register_buffer("tangent_le_const", tangent_le_const.clone())
        self.anchor_mode = str(anchor_mode)
        self.state_net = nn.Sequential(
            nn.Linear(q_dim + geom_dim, hidden),
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
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 6),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def residual(self, q_norm: torch.Tensor, geom_norm: torch.Tensor, trunk_norm: torch.Tensor) -> torch.Tensor:
        if q_norm.ndim == 1:
            q_norm = q_norm.unsqueeze(0)
        batch = int(q_norm.shape[0])
        points = int(trunk_norm.shape[0])
        geom_expand = geom_norm.reshape(1, -1).expand(batch, -1)
        state = self.state_net(torch.cat([q_norm, geom_expand], dim=-1))
        point = self.point_net(trunk_norm)
        state_expand = state[:, None, :].expand(batch, points, state.shape[-1])
        point_expand = point[None, :, :].expand(batch, points, point.shape[-1])
        return self.head(torch.cat([state_expand, point_expand], dim=-1).reshape(batch * points, -1)).reshape(batch, points, 6)

    def anchor(
        self,
        case_index: int,
        q_raw: torch.Tensor,
        point_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q_raw.ndim == 1:
            q_raw = q_raw.unsqueeze(0)
        if point_idx is None:
            b_prior = self.b_case_mean[case_index]
            le_mean = self.le_case_mean[case_index]
        else:
            b_prior = self.b_case_mean[case_index].index_select(0, point_idx)
            le_mean = self.le_case_mean[case_index].index_select(0, point_idx)
        if self.anchor_mode == "tangent-cubic":
            q_base = q_raw - self.q_case_mean[case_index].reshape(1, -1)
            q_dir = self.q_case_dir[case_index].reshape(1, -1)
            signed_amp = torch.sum(q_base * q_dir, dim=1)
            q_perp = q_base - signed_amp.reshape(-1, 1) * q_dir
            if point_idx is None:
                coeff = self.tangent_b_coeff[case_index]
                const = self.tangent_le_const[case_index]
            else:
                coeff = self.tangent_b_coeff[case_index].index_select(1, point_idx)
                const = self.tangent_le_const[case_index].index_select(0, point_idx)
            powers = torch.stack([signed_amp**power for power in range(int(coeff.shape[0]))], dim=1)
            b_poly = torch.einsum("nm,mpak->npak", powers, coeff)
            integ = torch.zeros(
                (q_raw.shape[0], coeff.shape[1], coeff.shape[2]),
                dtype=q_raw.dtype,
                device=q_raw.device,
            )
            slope_coeff = torch.einsum("mpak,k->mpa", coeff, self.q_case_dir[case_index])
            for power in range(int(coeff.shape[0])):
                integ = integ + (signed_amp ** (power + 1)).reshape(-1, 1, 1) * slope_coeff[power].reshape(1, coeff.shape[1], coeff.shape[2]) / float(power + 1)
            return const.reshape(1, const.shape[0], 6) + integ + torch.einsum("npak,nk->npa", b_poly, q_perp)
        if self.anchor_mode == "linear":
            q_base = q_raw
            offset = 0.0
        elif self.anchor_mode in {"affine", "affine-quadratic"}:
            q_base = q_raw - self.q_case_mean[case_index].reshape(1, -1)
            offset = le_mean.reshape(1, le_mean.shape[0], 6)
        else:
            raise ValueError(f"unknown anchor_mode {self.anchor_mode!r}")
        anchor = offset + torch.einsum("pak,nk->npa", b_prior, q_base)
        if self.anchor_mode == "affine-quadratic":
            if point_idx is None:
                coeff = self.quadratic_case_coeff[case_index]
            else:
                coeff = self.quadratic_case_coeff[case_index].index_select(1, point_idx)
            signed_amp = torch.sum(q_base * self.q_case_dir[case_index].reshape(1, -1), dim=1)
            powers = torch.stack([torch.ones_like(signed_amp), signed_amp, signed_amp * signed_amp], dim=1)
            anchor = anchor + torch.einsum("nm,mpa->npa", powers, coeff)
        return anchor

    def forward_case(
        self,
        case_index: int,
        q_raw: torch.Tensor,
        q_norm: torch.Tensor,
        geom_norm: torch.Tensor,
        trunk_norm: torch.Tensor,
        point_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q_raw.ndim == 1:
            q_raw = q_raw.unsqueeze(0)
        if q_norm.ndim == 1:
            q_norm = q_norm.unsqueeze(0)
        trunk_use = trunk_norm if point_idx is None else trunk_norm.index_select(0, point_idx)
        anchor = self.anchor(case_index, q_raw, point_idx=point_idx)
        residual_q = self.residual(q_norm, geom_norm, trunk_use)
        residual0 = self.residual(torch.zeros_like(q_norm), geom_norm, trunk_use)
        return anchor + residual_q - residual0


def jacobian_case(
    model: V3FormalObjectivePrototype,
    case_index: int,
    q_norm: torch.Tensor,
    geom_norm: torch.Tensor,
    trunk_norm: torch.Tensor,
    q_mean_global: torch.Tensor,
    q_std: torch.Tensor,
    point_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    def one(q_single_norm: torch.Tensor) -> torch.Tensor:
        q_single_raw = q_single_norm * q_std + q_mean_global
        return model.forward_case(case_index, q_single_raw, q_single_norm, geom_norm, trunk_norm, point_idx=point_idx)[0]

    jac_norm = vmap(jacrev(one))(q_norm)
    return jac_norm / q_std.reshape(1, 1, 1, -1)


def raw_project_b(ad_b_local_hat: torch.Tensor, t_eps: torch.Tensor, t_q_hat: torch.Tensor) -> torch.Tensor:
    b_abq = torch.einsum("pab,npbk->npak", t_eps, ad_b_local_hat)
    return torch.einsum("npak,kj->npaj", b_abq, t_q_hat)


def evaluate(
    model: V3FormalObjectivePrototype,
    data: dict[str, torch.Tensor],
    case_ids: list[int],
    *,
    point_chunk: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    q_mean_global = data["q_mean_global"]
    q_std = data["q_std"]
    overall = {key: torch.zeros((), device=q_std.device) for key in [
        "le_diff_sq", "le_den_sq", "ad_diff_sq", "ad_den_sq", "ad_dot", "ad_pred_sq",
        "braw_diff_sq", "braw_den_sq",
    ]}
    case_rows: list[dict[str, Any]] = []
    for c, case_id in enumerate(case_ids):
        q_raw = data["q"][c]
        q_norm = data["q_norm"][c]
        geom_norm = data["geom_norm"][c]
        trunk_norm = data["trunk_norm"][c]
        le = data["le"][c]
        b_local = data["b"][c]
        t_eps = data["t_eps"][c]
        t_q_hat = data["t_q_hat"][c]
        b_raw = data["b_raw"][c]
        with torch.no_grad():
            pred = model.forward_case(c, q_raw, q_norm, geom_norm, trunk_norm)
            le_diff = pred - le
            le_diff_sq = torch.sum(le_diff * le_diff)
            le_den_sq = torch.sum(le * le)
        ad_diff_sq = torch.zeros((), device=q_std.device)
        ad_den_sq = torch.zeros((), device=q_std.device)
        ad_dot = torch.zeros((), device=q_std.device)
        ad_pred_sq = torch.zeros((), device=q_std.device)
        braw_diff_sq = torch.zeros((), device=q_std.device)
        braw_den_sq = torch.zeros((), device=q_std.device)
        for p0 in range(0, int(trunk_norm.shape[0]), int(point_chunk)):
            p1 = min(p0 + int(point_chunk), int(trunk_norm.shape[0]))
            point_idx = torch.arange(p0, p1, device=q_std.device)
            ad_b = jacobian_case(model, c, q_norm, geom_norm, trunk_norm, q_mean_global, q_std, point_idx=point_idx)
            target = b_local[:, p0:p1]
            diff = ad_b - target
            with torch.no_grad():
                ad_diff_sq = ad_diff_sq + torch.sum(diff.detach() * diff.detach())
                ad_den_sq = ad_den_sq + torch.sum(target * target)
                ad_dot = ad_dot + torch.sum(ad_b.detach() * target)
                ad_pred_sq = ad_pred_sq + torch.sum(ad_b.detach() * ad_b.detach())
                b_model_raw = raw_project_b(ad_b.detach(), t_eps[p0:p1], t_q_hat)
                b_raw_chunk = b_raw[:, p0:p1]
                braw_diff = b_model_raw - b_raw_chunk
                braw_diff_sq = braw_diff_sq + torch.sum(braw_diff * braw_diff)
                braw_den_sq = braw_den_sq + torch.sum(b_raw_chunk * b_raw_chunk)
        for key, value in [
            ("le_diff_sq", le_diff_sq),
            ("le_den_sq", le_den_sq),
            ("ad_diff_sq", ad_diff_sq),
            ("ad_den_sq", ad_den_sq),
            ("ad_dot", ad_dot),
            ("ad_pred_sq", ad_pred_sq),
            ("braw_diff_sq", braw_diff_sq),
            ("braw_den_sq", braw_den_sq),
        ]:
            overall[key] = overall[key] + value.detach()
        le_rel = rel_from_squares(le_diff_sq, le_den_sq)
        ad_rel = rel_from_squares(ad_diff_sq, ad_den_sq)
        ad_cos = ad_dot / torch.clamp(torch.sqrt(ad_pred_sq) * torch.sqrt(ad_den_sq), min=1.0e-30)
        braw_rel = rel_from_squares(braw_diff_sq, braw_den_sq)
        case_rows.append({
            "case_index": int(c),
            "case_id": int(case_id),
            "train_LE_local_stack_rel": scalar_float(le_rel),
            "train_AD_B_local_useful_hat_rel": scalar_float(ad_rel),
            "train_AD_B_local_useful_hat_cos": scalar_float(ad_cos),
            "B_model_raw_rel": scalar_float(braw_rel),
            "selection_score": scalar_float(le_rel + ad_rel),
        })
    le_rel = rel_from_squares(overall["le_diff_sq"], overall["le_den_sq"])
    ad_rel = rel_from_squares(overall["ad_diff_sq"], overall["ad_den_sq"])
    ad_cos = overall["ad_dot"] / torch.clamp(torch.sqrt(overall["ad_pred_sq"]) * torch.sqrt(overall["ad_den_sq"]), min=1.0e-30)
    braw_rel = rel_from_squares(overall["braw_diff_sq"], overall["braw_den_sq"])
    return {
        "train_LE_local_stack_rel": scalar_float(le_rel),
        "train_AD_B_local_useful_hat_rel": scalar_float(ad_rel),
        "train_AD_B_local_useful_hat_cos": scalar_float(ad_cos),
        "B_model_raw_rel": scalar_float(braw_rel),
        "selection_score": scalar_float(le_rel + ad_rel),
    }, case_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    preferred = [
        "step", "case_index", "case_id", "loss", "le_loss", "b_loss",
        "train_LE_local_stack_rel", "train_AD_B_local_useful_hat_rel",
        "train_AD_B_local_useful_hat_cos", "B_model_raw_rel", "selection_score",
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
    model = V3FormalObjectivePrototype(
        q_dim=int(data["q_norm"].shape[2]),
        geom_dim=int(data["geom_norm"].shape[1]),
        trunk_dim=int(data["trunk_norm"].shape[2]),
        le_case_mean=data["le_case_mean"],
        b_case_mean=data["b_case_mean"],
        q_case_mean=data["q_case_mean"],
        q_case_dir=data["q_case_dir"],
        quadratic_case_coeff=data["quadratic_case_coeff"],
        tangent_b_coeff=data["tangent_b_coeff"],
        tangent_le_const=data["tangent_le_const"],
        hidden=int(args.hidden),
        anchor_mode=str(args.anchor_mode),
    ).to(device=device, dtype=torch.float32)
    if args.freeze_b_anchor:
        model.b_case_mean.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    le_scale_case = torch.clamp(torch.sqrt(torch.mean(data["le"] * data["le"], dim=(1, 2, 3))), min=1.0e-12)
    b_scale_case = torch.clamp(torch.sqrt(torch.mean(data["b"] * data["b"], dim=(1, 2, 3, 4))), min=1.0e-12)
    case_count = int(data["q"].shape[0])
    frame_count = int(data["q"].shape[1])
    point_count = int(data["trunk_norm"].shape[1])
    case_batch = min(int(args.case_batch), case_count)
    frame_batch = min(int(args.frame_batch), frame_count)
    le_point_batch = min(int(args.le_point_batch), point_count)
    ad_point_batch = min(int(args.ad_point_batch), point_count)
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
        selected_cases = torch.randperm(case_count, device=device)[:case_batch].tolist()
        losses: list[torch.Tensor] = []
        le_terms: list[torch.Tensor] = []
        b_terms: list[torch.Tensor] = []
        for case_index in selected_cases:
            frame_idx = torch.randperm(frame_count, device=device)[:frame_batch]
            le_idx = torch.randperm(point_count, device=device)[:le_point_batch]
            q_raw_batch = data["q"][case_index].index_select(0, frame_idx)
            q_norm_batch = data["q_norm"][case_index].index_select(0, frame_idx)
            pred = model.forward_case(
                int(case_index),
                q_raw_batch,
                q_norm_batch,
                data["geom_norm"][case_index],
                data["trunk_norm"][case_index],
                point_idx=le_idx,
            )
            le_target = data["le"][case_index].index_select(0, frame_idx).index_select(1, le_idx)
            le_loss = torch.mean(((pred - le_target) / le_scale_case[case_index]) ** 2)
            ad_idx = torch.randperm(point_count, device=device)[:ad_point_batch]
            b_target = data["b"][case_index].index_select(0, frame_idx).index_select(1, ad_idx)
            ad_b = jacobian_case(
                model,
                int(case_index),
                q_norm_batch,
                data["geom_norm"][case_index],
                data["trunk_norm"][case_index],
                data["q_mean_global"],
                data["q_std"],
                point_idx=ad_idx,
            )
            b_loss = torch.mean(((ad_b - b_target) / b_scale_case[case_index]) ** 2)
            losses.append(float(args.le_weight) * le_loss + float(args.b_weight) * b_loss)
            le_terms.append(le_loss.detach())
            b_terms.append(b_loss.detach())
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
                "le_loss": scalar_float(torch.stack(le_terms).mean()),
                "b_loss": scalar_float(torch.stack(b_terms).mean()),
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
    summary: dict[str, Any] = {
        "audit_name": "v3_css8_formal_objective_prototype",
        "out_root": str(out_root),
        "device": str(device),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "case_ids": case_ids,
        "compact_paths": list(data_np["compact_paths"]),
        "case_count": case_count,
        "frame_count_per_case": frame_count,
        "point_count": point_count,
        "q_useful_hat_dim": int(data["q"].shape[2]),
        "geometry_global_hat_dim": int(data["geom_norm"].shape[1]),
        "trunk_features_hat_dim": int(data["trunk_norm"].shape[2]),
        "anchor_mode": str(args.anchor_mode),
        "quadratic_anchor_coeff_source": "closed-form train-pool per case" if str(args.anchor_mode) == "affine-quadratic" else None,
        "tangent_anchor_coeff_source": "closed-form train-pool per case" if str(args.anchor_mode) == "tangent-cubic" else None,
        "tangent_anchor_degree": int(data_np.get("tangent_anchor_degree", -1)) if str(args.anchor_mode) == "tangent-cubic" else None,
        "b_anchor_trainable": bool(model.b_case_mean.requires_grad),
        "loss_scale_mode": "per-case",
        "model_inputs": ["q_useful_hat", "geometry_global_hat", "trunk_features_hat"],
        "model_output": "LE_local_stack",
        "ad_target": "dLE_local_stack/dq_useful_hat",
        "formal_training": False,
        "checkpoint_written": False,
        "uses_old_true176_labels_as_v3_labels": False,
        "initial_metrics": initial,
        "best_metrics": best,
        "latest_metrics": latest,
        "latest_case_metrics": latest_cases,
        "case031_latest_metrics": case031_latest,
    }
    summary_path = out_root / "v3_formal_objective_prototype_summary.json"
    history_path = out_root / "v3_formal_objective_prototype_history.csv"
    case_history_path = out_root / "v3_formal_objective_prototype_case_history.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    write_csv(history_path, history)
    write_csv(case_history_path, case_history)
    print(json.dumps({**summary, "summary_path": str(summary_path), "history_path": str(history_path), "case_history_path": str(case_history_path)}, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", default="")
    parser.add_argument("--compact", action="append", default=[])
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--le-weight", type=float, default=1.0)
    parser.add_argument("--b-weight", type=float, default=0.1)
    parser.add_argument("--case-batch", type=int, default=4)
    parser.add_argument("--frame-batch", type=int, default=10)
    parser.add_argument("--le-point-batch", type=int, default=128)
    parser.add_argument("--ad-point-batch", type=int, default=8)
    parser.add_argument("--eval-point-batch", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--anchor-mode", choices=["linear", "affine", "affine-quadratic", "tangent-cubic"], default="affine")
    parser.add_argument("--tangent-anchor-degree", type=int, default=3)
    parser.add_argument("--freeze-b-anchor", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if int(args.steps) < 0:
        raise SystemExit("--steps must be non-negative")
    if int(args.eval_every) < 1:
        raise SystemExit("--eval-every must be positive")
    train(args)


if __name__ == "__main__":
    main()
