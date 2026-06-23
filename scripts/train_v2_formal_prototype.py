#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a script-local formal v2 prototype with train-only normalization.

This is a prototype audit, not a final production trainer.  It implements the
v2f A+C normalization design:

    q_norm = q_useful / q_std_train
    LE_norm = LE_local / LE_std_train
    B_norm[a,k] = B_local[a,k] * q_std_train[k] / LE_std_train[a]

The model uses q_amp as an explicit descriptor, preserves the anchored zero-q
structure, and evaluates AD-B plus raw-B backprojection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.func import jacrev, vmap


B_USEFUL_KEYS = ("B_standard_useful", "B_local_useful")

V2E_BASELINE_LATEST = {
    "train_LE_local_rel": 1.1830968856811523,
    "val_LE_local_rel": 18.03921890258789,
    "train_AD_B_local_rel": 0.3041848838329315,
    "val_AD_B_local_rel": 0.2089325487613678,
    "train_AD_B_local_cos": 0.9526196718215942,
    "val_AD_B_local_cos": 0.9906823635101318,
    "train_B_model_raw_projected_rel": 0.30487608909606934,
    "val_B_model_raw_projected_rel": 0.2097579836845398,
    "zero_q_LE_local_rms": 0.0,
}


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


def scalar_int(value: Any, default: int | None = None) -> int | None:
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return default
        return int(arr.reshape(-1)[0])
    except Exception:
        return default


def parse_case_id(path: Path, z: np.lib.npyio.NpzFile | None = None) -> int:
    if z is not None:
        for key in ("v2b_pilot_case_id", "v2a_pilot_case_id", "case_id"):
            if key in z.files:
                val = scalar_int(z[key])
                if val is not None:
                    return int(val)
    match = re.search(r"case[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot parse case id from {path}")
    return int(match.group(1))


def find_b_useful_key(files: list[str]) -> str:
    for key in B_USEFUL_KEYS:
        if key in files:
            return key
    raise KeyError(f"missing one of {B_USEFUL_KEYS}")


def read_compact_list(path: Path) -> list[Path]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            out.append(Path(item).resolve())
    return out


def parse_cases(text: str) -> list[int]:
    vals = []
    for part in str(text).replace(";", ",").split(","):
        item = part.strip()
        if item:
            vals.append(int(item))
    return vals


def rel_norm_torch(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(num.reshape(-1)) / torch.clamp(torch.linalg.vector_norm(den.reshape(-1)), min=1.0e-30)


def cosine_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    denom = torch.clamp(torch.linalg.vector_norm(aa) * torch.linalg.vector_norm(bb), min=1.0e-30)
    return torch.dot(aa, bb) / denom


def scalar_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def broadcast_t_eps_np(t_eps: np.ndarray, n_frames: int) -> np.ndarray:
    vals = np.asarray(t_eps, dtype=np.float32)
    if vals.shape == (128, 6, 6):
        return np.broadcast_to(vals.reshape(1, 128, 6, 6), (int(n_frames), 128, 6, 6)).copy()
    if vals.shape == (int(n_frames), 128, 6, 6):
        return vals.copy()
    raise ValueError(f"T_eps_to_abq must be [128,6,6] or [N,128,6,6], got {vals.shape}")


@dataclass
class CaseData:
    case_id: int
    compact: str
    q: np.ndarray
    le: np.ndarray
    b: np.ndarray
    t_q: np.ndarray
    t_eps: np.ndarray
    b_raw: np.ndarray
    ip_xi: np.ndarray
    frame_case_ids: np.ndarray
    frame_indices: np.ndarray


def load_case(path: Path) -> CaseData:
    with np.load(str(path), allow_pickle=True) as z:
        required = {
            "q_useful",
            "LE128_local",
            "T_q_raw_to_useful",
            "T_eps_to_abq",
            "B_LE128_forward",
            "ip_xi",
        }
        missing = sorted(required.difference(z.files))
        if missing:
            raise KeyError(f"{path}: missing required v2g fields {missing}")
        case_id = parse_case_id(path, z)
        b_key = find_b_useful_key(list(z.files))
        q = np.asarray(z["q_useful"], dtype=np.float32)
        le = np.asarray(z["LE128_local"], dtype=np.float32)
        b = np.asarray(z[b_key], dtype=np.float32)
        t_q = np.asarray(z["T_q_raw_to_useful"], dtype=np.float32)
        t_eps = np.asarray(z["T_eps_to_abq"], dtype=np.float32)
        b_raw = np.asarray(z["B_LE128_forward"], dtype=np.float32)
        ip_xi = np.asarray(z["ip_xi"], dtype=np.float32)
        if ip_xi.ndim == 3:
            ip_xi = ip_xi[0]
        b_q_coord = scalar_text(z["B_label_q_coordinate"]) if "B_label_q_coordinate" in z.files else "unknown"
        b_out_coord = scalar_text(z["B_label_output_coordinate"]) if "B_label_output_coordinate" in z.files else "unknown"
        old_true176 = bool(np.asarray(z["uses_old_true176_labels_as_v2_labels"]).reshape(-1)[0]) if "uses_old_true176_labels_as_v2_labels" in z.files else False

    if old_true176:
        raise ValueError(f"{path}: uses_old_true176_labels_as_v2_labels is true")
    if b_q_coord != "q_useful":
        raise ValueError(f"{path}: B_label_q_coordinate must be q_useful, got {b_q_coord!r}")
    if b_out_coord != "local_jacobian_frame":
        raise ValueError(f"{path}: B_label_output_coordinate must be local_jacobian_frame, got {b_out_coord!r}")
    if q.shape[1:] != (42,):
        raise ValueError(f"{path}: q_useful must be [N,42], got {q.shape}")
    if le.shape != (q.shape[0], 128, 6):
        raise ValueError(f"{path}: LE128_local must be [N,128,6], got {le.shape}")
    if b.shape != (q.shape[0], 128, 6, 42):
        raise ValueError(f"{path}: {b_key} must be [N,128,6,42], got {b.shape}")
    if t_q.shape != (42, 48):
        raise ValueError(f"{path}: T_q_raw_to_useful must be [42,48], got {t_q.shape}")
    if b_raw.shape != (q.shape[0], 128, 6, 48):
        raise ValueError(f"{path}: B_LE128_forward must be [N,128,6,48], got {b_raw.shape}")
    if ip_xi.shape != (128, 3):
        raise ValueError(f"{path}: ip_xi must be [128,3], got {ip_xi.shape}")

    return CaseData(
        case_id=int(case_id),
        compact=str(path),
        q=q,
        le=le,
        b=b,
        t_q=t_q,
        t_eps=broadcast_t_eps_np(t_eps, q.shape[0]),
        b_raw=b_raw,
        ip_xi=ip_xi,
        frame_case_ids=np.full((q.shape[0],), int(case_id), dtype=np.int64),
        frame_indices=np.arange(q.shape[0], dtype=np.int64),
    )


def concat_cases(cases: list[CaseData]) -> dict[str, np.ndarray]:
    if not cases:
        raise ValueError("empty case selection")
    ip_xi = cases[0].ip_xi
    for item in cases[1:]:
        if not np.allclose(item.ip_xi, ip_xi, rtol=0.0, atol=1.0e-7):
            raise ValueError("ip_xi differs across cases; this prototype assumes one shared geometry")
    return {
        "q": np.concatenate([c.q for c in cases], axis=0),
        "le": np.concatenate([c.le for c in cases], axis=0),
        "b": np.concatenate([c.b for c in cases], axis=0),
        "t_q": np.concatenate([np.broadcast_to(c.t_q.reshape(1, 42, 48), (c.q.shape[0], 42, 48)) for c in cases], axis=0),
        "t_eps": np.concatenate([c.t_eps for c in cases], axis=0),
        "b_raw": np.concatenate([c.b_raw for c in cases], axis=0),
        "case_ids": np.concatenate([c.frame_case_ids for c in cases], axis=0),
        "frame_indices": np.concatenate([c.frame_indices for c in cases], axis=0),
        "ip_xi": ip_xi,
    }


def compute_normalization(train_np: dict[str, np.ndarray]) -> dict[str, np.ndarray | float | int]:
    q = np.asarray(train_np["q"], dtype=np.float64)
    le = np.asarray(train_np["le"], dtype=np.float64)
    q_std = np.std(q, axis=0)
    q_mean = np.mean(q, axis=0)
    positive_q = q_std[q_std > 0.0]
    q_median = float(np.median(positive_q)) if positive_q.size else 1.0
    q_floor_value = max(q_median * 1.0e-3, 1.0e-12)
    q_std_floor = np.maximum(q_std, q_floor_value)

    le_std = np.std(le, axis=(0, 1))
    le_mean = np.mean(le, axis=(0, 1))
    positive_le = le_std[le_std > 0.0]
    le_median = float(np.median(positive_le)) if positive_le.size else 1.0
    le_floor_value = max(le_median * 1.0e-3, 1.0e-12)
    le_std_floor = np.maximum(le_std, le_floor_value)

    q_amp = np.linalg.norm(q, axis=1)
    q_amp_rms = float(np.sqrt(np.mean(q_amp * q_amp)))
    q_amp_ref = max(q_amp_rms, 1.0e-12)
    return {
        "q_mean": q_mean.astype(np.float32),
        "q_std": q_std.astype(np.float32),
        "q_std_floor": q_std_floor.astype(np.float32),
        "q_std_floor_value": float(q_floor_value),
        "q_std_floored_count": int(np.sum(q_std < q_floor_value)),
        "q_std_min": float(np.min(q_std)),
        "q_std_max": float(np.max(q_std)),
        "q_std_floor_min": float(np.min(q_std_floor)),
        "q_std_floor_max": float(np.max(q_std_floor)),
        "q_std_ratio_before_floor": float(np.max(q_std) / max(np.min(q_std), 1.0e-30)),
        "q_std_ratio_after_floor": float(np.max(q_std_floor) / max(np.min(q_std_floor), 1.0e-30)),
        "le_mean": le_mean.astype(np.float32),
        "le_std": le_std.astype(np.float32),
        "le_std_floor": le_std_floor.astype(np.float32),
        "le_std_floor_value": float(le_floor_value),
        "le_std_floored_count": int(np.sum(le_std < le_floor_value)),
        "le_std_min": float(np.min(le_std)),
        "le_std_max": float(np.max(le_std)),
        "le_std_floor_min": float(np.min(le_std_floor)),
        "le_std_floor_max": float(np.max(le_std_floor)),
        "le_std_ratio_before_floor": float(np.max(le_std) / max(np.min(le_std), 1.0e-30)),
        "le_std_ratio_after_floor": float(np.max(le_std_floor) / max(np.min(le_std_floor), 1.0e-30)),
        "q_amp_mean": float(np.mean(q_amp)),
        "q_amp_std": float(np.std(q_amp)),
        "q_amp_rms": q_amp_rms,
        "q_amp_ref": q_amp_ref,
        "B_norm_scale_min": float(np.min(q_std_floor.reshape(1, 42) / le_std_floor.reshape(6, 1))),
        "B_norm_scale_max": float(np.max(q_std_floor.reshape(1, 42) / le_std_floor.reshape(6, 1))),
        "B_norm_scale_rms": float(np.sqrt(np.mean((q_std_floor.reshape(1, 42) / le_std_floor.reshape(6, 1)) ** 2))),
    }


def normalize_dataset(data: dict[str, np.ndarray], norm: dict[str, Any]) -> dict[str, np.ndarray]:
    q_std = np.asarray(norm["q_std_floor"], dtype=np.float32)
    le_std = np.asarray(norm["le_std_floor"], dtype=np.float32)
    q = np.asarray(data["q"], dtype=np.float32)
    le = np.asarray(data["le"], dtype=np.float32)
    b = np.asarray(data["b"], dtype=np.float32)
    q_norm = q / q_std.reshape(1, 42)
    le_norm = le / le_std.reshape(1, 1, 6)
    b_norm = b * q_std.reshape(1, 1, 1, 42) / le_std.reshape(1, 1, 6, 1)
    out = dict(data)
    out.update({"q_norm": q_norm, "le_norm": le_norm, "b_norm": b_norm})
    return out


def make_tensors(data: dict[str, np.ndarray], *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "q": torch.as_tensor(data["q"], dtype=dtype, device=device),
        "q_norm": torch.as_tensor(data["q_norm"], dtype=dtype, device=device),
        "le": torch.as_tensor(data["le"], dtype=dtype, device=device),
        "le_norm": torch.as_tensor(data["le_norm"], dtype=dtype, device=device),
        "b": torch.as_tensor(data["b"], dtype=dtype, device=device),
        "b_norm": torch.as_tensor(data["b_norm"], dtype=dtype, device=device),
        "t_q": torch.as_tensor(data["t_q"], dtype=dtype, device=device),
        "t_eps": torch.as_tensor(data["t_eps"], dtype=dtype, device=device),
        "b_raw": torch.as_tensor(data["b_raw"], dtype=dtype, device=device),
        "case_ids": torch.as_tensor(data["case_ids"], dtype=torch.long, device=device),
    }


class FormalV2Prototype(nn.Module):
    """Anchored formal v2 prototype in normalized coordinates."""

    def __init__(
        self,
        ip_xi: torch.Tensor,
        b_prior_norm_mean: torch.Tensor,
        q_std: torch.Tensor,
        *,
        q_amp_ref: float,
        hidden: int,
        use_q_amp: bool,
        use_q_dir: bool,
        gate_c: float,
    ) -> None:
        super().__init__()
        self.register_buffer("ip_xi", ip_xi)
        self.register_buffer("b_prior_base", b_prior_norm_mean.clone())
        self.register_buffer("q_std", q_std.clone())
        self.q_amp_ref = float(q_amp_ref)
        self.use_q_amp = bool(use_q_amp)
        self.use_q_dir = bool(use_q_dir)
        self.gate_c = float(gate_c)

        branch_dim = 42 + (1 if self.use_q_amp else 0) + (42 if self.use_q_dir else 0)
        self.point_encoder = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.branch_encoder = nn.Sequential(nn.Linear(branch_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.b_delta_head = nn.Linear(hidden, 6 * 42)
        self.residual = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 6),
        )
        nn.init.zeros_(self.b_delta_head.weight)
        nn.init.zeros_(self.b_delta_head.bias)

    def branch_features(self, q_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_useful = q_norm * self.q_std.reshape(1, 42)
        q_amp_raw = torch.sqrt(torch.sum(q_useful * q_useful, dim=-1, keepdim=True) + 1.0e-24)
        q_amp_scaled = q_amp_raw / max(self.q_amp_ref, 1.0e-12)
        pieces = [q_norm]
        if self.use_q_amp:
            pieces.append(q_amp_scaled)
        if self.use_q_dir:
            pieces.append(q_useful / torch.clamp(q_amp_raw, min=1.0e-12))
        return torch.cat(pieces, dim=-1), q_amp_scaled

    def b_prior(self, point_idx: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if point_idx is None:
            xi = self.ip_xi
            base = self.b_prior_base
        else:
            xi = self.ip_xi.index_select(0, point_idx)
            base = self.b_prior_base.index_select(0, point_idx)
        point_latent = self.point_encoder(xi)
        delta = self.b_delta_head(point_latent).reshape(xi.shape[0], 6, 42)
        return base + delta, point_latent

    def forward(self, q_norm: torch.Tensor, point_idx: torch.Tensor | None = None) -> torch.Tensor:
        if q_norm.ndim == 1:
            q_norm = q_norm.unsqueeze(0)
        b_prior, point_latent = self.b_prior(point_idx)
        branch_in, q_amp = self.branch_features(q_norm)
        branch_latent = self.branch_encoder(branch_in)
        linear = torch.einsum("pak,bk->bpa", b_prior, q_norm)

        batch = q_norm.shape[0]
        point_count = point_latent.shape[0]
        point_expand = point_latent[None, :, :].expand(batch, point_count, point_latent.shape[-1])
        branch_expand = branch_latent[:, None, :].expand(batch, point_count, branch_latent.shape[-1])
        amp_expand = q_amp[:, None, :].expand(batch, point_count, 1)
        residual_in = torch.cat([point_expand, branch_expand, amp_expand], dim=-1)
        residual_q = self.residual(residual_in.reshape(batch * point_count, -1)).reshape(batch, point_count, 6)

        q_zero = torch.zeros_like(q_norm)
        branch0, amp0 = self.branch_features(q_zero)
        branch_latent0 = self.branch_encoder(branch0)
        branch0_expand = branch_latent0[:, None, :].expand(batch, point_count, branch_latent0.shape[-1])
        amp0_expand = amp0[:, None, :].expand(batch, point_count, 1)
        residual0_in = torch.cat([point_expand, branch0_expand, amp0_expand], dim=-1)
        residual0 = self.residual(residual0_in.reshape(batch * point_count, -1)).reshape(batch, point_count, 6)
        gate = q_amp[:, None, :] / (q_amp[:, None, :] + float(self.gate_c))
        return linear + gate * (residual_q - residual0)


def jacobian_norm(model: FormalV2Prototype, q_norm: torch.Tensor, point_idx: torch.Tensor | None = None) -> torch.Tensor:
    def one(q_single: torch.Tensor) -> torch.Tensor:
        return model(q_single, point_idx=point_idx)[0]

    return vmap(jacrev(one))(q_norm)


def denormalize_b(ad_b_norm: torch.Tensor, le_std: torch.Tensor, q_std: torch.Tensor) -> torch.Tensor:
    return ad_b_norm * le_std.reshape(1, 1, 6, 1) / q_std.reshape(1, 1, 1, 42)


def raw_project_b(b_local: torch.Tensor, t_eps: torch.Tensor, t_q: torch.Tensor) -> torch.Tensor:
    b_abq = torch.einsum("npab,npbk->npak", t_eps, b_local)
    return torch.einsum("npak,nkj->npaj", b_abq, t_q)


def eval_split(
    model: FormalV2Prototype,
    data: dict[str, torch.Tensor],
    *,
    prefix: str,
    q_std: torch.Tensor,
    le_std: torch.Tensor,
    frame_chunk: int,
    point_chunk: int,
) -> dict[str, float]:
    q_norm = data["q_norm"]
    le_norm = data["le_norm"]
    le = data["le"]
    b_norm = data["b_norm"]
    b_local = data["b"]
    t_eps = data["t_eps"]
    t_q = data["t_q"]
    b_raw = data["b_raw"]
    model.eval()

    le_norm_diff_sq = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    le_norm_den_sq = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    le_diff_sq = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    le_den_sq = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    le_count = 0
    with torch.no_grad():
        for f0 in range(0, int(q_norm.shape[0]), int(frame_chunk)):
            f1 = min(f0 + int(frame_chunk), int(q_norm.shape[0]))
            pred_norm = model(q_norm[f0:f1])
            pred_local = pred_norm * le_std.reshape(1, 1, 6)
            target_norm = le_norm[f0:f1]
            target_local = le[f0:f1]
            diff_norm = pred_norm - target_norm
            diff_local = pred_local - target_local
            le_norm_diff_sq = le_norm_diff_sq + torch.sum(diff_norm * diff_norm)
            le_norm_den_sq = le_norm_den_sq + torch.sum(target_norm * target_norm)
            le_diff_sq = le_diff_sq + torch.sum(diff_local * diff_local)
            le_den_sq = le_den_sq + torch.sum(target_local * target_local)
            le_count += int(diff_local.numel())
        zero_norm = model(torch.zeros(1, q_norm.shape[-1], device=q_norm.device, dtype=q_norm.dtype))
        zero_local = zero_norm * le_std.reshape(1, 1, 6)

    ad_norm_diff_sq = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    ad_norm_den_sq = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    ad_local_diff_sq = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    ad_local_den_sq = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    ad_local_dot = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    ad_local_pred_sq = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    bproj_diff_sq = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    bproj_den_sq = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    braw_diff_sq = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    braw_den_sq = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)

    for f0 in range(0, int(q_norm.shape[0]), int(frame_chunk)):
        f1 = min(f0 + int(frame_chunk), int(q_norm.shape[0]))
        q_chunk = q_norm[f0:f1]
        t_q_chunk = t_q[f0:f1]
        p_useful = torch.einsum("fki,fkj->fij", t_q_chunk, t_q_chunk)
        for p0 in range(0, int(le.shape[1]), int(point_chunk)):
            p1 = min(p0 + int(point_chunk), int(le.shape[1]))
            pidx = torch.arange(p0, p1, device=q_norm.device, dtype=torch.long)
            ad_norm = jacobian_norm(model, q_chunk, point_idx=pidx)
            target_norm = b_norm[f0:f1, p0:p1]
            target_local = b_local[f0:f1, p0:p1]
            ad_local = denormalize_b(ad_norm, le_std, q_std)
            diff_norm = ad_norm - target_norm
            diff_local = ad_local - target_local
            with torch.no_grad():
                ad_norm_diff_sq = ad_norm_diff_sq + torch.sum(diff_norm.detach() * diff_norm.detach())
                ad_norm_den_sq = ad_norm_den_sq + torch.sum(target_norm * target_norm)
                ad_local_diff_sq = ad_local_diff_sq + torch.sum(diff_local.detach() * diff_local.detach())
                ad_local_den_sq = ad_local_den_sq + torch.sum(target_local * target_local)
                ad_local_dot = ad_local_dot + torch.sum(ad_local.detach() * target_local)
                ad_local_pred_sq = ad_local_pred_sq + torch.sum(ad_local.detach() * ad_local.detach())
                t_eps_chunk = t_eps[f0:f1, p0:p1]
                b_raw_chunk = b_raw[f0:f1, p0:p1]
                b_model_raw = raw_project_b(ad_local.detach(), t_eps_chunk, t_q_chunk)
                b_raw_projected = torch.einsum("fpaj,fjk->fpak", b_raw_chunk, p_useful)
                bproj_diff = b_model_raw - b_raw_projected
                braw_diff = b_model_raw - b_raw_chunk
                bproj_diff_sq = bproj_diff_sq + torch.sum(bproj_diff * bproj_diff)
                bproj_den_sq = bproj_den_sq + torch.sum(b_raw_projected * b_raw_projected)
                braw_diff_sq = braw_diff_sq + torch.sum(braw_diff * braw_diff)
                braw_den_sq = braw_den_sq + torch.sum(b_raw_chunk * b_raw_chunk)

    with torch.no_grad():
        le_norm_rel = torch.sqrt(le_norm_diff_sq) / torch.clamp(torch.sqrt(le_norm_den_sq), min=1.0e-30)
        le_local_rel = torch.sqrt(le_diff_sq) / torch.clamp(torch.sqrt(le_den_sq), min=1.0e-30)
        le_rmse = torch.sqrt(le_diff_sq / max(le_count, 1))
        le_target_rms = torch.sqrt(le_den_sq / max(le_count, 1))
        ad_norm_rel = torch.sqrt(ad_norm_diff_sq) / torch.clamp(torch.sqrt(ad_norm_den_sq), min=1.0e-30)
        ad_local_rel = torch.sqrt(ad_local_diff_sq) / torch.clamp(torch.sqrt(ad_local_den_sq), min=1.0e-30)
        ad_local_cos = ad_local_dot / torch.clamp(torch.sqrt(ad_local_pred_sq) * torch.sqrt(ad_local_den_sq), min=1.0e-30)
        b_projected_rel = torch.sqrt(bproj_diff_sq) / torch.clamp(torch.sqrt(bproj_den_sq), min=1.0e-30)
        b_raw_rel = torch.sqrt(braw_diff_sq) / torch.clamp(torch.sqrt(braw_den_sq), min=1.0e-30)
        zero_norm_rms = torch.sqrt(torch.mean(zero_norm * zero_norm))
        zero_local_rms = torch.sqrt(torch.mean(zero_local * zero_local))

    return {
        f"{prefix}_LE_norm_rel": scalar_float(le_norm_rel),
        f"{prefix}_AD_B_norm_rel": scalar_float(ad_norm_rel),
        f"{prefix}_LE_local_rel": scalar_float(le_local_rel),
        f"{prefix}_LE_local_rmse": scalar_float(le_rmse),
        f"{prefix}_LE_target_rms": scalar_float(le_target_rms),
        f"{prefix}_AD_B_local_rel": scalar_float(ad_local_rel),
        f"{prefix}_AD_B_local_cos": scalar_float(ad_local_cos),
        f"{prefix}_B_model_raw_projected_rel": scalar_float(b_projected_rel),
        f"{prefix}_B_model_raw_rel": scalar_float(b_raw_rel),
        f"{prefix}_zero_q_LE_norm_rms": scalar_float(zero_norm_rms),
        f"{prefix}_zero_q_LE_local_rms": scalar_float(zero_local_rms),
    }


def checkpoint_score(metrics: dict[str, float]) -> float:
    return float(
        metrics["val_LE_local_rel"]
        + metrics["val_AD_B_local_rel"]
        + 0.5 * metrics["val_B_model_raw_projected_rel"]
        + 10.0 * metrics["val_zero_q_LE_norm_rms"]
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "step",
        "split",
        "case_id",
        "frame_count",
        "loss",
        "le_loss",
        "b_loss",
        "zero_loss",
        "score",
        "LE_norm_rel",
        "LE_local_rel",
        "LE_local_rmse",
        "LE_target_rms",
        "AD_B_norm_rel",
        "AD_B_local_rel",
        "AD_B_local_cos",
        "B_raw_projected_rel",
        "B_raw_rel",
        "q_norm_mean",
        "q_amp_mean",
    ]
    ordered = [key for key in preferred if key in fields]
    ordered.extend(key for key in fields if key not in ordered)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def per_case_metrics(
    model: FormalV2Prototype,
    data: dict[str, torch.Tensor],
    *,
    split: str,
    q_std: torch.Tensor,
    le_std: torch.Tensor,
    q_amp_ref: float,
    frame_chunk: int,
    point_chunk: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_ids = sorted(set(int(v) for v in data["case_ids"].detach().cpu().numpy().tolist()))
    for case_id in case_ids:
        idx = torch.nonzero(data["case_ids"] == int(case_id), as_tuple=False).reshape(-1)
        subset = {key: value.index_select(0, idx) for key, value in data.items() if key != "case_ids"}
        subset["case_ids"] = data["case_ids"].index_select(0, idx)
        metrics = eval_split(model, subset, prefix=split, q_std=q_std, le_std=le_std, frame_chunk=frame_chunk, point_chunk=point_chunk)
        q_useful = subset["q"]
        q_amp = torch.linalg.vector_norm(q_useful, dim=1)
        rows.append(
            {
                "split": split,
                "case_id": int(case_id),
                "frame_count": int(idx.numel()),
                "LE_norm_rel": metrics[f"{split}_LE_norm_rel"],
                "LE_local_rel": metrics[f"{split}_LE_local_rel"],
                "LE_local_rmse": metrics[f"{split}_LE_local_rmse"],
                "LE_target_rms": metrics[f"{split}_LE_target_rms"],
                "AD_B_norm_rel": metrics[f"{split}_AD_B_norm_rel"],
                "AD_B_local_rel": metrics[f"{split}_AD_B_local_rel"],
                "AD_B_local_cos": metrics[f"{split}_AD_B_local_cos"],
                "B_raw_projected_rel": metrics[f"{split}_B_model_raw_projected_rel"],
                "B_raw_rel": metrics[f"{split}_B_model_raw_rel"],
                "zero_q_LE_norm_rms": metrics[f"{split}_zero_q_LE_norm_rms"],
                "zero_q_LE_local_rms": metrics[f"{split}_zero_q_LE_local_rms"],
                "q_norm_mean": scalar_float(torch.mean(torch.linalg.vector_norm(subset["q_norm"], dim=1))),
                "q_amp_mean": scalar_float(torch.mean(q_amp)),
                "q_amp_scaled_mean": scalar_float(torch.mean(q_amp / max(q_amp_ref, 1.0e-12))),
            }
        )
    return rows


def update_best(bests: dict[str, dict[str, Any]], row: dict[str, Any]) -> None:
    candidates = {
        "best_combined": row["score"],
        "best_LE": row["val_LE_local_rel"],
        "best_B": row["val_AD_B_local_rel"],
        "best_raw_projected": row["val_B_model_raw_projected_rel"],
    }
    for key, score in candidates.items():
        if key not in bests or float(score) < float(bests[key]["_selection_value"]):
            bests[key] = {**row, "_selection_value": float(score)}


def strip_selection(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out.pop("_selection_value", None)
    return out


def train(args: argparse.Namespace) -> dict[str, Any]:
    compact_list = Path(args.compact_list).resolve()
    cases = [load_case(path) for path in read_compact_list(compact_list)]
    if not cases:
        raise SystemExit("compact list is empty")
    case_ids = sorted(c.case_id for c in cases)
    val_cases = parse_cases(args.val_cases)
    train_cases = [case for case in case_ids if case not in set(val_cases)]
    if set(train_cases).intersection(val_cases):
        raise SystemExit("train/val split overlaps")
    unknown_val = sorted(set(val_cases).difference(case_ids))
    if unknown_val:
        raise SystemExit(f"val cases not found: {unknown_val}")

    train_np_raw = concat_cases([c for c in cases if c.case_id in set(train_cases)])
    val_np_raw = concat_cases([c for c in cases if c.case_id in set(val_cases)])
    norm = compute_normalization(train_np_raw)
    train_np = normalize_dataset(train_np_raw, norm)
    val_np = normalize_dataset(val_np_raw, norm)

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    norm_summary_path = out_root / "normalization_summary.json"
    norm_summary = {
        "train_only": True,
        "train_cases": train_cases,
        "val_cases_excluded": val_cases,
        "q_std_min": norm["q_std_min"],
        "q_std_max": norm["q_std_max"],
        "q_std_ratio_before_floor": norm["q_std_ratio_before_floor"],
        "q_std_ratio_after_floor": norm["q_std_ratio_after_floor"],
        "q_std_floor_value": norm["q_std_floor_value"],
        "q_std_floored_count": norm["q_std_floored_count"],
        "LE_std": np.asarray(norm["le_std"]).tolist(),
        "LE_std_floor": np.asarray(norm["le_std_floor"]).tolist(),
        "LE_std_ratio_before_floor": norm["le_std_ratio_before_floor"],
        "LE_std_ratio_after_floor": norm["le_std_ratio_after_floor"],
        "LE_std_floor_value": norm["le_std_floor_value"],
        "LE_std_floored_count": norm["le_std_floored_count"],
        "q_amp_mean": norm["q_amp_mean"],
        "q_amp_std": norm["q_amp_std"],
        "q_amp_rms": norm["q_amp_rms"],
        "q_amp_ref": norm["q_amp_ref"],
        "B_norm_scale_min": norm["B_norm_scale_min"],
        "B_norm_scale_max": norm["B_norm_scale_max"],
        "B_norm_scale_rms": norm["B_norm_scale_rms"],
    }
    norm_summary_path.write_text(json.dumps(norm_summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")

    set_seed(int(args.seed))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32
    train_data = make_tensors(train_np, device=device, dtype=dtype)
    val_data = make_tensors(val_np, device=device, dtype=dtype)
    ip_xi = torch.as_tensor(train_np["ip_xi"], dtype=dtype, device=device)
    q_std = torch.as_tensor(norm["q_std_floor"], dtype=dtype, device=device)
    le_std = torch.as_tensor(norm["le_std_floor"], dtype=dtype, device=device)
    b_prior_norm_mean = train_data["b_norm"].mean(dim=0)
    model = FormalV2Prototype(
        ip_xi,
        b_prior_norm_mean,
        q_std,
        q_amp_ref=float(norm["q_amp_ref"]),
        hidden=int(args.hidden),
        use_q_amp=bool(args.use_q_amp),
        use_q_dir=bool(args.use_q_dir),
        gate_c=float(args.gate_c),
    ).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    frame_count = int(train_data["q_norm"].shape[0])
    point_count = int(train_data["le_norm"].shape[1])
    frame_batch = min(int(args.frame_batch), frame_count)
    le_point_batch = min(int(args.le_point_batch), point_count)
    ad_point_batch = min(int(args.ad_point_batch), point_count)
    eval_steps = set(range(0, int(args.steps) + 1, int(args.eval_every)))
    eval_steps.add(int(args.steps))

    history: list[dict[str, Any]] = []
    bests: dict[str, dict[str, Any]] = {}

    def eval_all(step: int, extra: dict[str, float] | None = None) -> dict[str, Any]:
        row = {
            "step": int(step),
            **(extra or {}),
            **eval_split(
                model,
                train_data,
                prefix="train",
                q_std=q_std,
                le_std=le_std,
                frame_chunk=int(args.eval_frame_batch),
                point_chunk=int(args.eval_point_batch),
            ),
            **eval_split(
                model,
                val_data,
                prefix="val",
                q_std=q_std,
                le_std=le_std,
                frame_chunk=int(args.eval_frame_batch),
                point_chunk=int(args.eval_point_batch),
            ),
        }
        row["score"] = checkpoint_score(row)
        return row

    initial = eval_all(0)
    history.append(initial)
    update_best(bests, initial)

    for step in range(1, int(args.steps) + 1):
        model.train()
        frame_idx = torch.randperm(frame_count, device=device)[:frame_batch]
        q_batch = train_data["q_norm"].index_select(0, frame_idx)
        if le_point_batch == point_count:
            le_idx = None
            le_target = train_data["le_norm"].index_select(0, frame_idx)
        else:
            le_idx = torch.randperm(point_count, device=device)[:le_point_batch]
            le_target = train_data["le_norm"].index_select(0, frame_idx).index_select(1, le_idx)
        pred = model(q_batch, point_idx=le_idx)
        le_loss = torch.mean((pred - le_target) ** 2)

        if ad_point_batch == point_count:
            ad_idx = None
            b_target = train_data["b_norm"].index_select(0, frame_idx)
        else:
            ad_idx = torch.randperm(point_count, device=device)[:ad_point_batch]
            b_target = train_data["b_norm"].index_select(0, frame_idx).index_select(1, ad_idx)
        ad_b = jacobian_norm(model, q_batch, point_idx=ad_idx)
        b_loss = torch.mean((ad_b - b_target) ** 2)
        zero_pred = model(torch.zeros(1, q_batch.shape[-1], device=device, dtype=dtype))
        zero_loss = torch.mean(zero_pred * zero_pred)
        loss = float(args.le_weight) * le_loss + float(args.b_weight) * b_loss + float(args.zero_weight) * zero_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        opt.step()

        if step in eval_steps:
            row = eval_all(
                step,
                {
                    "loss": scalar_float(loss.detach()),
                    "le_loss": scalar_float(le_loss.detach()),
                    "b_loss": scalar_float(b_loss.detach()),
                    "zero_loss": scalar_float(zero_loss.detach()),
                },
            )
            history.append(row)
            update_best(bests, row)
            print(json.dumps(row, sort_keys=True), flush=True)

    latest = history[-1]
    per_case_rows = per_case_metrics(
        model,
        train_data,
        split="train",
        q_std=q_std,
        le_std=le_std,
        q_amp_ref=float(norm["q_amp_ref"]),
        frame_chunk=int(args.eval_frame_batch),
        point_chunk=int(args.eval_point_batch),
    ) + per_case_metrics(
        model,
        val_data,
        split="val",
        q_std=q_std,
        le_std=le_std,
        q_amp_ref=float(norm["q_amp_ref"]),
        frame_chunk=int(args.eval_frame_batch),
        point_chunk=int(args.eval_point_batch),
    )

    paths = {
        "training_summary": out_root / "training_summary.json",
        "metrics_history": out_root / "metrics_history.csv",
        "per_case_attribution": out_root / "per_case_attribution.csv",
        "best_combined": out_root / "best_combined_metrics.json",
        "best_LE": out_root / "best_LE_metrics.json",
        "best_B": out_root / "best_B_metrics.json",
        "best_raw_projected": out_root / "best_raw_projected_metrics.json",
        "latest": out_root / "latest_metrics.json",
    }
    for key in ("best_combined", "best_LE", "best_B", "best_raw_projected"):
        paths[key].write_text(json.dumps(strip_selection(bests[key]), indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    paths["latest"].write_text(json.dumps(latest, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    write_csv(paths["metrics_history"], history)
    write_csv(paths["per_case_attribution"], per_case_rows)

    summary = {
        "audit_name": "v2g_formal_prototype",
        "compact_list": str(compact_list),
        "out_root": str(out_root),
        "compact_count": len(cases),
        "case_ids": case_ids,
        "train_cases": train_cases,
        "val_cases": val_cases,
        "validation_is_overlapping": False,
        "normalization_train_only": True,
        "normalization_summary": str(norm_summary_path),
        "use_q_amp": bool(args.use_q_amp),
        "use_q_dir": bool(args.use_q_dir),
        "use_geometry_features": False,
        "gate_c": float(args.gate_c),
        "steps": int(args.steps),
        "latest_step": int(latest["step"]),
        "best_combined": strip_selection(bests["best_combined"]),
        "best_LE": strip_selection(bests["best_LE"]),
        "best_B": strip_selection(bests["best_B"]),
        "best_raw_projected": strip_selection(bests["best_raw_projected"]),
        "latest": latest,
        "v2e_baseline_latest": V2E_BASELINE_LATEST,
        "comparison_to_v2e_latest": {
            "val_LE_local_rel_delta": float(latest["val_LE_local_rel"] - V2E_BASELINE_LATEST["val_LE_local_rel"]),
            "val_AD_B_local_rel_delta": float(latest["val_AD_B_local_rel"] - V2E_BASELINE_LATEST["val_AD_B_local_rel"]),
            "val_B_raw_projected_rel_delta": float(latest["val_B_model_raw_projected_rel"] - V2E_BASELINE_LATEST["val_B_model_raw_projected_rel"]),
            "train_LE_local_rel_delta": float(latest["train_LE_local_rel"] - V2E_BASELINE_LATEST["train_LE_local_rel"]),
        },
        "formal_prototype_not_final_result": True,
        "uses_old_true176_labels_as_v2_labels": False,
        "checkpoint_written": False,
        "output_paths": {key: str(path) for key, path in paths.items()},
    }
    paths["training_summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--val-cases", default="44,46")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--le-weight", type=float, default=1.0)
    parser.add_argument("--b-weight", type=float, default=1.0)
    parser.add_argument("--zero-weight", type=float, default=10.0)
    parser.add_argument("--frame-batch", type=int, default=32)
    parser.add_argument("--le-point-batch", type=int, default=128)
    parser.add_argument("--ad-point-batch", type=int, default=16)
    parser.add_argument("--eval-frame-batch", type=int, default=16)
    parser.add_argument("--eval-point-batch", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--use-q-amp", action="store_true")
    parser.add_argument("--use-q-dir", action="store_true")
    parser.add_argument("--gate-c", type=float, default=0.1)
    args = parser.parse_args()

    if int(args.steps) < 1:
        raise SystemExit("--steps must be positive")
    if int(args.eval_every) < 1:
        raise SystemExit("--eval-every must be positive")
    train(args)


if __name__ == "__main__":
    main()
