#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a script-local v2 multi-case tiny smoke model.

This is not the formal v2 model.  It verifies that the multi-case v2b compact
pool can be loaded, split by case, trained in local strain coordinates,
differentiated with respect to q_useful, and mapped back to raw Abaqus B.

No checkpoint is written by default; only lightweight metrics are saved under
the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
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


def broadcast_t_eps_np(t_eps: np.ndarray, n_frames: int) -> np.ndarray:
    vals = np.asarray(t_eps, dtype=np.float32)
    if vals.shape == (128, 6, 6):
        return np.broadcast_to(vals.reshape(1, 128, 6, 6), (int(n_frames), 128, 6, 6)).copy()
    if vals.shape == (int(n_frames), 128, 6, 6):
        return vals.copy()
    raise ValueError(f"T_eps_to_abq must be [128,6,6] or [N,128,6,6], got {vals.shape}")


def rel_norm_torch(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(num.reshape(-1)) / torch.clamp(torch.linalg.vector_norm(den.reshape(-1)), min=1.0e-30)


def cosine_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    denom = torch.clamp(torch.linalg.vector_norm(aa) * torch.linalg.vector_norm(bb), min=1.0e-30)
    return torch.dot(aa, bb) / denom


def scalar_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


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
            raise KeyError(f"{path}: missing required v2e fields {missing}")
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
            raise ValueError("ip_xi differs across cases; this tiny smoke assumes one shared geometry")
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


class TinyAnchoredLocalModel(nn.Module):
    """Script-local anchored v2 tiny model with no case-id input."""

    def __init__(self, ip_xi: torch.Tensor, b_mean: torch.Tensor, *, hidden: int) -> None:
        super().__init__()
        self.register_buffer("ip_xi", ip_xi)
        self.b_prior = nn.Parameter(b_mean.clone())
        q_dim = int(b_mean.shape[-1])
        self.residual = nn.Sequential(
            nn.Linear(q_dim + 3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 6),
        )

    def forward(self, q: torch.Tensor, point_idx: torch.Tensor | None = None) -> torch.Tensor:
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if point_idx is None:
            xi = self.ip_xi
            b_prior = self.b_prior
        else:
            xi = self.ip_xi.index_select(0, point_idx)
            b_prior = self.b_prior.index_select(0, point_idx)
        batch = q.shape[0]
        point_count = xi.shape[0]
        linear = torch.einsum("pak,bk->bpa", b_prior, q)
        q_expand = q[:, None, :].expand(batch, point_count, q.shape[-1])
        xi_expand = xi[None, :, :].expand(batch, point_count, 3)
        residual_input = torch.cat([q_expand, xi_expand], dim=-1)
        residual_q = self.residual(residual_input.reshape(batch * point_count, -1)).reshape(batch, point_count, 6)
        q_zero = torch.zeros_like(q)
        q0_expand = q_zero[:, None, :].expand(batch, point_count, q.shape[-1])
        residual0_input = torch.cat([q0_expand, xi_expand], dim=-1)
        residual0 = self.residual(residual0_input.reshape(batch * point_count, -1)).reshape(batch, point_count, 6)
        return linear + residual_q - residual0


def jacobian_local(model: TinyAnchoredLocalModel, q: torch.Tensor, point_idx: torch.Tensor | None = None) -> torch.Tensor:
    def one(q_single: torch.Tensor) -> torch.Tensor:
        return model(q_single, point_idx=point_idx)[0]

    return vmap(jacrev(one))(q)


def raw_project_b(ad_b_local: torch.Tensor, t_eps: torch.Tensor, t_q: torch.Tensor) -> torch.Tensor:
    b_abq = torch.einsum("npab,npbk->npak", t_eps, ad_b_local)
    return torch.einsum("npak,nkj->npaj", b_abq, t_q)


def make_tensors(data: dict[str, np.ndarray], *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "q": torch.as_tensor(data["q"], dtype=dtype, device=device),
        "le": torch.as_tensor(data["le"], dtype=dtype, device=device),
        "b": torch.as_tensor(data["b"], dtype=dtype, device=device),
        "t_q": torch.as_tensor(data["t_q"], dtype=dtype, device=device),
        "t_eps": torch.as_tensor(data["t_eps"], dtype=dtype, device=device),
        "b_raw": torch.as_tensor(data["b_raw"], dtype=dtype, device=device),
        "case_ids": torch.as_tensor(data["case_ids"], dtype=torch.long, device=device),
    }


def eval_split(
    model: TinyAnchoredLocalModel,
    data: dict[str, torch.Tensor],
    *,
    prefix: str,
    frame_chunk: int,
    point_chunk: int,
) -> dict[str, float]:
    q = data["q"]
    le = data["le"]
    b_local = data["b"]
    t_eps = data["t_eps"]
    t_q = data["t_q"]
    b_raw = data["b_raw"]
    model.eval()

    le_diff_sq = torch.zeros((), device=q.device, dtype=q.dtype)
    le_den_sq = torch.zeros((), device=q.device, dtype=q.dtype)
    with torch.no_grad():
        for start in range(0, int(q.shape[0]), int(frame_chunk)):
            end = min(start + int(frame_chunk), int(q.shape[0]))
            pred = model(q[start:end])
            target = le[start:end]
            diff = pred - target
            le_diff_sq = le_diff_sq + torch.sum(diff * diff)
            le_den_sq = le_den_sq + torch.sum(target * target)
        le_rel = torch.sqrt(le_diff_sq) / torch.clamp(torch.sqrt(le_den_sq), min=1.0e-30)
        zero_pred = model(torch.zeros(1, q.shape[-1], device=q.device, dtype=q.dtype))
        zero_q = torch.sqrt(torch.mean(zero_pred * zero_pred))

    ad_diff_sq = torch.zeros((), device=q.device, dtype=q.dtype)
    ad_den_sq = torch.zeros((), device=q.device, dtype=q.dtype)
    ad_dot = torch.zeros((), device=q.device, dtype=q.dtype)
    ad_pred_sq = torch.zeros((), device=q.device, dtype=q.dtype)
    bproj_diff_sq = torch.zeros((), device=q.device, dtype=q.dtype)
    bproj_den_sq = torch.zeros((), device=q.device, dtype=q.dtype)
    braw_diff_sq = torch.zeros((), device=q.device, dtype=q.dtype)
    braw_den_sq = torch.zeros((), device=q.device, dtype=q.dtype)

    for f0 in range(0, int(q.shape[0]), int(frame_chunk)):
        f1 = min(f0 + int(frame_chunk), int(q.shape[0]))
        q_chunk = q[f0:f1]
        t_q_chunk = t_q[f0:f1]
        p_useful = torch.einsum("fki,fkj->fij", t_q_chunk, t_q_chunk)
        for p0 in range(0, int(le.shape[1]), int(point_chunk)):
            p1 = min(p0 + int(point_chunk), int(le.shape[1]))
            pidx = torch.arange(p0, p1, device=q.device, dtype=torch.long)
            ad_b = jacobian_local(model, q_chunk, point_idx=pidx)
            target = b_local[f0:f1, p0:p1]
            diff = ad_b - target
            with torch.no_grad():
                ad_diff_sq = ad_diff_sq + torch.sum(diff.detach() * diff.detach())
                ad_den_sq = ad_den_sq + torch.sum(target * target)
                ad_dot = ad_dot + torch.sum(ad_b.detach() * target)
                ad_pred_sq = ad_pred_sq + torch.sum(ad_b.detach() * ad_b.detach())
                t_eps_chunk = t_eps[f0:f1, p0:p1]
                b_raw_chunk = b_raw[f0:f1, p0:p1]
                b_model_raw = raw_project_b(ad_b.detach(), t_eps_chunk, t_q_chunk)
                b_raw_projected = torch.einsum("fpaj,fjk->fpak", b_raw_chunk, p_useful)
                bproj_diff = b_model_raw - b_raw_projected
                braw_diff = b_model_raw - b_raw_chunk
                bproj_diff_sq = bproj_diff_sq + torch.sum(bproj_diff * bproj_diff)
                bproj_den_sq = bproj_den_sq + torch.sum(b_raw_projected * b_raw_projected)
                braw_diff_sq = braw_diff_sq + torch.sum(braw_diff * braw_diff)
                braw_den_sq = braw_den_sq + torch.sum(b_raw_chunk * b_raw_chunk)

    with torch.no_grad():
        ad_rel = torch.sqrt(ad_diff_sq) / torch.clamp(torch.sqrt(ad_den_sq), min=1.0e-30)
        ad_cos = ad_dot / torch.clamp(torch.sqrt(ad_pred_sq) * torch.sqrt(ad_den_sq), min=1.0e-30)
        b_projected_rel = torch.sqrt(bproj_diff_sq) / torch.clamp(torch.sqrt(bproj_den_sq), min=1.0e-30)
        b_raw_rel = torch.sqrt(braw_diff_sq) / torch.clamp(torch.sqrt(braw_den_sq), min=1.0e-30)
    return {
        f"{prefix}_LE_local_rel": scalar_float(le_rel),
        f"{prefix}_AD_B_local_rel": scalar_float(ad_rel),
        f"{prefix}_AD_B_local_cos": scalar_float(ad_cos),
        f"{prefix}_zero_q_LE_local_rms": scalar_float(zero_q),
        f"{prefix}_B_model_raw_projected_rel": scalar_float(b_projected_rel),
        f"{prefix}_B_model_raw_rel": scalar_float(b_raw_rel),
    }


def per_case_metrics(
    model: TinyAnchoredLocalModel,
    data: dict[str, torch.Tensor],
    *,
    split: str,
    frame_chunk: int,
    point_chunk: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_ids = sorted(set(int(v) for v in data["case_ids"].detach().cpu().numpy().tolist()))
    for case_id in case_ids:
        mask = data["case_ids"] == int(case_id)
        idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
        subset = {
            "q": data["q"].index_select(0, idx),
            "le": data["le"].index_select(0, idx),
            "b": data["b"].index_select(0, idx),
            "t_q": data["t_q"].index_select(0, idx),
            "t_eps": data["t_eps"].index_select(0, idx),
            "b_raw": data["b_raw"].index_select(0, idx),
            "case_ids": data["case_ids"].index_select(0, idx),
        }
        metrics = eval_split(model, subset, prefix=split, frame_chunk=frame_chunk, point_chunk=point_chunk)
        rows.append(
            {
                "split": split,
                "case_id": int(case_id),
                "frame_count": int(idx.numel()),
                "LE_local_rel": metrics[f"{split}_LE_local_rel"],
                "AD_B_local_rel": metrics[f"{split}_AD_B_local_rel"],
                "AD_B_local_cos": metrics[f"{split}_AD_B_local_cos"],
                "zero_q_LE_local_rms": metrics[f"{split}_zero_q_LE_local_rms"],
                "B_model_raw_projected_rel": metrics[f"{split}_B_model_raw_projected_rel"],
                "B_model_raw_rel": metrics[f"{split}_B_model_raw_rel"],
            }
        )
    return rows


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
        "train_LE_local_rel",
        "val_LE_local_rel",
        "train_AD_B_local_rel",
        "val_AD_B_local_rel",
        "train_AD_B_local_cos",
        "val_AD_B_local_cos",
        "LE_local_rel",
        "AD_B_local_rel",
        "AD_B_local_cos",
        "zero_q_LE_local_rms",
        "B_model_raw_projected_rel",
        "B_model_raw_rel",
    ]
    ordered = [key for key in preferred if key in fields]
    ordered.extend(key for key in fields if key not in ordered)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def selection_score(metrics: dict[str, float]) -> float:
    return float(metrics["val_LE_local_rel"] + metrics["val_AD_B_local_rel"])


def train(args: argparse.Namespace) -> dict[str, Any]:
    compact_list = Path(args.compact_list).resolve()
    paths = read_compact_list(compact_list)
    if not paths:
        raise SystemExit("compact list is empty")
    cases = [load_case(path) for path in paths]
    case_ids = sorted(c.case_id for c in cases)
    val_cases = parse_cases(args.val_cases)
    if not val_cases:
        raise SystemExit("--val-cases must not be empty")
    train_cases = [case for case in case_ids if case not in set(val_cases)]
    if not train_cases:
        raise SystemExit("no train cases remain after val split")
    overlap = bool(set(train_cases).intersection(val_cases))
    if overlap:
        raise SystemExit("train/val case split overlaps")
    unknown_val = sorted(set(val_cases).difference(case_ids))
    if unknown_val:
        raise SystemExit(f"val cases not found in compact list: {unknown_val}")

    train_np = concat_cases([c for c in cases if c.case_id in set(train_cases)])
    val_np = concat_cases([c for c in cases if c.case_id in set(val_cases)])
    set_seed(int(args.seed))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32
    train_data = make_tensors(train_np, device=device, dtype=dtype)
    val_data = make_tensors(val_np, device=device, dtype=dtype)
    ip_xi = torch.as_tensor(train_np["ip_xi"], dtype=dtype, device=device)
    b_mean = train_data["b"].mean(dim=0)
    model = TinyAnchoredLocalModel(ip_xi, b_mean, hidden=int(args.hidden)).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    le_scale = torch.clamp(torch.sqrt(torch.mean(train_data["le"] * train_data["le"])), min=1.0e-12)
    b_scale = torch.clamp(torch.sqrt(torch.mean(train_data["b"] * train_data["b"])), min=1.0e-12)
    frame_count = int(train_data["q"].shape[0])
    point_count = int(train_data["le"].shape[1])
    frame_batch = min(int(args.frame_batch), frame_count)
    le_point_batch = min(int(args.le_point_batch), point_count)
    ad_point_batch = min(int(args.ad_point_batch), point_count)

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    eval_steps = set(range(0, int(args.steps) + 1, int(args.eval_every)))
    eval_steps.add(int(args.steps))

    def eval_all(step: int, extra: dict[str, float] | None = None) -> dict[str, Any]:
        metrics = {
            "step": int(step),
            **(extra or {}),
            **eval_split(
                model,
                train_data,
                prefix="train",
                frame_chunk=int(args.eval_frame_batch),
                point_chunk=int(args.eval_point_batch),
            ),
            **eval_split(
                model,
                val_data,
                prefix="val",
                frame_chunk=int(args.eval_frame_batch),
                point_chunk=int(args.eval_point_batch),
            ),
        }
        metrics["selection_score"] = selection_score(metrics)
        return metrics

    initial = eval_all(0)
    best = dict(initial)
    history.append(initial)

    for step in range(1, int(args.steps) + 1):
        model.train()
        frame_idx = torch.randperm(frame_count, device=device)[:frame_batch]
        if le_point_batch == point_count:
            le_idx = None
            le_target = train_data["le"].index_select(0, frame_idx)
        else:
            le_idx = torch.randperm(point_count, device=device)[:le_point_batch]
            le_target = train_data["le"].index_select(0, frame_idx).index_select(1, le_idx)
        q_batch = train_data["q"].index_select(0, frame_idx)
        pred = model(q_batch, point_idx=le_idx)
        le_loss = torch.mean(((pred - le_target) / le_scale) ** 2)

        if ad_point_batch == point_count:
            ad_idx = None
            b_target = train_data["b"].index_select(0, frame_idx)
        else:
            ad_idx = torch.randperm(point_count, device=device)[:ad_point_batch]
            b_target = train_data["b"].index_select(0, frame_idx).index_select(1, ad_idx)
        ad_b = jacobian_local(model, q_batch, point_idx=ad_idx)
        b_loss = torch.mean(((ad_b - b_target) / b_scale) ** 2)
        zero_pred = model(torch.zeros(1, q_batch.shape[-1], device=device, dtype=dtype))
        zero_loss = torch.mean((zero_pred / le_scale) ** 2)
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
            if float(row["selection_score"]) < float(best["selection_score"]):
                best = dict(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    latest = history[-1]
    per_case_rows = per_case_metrics(
        model,
        train_data,
        split="train",
        frame_chunk=int(args.eval_frame_batch),
        point_chunk=int(args.eval_point_batch),
    ) + per_case_metrics(
        model,
        val_data,
        split="val",
        frame_chunk=int(args.eval_frame_batch),
        point_chunk=int(args.eval_point_batch),
    )
    best_metrics_path = out_root / "best_metrics.json"
    summary_path = out_root / "training_summary.json"
    history_path = out_root / "metrics_history.csv"
    per_case_path = out_root / "per_case_attribution.csv"

    summary: dict[str, Any] = {
        "audit_name": "v2e_multi_case_tiny_training_smoke",
        "compact_list": str(compact_list),
        "out_root": str(out_root),
        "compact_count": len(cases),
        "case_ids": case_ids,
        "train_cases": train_cases,
        "val_cases": val_cases,
        "validation_is_overlapping": False,
        "train_frame_count": int(train_data["q"].shape[0]),
        "val_frame_count": int(val_data["q"].shape[0]),
        "point_count": point_count,
        "q_useful_dim": 42,
        "device": str(device),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "eval_frame_batch": int(args.eval_frame_batch),
        "eval_point_batch": int(args.eval_point_batch),
        "latest_step": int(latest["step"]),
        "best_step": int(best["step"]),
        "branch_input_coordinate": "q_useful",
        "trunk_input_coordinate": "ip_xi",
        "output_coordinate": "LE_local_jacobian_frame",
        "ad_target_coordinate": "d(LE_local_jacobian_frame)/d(q_useful)",
        "raw_backprojection": "B_raw_hat_model = T_eps_to_abq @ AD_B_local_hat @ T_q_raw_to_useful",
        "model_form": "LE_hat = B_prior_table(point) @ q_useful + R(q_useful,ip_xi) - R(0,ip_xi)",
        "case_id_used_as_input": False,
        "formal_training": False,
        "uses_old_true176_labels_as_v2_labels": False,
        "checkpoint_written": False,
        "best_metrics": best,
        "latest_metrics": latest,
        "per_case_attribution_csv": str(per_case_path),
        "metrics_history_csv": str(history_path),
        "best_metrics_json": str(best_metrics_path),
    }

    best_metrics_path.write_text(json.dumps(best, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    write_csv(history_path, history)
    write_csv(per_case_path, per_case_rows)
    print(json.dumps({**summary, "training_summary_json": str(summary_path)}, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--val-cases", default="44,46")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--le-weight", type=float, default=1.0)
    parser.add_argument("--b-weight", type=float, default=1.0)
    parser.add_argument("--zero-weight", type=float, default=1.0)
    parser.add_argument("--frame-batch", type=int, default=32)
    parser.add_argument("--le-point-batch", type=int, default=128)
    parser.add_argument("--ad-point-batch", type=int, default=16)
    parser.add_argument("--eval-frame-batch", type=int, default=16)
    parser.add_argument("--eval-point-batch", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    args = parser.parse_args()
    if int(args.steps) < 1:
        raise SystemExit("--steps must be positive")
    if int(args.eval_every) < 1:
        raise SystemExit("--eval-every must be positive")
    train(args)


if __name__ == "__main__":
    main()
