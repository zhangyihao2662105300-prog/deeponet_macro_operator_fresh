#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-case loader/overfit smoke for v3 CSS8 standard-operator compacts.

This is not formal training.  It verifies that multiple v3 compacts can be
loaded together, normalized over the selected smoke pool, batched by case, and
fit by a script-local anchored model:

    q_useful_hat + geometry_global_hat + trunk_features_hat -> LE_local_stack

The script also differentiates the output with respect to ``q_useful_hat`` and
maps the resulting local B back to raw Abaqus coordinates for an audit metric.
No checkpoint is written by default; only lightweight JSON/CSV metrics are
saved under the requested output directory.
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


def parse_case_ids(text: str | None) -> set[int] | None:
    if text is None or not text.strip():
        return None
    out: set[int] = set()
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        out.add(int(item.removeprefix("case")))
    return out


def load_compact_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.compact_list:
        list_path = Path(args.compact_list).resolve()
        for line in list_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                paths.append(Path(stripped).resolve())
    for item in args.compact:
        paths.append(Path(item).resolve())
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    if not unique:
        raise SystemExit("Provide --compact-list or at least one --compact")
    return unique


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
        case_id = int(np.asarray(z["case_id"]).reshape(-1)[0]) if "case_id" in z.files else -1
        q = np.asarray(z["q_useful_hat"], dtype=np.float64)
        geom = np.asarray(z["geometry_global_hat"], dtype=np.float64).reshape(-1)
        trunk = np.asarray(z["trunk_features_hat"], dtype=np.float64)
        le = np.asarray(z["LE_local_stack"], dtype=np.float64)
        b = np.asarray(z["B_local_useful_stack_hat"], dtype=np.float64)
        t_eps = np.asarray(z["T_eps_to_abq_stack"], dtype=np.float64)
        t_q_hat = np.asarray(z["T_q_raw_to_useful_hat"], dtype=np.float64)
        b_raw = np.asarray(z["B_LE128_forward"], dtype=np.float64)

    if q.ndim != 2 or q.shape[1] != 42:
        raise ValueError(f"{path}: q_useful_hat must be [N,42], got {q.shape}")
    if trunk.ndim != 2:
        raise ValueError(f"{path}: trunk_features_hat must be [P,F], got {trunk.shape}")
    if le.shape != (q.shape[0], trunk.shape[0], 6):
        raise ValueError(f"{path}: LE_local_stack must be [N,P,6], got {le.shape}")
    if b.shape != (q.shape[0], trunk.shape[0], 6, 42):
        raise ValueError(f"{path}: B_local_useful_stack_hat must be [N,P,6,42], got {b.shape}")
    if t_eps.shape != (trunk.shape[0], 6, 6):
        raise ValueError(f"{path}: T_eps_to_abq_stack must be [P,6,6], got {t_eps.shape}")
    if t_q_hat.shape != (42, 48):
        raise ValueError(f"{path}: T_q_raw_to_useful_hat must be [42,48], got {t_q_hat.shape}")
    if b_raw.shape != (q.shape[0], trunk.shape[0], 6, 48):
        raise ValueError(f"{path}: B_LE128_forward must be [N,P,6,48], got {b_raw.shape}")
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
    requested_case_ids = parse_case_ids(args.case_ids)
    rows = [load_one(path) for path in load_compact_paths(args)]
    if requested_case_ids is not None:
        rows = [row for row in rows if int(row["case_id"]) in requested_case_ids]
    rows = sorted(rows, key=lambda row: int(row["case_id"]))
    if args.max_cases and int(args.max_cases) > 0:
        rows = rows[: int(args.max_cases)]
    if len(rows) < 2:
        raise SystemExit("Multi-case smoke needs at least two selected compacts")

    q_shape = rows[0]["q"].shape
    geom_shape = rows[0]["geom"].shape
    trunk_shape = rows[0]["trunk"].shape
    for row in rows:
        if row["q"].shape != q_shape:
            raise ValueError("Selected compacts must share q_useful_hat shape for this smoke")
        if row["geom"].shape != geom_shape:
            raise ValueError("Selected compacts must share geometry_global_hat shape for this smoke")
        if row["trunk"].shape != trunk_shape:
            raise ValueError("Selected compacts must share trunk_features_hat shape for this smoke")

    q = np.stack([row["q"] for row in rows], axis=0)
    geom = np.stack([row["geom"] for row in rows], axis=0)
    trunk = np.stack([row["trunk"] for row in rows], axis=0)
    le = np.stack([row["le"] for row in rows], axis=0)
    b = np.stack([row["b"] for row in rows], axis=0)
    t_eps = np.stack([row["t_eps"] for row in rows], axis=0)
    t_q_hat = np.stack([row["t_q_hat"] for row in rows], axis=0)
    b_raw = np.stack([row["b_raw"] for row in rows], axis=0)

    q_norm, q_mean, q_std = standardize_np(q, axis=(0, 1))
    geom_norm, geom_mean, geom_std = standardize_np(geom, axis=0)
    trunk_norm, trunk_mean, trunk_std = standardize_np(trunk, axis=(0, 1))

    return {
        "case_ids": [int(row["case_id"]) for row in rows],
        "compact_paths": [row["path"] for row in rows],
        "q": q.astype(np.float32),
        "q_norm": q_norm.astype(np.float32),
        "q_mean": q_mean.astype(np.float64),
        "q_std": q_std.astype(np.float64),
        "geom": geom.astype(np.float32),
        "geom_norm": geom_norm.astype(np.float32),
        "geom_mean": geom_mean.astype(np.float64),
        "geom_std": geom_std.astype(np.float64),
        "trunk": trunk.astype(np.float32),
        "trunk_norm": trunk_norm.astype(np.float32),
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
        "trunk_norm": torch.as_tensor(data_np["trunk_norm"], dtype=dtype, device=device),
        "le": torch.as_tensor(data_np["le"], dtype=dtype, device=device),
        "b": torch.as_tensor(data_np["b"], dtype=dtype, device=device),
        "t_eps": torch.as_tensor(data_np["t_eps"], dtype=dtype, device=device),
        "t_q_hat": torch.as_tensor(data_np["t_q_hat"], dtype=dtype, device=device),
        "b_raw": torch.as_tensor(data_np["b_raw"], dtype=dtype, device=device),
    }


class V3MultiCaseTinyOperator(nn.Module):
    """Small anchored model for multi-case v3 loader/AD smoke.

    The case-indexed B prior is a smoke-test anchor, not a final architecture.
    It keeps the gate focused on multi-case loading, normalization, batching,
    geometry/trunk consumption, and autograd plumbing.
    """

    def __init__(
        self,
        *,
        q_dim: int,
        geom_dim: int,
        trunk_dim: int,
        b_prior: torch.Tensor,
        hidden: int,
    ) -> None:
        super().__init__()
        self.b_prior = nn.Parameter(b_prior.clone())
        self.state_net = nn.Sequential(
            nn.Linear(q_dim + geom_dim, hidden),
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
        if point_idx is None:
            b_prior = self.b_prior[case_index]
            trunk_use = trunk_norm
        else:
            b_prior = self.b_prior[case_index].index_select(0, point_idx)
            trunk_use = trunk_norm.index_select(0, point_idx)
        linear = torch.einsum("pak,nk->npa", b_prior, q_raw)
        residual_q = self.residual(q_norm, geom_norm, trunk_use)
        residual0 = self.residual(torch.zeros_like(q_norm), geom_norm, trunk_use)
        return linear + residual_q - residual0


def jacobian_case(
    model: V3MultiCaseTinyOperator,
    case_index: int,
    q_norm: torch.Tensor,
    geom_norm: torch.Tensor,
    trunk_norm: torch.Tensor,
    q_mean: torch.Tensor,
    q_std: torch.Tensor,
    point_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return dLE/dq_useful_hat for one case and a batch of frames."""

    def one(q_single_norm: torch.Tensor) -> torch.Tensor:
        q_single_raw = q_single_norm * q_std + q_mean
        return model.forward_case(
            case_index,
            q_single_raw,
            q_single_norm,
            geom_norm,
            trunk_norm,
            point_idx=point_idx,
        )[0]

    jac_norm = vmap(jacrev(one))(q_norm)
    return jac_norm / q_std.reshape(1, 1, 1, -1)


def raw_project_b(ad_b_local_hat: torch.Tensor, t_eps: torch.Tensor, t_q_hat: torch.Tensor) -> torch.Tensor:
    b_abq = torch.einsum("pab,npbk->npak", t_eps, ad_b_local_hat)
    return torch.einsum("npak,kj->npaj", b_abq, t_q_hat)


def evaluate(
    model: V3MultiCaseTinyOperator,
    data: dict[str, torch.Tensor],
    case_ids: list[int],
    *,
    point_chunk: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    q_mean = data["q_mean"]
    q_std = data["q_std"]
    overall = {
        "le_diff_sq": torch.zeros((), device=q_mean.device),
        "le_den_sq": torch.zeros((), device=q_mean.device),
        "ad_diff_sq": torch.zeros((), device=q_mean.device),
        "ad_den_sq": torch.zeros((), device=q_mean.device),
        "ad_dot": torch.zeros((), device=q_mean.device),
        "ad_pred_sq": torch.zeros((), device=q_mean.device),
        "bproj_diff_sq": torch.zeros((), device=q_mean.device),
        "bproj_den_sq": torch.zeros((), device=q_mean.device),
        "braw_diff_sq": torch.zeros((), device=q_mean.device),
        "braw_den_sq": torch.zeros((), device=q_mean.device),
    }
    case_rows: list[dict[str, Any]] = []
    case_count = int(data["q"].shape[0])

    for c in range(case_count):
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

        ad_diff_sq = torch.zeros((), device=q_mean.device)
        ad_den_sq = torch.zeros((), device=q_mean.device)
        ad_dot = torch.zeros((), device=q_mean.device)
        ad_pred_sq = torch.zeros((), device=q_mean.device)
        bproj_diff_sq = torch.zeros((), device=q_mean.device)
        bproj_den_sq = torch.zeros((), device=q_mean.device)
        braw_diff_sq = torch.zeros((), device=q_mean.device)
        braw_den_sq = torch.zeros((), device=q_mean.device)
        p_useful = t_q_hat.T @ t_q_hat
        for p0 in range(0, int(trunk_norm.shape[0]), int(point_chunk)):
            p1 = min(p0 + int(point_chunk), int(trunk_norm.shape[0]))
            point_idx = torch.arange(p0, p1, device=q_mean.device)
            ad_b = jacobian_case(model, c, q_norm, geom_norm, trunk_norm, q_mean, q_std, point_idx=point_idx)
            target = b_local[:, p0:p1]
            diff = ad_b - target
            with torch.no_grad():
                ad_diff_sq = ad_diff_sq + torch.sum(diff.detach() * diff.detach())
                ad_den_sq = ad_den_sq + torch.sum(target * target)
                ad_dot = ad_dot + torch.sum(ad_b.detach() * target)
                ad_pred_sq = ad_pred_sq + torch.sum(ad_b.detach() * ad_b.detach())
                b_model_raw = raw_project_b(ad_b.detach(), t_eps[p0:p1], t_q_hat)
                b_raw_chunk = b_raw[:, p0:p1]
                b_raw_projected = torch.einsum("npaj,jk->npak", b_raw_chunk, p_useful)
                bproj_diff = b_model_raw - b_raw_projected
                braw_diff = b_model_raw - b_raw_chunk
                bproj_diff_sq = bproj_diff_sq + torch.sum(bproj_diff * bproj_diff)
                bproj_den_sq = bproj_den_sq + torch.sum(b_raw_projected * b_raw_projected)
                braw_diff_sq = braw_diff_sq + torch.sum(braw_diff * braw_diff)
                braw_den_sq = braw_den_sq + torch.sum(b_raw_chunk * b_raw_chunk)

        for key, value in [
            ("le_diff_sq", le_diff_sq),
            ("le_den_sq", le_den_sq),
            ("ad_diff_sq", ad_diff_sq),
            ("ad_den_sq", ad_den_sq),
            ("ad_dot", ad_dot),
            ("ad_pred_sq", ad_pred_sq),
            ("bproj_diff_sq", bproj_diff_sq),
            ("bproj_den_sq", bproj_den_sq),
            ("braw_diff_sq", braw_diff_sq),
            ("braw_den_sq", braw_den_sq),
        ]:
            overall[key] = overall[key] + value.detach()

        le_rel = rel_from_squares(le_diff_sq, le_den_sq)
        ad_rel = rel_from_squares(ad_diff_sq, ad_den_sq)
        ad_cos = ad_dot / torch.clamp(torch.sqrt(ad_pred_sq) * torch.sqrt(ad_den_sq), min=1.0e-30)
        bproj_rel = rel_from_squares(bproj_diff_sq, bproj_den_sq)
        braw_rel = rel_from_squares(braw_diff_sq, braw_den_sq)
        case_rows.append(
            {
                "case_index": c,
                "case_id": int(case_ids[c]),
                "train_LE_local_stack_rel": scalar_float(le_rel),
                "train_AD_B_local_useful_hat_rel": scalar_float(ad_rel),
                "train_AD_B_local_useful_hat_cos": scalar_float(ad_cos),
                "B_model_raw_projected_rel": scalar_float(bproj_rel),
                "B_model_raw_rel": scalar_float(braw_rel),
                "selection_score": scalar_float(le_rel + ad_rel),
            }
        )

    le_rel = rel_from_squares(overall["le_diff_sq"], overall["le_den_sq"])
    ad_rel = rel_from_squares(overall["ad_diff_sq"], overall["ad_den_sq"])
    ad_cos = overall["ad_dot"] / torch.clamp(torch.sqrt(overall["ad_pred_sq"]) * torch.sqrt(overall["ad_den_sq"]), min=1.0e-30)
    bproj_rel = rel_from_squares(overall["bproj_diff_sq"], overall["bproj_den_sq"])
    braw_rel = rel_from_squares(overall["braw_diff_sq"], overall["braw_den_sq"])
    metrics = {
        "train_LE_local_stack_rel": scalar_float(le_rel),
        "train_AD_B_local_useful_hat_rel": scalar_float(ad_rel),
        "train_AD_B_local_useful_hat_cos": scalar_float(ad_cos),
        "B_model_raw_projected_rel": scalar_float(bproj_rel),
        "B_model_raw_rel": scalar_float(braw_rel),
        "selection_score": scalar_float(le_rel + ad_rel),
    }
    return metrics, case_rows


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
        "train_LE_local_stack_rel",
        "train_AD_B_local_useful_hat_rel",
        "train_AD_B_local_useful_hat_cos",
        "B_model_raw_projected_rel",
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


def train(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))
    data_np = load_pool(args)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    data = make_tensors(data_np, device=device)
    case_ids = list(data_np["case_ids"])

    model = V3MultiCaseTinyOperator(
        q_dim=int(data["q_norm"].shape[2]),
        geom_dim=int(data["geom_norm"].shape[1]),
        trunk_dim=int(data["trunk_norm"].shape[2]),
        b_prior=data["b"].mean(dim=1),
        hidden=int(args.hidden),
    ).to(device=device, dtype=torch.float32)
    if args.freeze_b_prior:
        model.b_prior.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    le_scale = torch.clamp(torch.sqrt(torch.mean(data["le"] * data["le"])), min=1.0e-12)
    b_scale = torch.clamp(torch.sqrt(torch.mean(data["b"] * data["b"])), min=1.0e-12)
    le_scale_case = torch.clamp(torch.sqrt(torch.mean(data["le"] * data["le"], dim=(1, 2, 3))), min=1.0e-12)
    b_scale_case = torch.clamp(torch.sqrt(torch.mean(data["b"] * data["b"], dim=(1, 2, 3, 4))), min=1.0e-12)

    case_count = int(data["q_norm"].shape[0])
    frame_count = int(data["q_norm"].shape[1])
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
        loss_terms: list[torch.Tensor] = []
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
            cur_le_scale = le_scale_case[case_index] if args.loss_scale_mode == "per-case" else le_scale
            le_loss = torch.mean(((pred - le_target) / cur_le_scale) ** 2)

            ad_idx = torch.randperm(point_count, device=device)[:ad_point_batch]
            b_target = data["b"][case_index].index_select(0, frame_idx).index_select(1, ad_idx)
            ad_b = jacobian_case(
                model,
                int(case_index),
                q_norm_batch,
                data["geom_norm"][case_index],
                data["trunk_norm"][case_index],
                data["q_mean"],
                data["q_std"],
                point_idx=ad_idx,
            )
            cur_b_scale = b_scale_case[case_index] if args.loss_scale_mode == "per-case" else b_scale
            b_loss = torch.mean(((ad_b - b_target) / cur_b_scale) ** 2)
            loss_terms.append(float(args.le_weight) * le_loss + float(args.b_weight) * b_loss)
            le_terms.append(le_loss.detach())
            b_terms.append(b_loss.detach())

        loss = torch.stack(loss_terms).mean()
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
    summary: dict[str, Any] = {
        "audit_name": "v3_css8_multi_case_loader_overfit_smoke",
        "out_root": str(out_root),
        "device": str(device),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "case_ids": case_ids,
        "compact_paths": list(data_np["compact_paths"]),
        "case_count": case_count,
        "frame_count_per_case": frame_count,
        "point_count": point_count,
        "q_useful_hat_dim": int(data["q_norm"].shape[2]),
        "geometry_global_hat_dim": int(data["geom_norm"].shape[1]),
        "trunk_features_hat_dim": int(data["trunk_norm"].shape[2]),
        "model_inputs": ["q_useful_hat", "geometry_global_hat", "trunk_features_hat"],
        "model_output": "LE_local_stack",
        "ad_target": "dLE_local_stack/dq_useful_hat",
        "raw_backprojection": "B_raw_hat = T_eps_to_abq_stack @ AD_B_local_hat @ T_q_raw_to_useful_hat",
        "model_form": "script-local multi-case anchored B-prior smoke: B_prior(case,point) @ q_useful_hat + R(q,geometry,trunk) - R(0,geometry,trunk); not final v3 architecture",
        "b_prior_trainable": bool(model.b_prior.requires_grad),
        "loss_scale_mode": str(args.loss_scale_mode),
        "formal_training": False,
        "checkpoint_written": False,
        "uses_old_true176_labels_as_v3_labels": False,
        "train_only_normalization": {
            "q_useful_hat": "mean/std over all frames of selected smoke cases",
            "geometry_global_hat": "mean/std over selected smoke cases",
            "trunk_features_hat": "mean/std over all points of selected smoke cases",
        },
        "initial_metrics": initial,
        "best_metrics": best,
        "latest_metrics": latest,
        "latest_case_metrics": latest_cases,
    }
    summary_path = out_root / "v3_multi_case_overfit_summary.json"
    history_path = out_root / "v3_multi_case_overfit_history.csv"
    case_history_path = out_root / "v3_multi_case_overfit_case_history.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    write_csv(history_path, history)
    write_csv(case_history_path, case_history)
    print(
        json.dumps(
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
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", default="")
    parser.add_argument("--compact", action="append", default=[])
    parser.add_argument("--case-ids", default="", help="Optional comma-separated case IDs, e.g. 041,045,050")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--le-weight", type=float, default=1.0)
    parser.add_argument("--b-weight", type=float, default=0.1)
    parser.add_argument("--case-batch", type=int, default=3)
    parser.add_argument("--frame-batch", type=int, default=6)
    parser.add_argument("--le-point-batch", type=int, default=64)
    parser.add_argument("--ad-point-batch", type=int, default=8)
    parser.add_argument("--eval-point-batch", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--freeze-b-prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss-scale-mode", choices=["global", "per-case"], default="global")
    args = parser.parse_args()
    if int(args.steps) < 1:
        raise SystemExit("--steps must be positive")
    if int(args.eval_every) < 1:
        raise SystemExit("--eval-every must be positive")
    train(args)


if __name__ == "__main__":
    main()
