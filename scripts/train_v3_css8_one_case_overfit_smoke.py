#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-case loader/overfit smoke for the v3 CSS8 standard-operator compact.

This is not formal training.  It verifies that a small script-local model can
read the v3 fields:

    q_useful_hat + geometry_global_hat + trunk_features_hat -> LE_local_stack

and differentiate the output with respect to ``q_useful_hat``.  No checkpoint is
written by default; only lightweight JSON/CSV metrics are saved.
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


def rel_norm_torch(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(num.reshape(-1)) / torch.clamp(torch.linalg.vector_norm(den.reshape(-1)), min=1.0e-30)


def cosine_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    denom = torch.clamp(torch.linalg.vector_norm(aa) * torch.linalg.vector_norm(bb), min=1.0e-30)
    return torch.dot(aa, bb) / denom


def scalar_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def standardize_np(value: np.ndarray, *, eps: float = 1.0e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = np.asarray(value, dtype=np.float64)
    mean = np.mean(vals, axis=0, keepdims=True)
    std = np.std(vals, axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (vals - mean) / std, mean.reshape(-1), std.reshape(-1)


def load_v3_compact(path: Path) -> dict[str, np.ndarray | str | int]:
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
            raise KeyError(f"{path}: missing required v3 smoke fields {missing}")
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
        case_id = int(np.asarray(z["case_id"]).reshape(-1)[0]) if "case_id" in z.files else -1

    if q.ndim != 2 or q.shape[1] != 42:
        raise ValueError(f"q_useful_hat must be [N,42], got {q.shape}")
    if trunk.ndim != 2:
        raise ValueError(f"trunk_features_hat must be [P,F], got {trunk.shape}")
    if le.shape != (q.shape[0], trunk.shape[0], 6):
        raise ValueError(f"LE_local_stack must be [N,P,6], got {le.shape}")
    if b.shape != (q.shape[0], trunk.shape[0], 6, 42):
        raise ValueError(f"B_local_useful_stack_hat must be [N,P,6,42], got {b.shape}")
    if t_eps.shape != (trunk.shape[0], 6, 6):
        raise ValueError(f"T_eps_to_abq_stack must be [P,6,6], got {t_eps.shape}")
    if t_q_hat.shape != (42, 48):
        raise ValueError(f"T_q_raw_to_useful_hat must be [42,48], got {t_q_hat.shape}")
    if b_raw.shape != (q.shape[0], trunk.shape[0], 6, 48):
        raise ValueError(f"B_LE128_forward must be [N,P,6,48], got {b_raw.shape}")

    q_norm, q_mean, q_std = standardize_np(q)
    trunk_norm, trunk_mean, trunk_std = standardize_np(trunk)
    geom_norm = geom.copy()
    geom_mean = np.zeros_like(geom)
    geom_std = np.ones_like(geom)
    # One-case smoke has a single global geometry vector.  Passing it through a
    # linear layer is still a loader check, but normalization cannot learn
    # between-geometry statistics from one sample.
    return {
        "case_id": case_id,
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


class V3TinyOperator(nn.Module):
    """Small anchored local model for a v3 one-case smoke.

    The model deliberately consumes q, global geometry, and trunk features.  It
    uses a trainable B-prior table initialized from the one-case v3 B labels so
    this smoke validates the loader/AD chain instead of asking a tiny MLP to
    rediscover the finite-element tangent from scratch.
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

    def forward(self, q_norm: torch.Tensor, geom_norm: torch.Tensor, trunk_norm: torch.Tensor) -> torch.Tensor:
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

    def forward_with_point_index(
        self,
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
            b_prior = self.b_prior
            trunk_use = trunk_norm
        else:
            b_prior = self.b_prior.index_select(0, point_idx)
            trunk_use = trunk_norm.index_select(0, point_idx)
        linear = torch.einsum("pak,nk->npa", b_prior, q_raw)
        residual_q = self(q_norm, geom_norm, trunk_use)
        q0_norm = torch.zeros_like(q_norm)
        residual0 = self(q0_norm, geom_norm, trunk_use)
        return linear + residual_q - residual0


def jacobian_local(
    model: V3TinyOperator,
    q_norm: torch.Tensor,
    geom_norm: torch.Tensor,
    trunk_norm: torch.Tensor,
    q_mean: torch.Tensor,
    q_std: torch.Tensor,
    point_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return dLE/dq_useful_hat from a model that internally sees normalized q."""

    def one(q_single_norm: torch.Tensor) -> torch.Tensor:
        q_single_raw = q_single_norm * q_std + q_mean
        return model.forward_with_point_index(
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
    model: V3TinyOperator,
    data: dict[str, torch.Tensor],
    *,
    point_chunk: int,
) -> dict[str, float]:
    q_norm = data["q_norm"]
    q_raw = data["q"]
    geom_norm = data["geom_norm"]
    trunk_norm = data["trunk_norm"]
    le = data["le"]
    b_local = data["b"]
    q_mean = data["q_mean"]
    q_std = data["q_std"]
    t_eps = data["t_eps"]
    t_q_hat = data["t_q_hat"]
    b_raw = data["b_raw"]
    model.eval()

    with torch.no_grad():
        pred = model.forward_with_point_index(q_raw, q_norm, geom_norm, trunk_norm)
        le_rel = rel_norm_torch(pred - le, le)

    ad_diff_sq = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    ad_den_sq = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    ad_dot = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    ad_pred_sq = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    bproj_diff_sq = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    bproj_den_sq = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    braw_diff_sq = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    braw_den_sq = torch.zeros((), device=q_norm.device, dtype=q_norm.dtype)
    p_useful = t_q_hat.T @ t_q_hat
    for p0 in range(0, int(trunk_norm.shape[0]), int(point_chunk)):
        p1 = min(p0 + int(point_chunk), int(trunk_norm.shape[0]))
        ad_b = jacobian_local(model, q_norm, geom_norm, trunk_norm, q_mean, q_std, point_idx=torch.arange(p0, p1, device=q_norm.device))
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

    with torch.no_grad():
        ad_rel = torch.sqrt(ad_diff_sq) / torch.clamp(torch.sqrt(ad_den_sq), min=1.0e-30)
        ad_cos = ad_dot / torch.clamp(torch.sqrt(ad_pred_sq) * torch.sqrt(ad_den_sq), min=1.0e-30)
        b_projected_rel = torch.sqrt(bproj_diff_sq) / torch.clamp(torch.sqrt(bproj_den_sq), min=1.0e-30)
        b_raw_rel = torch.sqrt(braw_diff_sq) / torch.clamp(torch.sqrt(braw_den_sq), min=1.0e-30)
    return {
        "train_LE_local_stack_rel": scalar_float(le_rel),
        "train_AD_B_local_useful_hat_rel": scalar_float(ad_rel),
        "train_AD_B_local_useful_hat_cos": scalar_float(ad_cos),
        "B_model_raw_projected_rel": scalar_float(b_projected_rel),
        "B_model_raw_rel": scalar_float(b_raw_rel),
        "selection_score": scalar_float(le_rel + ad_rel),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    preferred = [
        "step",
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


def make_tensors(data_np: dict[str, np.ndarray | str | int], *, device: torch.device) -> dict[str, torch.Tensor]:
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


def train(args: argparse.Namespace) -> dict[str, Any]:
    compact = Path(args.compact).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))
    data_np = load_v3_compact(compact)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    data = make_tensors(data_np, device=device)

    model = V3TinyOperator(
        q_dim=int(data["q_norm"].shape[1]),
        geom_dim=int(data["geom_norm"].shape[0]),
        trunk_dim=int(data["trunk_norm"].shape[1]),
        b_prior=data["b"].mean(dim=0),
        hidden=int(args.hidden),
    ).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    le_scale = torch.clamp(torch.sqrt(torch.mean(data["le"] * data["le"])), min=1.0e-12)
    b_scale = torch.clamp(torch.sqrt(torch.mean(data["b"] * data["b"])), min=1.0e-12)
    frame_count = int(data["q_norm"].shape[0])
    point_count = int(data["trunk_norm"].shape[0])
    frame_batch = min(int(args.frame_batch), frame_count)
    le_point_batch = min(int(args.le_point_batch), point_count)
    ad_point_batch = min(int(args.ad_point_batch), point_count)

    history: list[dict[str, Any]] = []
    initial = evaluate(model, data, point_chunk=int(args.eval_point_batch))
    best = {"step": 0, **initial}
    history.append(best)
    eval_steps = set(range(0, int(args.steps) + 1, int(args.eval_every)))
    eval_steps.add(int(args.steps))

    for step in range(1, int(args.steps) + 1):
        model.train()
        frame_idx = torch.randperm(frame_count, device=device)[:frame_batch]
        le_idx = torch.randperm(point_count, device=device)[:le_point_batch]
        q_raw_batch = data["q"].index_select(0, frame_idx)
        q_norm_batch = data["q_norm"].index_select(0, frame_idx)
        le_target = data["le"].index_select(0, frame_idx).index_select(1, le_idx)
        pred = model.forward_with_point_index(
            q_raw_batch,
            q_norm_batch,
            data["geom_norm"],
            data["trunk_norm"],
            point_idx=le_idx,
        )
        le_loss = torch.mean(((pred - le_target) / le_scale) ** 2)

        ad_idx = torch.randperm(point_count, device=device)[:ad_point_batch]
        b_target = data["b"].index_select(0, frame_idx).index_select(1, ad_idx)
        ad_b = jacobian_local(
            model,
            q_norm_batch,
            data["geom_norm"],
            data["trunk_norm"],
            data["q_mean"],
            data["q_std"],
            point_idx=ad_idx,
        )
        b_loss = torch.mean(((ad_b - b_target) / b_scale) ** 2)
        loss = float(args.le_weight) * le_loss + float(args.b_weight) * b_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        opt.step()

        if step in eval_steps:
            metrics = evaluate(model, data, point_chunk=int(args.eval_point_batch))
            row = {
                "step": int(step),
                "loss": scalar_float(loss.detach()),
                "le_loss": scalar_float(le_loss.detach()),
                "b_loss": scalar_float(b_loss.detach()),
                **metrics,
            }
            history.append(row)
            if float(row["selection_score"]) < float(best["selection_score"]):
                best = dict(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    latest = history[-1]
    summary: dict[str, Any] = {
        "audit_name": "v3_css8_one_case_loader_overfit_smoke",
        "compact": str(compact),
        "case_id": int(data_np["case_id"]),
        "out_root": str(out_root),
        "device": str(device),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "frame_count": frame_count,
        "point_count": point_count,
        "q_useful_hat_dim": int(data["q_norm"].shape[1]),
        "geometry_global_hat_dim": int(data["geom_norm"].shape[0]),
        "trunk_features_hat_dim": int(data["trunk_norm"].shape[1]),
        "model_inputs": ["q_useful_hat", "geometry_global_hat", "trunk_features_hat"],
        "model_output": "LE_local_stack",
        "ad_target": "dLE_local_stack/dq_useful_hat",
        "raw_backprojection": "B_raw_hat = T_eps_to_abq_stack @ AD_B_local_hat @ T_q_raw_to_useful_hat",
        "model_form": "script-local anchored B-prior smoke: B_prior(point) @ q_useful_hat + R(q,geometry,trunk) - R(0,geometry,trunk); not final v3 architecture",
        "formal_training": False,
        "checkpoint_written": False,
        "uses_old_true176_labels_as_v3_labels": False,
        "train_only_normalization": {
            "q_useful_hat": "mean/std over frames of this one case",
            "trunk_features_hat": "mean/std over points of this one case",
            "geometry_global_hat": "single vector, passed through unchanged for one-case smoke",
        },
        "initial_metrics": initial,
        "best_metrics": best,
        "latest_metrics": latest,
    }
    summary_path = out_root / "v3_one_case_overfit_summary.json"
    history_path = out_root / "v3_one_case_overfit_history.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    write_csv(history_path, history)
    print(
        json.dumps(
            {**summary, "summary_path": str(summary_path), "history_path": str(history_path)},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=json_default,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--le-weight", type=float, default=1.0)
    parser.add_argument("--b-weight", type=float, default=0.1)
    parser.add_argument("--frame-batch", type=int, default=10)
    parser.add_argument("--le-point-batch", type=int, default=128)
    parser.add_argument("--ad-point-batch", type=int, default=16)
    parser.add_argument("--eval-point-batch", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    args = parser.parse_args()
    if int(args.steps) < 1:
        raise SystemExit("--steps must be positive")
    if int(args.eval_every) < 1:
        raise SystemExit("--eval-every must be positive")
    train(args)


if __name__ == "__main__":
    main()
