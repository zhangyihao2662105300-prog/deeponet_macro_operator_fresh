#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a script-local v2 one-case tiny smoke model.

This is not the formal query-point model.  It is a small, self-contained smoke
test that verifies the v2b compact can be read by a network, differentiated
with respect to q_useful, and mapped back to raw Abaqus B coordinates.

No checkpoint is written by default; only lightweight JSON/CSV metrics are
saved under the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
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


def rel_norm_torch(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(num.reshape(-1)) / torch.clamp(torch.linalg.vector_norm(den.reshape(-1)), min=1.0e-30)


def cosine_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    denom = torch.clamp(torch.linalg.vector_norm(aa) * torch.linalg.vector_norm(bb), min=1.0e-30)
    return torch.dot(aa, bb) / denom


def scalar_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


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


def load_compact(path: Path) -> dict[str, np.ndarray | str]:
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
            raise KeyError(f"{path}: missing required v2c tiny-train fields {missing}")
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

    if q.ndim != 2:
        raise ValueError(f"q_useful must be [N,K], got {q.shape}")
    if le.shape != (q.shape[0], 128, 6):
        raise ValueError(f"LE128_local must be [N,128,6], got {le.shape}")
    if b.shape != (q.shape[0], 128, 6, q.shape[1]):
        raise ValueError(f"{b_key} must be [N,128,6,K], got {b.shape}")
    if t_q.shape != (q.shape[1], 48):
        raise ValueError(f"T_q_raw_to_useful must be [K,48], got {t_q.shape}")
    if b_raw.shape != (q.shape[0], 128, 6, 48):
        raise ValueError(f"B_LE128_forward must be [N,128,6,48], got {b_raw.shape}")
    if ip_xi.shape != (128, 3):
        raise ValueError(f"ip_xi must be [128,3], got {ip_xi.shape}")

    return {
        "q": q,
        "le": le,
        "b": b,
        "b_key": b_key,
        "t_q": t_q,
        "t_eps": broadcast_t_eps_np(t_eps, q.shape[0]),
        "b_raw": b_raw,
        "ip_xi": ip_xi,
    }


class TinyAnchoredLocalModel(nn.Module):
    """One-case smoke model with an anchored local-strain output.

    The trainable B table is initialized from the frame-mean local B label.  This
    is deliberate for the smoke test: the goal is to verify the coordinate/AD
    chain, not to claim a deployable architecture or cross-case generalization.
    """

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

        linear = torch.einsum("pak,bk->bpa", b_prior, q)
        batch = q.shape[0]
        points = xi.shape[0]
        q_expand = q[:, None, :].expand(batch, points, q.shape[-1])
        xi_expand = xi[None, :, :].expand(batch, points, 3)
        residual_input = torch.cat([q_expand, xi_expand], dim=-1)
        residual_q = self.residual(residual_input.reshape(batch * points, -1)).reshape(batch, points, 6)

        q_zero = torch.zeros_like(q)
        q0_expand = q_zero[:, None, :].expand(batch, points, q.shape[-1])
        residual0_input = torch.cat([q0_expand, xi_expand], dim=-1)
        residual0 = self.residual(residual0_input.reshape(batch * points, -1)).reshape(batch, points, 6)
        return linear + residual_q - residual0


def jacobian_local(model: TinyAnchoredLocalModel, q: torch.Tensor, point_idx: torch.Tensor | None = None) -> torch.Tensor:
    def one(q_single: torch.Tensor) -> torch.Tensor:
        return model(q_single, point_idx=point_idx)[0]

    return vmap(jacrev(one))(q)


def raw_project_b(ad_b_local: torch.Tensor, t_eps: torch.Tensor, t_q: torch.Tensor) -> torch.Tensor:
    b_abq = torch.einsum("npab,npbk->npak", t_eps, ad_b_local)
    return torch.einsum("npak,kj->npaj", b_abq, t_q)


@torch.no_grad()
def point_norm_range(q: torch.Tensor) -> tuple[float, float]:
    norms = torch.linalg.vector_norm(q, dim=1)
    return float(norms.min().cpu()), float(norms.max().cpu())


def evaluate(
    model: TinyAnchoredLocalModel,
    q: torch.Tensor,
    le: torch.Tensor,
    b_local: torch.Tensor,
    t_eps: torch.Tensor,
    t_q: torch.Tensor,
    b_raw: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        pred = model(q)
        le_rel = rel_norm_torch(pred - le, le)
        zero_pred = model(torch.zeros(1, q.shape[-1], device=q.device, dtype=q.dtype))
        zero_q = torch.sqrt(torch.mean(zero_pred * zero_pred))
    ad_b = jacobian_local(model, q)
    with torch.no_grad():
        ad_rel = rel_norm_torch(ad_b - b_local, b_local)
        ad_cos = cosine_torch(ad_b, b_local)
        b_model_raw = raw_project_b(ad_b, t_eps, t_q)
        p_useful = t_q.T @ t_q
        b_raw_projected = torch.einsum("npaj,jk->npak", b_raw, p_useful)
        b_model_raw_projected_rel = rel_norm_torch(b_model_raw - b_raw_projected, b_raw_projected)
        b_model_raw_rel = rel_norm_torch(b_model_raw - b_raw, b_raw)
        loss_like = le_rel + ad_rel
    return {
        "train_LE_local_rel": scalar_float(le_rel),
        "train_AD_B_local_rel": scalar_float(ad_rel),
        "train_AD_B_local_cos": scalar_float(ad_cos),
        "zero_q_LE_local_rms": scalar_float(zero_q),
        "B_model_raw_projected_rel": scalar_float(b_model_raw_projected_rel),
        "B_model_raw_rel": scalar_float(b_model_raw_rel),
        "selection_score": scalar_float(loss_like),
    }


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    preferred = [
        "step",
        "loss",
        "le_loss",
        "b_loss",
        "zero_loss",
        "train_LE_local_rel",
        "train_AD_B_local_rel",
        "train_AD_B_local_cos",
        "zero_q_LE_local_rms",
        "B_model_raw_projected_rel",
        "B_model_raw_rel",
        "selection_score",
    ]
    present = {key for row in rows for key in row.keys()}
    keys = [key for key in preferred if key in present]
    keys.extend(sorted(present.difference(keys)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def train(args: argparse.Namespace) -> dict[str, Any]:
    compact = Path(args.compact).resolve()
    data = load_compact(compact)
    set_seed(int(args.seed))

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32
    q = torch.as_tensor(data["q"], dtype=dtype, device=device)
    le = torch.as_tensor(data["le"], dtype=dtype, device=device)
    b_local = torch.as_tensor(data["b"], dtype=dtype, device=device)
    t_q = torch.as_tensor(data["t_q"], dtype=dtype, device=device)
    t_eps = torch.as_tensor(data["t_eps"], dtype=dtype, device=device)
    b_raw = torch.as_tensor(data["b_raw"], dtype=dtype, device=device)
    ip_xi = torch.as_tensor(data["ip_xi"], dtype=dtype, device=device)
    b_mean = b_local.mean(dim=0)

    model = TinyAnchoredLocalModel(ip_xi, b_mean, hidden=int(args.hidden)).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    le_scale = torch.clamp(torch.sqrt(torch.mean(le * le)), min=1.0e-12)
    b_scale = torch.clamp(torch.sqrt(torch.mean(b_local * b_local)), min=1.0e-12)
    point_count = int(le.shape[1])
    ad_point_batch = min(int(args.ad_point_batch), point_count)
    le_point_batch = min(int(args.le_point_batch), point_count)

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    best: dict[str, Any] | None = None
    eval_steps = set(range(0, int(args.steps) + 1, int(args.eval_every)))
    eval_steps.add(int(args.steps))

    initial = evaluate(model, q, le, b_local, t_eps, t_q, b_raw)
    best = {"step": 0, **initial}
    history.append({"step": 0, **initial})

    for step in range(1, int(args.steps) + 1):
        model.train()
        if le_point_batch == point_count:
            le_idx = None
            le_target = le
        else:
            le_idx = torch.randperm(point_count, device=device)[:le_point_batch]
            le_target = le.index_select(1, le_idx)
        pred = model(q, point_idx=le_idx)
        le_loss = torch.mean(((pred - le_target) / le_scale) ** 2)

        if ad_point_batch == point_count:
            ad_idx = None
            b_target = b_local
        else:
            ad_idx = torch.randperm(point_count, device=device)[:ad_point_batch]
            b_target = b_local.index_select(1, ad_idx)
        ad_b = jacobian_local(model, q, point_idx=ad_idx)
        b_loss = torch.mean(((ad_b - b_target) / b_scale) ** 2)
        zero_pred = model(torch.zeros(1, q.shape[-1], device=device, dtype=dtype))
        zero_loss = torch.mean((zero_pred / le_scale) ** 2)
        loss = float(args.le_weight) * le_loss + float(args.b_weight) * b_loss + float(args.zero_weight) * zero_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        opt.step()

        if step in eval_steps:
            metrics = evaluate(model, q, le, b_local, t_eps, t_q, b_raw)
            row = {
                "step": int(step),
                "loss": scalar_float(loss.detach()),
                "le_loss": scalar_float(le_loss.detach()),
                "b_loss": scalar_float(b_loss.detach()),
                "zero_loss": scalar_float(zero_loss.detach()),
                **metrics,
            }
            history.append(row)
            if best is None or float(metrics["selection_score"]) < float(best["selection_score"]):
                best = {"step": int(step), **metrics}
            print(json.dumps(row, sort_keys=True), flush=True)

    assert best is not None
    q_norm_min, q_norm_max = point_norm_range(q)
    summary: dict[str, Any] = {
        "audit_name": "v2c_one_case_tiny_training_smoke",
        "compact": str(compact),
        "out_root": str(out_root),
        "device": str(device),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "eval_every": int(args.eval_every),
        "frame_count": int(q.shape[0]),
        "point_count": int(le.shape[1]),
        "q_useful_dim": int(q.shape[1]),
        "branch_input_coordinate": "q_useful",
        "trunk_input_coordinate": "ip_xi",
        "output_coordinate": "LE_local_jacobian_frame",
        "ad_target_key": str(data["b_key"]),
        "ad_target_coordinate": "d(LE_local_jacobian_frame)/d(q_useful)",
        "raw_backprojection": "B_raw_hat_model = T_eps_to_abq @ AD_B_local_hat @ T_q_raw_to_useful",
        "model_form": "LE_hat = B_prior_table(point) @ q_useful + R(q_useful,ip_xi) - R(0,ip_xi)",
        "formal_training": False,
        "uses_old_true176_labels_as_v2_labels": False,
        "checkpoint_written": False,
        "q_useful_norm_min": q_norm_min,
        "q_useful_norm_max": q_norm_max,
        "initial_metrics": initial,
        "best_step": int(best["step"]),
        "train_LE_local_rel": float(best["train_LE_local_rel"]),
        "train_AD_B_local_rel": float(best["train_AD_B_local_rel"]),
        "train_AD_B_local_cos": float(best["train_AD_B_local_cos"]),
        "zero_q_LE_local_rms": float(best["zero_q_LE_local_rms"]),
        "B_model_raw_projected_rel": float(best["B_model_raw_projected_rel"]),
        "B_model_raw_rel": float(best["B_model_raw_rel"]),
        "selection_score": float(best["selection_score"]),
    }

    summary_path = out_root / "tiny_train_summary.json"
    history_path = out_root / "tiny_train_history.csv"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    write_history(history_path, history)
    print(
        json.dumps(
            {**summary, "tiny_train_summary_path": str(summary_path), "tiny_train_history_path": str(history_path)},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=json_default,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", required=True, help="Input v2b local-strain pilot compact")
    parser.add_argument("--out-root", required=True, help="Output directory for lightweight metrics")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--le-weight", type=float, default=1.0)
    parser.add_argument("--b-weight", type=float, default=1.0)
    parser.add_argument("--zero-weight", type=float, default=0.01)
    parser.add_argument("--le-point-batch", type=int, default=128)
    parser.add_argument("--ad-point-batch", type=int, default=16)
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
