#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit a point-conditioned B prior directly on Macro16 compact data.

This is a small diagnostic for Gate 11.  It does not change the production
model, data contract, q48 order, LE order, or integration rule.
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
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macro_deeponet.macro16_geometry import macro16_source128_point_table  # noqa: E402
from macro_deeponet.models import MLP  # noqa: E402
from macro_deeponet.train_macro16_boundary_sobolev import (  # noqa: E402
    _le_std_scale,
    build_physical_b_loss_scale,
    compact_paths_from_args,
    load_macro16_compacts,
    read_path_list,
    split_indices,
    standardize,
)
from macro_deeponet.true176_data import stats  # noqa: E402


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


def rel_np(pred: np.ndarray, true: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(true, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(p - t) / max(float(np.linalg.norm(t)), 1.0e-300))


class PointBPrior(nn.Module):
    def __init__(self, point_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.net = MLP(point_dim, 6 * 48, hidden_dim=hidden_dim, depth=depth, activation=nn.Tanh, zero_last=False)

    def forward(self, point: torch.Tensor) -> torch.Tensor:
        return self.net(point).view(point.shape[0], point.shape[1], 6, 48)


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {str(k): v.detach().cpu().clone() for k, v in module.state_dict().items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device("cuda" if bool(args.cuda) and torch.cuda.is_available() else "cpu")
    paths = compact_paths_from_args(args)
    data = load_macro16_compacts(
        paths,
        point_table=macro16_source128_point_table(),
        frame_stride=1,
        max_frames_per_compact=0,
        scale_mode="normalized",
        b_label_coordinate="auto",
    )
    train_idx, val_idx, split_meta = split_indices(
        data.case_id,
        val_fraction=float(args.val_fraction),
        val_cases=str(args.val_cases),
        seed=int(args.seed),
    )
    point_mean, point_std = stats(data.point_features_hat[train_idx].reshape(-1, data.point_features_hat.shape[-1]), axis=0)
    branch_raw = np.concatenate(
        [data.q48_hat, data.x16_hat.reshape(data.x16_hat.shape[0], -1), data.length_scale],
        axis=1,
    )
    branch_mean, branch_std = stats(branch_raw[train_idx], axis=0)
    _le_mean, le_std = stats(data.le[train_idx], axis=0)
    q_std = branch_std.reshape(-1)[:48]
    le_std_scale = _le_std_scale(le_std, data.le.shape[1])
    j_target = data.b * q_std.reshape(1, 1, 1, 48) / le_std_scale
    global_b_norm = np.mean(j_target[train_idx], axis=(0, 1)).astype(np.float32)
    b_scale = build_physical_b_loss_scale(data.b[train_idx])
    point_norm = standardize(data.point_features_hat, point_mean.reshape(1, 1, -1), point_std.reshape(1, 1, -1))
    model = PointBPrior(point_dim=point_norm.shape[-1], hidden_dim=int(args.hidden_dim), depth=int(args.depth)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    p_train = torch.as_tensor(point_norm[train_idx], dtype=torch.float32, device=device)
    b_train = torch.as_tensor(data.b[train_idx], dtype=torch.float32, device=device)
    scale = torch.as_tensor(b_scale, dtype=torch.float32, device=device)
    global_j = torch.as_tensor(global_b_norm, dtype=torch.float32, device=device).view(1, 1, 6, 48)
    q_std_t = torch.as_tensor(q_std.astype(np.float32), dtype=torch.float32, device=device).view(1, 1, 1, 48)
    le_std_t = torch.as_tensor(le_std_scale.astype(np.float32), dtype=torch.float32, device=device)
    history = []
    best: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred_j = global_j + float(args.baseline_scale) * model(p_train)
        pred_b = pred_j * le_std_t / torch.clamp(q_std_t, min=1.0e-12)
        loss = (((pred_b - b_train) / scale) ** 2).mean(dim=(0, 1)).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite B prior loss")
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % int(args.eval_every) == 0 or epoch == int(args.epochs):
            model.eval()
            with torch.no_grad():
                delta_all = model(torch.as_tensor(point_norm, dtype=torch.float32, device=device)).cpu().numpy()
            pred_j_all = global_b_norm.reshape(1, 1, 6, 48) + float(args.baseline_scale) * delta_all
            pred_b_all = pred_j_all * le_std_scale / q_std.reshape(1, 1, 1, 48)
            row = {
                "epoch": int(epoch),
                "loss": float(loss.detach().cpu()),
                "train_B_rel": rel_np(pred_b_all[train_idx], data.b[train_idx]),
                "val_B_rel": rel_np(pred_b_all[val_idx], data.b[val_idx]),
                "train_J_rel": rel_np(
                    pred_j_all[train_idx],
                    j_target[train_idx],
                ),
                "val_J_rel": rel_np(
                    pred_j_all[val_idx],
                    j_target[val_idx],
                ),
            }
            history.append(row)
            if float(row["train_B_rel"]) < best_score:
                best_score = float(row["train_B_rel"])
                best = dict(row)
                best_state = _cpu_state_dict(model.net)
    checkpoint_path = str(getattr(args, "checkpoint_out", "")).strip()
    if checkpoint_path:
        if best_state is None:
            best_state = _cpu_state_dict(model.net)
        checkpoint = {
            "b_prior_checkpoint_version": "macro16-point-b-prior-v1",
            "b_prior_kind": "global_plus_point",
            "target_coordinate": "J_norm",
            "global_b_norm": torch.as_tensor(global_b_norm, dtype=torch.float32),
            "point_b_net_state": best_state,
            "baseline_scale": float(args.baseline_scale),
            "model": {
                "type": "Macro16 point_b_net compatible MLP",
                "point_dim": int(point_norm.shape[-1]),
                "hidden_dim": int(args.hidden_dim),
                "depth": int(args.depth),
                "activation": "tanh",
                "strain_dim": 6,
                "q_dim": 48,
            },
            "norms": {
                "point_mean": torch.as_tensor(point_mean.astype(np.float32)),
                "point_std": torch.as_tensor(point_std.astype(np.float32)),
                "branch_mean": torch.as_tensor(branch_mean.astype(np.float32)),
                "branch_std": torch.as_tensor(branch_std.astype(np.float32)),
                "le_std": torch.as_tensor(le_std.astype(np.float32)),
                "q_std": torch.as_tensor(q_std.astype(np.float32)),
            },
            "split": {
                **split_meta,
                "train_case_ids": sorted(np.unique(data.case_id[train_idx]).astype(int).tolist()),
                "val_case_ids_excluded": sorted(np.unique(data.case_id[val_idx]).astype(int).tolist()),
                "train_frame_count": int(train_idx.size),
                "val_frame_count": int(val_idx.size),
            },
            "compact_paths": paths,
            "best": best,
        }
        out_path = Path(checkpoint_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, out_path)
    return {
        "status": "done",
        "device": str(device),
        "compact_paths": paths,
        "split": {
            **split_meta,
            "train_frame_count": int(train_idx.size),
            "val_frame_count": int(val_idx.size),
            "train_case_ids": sorted(np.unique(data.case_id[train_idx]).astype(int).tolist()),
            "val_case_ids": sorted(np.unique(data.case_id[val_idx]).astype(int).tolist()),
        },
        "model": {
            "type": "diagnostic PointBPrior",
            "target_coordinate": "J_norm",
            "b_prior_kind": "global_plus_point",
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "baseline_scale": float(args.baseline_scale),
        },
        "checkpoint_out": checkpoint_path,
        "best": best,
        "latest": history[-1] if history else None,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-list", default="")
    parser.add_argument("--compact", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-cases", default="")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--baseline-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint-out", default="")
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if args.compact_list:
        _ = read_path_list(Path(args.compact_list))
    payload = run(args)
    write_json(Path(args.out), payload)
    print(json.dumps({"best": payload["best"], "latest": payload["latest"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
