#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate 17 Macro16 state-dependent B baseline prototype.

This script is intentionally outside the production Macro16 trainer.  It tests
whether a q-state-conditioned B baseline can close the tiny case031 LE/B/force
diagnostic without changing q48 order, LE order, or the 128-point rule.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from pathlib import Path
import random
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
sys.path.insert(0, str(ROOT / "scripts"))

from audit_macro16_force_stiffness import metric  # noqa: E402
from audit_macro16_trained_force_closure import qdef_projected_rf, torch_load_cross_platform  # noqa: E402
from macro_deeponet.macro16_geometry import MACRO16_CONTRACT_VERSION, macro16_source128_point_table  # noqa: E402
from macro_deeponet.models import MLP  # noqa: E402
from macro_deeponet.train_macro16_boundary_sobolev import (  # noqa: E402
    compact_paths_from_args,
    load_macro16_compacts,
    parse_int_list,
    read_path_list,
    split_indices,
    standardize,
)
from macro_deeponet.train_true176_deeponet_sobolev import cos_np, rel_np  # noqa: E402
from macro_deeponet.true176_data import stats  # noqa: E402


class _CrossPlatformPosixPath(pathlib.PurePosixPath):
    pass


class _CrossPlatformWindowsPath(pathlib.PureWindowsPath):
    pass


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[str(key)] = str(value) if isinstance(value, Path) else value
    return out


def rms_np(arr: np.ndarray) -> float:
    vals = np.asarray(arr, dtype=np.float64)
    return float(np.sqrt(np.mean(vals * vals))) if vals.size else 0.0


def scale_from_train(values: np.ndarray, axes: tuple[int, ...], *, floor_rel: float, floor_abs: float) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    rms = np.sqrt(np.mean(vals * vals, axis=axes, keepdims=True))
    global_rms = float(np.sqrt(np.mean(vals * vals))) if vals.size else 0.0
    floor = max(float(floor_abs), float(floor_rel) * global_rms, 1.0e-12)
    return np.maximum(rms, floor).astype(np.float32)


class Macro16StateBPrototype(nn.Module):
    """Fixed-128 prototype with a q-state-conditioned B baseline."""

    def __init__(
        self,
        *,
        q_dim: int,
        geom_dim: int,
        point_dim: int,
        point_count: int,
        hidden_dim: int,
        depth: int,
        state_rank: int,
        state_scale: float,
        point_delta_scale: float,
        q_mean: np.ndarray,
        q_std: np.ndarray,
        b_init: np.ndarray,
        le0_init: np.ndarray,
        detach_state: bool,
    ) -> None:
        super().__init__()
        if int(q_dim) != 48:
            raise ValueError("Macro16 prototype requires q_dim=48")
        if int(state_rank) < 1:
            raise ValueError("state_rank must be positive")
        self.q_dim = int(q_dim)
        self.geom_dim = int(geom_dim)
        self.point_dim = int(point_dim)
        self.point_count = int(point_count)
        self.strain_dim = 6
        self.state_rank = int(state_rank)
        self.state_scale = float(state_scale)
        self.point_delta_scale = float(point_delta_scale)
        self.detach_state = bool(detach_state)
        self.register_buffer("q_mean", torch.as_tensor(q_mean, dtype=torch.float32).reshape(self.q_dim))
        self.register_buffer("q_std", torch.clamp(torch.as_tensor(q_std, dtype=torch.float32).reshape(self.q_dim), min=1.0e-12))
        self.static_b = nn.Parameter(torch.as_tensor(b_init, dtype=torch.float32).reshape(self.point_count, 6, self.q_dim))
        self.static_le0 = nn.Parameter(torch.as_tensor(le0_init, dtype=torch.float32).reshape(self.point_count, 6))
        self.point_b_net = MLP(
            self.point_dim,
            6 * self.q_dim,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            activation=nn.Tanh,
            zero_last=True,
        )
        self.point_le0_net = MLP(
            self.point_dim,
            6,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            activation=nn.Tanh,
            zero_last=True,
        )
        self.point_state_net = MLP(
            self.point_dim,
            6 * self.q_dim * self.state_rank,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            activation=nn.Tanh,
            zero_last=False,
        )
        self.state_coeff_net = MLP(
            self.q_dim + self.geom_dim,
            self.state_rank,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            activation=nn.Tanh,
            zero_last=True,
        )

    def q_hat(self, q_norm: torch.Tensor) -> torch.Tensor:
        return q_norm * self.q_std.view(1, -1) + self.q_mean.view(1, -1)

    def state_input(self, q_norm: torch.Tensor, geom_norm: torch.Tensor) -> torch.Tensor:
        q_state = q_norm.detach() if self.detach_state else q_norm
        geom_state = geom_norm.detach() if self.detach_state else geom_norm
        return torch.cat([q_state, geom_state], dim=-1)

    def le0(self, point_norm: torch.Tensor) -> torch.Tensor:
        batch, point_count, _ = point_norm.shape
        delta = self.point_le0_net(point_norm.reshape(-1, self.point_dim)).reshape(batch, point_count, 6)
        return self.static_le0.view(1, self.point_count, 6) + delta

    def b_state(self, q_norm: torch.Tensor, geom_norm: torch.Tensor, point_norm: torch.Tensor) -> torch.Tensor:
        batch, point_count, _ = point_norm.shape
        point_delta = self.point_b_net(point_norm.reshape(-1, self.point_dim)).reshape(batch, point_count, 6, self.q_dim)
        point_b = self.static_b.view(1, self.point_count, 6, self.q_dim) + self.point_delta_scale * point_delta
        coeff = self.state_coeff_net(self.state_input(q_norm, geom_norm)).reshape(batch, self.state_rank)
        basis = self.point_state_net(point_norm.reshape(-1, self.point_dim)).reshape(
            batch,
            point_count,
            6,
            self.q_dim,
            self.state_rank,
        )
        state_delta = torch.einsum("bpakr,br->bpak", basis, coeff)
        return point_b + self.state_scale * state_delta

    def forward(self, q_norm: torch.Tensor, geom_norm: torch.Tensor, point_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = self.b_state(q_norm, geom_norm, point_norm)
        q_hat = self.q_hat(q_norm)
        le = self.le0(point_norm) + torch.einsum("bpak,bk->bpa", b, q_hat)
        return le, b


def prepare_arrays(
    compact_paths: list[str],
    *,
    val_cases: str,
    val_fraction: float,
    seed: int,
) -> dict[str, Any]:
    data = load_macro16_compacts(
        compact_paths,
        point_table=macro16_source128_point_table(),
        frame_stride=1,
        max_frames_per_compact=0,
        scale_mode="normalized",
        b_label_coordinate="auto",
    )
    train_idx, val_idx, split_meta = split_indices(
        data.case_id,
        val_fraction=float(val_fraction),
        val_cases=str(val_cases),
        seed=int(seed),
    )
    geom_raw = np.concatenate([data.x16_hat.reshape(data.x16_hat.shape[0], -1), data.length_scale], axis=1).astype(np.float32)
    q_mean, q_std = stats(data.q48_hat[train_idx], axis=0)
    geom_mean, geom_std = stats(geom_raw[train_idx], axis=0)
    point_mean, point_std = stats(data.point_features_hat[train_idx].reshape(-1, data.point_features_hat.shape[-1]), axis=0)
    q_norm = standardize(data.q48_hat, q_mean, q_std)
    geom_norm = standardize(geom_raw, geom_mean, geom_std)
    point_norm = standardize(data.point_features_hat, point_mean.reshape(1, 1, -1), point_std.reshape(1, 1, -1))
    b_init = np.mean(data.b[train_idx], axis=0).astype(np.float32)
    le0_star = data.le - np.einsum("npak,nk->npa", data.b, data.q48_hat)
    le0_init = np.mean(le0_star[train_idx], axis=0).astype(np.float32)
    le_scale = scale_from_train(data.le[train_idx], (0, 1), floor_rel=2.0e-2, floor_abs=1.0e-10).reshape(1, 1, 6)
    b_scale = scale_from_train(data.b[train_idx], (0, 1), floor_rel=2.0e-2, floor_abs=1.0e-8).reshape(1, 1, 6, 48)
    return {
        "data": data,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "split_meta": split_meta,
        "q_norm": q_norm.astype(np.float32),
        "geom_norm": geom_norm.astype(np.float32),
        "point_norm": point_norm.astype(np.float32),
        "geom_raw": geom_raw,
        "q_mean": q_mean.astype(np.float32),
        "q_std": q_std.astype(np.float32),
        "geom_mean": geom_mean.astype(np.float32),
        "geom_std": geom_std.astype(np.float32),
        "point_mean": point_mean.astype(np.float32),
        "point_std": point_std.astype(np.float32),
        "b_init": b_init,
        "le0_init": le0_init,
        "le_scale": le_scale.astype(np.float32),
        "b_scale": b_scale.astype(np.float32),
    }


def build_model(arrays: dict[str, Any], args: argparse.Namespace, device: torch.device) -> Macro16StateBPrototype:
    data = arrays["data"]
    return Macro16StateBPrototype(
        q_dim=48,
        geom_dim=int(arrays["geom_norm"].shape[-1]),
        point_dim=int(arrays["point_norm"].shape[-1]),
        point_count=int(data.le.shape[1]),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        state_rank=int(args.state_rank),
        state_scale=float(args.state_scale),
        point_delta_scale=float(args.point_delta_scale),
        q_mean=arrays["q_mean"],
        q_std=arrays["q_std"],
        b_init=arrays["b_init"],
        le0_init=arrays["le0_init"],
        detach_state=not bool(args.no_detach_state),
    ).to(device)


def evaluate_model(
    model: Macro16StateBPrototype,
    arrays: dict[str, Any],
    indices: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, float]:
    data = arrays["data"]
    idx = np.asarray(indices, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        le_pred, b_pred = model(
            torch.as_tensor(arrays["q_norm"][idx], dtype=torch.float32, device=device),
            torch.as_tensor(arrays["geom_norm"][idx], dtype=torch.float32, device=device),
            torch.as_tensor(arrays["point_norm"][idx], dtype=torch.float32, device=device),
        )
    le_np = le_pred.detach().cpu().numpy().astype(np.float64)
    b_np = b_pred.detach().cpu().numpy().astype(np.float64)
    le_true = np.asarray(data.le[idx], dtype=np.float64)
    b_true = np.asarray(data.b[idx], dtype=np.float64)
    return {
        "frames": int(idx.size),
        "LE_rel": rel_np(le_np, le_true),
        "LE_cos": cos_np(le_np, le_true),
        "B_rel": rel_np(b_np, b_true),
        "B_cos": cos_np(b_np, b_true),
        "LE_rms": rms_np(le_np - le_true),
        "B_rms": rms_np(b_np - b_true),
    }


def checkpoint_payload(
    *,
    model: Macro16StateBPrototype,
    arrays: dict[str, Any],
    args: argparse.Namespace,
    compact_paths: list[str],
    best: dict[str, Any],
    latest: dict[str, Any],
) -> dict[str, Any]:
    data = arrays["data"]
    return {
        "checkpoint_version": "macro16-state-b-prototype-v1",
        "contract_version": MACRO16_CONTRACT_VERSION,
        "model_kind": "state_dependent_b_baseline_prototype",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": {
            "q_dim": 48,
            "geom_dim": int(arrays["geom_norm"].shape[-1]),
            "point_dim": int(arrays["point_norm"].shape[-1]),
            "point_count": int(data.le.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "state_rank": int(args.state_rank),
            "state_scale": float(args.state_scale),
            "point_delta_scale": float(args.point_delta_scale),
            "detach_state": not bool(args.no_detach_state),
        },
        "norms": {
            "q_mean": arrays["q_mean"],
            "q_std": arrays["q_std"],
            "geom_mean": arrays["geom_mean"],
            "geom_std": arrays["geom_std"],
            "point_mean": arrays["point_mean"],
            "point_std": arrays["point_std"],
        },
        "split": {
            **arrays["split_meta"],
            "train_case_ids": sorted(np.unique(data.case_id[arrays["train_idx"]]).astype(int).tolist()),
            "val_case_ids": sorted(np.unique(data.case_id[arrays["val_idx"]]).astype(int).tolist()),
            "train_frame_count": int(arrays["train_idx"].size),
            "val_frame_count": int(arrays["val_idx"].size),
        },
        "compact_paths": compact_paths,
        "args": jsonable_args(args),
        "best": best,
        "latest": latest,
        "invariants": {
            "q48_order_changed": False,
            "LE_order_changed": False,
            "point_rule": "macro16_source128",
            "production_model_changed": False,
        },
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device("cuda" if bool(args.cuda) and torch.cuda.is_available() else "cpu")
    compact_paths = compact_paths_from_args(args)
    arrays = prepare_arrays(
        compact_paths,
        val_cases=str(args.val_cases),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )
    model = build_model(arrays, args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(args.lr_decay))
    train_idx = arrays["train_idx"]
    val_idx = arrays["val_idx"]
    all_idx = np.arange(arrays["data"].q48_hat.shape[0], dtype=np.int64)
    le_scale = torch.as_tensor(arrays["le_scale"], dtype=torch.float32, device=device)
    b_scale = torch.as_tensor(arrays["b_scale"], dtype=torch.float32, device=device)
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_row: dict[str, Any] = {}
    best_state: dict[str, torch.Tensor] | None = None
    q_train = torch.as_tensor(arrays["q_norm"][train_idx], dtype=torch.float32, device=device)
    geom_train = torch.as_tensor(arrays["geom_norm"][train_idx], dtype=torch.float32, device=device)
    point_train = torch.as_tensor(arrays["point_norm"][train_idx], dtype=torch.float32, device=device)
    le_train = torch.as_tensor(arrays["data"].le[train_idx], dtype=torch.float32, device=device)
    b_train = torch.as_tensor(arrays["data"].b[train_idx], dtype=torch.float32, device=device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, int(args.steps) + 1):
        model.train()
        le_pred, b_pred = model(q_train, geom_train, point_train)
        le_loss = torch.mean(((le_pred - le_train) / le_scale) ** 2)
        b_loss = torch.mean(((b_pred - b_train) / b_scale) ** 2)
        loss = float(args.le_weight) * le_loss + float(args.b_weight) * b_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        scheduler.step()
        if step == 1 or step % int(args.eval_every) == 0 or step == int(args.steps):
            row = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "le_loss": float(le_loss.detach().cpu()),
                "b_loss": float(b_loss.detach().cpu()),
                "lr": float(scheduler.get_last_lr()[0]),
                "train": evaluate_model(model, arrays, train_idx, device=device),
                "val": evaluate_model(model, arrays, val_idx, device=device),
                "all": evaluate_model(model, arrays, all_idx, device=device),
            }
            row["score"] = float(row["val"]["LE_rel"] + row["val"]["B_rel"])
            history.append(row)
            if float(row["score"]) < best_score:
                best_score = float(row["score"])
                best_row = dict(row)
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(json.dumps(row, ensure_ascii=False, sort_keys=True, default=json_default), flush=True)

    latest = history[-1]
    if best_state is not None:
        model.load_state_dict(best_state)
    payload = checkpoint_payload(
        model=model,
        arrays=arrays,
        args=args,
        compact_paths=compact_paths,
        best=best_row,
        latest=latest,
    )
    torch.save(payload, out_dir / "best.pt")
    write_json(out_dir / "training_summary.json", {k: v for k, v in payload.items() if k != "model_state"})
    write_json(
        out_dir / "history.json",
        {
            "history": history,
            "best": best_row,
            "latest": latest,
        },
    )
    return {k: v for k, v in payload.items() if k != "model_state"}


def load_prototype_checkpoint(path: Path, device: torch.device) -> tuple[Macro16StateBPrototype, dict[str, Any]]:
    checkpoint = torch_load_cross_platform(path, device)
    if str(checkpoint.get("checkpoint_version", "")) != "macro16-state-b-prototype-v1":
        raise ValueError(f"{path}: unsupported checkpoint_version {checkpoint.get('checkpoint_version')}")
    cfg = checkpoint["model_config"]
    norms = checkpoint["norms"]
    state = checkpoint["model_state"]
    static_b = state["static_b"].detach().cpu().numpy()
    static_le0 = state["static_le0"].detach().cpu().numpy()
    model = Macro16StateBPrototype(
        q_dim=int(cfg["q_dim"]),
        geom_dim=int(cfg["geom_dim"]),
        point_dim=int(cfg["point_dim"]),
        point_count=int(cfg["point_count"]),
        hidden_dim=int(cfg["hidden_dim"]),
        depth=int(cfg["depth"]),
        state_rank=int(cfg["state_rank"]),
        state_scale=float(cfg["state_scale"]),
        point_delta_scale=float(cfg["point_delta_scale"]),
        q_mean=np.asarray(norms["q_mean"], dtype=np.float32),
        q_std=np.asarray(norms["q_std"], dtype=np.float32),
        b_init=static_b,
        le0_init=static_le0,
        detach_state=bool(cfg["detach_state"]),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def force_audit(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if bool(args.cuda) and torch.cuda.is_available() else "cpu")
    model, checkpoint = load_prototype_checkpoint(Path(args.checkpoint).resolve(), device)
    compact_paths = read_path_list(Path(args.compact_list))
    arrays = prepare_arrays(
        compact_paths,
        val_cases=str(args.val_cases),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )
    data = arrays["data"]
    norms = checkpoint["norms"]
    geom_raw = np.concatenate([data.x16_hat.reshape(data.x16_hat.shape[0], -1), data.length_scale], axis=1).astype(np.float32)
    q_norm = standardize(data.q48_hat, np.asarray(norms["q_mean"], dtype=np.float32), np.asarray(norms["q_std"], dtype=np.float32))
    geom_norm = standardize(geom_raw, np.asarray(norms["geom_mean"], dtype=np.float32), np.asarray(norms["geom_std"], dtype=np.float32))
    point_norm = standardize(
        data.point_features_hat,
        np.asarray(norms["point_mean"], dtype=np.float32).reshape(1, 1, -1),
        np.asarray(norms["point_std"], dtype=np.float32).reshape(1, 1, -1),
    )
    indices = np.arange(data.q48_hat.shape[0], dtype=np.int64)
    case_ids = set(parse_int_list(str(args.case_list))) if str(args.case_list).strip() else set()
    if case_ids:
        indices = indices[np.isin(data.case_id[indices], np.asarray(sorted(case_ids), dtype=np.int64))]
    if int(args.max_frames) > 0:
        indices = indices[: int(args.max_frames)]
    if indices.size == 0:
        raise ValueError("no frames selected for prototype force audit")
    le_rows: list[np.ndarray] = []
    b_rows: list[np.ndarray] = []
    for start in range(0, indices.size, int(args.batch_size)):
        sub = indices[start : start + int(args.batch_size)]
        with torch.no_grad():
            le_pred, b_pred = model(
                torch.as_tensor(q_norm[sub], dtype=torch.float32, device=device),
                torch.as_tensor(geom_norm[sub], dtype=torch.float32, device=device),
                torch.as_tensor(point_norm[sub], dtype=torch.float32, device=device),
            )
        le_rows.append(le_pred.detach().cpu().numpy().astype(np.float64))
        b_rows.append(b_pred.detach().cpu().numpy().astype(np.float64))
    le_pred_np = np.concatenate(le_rows, axis=0)
    b_pred_hat = np.concatenate(b_rows, axis=0)
    l_ref = np.asarray(data.length_scale[indices], dtype=np.float64).reshape(-1, 1, 1, 1)
    b_pred_raw = b_pred_hat / np.maximum(l_ref, 1.0e-12)
    b_true_raw = np.asarray(data.b[indices], dtype=np.float64) / np.maximum(l_ref, 1.0e-12)
    rf_qdef, weights_selected, elastic_d = qdef_projected_rf(
        data.compact_paths,
        data.source_index[indices],
        data.source_row[indices],
    )
    stress_pred = np.einsum("nab,npb->npa", elastic_d, le_pred_np)
    force_pred = np.einsum("npaj,npa,np->nj", b_pred_raw, stress_pred, weights_selected)
    stress_true = np.einsum("nab,npb->npa", elastic_d, np.asarray(data.le[indices], dtype=np.float64))
    force_teacher = np.einsum("npaj,npa,np->nj", b_true_raw, stress_true, weights_selected)
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "case_ids": sorted(np.unique(data.case_id[indices]).astype(int).tolist()),
        "frame_count": int(indices.size),
        "point_count": int(data.le.shape[1]),
        "device": str(device),
        "LE_rel": rel_np(le_pred_np, data.le[indices]),
        "B_qdef_hat_rel": rel_np(b_pred_hat, data.b[indices]),
        "selected_frame_force_qdef": metric(force_pred, rf_qdef),
        "teacher_selected_frame_force_qdef": metric(force_teacher, rf_qdef),
        "volume_mode": "selected-frame",
        "force_coordinate": "qdef coordinate, compared against rigid_projection_P^T RF_projected_macro",
        "prototype": {
            "kind": "state_dependent_b_baseline",
            "production_model_changed": False,
            "q48_order_changed": False,
            "LE_order_changed": False,
            "point_rule": "macro16_source128",
        },
    }
    write_json(Path(args.out), payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=json_default))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "force"], default="train")
    parser.add_argument("--compact", action="append", default=[])
    parser.add_argument("--compact-list", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--val-cases", default="")
    parser.add_argument("--case-list", default="")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--state-rank", type=int, default=8)
    parser.add_argument("--state-scale", type=float, default=1.0)
    parser.add_argument("--point-delta-scale", type=float, default=1.0)
    parser.add_argument("--le-weight", type=float, default=1.0)
    parser.add_argument("--b-weight", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--lr-decay", type=float, default=0.9995)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--no-detach-state", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if str(args.mode) == "train":
        if not str(args.out_dir).strip():
            raise SystemExit("--out-dir is required for --mode train")
        if int(args.steps) < 1:
            raise SystemExit("--steps must be positive")
        if int(args.eval_every) < 1:
            raise SystemExit("--eval-every must be positive")
    if str(args.mode) == "force":
        if not str(args.checkpoint).strip():
            raise SystemExit("--checkpoint is required for --mode force")
        if not str(args.out).strip():
            raise SystemExit("--out is required for --mode force")
    if int(args.state_rank) < 1:
        raise SystemExit("--state-rank must be positive")
    if float(args.state_scale) < 0.0:
        raise SystemExit("--state-scale must be non-negative")
    if float(args.point_delta_scale) < 0.0:
        raise SystemExit("--point-delta-scale must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    if str(args.mode) == "train":
        summary = train(args)
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))
    else:
        force_audit(args)


if __name__ == "__main__":
    main()
