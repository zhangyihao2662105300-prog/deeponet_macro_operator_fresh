"""Tiny overfit trainer for the Macro16 coordinate-DeepONet experiment.

This is not the production Macro16 trainer.  It is a first-stage route check:

    Branch: q48_def_hat[48] + geometry_g[14]
    Trunk:  x_gp_hat[3]
    Output: LE[6]
    B:      AD dLE/dq48_def_hat, supervised by B_macro_qdef
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from coordinate_deeponet_model import Macro16CoordinateDeepONet, Macro16CoordinateLinearResidualDeepONet
    from prepare_coordinate_dataset import load_coordinate_arrays, paths_from_args
else:
    from .coordinate_deeponet_model import Macro16CoordinateDeepONet, Macro16CoordinateLinearResidualDeepONet
    from .prepare_coordinate_dataset import load_coordinate_arrays, paths_from_args


class CoordinateDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray) -> None:
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        return (
            torch.as_tensor(idx, dtype=torch.long),
            torch.from_numpy(self.arrays["q"][idx]),
            torch.from_numpy(self.arrays["g"][idx]),
            torch.from_numpy(self.arrays["x"][idx]),
            torch.from_numpy(self.arrays["le"][idx]),
            torch.from_numpy(self.arrays["b"][idx]),
        )


def stats(x: np.ndarray, *, axis: int | tuple[int, ...], floor: float) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(x, dtype=np.float64)
    mean = np.mean(vals, axis=axis, keepdims=False)
    std = np.std(vals, axis=axis, keepdims=False)
    std = np.maximum(std, float(floor))
    return mean.astype(np.float32), std.astype(np.float32)


def rel_error(pred: np.ndarray, ref: np.ndarray, eps: float = 1.0e-12) -> float:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    den = max(float(np.linalg.norm(np.asarray(ref, dtype=np.float64).reshape(-1))), eps)
    return float(np.linalg.norm(diff.reshape(-1)) / den)


def parse_int_list(text: str, default: list[int]) -> list[int]:
    raw = str(text).strip().lower()
    if not raw or raw == "default":
        return list(default)
    if raw == "all":
        return list(range(48))
    return [int(v) for v in raw.replace(";", ",").split(",") if v.strip()]


def split_indices(case_id: np.ndarray, *, val_cases: str, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    all_idx = np.arange(case_id.shape[0], dtype=np.int64)
    text = str(val_cases).strip()
    if text:
        wanted = {int(v) for v in text.replace(";", ",").split(",") if v.strip()}
        mask = np.isin(case_id, np.asarray(sorted(wanted), dtype=np.int64))
        return all_idx[~mask], all_idx[mask], {"mode": "case", "val_cases": sorted(wanted)}
    if float(val_fraction) <= 0.0:
        return all_idx, np.empty((0,), dtype=np.int64), {"mode": "none"}
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(all_idx)
    n_val = max(1, int(round(float(val_fraction) * all_idx.size)))
    val = np.sort(perm[:n_val])
    train = np.sort(perm[n_val:])
    return train, val, {"mode": "fraction", "val_fraction": float(val_fraction)}


def make_branch_raw(q: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return torch.cat([q, g], dim=-1)


def predict_le_norm(
    model: Macro16CoordinateDeepONet,
    q: torch.Tensor,
    g: torch.Tensor,
    x: torch.Tensor,
    *,
    branch_mean: torch.Tensor,
    branch_std: torch.Tensor,
) -> torch.Tensor:
    branch = (make_branch_raw(q, g) - branch_mean) / branch_std
    return model(branch, x)


def predict_le_phys(
    model: Macro16CoordinateDeepONet,
    q: torch.Tensor,
    g: torch.Tensor,
    x: torch.Tensor,
    *,
    branch_mean: torch.Tensor,
    branch_std: torch.Tensor,
    le_mean: torch.Tensor,
    le_std: torch.Tensor,
) -> torch.Tensor:
    le_norm = predict_le_norm(model, q, g, x, branch_mean=branch_mean, branch_std=branch_std)
    return le_norm * le_std + le_mean


def b_jvp_columns(
    model: Macro16CoordinateDeepONet,
    q: torch.Tensor,
    g: torch.Tensor,
    x: torch.Tensor,
    columns: list[int],
    *,
    branch_mean: torch.Tensor,
    branch_std: torch.Tensor,
    le_mean: torch.Tensor,
    le_std: torch.Tensor,
) -> torch.Tensor:
    def fn(q_in: torch.Tensor) -> torch.Tensor:
        return predict_le_phys(
            model,
            q_in,
            g,
            x,
            branch_mean=branch_mean,
            branch_std=branch_std,
            le_mean=le_mean,
            le_std=le_std,
        )

    outs: list[torch.Tensor] = []
    for col in columns:
        tangent = torch.zeros_like(q)
        tangent[:, int(col)] = 1.0
        _base, deriv = torch.func.jvp(fn, (q,), (tangent,))
        outs.append(deriv)
    return torch.stack(outs, dim=-1)


def b_forward_diff_columns(
    model: Macro16CoordinateDeepONet,
    q: torch.Tensor,
    g: torch.Tensor,
    x: torch.Tensor,
    columns: list[int],
    *,
    branch_mean: torch.Tensor,
    branch_std: torch.Tensor,
    le_mean: torch.Tensor,
    le_std: torch.Tensor,
    fd_step: float,
    column_chunk: int,
) -> torch.Tensor:
    if not columns:
        raise ValueError("columns must not be empty")
    h = float(fd_step)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("fd_step must be positive")
    base = predict_le_phys(
        model,
        q,
        g,
        x,
        branch_mean=branch_mean,
        branch_std=branch_std,
        le_mean=le_mean,
        le_std=le_std,
    )
    preds: list[torch.Tensor] = []
    chunk = max(1, int(column_chunk))
    batch, point_count, _ = x.shape
    for start in range(0, len(columns), chunk):
        cur = columns[start : start + chunk]
        q_plus = q[:, None, :].expand(batch, len(cur), q.shape[-1]).clone()
        for local_idx, col in enumerate(cur):
            q_plus[:, local_idx, int(col)] = q_plus[:, local_idx, int(col)] + h
        flat_q = q_plus.reshape(batch * len(cur), q.shape[-1])
        flat_g = g[:, None, :].expand(batch, len(cur), g.shape[-1]).reshape(batch * len(cur), g.shape[-1])
        flat_x = x[:, None, :, :].expand(batch, len(cur), point_count, 3).reshape(batch * len(cur), point_count, 3)
        plus = predict_le_phys(
            model,
            flat_q,
            flat_g,
            flat_x,
            branch_mean=branch_mean,
            branch_std=branch_std,
            le_mean=le_mean,
            le_std=le_std,
        ).view(batch, len(cur), point_count, 6)
        slope = (plus - base[:, None, :, :]) / h
        preds.append(slope.permute(0, 2, 3, 1).contiguous())
    return torch.cat(preds, dim=-1)


def balanced_b_loss(pred: torch.Tensor, target: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.mean(((pred - target) / scale) ** 2)


def source_plus_fd_loss(
    model: Macro16CoordinateDeepONet,
    q: torch.Tensor,
    g: torch.Tensor,
    x: torch.Tensor,
    le_true: torch.Tensor,
    q_plus: torch.Tensor,
    le_plus_true: torch.Tensor,
    columns: list[int],
    *,
    branch_mean: torch.Tensor,
    branch_std: torch.Tensor,
    le_mean: torch.Tensor,
    le_std: torch.Tensor,
    plus_delta_raw: float,
    plus_slope_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not columns:
        raise ValueError("columns must not be empty")
    delta = float(plus_delta_raw)
    if not math.isfinite(delta) or delta == 0.0:
        raise ValueError("plus_delta_raw must be finite and nonzero")
    batch, point_count, _ = x.shape
    base = predict_le_phys(
        model,
        q,
        g,
        x,
        branch_mean=branch_mean,
        branch_std=branch_std,
        le_mean=le_mean,
        le_std=le_std,
    )
    qp = q_plus[:, columns, :].reshape(batch * len(columns), q.shape[-1])
    gp = g[:, None, :].expand(batch, len(columns), g.shape[-1]).reshape(batch * len(columns), g.shape[-1])
    xp = x[:, None, :, :].expand(batch, len(columns), point_count, 3).reshape(batch * len(columns), point_count, 3)
    plus = predict_le_phys(
        model,
        qp,
        gp,
        xp,
        branch_mean=branch_mean,
        branch_std=branch_std,
        le_mean=le_mean,
        le_std=le_std,
    ).view(batch, len(columns), point_count, 6)
    pred_slope = ((plus - base[:, None, :, :]) / delta).permute(0, 2, 3, 1).contiguous()
    target_slope = ((le_plus_true[:, columns, :, :] - le_true[:, None, :, :]) / delta).permute(0, 2, 3, 1).contiguous()
    scale = plus_slope_scale[:, :, :, columns]
    return balanced_b_loss(pred_slope, target_slope, scale), pred_slope, target_slope


def evaluate(
    model: Macro16CoordinateDeepONet,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    device: torch.device,
    branch_mean: torch.Tensor,
    branch_std: torch.Tensor,
    le_mean: torch.Tensor,
    le_std: torch.Tensor,
    b_scale: torch.Tensor,
    eval_columns: list[int],
    fd_step: float,
    fd_column_chunk: int,
    prefix: str,
) -> dict[str, float]:
    if indices.size == 0:
        return {}
    model.eval()
    q = torch.as_tensor(arrays["q"][indices], dtype=torch.float32, device=device)
    g = torch.as_tensor(arrays["g"][indices], dtype=torch.float32, device=device)
    x = torch.as_tensor(arrays["x"][indices], dtype=torch.float32, device=device)
    le_true = torch.as_tensor(arrays["le"][indices], dtype=torch.float32, device=device)
    b_true = torch.as_tensor(arrays["b"][indices][:, :, :, eval_columns], dtype=torch.float32, device=device)
    with torch.no_grad():
        le_pred = predict_le_phys(
            model,
            q,
            g,
            x,
            branch_mean=branch_mean,
            branch_std=branch_std,
            le_mean=le_mean,
            le_std=le_std,
        )
    with torch.enable_grad():
        q_req = q.detach().requires_grad_(True)
        b_pred = b_jvp_columns(
            model,
            q_req,
            g,
            x,
            eval_columns,
            branch_mean=branch_mean,
            branch_std=branch_std,
            le_mean=le_mean,
            le_std=le_std,
        )
        b_fd_pred = b_forward_diff_columns(
            model,
            q,
            g,
            x,
            eval_columns,
            branch_mean=branch_mean,
            branch_std=branch_std,
            le_mean=le_mean,
            le_std=le_std,
            fd_step=fd_step,
            column_chunk=fd_column_chunk,
        )
    le_np = le_pred.detach().cpu().numpy()
    le_ref = le_true.detach().cpu().numpy()
    b_np = b_pred.detach().cpu().numpy()
    b_fd_np = b_fd_pred.detach().cpu().numpy()
    b_ref = b_true.detach().cpu().numpy()
    scale = b_scale[:, :, :, eval_columns]
    b_loss = balanced_b_loss(b_pred, b_true, scale).detach().cpu().item()
    return {
        f"{prefix}_LE_rel": rel_error(le_np, le_ref),
        f"{prefix}_AD_B_rel": rel_error(b_np, b_ref),
        f"{prefix}_FD_B_rel": rel_error(b_fd_np, b_ref),
        f"{prefix}_LE_mse": float(np.mean((le_np - le_ref) ** 2)),
        f"{prefix}_balanced_B_mse": float(b_loss),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if bool(args.cuda) and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if str(args.prepared).strip():
        with np.load(str(args.prepared), allow_pickle=True) as z:
            arrays_obj = {
                "q": np.asarray(z["q48_def_hat"], dtype=np.float32),
                "g": np.asarray(z["geometry_g"], dtype=np.float32),
                "x": np.asarray(z["x_gp_hat"], dtype=np.float32),
                "le": np.asarray(z["LE_macro"], dtype=np.float32),
                "b": np.asarray(z["B_macro_qdef"], dtype=np.float32),
                "case_id": np.asarray(z["case_id"], dtype=np.int64),
            }
            if "q48_def_hat_plus" in z.files and "LE_macro_plus" in z.files:
                arrays_obj["q_plus"] = np.asarray(z["q48_def_hat_plus"], dtype=np.float32)
                arrays_obj["le_plus"] = np.asarray(z["LE_macro_plus"], dtype=np.float32)
                arrays_obj["plus_delta_raw"] = np.asarray(z["plus_delta_raw"], dtype=np.float64)
                arrays_obj["plus_raw_directions"] = np.asarray(z["plus_raw_directions"], dtype=np.int64)
            compact_paths = [str(v) for v in np.asarray(z["compact_paths"], dtype=object).reshape(-1)]
            gp_source = str(np.asarray(z["gp_coordinate_source"], dtype=object).reshape(-1)[0])
            geometry_param_names = [str(v) for v in np.asarray(z["geometry_param_names"], dtype=object).reshape(-1)]
    else:
        loaded = load_coordinate_arrays(
            paths_from_args(args),
            frame_stride=int(args.frame_stride),
            max_frames_per_compact=int(args.max_frames_per_compact),
        )
        arrays_obj = {
            "q": loaded.q48_def_hat.astype(np.float32),
            "g": loaded.geometry_g.astype(np.float32),
            "x": loaded.x_gp_hat.astype(np.float32),
            "le": loaded.le_macro.astype(np.float32),
            "b": loaded.b_macro_qdef.astype(np.float32),
            "case_id": loaded.case_id.astype(np.int64),
        }
        if loaded.q48_def_hat_plus is not None and loaded.le_macro_plus is not None:
            arrays_obj["q_plus"] = loaded.q48_def_hat_plus.astype(np.float32)
            arrays_obj["le_plus"] = loaded.le_macro_plus.astype(np.float32)
            arrays_obj["plus_delta_raw"] = np.asarray([float(loaded.plus_delta_raw)], dtype=np.float64)
            arrays_obj["plus_raw_directions"] = loaded.plus_raw_directions.astype(np.int64)
        compact_paths = loaded.compact_paths
        gp_source = loaded.gp_coordinate_source
        geometry_param_names = loaded.geometry_param_names

    if arrays_obj["x"].shape[1] != 128:
        raise ValueError("this experiment keeps the fixed 128 point rule")
    train_idx, val_idx, split_meta = split_indices(
        arrays_obj["case_id"],
        val_cases=str(args.val_cases),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )
    if train_idx.size == 0:
        raise ValueError("empty training split")

    branch_raw = np.concatenate([arrays_obj["q"], arrays_obj["g"]], axis=1)
    branch_mean, branch_std = stats(branch_raw[train_idx], axis=0, floor=float(args.branch_std_floor))
    q_std_before_floor = np.std(branch_raw[train_idx, :48].astype(np.float64), axis=0)
    le_mean, le_std = stats(arrays_obj["le"][train_idx], axis=0, floor=float(args.le_std_floor))
    b_rms = np.sqrt(np.mean(arrays_obj["b"][train_idx].astype(np.float64) ** 2, axis=0, keepdims=True)).astype(np.float32)
    b_scale_np = np.maximum(b_rms, float(args.b_scale_floor)).astype(np.float32)
    plus_available = "q_plus" in arrays_obj and "le_plus" in arrays_obj
    if plus_available:
        plus_delta_raw = float(np.asarray(arrays_obj["plus_delta_raw"], dtype=np.float64).reshape(-1)[0])
        plus_slope = (
            np.asarray(arrays_obj["le_plus"], dtype=np.float64)[train_idx]
            - np.asarray(arrays_obj["le"], dtype=np.float64)[train_idx, None, :, :]
        ) / plus_delta_raw
        plus_slope = np.transpose(plus_slope, (0, 2, 3, 1))
        plus_slope_scale_np = np.maximum(
            np.sqrt(np.mean(plus_slope**2, axis=0, keepdims=True)),
            float(args.b_scale_floor),
        ).astype(np.float32)
    else:
        plus_delta_raw = float("nan")
        plus_slope_scale_np = np.ones((1, 128, 6, 48), dtype=np.float32)

    branch_mean_t = torch.as_tensor(branch_mean.reshape(1, -1), dtype=torch.float32, device=device)
    branch_std_t = torch.as_tensor(branch_std.reshape(1, -1), dtype=torch.float32, device=device)
    le_mean_t = torch.as_tensor(le_mean.reshape(1, 128, 6), dtype=torch.float32, device=device)
    le_std_t = torch.as_tensor(le_std.reshape(1, 128, 6), dtype=torch.float32, device=device)
    b_scale_t = torch.as_tensor(b_scale_np, dtype=torch.float32, device=device)
    plus_slope_scale_t = torch.as_tensor(plus_slope_scale_np, dtype=torch.float32, device=device)

    style = str(args.model_style).strip().lower().replace("_", "-")
    if style == "linear-residual":
        model = Macro16CoordinateLinearResidualDeepONet(
            branch_dim=int(branch_raw.shape[1]),
            trunk_dim=3,
            q_dim=48,
            strain_dim=6,
            basis_dim=int(args.basis_dim),
            hidden_dim=int(args.hidden_dim),
            branch_depth=int(args.branch_depth),
            trunk_depth=int(args.trunk_depth),
            activation=str(args.activation),
            zero_init_residual=not bool(args.no_zero_init_residual),
        ).to(device)
    elif style == "plain":
        model = Macro16CoordinateDeepONet(
            branch_dim=int(branch_raw.shape[1]),
            trunk_dim=3,
            strain_dim=6,
            basis_dim=int(args.basis_dim),
            hidden_dim=int(args.hidden_dim),
            branch_depth=int(args.branch_depth),
            trunk_depth=int(args.trunk_depth),
            activation=str(args.activation),
        ).to(device)
    else:
        raise ValueError("--model-style must be plain or linear-residual")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    train_set = CoordinateDataset(arrays_obj, train_idx)
    loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, drop_last=False)
    rng = np.random.default_rng(int(args.seed) + 17)
    all_columns = list(range(48))
    eval_columns = parse_int_list(str(args.eval_columns), all_columns)
    history: list[dict[str, Any]] = []
    initial_report: dict[str, Any] | None = None
    best_report: dict[str, Any] | None = None
    best_score = math.inf

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        sum_loss = 0.0
        sum_le = 0.0
        sum_b = 0.0
        count = 0
        for idxb, q, g, x, le, b in loader:
            idx_np = idxb.numpy().astype(np.int64)
            q = q.to(device)
            g = g.to(device)
            x = x.to(device)
            le = le.to(device)
            b = b.to(device)
            q_req = q.detach().requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            le_norm_pred = predict_le_norm(
                model,
                q_req,
                g,
                x,
                branch_mean=branch_mean_t,
                branch_std=branch_std_t,
            )
            le_norm_true = (le - le_mean_t) / le_std_t
            le_loss = nn.functional.mse_loss(le_norm_pred, le_norm_true)
            if int(args.b_columns_per_step) >= 48:
                columns = all_columns
            else:
                columns = sorted(rng.choice(48, size=max(1, int(args.b_columns_per_step)), replace=False).tolist())
            le_phys_pred = le_norm_pred * le_std_t + le_mean_t
            del le_phys_pred
            b_loss_source = str(args.b_loss_source).strip().lower().replace("_", "-")
            if b_loss_source in {"ad", "ad-only"}:
                b_pred = b_jvp_columns(
                    model,
                    q_req,
                    g,
                    x,
                    columns,
                    branch_mean=branch_mean_t,
                    branch_std=branch_std_t,
                    le_mean=le_mean_t,
                    le_std=le_std_t,
                )
                b_loss = balanced_b_loss(b_pred, b[:, :, :, columns], b_scale_t[:, :, :, columns])
            elif b_loss_source in {"fd", "forward-diff", "forward-difference"}:
                b_pred_fd = b_forward_diff_columns(
                    model,
                    q_req,
                    g,
                    x,
                    columns,
                    branch_mean=branch_mean_t,
                    branch_std=branch_std_t,
                    le_mean=le_mean_t,
                    le_std=le_std_t,
                    fd_step=float(args.fd_step),
                    column_chunk=int(args.fd_column_chunk),
                )
                b_loss = balanced_b_loss(b_pred_fd, b[:, :, :, columns], b_scale_t[:, :, :, columns])
            elif b_loss_source in {"plus-fd", "source-plus-fd", "data-fd"}:
                if not plus_available:
                    raise ValueError("--b-loss-source plus-fd requires q48_def_hat_plus and LE_macro_plus")
                q_plus = torch.as_tensor(arrays_obj["q_plus"][idx_np], dtype=torch.float32, device=device)
                le_plus = torch.as_tensor(arrays_obj["le_plus"][idx_np], dtype=torch.float32, device=device)
                b_loss, _pred_slope, _target_slope = source_plus_fd_loss(
                    model,
                    q_req,
                    g,
                    x,
                    le,
                    q_plus,
                    le_plus,
                    columns,
                    branch_mean=branch_mean_t,
                    branch_std=branch_std_t,
                    le_mean=le_mean_t,
                    le_std=le_std_t,
                    plus_delta_raw=plus_delta_raw,
                    plus_slope_scale=plus_slope_scale_t,
                )
            elif b_loss_source in {"ad-plus-fd", "fd-plus-ad"}:
                b_pred_ad = b_jvp_columns(
                    model,
                    q_req,
                    g,
                    x,
                    columns,
                    branch_mean=branch_mean_t,
                    branch_std=branch_std_t,
                    le_mean=le_mean_t,
                    le_std=le_std_t,
                )
                b_pred_fd = b_forward_diff_columns(
                    model,
                    q_req,
                    g,
                    x,
                    columns,
                    branch_mean=branch_mean_t,
                    branch_std=branch_std_t,
                    le_mean=le_mean_t,
                    le_std=le_std_t,
                    fd_step=float(args.fd_step),
                    column_chunk=int(args.fd_column_chunk),
                )
                target_cols = b[:, :, :, columns]
                scale_cols = b_scale_t[:, :, :, columns]
                b_loss = 0.5 * (
                    balanced_b_loss(b_pred_ad, target_cols, scale_cols)
                    + balanced_b_loss(b_pred_fd, target_cols, scale_cols)
                )
            else:
                raise ValueError("--b-loss-source must be ad, fd, plus-fd, or ad-plus-fd")
            loss = float(args.le_weight) * le_loss + float(args.b_weight) * b_loss
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            batch_n = int(q.shape[0])
            count += batch_n
            sum_loss += float(loss.detach().cpu()) * batch_n
            sum_le += float(le_loss.detach().cpu()) * batch_n
            sum_b += float(b_loss.detach().cpu()) * batch_n

        row: dict[str, Any] = {
            "epoch": epoch,
            "loss": sum_loss / max(count, 1),
            "le_norm_mse": sum_le / max(count, 1),
            "balanced_b_mse": sum_b / max(count, 1),
        }
        do_eval = epoch == 1 or epoch % int(args.eval_every) == 0 or epoch == int(args.epochs)
        if do_eval:
            row.update(
                evaluate(
                    model,
                    arrays_obj,
                    train_idx,
                    device=device,
                    branch_mean=branch_mean_t,
                    branch_std=branch_std_t,
                    le_mean=le_mean_t,
                    le_std=le_std_t,
                    b_scale=b_scale_t,
                    eval_columns=eval_columns,
                    fd_step=float(args.fd_step),
                    fd_column_chunk=int(args.fd_column_chunk),
                    prefix="train",
                )
            )
            row.update(
                evaluate(
                    model,
                    arrays_obj,
                    val_idx,
                    device=device,
                    branch_mean=branch_mean_t,
                    branch_std=branch_std_t,
                    le_mean=le_mean_t,
                    le_std=le_std_t,
                    b_scale=b_scale_t,
                    eval_columns=eval_columns,
                    fd_step=float(args.fd_step),
                    fd_column_chunk=int(args.fd_column_chunk),
                    prefix="val",
                )
            )
            if initial_report is None:
                initial_report = dict(row)
            score = float(row.get("train_LE_rel", row["loss"])) + float(row.get("train_AD_B_rel", 0.0))
            if score < best_score:
                best_score = score
                best_report = dict(row)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "norms": {
                            "branch_mean": branch_mean,
                            "branch_std": branch_std,
                            "le_mean": le_mean,
                            "le_std": le_std,
                            "b_scale": b_scale_np,
                        },
                        "args": vars(args),
                        "geometry_param_names": geometry_param_names,
                        "compact_paths": compact_paths,
                        "best_report": best_report,
                    },
                    out_dir / "best.pt",
                )
        history.append(row)
        if epoch % int(args.log_every) == 0 or do_eval:
            print(json.dumps(row, sort_keys=True), flush=True)

    latest = history[-1] if history else {}
    summary = {
        "experiment": "macro16_coordinate_deeponet",
        "status": "smoke_complete",
        "compact_paths": compact_paths,
        "frame_count": int(arrays_obj["q"].shape[0]),
        "train_frames": int(train_idx.size),
        "val_frames": int(val_idx.size),
        "point_count": int(arrays_obj["x"].shape[1]),
        "branch_input": "q48_def_hat[48] + geometry_g[14]",
        "trunk_input": "x_gp_hat[3]",
        "output": "LE[6]",
        "b_definition": "AD dLE/dq48_def_hat",
        "b_target": "B_macro_qdef",
        "b_loss_source": str(args.b_loss_source),
        "fd_step": float(args.fd_step),
        "source_plus_available": bool(plus_available),
        "source_plus_delta_raw": None if not plus_available else float(plus_delta_raw),
        "source_plus_raw_directions": (
            None
            if not plus_available
            else [int(v) for v in np.asarray(arrays_obj["plus_raw_directions"], dtype=np.int64).reshape(-1)]
        ),
        "model_style": style,
        "gp_coordinate_source": gp_source,
        "geometry_param_names": geometry_param_names,
        "split": split_meta,
        "device": str(device),
        "q_std_before_floor_min": float(np.min(q_std_before_floor)),
        "q_std_before_floor_max": float(np.max(q_std_before_floor)),
        "initial_report": initial_report,
        "best_report": best_report,
        "latest_report": latest,
        "history": history,
        "best_checkpoint": str(out_dir / "best.pt"),
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="append", default=[])
    parser.add_argument("--compact-list", default="")
    parser.add_argument("--prepared", type=Path, default=Path(""))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-compact", type=int, default=0)
    parser.add_argument("--val-cases", default="")
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--basis-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--branch-depth", type=int, default=4)
    parser.add_argument("--trunk-depth", type=int, default=4)
    parser.add_argument("--activation", default="tanh")
    parser.add_argument("--model-style", default="plain", choices=["plain", "linear-residual"])
    parser.add_argument("--no-zero-init-residual", action="store_true")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--le-weight", type=float, default=1.0)
    parser.add_argument("--b-weight", type=float, default=0.05)
    parser.add_argument("--b-loss-source", default="ad", choices=["ad", "fd", "plus-fd", "ad-plus-fd"])
    parser.add_argument("--b-columns-per-step", type=int, default=12)
    parser.add_argument("--fd-step", type=float, default=1.0e-4)
    parser.add_argument("--fd-column-chunk", type=int, default=12)
    parser.add_argument("--eval-columns", default="all")
    parser.add_argument("--branch-std-floor", type=float, default=1.0e-4)
    parser.add_argument("--le-std-floor", type=float, default=1.0e-8)
    parser.add_argument("--b-scale-floor", type=float, default=1.0e-8)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
