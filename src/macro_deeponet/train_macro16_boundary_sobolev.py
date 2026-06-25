"""Train the Macro16 boundary-control shell operator.

This route is intentionally separate from the v3 CSS8 scripts.  Model-visible
geometry is limited to 16 boundary control nodes, while q keeps all 48 boundary
displacement components.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .macro16_geometry import (
    MACRO16_CONTRACT_VERSION,
    Macro16PointTable,
    build_macro16_feature_batch,
    macro16_standard_point_table,
)
from .models import Macro16BoundaryDeepONet
from .train_true176_deeponet_sobolev import ad_jacobian, cos_np, rel_np, write_json
from .true176_data import stats


@dataclass(frozen=True)
class Macro16Arrays:
    compact_paths: list[str]
    q48_hat: np.ndarray
    x16_hat: np.ndarray
    point_features_hat: np.ndarray
    le: np.ndarray
    b: np.ndarray
    weights: np.ndarray
    length_scale: np.ndarray
    x_center: np.ndarray
    case_id: np.ndarray
    source_index: np.ndarray
    source_row: np.ndarray
    point_meta: dict[str, Any]


class Macro16Dataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        branch: np.ndarray,
        point: np.ndarray,
        le: np.ndarray,
        j: np.ndarray,
        weights: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.branch = np.asarray(branch, dtype=np.float32)
        self.point = np.asarray(point, dtype=np.float32)
        self.le = np.asarray(le, dtype=np.float32)
        self.j = np.asarray(j, dtype=np.float32)
        self.weights = np.asarray(weights, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        return (
            torch.from_numpy(self.branch[idx]),
            torch.from_numpy(self.point[idx]),
            torch.from_numpy(self.le[idx]),
            torch.from_numpy(self.j[idx]),
            torch.from_numpy(self.weights[idx]),
        )


def _scalar_text(z: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    if arr.size == 0:
        return default
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _infer_n(z: np.lib.npyio.NpzFile, path: Path) -> int:
    for key in ("q48_raw", "q48_hat", "q_boundary", "LE_macro", "LE128_base", "B_macro", "B_LE128_forward"):
        if key in z.files:
            arr = np.asarray(z[key])
            if arr.ndim >= 1:
                return int(arr.shape[0])
    raise KeyError(f"{path}: cannot infer frame count")


def _case_id_from_path(path: Path) -> int:
    import re

    match = re.search(r"case[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def _as_frame_x16(z: np.lib.npyio.NpzFile, path: Path, n_total: int, rows: np.ndarray) -> np.ndarray:
    candidates = (
        "X16",
        "X16_ref",
        "X_keep",
        "X_keep_ref",
        "control_node_coords",
        "control_nodes16",
        "keep_node_coords",
    )
    for key in candidates:
        if key not in z.files:
            continue
        vals = np.asarray(z[key], dtype=np.float32)
        if vals.shape == (16, 3):
            return np.broadcast_to(vals.reshape(1, 16, 3), (rows.size, 16, 3)).copy()
        if vals.shape == (n_total, 16, 3):
            return vals[rows]
        raise ValueError(f"{path}: {key} must have shape [16,3] or [{n_total},16,3], got {vals.shape}")
    raise KeyError(f"{path}: missing X16 or X_keep. Macro16 training must not use X_macro or CSS8 internal nodes.")


def _as_q48(z: np.lib.npyio.NpzFile, path: Path, n_total: int, rows: np.ndarray) -> np.ndarray:
    for key in ("q48_raw", "q48_hat", "q_boundary"):
        if key not in z.files:
            continue
        vals = np.asarray(z[key], dtype=np.float32)
        if vals.shape == (n_total, 48):
            return vals[rows]
        if vals.shape == (n_total, 16, 3):
            return vals[rows].reshape(rows.size, 48)
        raise ValueError(f"{path}: {key} must have shape [{n_total},48] or [{n_total},16,3], got {vals.shape}")
    raise KeyError(f"{path}: missing q48_raw q48_hat or q_boundary")


def _as_le(z: np.lib.npyio.NpzFile, path: Path, n_total: int, rows: np.ndarray, point_count: int) -> np.ndarray:
    for key in ("LE_macro", "LE16_macro", "LE_local_stack", "LE128_base", "LE_abq", "le"):
        if key not in z.files:
            continue
        vals = np.asarray(z[key], dtype=np.float32)
        if vals.shape == (n_total, point_count, 6):
            return vals[rows]
        raise ValueError(
            f"{path}: {key} must align with Macro16 point count {point_count}, got {vals.shape}. "
            "Do not feed CSS8 128-row labels unless the chosen Macro16 point table also has 128 rows."
        )
    raise KeyError(f"{path}: missing LE_macro or compatible LE label")


def _as_b(z: np.lib.npyio.NpzFile, path: Path, n_total: int, rows: np.ndarray, point_count: int) -> np.ndarray:
    for key in ("B_macro", "B_macro_q48", "B_LE_macro", "B_LE128_forward", "b"):
        if key not in z.files:
            continue
        vals = np.asarray(z[key], dtype=np.float32)
        if vals.shape == (n_total, point_count, 6, 48):
            return vals[rows]
        raise ValueError(f"{path}: {key} must have shape [{n_total},{point_count},6,48], got {vals.shape}")
    raise KeyError(f"{path}: missing B_macro or compatible B label with 48 q columns")


def _length_scale(z: np.lib.npyio.NpzFile, n_total: int, rows: np.ndarray) -> np.ndarray:
    for key in ("H", "length_scale", "length_scale_H", "macro_length_scale"):
        if key not in z.files:
            continue
        vals = np.asarray(z[key], dtype=np.float32)
        if vals.ndim == 0 or vals.shape == (1,):
            return np.full((rows.size, 1), float(vals.reshape(-1)[0]), dtype=np.float32)
        if vals.shape == (n_total,):
            return vals[rows].reshape(-1, 1).astype(np.float32)
        if vals.shape == (n_total, 1):
            return vals[rows].astype(np.float32)
        raise ValueError(f"{key} must be scalar [N] or [N,1], got {vals.shape}")
    return np.ones((rows.size, 1), dtype=np.float32)


def _rows(n_total: int, *, frame_stride: int, max_frames: int) -> np.ndarray:
    rows = np.arange(0, int(n_total), max(1, int(frame_stride)), dtype=np.int64)
    if int(max_frames) > 0:
        rows = rows[: int(max_frames)]
    return rows


def read_path_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def compact_paths_from_args(args: argparse.Namespace) -> list[str]:
    paths = [str(v) for v in getattr(args, "compact", [])]
    if str(getattr(args, "compact_list", "")).strip():
        paths.extend(read_path_list(Path(args.compact_list)))
    unique: list[str] = []
    seen: set[str] = set()
    for text in paths:
        key = str(Path(text).resolve())
        if key not in seen:
            seen.add(key)
            unique.append(key)
    if not unique:
        raise ValueError("provide --compact or --compact-list")
    return unique


def load_macro16_compacts(
    paths: Iterable[str],
    *,
    point_table: Macro16PointTable,
    frame_stride: int = 1,
    max_frames_per_compact: int = 0,
    scale_mode: str = "normalized",
    b_label_coordinate: str = "auto",
) -> Macro16Arrays:
    q_chunks: list[np.ndarray] = []
    xhat_chunks: list[np.ndarray] = []
    point_chunks: list[np.ndarray] = []
    le_chunks: list[np.ndarray] = []
    b_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    length_chunks: list[np.ndarray] = []
    center_chunks: list[np.ndarray] = []
    case_chunks: list[np.ndarray] = []
    source_chunks: list[np.ndarray] = []
    row_chunks: list[np.ndarray] = []
    compact_paths: list[str] = []
    point_meta: dict[str, Any] | None = None
    mode = str(scale_mode).strip().lower()
    if mode not in {"normalized", "physical"}:
        raise ValueError("scale_mode must be normalized or physical")

    for source_id, text in enumerate(paths):
        path = Path(text).resolve()
        with np.load(str(path), allow_pickle=True) as z:
            n_total = _infer_n(z, path)
            rows = _rows(n_total, frame_stride=frame_stride, max_frames=max_frames_per_compact)
            x16_raw = _as_frame_x16(z, path, n_total, rows)
            q_raw = _as_q48(z, path, n_total, rows)
            le = _as_le(z, path, n_total, rows, point_table.xi.shape[0])
            b_raw = _as_b(z, path, n_total, rows, point_table.xi.shape[0])
            h = _length_scale(z, n_total, rows)
            if mode == "physical":
                if np.allclose(h, 1.0) and not any(k in z.files for k in ("H", "length_scale", "length_scale_H", "macro_length_scale")):
                    raise ValueError(f"{path}: scale_mode=physical requires explicit H or length_scale")
            x16_hat, point_features, weights, l_ref, meta = build_macro16_feature_batch(x16_raw, point_table)
            if mode == "physical":
                q_hat = q_raw / np.maximum(l_ref, 1.0e-12)
            else:
                q_hat = q_raw
            coord = str(b_label_coordinate).strip().lower().replace("_", "-")
            if coord == "auto":
                coord = "physical" if mode == "physical" else "dimensionless"
            if coord in {"physical", "dimensional", "q-phys"} and mode == "physical":
                b = b_raw * l_ref.reshape(l_ref.shape[0], 1, 1, 1)
            elif coord in {"dimensionless", "normalized", "hat", "q-hat"}:
                b = b_raw
            elif coord in {"physical", "dimensional", "q-phys"} and mode == "normalized":
                b = b_raw
            else:
                raise ValueError("b_label_coordinate must be auto physical or dimensionless")
            case_id = np.full(rows.size, _case_id_from_path(path), dtype=np.int64)
            if "case_id" in z.files:
                vals = np.asarray(z["case_id"], dtype=np.int64)
                if vals.ndim == 0 or vals.shape == (1,):
                    case_id[:] = int(vals.reshape(-1)[0])
                elif vals.shape == (n_total,):
                    case_id = vals[rows]
            compact_paths.append(str(path))
            q_chunks.append(q_hat.astype(np.float32))
            xhat_chunks.append(x16_hat.astype(np.float32))
            point_chunks.append(point_features.astype(np.float32))
            le_chunks.append(le.astype(np.float32))
            b_chunks.append(b.astype(np.float32))
            weight_chunks.append(weights.astype(np.float32))
            length_chunks.append(l_ref.astype(np.float32))
            center_chunks.append(np.asarray(meta["X_center"], dtype=np.float32))
            case_chunks.append(case_id)
            source_chunks.append(np.full(rows.size, int(source_id), dtype=np.int64))
            row_chunks.append(rows.astype(np.int64))
            point_meta = meta
            version = _scalar_text(z, "standard_operator_contract_version", "")
            if version and version != MACRO16_CONTRACT_VERSION:
                raise ValueError(f"{path}: expected {MACRO16_CONTRACT_VERSION}, got {version}")

    if not q_chunks:
        raise ValueError("no Macro16 frames loaded")
    return Macro16Arrays(
        compact_paths=compact_paths,
        q48_hat=np.concatenate(q_chunks, axis=0),
        x16_hat=np.concatenate(xhat_chunks, axis=0),
        point_features_hat=np.concatenate(point_chunks, axis=0),
        le=np.concatenate(le_chunks, axis=0),
        b=np.concatenate(b_chunks, axis=0),
        weights=np.concatenate(weight_chunks, axis=0),
        length_scale=np.concatenate(length_chunks, axis=0),
        x_center=np.concatenate(center_chunks, axis=0),
        case_id=np.concatenate(case_chunks, axis=0),
        source_index=np.concatenate(source_chunks, axis=0),
        source_row=np.concatenate(row_chunks, axis=0),
        point_meta=point_meta or {},
    )


def split_indices(case_id: np.ndarray, *, val_fraction: float, val_cases: str, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n = int(case_id.shape[0])
    all_idx = np.arange(n, dtype=np.int64)
    text = str(val_cases).strip()
    if text:
        wanted = {int(v) for v in text.replace(";", ",").split(",") if v.strip()}
        mask = np.isin(case_id, np.asarray(sorted(wanted), dtype=np.int64))
        train = all_idx[~mask]
        val = all_idx[mask]
        if train.size == 0 or val.size == 0:
            raise ValueError("val_cases produced empty train or validation split")
        return train, val, {
            "validation_split_mode": "case",
            "validation_is_overlapping": False,
            "val_cases": sorted(wanted),
        }
    valid = np.asarray(case_id, dtype=np.int64)
    unique = np.unique(valid[valid >= 0])
    if unique.size >= 2:
        rng = np.random.default_rng(int(seed))
        shuffled = rng.permutation(unique)
        n_val = min(max(1, int(round(max(float(val_fraction), 0.2) * unique.size))), unique.size - 1)
        val_groups = np.sort(shuffled[:n_val])
        val = all_idx[np.isin(valid, val_groups)]
        train = all_idx[~np.isin(valid, val_groups)]
        return train, val, {
            "validation_split_mode": "case",
            "validation_is_overlapping": False,
            "val_cases": val_groups.astype(int).tolist(),
        }
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_val = int(round(max(float(val_fraction), 0.2) * n))
    n_val = min(max(n_val, 1), n - 1)
    return perm[n_val:], perm[:n_val], {
        "validation_split_mode": "frame",
        "validation_is_overlapping": False,
        "warning": "frame split used because case metadata has fewer than two cases",
    }


def rigid_q48_modes(x16_hat: np.ndarray) -> np.ndarray:
    """Return six infinitesimal rigid displacement modes for each frame."""

    x = np.asarray(x16_hat, dtype=np.float32).reshape(-1, 16, 3)
    modes = np.zeros((x.shape[0], 6, 16, 3), dtype=np.float32)
    modes[:, 0, :, 0] = 1.0
    modes[:, 1, :, 1] = 1.0
    modes[:, 2, :, 2] = 1.0
    axes = np.eye(3, dtype=np.float32)
    for i, axis in enumerate(axes):
        modes[:, 3 + i] = np.cross(axis.reshape(1, 1, 3), x, axis=-1)
    return modes.reshape(x.shape[0], 6, 48)


def standardize(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(arr, dtype=np.float32) - mean) / std).astype(np.float32)


def _le_std_scale(le_std: np.ndarray, point_count: int) -> np.ndarray:
    vals = np.asarray(le_std, dtype=np.float32)
    if vals.ndim == 1:
        vals = vals.reshape(1, 1, -1)
    elif vals.ndim == 2:
        vals = vals.reshape(1, vals.shape[0], vals.shape[1])
    if vals.shape[1] not in {1, point_count}:
        raise ValueError(f"LE std point dimension cannot broadcast to {point_count}")
    return np.maximum(vals, 1.0e-12)[:, :, :, None]


def sample_columns(columns: list[int], count: int, rng: np.random.Generator) -> list[int]:
    if int(count) <= 0 or int(count) >= len(columns):
        return list(columns)
    picked = rng.choice(np.asarray(columns, dtype=np.int64), size=int(count), replace=False)
    return [int(v) for v in picked.tolist()]


def parse_int_list(text: str) -> list[int]:
    raw = str(text).strip()
    if not raw or raw.lower() == "all":
        return list(range(48))
    return [int(v) for v in raw.replace(";", ",").split(",") if v.strip()]


def evaluate(
    model: nn.Module,
    *,
    branch_norm: np.ndarray,
    point_norm: np.ndarray,
    le_raw: np.ndarray,
    b_raw: np.ndarray,
    j_norm_target: np.ndarray,
    norms: dict[str, np.ndarray],
    indices: np.ndarray,
    columns: list[int],
    batch_size: int,
    device: torch.device,
    prefix: str,
) -> dict[str, Any]:
    idx = np.asarray(indices, dtype=np.int64)
    le_rows: list[np.ndarray] = []
    j_rows: list[np.ndarray] = []
    b_rows: list[np.ndarray] = []
    q_std_cols = norms["branch_std"].reshape(-1)[:48][np.asarray(columns, dtype=np.int64)]
    le_std = norms["le_std"].astype(np.float32)
    model.eval()
    for start in range(0, idx.size, int(batch_size)):
        sub = idx[start : start + int(batch_size)]
        xb = torch.as_tensor(branch_norm[sub], dtype=torch.float32, device=device)
        pb = torch.as_tensor(point_norm[sub], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_norm = model(xb, pb).detach().cpu().numpy()
        le_rows.append(pred_norm * norms["le_std"] + norms["le_mean"])
        with torch.enable_grad():
            jn = ad_jacobian(model, xb, pb, columns, create_graph=False, method="forward")
        j_np = jn.detach().cpu().numpy().astype(np.float64)
        j_rows.append(j_np)
        b_rows.append(j_np * _le_std_scale(le_std, j_np.shape[1]) / q_std_cols.reshape(1, 1, 1, -1))
    le_pred = np.concatenate(le_rows, axis=0).astype(np.float64)
    j_pred = np.concatenate(j_rows, axis=0).astype(np.float64)
    b_pred = np.concatenate(b_rows, axis=0).astype(np.float64)
    le_true = le_raw[idx].astype(np.float64)
    j_true = j_norm_target[idx][:, :, :, columns].astype(np.float64)
    b_true = b_raw[idx][:, :, :, columns].astype(np.float64)
    return {
        f"{prefix}_frames": int(idx.size),
        f"{prefix}_LE_rel": rel_np(le_pred, le_true),
        f"{prefix}_LE_cos": cos_np(le_pred, le_true),
        f"{prefix}_AD_B_norm_rel": rel_np(j_pred, j_true),
        f"{prefix}_AD_B_rel": rel_np(b_pred, b_true),
        f"{prefix}_AD_B_cos": cos_np(b_pred, b_true),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if bool(args.cuda) and torch.cuda.is_available() else "cpu")
    point_table = macro16_standard_point_table(
        plane_order=int(args.plane_gauss_order),
        thickness_order=int(args.thickness_gauss_order),
    )
    compact_paths = compact_paths_from_args(args)
    data = load_macro16_compacts(
        compact_paths,
        point_table=point_table,
        frame_stride=int(args.frame_stride),
        max_frames_per_compact=int(args.max_frames_per_compact),
        scale_mode=str(args.scale_mode),
        b_label_coordinate=str(args.b_label_coordinate),
    )
    train_idx, val_idx, split_meta = split_indices(
        data.case_id,
        val_fraction=float(args.val_fraction),
        val_cases=str(args.val_cases),
        seed=int(args.seed),
    )

    branch_raw = np.concatenate([data.q48_hat, data.x16_hat.reshape(data.x16_hat.shape[0], -1), data.length_scale], axis=1)
    branch_mean, branch_std = stats(branch_raw[train_idx], axis=0)
    point_mean, point_std = stats(data.point_features_hat[train_idx].reshape(-1, data.point_features_hat.shape[-1]), axis=0)
    le_mean, le_std = stats(data.le[train_idx], axis=0)
    branch_norm = standardize(branch_raw, branch_mean, branch_std)
    point_norm = standardize(data.point_features_hat, point_mean.reshape(1, 1, -1), point_std.reshape(1, 1, -1))
    le_norm = standardize(data.le, le_mean, le_std)
    q_std = branch_std.reshape(-1)[:48]
    j_norm_target = (data.b * q_std.reshape(1, 1, 1, 48) / _le_std_scale(le_std, data.le.shape[1])).astype(np.float32)
    skip_init = np.mean(j_norm_target[train_idx], axis=0).astype(np.float32)
    if bool(args.global_b_prior):
        skip_init = np.mean(skip_init, axis=0)
    norms = {
        "branch_mean": branch_mean.astype(np.float32),
        "branch_std": branch_std.astype(np.float32),
        "point_mean": point_mean.astype(np.float32),
        "point_std": point_std.astype(np.float32),
        "le_mean": le_mean.astype(np.float32),
        "le_std": le_std.astype(np.float32),
        "q_start": np.asarray(0, dtype=np.int64),
        "q_dim": np.asarray(48, dtype=np.int64),
    }

    train_set = Macro16Dataset(branch_norm, point_norm, le_norm, j_norm_target, data.weights, train_idx)
    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, drop_last=False)
    model = Macro16BoundaryDeepONet(
        input_dim=int(branch_norm.shape[-1]),
        point_dim=int(point_norm.shape[-1]),
        ip_count=int(point_norm.shape[1]),
        q_start=0,
        q_dim=48,
        basis_dim=int(args.basis_dim),
        hidden_dim=int(args.hidden_dim),
        branch_depth=int(args.branch_depth),
        trunk_depth=int(args.trunk_depth),
        activation=str(args.activation),
        skip_init=torch.as_tensor(skip_init, dtype=torch.float32),
        train_skip=not bool(args.freeze_skip),
        residual_scale=float(args.residual_scale),
        baseline_scale=float(args.fe_baseline_scale),
        train_point_baseline=not bool(args.freeze_fe_point_baseline),
        q_zero_norm=((0.0 - branch_mean.reshape(-1)[:48]) / branch_std.reshape(-1)[:48]).astype(np.float32),
        le_zero_norm=((0.0 - le_mean) / le_std).astype(np.float32),
        q_raw_mean=branch_mean.reshape(-1)[:48].astype(np.float32),
        q_raw_std=branch_std.reshape(-1)[:48].astype(np.float32),
        gate_q0=float(args.anchored_residual_gate_q0),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(args.lr_decay))
    columns_all = parse_int_list(str(args.jacobian_columns))
    eval_columns = parse_int_list(str(args.eval_columns))
    col_rng = np.random.default_rng(int(args.seed) + 307)
    rigid_rng = np.random.default_rng(int(args.seed) + 1307)
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_report: dict[str, Any] | None = None

    write_json(
        out_dir / "config.json",
        {
            "standard_operator_contract_version": MACRO16_CONTRACT_VERSION,
            "args": vars(args),
            "compact_paths": data.compact_paths,
            "train_frames": int(train_idx.size),
            "val_frames": int(val_idx.size),
            "validation_split": split_meta,
            "branch_contract": "q48_hat[48] plus X16_hat[16,3] plus L_ref",
            "q_dim": 48,
            "rigid_modes_removed_from_input": False,
            "fine_grid_geometry_visible": False,
            "css8_internal_geometry_visible": False,
            "point_meta": data.point_meta,
            "point_feature_names": data.point_meta.get("point_feature_names", []),
            "model_meta": {
                "model_style": "macro16-boundary-deeponet",
                "input_dim": int(branch_norm.shape[-1]),
                "point_dim": int(point_norm.shape[-1]),
                "ip_count": int(point_norm.shape[1]),
                "ad_target": "dLE/dq48_hat",
            },
        },
    )

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        t0 = time.time()
        sums = {"loss": 0.0, "le": 0.0, "j": 0.0, "rigid": 0.0}
        count = 0
        for xb, pb, leb, jb, _wb in train_loader:
            xb = xb.to(device)
            pb = pb.to(device)
            leb = leb.to(device)
            jb = jb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, pb)
            le_loss = nn.functional.mse_loss(pred, leb)
            columns = sample_columns(columns_all, int(args.jacobian_columns_per_batch), col_rng)
            if columns:
                j_pred = ad_jacobian(model, xb, pb, columns, create_graph=True, method="forward")
                j_loss = nn.functional.mse_loss(j_pred, jb[:, :, :, columns])
            else:
                j_loss = torch.zeros((), dtype=xb.dtype, device=device)
            rigid_loss = torch.zeros((), dtype=xb.dtype, device=device)
            if float(args.rigid_loss_weight) > 0.0:
                batch_x16_norm = xb[:, 48:96] * torch.as_tensor(
                    branch_std.reshape(-1)[48:96], dtype=xb.dtype, device=device
                ) + torch.as_tensor(branch_mean.reshape(-1)[48:96], dtype=xb.dtype, device=device)
                x16_np = batch_x16_norm.detach().cpu().numpy().reshape(-1, 16, 3)
                modes = rigid_q48_modes(x16_np)
                coeff = rigid_rng.normal(scale=float(args.rigid_mode_scale), size=(x16_np.shape[0], 6)).astype(np.float32)
                q_rigid = np.einsum("bm,bmj->bj", coeff, modes).astype(np.float32)
                q_mean = branch_mean.reshape(-1)[:48]
                q_scale = branch_std.reshape(-1)[:48]
                rigid_branch = xb.detach().clone()
                rigid_branch[:, :48] = torch.as_tensor((q_rigid - q_mean) / q_scale, dtype=xb.dtype, device=device)
                rigid_pred = model(rigid_branch, pb)
                rigid_zero = torch.as_tensor((0.0 - le_mean) / le_std, dtype=xb.dtype, device=device)
                rigid_loss = nn.functional.mse_loss(rigid_pred, rigid_zero.expand_as(rigid_pred))
            loss = float(args.le_loss_weight) * le_loss + float(args.jacobian_loss_weight) * j_loss + float(args.rigid_loss_weight) * rigid_loss
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            batch_n = int(xb.shape[0])
            count += batch_n
            sums["loss"] += float(loss.detach().cpu()) * batch_n
            sums["le"] += float(le_loss.detach().cpu()) * batch_n
            sums["j"] += float(j_loss.detach().cpu()) * batch_n
            sums["rigid"] += float(rigid_loss.detach().cpu()) * batch_n
        scheduler.step()
        denom = max(count, 1)
        row: dict[str, Any] = {
            "epoch": int(epoch),
            "loss": sums["loss"] / denom,
            "le_loss_norm_mse": sums["le"] / denom,
            "j_loss_norm_mse": sums["j"] / denom,
            "rigid_loss_norm_mse": sums["rigid"] / denom,
            "lr": float(scheduler.get_last_lr()[0]),
            "seconds": time.time() - t0,
        }
        if epoch == 1 or epoch % int(args.eval_every) == 0 or epoch == int(args.epochs):
            row.update(
                evaluate(
                    model,
                    branch_norm=branch_norm,
                    point_norm=point_norm,
                    le_raw=data.le,
                    b_raw=data.b,
                    j_norm_target=j_norm_target,
                    norms=norms,
                    indices=train_idx[: min(train_idx.size, int(args.max_eval_frames))],
                    columns=eval_columns,
                    batch_size=int(args.eval_batch_size),
                    device=device,
                    prefix="train",
                )
            )
            row.update(
                evaluate(
                    model,
                    branch_norm=branch_norm,
                    point_norm=point_norm,
                    le_raw=data.le,
                    b_raw=data.b,
                    j_norm_target=j_norm_target,
                    norms=norms,
                    indices=val_idx[: min(val_idx.size, int(args.max_eval_frames))],
                    columns=eval_columns,
                    batch_size=int(args.eval_batch_size),
                    device=device,
                    prefix="val",
                )
            )
            row["score"] = float(row.get("val_LE_rel", row["loss"])) + float(row.get("val_AD_B_rel", 0.0))
            if float(row["score"]) < best_score:
                best_score = float(row["score"])
                best_report = dict(row)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "norms": norms,
                        "args": vars(args),
                        "contract_version": MACRO16_CONTRACT_VERSION,
                        "point_meta": data.point_meta,
                        "validation_split": split_meta,
                        "best_report": best_report,
                    },
                    out_dir / "best.pt",
                )
        history.append(row)
        torch.save(
            {
                "model_state": model.state_dict(),
                "norms": norms,
                "args": vars(args),
                "contract_version": MACRO16_CONTRACT_VERSION,
                "point_meta": data.point_meta,
                "validation_split": split_meta,
                "latest_report": row,
                "best_report": best_report,
            },
            out_dir / "latest.pt",
        )
        write_json(out_dir / "latest_metrics.json", row)
        print(json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "standard_operator_contract_version": MACRO16_CONTRACT_VERSION,
        "args": vars(args),
        "history": history,
        "best_score": best_score,
        "best_report": best_report,
        "latest_report": history[-1] if history else None,
        "best_checkpoint": str(out_dir / "best.pt"),
        "latest_checkpoint": str(out_dir / "latest.pt"),
        "model_visible_inputs": ["q48_hat", "X16_hat", "macro16_point_features_hat"],
        "q_dim": 48,
        "rigid_modes_removed_from_input": False,
        "fine_grid_geometry_visible": False,
    }
    write_json(out_dir / "training_summary.json", summary)
    return {"best_score": best_score, "out_dir": str(out_dir)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compact", action="append", default=[])
    p.add_argument("--compact-list", default="")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--plane-gauss-order", type=int, default=3)
    p.add_argument("--thickness-gauss-order", type=int, default=2)
    p.add_argument("--scale-mode", default="normalized", choices=["normalized", "physical"])
    p.add_argument("--b-label-coordinate", default="auto", choices=["auto", "physical", "dimensionless"])
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames-per-compact", type=int, default=0)
    p.add_argument("--max-eval-frames", type=int, default=256)
    p.add_argument("--basis-dim", type=int, default=96)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--branch-depth", type=int, default=5)
    p.add_argument("--trunk-depth", type=int, default=5)
    p.add_argument("--activation", default="tanh")
    p.add_argument("--residual-scale", type=float, default=1.0)
    p.add_argument("--fe-baseline-scale", type=float, default=1.0)
    p.add_argument("--freeze-fe-point-baseline", action="store_true")
    p.add_argument("--freeze-skip", action="store_true")
    p.add_argument("--global-b-prior", action="store_true")
    p.add_argument("--anchored-residual-gate-q0", type=float, default=0.0)
    p.add_argument("--le-loss-weight", type=float, default=1.0)
    p.add_argument("--jacobian-loss-weight", type=float, default=1.0)
    p.add_argument("--rigid-loss-weight", type=float, default=0.1)
    p.add_argument("--rigid-mode-scale", type=float, default=0.1)
    p.add_argument("--jacobian-columns", default="all")
    p.add_argument("--jacobian-columns-per-batch", type=int, default=8)
    p.add_argument("--eval-columns", default="0,1,2,3,4,5,6,7,8,9,10,11")
    p.add_argument("--lr", type=float, default=8.0e-5)
    p.add_argument("--lr-decay", type=float, default=0.9995)
    p.add_argument("--weight-decay", type=float, default=1.0e-5)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--val-cases", default="")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260625)
    p.add_argument("--cuda", action="store_true")
    return p.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
