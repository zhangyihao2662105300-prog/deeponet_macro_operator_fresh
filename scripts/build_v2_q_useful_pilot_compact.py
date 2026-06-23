#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a one-case v2a q-useful pilot compact.

This script is a coordinate-contract pilot, not a training tool.  It reads one
strict v1.x complete compact, builds a linear rigid-body projection

    q_useful = T_q_raw_to_useful @ q48_raw

from the q48 control-node reference coordinates, and writes a copied pilot NPZ
with v2a coordinate metadata and B chain-rule audit fields.

The v2a pilot keeps the output strain coordinate as Abaqus global LE:

    T_eps_to_abq = I_6
    B_useful_abq = B_raw @ T_q_raw_to_useful.T
    B_raw_hat    = B_useful_abq @ T_q_raw_to_useful

It does not create standard/local strain labels and does not train a model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


X_Q48_KEYS = (
    "X_keep_ref",
    "X_keep",
    "keep_node_coords",
    "coords48",
    "coords48_for_321",
)


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


def scalar_text(value: Any) -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def find_key(files: list[str], candidates: tuple[str, ...]) -> str | None:
    available = set(files)
    for key in candidates:
        if key in available:
            return key
    return None


def load_x_q48(z: np.lib.npyio.NpzFile, source: str) -> tuple[np.ndarray, str]:
    if source:
        key = str(source)
        if key not in z.files:
            raise KeyError(f"requested q48 node coordinate source {key!r} is not present in compact")
    else:
        key = find_key(list(z.files), X_Q48_KEYS)
        if key is None:
            raise KeyError(
                "cannot find q48 control-node coordinates. Expected one of "
                f"{list(X_Q48_KEYS)}; do not guess q48 node order."
            )
    vals = np.asarray(z[key], dtype=np.float64)
    if vals.shape == (16, 3):
        return vals.copy(), key
    if vals.ndim == 3 and vals.shape[1:] == (16, 3):
        first = vals[0].copy()
        if not np.allclose(vals, first.reshape(1, 16, 3), rtol=0.0, atol=1.0e-8):
            raise ValueError(
                f"{key}: frame-dependent q48 node coordinates are not supported by this v2a pilot; "
                "provide a single reference [16,3] field."
            )
        return first, key
    raise ValueError(f"{key}: expected [16,3] or [N,16,3], got {vals.shape}")


def rigid_modes_from_nodes(x_q48: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(x_q48, dtype=np.float64).reshape(16, 3)
    centroid = np.mean(x, axis=0)
    rel = x - centroid.reshape(1, 3)
    modes = np.zeros((48, 6), dtype=np.float64)

    # translations
    for node in range(16):
        base = 3 * node
        modes[base + 0, 0] = 1.0
        modes[base + 1, 1] = 1.0
        modes[base + 2, 2] = 1.0

    axes = np.eye(3, dtype=np.float64)
    for axis_i, omega in enumerate(axes):
        disp = np.cross(np.broadcast_to(omega.reshape(1, 3), rel.shape), rel)
        modes[:, 3 + axis_i] = disp.reshape(48)

    u, s, _vh = np.linalg.svd(modes, full_matrices=False)
    rank = int(np.sum(s > max(float(s[0]) * 1.0e-10, 1.0e-12))) if s.size else 0
    if rank != 6:
        raise ValueError(f"rigid mode matrix rank must be 6, got {rank}; singular_values={s.tolist()}")
    q_rigid = u[:, :6]
    return q_rigid, {
        "centroid": centroid.tolist(),
        "rigid_mode_rank": rank,
        "rigid_mode_singular_values": s.tolist(),
        "rigid_mode_definition": "columns=[translation_x,y,z, small_rotation_x,y,z], rotation displacement u=omega_cross_(X-centroid)",
    }


def useful_basis_from_rigid(q_rigid: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    # Complete QR on the rigid subspace gives the orthogonal complement in the
    # remaining columns.  N has shape [48,42].
    q_full, _r = np.linalg.qr(np.asarray(q_rigid, dtype=np.float64), mode="complete")
    rigid_basis = q_full[:, :6]
    n_useful = q_full[:, 6:]
    if n_useful.shape != (48, 42):
        raise ValueError(f"expected useful basis [48,42], got {n_useful.shape}")
    orthonormal_max = float(np.max(np.abs(n_useful.T @ n_useful - np.eye(42))))
    rigid_annihilation_max = float(np.max(np.abs(n_useful.T @ rigid_basis)))
    return n_useful, {
        "useful_basis_shape": list(n_useful.shape),
        "useful_basis_orthonormal_max": orthonormal_max,
        "rigid_annihilation_max": rigid_annihilation_max,
    }


def rel_norm(num: np.ndarray, den: np.ndarray) -> float:
    n = float(np.linalg.norm(np.asarray(num, dtype=np.float64).reshape(-1)))
    d = float(np.linalg.norm(np.asarray(den, dtype=np.float64).reshape(-1)))
    return n / max(d, 1.0e-30)


def build_payload(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    compact_path = Path(args.compact).resolve()
    if not compact_path.exists():
        raise FileNotFoundError(compact_path)

    with np.load(str(compact_path), allow_pickle=True) as z:
        required = {"q48_raw", "B_LE128_forward"}
        missing = sorted(required.difference(z.files))
        if missing:
            raise KeyError(f"{compact_path}: missing required fields {missing}")
        payload = {key: np.asarray(z[key]) for key in z.files}
        q48 = np.asarray(z["q48_raw"], dtype=np.float64)
        b_raw = np.asarray(z["B_LE128_forward"], dtype=np.float64)
        if q48.ndim != 2 or q48.shape[1] != 48:
            raise ValueError(f"q48_raw must be [N,48], got {q48.shape}")
        if b_raw.ndim != 4 or b_raw.shape[0] != q48.shape[0] or b_raw.shape[-1] != 48:
            raise ValueError(f"B_LE128_forward must be [N,P,6,48] aligned to q48, got {b_raw.shape}")

        x_q48, x_source = load_x_q48(z, str(args.q48_node_coordinate_source).strip())
        keep_nodes = np.asarray(z["keep_nodes"], dtype=np.int64).reshape(-1) if "keep_nodes" in z.files else None
        keep_node_labels = (
            np.asarray(z["keep_node_labels"], dtype=np.int64).reshape(-1) if "keep_node_labels" in z.files else None
        )

    q_rigid, rigid_meta = rigid_modes_from_nodes(x_q48)
    n_useful, useful_meta = useful_basis_from_rigid(q_rigid)
    t_q = n_useful.T
    p_useful = n_useful @ n_useful.T

    q_useful = q48 @ t_q.T
    q_projected = q48 @ p_useful
    q_removed = q48 - q_projected
    q_useful_reconstruction_rel = rel_norm(q_projected - q48, q48)
    q_useful_removed_rigid_rel = rel_norm(q_removed, q48)

    b_useful = np.einsum("npaj,jk->npak", b_raw, n_useful)
    b_raw_hat = np.einsum("npak,kj->npaj", b_useful, t_q)
    b_projected = np.einsum("npaj,jk->npak", b_raw, p_useful)
    b_rigid = b_raw - b_projected
    b_chain_rule_raw_rel = rel_norm(b_raw_hat - b_raw, b_raw)
    b_chain_rule_projected_rel = rel_norm(b_raw_hat - b_projected, b_projected)
    b_rigid_residual_rel = rel_norm(b_rigid, b_raw)
    b_chain_rule_raw_max_abs = float(np.max(np.abs(b_raw_hat - b_raw)))
    b_chain_rule_projected_max_abs = float(np.max(np.abs(b_raw_hat - b_projected)))

    t_eps = np.eye(6, dtype=np.float64)
    chain_rule = "B_raw_hat = T_eps_to_abq @ B_useful_abq @ T_q_raw_to_useful"
    q_definition = (
        "v2a linear small-rotation rigid-body projection built from q48 reference "
        "control-node coordinates; q_useful = T_q_raw_to_useful @ q48_raw."
    )
    q_local_frame = (
        "Abaqus global q48 control-node displacement components; rigid modes are "
        "translations and small rotations about the X_q48 centroid. No Kabsch/Procrustes per-frame fit."
    )
    b_meta = {
        "B_raw_key": "B_LE128_forward",
        "B_useful_key": "B_useful_abq",
        "B_raw_shape": list(b_raw.shape),
        "B_useful_shape": list(b_useful.shape),
        "T_q_shape": list(t_q.shape),
        "T_eps_to_abq_shape": [6, 6],
        "B_chain_rule_raw_rel": b_chain_rule_raw_rel,
        "B_chain_rule_projected_rel": b_chain_rule_projected_rel,
        "B_rigid_residual_rel": b_rigid_residual_rel,
        "interpretation": (
            "B_chain_rule_projected_rel checks implementation of B_useful_abq @ T_q. "
            "B_chain_rule_raw_rel equals the raw-space loss caused by removing rigid directions. "
            "If B_rigid_residual_rel is large, raw B has non-negligible rigid-mode response."
        ),
    }

    payload.update(
        {
            "q_useful": q_useful.astype(np.float32),
            "T_q_raw_to_useful": t_q.astype(np.float64),
            "P_q_useful": p_useful.astype(np.float64),
            "q_rigid_modes": q_rigid.astype(np.float64),
            "X_q48_ref": x_q48.astype(np.float64),
            "X_q48_ref_source": np.asarray(x_source, dtype=object),
            "q_useful_coordinate": np.asarray("local_rigid_projected", dtype=object),
            "q_useful_rank": np.asarray(42, dtype=np.int64),
            "q_useful_definition": np.asarray(q_definition, dtype=object),
            "q_useful_local_frame": np.asarray(q_local_frame, dtype=object),
            "q_useful_rigid_mode_rank": np.asarray(rigid_meta["rigid_mode_rank"], dtype=np.int64),
            "q_useful_rigid_mode_singular_values": np.asarray(rigid_meta["rigid_mode_singular_values"], dtype=np.float64),
            "q_useful_rigid_annihilation_max": np.asarray(useful_meta["rigid_annihilation_max"], dtype=np.float64),
            "q_useful_basis_orthonormal_max": np.asarray(useful_meta["useful_basis_orthonormal_max"], dtype=np.float64),
            "q_useful_reconstruction_rel": np.asarray(q_useful_reconstruction_rel, dtype=np.float64),
            "q_useful_removed_rigid_rel": np.asarray(q_useful_removed_rigid_rel, dtype=np.float64),
            "strain_output_coordinate": np.asarray("abaqus_global", dtype=object),
            "T_eps_to_abq": t_eps,
            "T_eps_from_abq": t_eps,
            "strain_transform_note": np.asarray(
                "v2a pilot keeps Abaqus global LE; no standard/local strain transform yet.",
                dtype=object,
            ),
            "B_label_q_coordinate": np.asarray("q48_raw", dtype=object),
            "B_label_output_coordinate": np.asarray("abaqus_global", dtype=object),
            "B_chain_rule": np.asarray(chain_rule, dtype=object),
            "B_chain_rule_metadata": np.asarray(json.dumps(b_meta, sort_keys=True), dtype=object),
            "B_useful_abq": b_useful.astype(np.float32),
            "B_chain_rule_raw_rel": np.asarray(b_chain_rule_raw_rel, dtype=np.float64),
            "B_chain_rule_projected_rel": np.asarray(b_chain_rule_projected_rel, dtype=np.float64),
            "B_rigid_residual_rel": np.asarray(b_rigid_residual_rel, dtype=np.float64),
            "B_chain_rule_raw_max_abs": np.asarray(b_chain_rule_raw_max_abs, dtype=np.float64),
            "B_chain_rule_projected_max_abs": np.asarray(b_chain_rule_projected_max_abs, dtype=np.float64),
            "v2a_pilot": np.asarray(True, dtype=np.bool_),
            "v2a_pilot_source_compact": np.asarray(str(compact_path), dtype=object),
            "uses_old_true176_labels_as_v2_labels": np.asarray(False, dtype=np.bool_),
            "standard_or_local_strain_labels_generated": np.asarray(False, dtype=np.bool_),
        }
    )
    if args.case_id is not None:
        payload["v2a_pilot_case_id"] = np.asarray(int(args.case_id), dtype=np.int64)
    if keep_nodes is not None:
        payload["q48_keep_nodes_source"] = np.asarray("keep_nodes", dtype=object)
        payload["q48_keep_nodes"] = keep_nodes
    elif keep_node_labels is not None:
        payload["q48_keep_nodes_source"] = np.asarray("keep_node_labels", dtype=object)
        payload["q48_keep_nodes"] = keep_node_labels

    summary = {
        "audit_name": "v2a_q_useful_pilot",
        "source_compact": str(compact_path),
        "case_id": int(args.case_id) if args.case_id is not None else None,
        "q48_node_coordinate_source": x_source,
        "q48_shape": list(q48.shape),
        "q_useful_shape": list(q_useful.shape),
        "T_q_raw_to_useful_shape": list(t_q.shape),
        "B_raw_shape": list(b_raw.shape),
        "B_useful_abq_shape": list(b_useful.shape),
        "rigid_mode_rank": rigid_meta["rigid_mode_rank"],
        "rigid_mode_singular_values": rigid_meta["rigid_mode_singular_values"],
        "rigid_annihilation_max": useful_meta["rigid_annihilation_max"],
        "useful_basis_orthonormal_max": useful_meta["useful_basis_orthonormal_max"],
        "q_useful_reconstruction_rel": q_useful_reconstruction_rel,
        "q_useful_removed_rigid_rel": q_useful_removed_rigid_rel,
        "B_chain_rule_raw_rel": b_chain_rule_raw_rel,
        "B_chain_rule_projected_rel": b_chain_rule_projected_rel,
        "B_rigid_residual_rel": b_rigid_residual_rel,
        "B_chain_rule_raw_max_abs": b_chain_rule_raw_max_abs,
        "B_chain_rule_projected_max_abs": b_chain_rule_projected_max_abs,
        "strain_output_coordinate": "abaqus_global",
        "standard_or_local_strain_labels_generated": False,
        "model_training_performed": False,
        "uses_old_true176_labels_as_v2_labels": False,
        "notes": [
            "q_useful_reconstruction_rel is the fraction of q48 removed by the useful projection.",
            "B_chain_rule_projected_rel should be near zero if the B_useful_abq @ T_q implementation is correct.",
            "B_chain_rule_raw_rel tracks the B residual in removed rigid directions.",
        ],
    }
    return payload, summary


def write_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", required=True, help="Input strict v1.x complete compact")
    parser.add_argument("--out", required=True, help="Output v2a pilot compact NPZ")
    parser.add_argument("--case-id", type=int, default=None)
    parser.add_argument(
        "--q48-node-coordinate-source",
        default="",
        help="Compact field to use for q48 control-node reference coordinates; defaults to X_keep_ref/X_keep/etc.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if key v2a roundtrip checks are not finite/sane")
    args = parser.parse_args()

    payload, summary = build_payload(args)
    if args.strict:
        if summary["rigid_mode_rank"] != 6:
            raise SystemExit("rigid_mode_rank is not 6")
        if not math.isfinite(float(summary["rigid_annihilation_max"])) or float(summary["rigid_annihilation_max"]) > 1.0e-10:
            raise SystemExit(f"rigid annihilation too large: {summary['rigid_annihilation_max']}")
        if not math.isfinite(float(summary["B_chain_rule_projected_rel"])) or float(summary["B_chain_rule_projected_rel"]) > 1.0e-6:
            raise SystemExit(f"B projected chain-rule error too large: {summary['B_chain_rule_projected_rel']}")

    out = Path(args.out).resolve()
    write_npz(out, payload)
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    print(json.dumps({**summary, "out": str(out), "summary_path": str(summary_path)}, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
