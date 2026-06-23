#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a one-case v2b local-strain pilot compact.

This script extends a v2a q-useful pilot compact with a local orthonormal strain
coordinate at each integration point:

    E_local = Q.T @ E_abq @ Q
    E_abq   = Q @ E_local @ Q.T

where Q columns are local basis vectors represented in Abaqus/global
coordinates.  The 6x6 Voigt transforms are generated numerically from basis
tensors to avoid hand-coded shear mistakes.

The script does not train a model and does not overwrite the input compact.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


VOIGT_ORDER = "LE11_LE22_LE33_LE12_LE13_LE23"
SHEAR_CONVENTION = "tensor_shear_not_engineering_gamma"


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


def voigt_to_tensor(v: np.ndarray) -> np.ndarray:
    vals = np.asarray(v, dtype=np.float64).reshape(6)
    out = np.zeros((3, 3), dtype=np.float64)
    out[0, 0] = vals[0]
    out[1, 1] = vals[1]
    out[2, 2] = vals[2]
    out[0, 1] = out[1, 0] = vals[3]
    out[0, 2] = out[2, 0] = vals[4]
    out[1, 2] = out[2, 1] = vals[5]
    return out


def tensor_to_voigt(tensor: np.ndarray) -> np.ndarray:
    t = np.asarray(tensor, dtype=np.float64).reshape(3, 3)
    return np.asarray([t[0, 0], t[1, 1], t[2, 2], t[0, 1], t[0, 2], t[1, 2]], dtype=np.float64)


def normalize(vec: np.ndarray, eps: float) -> tuple[np.ndarray, float]:
    vals = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vals))
    if not math.isfinite(norm) or norm <= float(eps):
        raise ValueError(f"degenerate vector norm {norm:g}")
    return vals / norm, norm


def local_frame_from_j(jmat: np.ndarray, *, convention: str, eps: float) -> tuple[np.ndarray, float]:
    j = np.asarray(jmat, dtype=np.float64).reshape(3, 3)
    key = str(convention).strip().lower().replace("_", "-")
    if key in {"rows", "row", "auto-rows", "ip-j-rows"}:
        g1, g2, g3 = j[0], j[1], j[2]
        source = "ip_J_rows"
    elif key in {"cols", "columns", "column", "ip-j-columns"}:
        g1, g2, g3 = j[:, 0], j[:, 1], j[:, 2]
        source = "ip_J_columns"
    else:
        raise ValueError("ip_j_axis_convention must be rows or columns")

    e1, _n1 = normalize(g1, eps)
    g2_orth = g2 - float(np.dot(g2, e1)) * e1
    e2, _n2 = normalize(g2_orth, eps)
    e3 = np.cross(e1, e2)
    e3, _n3 = normalize(e3, eps)
    orientation = float(np.dot(e3, g3) / max(float(np.linalg.norm(g3)), eps))
    if orientation < 0.0:
        e2 = -e2
        e3 = np.cross(e1, e2)
        e3, _n3 = normalize(e3, eps)
        orientation = float(np.dot(e3, g3) / max(float(np.linalg.norm(g3)), eps))

    q = np.stack([e1, e2, e3], axis=1)
    # Attach lightweight provenance for callers that want it without changing
    # the return type too much.
    _ = source
    return q, orientation


def build_local_frames(ip_j: np.ndarray, *, convention: str, eps: float) -> tuple[np.ndarray, dict[str, Any]]:
    vals = np.asarray(ip_j, dtype=np.float64)
    if vals.shape == (128, 3, 3):
        frame_shape = (128,)
        flat = vals.reshape(128, 3, 3)
        out_shape = (128, 3, 3)
    elif vals.ndim == 4 and vals.shape[1:] == (128, 3, 3):
        frame_shape = (vals.shape[0], 128)
        flat = vals.reshape(-1, 3, 3)
        out_shape = (vals.shape[0], 128, 3, 3)
    else:
        raise ValueError(f"ip_J must be [128,3,3] or [N,128,3,3], got {vals.shape}")

    frames = []
    orientations = []
    for jmat in flat:
        q, orient = local_frame_from_j(jmat, convention=convention, eps=eps)
        frames.append(q)
        orientations.append(orient)
    q_all = np.asarray(frames, dtype=np.float64).reshape(out_shape)
    orient_arr = np.asarray(orientations, dtype=np.float64).reshape(frame_shape)
    qtq = np.einsum("...ia,...ib->...ab", q_all, q_all)
    det = np.linalg.det(q_all)
    meta = {
        "local_frame_orthonormal_max": float(np.max(np.abs(qtq - np.eye(3)))),
        "local_frame_det_min": float(np.min(det)),
        "local_frame_det_max": float(np.max(det)),
        "local_frame_orientation_min": float(np.min(orient_arr)),
        "local_frame_orientation_max": float(np.max(orient_arr)),
    }
    return q_all, meta


def strain_transform_matrices(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frames = np.asarray(q, dtype=np.float64)
    if frames.shape[-2:] != (3, 3):
        raise ValueError(f"local frame Q must end with [3,3], got {frames.shape}")
    out_shape = frames.shape[:-2] + (6, 6)
    t_from = np.empty(out_shape, dtype=np.float64)
    t_to = np.empty(out_shape, dtype=np.float64)
    basis = np.eye(6, dtype=np.float64)

    for idx in np.ndindex(frames.shape[:-2]):
        qi = frames[idx]
        for col in range(6):
            e_global = voigt_to_tensor(basis[col])
            e_local = qi.T @ e_global @ qi
            t_from[idx + (slice(None), col)] = tensor_to_voigt(e_local)

            basis_local = voigt_to_tensor(basis[col])
            back = qi @ basis_local @ qi.T
            t_to[idx + (slice(None), col)] = tensor_to_voigt(back)
    return t_from, t_to


def rel_norm(num: np.ndarray, den: np.ndarray) -> float:
    n = float(np.linalg.norm(np.asarray(num, dtype=np.float64).reshape(-1)))
    d = float(np.linalg.norm(np.asarray(den, dtype=np.float64).reshape(-1)))
    return n / max(d, 1.0e-30)


def max_abs(value: np.ndarray) -> float:
    vals = np.asarray(value, dtype=np.float64)
    return float(np.max(np.abs(vals))) if vals.size else 0.0


def broadcast_ip_matrix(mat: np.ndarray, n_frames: int) -> np.ndarray:
    vals = np.asarray(mat, dtype=np.float64)
    if vals.shape == (128, 6, 6):
        return np.broadcast_to(vals.reshape(1, 128, 6, 6), (int(n_frames), 128, 6, 6)).copy()
    if vals.shape == (int(n_frames), 128, 6, 6):
        return vals.copy()
    raise ValueError(f"expected transform [128,6,6] or [N,128,6,6], got {vals.shape}")


def apply_strain_transform(t: np.ndarray, le: np.ndarray) -> np.ndarray:
    tt = broadcast_ip_matrix(t, int(le.shape[0]))
    return np.einsum("npab,npb->npa", tt, np.asarray(le, dtype=np.float64))


def apply_b_transform(t: np.ndarray, b: np.ndarray) -> np.ndarray:
    tt = broadcast_ip_matrix(t, int(b.shape[0]))
    return np.einsum("npab,npbk->npak", tt, np.asarray(b, dtype=np.float64))


def build_payload(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    compact_path = Path(args.compact).resolve()
    if not compact_path.exists():
        raise FileNotFoundError(compact_path)
    with np.load(str(compact_path), allow_pickle=True) as z:
        required = {"LE128_base", "B_LE128_forward", "B_useful_abq", "T_q_raw_to_useful", "ip_J"}
        missing = sorted(required.difference(z.files))
        if missing:
            raise KeyError(f"{compact_path}: missing required v2b inputs {missing}")
        payload = {key: np.asarray(z[key]) for key in z.files}
        le_abq = np.asarray(z["LE128_base"], dtype=np.float64)
        b_raw = np.asarray(z["B_LE128_forward"], dtype=np.float64)
        b_useful_abq = np.asarray(z["B_useful_abq"], dtype=np.float64)
        t_q = np.asarray(z["T_q_raw_to_useful"], dtype=np.float64)
        ip_j = np.asarray(z["ip_J"], dtype=np.float64)
        ip_j_convention = scalar_text(z["ip_J_convention"]) if "ip_J_convention" in z.files else ""

    if le_abq.ndim != 3 or le_abq.shape[1:] != (128, 6):
        raise ValueError(f"LE128_base must be [N,128,6], got {le_abq.shape}")
    if b_raw.shape != (le_abq.shape[0], 128, 6, 48):
        raise ValueError(f"B_LE128_forward must be [N,128,6,48], got {b_raw.shape}")
    if b_useful_abq.shape[:3] != (le_abq.shape[0], 128, 6):
        raise ValueError(f"B_useful_abq must align to LE frames/IP/components, got {b_useful_abq.shape}")
    if t_q.shape != (b_useful_abq.shape[-1], 48):
        raise ValueError(f"T_q_raw_to_useful shape {t_q.shape} does not match B_useful_abq K={b_useful_abq.shape[-1]}")

    convention = str(args.ip_j_axis_convention).strip().lower()
    if convention == "auto":
        # Current exporter metadata says rows are physical derivatives wrt
        # local natural coordinates.  Use that when present; otherwise default
        # to rows and record the choice.
        convention = "rows"
        if "columns" in ip_j_convention.lower() or "cols" in ip_j_convention.lower():
            convention = "columns"

    local_q, frame_meta = build_local_frames(ip_j, convention=convention, eps=float(args.degenerate_tol))
    t_from, t_to = strain_transform_matrices(local_q)
    t_from_full = broadcast_ip_matrix(t_from, int(le_abq.shape[0]))
    t_to_full = broadcast_ip_matrix(t_to, int(le_abq.shape[0]))

    eye6 = np.eye(6, dtype=np.float64)
    roundtrip_local = np.einsum("npab,npbc->npac", t_from_full, t_to_full)
    roundtrip_abq = np.einsum("npab,npbc->npac", t_to_full, t_from_full)
    t_eps_roundtrip_local_rel = rel_norm(roundtrip_local - eye6.reshape(1, 1, 6, 6), eye6)
    t_eps_roundtrip_abq_rel = rel_norm(roundtrip_abq - eye6.reshape(1, 1, 6, 6), eye6)

    le_local = apply_strain_transform(t_from, le_abq)
    le_abq_hat = apply_strain_transform(t_to, le_local)
    le_local_roundtrip_rel = rel_norm(le_abq_hat - le_abq, le_abq)
    le_local_roundtrip_max_abs = max_abs(le_abq_hat - le_abq)

    b_local_useful = apply_b_transform(t_from, b_useful_abq)
    b_abq_from_local = apply_b_transform(t_to, b_local_useful)
    b_raw_hat = np.einsum("npak,kj->npaj", b_abq_from_local, t_q)
    p_useful = t_q.T @ t_q
    b_raw_projected = np.einsum("npaj,jk->npak", b_raw, p_useful)

    b_local_chain_rule_projected_rel = rel_norm(b_raw_hat - b_raw_projected, b_raw_projected)
    b_local_chain_rule_raw_rel = rel_norm(b_raw_hat - b_raw, b_raw)
    b_local_chain_rule_projected_max_abs = max_abs(b_raw_hat - b_raw_projected)
    b_local_chain_rule_raw_max_abs = max_abs(b_raw_hat - b_raw)

    chain_meta = {
        "B_raw_key": "B_LE128_forward",
        "B_input_useful_key": "B_useful_abq",
        "B_local_useful_key": "B_local_useful",
        "B_standard_useful_key": "B_standard_useful",
        "T_q_shape": list(t_q.shape),
        "T_eps_to_abq_shape": list(t_to.shape),
        "T_eps_from_abq_shape": list(t_from.shape),
        "B_local_chain_rule_projected_rel": b_local_chain_rule_projected_rel,
        "B_local_chain_rule_raw_rel": b_local_chain_rule_raw_rel,
        "strain_coordinate": "local_jacobian_frame",
    }
    transform_note = (
        "v2b pilot uses local orthonormal tensor component transform from Abaqus global LE. "
        "Assumes LE components are global tensor components in order [11,22,33,12,13,23]; "
        "off-diagonal components are tensor shear, not engineering gamma."
    )
    frame_definition = (
        "Q columns are local basis vectors in Abaqus/global coordinates. "
        "Basis is built by Gram-Schmidt from ip_J rows/columns according to local_frame_ip_J_axis_convention."
    )

    payload.update(
        {
            "local_frame_Q": local_q.astype(np.float64),
            "local_frame_source": np.asarray("ip_J_gram_schmidt", dtype=object),
            "local_frame_ip_J_axis_convention": np.asarray(convention, dtype=object),
            "local_frame_definition": np.asarray(frame_definition, dtype=object),
            "local_frame_orthonormal_max": np.asarray(frame_meta["local_frame_orthonormal_max"], dtype=np.float64),
            "local_frame_det_min": np.asarray(frame_meta["local_frame_det_min"], dtype=np.float64),
            "local_frame_det_max": np.asarray(frame_meta["local_frame_det_max"], dtype=np.float64),
            "local_frame_orientation_min": np.asarray(frame_meta["local_frame_orientation_min"], dtype=np.float64),
            "local_frame_orientation_max": np.asarray(frame_meta["local_frame_orientation_max"], dtype=np.float64),
            "LE128_local": le_local.astype(np.float32),
            "strain_output_coordinate": np.asarray("local_jacobian_frame", dtype=object),
            "T_eps_to_abq": t_to.astype(np.float64),
            "T_eps_from_abq": t_from.astype(np.float64),
            "strain_voigt_order": np.asarray(VOIGT_ORDER, dtype=object),
            "strain_shear_convention": np.asarray(SHEAR_CONVENTION, dtype=object),
            "strain_transform_note": np.asarray(transform_note, dtype=object),
            "B_local_useful": b_local_useful.astype(np.float64),
            "B_standard_useful": b_local_useful.astype(np.float64),
            "B_standard_useful_coordinate": np.asarray("local_jacobian_frame", dtype=object),
            "B_standard_useful_note": np.asarray(
                "v2b local strain coordinate, not covariant standard-coordinate strain.",
                dtype=object,
            ),
            "B_label_q_coordinate": np.asarray("q_useful", dtype=object),
            "B_label_output_coordinate": np.asarray("local_jacobian_frame", dtype=object),
            "B_chain_rule": np.asarray(
                "B_raw_hat = T_eps_to_abq @ B_local_useful @ T_q_raw_to_useful",
                dtype=object,
            ),
            "B_chain_rule_metadata": np.asarray(json.dumps(chain_meta, sort_keys=True), dtype=object),
            "T_eps_roundtrip_local_rel": np.asarray(t_eps_roundtrip_local_rel, dtype=np.float64),
            "T_eps_roundtrip_abq_rel": np.asarray(t_eps_roundtrip_abq_rel, dtype=np.float64),
            "LE_local_roundtrip_rel": np.asarray(le_local_roundtrip_rel, dtype=np.float64),
            "LE_local_roundtrip_max_abs": np.asarray(le_local_roundtrip_max_abs, dtype=np.float64),
            "B_local_chain_rule_projected_rel": np.asarray(b_local_chain_rule_projected_rel, dtype=np.float64),
            "B_local_chain_rule_raw_rel": np.asarray(b_local_chain_rule_raw_rel, dtype=np.float64),
            "B_local_chain_rule_projected_max_abs": np.asarray(b_local_chain_rule_projected_max_abs, dtype=np.float64),
            "B_local_chain_rule_raw_max_abs": np.asarray(b_local_chain_rule_raw_max_abs, dtype=np.float64),
            "v2b_pilot": np.asarray(True, dtype=np.bool_),
            "v2b_pilot_source_compact": np.asarray(str(compact_path), dtype=object),
            "standard_or_local_strain_labels_generated": np.asarray(True, dtype=np.bool_),
            "uses_old_true176_labels_as_v2_labels": np.asarray(False, dtype=np.bool_),
        }
    )
    if args.case_id is not None:
        payload["v2b_pilot_case_id"] = np.asarray(int(args.case_id), dtype=np.int64)

    summary = {
        "audit_name": "v2b_local_strain_pilot",
        "source_compact": str(compact_path),
        "case_id": int(args.case_id) if args.case_id is not None else None,
        "local_frame_Q_shape": list(local_q.shape),
        "local_frame_source": "ip_J_gram_schmidt",
        "local_frame_ip_J_axis_convention": convention,
        **frame_meta,
        "T_eps_to_abq_shape": list(t_to.shape),
        "T_eps_from_abq_shape": list(t_from.shape),
        "T_eps_roundtrip_local_rel": t_eps_roundtrip_local_rel,
        "T_eps_roundtrip_abq_rel": t_eps_roundtrip_abq_rel,
        "LE128_local_shape": list(le_local.shape),
        "LE_local_roundtrip_rel": le_local_roundtrip_rel,
        "LE_local_roundtrip_max_abs": le_local_roundtrip_max_abs,
        "B_local_useful_shape": list(b_local_useful.shape),
        "B_local_chain_rule_projected_rel": b_local_chain_rule_projected_rel,
        "B_local_chain_rule_raw_rel": b_local_chain_rule_raw_rel,
        "B_local_chain_rule_projected_max_abs": b_local_chain_rule_projected_max_abs,
        "B_local_chain_rule_raw_max_abs": b_local_chain_rule_raw_max_abs,
        "strain_output_coordinate": "local_jacobian_frame",
        "strain_voigt_order": VOIGT_ORDER,
        "strain_shear_convention": SHEAR_CONVENTION,
        "model_training_performed": False,
        "uses_old_true176_labels_as_v2_labels": False,
    }
    return payload, summary


def write_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", required=True, help="Input v2a q-useful pilot compact")
    parser.add_argument("--out", required=True, help="Output v2b local-strain pilot compact NPZ")
    parser.add_argument("--case-id", type=int, default=None)
    parser.add_argument(
        "--ip-j-axis-convention",
        default="auto",
        choices=("auto", "rows", "columns"),
        help="Use rows or columns of ip_J as g1,g2,g3. auto follows compact metadata and defaults to rows.",
    )
    parser.add_argument("--degenerate-tol", type=float, default=1.0e-12)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    payload, summary = build_payload(args)
    if args.strict:
        checks = {
            "local_frame_orthonormal_max": 1.0e-10,
            "T_eps_roundtrip_local_rel": 1.0e-10,
            "T_eps_roundtrip_abq_rel": 1.0e-10,
            "LE_local_roundtrip_rel": 1.0e-10,
            "B_local_chain_rule_projected_rel": 1.0e-8,
        }
        for key, tol in checks.items():
            val = float(summary[key])
            if not math.isfinite(val) or val > float(tol):
                raise SystemExit(f"{key}={val:g} exceeds strict tolerance {tol:g}")
        if float(summary["local_frame_det_min"]) < 1.0 - 1.0e-10:
            raise SystemExit(f"local_frame_det_min={summary['local_frame_det_min']} is not right-handed")

    out = Path(args.out).resolve()
    write_npz(out, payload)
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=json_default), encoding="utf-8")
    print(json.dumps({**summary, "out": str(out), "summary_path": str(summary_path)}, indent=2, ensure_ascii=False, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
