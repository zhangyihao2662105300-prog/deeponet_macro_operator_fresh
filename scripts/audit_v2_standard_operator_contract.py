#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit a v2 standard-operator compact.

The strict checks verify that preprocessing/postprocessing transforms close:

    q_useful = q48_raw @ T_q_raw_to_useful.T
    LE_abq   = T_eps_to_abq @ LE_local
    B_raw projected useful = T_eps_to_abq @ B_local_useful @ T_q_raw_to_useful

The removed rigid-mode residual is reported but is not a strict failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_MODEL_VISIBLE = {
    "q_useful": (None, 42),
    "ip_xi": (128, 3),
    "LE_local": (None, 128, 6),
    "B_local_useful": (None, 128, 6, 42),
}
REQUIRED_AUDIT = ("q48_raw", "B_LE128_forward", "T_q_raw_to_useful", "T_eps_to_abq")


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


def scalar_text(value: Any, default: str = "unknown") -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return str(default)
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def rel_norm(diff: np.ndarray, ref: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(ref.reshape(-1))), 1.0e-30)
    return float(np.linalg.norm(diff.reshape(-1)) / denom)


def apply_t_q(q_raw: np.ndarray, t_q: np.ndarray) -> np.ndarray:
    if t_q.shape == (42, 48):
        return q_raw @ t_q.T
    if t_q.ndim == 3 and t_q.shape[0] == q_raw.shape[0] and t_q.shape[1:] == (42, 48):
        return np.einsum("nki,ni->nk", t_q, q_raw)
    raise ValueError(f"T_q_raw_to_useful must be [42,48] or [N,42,48], got {t_q.shape}")


def apply_t_eps_to_strain(t_eps: np.ndarray, strain: np.ndarray) -> np.ndarray:
    if t_eps.shape == (128, 6, 6):
        return np.einsum("pab,npb->npa", t_eps, strain)
    if t_eps.shape == (strain.shape[0], 128, 6, 6):
        return np.einsum("npab,npb->npa", t_eps, strain)
    raise ValueError(f"T_eps_to_abq must be [128,6,6] or [N,128,6,6], got {t_eps.shape}")


def apply_t_eps_to_b(t_eps: np.ndarray, b_local: np.ndarray) -> np.ndarray:
    if t_eps.shape == (128, 6, 6):
        return np.einsum("pab,npbk->npak", t_eps, b_local)
    if t_eps.shape == (b_local.shape[0], 128, 6, 6):
        return np.einsum("npab,npbk->npak", t_eps, b_local)
    raise ValueError(f"T_eps_to_abq must be [128,6,6] or [N,128,6,6], got {t_eps.shape}")


def useful_to_raw_projected(b_local_useful: np.ndarray, t_eps_to_abq: np.ndarray, t_q: np.ndarray) -> np.ndarray:
    b_abq_useful = apply_t_eps_to_b(t_eps_to_abq, b_local_useful)
    if t_q.shape == (42, 48):
        return np.einsum("npak,kj->npaj", b_abq_useful, t_q)
    if t_q.ndim == 3 and t_q.shape[0] == b_local_useful.shape[0] and t_q.shape[1:] == (42, 48):
        return np.einsum("npak,nkj->npaj", b_abq_useful, t_q)
    raise ValueError(f"T_q_raw_to_useful must be [42,48] or [N,42,48], got {t_q.shape}")


def project_raw_b(b_raw: np.ndarray, t_q: np.ndarray) -> np.ndarray:
    if t_q.shape == (42, 48):
        p_useful = t_q.T @ t_q
        return np.einsum("npaj,jk->npak", b_raw, p_useful)
    if t_q.ndim == 3 and t_q.shape[0] == b_raw.shape[0] and t_q.shape[1:] == (42, 48):
        p_useful = np.einsum("nki,nkj->nij", t_q, t_q)
        return np.einsum("npaj,njk->npak", b_raw, p_useful)
    raise ValueError(f"T_q_raw_to_useful must be [42,48] or [N,42,48], got {t_q.shape}")


def shape_matches(actual: tuple[int, ...], expected: tuple[int | None, ...], n_frames: int | None = None) -> bool:
    if len(actual) != len(expected):
        return False
    for idx, (got, want) in enumerate(zip(actual, expected)):
        if want is None:
            if idx == 0 and n_frames is not None and got != n_frames:
                return False
            continue
        if got != want:
            return False
    return True


def audit(path: Path, *, strict: bool, tol: float) -> dict[str, Any]:
    if not path.exists():
        return {
            "standard_compact": str(path),
            "strict_pass": False,
            "strict_failures": ["standard_compact_missing"],
        }

    failures: list[str] = []
    with np.load(str(path), allow_pickle=True) as z:
        missing = [key for key in list(REQUIRED_MODEL_VISIBLE) + list(REQUIRED_AUDIT) if key not in z.files]
        if missing:
            return {
                "standard_compact": str(path),
                "strict_pass": False,
                "strict_failures": [f"missing_{key}" for key in missing],
            }

        q_useful = np.asarray(z["q_useful"], dtype=np.float64)
        ip_xi = np.asarray(z["ip_xi"], dtype=np.float64)
        le_local = np.asarray(z["LE_local"], dtype=np.float64)
        b_local_useful = np.asarray(z["B_local_useful"], dtype=np.float64)
        q48_raw = np.asarray(z["q48_raw"], dtype=np.float64)
        b_raw = np.asarray(z["B_LE128_forward"], dtype=np.float64)
        t_q = np.asarray(z["T_q_raw_to_useful"], dtype=np.float64)
        t_eps_to_abq = np.asarray(z["T_eps_to_abq"], dtype=np.float64)
        le_abq = np.asarray(z["LE_abq"], dtype=np.float64) if "LE_abq" in z.files else None

        n_frames = int(q_useful.shape[0]) if q_useful.ndim else None
        shapes = {
            "q_useful": list(q_useful.shape),
            "ip_xi": list(ip_xi.shape),
            "LE_local": list(le_local.shape),
            "B_local_useful": list(b_local_useful.shape),
            "q48_raw": list(q48_raw.shape),
            "B_LE128_forward": list(b_raw.shape),
            "T_q_raw_to_useful": list(t_q.shape),
            "T_eps_to_abq": list(t_eps_to_abq.shape),
            "LE_abq": list(le_abq.shape) if le_abq is not None else None,
        }

        if not shape_matches(q_useful.shape, (None, 42), n_frames):
            failures.append(f"q_useful_shape_{q_useful.shape}")
        if not shape_matches(ip_xi.shape, (128, 3), n_frames):
            failures.append(f"ip_xi_shape_{ip_xi.shape}")
        if not shape_matches(le_local.shape, (None, 128, 6), n_frames):
            failures.append(f"LE_local_shape_{le_local.shape}")
        if not shape_matches(b_local_useful.shape, (None, 128, 6, 42), n_frames):
            failures.append(f"B_local_useful_shape_{b_local_useful.shape}")
        if q48_raw.shape != (n_frames, 48):
            failures.append(f"q48_raw_shape_{q48_raw.shape}")
        if b_raw.shape != (n_frames, 128, 6, 48):
            failures.append(f"B_LE128_forward_shape_{b_raw.shape}")

        q_check = apply_t_q(q48_raw, t_q)
        q_diff = q_check - q_useful
        q_rel = rel_norm(q_diff, q_useful)
        q_max_abs = float(np.max(np.abs(q_diff))) if q_diff.size else 0.0

        le_roundtrip_rel = None
        le_roundtrip_max_abs = None
        le_roundtrip_skipped = le_abq is None
        if le_abq is not None:
            le_abq_recovered = apply_t_eps_to_strain(t_eps_to_abq, le_local)
            le_diff = le_abq_recovered - le_abq
            le_roundtrip_rel = rel_norm(le_diff, le_abq)
            le_roundtrip_max_abs = float(np.max(np.abs(le_diff))) if le_diff.size else 0.0

        b_raw_hat_projected = useful_to_raw_projected(b_local_useful, t_eps_to_abq, t_q)
        b_raw_projected = project_raw_b(b_raw, t_q)
        b_projected_diff = b_raw_hat_projected - b_raw_projected
        b_raw_diff = b_raw_hat_projected - b_raw
        b_projected_rel = rel_norm(b_projected_diff, b_raw_projected)
        b_projected_max_abs = float(np.max(np.abs(b_projected_diff))) if b_projected_diff.size else 0.0
        b_rigid_rel = rel_norm(b_raw_diff, b_raw)
        b_rigid_max_abs = float(np.max(np.abs(b_raw_diff))) if b_raw_diff.size else 0.0

        if strict:
            if q_rel >= tol:
                failures.append(f"q_useful_transform_rel_ge_{tol:g}")
            if le_roundtrip_rel is not None and le_roundtrip_rel >= tol:
                failures.append(f"LE_local_to_abq_roundtrip_rel_ge_{tol:g}")
            if b_projected_rel >= tol:
                failures.append(f"B_raw_projected_rel_ge_{tol:g}")

        contract_version = (
            scalar_text(z["standard_operator_contract_version"])
            if "standard_operator_contract_version" in z.files
            else "unknown"
        )
        case_id = int(np.asarray(z["case_id"]).reshape(-1)[0]) if "case_id" in z.files else None

    return {
        "standard_compact": str(path),
        "case_id": case_id,
        "standard_operator_contract_version": contract_version,
        "strict_requested": bool(strict),
        "strict_pass": len(failures) == 0,
        "strict_failures": failures,
        "shapes": shapes,
        "q_useful_transform_rel": q_rel,
        "q_useful_transform_max_abs": q_max_abs,
        "LE_roundtrip_skipped": le_roundtrip_skipped,
        "LE_local_to_abq_roundtrip_rel": le_roundtrip_rel,
        "LE_local_to_abq_roundtrip_max_abs": le_roundtrip_max_abs,
        "B_raw_projected_rel": b_projected_rel,
        "B_raw_projected_max_abs": b_projected_max_abs,
        "B_rigid_residual_rel": b_rigid_rel,
        "B_rigid_residual_max_abs": b_rigid_max_abs,
        "B_rigid_residual_strict_failure": False,
        "trained_model": False,
        "used_old_true176_labels": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-compact", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--tol", type=float, default=1.0e-10)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary = audit(args.standard_compact, strict=args.strict, tol=args.tol)
    summary_path = args.out_root / "standard_operator_audit_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))
    if args.strict and not summary["strict_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
