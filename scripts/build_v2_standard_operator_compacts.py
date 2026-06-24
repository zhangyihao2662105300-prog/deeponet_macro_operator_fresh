#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build v2 standard-operator compacts from coordinate-consistent v2b compacts.

This is a preprocessing/export utility only.  It does not train a model and it
does not use old TRUE176 LE/B labels.  The generated compact separates
model-visible fields from transform/audit fields:

    NN-visible: q_useful, ip_xi, LE_local, B_local_useful
    audit/postprocess: q48_raw, LE_abq, B_LE128_forward, T_q, T_eps
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


CONTRACT_VERSION = "v2-standard-operator-001"
MODEL_VISIBLE_FIELDS = ("q_useful", "ip_xi", "LE_local", "B_local_useful")
AUDIT_POSTPROCESS_FIELDS = (
    "q48_raw",
    "LE_abq",
    "B_LE128_forward",
    "T_q_raw_to_useful",
    "T_eps_to_abq",
    "T_eps_from_abq",
    "local_frame_Q",
    "case_id",
    "metadata",
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


def read_compact_list(path: Path) -> list[Path]:
    return [
        Path(line.strip()).resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def scalar_text(value: Any, default: str = "unknown") -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return str(default)
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def scalar_int(value: Any, default: int | None = None) -> int | None:
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return default
        return int(arr.reshape(-1)[0])
    except Exception:
        return default


def parse_case_id(path: Path, z: np.lib.npyio.NpzFile | None = None) -> int:
    if z is not None:
        for key in ("v2b_pilot_case_id", "v2a_pilot_case_id", "case_id"):
            if key in z.files:
                val = scalar_int(z[key])
                if val is not None:
                    return int(val)
    match = re.search(r"case[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot parse case id from {path}")
    return int(match.group(1))


def first_key(files: list[str], candidates: tuple[str, ...]) -> str | None:
    present = set(files)
    for key in candidates:
        if key in present:
            return key
    return None


def as_float64(z: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    return np.asarray(z[key], dtype=np.float64)


def as_ip_xi(z: np.lib.npyio.NpzFile) -> np.ndarray:
    ip_xi = as_float64(z, "ip_xi")
    if ip_xi.shape == (128, 3):
        return ip_xi
    if ip_xi.ndim == 3 and ip_xi.shape[1:] == (128, 3):
        return ip_xi[0]
    raise ValueError(f"ip_xi must be [128,3] or [N,128,3], got {ip_xi.shape}")


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
    raise ValueError(f"T_eps shape must be [128,6,6] or [N,128,6,6], got {t_eps.shape}")


def apply_t_eps_to_b(t_eps: np.ndarray, b_raw: np.ndarray) -> np.ndarray:
    if t_eps.shape == (128, 6, 6):
        return np.einsum("pab,npbj->npaj", t_eps, b_raw)
    if t_eps.shape == (b_raw.shape[0], 128, 6, 6):
        return np.einsum("npab,npbj->npaj", t_eps, b_raw)
    raise ValueError(f"T_eps shape must be [128,6,6] or [N,128,6,6], got {t_eps.shape}")


def right_multiply_t_q_transpose(b_local_raw: np.ndarray, t_q: np.ndarray) -> np.ndarray:
    if t_q.shape == (42, 48):
        return np.einsum("npaj,kj->npak", b_local_raw, t_q)
    if t_q.ndim == 3 and t_q.shape[0] == b_local_raw.shape[0] and t_q.shape[1:] == (42, 48):
        return np.einsum("npaj,nkj->npak", b_local_raw, t_q)
    raise ValueError(f"T_q_raw_to_useful must be [42,48] or [N,42,48], got {t_q.shape}")


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


def rel_norm(diff: np.ndarray, ref: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(ref.reshape(-1))), 1.0e-30)
    return float(np.linalg.norm(diff.reshape(-1)) / denom)


def audit_arrays(
    *,
    q48_raw: np.ndarray,
    q_useful: np.ndarray,
    le_abq: np.ndarray | None,
    le_local: np.ndarray,
    b_raw: np.ndarray,
    b_local_useful: np.ndarray,
    t_q: np.ndarray,
    t_eps_to_abq: np.ndarray,
    strict: bool,
    tol: float,
) -> dict[str, Any]:
    q_check = apply_t_q(q48_raw, t_q)
    q_diff = q_check - q_useful
    le_roundtrip_rel = None
    le_roundtrip_max_abs = None
    if le_abq is not None:
        le_abq_recovered = apply_t_eps_to_strain(t_eps_to_abq, le_local)
        le_diff = le_abq_recovered - le_abq
        le_roundtrip_rel = rel_norm(le_diff, le_abq)
        le_roundtrip_max_abs = float(np.max(np.abs(le_diff))) if le_diff.size else 0.0

    b_raw_hat_projected = useful_to_raw_projected(b_local_useful, t_eps_to_abq, t_q)
    b_raw_projected = project_raw_b(b_raw, t_q)
    b_projected_diff = b_raw_hat_projected - b_raw_projected
    b_raw_diff = b_raw_hat_projected - b_raw

    failures: list[str] = []
    if q_useful.shape != (q48_raw.shape[0], 42):
        failures.append(f"q_useful_shape_{q_useful.shape}")
    if le_local.shape != (q48_raw.shape[0], 128, 6):
        failures.append(f"LE_local_shape_{le_local.shape}")
    if b_local_useful.shape != (q48_raw.shape[0], 128, 6, 42):
        failures.append(f"B_local_useful_shape_{b_local_useful.shape}")
    q_rel = rel_norm(q_diff, q_useful)
    b_rel = rel_norm(b_projected_diff, b_raw_projected)
    if strict:
        if q_rel >= tol:
            failures.append(f"q_useful_transform_rel_ge_{tol:g}")
        if le_roundtrip_rel is not None and le_roundtrip_rel >= tol:
            failures.append(f"LE_local_to_abq_roundtrip_rel_ge_{tol:g}")
        if b_rel >= tol:
            failures.append(f"B_raw_projected_rel_ge_{tol:g}")

    return {
        "strict_pass": len(failures) == 0,
        "strict_failures": failures,
        "q_useful_transform_rel": q_rel,
        "q_useful_transform_max_abs": float(np.max(np.abs(q_diff))) if q_diff.size else 0.0,
        "LE_local_to_abq_roundtrip_rel": le_roundtrip_rel,
        "LE_local_to_abq_roundtrip_max_abs": le_roundtrip_max_abs,
        "LE_roundtrip_skipped": le_abq is None,
        "B_raw_projected_rel": b_rel,
        "B_raw_projected_max_abs": float(np.max(np.abs(b_projected_diff))) if b_projected_diff.size else 0.0,
        "B_rigid_residual_rel": rel_norm(b_raw_diff, b_raw),
        "B_rigid_residual_max_abs": float(np.max(np.abs(b_raw_diff))) if b_raw_diff.size else 0.0,
    }


def build_one(path: Path, out_root: Path, *, strict: bool, tol: float) -> dict[str, Any]:
    with np.load(str(path), allow_pickle=True) as z:
        files = list(z.files)
        required = {
            "q48_raw",
            "B_LE128_forward",
            "T_q_raw_to_useful",
            "T_eps_to_abq",
            "T_eps_from_abq",
            "ip_xi",
        }
        missing = sorted(required.difference(files))
        if missing:
            raise KeyError(f"{path}: missing required fields {missing}")
        if bool(np.asarray(z["uses_old_true176_labels_as_v2_labels"]).reshape(-1)[0]) if "uses_old_true176_labels_as_v2_labels" in files else False:
            raise ValueError(f"{path}: uses_old_true176_labels_as_v2_labels is true")

        le_abq_key = first_key(files, ("LE_abq", "LE128_base", "le"))
        le_local_key = first_key(files, ("LE_local", "LE128_local"))
        if le_abq_key is None and le_local_key is None:
            raise KeyError(f"{path}: missing LE_abq/LE128_base or LE_local/LE128_local")

        case_id = parse_case_id(path, z)
        case_name = f"case{case_id:03d}"
        q48_raw = as_float64(z, "q48_raw")
        b_raw = as_float64(z, "B_LE128_forward")
        t_q = as_float64(z, "T_q_raw_to_useful")
        t_eps_to_abq = as_float64(z, "T_eps_to_abq")
        t_eps_from_abq = as_float64(z, "T_eps_from_abq")
        ip_xi = as_ip_xi(z)

        q_useful = apply_t_q(q48_raw, t_q)
        le_abq = as_float64(z, le_abq_key) if le_abq_key else None
        if le_abq is not None:
            le_local = apply_t_eps_to_strain(t_eps_from_abq, le_abq)
        else:
            le_local = as_float64(z, le_local_key)  # type: ignore[arg-type]
        b_local_raw = apply_t_eps_to_b(t_eps_from_abq, b_raw)
        b_local_useful = right_multiply_t_q_transpose(b_local_raw, t_q)

        if q48_raw.shape != (q_useful.shape[0], 48):
            raise ValueError(f"{path}: q48_raw must be [N,48], got {q48_raw.shape}")
        if b_raw.shape != (q_useful.shape[0], 128, 6, 48):
            raise ValueError(f"{path}: B_LE128_forward must be [N,128,6,48], got {b_raw.shape}")
        if le_abq is not None and le_abq.shape != (q_useful.shape[0], 128, 6):
            raise ValueError(f"{path}: {le_abq_key} must be [N,128,6], got {le_abq.shape}")
        if le_local.shape != (q_useful.shape[0], 128, 6):
            raise ValueError(f"{path}: LE_local must be [N,128,6], got {le_local.shape}")
        if ip_xi.shape != (128, 3):
            raise ValueError(f"{path}: ip_xi must be [128,3], got {ip_xi.shape}")

        audit = audit_arrays(
            q48_raw=q48_raw,
            q_useful=q_useful,
            le_abq=le_abq,
            le_local=le_local,
            b_raw=b_raw,
            b_local_useful=b_local_useful,
            t_q=t_q,
            t_eps_to_abq=t_eps_to_abq,
            strict=strict,
            tol=tol,
        )

        metadata = {
            "standard_operator_contract_version": CONTRACT_VERSION,
            "model_input_coordinate": "q_useful",
            "model_point_coordinate": "ip_xi_standard_or_parent",
            "model_output_coordinate": "local_jacobian_frame",
            "model_output_quantity": "LE_local",
            "model_derivative": "dLE_local/dq_useful",
            "raw_q_coordinate": "q48_raw_abaqus_global",
            "raw_B_coordinate": "dLE_abq_global/dq48_raw",
            "postprocess_B_formula": "B_raw_hat = T_eps_to_abq @ B_local_useful_hat @ T_q_raw_to_useful",
            "preprocess_q_formula": "q_useful = T_q_raw_to_useful @ q48_raw",
            "preprocess_LE_formula": "LE_local = T_eps_from_abq @ LE_abq",
            "preprocess_B_formula": "B_local_useful = T_eps_from_abq @ B_raw @ T_q_raw_to_useful.T",
            "network_must_not_learn_transforms": True,
            "model_visible_fields": list(MODEL_VISIBLE_FIELDS),
            "audit_postprocess_only_fields": list(AUDIT_POSTPROCESS_FIELDS),
        }

        case_dir = out_root / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        out_npz = case_dir / f"{case_name}_standard_operator.npz"
        out_summary = case_dir / f"{case_name}_standard_operator.summary.json"

        arrays: dict[str, Any] = {
            "q_useful": q_useful,
            "ip_xi": ip_xi,
            "LE_local": le_local,
            "B_local_useful": b_local_useful,
            "q48_raw": q48_raw,
            "B_LE128_forward": b_raw,
            "T_q_raw_to_useful": t_q,
            "T_eps_to_abq": t_eps_to_abq,
            "T_eps_from_abq": t_eps_from_abq,
            "case_id": np.array(case_id, dtype=np.int64),
            "source_compact": np.array(str(path)),
            "source_LE_abq_key": np.array(le_abq_key or ""),
            "metadata": np.array(json.dumps(metadata, ensure_ascii=False, default=json_default)),
            "standard_operator_contract_version": np.array(CONTRACT_VERSION),
            "model_input_coordinate": np.array("q_useful"),
            "model_point_coordinate": np.array("ip_xi_standard_or_parent"),
            "model_output_coordinate": np.array("local_jacobian_frame"),
            "model_output_quantity": np.array("LE_local"),
            "model_derivative": np.array("dLE_local/dq_useful"),
            "raw_q_coordinate": np.array("q48_raw_abaqus_global"),
            "raw_B_coordinate": np.array("dLE_abq_global/dq48_raw"),
            "postprocess_B_formula": np.array("B_raw_hat = T_eps_to_abq @ B_local_useful_hat @ T_q_raw_to_useful"),
            "preprocess_q_formula": np.array("q_useful = T_q_raw_to_useful @ q48_raw"),
            "preprocess_LE_formula": np.array("LE_local = T_eps_from_abq @ LE_abq"),
            "preprocess_B_formula": np.array("B_local_useful = T_eps_from_abq @ B_raw @ T_q_raw_to_useful.T"),
            "network_must_not_learn_transforms": np.array(True),
            "uses_old_true176_labels_as_v2_labels": np.array(False),
            "model_visible_fields": np.asarray(MODEL_VISIBLE_FIELDS),
            "audit_postprocess_only_fields": np.asarray(AUDIT_POSTPROCESS_FIELDS),
        }
        if le_abq is not None:
            arrays["LE_abq"] = le_abq
        if "local_frame_Q" in files:
            arrays["local_frame_Q"] = as_float64(z, "local_frame_Q")
        for key in (
            "strain_field",
            "B_label_strain_field",
            "strain_voigt_order",
            "strain_shear_convention",
            "q_useful_coordinate",
            "q_useful_definition",
        ):
            if key in files:
                arrays[key] = np.asarray(z[key])

        np.savez_compressed(out_npz, **arrays)

    summary = {
        "case_id": int(case_id),
        "case_name": case_name,
        "source_compact": str(path),
        "standard_compact": str(out_npz),
        "summary_path": str(out_summary),
        "standard_operator_contract_version": CONTRACT_VERSION,
        "model_visible_fields": list(MODEL_VISIBLE_FIELDS),
        "audit_postprocess_only_fields": list(AUDIT_POSTPROCESS_FIELDS),
        "frame_count": int(q_useful.shape[0]),
        "point_count": int(ip_xi.shape[0]),
        "q_useful_shape": list(q_useful.shape),
        "ip_xi_shape": list(ip_xi.shape),
        "LE_local_shape": list(le_local.shape),
        "B_local_useful_shape": list(b_local_useful.shape),
        "LE_abq_present": le_abq is not None,
        "strict_requested": bool(strict),
        **audit,
        "trained_model": False,
        "used_old_true176_labels": False,
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--tol", type=float, default=1.0e-10)
    args = parser.parse_args()

    paths = read_compact_list(args.compact_list)
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows = [build_one(path, args.out_root, strict=args.strict, tol=args.tol) for path in paths]
    compact_list = args.out_root / "standard_operator_compact_list.txt"
    compact_list.write_text(
        "\n".join(str(row["standard_compact"]) for row in rows) + "\n",
        encoding="utf-8",
    )

    strict_pass_count = sum(1 for row in rows if row["strict_pass"])
    summary = {
        "standard_operator_contract_version": CONTRACT_VERSION,
        "compact_count": len(rows),
        "strict_requested": bool(args.strict),
        "strict_pass_count": int(strict_pass_count),
        "strict_fail_count": int(len(rows) - strict_pass_count),
        "standard_operator_compact_list": str(compact_list),
        "model_visible_fields": list(MODEL_VISIBLE_FIELDS),
        "audit_postprocess_only_fields": list(AUDIT_POSTPROCESS_FIELDS),
        "trained_model": False,
        "used_old_true176_labels": False,
        "cases": rows,
    }
    summary_path = args.out_root / "standard_operator_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")

    print(json.dumps({
        "standard_operator_summary": str(summary_path),
        "standard_operator_compact_list": str(compact_list),
        "compact_count": len(rows),
        "strict_pass_count": int(strict_pass_count),
        "strict_fail_count": int(len(rows) - strict_pass_count),
    }, indent=2, ensure_ascii=False))
    if args.strict and strict_pass_count != len(rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
