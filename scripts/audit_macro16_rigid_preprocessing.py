#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit Macro16 rigid-motion preprocessing before any training run."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macro_deeponet.macro16_geometry import (  # noqa: E402
    macro16_standard_point_table,
    scale_consistency_report,
)
from macro_deeponet.macro16_rigid import remove_rigid_motion, rigid_consistency_report  # noqa: E402
from macro_deeponet.train_macro16_boundary_sobolev import load_macro16_compacts  # noqa: E402


REQUIRED_RIGID_COMPACT_FIELDS = (
    "X16_raw",
    "X_center",
    "L_ref",
    "X16_hat",
    "q48_raw",
    "q48_hat",
    "q48_rigid_raw",
    "q48_def_raw",
    "q48_def_hat",
    "B_macro_qraw",
    "B_macro_qhat",
    "B_macro_qdef_raw",
    "B_macro_qdef",
    "integration_weight_hat",
    "integration_weight_phys",
    "rigid_projection_P",
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def read_path_list(path: Path) -> list[Path]:
    return [
        Path(line.strip()).resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def collect_paths(compacts: Iterable[Path], compact_lists: Iterable[Path], compact_globs: Iterable[str], *, case_limit: int) -> list[Path]:
    paths = [Path(p).resolve() for p in compacts]
    for list_path in compact_lists:
        paths.extend(read_path_list(Path(list_path)))
    for pattern in compact_globs:
        paths.extend(Path(hit).resolve() for hit in sorted(glob.glob(str(pattern))))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if int(case_limit) > 0:
        unique = unique[: int(case_limit)]
    return unique


def scalar_text(z: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    if arr.size == 0:
        return default
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def infer_n(z: np.lib.npyio.NpzFile, path: Path) -> int:
    for key in ("q48_raw", "q48_def_hat", "LE_macro", "B_macro_qraw", "B_macro"):
        if key in z.files:
            arr = np.asarray(z[key])
            if arr.ndim >= 1:
                return int(arr.shape[0])
    raise KeyError(f"{path}: cannot infer frame count")


def frame_rows(n_total: int, *, frame_stride: int, max_frames_per_compact: int) -> np.ndarray:
    rows = np.arange(0, int(n_total), max(1, int(frame_stride)), dtype=np.int64)
    if int(max_frames_per_compact) > 0:
        rows = rows[: int(max_frames_per_compact)]
    return rows


def frame_array(
    z: np.lib.npyio.NpzFile,
    path: Path,
    n_total: int,
    rows: np.ndarray,
    key: str,
    tail: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(z[key], dtype=np.float64)
    if arr.shape == tuple(tail):
        return np.broadcast_to(arr.reshape((1,) + tuple(tail)), (rows.size,) + tuple(tail)).copy()
    if arr.shape == (1,) + tuple(tail):
        return np.broadcast_to(arr.reshape((1,) + tuple(tail)), (rows.size,) + tuple(tail)).copy()
    if tail == (1,) and arr.shape == (n_total,):
        return arr[rows].reshape(rows.size, 1)
    if arr.shape == (n_total,) + tuple(tail):
        return arr[rows]
    raise ValueError(f"{path}: {key} must have shape {tail}, [1,{tail}], or [{n_total},{tail}], got {arr.shape}")


def rel_norm(diff: np.ndarray, ref: np.ndarray) -> float:
    ref_arr = np.asarray(ref, dtype=np.float64)
    den = max(float(np.linalg.norm(ref_arr.reshape(-1))), 1.0e-30)
    return float(np.linalg.norm(np.asarray(diff, dtype=np.float64).reshape(-1)) / den)


def metric(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    pred_arr = np.asarray(pred, dtype=np.float64)
    ref_arr = np.asarray(ref, dtype=np.float64)
    diff = pred_arr - ref_arr
    return {
        "rel": rel_norm(diff, ref_arr),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
    }


def check_leq(name: str, value: float | None, limit: float) -> dict[str, Any]:
    enforced = bool(float(limit) > 0.0 and value is not None)
    passed = (not enforced) or (math.isfinite(float(value)) and float(value) <= float(limit))
    return {"name": name, "value": value, "limit": float(limit), "enforced": enforced, "passed": bool(passed)}


def flat_macro16_x16(thickness: float = 0.2) -> np.ndarray:
    bottom = np.asarray(
        [
            [-1.0, -1.0, -0.5 * thickness],
            [0.0, -1.0, -0.5 * thickness],
            [1.0, -1.0, -0.5 * thickness],
            [1.0, 0.0, -0.5 * thickness],
            [1.0, 1.0, -0.5 * thickness],
            [0.0, 1.0, -0.5 * thickness],
            [-1.0, 1.0, -0.5 * thickness],
            [-1.0, 0.0, -0.5 * thickness],
        ],
        dtype=np.float64,
    )
    top = bottom.copy()
    top[:, 2] = 0.5 * thickness
    return np.concatenate([bottom, top], axis=0)


def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    a = a / np.linalg.norm(a)
    k = np.asarray([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]], dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    return eye + math.sin(float(angle)) * k + (1.0 - math.cos(float(angle))) * (k @ k)


def rigid_q(x16: np.ndarray, *, rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    r = np.eye(3, dtype=np.float64) if rotation is None else np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    t = np.zeros(3, dtype=np.float64) if translation is None else np.asarray(translation, dtype=np.float64).reshape(3)
    x = np.asarray(x16, dtype=np.float64).reshape(16, 3)
    return ((r @ x.T).T + t.reshape(1, 3) - x).reshape(48)


def audit_synthetic_rigid(tol: float) -> dict[str, Any]:
    x16 = flat_macro16_x16()
    cases: list[tuple[str, np.ndarray, bool]] = [
        ("pure_translation", rigid_q(x16, translation=np.asarray([0.3, -0.2, 0.15])), True),
        ("pure_rotation_x", rigid_q(x16, rotation=rodrigues(np.asarray([1.0, 0.0, 0.0]), 2.0e-3)), True),
        ("pure_rotation_y", rigid_q(x16, rotation=rodrigues(np.asarray([0.0, 1.0, 0.0]), -2.5e-3)), True),
        ("pure_rotation_z", rigid_q(x16, rotation=rodrigues(np.asarray([0.0, 0.0, 1.0]), 3.0e-3)), True),
    ]
    deformation = np.zeros((16, 3), dtype=np.float64)
    deformation[:, 2] = np.linspace(-0.01, 0.01, 16)
    mixed = rigid_q(
        x16,
        rotation=rodrigues(np.asarray([0.2, -0.1, 1.0]), 1.5e-3),
        translation=np.asarray([0.05, -0.02, 0.01]),
    ) + deformation.reshape(48)
    cases.append(("translation_rotation_plus_deformation", mixed, False))

    rows: list[dict[str, Any]] = []
    passed = True
    for name, q_raw, should_zero in cases:
        q_def, q_rigid, r, t = remove_rigid_motion(x16, q_raw)
        closure = q_rigid + q_def - q_raw
        r_orth = r.T @ r - np.eye(3, dtype=np.float64)
        row = {
            "name": name,
            "q48_raw_shape": list(q_raw.shape),
            "q48_def_raw_shape": list(q_def.shape),
            "q48_rigid_raw_shape": list(q_rigid.shape),
            "translation_shape": list(t.shape),
            "rotation_shape": list(r.shape),
            "output_remains_48d": bool(q_def.shape == (48,) and q_rigid.shape == (48,)),
            "q_raw_vs_rigid_plus_def_max_abs": float(np.max(np.abs(closure))),
            "rotation_orthogonality_max_abs": float(np.max(np.abs(r_orth))),
            "rotation_det": float(np.linalg.det(r)),
            "q_def_max_abs": float(np.max(np.abs(q_def))),
            "pure_rigid_expected_zero_def": bool(should_zero),
        }
        row["passed"] = bool(
            row["output_remains_48d"]
            and row["q_raw_vs_rigid_plus_def_max_abs"] <= float(tol)
            and row["rotation_orthogonality_max_abs"] <= float(tol)
            and row["rotation_det"] > 0.0
            and ((not should_zero) or row["q_def_max_abs"] <= float(tol))
        )
        passed = bool(passed and row["passed"])
        rows.append(row)
    return {
        "available": True,
        "passed": bool(passed),
        "tolerance": float(tol),
        "case_count": int(len(rows)),
        "cases": rows,
        "input_dimension_note": "Rigid motion is removed in preprocessing, but q remains 48D; no 42D projection is used.",
    }


def audit_loader(path: Path, *, frame_stride: int, max_frames_per_compact: int) -> dict[str, Any]:
    try:
        data = load_macro16_compacts(
            [str(path)],
            point_table=macro16_standard_point_table(),
            frame_stride=frame_stride,
            max_frames_per_compact=max_frames_per_compact,
        )
    except Exception as exc:  # pragma: no cover - exact exception text is more useful in JSON than stack type here.
        return {"available": False, "passed": False, "error": str(exc)}
    q_shape = list(data.q48_hat.shape)
    b_shape = list(data.b.shape)
    passed = bool(
        data.point_meta.get("model_visible_q") == "q48_def_hat"
        and data.point_meta.get("model_visible_B") == "B_macro_qdef"
        and data.point_meta.get("rigid_motion_removed_by_preprocessing") is True
        and len(q_shape) == 2
        and q_shape[1] == 48
        and len(b_shape) == 4
        and b_shape[-1] == 48
    )
    return {
        "available": True,
        "passed": passed,
        "model_visible_q": data.point_meta.get("model_visible_q"),
        "model_visible_B": data.point_meta.get("model_visible_B"),
        "rigid_motion_removed_by_preprocessing": data.point_meta.get("rigid_motion_removed_by_preprocessing"),
        "q48_hat_shape_loaded_for_model": q_shape,
        "B_shape_loaded_for_model": b_shape,
        "weight_shape_loaded_for_model": list(data.weights.shape),
    }


def audit_compact(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    path = Path(path).resolve()
    with np.load(str(path), allow_pickle=True) as z:
        n_total = infer_n(z, path)
        rows = frame_rows(n_total, frame_stride=int(args.frame_stride), max_frames_per_compact=int(args.max_frames_per_compact))
        missing = [key for key in REQUIRED_RIGID_COMPACT_FIELDS if key not in z.files]
        base: dict[str, Any] = {
            "compact": str(path),
            "frame_count_total": int(n_total),
            "frame_count_checked": int(rows.size),
            "macro16_teacher_contract_version": scalar_text(z, "macro16_teacher_contract_version", ""),
            "missing_required_fields": missing,
            "required_fields_present": bool(not missing),
        }
        if missing:
            base["passed"] = False
            base["reason"] = "compact is missing new rigid/scale contract fields; regenerate source128 compact with current builder"
            return base

        x16 = frame_array(z, path, n_total, rows, "X16_raw", (16, 3))
        l_ref = frame_array(z, path, n_total, rows, "L_ref", (1,))
        q_raw = frame_array(z, path, n_total, rows, "q48_raw", (48,))
        q_hat = frame_array(z, path, n_total, rows, "q48_hat", (48,))
        q_rigid = frame_array(z, path, n_total, rows, "q48_rigid_raw", (48,))
        q_def = frame_array(z, path, n_total, rows, "q48_def_raw", (48,))
        q_def_hat = frame_array(z, path, n_total, rows, "q48_def_hat", (48,))
        b_qraw = frame_array(z, path, n_total, rows, "B_macro_qraw", (np.asarray(z["B_macro_qraw"]).shape[1], 6, 48))
        b_qhat = frame_array(z, path, n_total, rows, "B_macro_qhat", (b_qraw.shape[1], 6, 48))
        b_qdef_raw = frame_array(z, path, n_total, rows, "B_macro_qdef_raw", (b_qraw.shape[1], 6, 48))
        b_qdef = frame_array(z, path, n_total, rows, "B_macro_qdef", (b_qraw.shape[1], 6, 48))
        w_hat = frame_array(z, path, n_total, rows, "integration_weight_hat", (b_qraw.shape[1],))
        w_phys = frame_array(z, path, n_total, rows, "integration_weight_phys", (b_qraw.shape[1],))
        p_rigid = frame_array(z, path, n_total, rows, "rigid_projection_P", (48, 48))

        rigid = rigid_consistency_report(
            x16_raw=x16,
            q48_raw=q_raw,
            q48_rigid_raw=q_rigid,
            q48_def_raw=q_def,
            q48_def_hat=q_def_hat,
            l_ref=l_ref,
            rigid_projection_p=p_rigid,
        )
        scale = scale_consistency_report(
            q48_raw=q_raw,
            q48_hat=q_hat,
            b_macro_qraw=b_qraw,
            b_macro_qhat=b_qhat,
            integration_weight_hat=w_hat,
            integration_weight_phys=w_phys,
            l_ref=l_ref,
        )
        b_qdef_scale = metric(b_qdef, b_qdef_raw * l_ref.reshape(l_ref.shape[0], 1, 1, 1))
        b_qdef_projection = metric(b_qdef_raw, np.einsum("npak,nkj->npaj", b_qraw, p_rigid))
        rotations = frame_array(z, path, n_total, rows, "rigid_rotation_R", (3, 3)) if "rigid_rotation_R" in z.files else None
        rotation_audit: dict[str, Any] = {"available": rotations is not None}
        if rotations is not None:
            eye = np.eye(3, dtype=np.float64).reshape(1, 3, 3)
            rtr = np.einsum("nki,nkj->nij", rotations, rotations)
            dets = np.linalg.det(rotations)
            rotation_audit.update(
                {
                    "rotation_orthogonality_max_abs": float(np.max(np.abs(rtr - eye))),
                    "rotation_det_min": float(np.min(dets)),
                    "rotation_det_max": float(np.max(dets)),
                }
            )

    loader = audit_loader(path, frame_stride=int(args.frame_stride), max_frames_per_compact=int(args.max_frames_per_compact))
    checks = [
        {"name": "required_new_fields", "passed": True, "enforced": True},
        {"name": "training_loader_uses_q48_def_hat_and_B_macro_qdef", "passed": bool(loader.get("passed")), "enforced": True},
        check_leq("q48_raw_vs_rigid_plus_def_rel", rigid.get("q48_raw_vs_rigid_plus_def_rel"), float(args.max_rigid_closure_rel)),
        check_leq("q48_raw_vs_rigid_plus_def_max_abs", rigid.get("q48_raw_vs_rigid_plus_def_max_abs"), float(args.max_rigid_closure_abs)),
        check_leq("q48_def_hat_times_L_ref_vs_q48_def_raw_rel", rigid.get("q48_def_hat_times_L_ref_vs_q48_def_raw_rel"), float(args.max_scale_rel)),
        check_leq("q48_def_hat_times_L_ref_vs_q48_def_raw_max_abs", rigid.get("q48_def_hat_times_L_ref_vs_q48_def_raw_max_abs"), float(args.max_scale_abs)),
        check_leq("B_macro_qdef_vs_B_macro_qdef_raw_times_L_ref_rel", b_qdef_scale["rel"], float(args.max_scale_rel)),
        check_leq("B_macro_qdef_vs_B_macro_qdef_raw_times_L_ref_max_abs", b_qdef_scale["max_abs"], float(args.max_scale_abs)),
        check_leq("B_macro_qdef_raw_vs_B_macro_qraw_times_rigid_projection_P_rel", b_qdef_projection["rel"], float(args.max_projection_rel)),
        check_leq("B_macro_qdef_raw_vs_B_macro_qraw_times_rigid_projection_P_max_abs", b_qdef_projection["max_abs"], float(args.max_projection_abs)),
        check_leq(
            "integration_weight_phys_vs_hat_times_L_ref3_rel",
            scale.get("integration_weight_phys_vs_hat_times_L_ref3_rel"),
            float(args.max_scale_rel),
        ),
        check_leq(
            "integration_weight_phys_vs_hat_times_L_ref3_max_abs",
            scale.get("integration_weight_phys_vs_hat_times_L_ref3_max_abs"),
            float(args.max_scale_abs),
        ),
        check_leq(
            "q48_def_translation_orthogonality_max_abs",
            rigid.get("q48_def_translation_orthogonality_max_abs"),
            float(args.max_qdef_translation_orthogonality),
        ),
        check_leq(
            "q48_def_rotation_orthogonality_max_abs",
            rigid.get("q48_def_rotation_orthogonality_max_abs"),
            float(args.max_qdef_rotation_orthogonality),
        ),
    ]
    if rotation_audit.get("available"):
        checks.append(
            check_leq(
                "rigid_rotation_R_orthogonality_max_abs",
                rotation_audit.get("rotation_orthogonality_max_abs"),
                float(args.max_rotation_matrix_orthogonality),
            )
        )
        checks.append(
            {
                "name": "rigid_rotation_R_det_positive",
                "value": rotation_audit.get("rotation_det_min"),
                "limit": 0.0,
                "enforced": True,
                "passed": bool(float(rotation_audit.get("rotation_det_min", -1.0)) > 0.0),
            }
        )
    passed = bool(all(bool(check.get("passed")) for check in checks))
    return {
        **base,
        "passed": passed,
        "point_count": int(b_qraw.shape[1]),
        "dimension_checks": {
            "q48_raw_shape": list(q_raw.shape),
            "q48_def_hat_shape": list(q_def_hat.shape),
            "B_macro_qdef_shape": list(b_qdef.shape),
            "q_input_remains_48d": bool(q_raw.shape[-1] == 48 and q_def_hat.shape[-1] == 48 and b_qdef.shape[-1] == 48),
            "no_42d_input": bool(q_def_hat.shape[-1] != 42 and b_qdef.shape[-1] != 42),
        },
        "loader": loader,
        "rigid_consistency": rigid,
        "scale_consistency": scale,
        "B_macro_qdef_scale_consistency": b_qdef_scale,
        "B_macro_qdef_linear_projection_consistency": b_qdef_projection,
        "rotation_matrix_consistency": rotation_audit,
        "threshold_checks": checks,
        "linear_projection_note": (
            "B_macro_qdef_raw is audited against B_macro_qraw @ rigid_projection_P. "
            "rigid_projection_P is the current small-rotation linearized chain-rule approximation, not a strict Kabsch Jacobian."
        ),
    }


def aggregate(compacts: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(path: tuple[str, ...]) -> list[float]:
        out: list[float] = []
        for row in compacts:
            cur: Any = row
            for key in path:
                if not isinstance(cur, dict) or key not in cur:
                    cur = None
                    break
                cur = cur[key]
            if cur is not None:
                val = float(cur)
                if math.isfinite(val):
                    out.append(val)
        return out

    def max_or_none(path: tuple[str, ...]) -> float | None:
        data = vals(path)
        return float(max(data)) if data else None

    return {
        "compact_count": int(len(compacts)),
        "passed_count": int(sum(1 for row in compacts if bool(row.get("passed")))),
        "missing_required_field_count": int(sum(1 for row in compacts if row.get("missing_required_fields"))),
        "q48_raw_vs_rigid_plus_def_rel_max": max_or_none(("rigid_consistency", "q48_raw_vs_rigid_plus_def_rel")),
        "q48_def_hat_times_L_ref_vs_q48_def_raw_rel_max": max_or_none(
            ("rigid_consistency", "q48_def_hat_times_L_ref_vs_q48_def_raw_rel")
        ),
        "B_macro_qdef_scale_rel_max": max_or_none(("B_macro_qdef_scale_consistency", "rel")),
        "B_macro_qdef_projection_rel_max": max_or_none(("B_macro_qdef_linear_projection_consistency", "rel")),
        "integration_weight_phys_scale_rel_max": max_or_none(("scale_consistency", "integration_weight_phys_vs_hat_times_L_ref3_rel")),
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    paths = collect_paths(args.compact, args.compact_list, args.compact_glob, case_limit=int(args.case_limit))
    if not paths and bool(args.skip_synthetic):
        raise ValueError("provide --compact/--compact-list/--compact-glob or omit --skip-synthetic")

    synthetic = None if bool(args.skip_synthetic) else audit_synthetic_rigid(float(args.synthetic_tol))
    compacts = [audit_compact(path, args) for path in paths]
    compact_aggregate = aggregate(compacts)
    strict_pass = bool((synthetic is None or synthetic["passed"]) and all(bool(row.get("passed")) for row in compacts))
    summary = {
        "script": "audit_macro16_rigid_preprocessing",
        "compact_paths": [str(path) for path in paths],
        "synthetic_rigid_cases": synthetic,
        "aggregate": compact_aggregate,
        "strict_pass": strict_pass,
        "compacts": compacts,
        "interpretation_note": (
            "This audit checks the preprocessing data contract only. It does not train a network. "
            "The current B chain-rule field is the linearized rigid_projection_P approximation; "
            "strict nonlinear Kabsch Jacobians are not claimed here."
        ),
    }
    out = Path(args.out).resolve()
    write_json(out, summary)
    print(
        json.dumps(
            {
                "summary": str(out),
                "strict_pass": strict_pass,
                "synthetic_passed": None if synthetic is None else bool(synthetic["passed"]),
                **compact_aggregate,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=json_default,
        )
    )
    if bool(args.strict) and not strict_pass:
        raise SystemExit(1)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="append", type=Path, default=[])
    parser.add_argument("--compact-list", action="append", type=Path, default=[])
    parser.add_argument("--compact-glob", action="append", default=[])
    parser.add_argument("--out", type=Path, default=Path("runs") / "macro16_rigid_preprocessing_audit.json")
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-compact", type=int, default=0)
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--synthetic-tol", type=float, default=1.0e-10)
    parser.add_argument("--max-scale-rel", type=float, default=1.0e-4)
    parser.add_argument("--max-scale-abs", type=float, default=1.0e-5)
    parser.add_argument("--max-projection-rel", type=float, default=1.0e-4)
    parser.add_argument("--max-projection-abs", type=float, default=1.0e-5)
    parser.add_argument("--max-rigid-closure-rel", type=float, default=1.0e-6)
    parser.add_argument("--max-rigid-closure-abs", type=float, default=1.0e-6)
    parser.add_argument("--max-rotation-matrix-orthogonality", type=float, default=1.0e-5)
    parser.add_argument("--max-qdef-translation-orthogonality", type=float, default=0.0)
    parser.add_argument("--max-qdef-rotation-orthogonality", type=float, default=0.0)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_audit(parse_args())


if __name__ == "__main__":
    main()
