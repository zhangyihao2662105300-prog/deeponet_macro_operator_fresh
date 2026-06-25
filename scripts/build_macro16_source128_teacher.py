#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Macro16 128-point teacher compacts without reducing integration points.

The output keeps the Macro16 boundary interface, X16 plus q48, but uses the
source 128 teacher points and Abaqus integration volumes directly.  This is a
force/stiffness closure candidate, not a reduced integration rule.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_macro16_from_128_teacher import (  # noqa: E402
    TRUE176_MACRO_TO_KEEP_FLAT,
    collect_paths,
    frame_rows,
    infer_n,
    load_case_ids,
    load_labels_by_mode,
    load_q48,
    load_x16,
    reorder_nodes_q_b_to_macro16,
    resolve_source_node_order,
    scalar_text,
    validate_ip_keys,
)
from macro_deeponet.macro16_geometry import (  # noqa: E402
    MACRO16_CONTRACT_VERSION,
    Macro16GeometryMap,
    macro16_source128_point_table,
    normalize_macro16_x16_batch,
    scale_consistency_report,
)
from macro_deeponet.macro16_rigid import rigid_consistency_report, rigid_preprocess_batch  # noqa: E402
from macro_deeponet.true176_data import standard_css8_row_map  # noqa: E402

SOURCE128_CONTRACT_VERSION = "macro16-source128-teacher-rigid-preprocess-003"


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


def row_select_or_broadcast(vals: np.ndarray, rows: np.ndarray, n_total: int, tail: tuple[int, ...], key: str, path: Path) -> np.ndarray:
    arr = np.asarray(vals)
    if arr.shape == tuple(tail):
        return np.broadcast_to(arr.reshape((1,) + tuple(tail)), (rows.size,) + tuple(tail)).copy()
    if arr.shape == (1,) + tuple(tail):
        return np.broadcast_to(arr.reshape((1,) + tuple(tail)), (rows.size,) + tuple(tail)).copy()
    if arr.shape == (int(n_total),) + tuple(tail):
        return arr[rows]
    raise ValueError(f"{path}: {key} must have shape {tail}, [1,{tail}], or [{n_total},{tail}], got {arr.shape}")


def load_volume_weights(z: np.lib.npyio.NpzFile, path: Path, n_total: int, rows: np.ndarray, mode: str) -> tuple[np.ndarray, str]:
    key_mode = str(mode).strip().lower().replace("_", "-")
    if key_mode == "macro16-x16":
        raise ValueError("macro16-x16 weights are generated after X16 node-order normalization")
    if key_mode == "ip-ivol":
        key = "ip_IVOL_abaqus"
    elif key_mode == "inferred-dle":
        key = "IVOL128_inferred_from_DLE"
    elif key_mode == "auto":
        key = "ip_IVOL_abaqus" if "ip_IVOL_abaqus" in z.files else "IVOL128_inferred_from_DLE"
    else:
        raise ValueError("volume_weight_mode must be auto, ip-ivol, or inferred-dle")
    if key not in z.files:
        raise KeyError(f"{path}: missing {key} for source128 physical integration weights")
    weights = row_select_or_broadcast(np.asarray(z[key], dtype=np.float64), rows, n_total, (128,), key, path)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError(f"{path}: {key} must be finite and positive")
    return weights.astype(np.float64), key


def source_macro_xi() -> np.ndarray:
    row = standard_css8_row_map()
    return row[:, 9:12].astype(np.float64)


def macro16_x16_source128_weights(x16: np.ndarray) -> np.ndarray:
    nodes = np.asarray(x16, dtype=np.float64).reshape(-1, 16, 3)
    point_table = macro16_source128_point_table()
    weights = np.empty((nodes.shape[0], point_table.xi.shape[0]), dtype=np.float64)
    for i, frame_nodes in enumerate(nodes):
        geom = Macro16GeometryMap(frame_nodes)
        fields = geom.eval_points(point_table)
        l_ref = float(geom.l_ref)
        weights[i] = np.asarray(fields["integration_weight_hat"], dtype=np.float64) * (l_ref ** 3)
    return weights


def macro16_source128_geometry_scale(x16_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x16_hat, x_center, l_ref = normalize_macro16_x16_batch(x16_raw)
    point_table = macro16_source128_point_table()
    weights_hat = np.empty((x16_hat.shape[0], point_table.xi.shape[0]), dtype=np.float64)
    for i, frame_nodes in enumerate(np.asarray(x16_raw, dtype=np.float64).reshape(-1, 16, 3)):
        fields = Macro16GeometryMap(frame_nodes).eval_points(point_table)
        weights_hat[i] = np.asarray(fields["integration_weight_hat"], dtype=np.float64)
    return x16_hat, x_center, l_ref, weights_hat


def build_one(
    path: Path,
    out_root: Path,
    *,
    source_index: int,
    frame_stride: int,
    max_frames_per_compact: int,
    allow_missing_ip_keys: bool,
    source_node_order: str,
    strain_coordinate_mode: str,
    volume_weight_mode: str,
) -> dict[str, Any]:
    path = Path(path).resolve()
    with np.load(str(path), allow_pickle=True) as z:
        n_total = infer_n(z, path)
        rows = frame_rows(n_total, frame_stride=frame_stride, max_frames=max_frames_per_compact)
        if rows.size == 0:
            raise ValueError(f"{path}: no frames selected")
        ip_key_meta = validate_ip_keys(z, path, n_total, rows, allow_missing=allow_missing_ip_keys)
        q48, q_key = load_q48(z, path, n_total, rows)
        x16, x16_source = load_x16(z, path, n_total, rows)
        case_id = load_case_ids(z, path, n_total, rows)
        le128, b128, le_key, b_key, strain_meta = load_labels_by_mode(
            z,
            path,
            n_total,
            rows,
            strain_coordinate_mode=strain_coordinate_mode,
        )
        key_mode = str(volume_weight_mode).strip().lower().replace("_", "-")
        if key_mode == "macro16-x16":
            weights = np.empty((rows.size, 128), dtype=np.float64)
            weight_key = "macro16_x16_source128_standard_rule"
        else:
            weights, weight_key = load_volume_weights(z, path, n_total, rows, volume_weight_mode)
        strain_field = scalar_text(z, "strain_field", scalar_text(z, "strain_label_key", "LE"))
        b_strain_field = scalar_text(z, "B_label_strain_field", strain_field)
    resolved_node_order = resolve_source_node_order(source_node_order, x16_source)
    x16, q48, b128, node_order_meta = reorder_nodes_q_b_to_macro16(
        x16=x16,
        q48=q48,
        b128=b128,
        source_node_order=resolved_node_order,
    )
    if not np.all(np.isfinite(le128)) or not np.all(np.isfinite(b128)):
        raise ValueError(f"{path}: non-finite source128 teacher labels")
    x16_hat, x_center, l_ref, weights_hat = macro16_source128_geometry_scale(x16)
    q48_hat = q48 / np.maximum(l_ref, 1.0e-12)
    q48_def_raw, q48_rigid_raw, rigid_r, rigid_t, rigid_projection_p = rigid_preprocess_batch(x16, q48)
    q48_def_hat = q48_def_raw / np.maximum(l_ref, 1.0e-12)
    b128_qraw = b128
    b128_qdef_raw = np.einsum("npak,nkj->npaj", b128_qraw, rigid_projection_p)
    b128_qdef = b128_qdef_raw * l_ref.reshape(l_ref.shape[0], 1, 1, 1)
    b128_qhat = b128_qraw * l_ref.reshape(l_ref.shape[0], 1, 1, 1)
    weights_phys = weights_hat * (l_ref**3)
    requested_weight_phys = weights_phys.copy() if weight_key == "macro16_x16_source128_standard_rule" else weights
    scale_audit = scale_consistency_report(
        q48_raw=q48,
        q48_hat=q48_hat,
        b_macro_qraw=b128_qraw,
        b_macro_qhat=b128_qhat,
        integration_weight_hat=weights_hat,
        integration_weight_phys=weights_phys,
        l_ref=l_ref,
    )
    rigid_audit = rigid_consistency_report(
        x16_raw=x16,
        q48_raw=q48,
        q48_rigid_raw=q48_rigid_raw,
        q48_def_raw=q48_def_raw,
        q48_def_hat=q48_def_hat,
        l_ref=l_ref,
        rigid_projection_p=rigid_projection_p,
    )
    first_case = int(case_id[0]) if case_id.size else -1
    case_name = f"case{first_case:03d}" if first_case >= 0 else "case_unknown"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{int(source_index):04d}_{case_name}_macro16_source128_teacher.npz"
    point_xi = source_macro_xi()
    np.savez_compressed(
        out_path,
        standard_operator_contract_version=np.asarray(MACRO16_CONTRACT_VERSION, dtype=object),
        macro16_teacher_contract_version=np.asarray(SOURCE128_CONTRACT_VERSION, dtype=object),
        q48_raw=q48.astype(np.float32),
        q48_hat=q48_hat.astype(np.float32),
        q48_rigid_raw=q48_rigid_raw.astype(np.float32),
        q48_def_raw=q48_def_raw.astype(np.float32),
        q48_def_hat=q48_def_hat.astype(np.float32),
        rigid_rotation_R=rigid_r.astype(np.float32),
        rigid_translation_t=rigid_t.astype(np.float32),
        rigid_projection_P=rigid_projection_p.astype(np.float32),
        X16_raw=x16.astype(np.float32),
        X_center=x_center.astype(np.float32),
        L_ref=l_ref.astype(np.float32),
        X16_hat=x16_hat.astype(np.float32),
        X16=x16.astype(np.float32),
        LE_macro=le128.astype(np.float32),
        B_macro_qraw=b128_qraw.astype(np.float32),
        B_macro_qdef_raw=b128_qdef_raw.astype(np.float32),
        B_macro_qdef=b128_qdef.astype(np.float32),
        B_macro_qhat=b128_qhat.astype(np.float32),
        B_macro=b128_qraw.astype(np.float32),
        integration_weight_hat=weights_hat.astype(np.float32),
        integration_weight_phys=weights_phys.astype(np.float32),
        requested_integration_weight_phys=requested_weight_phys.astype(np.float32),
        case_id=case_id.astype(np.int64),
        source_compact=np.asarray(str(path), dtype=object),
        source_index=np.full(rows.size, int(source_index), dtype=np.int64),
        source_row=rows.astype(np.int64),
        source_q_key=np.asarray(q_key, dtype=object),
        source_x16_key=np.asarray(x16_source, dtype=object),
        source_node_order=np.asarray(node_order_meta["source_node_order"], dtype=object),
        node_order_transform=np.asarray(node_order_meta["node_order_transform"], dtype=object),
        source_le128_key=np.asarray(le_key, dtype=object),
        source_b128_key=np.asarray(b_key, dtype=object),
        source_strain_coordinate_mode=np.asarray(strain_meta["strain_coordinate_mode"], dtype=object),
        source_strain_coordinate_note=np.asarray(strain_meta["strain_coordinate_note"], dtype=object),
        macro16_point_xi=point_xi.astype(np.float32),
        macro16_point_set=np.asarray("source128_css8_gauss_points", dtype=object),
        macro16_parent_interpolation_from_128=np.asarray("none_source_128_points_kept", dtype=object),
        integration_weight_source=np.asarray("macro16_x16_source128_standard_rule", dtype=object),
        requested_integration_weight_source=np.asarray(weight_key, dtype=object),
        integration_weight_coordinate=np.asarray("hat-dimensionless", dtype=object),
        integration_weight_phys_coordinate=np.asarray("physical-volume", dtype=object),
        integration_weight_rule=np.asarray("macro16_x16_standard_source128", dtype=object),
        requested_integration_weight_rule=np.asarray(
            "macro16_x16_standard_source128" if weight_key == "macro16_x16_source128_standard_rule" else "source_teacher_volume",
            dtype=object,
        ),
        strain_output_coordinate=np.asarray(strain_meta["strain_output_coordinate"], dtype=object),
        strain_field=np.asarray(strain_field, dtype=object),
        B_label_strain_field=np.asarray(b_strain_field, dtype=object),
        B_label_q_coordinate=np.asarray("q48_raw", dtype=object),
        B_macro_qraw_label_q_coordinate=np.asarray("q48_raw", dtype=object),
        B_macro_qdef_raw_label_q_coordinate=np.asarray("q48_def_raw", dtype=object),
        B_macro_qdef_label_q_coordinate=np.asarray("q48_def_hat", dtype=object),
        B_macro_qhat_label_q_coordinate=np.asarray("q48_hat", dtype=object),
        rigid_preprocessing=np.asarray("kabsch_remove_translation_and_rotation_keep_48d", dtype=object),
        rigid_projection_P_note=np.asarray(
            "rigid_projection_P is a small-rotation linear projector for first-pass B chain-rule labels; "
            "Kabsch preprocessing is nonlinear and can later use a numerical or analytic Jacobian.",
            dtype=object,
        ),
        compatibility_alias_B_macro=np.asarray("B_macro_qraw", dtype=object),
        compatibility_alias_X16=np.asarray("X16_raw", dtype=object),
        compatibility_note=np.asarray(
            "New readers prefer X16_hat/q48_def_hat/B_macro_qdef/integration_weight_hat for training and "
            "B_macro_qraw/integration_weight_phys for physical force audits; legacy X16 and B_macro are raw aliases.",
            dtype=object,
        ),
        fine_grid_geometry_visible_to_model=np.asarray(False),
        prepared_from=np.asarray("scripts/build_macro16_source128_teacher.py", dtype=object),
    )
    summary = {
        "source_compact": str(path),
        "macro16_compact": str(out_path),
        "standard_operator_contract_version": MACRO16_CONTRACT_VERSION,
        "macro16_teacher_contract_version": SOURCE128_CONTRACT_VERSION,
        "frame_count": int(rows.size),
        "source_rows_first_last": [int(rows[0]), int(rows[-1])],
        "point_count": 128,
        "q_key": q_key,
        "x16_source": x16_source,
        **node_order_meta,
        "source_le128_key": le_key,
        "source_b128_key": b_key,
        **strain_meta,
        "integration_weight_source": "macro16_x16_source128_standard_rule",
        "requested_integration_weight_source": weight_key,
        "integration_weight_coordinate": "hat-dimensionless",
        "integration_weight_phys_coordinate": "physical-volume",
        "integration_weight_rule": "macro16_x16_standard_source128",
        "requested_integration_weight_rule": "macro16_x16_standard_source128" if weight_key == "macro16_x16_source128_standard_rule" else "source_teacher_volume",
        "label_source": "128-IP TRUE176/CSS8 teacher labels kept as Macro16 source128 point set",
        "model_visible_arrays": ["q48_def_hat", "X16_hat", "macro16_point_xi/source128 point features"],
        "rigid_motion_removed_by_preprocessing": True,
        "force_audit_arrays": ["B_macro_qraw", "integration_weight_phys"],
        "fine_grid_geometry_visible_to_model": False,
        "writes_X_macro": False,
        "writes_css8_internal_geometry": False,
        "case_ids": sorted(np.unique(case_id).astype(np.int64).tolist()),
        "q48_shape": list(q48.shape),
        "q48_hat_shape": list(q48_hat.shape),
        "q48_def_raw_shape": list(q48_def_raw.shape),
        "q48_def_hat_shape": list(q48_def_hat.shape),
        "rigid_projection_P_shape": list(rigid_projection_p.shape),
        "X16_raw_shape": list(x16.shape),
        "X16_hat_shape": list(x16_hat.shape),
        "LE_macro_shape": list(le128.shape),
        "B_macro_qraw_shape": list(b128_qraw.shape),
        "B_macro_qdef_shape": list(b128_qdef.shape),
        "B_macro_qhat_shape": list(b128_qhat.shape),
        "integration_weight_hat_shape": list(weights_hat.shape),
        "integration_weight_phys_shape": list(weights_phys.shape),
        "integration_weight_hat_sum_min": float(np.min(np.sum(weights_hat, axis=1))),
        "integration_weight_hat_sum_max": float(np.max(np.sum(weights_hat, axis=1))),
        "integration_weight_phys_sum_min": float(np.min(np.sum(weights_phys, axis=1))),
        "integration_weight_phys_sum_max": float(np.max(np.sum(weights_phys, axis=1))),
        "L_ref_min": float(np.min(l_ref)),
        "L_ref_max": float(np.max(l_ref)),
        "scale_consistency": scale_audit,
        "rigid_consistency": rigid_audit,
        **ip_key_meta,
    }
    summary_path = out_path.with_suffix(".summary.json")
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def build_all(args: argparse.Namespace) -> dict[str, Any]:
    paths = collect_paths(args.compact, args.compact_list, case_limit=args.case_limit)
    if not paths:
        raise ValueError("no compact inputs provided")
    out_root = Path(args.out_root).resolve()
    rows = [
        build_one(
            path,
            out_root,
            source_index=i,
            frame_stride=int(args.frame_stride),
            max_frames_per_compact=int(args.max_frames_per_compact),
            allow_missing_ip_keys=bool(args.allow_missing_ip_keys),
            source_node_order=str(args.source_node_order),
            strain_coordinate_mode=str(args.strain_coordinate_mode),
            volume_weight_mode=str(args.volume_weight_mode),
        )
        for i, path in enumerate(paths)
    ]
    list_path = out_root / "macro16_source128_teacher_compact_list.txt"
    list_path.write_text("\n".join(str(row["macro16_compact"]) for row in rows) + "\n", encoding="utf-8")
    summary = {
        "script": "build_macro16_source128_teacher",
        "standard_operator_contract_version": MACRO16_CONTRACT_VERSION,
        "macro16_teacher_contract_version": SOURCE128_CONTRACT_VERSION,
        "source_compact_count": int(len(paths)),
        "macro16_compact_count": int(len(rows)),
        "total_frame_count": int(sum(int(row["frame_count"]) for row in rows)),
        "point_count": 128,
        "compact_list": str(list_path),
        "label_source": "128-IP TRUE176/CSS8 teacher labels kept; no 18-point reduction",
        "fine_grid_geometry_visible_to_model": False,
        "writes_X_macro": False,
        "cases": rows,
    }
    summary_path = out_root / "macro16_source128_teacher_summary.json"
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "compact_list": str(list_path),
                "macro16_compact_count": int(len(rows)),
                "total_frame_count": int(summary["total_frame_count"]),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=json_default,
        )
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="append", type=Path, default=[])
    parser.add_argument("--compact-list", action="append", type=Path, default=[])
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-compact", type=int, default=0)
    parser.add_argument("--allow-missing-ip-keys", action="store_true")
    parser.add_argument("--strain-coordinate-mode", default="global-to-macro-local", choices=("global-to-macro-local", "source-local"))
    parser.add_argument("--source-node-order", default="auto", choices=("auto", "macro16", "true176-keep"))
    parser.add_argument("--volume-weight-mode", default="auto", choices=("auto", "ip-ivol", "inferred-dle", "macro16-x16"))
    return parser.parse_args()


def main() -> None:
    build_all(parse_args())


if __name__ == "__main__":
    main()
