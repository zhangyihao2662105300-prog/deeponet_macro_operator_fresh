#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Gate 05 Macro16 integration-point reduction audit.

This script does not train a network and does not modify the model.  It audits
fixed parent-coordinate candidate rules against the current Macro16 source128
teacher compacts using the selected-frame physical-volume convention.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_macro16_force_stiffness import (  # noqa: E402
    assemble_force_stiffness,
    load_macro,
    load_source,
    metric,
    source_stiffness_to_macro_subset,
    source_vectors_to_macro,
    stiffness_symmetry,
)
from build_macro16_from_128_teacher import macro16_from_128_interpolation_matrix  # noqa: E402
from macro_deeponet.macro16_geometry import (  # noqa: E402
    Macro16GeometryMap,
    Macro16PointTable,
    macro16_source128_point_table,
    macro16_standard_point_table,
)

RULE_ORDER = (128, 96, 64, 32, 18)
FORCE_PASS_MAX = 0.02
DEFAULT_FORCE_REF_RMS_FLOOR = 1.0e-10


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


def read_path_list(path: Path) -> list[Path]:
    return [
        Path(line.strip()).resolve()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def scalar_text(value: Any) -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def rel_norm(diff: np.ndarray, ref: np.ndarray) -> float:
    den = max(float(np.linalg.norm(np.asarray(ref, dtype=np.float64).reshape(-1))), 1.0e-30)
    return float(np.linalg.norm(np.asarray(diff, dtype=np.float64).reshape(-1)) / den)


def max_abs(arr: np.ndarray) -> float:
    vals = np.asarray(arr, dtype=np.float64)
    return float(np.max(np.abs(vals))) if vals.size else 0.0


def stats(values: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {
        "count": int(len(vals)),
        "mean": float(np.mean(vals)) if vals else None,
        "max": float(np.max(vals)) if vals else None,
        "min": float(np.min(vals)) if vals else None,
    }


def point_table_hash(xi: np.ndarray) -> str:
    vals = np.asarray(xi, dtype=np.float64).reshape(-1, 3)
    return hashlib.sha256(vals.tobytes()).hexdigest()[:16]


def inplane_keep(ex: int, ey: int, per_thickness_count: int) -> list[int]:
    if per_thickness_count == 4:
        return [0, 1, 2, 3]
    parity = (int(ex) + int(ey)) % 2
    rotating = (int(ex) + 2 * int(ey)) % 4
    if per_thickness_count == 3:
        return [i for i in range(4) if i != rotating]
    if per_thickness_count == 2:
        return [0, 3] if parity == 0 else [1, 2]
    if per_thickness_count == 1:
        return [rotating]
    raise ValueError(f"unsupported per-thickness count {per_thickness_count}")


def subset_rule(point_count: int, source_table: Macro16PointTable) -> dict[str, Any]:
    if int(point_count) not in {128, 96, 64, 32}:
        raise ValueError("subset_rule supports only 128, 96, 64, or 32")
    per_thickness_count = int(point_count) // (16 * 2)
    keep_source: list[int] = []
    target_by_source = np.full(128, -1, dtype=np.int64)

    for ey in range(4):
        for ex in range(4):
            cell = ey * 4 + ex
            keep_inplane = inplane_keep(ex, ey, per_thickness_count)
            for lt in range(2):
                kept_for_slice: list[int] = []
                for q2 in keep_inplane:
                    lr = q2 % 2
                    ls = q2 // 2
                    ip = lr + 2 * ls + 4 * lt
                    source_index = cell * 8 + ip
                    target_index = len(keep_source)
                    keep_source.append(source_index)
                    target_by_source[source_index] = target_index
                    kept_for_slice.append(source_index)
                kept_xi = np.asarray(source_table.xi[kept_for_slice], dtype=np.float64)
                for q2 in range(4):
                    lr = q2 % 2
                    ls = q2 // 2
                    ip = lr + 2 * ls + 4 * lt
                    source_index = cell * 8 + ip
                    if target_by_source[source_index] >= 0:
                        continue
                    xi = np.asarray(source_table.xi[source_index], dtype=np.float64)
                    local = int(np.argmin(np.linalg.norm(kept_xi - xi.reshape(1, 3), axis=1)))
                    target_by_source[source_index] = target_by_source[kept_for_slice[local]]

    keep = np.asarray(keep_source, dtype=np.int64)
    if keep.size != int(point_count):
        raise RuntimeError(f"bad rule {point_count}: keep size {keep.size}")
    if np.any(target_by_source < 0):
        raise RuntimeError(f"bad rule {point_count}: unassigned source points")
    parent_weights = np.zeros(keep.size, dtype=np.float64)
    for source_index, target_index in enumerate(target_by_source.tolist()):
        parent_weights[int(target_index)] += float(source_table.weights[source_index])
    return {
        "point_count": int(point_count),
        "rule_name": f"subset_{point_count}_from_source128_balanced_cell_slices",
        "rule_kind": "source128_subset_with_selected_volume_aggregation",
        "xi": source_table.xi[keep].astype(np.float64),
        "parent_weights": parent_weights,
        "keep_indices": keep,
        "source_to_rule_assignment": target_by_source,
        "label_mode": "take_source128_at_kept_points",
        "weight_mode": "aggregate_source128_selected_frame_volume_to_kept_points",
    }


def nearest_assignment(source_xi: np.ndarray, target_xi: np.ndarray) -> np.ndarray:
    src = np.asarray(source_xi, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(target_xi, dtype=np.float64).reshape(-1, 3)
    d2 = np.sum((src[:, None, :] - tgt[None, :, :]) ** 2, axis=2)
    return np.argmin(d2, axis=1).astype(np.int64)


def standard18_rule(source_table: Macro16PointTable) -> dict[str, Any]:
    table18 = macro16_standard_point_table(plane_order=3, thickness_order=2)
    assignment = nearest_assignment(source_table.xi, table18.xi)
    parent_weights = np.zeros(table18.xi.shape[0], dtype=np.float64)
    for source_index, target_index in enumerate(assignment.tolist()):
        parent_weights[int(target_index)] += float(source_table.weights[source_index])
    interp = macro16_from_128_interpolation_matrix(table18)
    return {
        "point_count": 18,
        "rule_name": "standard_3x3x2_interpolated_from_source128_failure_control",
        "rule_kind": "standard18_interpolated_failure_control",
        "xi": table18.xi.astype(np.float64),
        "parent_weights": parent_weights,
        "keep_indices": None,
        "source_to_rule_assignment": assignment,
        "interpolation_matrix": interp,
        "label_mode": "linear_parent_interpolation_from_source128",
        "weight_mode": "nearest_aggregate_source128_selected_frame_volume",
    }


def build_rules(out_root: Path) -> dict[int, dict[str, Any]]:
    source_table = macro16_source128_point_table()
    rules: dict[int, dict[str, Any]] = {}
    for count in RULE_ORDER:
        rule = standard18_rule(source_table) if count == 18 else subset_rule(count, source_table)
        rule["xi_hash"] = point_table_hash(rule["xi"])
        rules[int(count)] = rule
    point_root = out_root / "point_rules"
    point_root.mkdir(parents=True, exist_ok=True)
    for count, rule in rules.items():
        np.savez_compressed(
            point_root / f"macro16_gate05_rule_{count}.npz",
            point_count=np.asarray(int(count), dtype=np.int64),
            rule_name=np.asarray(rule["rule_name"], dtype=object),
            rule_kind=np.asarray(rule["rule_kind"], dtype=object),
            xi=np.asarray(rule["xi"], dtype=np.float64),
            parent_weights=np.asarray(rule["parent_weights"], dtype=np.float64),
            keep_indices=np.asarray([] if rule["keep_indices"] is None else rule["keep_indices"], dtype=np.int64),
            source_to_rule_assignment=np.asarray(rule["source_to_rule_assignment"], dtype=np.int64),
            xi_hash=np.asarray(rule["xi_hash"], dtype=object),
            label_mode=np.asarray(rule["label_mode"], dtype=object),
            weight_mode=np.asarray(rule["weight_mode"], dtype=object),
        )
    return rules


def aggregate_weights(source_weights: np.ndarray, assignment: np.ndarray, point_count: int) -> np.ndarray:
    weights = np.asarray(source_weights, dtype=np.float64)
    assign = np.asarray(assignment, dtype=np.int64).reshape(-1)
    if weights.ndim != 2 or weights.shape[1] != assign.size:
        raise ValueError(f"source selected weights must be [N,{assign.size}], got {weights.shape}")
    out = np.zeros((weights.shape[0], int(point_count)), dtype=np.float64)
    for source_index, target_index in enumerate(assign.tolist()):
        out[:, int(target_index)] += weights[:, source_index]
    return out


def reduced_labels(macro: dict[str, Any], rule: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    le = np.asarray(macro["le"], dtype=np.float64)
    b = np.asarray(macro["b"], dtype=np.float64)
    if le.shape[1] != 128 or b.shape[1] != 128:
        raise ValueError(f"Gate 05 expects source128 macro labels, got LE {le.shape} B {b.shape}")
    keep = rule.get("keep_indices")
    if keep is not None:
        idx = np.asarray(keep, dtype=np.int64)
        return le[:, idx, :], b[:, idx, :, :]
    w = np.asarray(rule["interpolation_matrix"], dtype=np.float64)
    return np.einsum("pr,nra->npa", w, le), np.einsum("pr,nraj->npaj", w, b)


def detj_report(x16: np.ndarray, rule: dict[str, Any]) -> dict[str, Any]:
    xi = np.asarray(rule["xi"], dtype=np.float64)
    weights = np.asarray(rule["parent_weights"], dtype=np.float64)
    table = Macro16PointTable(xi=xi, weights=weights)
    det_min = math.inf
    det_max = -math.inf
    for nodes in np.asarray(x16, dtype=np.float64).reshape(-1, 16, 3):
        fields = Macro16GeometryMap(nodes).eval_points(table)
        det = np.asarray(fields["detJ_hat"], dtype=np.float64)
        det_min = min(det_min, float(np.min(det)))
        det_max = max(det_max, float(np.max(det)))
    return {"detJ_min": det_min, "detJ_max": det_max, "detJ_positive": bool(det_min > 0.0)}


def compact_point_contract(path: Path, macro: dict[str, Any]) -> dict[str, Any]:
    fields_missing: list[str] = []
    with np.load(str(path), allow_pickle=True) as z:
        for key in ("q48_raw", "q48_def_hat", "X16_raw", "X16_hat", "LE_macro", "B_macro_qraw", "B_macro_qdef"):
            if key not in z.files:
                fields_missing.append(key)
        forbidden_visible = [key for key in ("X_macro", "css8_connectivity_zero_based", "css8_point_table") if key in z.files]
        point_xi = np.asarray(z["macro16_point_xi"], dtype=np.float64) if "macro16_point_xi" in z.files else None
        fine_grid_visible = bool(np.asarray(z["fine_grid_geometry_visible_to_model"]).reshape(-1)[0]) if "fine_grid_geometry_visible_to_model" in z.files else False
        b_coord = scalar_text(z["B_label_q_coordinate"]) if "B_label_q_coordinate" in z.files else ""
    source128 = macro16_source128_point_table().xi
    point_match = point_xi is not None and point_xi.shape == source128.shape and np.allclose(point_xi, source128, rtol=1.0e-7, atol=1.0e-7)
    q = np.asarray(macro["q"], dtype=np.float64)
    x16 = np.asarray(macro["x16"], dtype=np.float64)
    le = np.asarray(macro["le"], dtype=np.float64)
    b = np.asarray(macro["b"], dtype=np.float64)
    checks = {
        "required_fields_present": not fields_missing,
        "missing_fields": fields_missing,
        "q48_raw_shape_ok": bool(q.ndim == 2 and q.shape[1] == 48),
        "X16_shape_ok": bool(x16.shape == (q.shape[0], 16, 3)),
        "LE_source128_shape_ok": bool(le.shape == (q.shape[0], 128, 6)),
        "B_source128_shape_ok": bool(b.shape == (q.shape[0], 128, 6, 48)),
        "macro16_point_xi_matches_source128": bool(point_match),
        "forbidden_visible_fields": forbidden_visible,
        "fine_grid_geometry_visible_to_model": bool(fine_grid_visible),
        "B_label_q_coordinate": b_coord,
        "B_label_qraw": b_coord in {"q48_raw", "q-48-raw", "qraw", ""},
    }
    checks["passed"] = bool(
        checks["required_fields_present"]
        and checks["q48_raw_shape_ok"]
        and checks["X16_shape_ok"]
        and checks["LE_source128_shape_ok"]
        and checks["B_source128_shape_ok"]
        and checks["macro16_point_xi_matches_source128"]
        and not checks["forbidden_visible_fields"]
        and not checks["fine_grid_geometry_visible_to_model"]
        and checks["B_label_qraw"]
    )
    return checks


def compact_case_ids(path: Path) -> list[int]:
    with np.load(str(path), allow_pickle=True) as z:
        if "case_id" not in z.files:
            return []
        vals = np.asarray(z["case_id"], dtype=np.int64).reshape(-1)
        return sorted(np.unique(vals).astype(int).tolist())


def audit_compact_rule(path: Path, rule: dict[str, Any]) -> dict[str, Any]:
    macro = load_macro(path)
    source_path = Path(str(macro["source_compact"]))
    source = load_source(source_path, np.asarray(macro["source_rows"], dtype=np.int64))
    if "source_selected_frame_volume" not in source:
        raise KeyError(f"{source_path}: missing ip_IVOL_abaqus_selected_frames")
    source_order = str(macro["source_node_order"])
    source_rf_macro = source_vectors_to_macro(np.asarray(source["rf"], dtype=np.float64), source_order)
    le_rule, b_rule = reduced_labels(macro, rule)
    weights_rule = aggregate_weights(
        np.asarray(source["source_selected_frame_volume"], dtype=np.float64),
        np.asarray(rule["source_to_rule_assignment"], dtype=np.int64),
        int(rule["point_count"]),
    )
    if not np.all(np.isfinite(weights_rule)) or np.any(weights_rule <= 0.0):
        raise ValueError(f"{path}: reduced selected-frame weights are non-finite or non-positive")
    force, stiffness, energy_twice = assemble_force_stiffness(
        le_rule,
        b_rule,
        weights_rule,
        np.asarray(source["elastic_D"], dtype=np.float64),
    )
    kfd_subset = None
    dirs_macro = None
    material_report: dict[str, Any] = {"available": False, "reason": "source compact lacks RF_projected_plus"}
    if "rf_plus" in source:
        if source.get("delta") is None:
            raise KeyError(f"{source_path}: RF_projected_plus requires delta")
        kfd_subset, dirs_macro = source_stiffness_to_macro_subset(
            np.asarray(source["rf"], dtype=np.float64),
            np.asarray(source["rf_plus"], dtype=np.float64),
            np.asarray(source["perturb_directions"], dtype=np.int64),
            float(source["delta"]),
            source_order,
        )
        material_report = {
            "available": True,
            "definition": "material-only B^T D B dV versus Abaqus RF finite-difference tangent; diagnostic only",
            "direction_count": int(dirs_macro.size),
            **metric(stiffness[:, :, dirs_macro], kfd_subset),
        }
    energy_report = {
        "macro_q_dot_F_vs_integral_LE_sigma": metric(
            np.einsum("nj,nj->n", np.asarray(macro["q"], dtype=np.float64), force),
            energy_twice,
        ),
        "source_q_dot_RF_vs_macro_q_dot_F": metric(
            np.einsum("nj,nj->n", np.asarray(macro["q"], dtype=np.float64), force),
            np.einsum("nj,nj->n", np.asarray(macro["q"], dtype=np.float64), source_rf_macro),
        ),
    }
    contract = compact_point_contract(path, macro)
    detj = detj_report(np.asarray(macro["x16"], dtype=np.float64), rule)
    force_report = metric(force, source_rf_macro)
    return {
        "compact": str(path),
        "source_compact": str(source_path),
        "case_ids": compact_case_ids(path),
        "frame_count": int(np.asarray(macro["q"]).shape[0]),
        "point_count": int(rule["point_count"]),
        "data_contract": contract,
        "detJ": detj,
        "selected_volume_sum_min": float(np.min(np.sum(weights_rule, axis=1))),
        "selected_volume_sum_max": float(np.max(np.sum(weights_rule, axis=1))),
        "force": force_report,
        "effective_force_for_relative_gate": bool(float(force_report["ref_rms"]) > DEFAULT_FORCE_REF_RMS_FLOOR),
        "force_ref_rms_floor": DEFAULT_FORCE_REF_RMS_FLOOR,
        "force_dof_max_abs_error": float(np.max(np.abs(force - source_rf_macro), axis=None)),
        "material_only_stiffness": material_report,
        "material_only_stiffness_symmetry": stiffness_symmetry(stiffness),
        "energy": energy_report,
        "rule_label_mode": rule["label_mode"],
        "rule_weight_mode": rule["weight_mode"],
    }


def _audit_job(job: tuple[str, str, int, dict[str, Any]]) -> tuple[str, int, dict[str, Any]]:
    class_name, path_text, rule_count, rule = job
    return class_name, rule_count, audit_compact_rule(Path(path_text), rule)


def aggregate_rule_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    force_vals_all = [float(row["force"]["rel"]) for row in rows]
    force_vals = [float(row["force"]["rel"]) for row in rows if bool(row.get("effective_force_for_relative_gate", True))]
    k_vals = [
        float(row["material_only_stiffness"]["rel"])
        for row in rows
        if row.get("material_only_stiffness", {}).get("available")
    ]
    sym_vals = [float(row["material_only_stiffness_symmetry"]["rel"]) for row in rows]
    det_vals = [float(row["detJ"]["detJ_min"]) for row in rows]
    return {
        "compact_count": int(len(rows)),
        "frame_count": int(sum(int(row["frame_count"]) for row in rows)),
        "case_ids": sorted({case for row in rows for case in row.get("case_ids", [])}),
        "data_contract_pass": bool(all(bool(row["data_contract"]["passed"]) for row in rows)),
        "detJ_positive": bool(all(bool(row["detJ"]["detJ_positive"]) for row in rows)),
        "detJ_min": float(np.min(det_vals)) if det_vals else None,
        "force_rel_all": stats(force_vals_all),
        "force_rel_effective": stats(force_vals),
        "near_zero_force_excluded_count": int(len(force_vals_all) - len(force_vals)),
        "force_pass": bool(force_vals and max(force_vals) < FORCE_PASS_MAX),
        "material_only_k_rel": stats(k_vals),
        "material_only_k_symmetry_rel": stats(sym_vals),
    }


def default_class_lists(root: Path) -> dict[str, Path]:
    return {
        "regular": root / "runs" / "gate04_generality" / "regular_compact_list.txt",
        "lightly_distorted": root / "runs" / "gate04_generality" / "lightly_distorted_compact_list.txt",
        "moderately_distorted": root / "runs" / "gate04_generality" / "moderately_distorted_compact_list.txt",
        "cylindrical_shell": root
        / "runs"
        / "gate04_wind_shell_generality"
        / "audits"
        / "cylindrical_shell"
        / "cylindrical_shell_macro16_source128_compact_list.txt",
        "conical_shell": root
        / "runs"
        / "gate04_wind_shell_generality"
        / "audits"
        / "conical_shell"
        / "conical_shell_macro16_source128_compact_list.txt",
        "thickness_varying_shell": root
        / "runs"
        / "gate04_wind_shell_generality"
        / "audits"
        / "thickness_varying_shell"
        / "thickness_varying_shell_macro16_source128_compact_list.txt",
        "mild_double_curvature_shell": root
        / "runs"
        / "gate04_wind_shell_generality"
        / "audits"
        / "mild_double_curvature_shell"
        / "mild_double_curvature_shell_macro16_source128_compact_list.txt",
    }


def parse_class_list_args(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError("--class-list must be NAME=PATH")
        name, path = raw.split("=", 1)
        out[name.strip()] = Path(path.strip()).resolve()
    return out


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Gate 05 IP Reduction Audit")
    lines.append("")
    lines.append(f"Date: {summary['date']}")
    lines.append("")
    lines.append("Role: Gate 05 IP Reduction Agent")
    lines.append("")
    lines.append("## 1. Task Goal")
    lines.append("")
    lines.append("开始 Macro16 source128 积分点降阶审计。")
    lines.append("")
    lines.append("本轮不训练网络，不改模型，不改 `q48` 顺序，不改 `LE` 顺序，不改当前标准 128 点积分规则。Material-only stiffness 只记录为 diagnostic，不作为 Gate 05 通过门槛。")
    lines.append("")
    lines.append("## 2. Data Used")
    lines.append("")
    for class_name, meta in summary["classes"].items():
        lines.append(f"- {class_name}: `{meta['compact_list']}`")
    lines.append("")
    lines.append("## 3. Commands Used")
    lines.append("")
    lines.append("```powershell")
    lines.append(summary["command"])
    lines.append("```")
    lines.append("")
    lines.append("## 4. Candidate Rules")
    lines.append("")
    lines.append("| Rule | Point Count | Rule Kind | Label Mode | Weight Mode | Xi Hash |")
    lines.append("|---|---:|---|---|---|---|")
    for count in RULE_ORDER:
        rule = summary["rules"][str(count)]
        lines.append(
            f"| {count} | {count} | {rule['rule_kind']} | {rule['label_mode']} | {rule['weight_mode']} | `{rule['xi_hash']}` |"
        )
    lines.append("")
    lines.append("说明：96、64、32 是 Gate 05 候选审计点表；128 仍是当前标准规则；18 是失败对照。")
    lines.append("")
    lines.append("## 5. Pass Standard")
    lines.append("")
    lines.append("Gate 05 当前通过标准只看 selected-frame volume 下的恢复力闭合：")
    lines.append("")
    lines.append("```text")
    lines.append("force max relative error < 0.02")
    lines.append("```")
    lines.append("")
    lines.append("Material-only K 只记录，不作为 full tangent 门槛。")
    lines.append("")
    lines.append("近零反力帧使用 Gate 04 相同口径排除在 relative force 门槛外，阈值为 `force_ref_rms <= 1e-10`；全量结果仍保存在 JSON 明细中。")
    lines.append("")
    lines.append("## 6. Overall Result")
    lines.append("")
    lines.append("| Rule | Data Contract | detJ Positive | Compact Count | Frame Count | Effective Force Count | Near-Zero Excluded | Force Mean | Force Max | Force Pass | Material-only K Mean | Material-only K Max |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for count in RULE_ORDER:
        row = summary["overall"][str(count)]
        fmean = row["force_rel_effective"]["mean"]
        fmax = row["force_rel_effective"]["max"]
        kmean = row["material_only_k_rel"]["mean"]
        kmax = row["material_only_k_rel"]["max"]
        lines.append(
            "| "
            f"{count} | {pass_text(row['data_contract_pass'])} | {pass_text(row['detJ_positive'])} | "
            f"{row['compact_count']} | {row['frame_count']} | {row['force_rel_effective']['count']} | "
            f"{row['near_zero_force_excluded_count']} | {fmt(fmean)} | {fmt(fmax)} | {pass_text(row['force_pass'])} | "
            f"{fmt(kmean)} | {fmt(kmax)} |"
        )
    lines.append("")
    lines.append("## 7. Per-Class Result")
    lines.append("")
    lines.append("| Class | Rule | Geometry Count | Frame Count | Effective Force Count | Near-Zero Excluded | detJ Positive | Data Contract | Force Mean | Force Max | Force Pass | Material-only K Mean | Material-only K Max |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---:|---:|")
    for class_name, class_summary in summary["classes"].items():
        for count in RULE_ORDER:
            row = class_summary["rules"][str(count)]
            fmean = row["force_rel_effective"]["mean"]
            fmax = row["force_rel_effective"]["max"]
            kmean = row["material_only_k_rel"]["mean"]
            kmax = row["material_only_k_rel"]["max"]
            lines.append(
                "| "
                f"{class_name} | {count} | {row['compact_count']} | {row['frame_count']} | "
                f"{row['force_rel_effective']['count']} | {row['near_zero_force_excluded_count']} | "
                f"{pass_text(row['detJ_positive'])} | {pass_text(row['data_contract_pass'])} | "
                f"{fmt(fmean)} | {fmt(fmax)} | {pass_text(row['force_pass'])} | "
                f"{fmt(kmean)} | {fmt(kmax)} |"
            )
    lines.append("")
    lines.append("## 8. Gate Result")
    lines.append("")
    lines.append(summary["gate_result_text"])
    lines.append("")
    lines.append("## 9. Unresolved Issues")
    lines.append("")
    lines.append("1. Material-only K 仍只是 diagnostic，不代表完整 Abaqus tangent。")
    lines.append("2. 96、64、32 当前是候选降阶点表，不能替代 128 标准规则，除非后续正式决策采用。")
    lines.append("3. 18 点继续作为失败对照，不阻塞。")
    lines.append("")
    lines.append("## 10. Recommended Next Action")
    lines.append("")
    lines.append(summary["recommended_next_action"])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"`{float(value):.9g}`"


def pass_text(value: Any) -> str:
    return "PASS" if bool(value) else "FAIL"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rules = build_rules(out_root)
    class_lists = default_class_lists(ROOT)
    class_lists.update(parse_class_list_args(args.class_list))
    class_paths: dict[str, list[Path]] = {}
    for name, list_path in class_lists.items():
        if not list_path.exists():
            raise FileNotFoundError(f"missing class list for {name}: {list_path}")
        paths = read_path_list(list_path)
        if int(args.case_limit) > 0:
            paths = paths[: int(args.case_limit)]
        class_paths[name] = paths

    jobs: list[tuple[str, str, int, dict[str, Any]]] = []
    for class_name, paths in class_paths.items():
        for path in paths:
            for count in RULE_ORDER:
                jobs.append((class_name, str(path), int(count), rules[int(count)]))

    workers = max(1, int(args.workers))
    results: dict[str, dict[int, list[dict[str, Any]]]] = {
        class_name: {int(count): [] for count in RULE_ORDER} for class_name in class_paths
    }
    errors: list[dict[str, Any]] = []
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            try:
                class_name, count, row = _audit_job(job)
                results[class_name][int(count)].append(row)
            except Exception as exc:
                errors.append({"class": job[0], "compact": job[1], "rule": job[2], "error": repr(exc)})
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            futures = {pool.submit(_audit_job, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    class_name, count, row = future.result()
                    results[class_name][int(count)].append(row)
                except Exception as exc:
                    errors.append({"class": job[0], "compact": job[1], "rule": job[2], "error": repr(exc)})

    class_summaries: dict[str, Any] = {}
    for class_name, paths in class_paths.items():
        class_rules: dict[str, Any] = {}
        for count in RULE_ORDER:
            rows = results[class_name][int(count)]
            class_rules[str(count)] = aggregate_rule_rows(rows) if rows else {
                "compact_count": 0,
                "frame_count": 0,
                "case_ids": [],
                "data_contract_pass": False,
                "detJ_positive": False,
                "detJ_min": None,
                "force_rel": stats([]),
                "force_pass": False,
                "material_only_k_rel": stats([]),
                "material_only_k_symmetry_rel": stats([]),
            }
        class_summaries[class_name] = {
            "compact_list": str(class_lists[class_name]),
            "compact_paths": [str(p) for p in paths],
            "rules": class_rules,
        }

    overall: dict[str, Any] = {}
    for count in RULE_ORDER:
        all_rows = [row for class_name in class_paths for row in results[class_name][int(count)]]
        overall[str(count)] = aggregate_rule_rows(all_rows)

    passed_reduced = [
        count
        for count in (96, 64, 32)
        if overall[str(count)]["data_contract_pass"] and overall[str(count)]["detJ_positive"] and overall[str(count)]["force_pass"]
    ]
    if passed_reduced:
        smallest = min(passed_reduced)
        gate_text = (
            f"Gate 05 force-only reduction audit: PASS for reduced candidate rules {passed_reduced}. "
            f"Smallest passing reduced candidate is {smallest} points. "
            "This does not adopt the rule automatically; 128 remains the active standard rule until an explicit project decision."
        )
        next_action = (
            f"Keep 128 as the active rule for now, but carry {smallest}-point and larger passing candidates into the next formal decision. "
            "Do not train until the remaining upstream gates are resolved or waived."
        )
    else:
        gate_text = (
            "Gate 05 force-only reduction audit: no reduced candidate passed the current force max threshold. "
            "Keep 128 points as the active rule."
        )
        next_action = "Keep 128 points and avoid network training; if reduction is still desired, design a new candidate rule instead of changing the model."

    summary: dict[str, Any] = {
        "script": "run_gate05_ip_reduction_audit",
        "date": "2026-06-25",
        "command": " ".join(sys.argv),
        "out_root": str(out_root),
        "workers": workers,
        "force_pass_max": FORCE_PASS_MAX,
        "rules": {
            str(count): {
                "point_count": int(count),
                "rule_name": str(rule["rule_name"]),
                "rule_kind": str(rule["rule_kind"]),
                "label_mode": str(rule["label_mode"]),
                "weight_mode": str(rule["weight_mode"]),
                "xi_hash": str(rule["xi_hash"]),
                "point_table": str(out_root / "point_rules" / f"macro16_gate05_rule_{count}.npz"),
            }
            for count, rule in rules.items()
        },
        "classes": class_summaries,
        "overall": overall,
        "errors": errors,
        "strict_success": bool(not errors),
        "gate_result_text": gate_text,
        "recommended_next_action": next_action,
        "material_only_k_policy": "diagnostic only; not a full tangent pass/fail gate",
        "network_training": False,
        "model_changed": False,
        "q48_order_changed": False,
        "LE_order_changed": False,
        "standard_128_rule_changed": False,
    }

    summary_path = out_root / "gate05_ip_reduction_summary.json"
    write_json(summary_path, summary)
    detail_path = out_root / "gate05_ip_reduction_details.json"
    write_json(
        detail_path,
        {
            "script": "run_gate05_ip_reduction_audit",
            "results": results,
            "errors": errors,
        },
    )
    report_path = Path(args.report).resolve()
    write_markdown_report(report_path, summary)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "details": str(detail_path),
                "report": str(report_path),
                "errors": len(errors),
                "overall_force_max": {str(count): overall[str(count)]["force_rel_effective"]["max"] for count in RULE_ORDER},
            },
            ensure_ascii=False,
            sort_keys=True,
            default=json_default,
        )
    )
    if errors and bool(args.strict):
        raise SystemExit(1)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs" / "gate05_ip_reduction")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "05_ip_reduction_audit.md")
    parser.add_argument("--class-list", action="append", default=[], help="Override/add class list as NAME=PATH")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
