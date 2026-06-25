#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit Macro16 source128 data readiness for the limited LE/B training gate.

This script is read-only with respect to compact data.  It does not train a
network, change model code, change q48/LE ordering, or modify the 128-point
integration rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from macro_deeponet.macro16_geometry import MACRO16_CONTRACT_VERSION, macro16_source128_point_table  # noqa: E402
from macro_deeponet.models import Macro16BoundaryDeepONetWithLE0  # noqa: E402,F401
from macro_deeponet.train_macro16_boundary_sobolev import load_macro16_compacts, split_indices  # noqa: E402


REQUIRED_FIELDS = (
    "q48_def_hat",
    "X16_hat",
    "LE_macro",
    "B_macro_qdef",
    "macro16_point_xi",
    "case_id",
)
FORBIDDEN_MODEL_VISIBLE_FIELDS = (
    "X_macro",
    "css8_connectivity_zero_based",
    "css8_point_table",
)


DEFAULT_CATEGORY_LISTS = {
    "regular": ROOT / "runs" / "gate04_generality" / "regular_compact_list.txt",
    "lightly_distorted": ROOT / "runs" / "gate04_generality" / "lightly_distorted_compact_list.txt",
    "moderately_distorted": ROOT / "runs" / "gate04_generality" / "moderately_distorted_compact_list.txt",
    "strong_non_flipped_repaired": Path(
        r"D:\IS-FEM\outputs\macro16_source128_generality_case061_repair\postprocess"
        r"\macro16_source128_x16weights\macro16_source128_teacher_compact_list.txt"
    ),
    "cylindrical_shell": ROOT
    / "runs"
    / "gate04_wind_shell_generality"
    / "audits"
    / "cylindrical_shell"
    / "cylindrical_shell_macro16_source128_compact_list.txt",
    "conical_shell": ROOT
    / "runs"
    / "gate04_wind_shell_generality"
    / "audits"
    / "conical_shell"
    / "conical_shell_macro16_source128_compact_list.txt",
    "thickness_varying_shell": ROOT
    / "runs"
    / "gate04_wind_shell_generality"
    / "audits"
    / "thickness_varying_shell"
    / "thickness_varying_shell_macro16_source128_compact_list.txt",
    "mild_double_curvature_shell": ROOT
    / "runs"
    / "gate04_wind_shell_generality"
    / "audits"
    / "mild_double_curvature_shell"
    / "mild_double_curvature_shell_macro16_source128_compact_list.txt",
}


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
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def scalar_text(z: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    if arr.size == 0:
        return default
    value = arr.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def infer_frame_count(z: np.lib.npyio.NpzFile) -> int:
    for key in ("q48_def_hat", "X16_hat", "LE_macro", "B_macro_qdef", "q48_raw"):
        if key in z.files:
            return int(np.asarray(z[key]).shape[0])
    return 0


def geometry_hash(x16_hat_first: np.ndarray) -> str:
    rounded = np.round(np.asarray(x16_hat_first, dtype=np.float64), 10)
    return hashlib.sha1(rounded.tobytes()).hexdigest()[:12]


def unique_ints(arr: np.ndarray) -> list[int]:
    return sorted(np.unique(np.asarray(arr, dtype=np.int64).reshape(-1)).astype(int).tolist())


def compact_audit(path: Path) -> dict[str, Any]:
    source128 = macro16_source128_point_table().xi
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        out.update({"passed": False, "reason": "missing compact"})
        return out

    with np.load(str(path), allow_pickle=True) as z:
        files = set(z.files)
        missing = [key for key in REQUIRED_FIELDS if key not in files]
        forbidden = [key for key in FORBIDDEN_MODEL_VISIBLE_FIELDS if key in files]
        fine_grid_visible = (
            bool(np.asarray(z["fine_grid_geometry_visible_to_model"]).reshape(-1)[0])
            if "fine_grid_geometry_visible_to_model" in files
            else False
        )
        n = infer_frame_count(z)
        shapes = {key: list(np.asarray(z[key]).shape) for key in REQUIRED_FIELDS if key in files}
        shape_checks = {
            "q48_def_hat": bool("q48_def_hat" in files and np.asarray(z["q48_def_hat"]).shape == (n, 48)),
            "X16_hat": bool("X16_hat" in files and np.asarray(z["X16_hat"]).shape == (n, 16, 3)),
            "LE_macro": bool("LE_macro" in files and np.asarray(z["LE_macro"]).shape == (n, 128, 6)),
            "B_macro_qdef": bool("B_macro_qdef" in files and np.asarray(z["B_macro_qdef"]).shape == (n, 128, 6, 48)),
            "case_id": bool("case_id" in files and np.asarray(z["case_id"]).reshape(-1).shape[0] in {1, n}),
        }
        xi = np.asarray(z["macro16_point_xi"], dtype=np.float64) if "macro16_point_xi" in files else np.empty((0, 3))
        xi_matches = bool(xi.shape == source128.shape and np.allclose(xi, source128, rtol=1.0e-7, atol=1.0e-7))
        xi_max_abs = float(np.max(np.abs(xi - source128))) if xi.shape == source128.shape else None
        case_ids = unique_ints(np.asarray(z["case_id"])) if "case_id" in files else []
        geom_hash = geometry_hash(np.asarray(z["X16_hat"])[0]) if "X16_hat" in files and n > 0 else ""
        contract_version = scalar_text(z, "standard_operator_contract_version")
        b_coord = scalar_text(z, "B_macro_qdef_label_q_coordinate") or scalar_text(z, "B_label_q_coordinate")

    passed = bool(
        path.exists()
        and not missing
        and all(shape_checks.values())
        and xi_matches
        and not forbidden
        and not fine_grid_visible
        and (not contract_version or contract_version == MACRO16_CONTRACT_VERSION)
    )
    out.update(
        {
            "passed": passed,
            "frame_count": n,
            "case_ids": case_ids,
            "geometry_hash": geom_hash,
            "missing_required_fields": missing,
            "shapes": shapes,
            "shape_checks": shape_checks,
            "macro16_point_xi_matches_source128": xi_matches,
            "macro16_point_xi_max_abs": xi_max_abs,
            "forbidden_visible_fields": forbidden,
            "fine_grid_geometry_visible_to_model": fine_grid_visible,
            "standard_operator_contract_version": contract_version,
            "B_macro_qdef_label_q_coordinate": b_coord,
        }
    )
    return out


def loader_audit(paths: list[Path]) -> dict[str, Any]:
    try:
        data = load_macro16_compacts(
            [str(path) for path in paths],
            point_table=macro16_source128_point_table(),
            scale_mode="normalized",
            b_label_coordinate="auto",
        )
        train_idx, val_idx, split_meta = split_indices(data.case_id, val_fraction=0.2, val_cases="", seed=20260625)
    except Exception as exc:  # pragma: no cover - report path
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "passed": True,
        "model": "Macro16BoundaryDeepONetWithLE0",
        "q48_def_hat_shape": list(data.q48_hat.shape),
        "X16_hat_shape": list(data.x16_hat.shape),
        "point_features_hat_shape": list(data.point_features_hat.shape),
        "LE_macro_shape": list(data.le.shape),
        "B_macro_qdef_shape": list(data.b.shape),
        "weights_shape": list(data.weights.shape),
        "model_visible_q": data.point_meta.get("model_visible_q"),
        "model_visible_B": data.point_meta.get("model_visible_B"),
        "point_count": int(data.le.shape[1]),
        "case_ids": unique_ints(data.case_id),
        "split": {
            **split_meta,
            "train_frame_count": int(train_idx.size),
            "val_frame_count": int(val_idx.size),
        },
    }


def audit_category(name: str, list_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"name": name, "compact_list": str(list_path), "list_exists": list_path.exists()}
    if not list_path.exists():
        row.update({"passed": False, "reason": "missing compact list", "compacts": []})
        return row
    paths = read_path_list(list_path)
    compacts = [compact_audit(path) for path in paths]
    loader = loader_audit(paths) if paths else {"passed": False, "error": "empty compact list"}
    frame_count = int(sum(int(c.get("frame_count", 0)) for c in compacts))
    case_ids = sorted({cid for c in compacts for cid in c.get("case_ids", [])})
    geometry_hashes = sorted({str(c.get("geometry_hash")) for c in compacts if c.get("geometry_hash")})
    row.update(
        {
            "passed": bool(compacts and all(bool(c.get("passed")) for c in compacts) and loader.get("passed")),
            "compact_count": len(compacts),
            "frame_count": frame_count,
            "case_ids": case_ids,
            "case_count": len(case_ids),
            "geometry_hashes": geometry_hashes,
            "geometry_count": len(geometry_hashes),
            "compacts": compacts,
            "loader": loader,
        }
    )
    return row


def aggregate(categories: list[dict[str, Any]]) -> dict[str, Any]:
    all_compacts = [c for cat in categories for c in cat.get("compacts", [])]
    all_cases = sorted({cid for cat in categories for cid in cat.get("case_ids", [])})
    all_geoms = sorted({gh for cat in categories for gh in cat.get("geometry_hashes", [])})
    required_category_names = {
        "regular",
        "lightly_distorted",
        "moderately_distorted",
        "strong_non_flipped_repaired",
        "cylindrical_shell",
        "conical_shell",
        "thickness_varying_shell",
        "mild_double_curvature_shell",
    }
    present_category_names = {str(cat.get("name")) for cat in categories if int(cat.get("compact_count", 0)) > 0}
    return {
        "passed": bool(categories and all(bool(cat.get("passed")) for cat in categories)),
        "category_count": len(categories),
        "required_categories_present": bool(required_category_names.issubset(present_category_names)),
        "missing_required_categories": sorted(required_category_names.difference(present_category_names)),
        "compact_count": len(all_compacts),
        "frame_count": int(sum(int(c.get("frame_count", 0)) for c in all_compacts)),
        "case_ids": all_cases,
        "case_count": len(all_cases),
        "geometry_hashes": all_geoms,
        "geometry_count": len(all_geoms),
        "all_required_fields_present": bool(all(not c.get("missing_required_fields") for c in all_compacts)),
        "all_source128_rule_match": bool(all(bool(c.get("macro16_point_xi_matches_source128")) for c in all_compacts)),
        "no_forbidden_visible_fields": bool(all(not c.get("forbidden_visible_fields") for c in all_compacts)),
        "no_fine_grid_geometry_visible": bool(all(not c.get("fine_grid_geometry_visible_to_model") for c in all_compacts)),
    }


def split_readiness(categories: list[dict[str, Any]]) -> dict[str, Any]:
    case_ids = sorted({cid for cat in categories for cid in cat.get("case_ids", [])})
    category_to_cases = {str(cat.get("name")): list(cat.get("case_ids", [])) for cat in categories}
    geometry_to_categories: dict[str, list[str]] = {}
    for cat in categories:
        for geom_hash in cat.get("geometry_hashes", []):
            geometry_to_categories.setdefault(str(geom_hash), []).append(str(cat.get("name")))
    case_split_possible = len(case_ids) >= 2
    category_holdout_possible = len(categories) >= 2 and all(int(cat.get("frame_count", 0)) > 0 for cat in categories)
    suggested_val_cases: list[int] = []
    for key in ("cylindrical_shell", "conical_shell", "thickness_varying_shell", "mild_double_curvature_shell"):
        suggested_val_cases.extend(category_to_cases.get(key, []))
    if not suggested_val_cases:
        suggested_val_cases = category_to_cases.get("strong_non_flipped_repaired", [])
    return {
        "case_split_possible": case_split_possible,
        "geometry_or_category_holdout_possible": category_holdout_possible,
        "case_ids": case_ids,
        "category_to_cases": category_to_cases,
        "geometry_to_categories": geometry_to_categories,
        "recommended_split": "case/category isolated",
        "recommended_val_cases": sorted(set(int(v) for v in suggested_val_cases)),
        "frame_random_split_recommended": False,
        "note": "Use explicit val_cases for a real holdout. Case isolation is supported by the loader; category-level holdout should be chosen intentionally.",
    }


def md_bool(value: Any) -> str:
    return "PASS" if bool(value) else "FAIL"


def write_report(path: Path, payload: dict[str, Any], json_path: Path) -> None:
    lines: list[str] = []
    summary = payload["summary"]
    split = payload["split_readiness"]
    lines.append("# Gate 06 Training Data Readiness")
    lines.append("")
    lines.append("Date: 2026-06-25")
    lines.append("")
    lines.append("Role: Training Data Readiness Agent")
    lines.append("")
    lines.append("## 1. Task Goal")
    lines.append("")
    lines.append("训练前确认 Macro16 source128 训练数据是否完整可用。")
    lines.append("")
    lines.append("本轮不训练网络，不改模型，不改 `q48` 顺序，不改 `LE` 顺序，不改 128 点积分规则。")
    lines.append("")
    lines.append("## 2. Data Used")
    lines.append("")
    for cat in payload["categories"]:
        lines.append(f"- {cat['name']}: `{cat['compact_list']}`")
    lines.append("")
    lines.append("## 3. Commands Used")
    lines.append("")
    lines.append("```powershell")
    lines.append("py -3 -m py_compile scripts\\audit_macro16_training_data_readiness.py")
    lines.append("py -3 scripts\\audit_macro16_training_data_readiness.py --strict")
    lines.append("```")
    lines.append("")
    lines.append("JSON detail:")
    lines.append("")
    lines.append(f"`{json_path}`")
    lines.append("")
    lines.append("## 4. Overall Result")
    lines.append("")
    lines.append(f"- Status: {md_bool(summary['passed'])}")
    lines.append(f"- Compact count: `{summary['compact_count']}`")
    lines.append(f"- Frame count: `{summary['frame_count']}`")
    lines.append(f"- Case count: `{summary['case_count']}`")
    lines.append(f"- Geometry hash count: `{summary['geometry_count']}`")
    lines.append(f"- Required categories present: {md_bool(summary['required_categories_present'])}")
    lines.append(f"- Required fields present: {md_bool(summary['all_required_fields_present'])}")
    lines.append(f"- Standard 128 point rule match: {md_bool(summary['all_source128_rule_match'])}")
    lines.append(f"- No `X_macro` or CSS8 fine-grid geometry exposed to model: {md_bool(summary['no_forbidden_visible_fields'] and summary['no_fine_grid_geometry_visible'])}")
    lines.append("")
    lines.append("## 5. Category Coverage")
    lines.append("")
    lines.append("| Category | Compact Count | Frame Count | Case IDs | Geometry Count | Field Contract | 128 Rule | Loader Ready |")
    lines.append("|---|---:|---:|---|---:|---|---|---|")
    for cat in payload["categories"]:
        loader = cat.get("loader", {})
        rule_pass = all(bool(c.get("macro16_point_xi_matches_source128")) for c in cat.get("compacts", []))
        field_pass = all(not c.get("missing_required_fields") and all(c.get("shape_checks", {}).values()) for c in cat.get("compacts", []))
        lines.append(
            "| "
            f"{cat['name']} | {cat.get('compact_count', 0)} | {cat.get('frame_count', 0)} | "
            f"`{cat.get('case_ids', [])}` | {cat.get('geometry_count', 0)} | "
            f"{md_bool(field_pass)} | {md_bool(rule_pass)} | {md_bool(loader.get('passed'))} |"
        )
    lines.append("")
    lines.append("## 6. Required Field Checks")
    lines.append("")
    lines.append("Required model/data fields:")
    lines.append("")
    lines.append("1. `q48_def_hat`: present, shape `[N,48]`.")
    lines.append("2. `X16_hat`: present, shape `[N,16,3]`.")
    lines.append("3. `LE_macro`: present, shape `[N,128,6]`.")
    lines.append("4. `B_macro_qdef`: present, shape `[N,128,6,48]`.")
    lines.append("5. `macro16_point_xi`: present and equal to `macro16_source128_point_table()`.")
    lines.append("6. `case_id`: present for case-isolated split.")
    lines.append("")
    lines.append("Result:")
    lines.append("")
    lines.append(f"- Required fields: {md_bool(summary['all_required_fields_present'])}")
    lines.append(f"- 128 point rule: {md_bool(summary['all_source128_rule_match'])}")
    lines.append("")
    lines.append("## 7. Train Val Split Readiness")
    lines.append("")
    lines.append(f"- Case split possible: {md_bool(split['case_split_possible'])}")
    lines.append(f"- Geometry/category holdout possible: {md_bool(split['geometry_or_category_holdout_possible'])}")
    lines.append(f"- Recommended split: `{split['recommended_split']}`")
    lines.append(f"- Recommended validation cases: `{split['recommended_val_cases']}`")
    lines.append("- Frame-random split: not recommended.")
    lines.append("")
    lines.append("说明：loader 支持 `val_cases` 做 case 隔离。若要严格几何泛化，应按类别或几何哈希选择验证集，而不是只做随机 frame split。")
    lines.append("")
    lines.append("## 8. Model Feed Readiness")
    lines.append("")
    lines.append("The data can be loaded by the current Macro16 training loader with:")
    lines.append("")
    lines.append("```text")
    lines.append("model = Macro16BoundaryDeepONetWithLE0")
    lines.append("q input = q48_def_hat")
    lines.append("geometry input = X16_hat")
    lines.append("label = LE_macro")
    lines.append("B target = B_macro_qdef")
    lines.append("integration rule = standard source128")
    lines.append("```")
    lines.append("")
    loader_ready = all(bool(cat.get("loader", {}).get("passed")) for cat in payload["categories"])
    lines.append(f"Result: {md_bool(loader_ready)}")
    lines.append("")
    lines.append("## 9. Decision")
    lines.append("")
    if summary["passed"] and loader_ready:
        lines.append("Training data readiness: PASS for limited LE/B training.")
        lines.append("")
        lines.append("可以直接喂给 `Macro16BoundaryDeepONetWithLE0` 做 limited LE/B training。")
    else:
        lines.append("Training data readiness: FAIL.")
        lines.append("")
        lines.append("缺失项见 JSON 明细，不能直接进入训练。")
    lines.append("")
    lines.append("边界：这不是 full tangent 通过，也不是 solver-ready 结论。训练后仍必须做 selected-frame force closure。Material-only K 只作为 diagnostic。")
    lines.append("")
    lines.append("## 10. Unresolved Risks")
    lines.append("")
    lines.append("1. `B_macro_qdef` 仍使用当前小转角线性刚体投影近似。")
    lines.append("2. Full tangent 仍待处理，不由本报告释放。")
    lines.append("3. Strong non-inverted repaired data可作为鲁棒性覆盖，但训练 split 应避免把同一几何的近似重复 frame 同时放入 train 和 val 来冒充泛化。")
    lines.append("")
    lines.append("## 11. Recommended Next Action")
    lines.append("")
    lines.append("1. 若开始 limited training，使用 128 点 source128 compact。")
    lines.append("2. 使用显式 `val_cases` 做 case 或类别隔离。")
    lines.append("3. 训练后必须重新跑 selected-frame force closure。")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    category_lists = dict(DEFAULT_CATEGORY_LISTS)
    categories = [audit_category(name, path) for name, path in category_lists.items()]
    payload = {
        "task": "Gate 06 training data readiness",
        "no_training": True,
        "model": "Macro16BoundaryDeepONetWithLE0",
        "categories": categories,
    }
    payload["summary"] = aggregate(categories)
    payload["split_readiness"] = split_readiness(categories)
    out_json = Path(args.out_json).resolve()
    out_report = Path(args.report).resolve()
    write_json(out_json, payload)
    write_report(out_report, payload, out_json)
    if bool(args.strict) and not bool(payload["summary"]["passed"]):
        raise SystemExit("training data readiness failed")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", default=str(ROOT / "runs" / "gate06_training_data_readiness" / "training_data_readiness.json"))
    parser.add_argument("--report", default=str(ROOT / "reports" / "06_training_data_readiness.md"))
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
