#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate 32 diagnosis for Macro16 wind-shell generalization failure.

This script is read-only. It does not train a network, change model structure,
change q48/LE ordering, or change the 128-point rule.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
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

from audit_macro16_force_stiffness import json_default, read_path_list, write_json  # noqa: E402
from audit_macro16_trained_force_closure import parse_path_map, qdef_projected_rf  # noqa: E402
from macro_deeponet.macro16_geometry import macro16_standard_point_table  # noqa: E402
from macro_deeponet.train_macro16_boundary_sobolev import load_macro16_compacts, parse_int_list  # noqa: E402
from macro_deeponet.true176_data import stats  # noqa: E402


def norm_rows(arr: np.ndarray) -> np.ndarray:
    vals = np.asarray(arr, dtype=np.float64)
    if vals.ndim == 1:
        return np.abs(vals)
    return np.linalg.norm(vals.reshape(vals.shape[0], -1), axis=1)


def finite_stats(vals: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(vals, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    out: dict[str, Any] = {
        "count": int(arr.size),
        "finite_count": int(finite.size),
    }
    if finite.size == 0:
        out.update({"min": None, "p05": None, "p50": None, "p95": None, "max": None, "mean": None})
        return out
    out.update(
        {
            "min": float(np.min(finite)),
            "p01": float(np.percentile(finite, 1)),
            "p05": float(np.percentile(finite, 5)),
            "p50": float(np.percentile(finite, 50)),
            "p95": float(np.percentile(finite, 95)),
            "p99": float(np.percentile(finite, 99)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
        }
    )
    return out


def rel_norm(pred: np.ndarray, ref: np.ndarray, *, floor: float = 1.0e-300) -> float:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    r = np.asarray(ref, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(p - r) / max(float(np.linalg.norm(r)), float(floor)))


def case_ids_from_text(text: str) -> set[int]:
    if not str(text).strip():
        return set()
    return set(int(v) for v in parse_int_list(str(text)))


def select_indices(case_id: np.ndarray, cases: set[int]) -> np.ndarray:
    idx = np.arange(case_id.shape[0], dtype=np.int64)
    if cases:
        idx = idx[np.isin(case_id, np.asarray(sorted(cases), dtype=np.int64))]
    return idx


def load_checkpoint_norms(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import pathlib
    import torch

    class _CrossPlatformPosixPath(pathlib.PurePosixPath):
        pass

    class _CrossPlatformWindowsPath(pathlib.PureWindowsPath):
        pass

    posix_original = pathlib.PosixPath
    windows_original = pathlib.WindowsPath
    try:
        if os.name == "nt":
            pathlib.PosixPath = _CrossPlatformPosixPath  # type: ignore[assignment]
        else:
            pathlib.WindowsPath = _CrossPlatformWindowsPath  # type: ignore[assignment]
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = posix_original  # type: ignore[assignment]
        pathlib.WindowsPath = windows_original  # type: ignore[assignment]
    norms = checkpoint.get("norms", {})
    if "branch_mean" not in norms or "branch_std" not in norms:
        raise KeyError(f"{path}: checkpoint missing norms.branch_mean or norms.branch_std")
    return np.asarray(norms["branch_mean"], dtype=np.float64), np.asarray(norms["branch_std"], dtype=np.float64)


def branch_normalization(
    branch_raw: np.ndarray,
    train_idx: np.ndarray,
    checkpoint: Path | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if checkpoint is not None:
        mean, std = load_checkpoint_norms(checkpoint)
        source = str(checkpoint)
    else:
        mean, std = stats(branch_raw[train_idx], axis=0)
        source = "computed_from_train_cases"
    std = np.maximum(np.asarray(std, dtype=np.float64), 1.0e-12)
    normed = (np.asarray(branch_raw, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) / std
    return normed, {"source": source, "branch_dim": int(branch_raw.shape[1])}


def teacher_force(
    data: Any,
    idx: np.ndarray,
    source_path_map: list[tuple[str, str]],
) -> dict[str, Any]:
    rf_qdef, weights, elastic_d = qdef_projected_rf(
        data.compact_paths,
        data.source_index[idx],
        data.source_row[idx],
        source_path_map=source_path_map,
    )
    l_ref = np.asarray(data.length_scale[idx], dtype=np.float64).reshape(-1, 1, 1, 1)
    b_raw = np.asarray(data.b[idx], dtype=np.float64) / np.maximum(l_ref, 1.0e-12)
    le = np.asarray(data.le[idx], dtype=np.float64)
    stress = np.einsum("nab,npb->npa", np.asarray(elastic_d, dtype=np.float64), le)
    force = np.einsum("npaj,npa,np->nj", b_raw, stress, np.asarray(weights, dtype=np.float64))
    err = force - rf_qdef
    force_norm = norm_rows(force)
    rf_norm = norm_rows(rf_qdef)
    err_norm = norm_rows(err)
    rel_frame = err_norm / np.maximum(rf_norm, 1.0e-300)
    return {
        "force": force,
        "rf": rf_qdef,
        "force_norm": force_norm,
        "rf_norm": rf_norm,
        "err_norm": err_norm,
        "rel_by_frame": rel_frame,
        "rel": rel_norm(force, rf_qdef),
        "weights_sum_by_frame": np.sum(np.asarray(weights, dtype=np.float64), axis=1),
    }


def group_stats(
    name: str,
    data: Any,
    idx: np.ndarray,
    branch_norm: np.ndarray,
    force_info: dict[str, Any] | None,
) -> dict[str, Any]:
    q_norm = norm_rows(data.q48_hat[idx])
    le_norm = norm_rows(data.le[idx])
    b_norm = norm_rows(data.b[idx])
    branch_all = norm_rows(branch_norm[idx])
    branch_q = norm_rows(branch_norm[idx, :48])
    branch_x = norm_rows(branch_norm[idx, 48:96])
    out: dict[str, Any] = {
        "name": name,
        "frame_count": int(idx.size),
        "case_count": int(np.unique(data.case_id[idx]).size) if idx.size else 0,
        "cases": sorted(np.unique(data.case_id[idx]).astype(int).tolist()) if idx.size else [],
        "q48_def_hat_norm": finite_stats(q_norm),
        "LE_macro_norm": finite_stats(le_norm),
        "B_macro_qdef_norm": finite_stats(b_norm),
        "branch_norm_all": finite_stats(branch_all),
        "branch_norm_q": finite_stats(branch_q),
        "branch_norm_x16": finite_stats(branch_x),
        "L_ref": finite_stats(np.asarray(data.length_scale[idx], dtype=np.float64).reshape(-1)),
    }
    if force_info is not None:
        out.update(
            {
                "teacher_assembled_force_norm": finite_stats(force_info["force_norm"]),
                "abaqus_rf_qdef_norm": finite_stats(force_info["rf_norm"]),
                "teacher_force_rel": float(force_info["rel"]),
                "teacher_force_rel_by_frame": finite_stats(force_info["rel_by_frame"]),
                "selected_frame_volume_sum": finite_stats(force_info["weights_sum_by_frame"]),
            }
        )
    return out


def per_case_stats(
    data: Any,
    branch_norm: np.ndarray,
    force_all: dict[str, Any] | None,
    global_indices: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in sorted(np.unique(data.case_id[global_indices]).astype(int).tolist()):
        local = global_indices[data.case_id[global_indices] == case]
        force_case: dict[str, Any] | None = None
        if force_all is not None:
            positions = np.where(np.isin(global_indices, local))[0]
            force_case = {
                "force_norm": force_all["force_norm"][positions],
                "rf_norm": force_all["rf_norm"][positions],
                "rel_by_frame": force_all["rel_by_frame"][positions],
                "weights_sum_by_frame": force_all["weights_sum_by_frame"][positions],
                "rel": rel_norm(force_all["force"][positions], force_all["rf"][positions]),
            }
        item = group_stats(f"case{case:03d}", data, local, branch_norm, force_case)
        item["case_id"] = int(case)
        rows.append(item)
    return rows


def extract_case_from_payload(path: Path, payload: dict[str, Any]) -> int | None:
    for key in ("case_ids", "cases"):
        vals = payload.get(key)
        if isinstance(vals, list) and vals:
            try:
                return int(vals[0])
            except Exception:
                pass
    match = re.search(r"case[_-]?(\d+)", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def nested_rel(payload: dict[str, Any], keys: list[tuple[str, str]]) -> float | None:
    for outer, inner in keys:
        val = payload.get(outer)
        if isinstance(val, dict) and inner in val:
            try:
                return float(val[inner])
            except Exception:
                continue
    return None


def collect_audit_payloads(audit_globs: list[str]) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for pattern in audit_globs:
        paths.extend(Path(p) for p in sorted(glob.glob(pattern)))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        case_id = extract_case_from_payload(path, payload)
        rows.append(
            {
                "path": str(path),
                "case_id": case_id,
                "model_force_rel": nested_rel(
                    payload,
                    [
                        ("selected_frame_force_qdef", "rel"),
                        ("force", "rel"),
                        ("model_force", "rel"),
                    ],
                ),
                "teacher_force_rel": nested_rel(
                    payload,
                    [
                        ("teacher_selected_frame_force_qdef", "rel"),
                        ("teacher_force", "rel"),
                        ("teacher_selected_frame_force", "rel"),
                    ],
                ),
            }
        )
    return rows


def ratio_to_ref(value: Any, ref: Any) -> float | None:
    if value is None or ref is None:
        return None
    v = float(value)
    r = float(ref)
    if not math.isfinite(v) or not math.isfinite(r) or abs(r) < 1.0e-300:
        return None
    return float(v / r)


def diagnose(summary: dict[str, Any]) -> dict[str, Any]:
    train = summary["groups"]["train"]
    wind = summary["groups"]["wind_shell"]
    train_branch_p99 = train["branch_norm_all"].get("p99")
    wind_branch_max = wind["branch_norm_all"].get("max")
    train_force_p50 = train.get("abaqus_rf_qdef_norm", {}).get("p50")
    wind_force_p50 = wind.get("abaqus_rf_qdef_norm", {}).get("p50")
    wind_teacher_rel = wind.get("teacher_force_rel")
    wind_branch_ratio = ratio_to_ref(wind_branch_max, train_branch_p99)
    wind_force_ratio = ratio_to_ref(wind_force_p50, train_force_p50)
    audits = summary.get("gate31_audits", [])
    wind_audits = [row for row in audits if row.get("case_id") in {70, 71, 72, 73}]
    case60_audits = [row for row in audits if row.get("case_id") == 60]
    return {
        "data_bad_evidence": bool(wind_teacher_rel is not None and wind_teacher_rel > 2.0e-2),
        "wind_teacher_force_passes_0p02": bool(wind_teacher_rel is not None and wind_teacher_rel <= 2.0e-2),
        "wind_branch_max_over_train_p99": wind_branch_ratio,
        "wind_rf_p50_over_train_rf_p50": wind_force_ratio,
        "wind_branch_outlier_flag": bool(wind_branch_ratio is not None and wind_branch_ratio > 1.5),
        "wind_force_small_flag": bool(wind_force_ratio is not None and wind_force_ratio < 1.0e-2),
        "wind_model_force_failed_in_gate31": bool(
            wind_audits and max(float(r.get("model_force_rel") or 0.0) for r in wind_audits) > 1.0e-1
        ),
        "case060_invalid_relative_gate_flag": bool(
            case60_audits
            and max(float(r.get("teacher_force_rel") or 0.0) for r in case60_audits) > 2.0e-2
        ),
        "recommended_next_step": "do_not_expand_training_before_explaining_wind_shell_distribution_shift",
    }


def fmt_num(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def group_table(groups: dict[str, Any]) -> str:
    lines = [
        "| group | cases | frames | q p50 | LE p50 | B p50 | branch max | RF p50 | teacher force rel |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, group in groups.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(name),
                    ",".join(str(v) for v in group.get("cases", [])),
                    str(group.get("frame_count", 0)),
                    fmt_num(group.get("q48_def_hat_norm", {}).get("p50")),
                    fmt_num(group.get("LE_macro_norm", {}).get("p50")),
                    fmt_num(group.get("B_macro_qdef_norm", {}).get("p50")),
                    fmt_num(group.get("branch_norm_all", {}).get("max")),
                    fmt_num(group.get("abaqus_rf_qdef_norm", {}).get("p50")),
                    fmt_num(group.get("teacher_force_rel")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def case_table(cases: list[dict[str, Any]]) -> str:
    lines = [
        "| case | frames | q p50 | LE p50 | B p50 | branch max | RF p50 | teacher force rel |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cases:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{int(row['case_id']):03d}",
                    str(row.get("frame_count", 0)),
                    fmt_num(row.get("q48_def_hat_norm", {}).get("p50")),
                    fmt_num(row.get("LE_macro_norm", {}).get("p50")),
                    fmt_num(row.get("B_macro_qdef_norm", {}).get("p50")),
                    fmt_num(row.get("branch_norm_all", {}).get("max")),
                    fmt_num(row.get("abaqus_rf_qdef_norm", {}).get("p50")),
                    fmt_num(row.get("teacher_force_rel")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def audit_table(audits: list[dict[str, Any]]) -> str:
    if not audits:
        return "No Gate 31 audit JSON files were provided."
    lines = [
        "| case | model force rel | teacher force rel | json |",
        "|---|---:|---:|---|",
    ]
    for row in sorted(audits, key=lambda r: (-1 if r.get("case_id") is None else int(r["case_id"]), str(r.get("path", "")))):
        lines.append(
            "| "
            + " | ".join(
                [
                    "NA" if row.get("case_id") is None else f"{int(row['case_id']):03d}",
                    fmt_num(row.get("model_force_rel")),
                    fmt_num(row.get("teacher_force_rel")),
                    str(row.get("path", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    diag = summary["diagnosis"]
    text = f"""# Gate 32 Wind Shell Generalization Diagnosis

日期：2026-06-26

## 1. 目标

诊断 Gate 31 中 `case070` 到 `case073` 风机壳验证集 force closure 失败原因。

不训练网络。

不改模型。

不改 `q48` 顺序。

不改 `LE` 顺序。

不改 128 点规则。

## 2. 数据

compact list：

```text
{summary["compact_list"]}
```

train cases：

```text
{",".join(str(v) for v in summary["train_cases"])}
```

wind shell cases：

```text
{",".join(str(v) for v in summary["wind_cases"])}
```

branch normalization：

```text
{summary["branch_normalization"]["source"]}
```

## 3. 分组统计

{group_table(summary["groups"])}

## 4. 逐 Case 统计

{case_table(summary["per_case"])}

## 5. Gate 31 Force Audit 对照

{audit_table(summary.get("gate31_audits", []))}

## 6. 诊断标记

```text
data_bad_evidence = {diag["data_bad_evidence"]}
wind_teacher_force_passes_0p02 = {diag["wind_teacher_force_passes_0p02"]}
wind_branch_max_over_train_p99 = {fmt_num(diag["wind_branch_max_over_train_p99"])}
wind_rf_p50_over_train_rf_p50 = {fmt_num(diag["wind_rf_p50_over_train_rf_p50"])}
wind_branch_outlier_flag = {diag["wind_branch_outlier_flag"]}
wind_force_small_flag = {diag["wind_force_small_flag"]}
wind_model_force_failed_in_gate31 = {diag["wind_model_force_failed_in_gate31"]}
case060_invalid_relative_gate_flag = {diag["case060_invalid_relative_gate_flag"]}
```

## 7. 当前结论

如果 `wind_teacher_force_passes_0p02 = True` 且 `wind_model_force_failed_in_gate31 = True`，则当前证据指向模型或分布泛化失败，不是老师力闭合坏。

如果 `wind_branch_outlier_flag = True`，下一步应优先处理风机壳 branch 分布离群或 split 策略。

如果 `wind_force_small_flag = True`，下一步应检查 relative force 门槛是否被小力范数放大。

如果 `case060_invalid_relative_gate_flag = True`，`case060` 不应作为普通 relative force gate 样本。

## 8. 下一步

不要扩大训练。

先根据本报告决定：

1. 是否调整 wind shell split。
2. 是否调整 branch normalization。
3. 是否对近零力样本使用 absolute floor 或单独 gate。
4. 是否需要专门的 wind shell 小样本训练对照。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    compact_paths = [str(p) for p in read_path_list(Path(args.compact_list))]
    point_table = macro16_standard_point_table(
        plane_order=int(args.plane_gauss_order),
        thickness_order=int(args.thickness_gauss_order),
    )
    data = load_macro16_compacts(
        compact_paths,
        point_table=point_table,
        frame_stride=1,
        max_frames_per_compact=0,
        scale_mode="normalized",
        b_label_coordinate="auto",
    )
    train_cases = case_ids_from_text(args.train_cases)
    wind_cases = case_ids_from_text(args.wind_cases)
    special_cases = case_ids_from_text(args.special_cases)
    train_idx = select_indices(data.case_id, train_cases)
    wind_idx = select_indices(data.case_id, wind_cases)
    if train_idx.size == 0:
        raise ValueError("no train frames selected")
    if wind_idx.size == 0:
        raise ValueError("no wind-shell frames selected")

    branch_raw = np.concatenate(
        [data.q48_hat, data.x16_hat.reshape(data.x16_hat.shape[0], -1), data.length_scale],
        axis=1,
    )
    checkpoint = Path(args.checkpoint).resolve() if str(args.checkpoint).strip() else None
    branch_norm, branch_meta = branch_normalization(branch_raw, train_idx, checkpoint)
    path_map = parse_path_map(list(args.source_path_map))

    selected = np.unique(np.concatenate([train_idx, wind_idx, select_indices(data.case_id, special_cases)]))
    force_selected = teacher_force(data, selected, path_map)
    selected_pos = {int(global_idx): int(pos) for pos, global_idx in enumerate(selected.tolist())}

    def subset_force(idx: np.ndarray) -> dict[str, Any]:
        pos = np.asarray([selected_pos[int(v)] for v in idx], dtype=np.int64)
        return {
            "force": force_selected["force"][pos],
            "rf": force_selected["rf"][pos],
            "force_norm": force_selected["force_norm"][pos],
            "rf_norm": force_selected["rf_norm"][pos],
            "rel_by_frame": force_selected["rel_by_frame"][pos],
            "weights_sum_by_frame": force_selected["weights_sum_by_frame"][pos],
            "rel": rel_norm(force_selected["force"][pos], force_selected["rf"][pos]),
        }

    audit_globs = list(args.audit_json_glob)
    if str(args.audit_dir).strip():
        audit_globs.append(str(Path(args.audit_dir) / "*.json"))
    summary: dict[str, Any] = {
        "script": "diagnose_macro16_wind_shell_generalization.py",
        "compact_list": str(Path(args.compact_list).resolve()),
        "compact_count": int(len(compact_paths)),
        "frame_count": int(data.q48_hat.shape[0]),
        "point_count": int(data.le.shape[1]),
        "train_cases": sorted(train_cases),
        "wind_cases": sorted(wind_cases),
        "special_cases": sorted(special_cases),
        "branch_normalization": branch_meta,
        "groups": {
            "train": group_stats("train", data, train_idx, branch_norm, subset_force(train_idx)),
            "wind_shell": group_stats("wind_shell", data, wind_idx, branch_norm, subset_force(wind_idx)),
        },
        "per_case": per_case_stats(data, branch_norm, force_selected, selected),
        "gate31_audits": collect_audit_payloads(audit_globs),
    }
    summary["diagnosis"] = diagnose(summary)
    if str(args.out).strip():
        write_json(Path(args.out).resolve(), summary)
    if str(args.report_out).strip():
        write_markdown(Path(args.report_out).resolve(), summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-list", required=True, type=Path)
    parser.add_argument("--out", default="")
    parser.add_argument("--report-out", default="")
    parser.add_argument("--train-cases", default="19,25,31,41,43,44,45,46,49,50,60,61")
    parser.add_argument("--wind-cases", default="70,71,72,73")
    parser.add_argument("--special-cases", default="60")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--audit-dir", default="")
    parser.add_argument("--audit-json-glob", action="append", default=[])
    parser.add_argument("--source-path-map", action="append", default=[])
    parser.add_argument("--plane-gauss-order", type=int, default=3)
    parser.add_argument("--thickness-gauss-order", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
