"""Evaluate a v1.2 query-point checkpoint case-by-case without training.

This script is intentionally narrow: it reconstructs the generic query-point
trainer preprocessing from a saved checkpoint, evaluates selected case ids with
the existing trainer metrics, and writes a lightweight JSON/CSV attribution
report.  It does not update model weights.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from macro_deeponet.point_features import load_point_features_from_compacts, transform_point_features_for_scale
from macro_deeponet.train_true176_deeponet_sobolev import evaluate
from macro_deeponet.train_true176_generic_sobolev import (
    _le_scale_np,
    _query_b_baseline_meta,
    build_model,
    validate_point_feature_source_for_scale,
)
from macro_deeponet.true176_data import (
    build_branch_features,
    canonical_scale_mode,
    load_compacts,
    parse_int_list,
    transform_b_target_for_q_coordinate,
)


def _read_compact_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]


def _as_numpy_norms(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    return {str(k): np.asarray(v) for k, v in raw.items()}


def _metric_subset(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    keys = [
        "frames",
        "LE_rel",
        "LE_cos",
        "LE_ip_rel_max",
        "AD_B_rel",
        "AD_B_cos",
        "AD_B_ip_rel_max",
        "AD_B_col_rel_median",
        "AD_B_col_rel_p95",
        "AD_B_col_rel_max",
        "phys_rand_dir_B_rel",
        "phys_rand_dir_B_cos",
        "rand_dir_B_rel",
        "AD_B_abs_rel_train_rms",
        "AD_B_rel_eps_rms",
    ]
    return {key: metrics.get(f"{prefix}_{key}") for key in keys}


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).resolve()
    compact_paths = _read_compact_list(Path(args.compact_list).resolve())
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    train_args = dict(ckpt.get("args", {}))
    ns = SimpleNamespace(**train_args)
    norms = _as_numpy_norms(ckpt["norms"])

    target_ips = [int(v) for v in ckpt.get("target_ips", parse_int_list(str(getattr(ns, "target_ips", "")), default=list(range(128))))]
    scale_mode = canonical_scale_mode(str(getattr(ns, "scale_mode", "normalized")))
    data = load_compacts(
        compact_paths,
        frame_stride=int(getattr(ns, "frame_stride", 1)),
        max_frames_per_compact=int(getattr(ns, "max_frames_per_compact", 0)),
        target_ips=target_ips,
    )

    x_raw, branch_meta = build_branch_features(
        shape4=data.shape4,
        q48_raw=data.q48_raw,
        mode=str(getattr(ns, "branch_feature_mode", "xkeep-qraw")),
        keep_node_coords=data.keep_node_coords,
        macro_nodes=data.macro_nodes,
        length_scale=data.length_scale,
        scale_mode=scale_mode,
    )
    le_raw = data.le.astype(np.float32, copy=False)
    b_raw = data.b.astype(np.float32, copy=False)
    b_train, b_scale_meta = transform_b_target_for_q_coordinate(
        b_raw,
        data.length_scale,
        scale_mode=scale_mode,
        b_label_coordinate=str(getattr(ns, "b_label_coordinate", "auto")),
    )

    point_raw, point_meta = load_point_features_from_compacts(
        compact_paths=data.compact_paths,
        source_index=data.source_index,
        source_row=data.source_row,
        shape4=data.shape4,
        target_ips=target_ips,
        source=str(getattr(ns, "point_feature_source", "data")),
        include_id_features=bool(getattr(ns, "include_id_features", False)),
        allow_shape4_fallback=bool(getattr(ns, "allow_shape4_point_feature_fallback", False)),
    )
    validate_point_feature_source_for_scale(
        scale_mode=scale_mode,
        requested_source=str(getattr(ns, "point_feature_source", "data")),
        point_meta=point_meta,
        allow_physical_shape4_trunk=bool(getattr(ns, "allow_physical_shape4_trunk", False)),
    )
    point_feature_names = list(point_meta.get("feature_names", [f"point_feature_{i}" for i in range(point_raw.shape[-1])]))
    point_raw, point_scale_meta = transform_point_features_for_scale(
        point_raw,
        point_feature_names,
        data.length_scale,
        scale_mode=scale_mode,
        detj_scale_dim=int(getattr(ns, "detj_scale_dim", 3)),
    )
    point_meta = {**point_meta, "scale_transform": point_scale_meta}

    x_norm = ((x_raw - norms["x_mean"]) / norms["x_std"]).astype(np.float32)
    point_norm = ((point_raw - norms["point_mean"].reshape(1, 1, -1)) / norms["point_std"].reshape(1, 1, -1)).astype(np.float32)
    q_start = int(np.asarray(norms.get("q_start", 4)).reshape(-1)[0])
    q_dim = int(np.asarray(norms.get("q_dim", 48)).reshape(-1)[0])
    q_std = norms["x_std"].reshape(-1)[q_start : q_start + q_dim].astype(np.float32)
    le_std = norms["le_std"].astype(np.float32)
    j_norm_target = (b_train * q_std.reshape(1, 1, 1, 48) / _le_scale_np(le_std, len(target_ips))).astype(np.float32)

    device = torch.device("cuda" if bool(args.cuda) and torch.cuda.is_available() else "cpu")
    model = build_model(
        ns,
        input_dim=int(x_norm.shape[-1]),
        point_dim=int(point_norm.shape[-1]),
        ip_count=len(target_ips),
        q_start=q_start,
        q_dim=q_dim,
        skip_init=np.zeros((len(target_ips), 6, 48), dtype=np.float32),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    requested_cases = parse_int_list(str(args.cases), default=[])
    if not requested_cases:
        requested_cases = sorted(np.unique(data.case_id[data.case_id >= 0]).astype(np.int64).tolist())
    eval_columns = parse_int_list(str(getattr(ns, "eval_columns", "")), default=list(range(12)))
    batch_size = max(1, int(getattr(ns, "eval_batch_size", 1)))
    b_global_rms = float(np.sqrt(np.mean(np.asarray(b_train, dtype=np.float64) ** 2)))

    case_reports: list[dict[str, Any]] = []
    for case_id in requested_cases:
        indices = np.flatnonzero(np.asarray(data.case_id, dtype=np.int64) == int(case_id)).astype(np.int64)
        if indices.size == 0:
            raise ValueError(f"case {case_id} selected no frames")
        prefix = f"case{int(case_id):03d}"
        metrics = evaluate(
            model,
            x_norm,
            point_norm,
            le_raw,
            b_train,
            j_norm_target,
            indices,
            norms,
            device,
            batch_size,
            eval_columns,
            str(getattr(ns, "jacobian_method", "forward")),
            prefix,
            seed=int(getattr(ns, "seed", 20260620)) + int(case_id),
            b_global_rms=b_global_rms,
            rel_eps_scale=float(getattr(ns, "physical_j_rel_eps_scale", 0.02)),
        )
        subset_label = f"case{int(case_id):03d}"
        b_prior = _query_b_baseline_meta(
            model,
            point_norm,
            j_norm_target,
            indices,
            b_target=b_train,
            le_std=le_std,
            q_std=q_std,
            eval_columns=eval_columns,
            device=device,
            batch_size=batch_size,
            prefix="b_prior_current",
            subset_label=subset_label,
        )
        b_prefix = f"b_prior_current_{subset_label}"
        row = {
            "case_id": int(case_id),
            "frame_count": int(indices.size),
            "checkpoint": str(checkpoint_path),
            "compact_sources": sorted(set(str(data.compact_paths[int(i)]) for i in np.unique(data.source_index[indices]))),
            "eval_columns": eval_columns,
        }
        row.update(_metric_subset(metrics, prefix))
        row.update(
            {
                "B_prior_rel": b_prior.get(f"{b_prefix}_evalcols_B_rel"),
                "B_prior_cos": b_prior.get(f"{b_prefix}_evalcols_B_cos"),
                "B_prior_norm_rel": b_prior.get(f"{b_prefix}_evalcols_norm_rel"),
                "B_prior_current_B_rel_all_columns": b_prior.get(f"{b_prefix}_B_rel"),
            }
        )
        case_reports.append(row)

    report = {
        "audit_name": "v1_2_checkpoint_case_attribution",
        "checkpoint": str(checkpoint_path),
        "compact_list": str(Path(args.compact_list).resolve()),
        "cases": requested_cases,
        "training_validation_split": ckpt.get("validation_split"),
        "strain_meta": ckpt.get("strain_meta"),
        "b_scale_meta": b_scale_meta,
        "case_reports": case_reports,
        "no_training_performed": True,
    }
    (out_root / "case_attribution.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in case_reports for key in row.keys() if key != "compact_sources"})
    with (out_root / "case_attribution.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + ["compact_sources"])
        writer.writeheader()
        for row in case_reports:
            out = {key: row.get(key) for key in fieldnames}
            out["compact_sources"] = ";".join(row.get("compact_sources", []))
            writer.writerow(out)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--compact-list", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--cases", default="")
    p.add_argument("--cuda", action="store_true")
    return p.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"out_cases": [r["case_id"] for r in report["case_reports"]], "no_training_performed": True}, sort_keys=True))


if __name__ == "__main__":
    main()
