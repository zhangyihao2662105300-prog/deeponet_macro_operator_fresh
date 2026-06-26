#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect wind-shell complete compacts for training.

This is a lightweight manifest builder. It does not modify compacts and does
not train a network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def compact_summary(path: Path) -> dict[str, Any]:
    with np.load(str(path), allow_pickle=True) as z:
        n = int(np.asarray(z["q48_raw"]).reshape(-1, 48).shape[0]) if "q48_raw" in z.files else 0
        return {
            "path": str(path),
            "frames": n,
            "has_LE128_base": "LE128_base" in z.files,
            "has_B_LE128_forward": "B_LE128_forward" in z.files,
            "has_X_keep": "X_keep" in z.files,
            "has_point_features": "point_features" in z.files,
            "point_feature_dim": int(np.asarray(z["point_features"]).shape[-1]) if "point_features" in z.files else 0,
            "length_scale_source": str(np.asarray(z["length_scale_source"]).reshape(-1)[0]) if "length_scale_source" in z.files else "",
            "q_generation_method": str(np.asarray(z["q_generation_method"]).reshape(-1)[0]) if "q_generation_method" in z.files else "",
            "geometry_family": str(np.asarray(z["geometry_family"]).reshape(-1)[0]) if "geometry_family" in z.files else "",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--out-list", required=True, type=Path)
    parser.add_argument("--out-summary", required=True, type=Path)
    parser.add_argument("--min-compacts", type=int, default=1)
    args = parser.parse_args()

    paths: list[Path] = []
    for root in args.root:
        paths.extend(sorted(Path(root).resolve().rglob("*_training_ready.npz")))
    unique = sorted({str(path.resolve()): path.resolve() for path in paths}.values())
    rows = [compact_summary(path) for path in unique]
    bad = [
        row
        for row in rows
        if not (row["has_LE128_base"] and row["has_B_LE128_forward"] and row["has_X_keep"] and row["has_point_features"])
    ]
    if len(rows) < int(args.min_compacts):
        raise RuntimeError(f"only found {len(rows)} compacts, expected at least {int(args.min_compacts)}")
    if bad:
        raise RuntimeError(f"{len(bad)} compacts are missing required training fields")

    args.out_list.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_list.write_text("\n".join(str(path) for path in unique) + "\n", encoding="utf-8")
    summary = {
        "compact_count": len(rows),
        "frame_count": int(sum(int(row["frames"]) for row in rows)),
        "families": sorted({str(row["geometry_family"]) for row in rows}),
        "point_feature_dims": sorted({int(row["point_feature_dim"]) for row in rows}),
        "length_scale_sources": sorted({str(row["length_scale_source"]) for row in rows}),
        "q_generation_methods": sorted({str(row["q_generation_method"]) for row in rows}),
        "compacts": rows,
    }
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "compacts"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
