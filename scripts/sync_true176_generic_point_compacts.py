#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synchronize TRUE176 compact files to the generic point-feature schema.

This script copies compact npz files and adds a canonical ``point_features``
array when explicit real integration-point fields are present.

Default behavior is safe: it refuses to generate point features from shape4,
because that would assume the standard row-map points match the LE/B labels.
Use ``--shape4-audited`` only after ``audit_true176_shape4_ip_contract.py`` or
an equivalent coordinate/order audit proves that this assumption is valid for
the dataset. ``--legacy-shape4-fallback`` remains as a compatibility spelling.

Canonical output fields added to each compact:

    point_features:      [N, 128, F]
    point_feature_names: string array [F]
    point_feature_source: scalar string

Accepted input fields are implemented in ``macro_deeponet.point_features``:
prebuilt ``point_features``/``ip_features``/``trunk_features`` or raw fields such
as ``ip_xi``, ``ip_xyz``, ``ip_J``, ``ip_invJ``, ``ip_detJ``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from macro_deeponet.point_features import load_point_features_from_compacts
from macro_deeponet.true176_data import parse_target_ips


def read_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def write_list(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def copy_with_point_features(src: Path, dst: Path, *, point_features: np.ndarray, feature_names: list[str], source_text: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with np.load(str(src), allow_pickle=True) as z:
        payload = {key: z[key] for key in z.files if key not in {"point_features", "point_feature_names", "point_feature_source"}}
    payload["point_features"] = np.asarray(point_features, dtype=np.float32)
    payload["point_feature_names"] = np.asarray(feature_names, dtype=object)
    payload["point_feature_source"] = np.asarray(source_text, dtype=object)
    np.savez_compressed(str(dst), **payload)


def load_shape4_only(src: Path) -> np.ndarray:
    with np.load(str(src), allow_pickle=True) as z:
        if "shape4" not in z.files:
            raise KeyError(f"{src}: missing shape4")
        shape4 = np.asarray(z["shape4"], dtype=np.float32).reshape(-1, 4)
    return shape4


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compact-list", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--target-ips", default=",".join(str(i) for i in range(128)))
    p.add_argument("--include-id-features", action="store_true")
    p.add_argument("--shape4-audited", action="store_true")
    p.add_argument("--legacy-shape4-fallback", action="store_true")
    p.add_argument("--out-list-name", default="compact_paths_generic_points.txt")
    args = p.parse_args()

    src_paths = [Path(v).resolve() for v in read_list(Path(args.compact_list))]
    out_root = Path(args.out_root).resolve()
    target_ips = parse_target_ips(str(args.target_ips))
    if target_ips != list(range(128)):
        raise ValueError("sync script currently writes full [N,128,F] point_features; use --target-ips 0..127")

    out_paths: list[str] = []
    reports = []
    for i, src in enumerate(src_paths):
        shape4 = load_shape4_only(src)
        n = int(shape4.shape[0])
        source_index = np.zeros(n, dtype=np.int64)
        source_row = np.arange(n, dtype=np.int64)
        point, meta = load_point_features_from_compacts(
            compact_paths=[str(src)],
            source_index=source_index,
            source_row=source_row,
            shape4=shape4,
            target_ips=target_ips,
            source="shape4-audited" if args.shape4_audited else ("auto" if args.legacy_shape4_fallback else "data"),
            include_id_features=bool(args.include_id_features),
            allow_shape4_fallback=bool(args.legacy_shape4_fallback),
        )
        rel = src.name if src.suffix == ".npz" else f"compact_{i:03d}.npz"
        parent_name = src.parent.name
        dst = out_root / parent_name / rel
        copy_with_point_features(
            src,
            dst,
            point_features=point,
            feature_names=list(meta.get("feature_names", [f"point_feature_{k}" for k in range(point.shape[-1])])),
            source_text=str(meta.get("point_feature_source", "data_generic")),
        )
        out_paths.append(str(dst))
        reports.append({"src": str(src), "dst": str(dst), "frames": n, "point_shape": list(point.shape), "meta": meta})
        print(json.dumps(reports[-1], ensure_ascii=False, sort_keys=True))

    out_list = out_root / args.out_list_name
    write_list(out_list, out_paths)
    (out_root / "point_feature_sync_report.json").write_text(
        json.dumps({"out_list": str(out_list), "compacts": reports}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {out_list}")


if __name__ == "__main__":
    main()
