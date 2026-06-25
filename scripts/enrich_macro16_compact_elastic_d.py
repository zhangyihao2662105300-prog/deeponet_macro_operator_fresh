"""Copy elastic_D from source compacts into Macro16 compacts.

This is a small preparation step for force-residual training.  It does not
change q48, LE, B, point rules, or any model-visible geometry field.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def read_path_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def collect_paths(compacts: list[str], compact_list: str) -> list[Path]:
    paths = [Path(v).resolve() for v in compacts]
    if compact_list:
        paths.extend(Path(v).resolve() for v in read_path_list(Path(compact_list)))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        raise ValueError("provide --compact or --compact-list")
    return unique


def parse_path_map(values: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"path map must be FROM=TO, got {item!r}")
        src, dst = item.split("=", 1)
        pairs.append((src, dst))
    return pairs


def map_source_path(path_text: str, path_map: list[tuple[str, str]]) -> Path:
    text = str(path_text)
    for src, dst in path_map:
        if text.startswith(src):
            text = dst + text[len(src) :]
            break
    text = text.replace("\\", os.sep)
    return Path(text).resolve()


def scalar_text(z: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in z.files:
        return ""
    arr = np.asarray(z[key])
    if arr.size == 0:
        return ""
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def infer_n(z: np.lib.npyio.NpzFile, path: Path) -> int:
    for key in ("q48_def_hat", "q48_raw", "q48_hat", "LE_macro", "LE128_base"):
        if key in z.files:
            arr = np.asarray(z[key])
            if arr.ndim >= 1:
                return int(arr.shape[0])
    raise KeyError(f"{path}: cannot infer frame count")


def source_rows(z: np.lib.npyio.NpzFile, n: int) -> np.ndarray:
    if "source_row" not in z.files:
        return np.arange(n, dtype=np.int64)
    rows = np.asarray(z["source_row"], dtype=np.int64)
    if rows.shape != (n,):
        raise ValueError(f"source_row must have shape [{n}], got {rows.shape}")
    return rows


def read_source_elastic_d(source_path: Path, rows: np.ndarray) -> tuple[np.ndarray, str]:
    with np.load(str(source_path), allow_pickle=True) as z:
        n_total = infer_n(z, source_path)
        for key in ("elastic_D", "material_D"):
            if key not in z.files:
                continue
            vals = np.asarray(z[key], dtype=np.float32)
            if vals.shape == (6, 6):
                return np.broadcast_to(vals.reshape(1, 6, 6), (rows.size, 6, 6)).copy(), key
            if vals.shape == (1, 6, 6):
                return np.broadcast_to(vals, (rows.size, 6, 6)).copy(), key
            if vals.shape == (n_total, 6, 6):
                return vals[rows].astype(np.float32), key
            raise ValueError(f"{source_path}: {key} must be [6,6], [1,6,6], or [{n_total},6,6], got {vals.shape}")
    raise KeyError(f"{source_path}: missing elastic_D or material_D")


def write_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.stem + ".", suffix=".npz", dir=str(path.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def enrich_one(path: Path, out_path: Path, path_map: list[tuple[str, str]]) -> dict[str, Any]:
    with np.load(str(path), allow_pickle=True) as z:
        n = infer_n(z, path)
        source_text = scalar_text(z, "source_compact")
        if not source_text:
            raise KeyError(f"{path}: missing source_compact")
        rows = source_rows(z, n)
        source_path = map_source_path(source_text, path_map)
        elastic_d, source_key = read_source_elastic_d(source_path, rows)
        payload = {key: z[key] for key in z.files}
    payload["elastic_D"] = elastic_d.astype(np.float32)
    payload["elastic_D_source"] = np.asarray(f"copied_from_source_{source_key}", dtype=object)
    payload["elastic_D_source_compact"] = np.asarray(str(source_path), dtype=object)
    write_npz(out_path, payload)
    return {
        "compact": str(path),
        "out": str(out_path),
        "source_compact": str(source_path),
        "source_key": source_key,
        "frame_count": int(elastic_d.shape[0]),
        "elastic_D_shape": list(elastic_d.shape),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = collect_paths(list(getattr(args, "compact", [])), str(getattr(args, "compact_list", "")))
    path_map = parse_path_map(list(getattr(args, "source_path_map", [])))
    in_place = bool(getattr(args, "in_place", False))
    out_dir_text = str(getattr(args, "out_dir", "") or "")
    if not in_place and not out_dir_text:
        raise ValueError("provide --out-dir or --in-place")
    out_dir = Path(out_dir_text).resolve() if out_dir_text else None
    rows: list[dict[str, Any]] = []
    for path in paths:
        out_path = path if in_place else (out_dir / path.name)  # type: ignore[operator]
        rows.append(enrich_one(path, out_path, path_map))
    summary = {
        "script": "enrich_macro16_compact_elastic_d.py",
        "compact_count": int(len(rows)),
        "in_place": in_place,
        "items": rows,
    }
    out_summary = str(getattr(args, "summary_out", "") or "")
    if out_summary:
        summary_path = Path(out_summary).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        summary["summary_out"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="append", default=[])
    parser.add_argument("--compact-list", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--source-path-map", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
