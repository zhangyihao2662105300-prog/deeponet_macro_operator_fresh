#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit whether TRUE176 compact labels can use shape4-reconstructed IP geometry.

The current compact files may contain only:

    shape4, q48_raw, LE128_base, B_LE128_forward

That is still enough to reconstruct the reference CSS8 integration-point
geometry if, and only if, the LE/B rows use the standard generation order:

    element 1 IP1..IP8, element 2 IP1..IP8, ..., element 16 IP1..IP8

Original per-sample NPZ files written by the generation scripts store
``ip_keys``.  This audit checks those keys against the standard order and writes
a JSON report that can be kept beside a training run.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

NX = 4
NY = 4
GAUSS_1D = np.asarray([-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)], dtype=np.float64)
GAUSS_POINTS = tuple((r, s, t) for t in GAUSS_1D for s in GAUSS_1D for r in GAUSS_1D)
NODE_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


def standard_ip_keys(nx: int = 4, ny: int = 4) -> np.ndarray:
    rows: list[list[int]] = []
    for elem in range(1, int(nx) * int(ny) + 1):
        for ip in range(1, 9):
            rows.append([elem, ip, 0])
    return np.asarray(rows, dtype=np.int64)


def _fine_axis_coords(n_sub: int = 4) -> np.ndarray:
    out: list[float] = []
    h = 2.0 / float(n_sub)
    for i in range(n_sub):
        center = -1.0 + (float(i) + 0.5) * h
        for g in GAUSS_1D:
            out.append(center + 0.5 * h * float(g))
    return np.asarray(out, dtype=np.float64)


def _standard_css8_row_map(nx: int = 4, ny: int = 4) -> np.ndarray:
    fine_x = _fine_axis_coords(nx)
    fine_y = _fine_axis_coords(ny)
    rows: list[list[float]] = []
    data_row = 0
    for elem in range(1, int(nx) * int(ny) + 1):
        ex = (elem - 1) % int(nx)
        ey = (elem - 1) // int(nx)
        for ip in range(1, 9):
            gp = ip - 1
            lr = gp % 2
            ls = (gp // 2) % 2
            iz = gp // 4
            ix = ex * 2 + lr
            iy = ey * 2 + ls
            rows.append([data_row, elem, ip, gp, ex, ey, ix, iy, iz, fine_x[ix], fine_y[iy], GAUSS_1D[iz]])
            data_row += 1
    return np.asarray(rows, dtype=np.float64)


def _node_id(i: int, j: int, k: int) -> int:
    return 1 + int(k) * (NX + 1) * (NY + 1) + int(j) * (NX + 1) + int(i)


def _css8_elements() -> dict[int, list[int]]:
    elements: dict[int, list[int]] = {}
    eid = 1
    for j in range(NY):
        for i in range(NX):
            elements[eid] = [
                _node_id(i, j, 0),
                _node_id(i + 1, j, 0),
                _node_id(i + 1, j + 1, 0),
                _node_id(i, j + 1, 0),
                _node_id(i, j, 1),
                _node_id(i + 1, j, 1),
                _node_id(i + 1, j + 1, 1),
                _node_id(i, j + 1, 1),
            ]
            eid += 1
    return elements


def _css8_shape(r: float, s: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    n = np.empty(8, dtype=np.float64)
    dndr = np.empty((8, 3), dtype=np.float64)
    for a, (ra, sa, ta) in enumerate(NODE_SIGNS):
        n[a] = 0.125 * (1.0 + ra * r) * (1.0 + sa * s) * (1.0 + ta * t)
        dndr[a, 0] = 0.125 * ra * (1.0 + sa * s) * (1.0 + ta * t)
        dndr[a, 1] = 0.125 * sa * (1.0 + ra * r) * (1.0 + ta * t)
        dndr[a, 2] = 0.125 * ta * (1.0 + ra * r) * (1.0 + sa * s)
    return n, dndr


def _derive_shape4(shape4: np.ndarray) -> dict[str, float | str | None]:
    lam, tau, chi, mu = [float(v) for v in np.asarray(shape4, dtype=np.float64).reshape(4)]
    if abs(chi) <= 1.0e-12:
        return {"shape_type": "flat", "lambda": lam, "tau": tau, "chi": 0.0, "mu": mu, "root": 1.0, "R_mid": None, "theta": 0.0}
    root = math.sqrt(1.0 + mu * mu)
    r_mid = 1.0 / (chi * root)
    theta = lam * chi * root
    return {"shape_type": "cylinder" if abs(mu) <= 1.0e-12 else "cone", "lambda": lam, "tau": tau, "chi": chi, "mu": mu, "root": root, "R_mid": r_mid, "theta": theta}


def _frame_at_params(shape4: np.ndarray, u: np.ndarray | float, v: np.ndarray | float) -> np.ndarray:
    d = _derive_shape4(shape4)
    uu, vv = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
    if d["shape_type"] == "flat":
        out = np.zeros(uu.shape + (3, 3), dtype=np.float64)
        out[..., 0, 0] = 1.0
        out[..., 1, 1] = 1.0
        out[..., 2, 2] = 1.0
        return out
    theta = float(d["theta"])
    mu = float(d["mu"])
    root = float(d["root"])
    phi = -vv * theta
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    e_width = np.stack([sin_p, np.zeros_like(phi), -cos_p], axis=-1)
    e_axial = np.stack([mu * cos_p / root, np.ones_like(phi) / root, mu * sin_p / root], axis=-1)
    e_normal = np.stack([cos_p / root, -mu * np.ones_like(phi) / root, sin_p / root], axis=-1)
    return np.stack([e_width, e_axial, e_normal], axis=-2)


def _point_at_params(shape4: np.ndarray, u: np.ndarray | float, v: np.ndarray | float, w: np.ndarray | float) -> np.ndarray:
    d = _derive_shape4(shape4)
    uu, vv, ww = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64), np.asarray(w, dtype=np.float64))
    lam = float(d["lambda"])
    tau = float(d["tau"])
    if d["shape_type"] == "flat":
        return np.stack([lam * vv, uu, tau * ww], axis=-1)
    theta = float(d["theta"])
    mu = float(d["mu"])
    r_mid = float(d["R_mid"])
    phi = -vv * theta
    radius = r_mid + mu * uu
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    normal = _frame_at_params(shape4, uu, vv)[..., 2, :]
    mid = np.stack([radius * cos_p, uu, radius * sin_p], axis=-1)
    return mid + (ww * tau)[..., None] * normal


def _build_shape4_nodes(shape4: np.ndarray) -> np.ndarray:
    nodes = np.empty(((NX + 1) * (NY + 1) * 2, 3), dtype=np.float64)
    for k in range(2):
        w = -0.5 + float(k)
        for j in range(NY + 1):
            u = -0.5 + float(j) / float(NY)
            for i in range(NX + 1):
                v = -0.5 + float(i) / float(NX)
                nodes[_node_id(i, j, k) - 1] = _point_at_params(shape4, u, v, w)
    return nodes


def _shape4_detj(shape4: np.ndarray) -> np.ndarray:
    row = _standard_css8_row_map()
    elements = _css8_elements()
    nodes = _build_shape4_nodes(shape4)
    detj = np.empty(128, dtype=np.float64)
    for ri, rr in enumerate(row):
        elem = int(rr[1])
        gp = int(rr[3])
        _nshape, dndr = _css8_shape(*GAUSS_POINTS[gp])
        x = nodes[np.asarray(elements[elem], dtype=np.int64) - 1]
        detj[ri] = float(np.linalg.det(dndr.T.dot(x)))
    return detj


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


def _read_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]


def _compact_paths(args: argparse.Namespace) -> list[Path]:
    out: list[Path] = []
    for item in args.compact:
        out.append(Path(item).resolve())
    if str(args.compact_list).strip():
        out.extend(Path(p).resolve() for p in _read_list(Path(args.compact_list)))
    seen: set[str] = set()
    uniq: list[Path] = []
    for path in out:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(path)
    return uniq


def _sample_paths_from_compacts(compacts: list[Path]) -> list[Path]:
    out: list[Path] = []
    for compact in compacts:
        with np.load(str(compact), allow_pickle=True) as z:
            if "sample_paths" not in z.files:
                continue
            out.extend(Path(str(p)).resolve() for p in np.asarray(z["sample_paths"]).reshape(-1))
    return out


def _sample_paths_from_args(args: argparse.Namespace, compacts: list[Path]) -> list[Path]:
    out: list[Path] = []
    out.extend(_sample_paths_from_compacts(compacts))
    for item in args.sample_npz:
        out.append(Path(item).resolve())
    for pattern in args.sample_glob:
        out.extend(Path(p).resolve() for p in glob.glob(str(pattern), recursive=True))
    seen: set[str] = set()
    uniq: list[Path] = []
    for path in out:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(path)
    if int(args.max_samples) > 0:
        uniq = uniq[: int(args.max_samples)]
    return uniq


def _first_mismatches(keys: np.ndarray, expected: np.ndarray, limit: int) -> list[dict[str, Any]]:
    kk = np.asarray(keys, dtype=np.int64).reshape(-1, 3)
    ee = np.asarray(expected, dtype=np.int64).reshape(-1, 3)
    if kk.shape != ee.shape:
        return [{"row": -1, "reason": f"shape mismatch {kk.shape} vs {ee.shape}"}]
    bad = np.flatnonzero(np.any(kk != ee, axis=1))
    rows: list[dict[str, Any]] = []
    for row in bad[: max(0, int(limit))]:
        rows.append(
            {
                "row": int(row),
                "actual": kk[row].tolist(),
                "expected": ee[row].tolist(),
            }
        )
    return rows


def _audit_sample(path: Path, expected: np.ndarray, mismatch_rows: int) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(path), "exists": bool(path.exists())}
    if not path.exists():
        row["status"] = "MISSING"
        return row
    try:
        with np.load(str(path), allow_pickle=True) as z:
            row["files"] = sorted(str(k) for k in z.files)
            row["has_ip_keys"] = bool("ip_keys" in z.files)
            if "shape4" in z.files:
                shape4 = np.asarray(z["shape4"], dtype=np.float64).reshape(4)
                row["shape4"] = shape4.tolist()
                detj = _shape4_detj(shape4)
                row["detJ_min"] = float(detj.min())
                row["detJ_max"] = float(detj.max())
                row["detJ_abs_min"] = float(np.abs(detj).min())
                row["detJ_positive_all"] = bool(np.all(detj > 0.0))
                if "IVOL128_inferred_from_DLE" in z.files:
                    ivol = np.asarray(z["IVOL128_inferred_from_DLE"], dtype=np.float64).reshape(-1, 128)
                    ivol_mean = ivol.mean(axis=0)
                    denom = np.maximum(np.maximum(np.abs(ivol_mean), np.abs(detj)), 1.0e-30)
                    rel = np.abs(ivol_mean - detj) / denom
                    row["ivol_mean_vs_detJ_rel_max"] = float(rel.max())
                    row["ivol_mean_vs_detJ_abs_max"] = float(np.abs(ivol_mean - detj).max())
                    row["ivol_min"] = float(ivol.min())
                    row["ivol_max"] = float(ivol.max())
            if "ip_keys" not in z.files:
                row["status"] = "NO_IP_KEYS"
                return row
            keys = np.asarray(z["ip_keys"], dtype=np.int64).reshape(-1, 3)
            match = bool(keys.shape == expected.shape and np.array_equal(keys, expected))
            row["ip_keys_shape"] = list(keys.shape)
            row["ip_keys_match_standard"] = match
            if match:
                row["ip_key_mismatch_count"] = 0
            elif keys.shape == expected.shape:
                row["ip_key_mismatch_count"] = int(np.count_nonzero(np.any(keys != expected, axis=1)))
            else:
                row["ip_key_mismatch_count"] = int(max(keys.shape[0], expected.shape[0]))
            row["mismatches"] = [] if match else _first_mismatches(keys, expected, mismatch_rows)
            row["status"] = "PASS" if match else "FAIL"
    except Exception as exc:
        row["status"] = "ERROR"
        row["error"] = f"{exc.__class__.__name__}: {exc}"
    return row


def _compact_summary(compacts: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in compacts:
        row: dict[str, Any] = {"path": str(path), "exists": bool(path.exists())}
        if path.exists():
            with np.load(str(path), allow_pickle=True) as z:
                row["files"] = sorted(str(k) for k in z.files)
                row["frames"] = int(z["shape4"].shape[0]) if "shape4" in z.files else 0
                row["has_point_fields"] = bool(
                    any(k in z.files for k in ("point_features", "ip_features", "trunk_features", "ip_xi", "ip_xyz", "ip_J", "ip_detJ"))
                )
                row["sample_paths"] = int(np.asarray(z["sample_paths"]).size) if "sample_paths" in z.files else 0
                if "shape4" in z.files:
                    row["unique_shape4_count"] = int(np.unique(np.asarray(z["shape4"], dtype=np.float32).reshape(-1, 4), axis=0).shape[0])
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    compacts = _compact_paths(args)
    sample_paths = _sample_paths_from_args(args, compacts)
    expected = standard_ip_keys()
    rows = [_audit_sample(path, expected, int(args.mismatch_rows)) for path in sample_paths]
    checked = [r for r in rows if r.get("status") in {"PASS", "FAIL", "NO_IP_KEYS"}]
    key_rows = [r for r in rows if r.get("has_ip_keys") is True]
    pass_rows = [r for r in rows if r.get("status") == "PASS"]
    fail_rows = [r for r in rows if r.get("status") == "FAIL"]
    summary = {
        "compact_count": int(len(compacts)),
        "sample_count": int(len(rows)),
        "checked_sample_count": int(len(checked)),
        "samples_with_ip_keys": int(len(key_rows)),
        "pass_count": int(len(pass_rows)),
        "fail_count": int(len(fail_rows)),
        "missing_count": int(sum(1 for r in rows if r.get("status") == "MISSING")),
        "error_count": int(sum(1 for r in rows if r.get("status") == "ERROR")),
        "no_ip_keys_count": int(sum(1 for r in rows if r.get("status") == "NO_IP_KEYS")),
        "available_ip_key_samples_all_match_standard": bool(key_rows and len(fail_rows) == 0),
        "can_use_shape4_audited": bool(key_rows and len(fail_rows) == 0),
        "can_use_shape4_audited_meaning": (
            "Available original sample NPZ files with ip_keys match the standard row order. "
            "Missing sample NPZ files are covered only by the assumption that they were produced by the same TRUE176 shape4 generator."
        ),
        "standard_ip_key_order": "element 1 IP1..IP8, ..., element 16 IP1..IP8",
        "compact_summary": _compact_summary(compacts),
    }
    report = {"summary": summary, "samples": rows}
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    print(f"wrote {out}")
    if bool(args.fail_on_error) and not bool(summary["can_use_shape4_audited"]):
        raise SystemExit(2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="append", default=[])
    parser.add_argument("--compact-list", default="")
    parser.add_argument("--sample-npz", action="append", default=[])
    parser.add_argument("--sample-glob", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--mismatch-rows", type=int, default=16)
    parser.add_argument("--out", default="runs/true176_shape4_ip_contract_audit.json")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
