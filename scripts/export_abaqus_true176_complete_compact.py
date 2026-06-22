#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export a TRUE176/CSS8 DeepONet-ready compact directly from an Abaqus ODB.

This is the production version of the ODB probe: the resulting NPZ contains the
branch inputs, query-point inputs, Abaqus strain labels, and optionally the
Sobolev B labels copied from an existing finite-difference compact.

Run with Abaqus Python:

    abaqus python scripts/export_abaqus_true176_complete_compact.py \
      --odb job.odb \
      --out job_complete_compact.npz \
      --merge-compact old_b_compact.npz \
      --require-b
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from abaqusConstants import INTEGRATION_POINT  # type: ignore
    from odbAccess import openOdb  # type: ignore
except Exception:  # pragma: no cover - available only inside Abaqus Python
    INTEGRATION_POINT = None  # type: ignore
    openOdb = None  # type: ignore


SCHEMA_VERSION = "abaqus_true176_complete_operator_compact_v1"
DEFAULT_KEEP_NODE_LABELS = (1, 3, 5, 11, 15, 21, 23, 25, 26, 28, 30, 36, 40, 46, 48, 50)
GAUSS_1D = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))
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


def parse_ints(text: str) -> list[int]:
    return [int(v) for v in str(text).replace(";", ",").split(",") if v.strip()]


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


def css8_shape(r: float, s: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    n = np.empty(8, dtype=np.float64)
    dndr = np.empty((8, 3), dtype=np.float64)
    for a, (ra, sa, ta) in enumerate(NODE_SIGNS):
        n[a] = 0.125 * (1.0 + ra * r) * (1.0 + sa * s) * (1.0 + ta * t)
        dndr[a, 0] = 0.125 * ra * (1.0 + sa * s) * (1.0 + ta * t)
        dndr[a, 1] = 0.125 * sa * (1.0 + ra * r) * (1.0 + ta * t)
        dndr[a, 2] = 0.125 * ta * (1.0 + ra * r) * (1.0 + sa * s)
    return n, dndr


def field_values(field: Any, *, integration_point_only: bool = False) -> list[Any]:
    if integration_point_only:
        if INTEGRATION_POINT is None:
            raise RuntimeError("Abaqus INTEGRATION_POINT constant is unavailable outside Abaqus Python")
        try:
            return list(field.getSubset(position=INTEGRATION_POINT).values)
        except Exception:
            pass
    return list(field.values)


def has_element_ip(value: Any) -> bool:
    try:
        return getattr(value, "elementLabel") is not None and getattr(value, "integrationPoint") is not None
    except Exception:
        return False


def section_number(value: Any) -> int:
    try:
        section = getattr(value, "sectionPoint")
        if section is None:
            return 0
        return int(getattr(section, "number", 0))
    except Exception:
        return 0


def value_key(value: Any) -> tuple[int, int, int]:
    return int(value.elementLabel), int(value.integrationPoint), section_number(value)


def tensor_field_by_ip(frame: Any, field_name: str, width: int) -> dict[tuple[int, int, int], np.ndarray]:
    if field_name not in frame.fieldOutputs:
        raise KeyError("frame is missing field output %s" % field_name)
    out: dict[tuple[int, int, int], np.ndarray] = {}
    for value in field_values(frame.fieldOutputs[field_name], integration_point_only=True):
        if not has_element_ip(value):
            continue
        data = np.asarray(value.data, dtype=np.float64).reshape(-1)
        if data.size < int(width):
            raise ValueError("%s value at %s has width %d < %d" % (field_name, value_key(value), data.size, width))
        out[value_key(value)] = data[: int(width)]
    return out


def nodal_field(frame: Any, field_name: str, width: int) -> dict[int, np.ndarray]:
    if field_name not in frame.fieldOutputs:
        raise KeyError("frame is missing nodal field output %s" % field_name)
    out: dict[int, np.ndarray] = {}
    for value in frame.fieldOutputs[field_name].values:
        node_label = getattr(value, "nodeLabel", None)
        if node_label is None:
            continue
        data = np.asarray(value.data, dtype=np.float64).reshape(-1)
        if data.size < int(width):
            raise ValueError("%s node value %s has width %d < %d" % (field_name, node_label, data.size, width))
        out[int(node_label)] = data[: int(width)]
    return out


def choose_instance(odb: Any, name: str = "") -> Any:
    if name:
        key_map = {str(k).upper(): v for k, v in odb.rootAssembly.instances.items()}
        key = str(name).upper()
        if key not in key_map:
            raise KeyError("instance %s not found; available=%s" % (name, sorted(odb.rootAssembly.instances.keys())))
        return key_map[key]
    candidates = [inst for inst in odb.rootAssembly.instances.values() if len(inst.elements) > 0]
    if not candidates:
        raise RuntimeError("ODB has no assembly instance with elements")
    if len(candidates) > 1:
        # TRUE176 jobs normally have one part instance; require an explicit name
        # when that is not true to avoid silently mixing element sets.
        names = [str(inst.name) for inst in candidates]
        raise RuntimeError("multiple element instances found; pass --instance. available=%s" % names)
    return candidates[0]


def frame_indices(step: Any, mode: str) -> list[int]:
    key = str(mode).strip().lower()
    if key == "last":
        return [len(step.frames) - 1]
    if key == "all":
        return list(range(len(step.frames)))
    if key == "noninitial":
        return [i for i, frame in enumerate(step.frames) if abs(float(frame.frameValue)) > 0.0]
    return parse_ints(key)


def element_and_node_maps(instance: Any) -> tuple[dict[int, Any], dict[int, np.ndarray], np.ndarray, np.ndarray]:
    elem_by_label = {int(elem.label): elem for elem in instance.elements}
    node_labels = np.asarray([int(node.label) for node in instance.nodes], dtype=np.int64)
    order = np.argsort(node_labels)
    node_labels = node_labels[order]
    node_xyz = np.asarray([np.asarray(instance.nodes[int(i)].coordinates, dtype=np.float64) for i in order], dtype=np.float64)
    node_by_label = {int(label): node_xyz[i] for i, label in enumerate(node_labels)}
    return elem_by_label, node_by_label, node_labels, node_xyz


def local_frame_from_j(jmat: np.ndarray) -> np.ndarray:
    a0 = np.asarray(jmat[0], dtype=np.float64)
    a1 = np.asarray(jmat[1], dtype=np.float64)
    a2 = np.asarray(jmat[2], dtype=np.float64)
    e0 = a0 / max(float(np.linalg.norm(a0)), 1.0e-30)
    a1p = a1 - float(np.dot(a1, e0)) * e0
    e1 = a1p / max(float(np.linalg.norm(a1p)), 1.0e-30)
    e2 = np.cross(e0, e1)
    e2 = e2 / max(float(np.linalg.norm(e2)), 1.0e-30)
    if float(np.dot(e2, a2)) < 0.0:
        e2 = -e2
    e1 = np.cross(e2, e0)
    e1 = e1 / max(float(np.linalg.norm(e1)), 1.0e-30)
    return np.stack([e0, e1, e2], axis=0)


def reference_ip_geometry(
    instance: Any,
    keys: list[tuple[int, int, int]],
) -> dict[str, np.ndarray]:
    elem_by_label, node_by_label, _node_labels, _node_xyz = element_and_node_maps(instance)
    p = len(keys)
    ip_xi = np.empty((p, 3), dtype=np.float64)
    ip_xyz = np.empty((p, 3), dtype=np.float64)
    ip_j = np.empty((p, 3, 3), dtype=np.float64)
    ip_invj = np.empty((p, 3, 3), dtype=np.float64)
    ip_detj = np.empty((p,), dtype=np.float64)
    ip_frame = np.empty((p, 3, 3), dtype=np.float64)
    element_types: list[str] = []
    for i, (elem_label, ip, _section) in enumerate(keys):
        if ip < 1 or ip > 8:
            raise ValueError("CSS8/C3D8 integration point must be 1..8; got %s" % (ip,))
        elem = elem_by_label[int(elem_label)]
        element_types.append(str(elem.type))
        conn = list(elem.connectivity)
        if len(conn) != 8:
            raise ValueError("element %s has %d connectivity nodes; expected 8-node CSS8/Hex8" % (elem_label, len(conn)))
        x = np.asarray([node_by_label[int(label)] for label in conn], dtype=np.float64)
        gp = GAUSS_POINTS[ip - 1]
        nshape, dndr = css8_shape(*gp)
        jmat = dndr.T.dot(x)
        detj = float(np.linalg.det(jmat))
        ip_xi[i] = np.asarray(gp, dtype=np.float64)
        ip_xyz[i] = nshape.dot(x)
        ip_j[i] = jmat
        ip_invj[i] = np.linalg.inv(jmat)
        ip_detj[i] = detj
        ip_frame[i] = local_frame_from_j(jmat)
    return {
        "ip_xi": ip_xi,
        "ip_xyz": ip_xyz,
        "ip_J": ip_j,
        "ip_invJ": ip_invj,
        "ip_detJ": ip_detj,
        "ip_frame": ip_frame,
        "element_types": np.asarray(sorted(set(element_types)), dtype=object),
    }


def point_features_from_raw(geom: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ip_xi = np.asarray(geom["ip_xi"], dtype=np.float64)
    ip_xyz = np.asarray(geom["ip_xyz"], dtype=np.float64)
    ip_frame = np.asarray(geom["ip_frame"], dtype=np.float64).reshape(ip_xi.shape[0], 9)
    ip_j = np.asarray(geom["ip_J"], dtype=np.float64).reshape(ip_xi.shape[0], 9)
    ip_invj = np.asarray(geom["ip_invJ"], dtype=np.float64).reshape(ip_xi.shape[0], 9)
    ip_detj = np.asarray(geom["ip_detJ"], dtype=np.float64).reshape(ip_xi.shape[0], 1)
    log_detj = np.log(np.maximum(np.abs(ip_detj), 1.0e-300))
    point = np.concatenate([ip_xi, ip_xyz, ip_frame, ip_j, ip_invj, ip_detj, log_detj], axis=1).astype(np.float32)
    names = ["ip_xi_r", "ip_xi_s", "ip_xi_t", "ip_xyz_x", "ip_xyz_y", "ip_xyz_z"]
    names += ["ip_frame_%d%d" % (i, j) for i in range(3) for j in range(3)]
    names += ["ip_J_%d%d" % (i, j) for i in range(3) for j in range(3)]
    names += ["ip_invJ_%d%d" % (i, j) for i in range(3) for j in range(3)]
    names += ["ip_detJ", "log_abs_detJ"]
    return point, np.asarray(names, dtype=object)


def load_shape4_arg(text: str) -> np.ndarray | None:
    raw = str(text).strip()
    if not raw:
        return None
    values = [float(v) for v in raw.replace(";", ",").split(",") if v.strip()]
    if len(values) != 4:
        raise ValueError("--shape4 must contain exactly four comma-separated numbers")
    return np.asarray(values, dtype=np.float32).reshape(4)


def load_shape4_json(path_text: str) -> np.ndarray | None:
    if not str(path_text).strip():
        return None
    path = Path(path_text).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("shape4", "shape"):
            if key in data:
                arr = np.asarray(data[key], dtype=np.float32)
                return arr.reshape(4) if arr.size == 4 else arr.reshape(-1, 4)
    arr = np.asarray(data, dtype=np.float32)
    return arr.reshape(4) if arr.size == 4 else arr.reshape(-1, 4)


def broadcast_or_match(arr: np.ndarray, n: int, tail: tuple[int, ...], name: str) -> np.ndarray:
    vals = np.asarray(arr)
    if vals.shape == tail:
        return np.broadcast_to(vals.reshape((1,) + tail), (int(n),) + tail).copy()
    if vals.shape == (int(n),) + tail:
        return vals
    raise ValueError("%s must have shape %s or %s, got %s" % (name, tail, (int(n),) + tail, vals.shape))


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _enforce_tolerance(name: str, diff: float, tol: float, *, allow_mismatch: bool) -> None:
    if float(diff) > float(tol) and not bool(allow_mismatch):
        raise ValueError(f"{name} mismatch exceeds tolerance: diff={diff:.9g}, tol={float(tol):.9g}")


def _scalar_text(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
    elif arr.size == 1:
        item = arr.reshape(-1)[0]
    else:
        item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item)


def canonical_strain_field(value: Any) -> str:
    text = _scalar_text(value).strip()
    if not text:
        return "unknown"
    key = text.upper()
    if key in {"UNKNOWN", "NONE", "NULL", "NA", "N/A"}:
        return "unknown"
    return key


def _known_strain_field(value: str) -> bool:
    return canonical_strain_field(value) != "unknown"


def _enforce_strain_field_match(name: str, current: str, merged: str, *, source: Path, allow_mismatch: bool) -> bool:
    cur = canonical_strain_field(current)
    old = canonical_strain_field(merged)
    if _known_strain_field(cur) and _known_strain_field(old) and cur != old:
        if not bool(allow_mismatch):
            raise ValueError(f"{name} mismatch for {source}: current={cur}, merged={old}")
        return False
    return True


def _check_ip_keys_match(current: np.ndarray, merged: np.ndarray, *, source: Path) -> None:
    cur = np.asarray(current, dtype=np.int64).reshape(-1, np.asarray(current).shape[-1])
    old = np.asarray(merged, dtype=np.int64)
    if old.ndim == 3:
        old = old[0]
    old = old.reshape(-1, old.shape[-1])
    if cur.shape != old.shape or not np.array_equal(cur, old):
        raise ValueError(
            f"merged ip_keys in {source} do not match current Abaqus ip_keys: "
            f"current_shape={cur.shape}, merged_shape={old.shape}"
        )


def merge_existing_payload(
    payload: dict[str, Any],
    merge_path: str,
    *,
    require_b: bool,
    merge_q_tol: float = 1.0e-8,
    merge_le_tol: float = 1.0e-8,
    require_merge_ip_keys: bool = False,
    allow_merge_mismatch: bool = False,
) -> dict[str, Any]:
    if not str(merge_path).strip():
        if require_b:
            raise ValueError("--require-b was set but --merge-compact was not provided")
        return payload
    path = Path(merge_path).resolve()
    n = int(payload["q48_raw"].shape[0])
    p = int(payload["ip_keys"].shape[0])
    with np.load(str(path), allow_pickle=True) as z:
        copied = []
        current_strain = canonical_strain_field(payload.get("strain_field", "unknown"))
        current_b_strain = canonical_strain_field(payload.get("B_label_strain_field", current_strain))
        merged_strain = canonical_strain_field(z["strain_field"]) if "strain_field" in z.files else "unknown"
        merged_b_strain = (
            canonical_strain_field(z["B_label_strain_field"]) if "B_label_strain_field" in z.files else merged_strain
        )
        strain_match = True
        strain_match = (
            _enforce_strain_field_match(
                "merge strain_field",
                current_strain,
                merged_strain,
                source=path,
                allow_mismatch=bool(allow_merge_mismatch),
            )
            and strain_match
        )
        strain_match = (
            _enforce_strain_field_match(
                "merge B_label_strain_field",
                current_b_strain,
                merged_b_strain,
                source=path,
                allow_mismatch=bool(allow_merge_mismatch),
            )
            and strain_match
        )
        strain_match = (
            _enforce_strain_field_match(
                "merge strain_field vs B_label_strain_field",
                merged_strain,
                merged_b_strain,
                source=path,
                allow_mismatch=bool(allow_merge_mismatch),
            )
            and strain_match
        )
        payload["merge_strain_field"] = np.asarray(merged_strain, dtype=object)
        payload["merge_B_label_strain_field"] = np.asarray(merged_b_strain, dtype=object)
        payload["merge_strain_field_match"] = np.asarray(bool(strain_match))
        for key in z.files:
            if key in payload:
                continue
            if key in {"point_features", "point_feature_names", "point_feature_source"}:
                continue
            payload[key] = z[key]
            copied.append(key)
        if "B_LE128_forward" in z.files:
            b = np.asarray(z["B_LE128_forward"], dtype=np.float32)
            if b.shape != (n, p, 6, 48):
                raise ValueError("merged B_LE128_forward expected shape %s, got %s" % ((n, p, 6, 48), b.shape))
            payload["B_LE128_forward"] = b
        elif require_b:
            raise ValueError("%s does not contain B_LE128_forward" % path)
        if "shape4" in z.files and "shape4" not in payload:
            payload["shape4"] = broadcast_or_match(np.asarray(z["shape4"], dtype=np.float32), n, (4,), "shape4")
        if "q48_raw" in z.files:
            q_old = broadcast_or_match(np.asarray(z["q48_raw"], dtype=np.float32), n, (48,), "q48_raw")
            q_diff = _max_abs_diff(q_old, payload["q48_raw"])
            payload["merge_q48_max_abs_diff"] = np.asarray(q_diff, dtype=np.float64)
            _enforce_tolerance("merge q48_raw", q_diff, float(merge_q_tol), allow_mismatch=bool(allow_merge_mismatch))
        if "LE128_base" in z.files:
            le_old = broadcast_or_match(np.asarray(z["LE128_base"], dtype=np.float32), n, (p, 6), "LE128_base")
            le_diff = _max_abs_diff(le_old, payload["LE128_base"])
            payload["merge_LE128_base_max_abs_diff"] = np.asarray(le_diff, dtype=np.float64)
            _enforce_tolerance("merge LE128_base", le_diff, float(merge_le_tol), allow_mismatch=bool(allow_merge_mismatch))
        if "ip_keys" in z.files:
            _check_ip_keys_match(payload["ip_keys"], z["ip_keys"], source=path)
            payload["merge_ip_keys_match"] = np.asarray(True)
        elif bool(require_merge_ip_keys):
            raise ValueError(f"{path} is missing ip_keys; use --require-merge-ip-keys only with keyed B compacts")
    payload["merged_compact"] = np.asarray(str(path), dtype=object)
    payload["merged_keys"] = np.asarray(copied, dtype=object)
    payload["merge_q_tol"] = np.asarray(float(merge_q_tol), dtype=np.float64)
    payload["merge_le_tol"] = np.asarray(float(merge_le_tol), dtype=np.float64)
    payload["allow_merge_mismatch"] = np.asarray(bool(allow_merge_mismatch))
    return payload


def enforce_ip_audit(
    payload: dict[str, Any],
    *,
    require_ip_audit: bool,
    ip_xyz_tol: float,
    detj_ivol_tol: float,
    skip_ivol_audit: bool,
) -> dict[str, Any]:
    if "ip_xyz_abaqus_coord" in payload:
        coords = np.asarray(payload["ip_xyz_abaqus_coord"], dtype=np.float64)
        xyz = np.asarray(payload["ip_xyz"], dtype=np.float64).reshape(1, 128, 3)
        diff = float(np.nanmax(np.abs(coords - xyz)))
        payload["audit_ref_ip_xyz_vs_abaqus_coord_max_abs"] = np.asarray(diff, dtype=np.float64)
        if float(diff) > float(ip_xyz_tol):
            raise ValueError(f"IP COORD audit failed: diff={diff:.9g}, tol={float(ip_xyz_tol):.9g}")
    elif bool(require_ip_audit):
        raise ValueError("--require-ip-audit requires Abaqus COORD field output")

    if bool(skip_ivol_audit):
        payload["audit_detJ_vs_IVOL_skipped"] = np.asarray(True)
    elif "ip_IVOL_abaqus" in payload:
        ivol = np.asarray(payload["ip_IVOL_abaqus"], dtype=np.float64)
        detj = np.asarray(payload["ip_detJ"], dtype=np.float64).reshape(1, 128)
        diff = float(np.nanmax(np.abs(ivol - detj)))
        payload["audit_detJ_vs_IVOL_max_abs"] = np.asarray(diff, dtype=np.float64)
        if float(diff) > float(detj_ivol_tol):
            raise ValueError(
                f"TRUE176/CSS8 detJ-vs-IVOL audit failed: diff={diff:.9g}, tol={float(detj_ivol_tol):.9g}"
            )
    elif bool(require_ip_audit):
        raise ValueError("--require-ip-audit requires Abaqus IVOL field output unless --skip-ivol-audit is set")

    payload["require_ip_audit"] = np.asarray(bool(require_ip_audit))
    payload["ip_xyz_tol"] = np.asarray(float(ip_xyz_tol), dtype=np.float64)
    payload["detj_ivol_tol"] = np.asarray(float(detj_ivol_tol), dtype=np.float64)
    payload["skip_ivol_audit"] = np.asarray(bool(skip_ivol_audit))
    payload["ip_audit_scope"] = np.asarray("TRUE176/CSS8 128-IP solid contract; IVOL compared directly to detJ", dtype=object)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odb", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--instance", default="")
    parser.add_argument("--step", default="")
    parser.add_argument("--frames", default="all", help="all, noninitial, last, or comma-separated frame indices")
    parser.add_argument("--strain-field", default="LE")
    parser.add_argument("--keep-node-labels", default=",".join(str(v) for v in DEFAULT_KEEP_NODE_LABELS))
    parser.add_argument("--merge-compact", default="")
    parser.add_argument("--require-b", action="store_true")
    parser.add_argument("--merge-q-tol", type=float, default=1.0e-8)
    parser.add_argument("--merge-le-tol", type=float, default=1.0e-8)
    parser.add_argument("--require-merge-ip-keys", action="store_true")
    parser.add_argument("--allow-merge-mismatch", action="store_true")
    parser.add_argument("--require-ip-audit", action="store_true")
    parser.add_argument("--ip-xyz-tol", type=float, default=1.0e-6)
    parser.add_argument("--detj-ivol-tol", type=float, default=1.0e-8)
    parser.add_argument("--skip-ivol-audit", action="store_true")
    parser.add_argument("--shape4", default="")
    parser.add_argument("--shape4-json", default="")
    parser.add_argument("--length-scale", type=float, default=0.0)
    args = parser.parse_args()

    odb_path = Path(args.odb).resolve()
    out_path = Path(args.out).resolve()
    keep_labels = parse_ints(str(args.keep_node_labels))
    if len(keep_labels) != 16:
        raise ValueError("--keep-node-labels must contain 16 node labels")
    if openOdb is None:
        raise RuntimeError("This exporter must be run with Abaqus Python; odbAccess.openOdb is unavailable")

    odb = openOdb(path=str(odb_path), readOnly=True)
    try:
        instance = choose_instance(odb, str(args.instance).strip())
        step_name = str(args.step).strip() or list(odb.steps.keys())[-1]
        step = odb.steps[step_name]
        selected_frames = frame_indices(step, str(args.frames))
        if not selected_frames:
            raise ValueError("no frames selected")
        elem_by_label, node_by_label, node_labels, node_xyz = element_and_node_maps(instance)
        del elem_by_label

        strain_field_name = canonical_strain_field(str(args.strain_field))
        first_frame = step.frames[int(selected_frames[0])]
        le0 = tensor_field_by_ip(first_frame, str(args.strain_field), 6)
        keys = sorted(le0.keys())
        if len(keys) != 128:
            raise ValueError("TRUE176 complete compact expects 128 integration rows; got %d" % len(keys))
        geom = reference_ip_geometry(instance, keys)
        point_features, point_feature_names = point_features_from_raw(geom)
        ip_keys = np.asarray(keys, dtype=np.int64)

        x_keep = np.asarray([node_by_label[int(label)] for label in keep_labels], dtype=np.float32)
        x_macro = node_xyz.astype(np.float32)
        q_rows: list[np.ndarray] = []
        le_rows: list[np.ndarray] = []
        coord_rows: list[np.ndarray] = []
        ivol_rows: list[np.ndarray] = []
        frame_values: list[float] = []
        frame_indices_out: list[int] = []
        for frame_index in selected_frames:
            frame = step.frames[int(frame_index)]
            u_by_node = nodal_field(frame, "U", 3)
            q_rows.append(np.asarray([u_by_node.get(int(label), np.zeros(3, dtype=np.float64)) for label in keep_labels], dtype=np.float64).reshape(48))
            le_map = tensor_field_by_ip(frame, str(args.strain_field), 6)
            if sorted(le_map.keys()) != keys:
                raise ValueError("frame %d integration-point keys differ from first exported frame" % int(frame_index))
            le_rows.append(np.asarray([le_map[key] for key in keys], dtype=np.float64))
            if "COORD" in frame.fieldOutputs:
                coord_map = tensor_field_by_ip(frame, "COORD", 3)
                coord_rows.append(np.asarray([coord_map.get(key, np.full(3, np.nan)) for key in keys], dtype=np.float64))
            if "IVOL" in frame.fieldOutputs:
                ivol_map = tensor_field_by_ip(frame, "IVOL", 1)
                ivol_rows.append(np.asarray([ivol_map.get(key, np.asarray([np.nan]))[0] for key in keys], dtype=np.float64))
            frame_values.append(float(frame.frameValue))
            frame_indices_out.append(int(frame_index))

        q48 = np.asarray(q_rows, dtype=np.float32)
        le128 = np.asarray(le_rows, dtype=np.float32)
        n_frame = int(q48.shape[0])
        shape4 = load_shape4_arg(str(args.shape4))
        shape4_json = load_shape4_json(str(args.shape4_json))
        if shape4 is None:
            shape4 = shape4_json
        if shape4 is not None:
            shape4 = broadcast_or_match(shape4, n_frame, (4,), "shape4").astype(np.float32)

        payload: dict[str, Any] = {
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=object),
            "source_odb": np.asarray(str(odb_path), dtype=object),
            "source_instance": np.asarray(str(instance.name), dtype=object),
            "source_step": np.asarray(str(step_name), dtype=object),
            "source_frame_indices": np.asarray(frame_indices_out, dtype=np.int64),
            "source_frame_values": np.asarray(frame_values, dtype=np.float64),
            "sample_paths": np.asarray([str(odb_path)], dtype=object),
            "q48_raw": q48,
            "X_keep": x_keep,
            "X_macro": x_macro,
            "keep_node_labels": np.asarray(keep_labels, dtype=np.int64),
            "macro_node_labels": node_labels.astype(np.int64),
            "LE128_base": le128,
            "strain_field": np.asarray(strain_field_name, dtype=object),
            "B_label_strain_field": np.asarray(strain_field_name, dtype=object),
            "strain_label_key": np.asarray("LE128_base", dtype=object),
            "B_label_key": np.asarray("B_LE128_forward", dtype=object),
            "ip_keys": ip_keys,
            "ip_xi": np.asarray(geom["ip_xi"], dtype=np.float32),
            "ip_xyz": np.asarray(geom["ip_xyz"], dtype=np.float32),
            "ip_J": np.asarray(geom["ip_J"], dtype=np.float32),
            "ip_invJ": np.asarray(geom["ip_invJ"], dtype=np.float32),
            "ip_detJ": np.asarray(geom["ip_detJ"], dtype=np.float32),
            "ip_frame": np.asarray(geom["ip_frame"], dtype=np.float32),
            "point_features": point_features.astype(np.float32),
            "point_feature_names": point_feature_names,
            "point_feature_source": np.asarray("abaqus_odb_reference_mesh", dtype=object),
            "point_feature_target_ips": np.arange(128, dtype=np.int64),
            "point_feature_ip_keys": ip_keys,
            "point_feature_axis": np.asarray("point_features[k, :] aligns with LE128_base[:, k, :] and B_LE128_forward[:, k, :, :]", dtype=object),
            "point_feature_alignment": np.asarray("Abaqus integration-point keys sorted by (elementLabel, integrationPoint, sectionPoint)", dtype=object),
            "ip_frame_method": np.asarray("orthonormalized rows of J=[dX/dr; dX/ds; dX/dt]", dtype=object),
            "ip_J_convention": np.asarray("rows are physical derivatives with respect to local natural coordinates r,s,t", dtype=object),
            "element_types": np.asarray(geom["element_types"], dtype=object),
            "complete_branch_inputs": np.asarray(True),
            "complete_point_inputs": np.asarray(True),
            "complete_le_labels": np.asarray(True),
        }
        if shape4 is not None:
            payload["shape4"] = shape4
        if float(args.length_scale) > 0.0:
            payload["length_scale"] = np.full((n_frame, 1), float(args.length_scale), dtype=np.float32)
        if coord_rows:
            coords = np.asarray(coord_rows, dtype=np.float32)
            payload["ip_xyz_abaqus_coord"] = coords
        if ivol_rows:
            payload["ip_IVOL_abaqus"] = np.asarray(ivol_rows, dtype=np.float32)

        payload = enforce_ip_audit(
            payload,
            require_ip_audit=bool(args.require_ip_audit),
            ip_xyz_tol=float(args.ip_xyz_tol),
            detj_ivol_tol=float(args.detj_ivol_tol),
            skip_ivol_audit=bool(args.skip_ivol_audit),
        )
        payload = merge_existing_payload(
            payload,
            str(args.merge_compact),
            require_b=bool(args.require_b),
            merge_q_tol=float(args.merge_q_tol),
            merge_le_tol=float(args.merge_le_tol),
            require_merge_ip_keys=bool(args.require_merge_ip_keys),
            allow_merge_mismatch=bool(args.allow_merge_mismatch),
        )
        payload["complete_b_labels"] = np.asarray("B_LE128_forward" in payload)
        payload["training_ready_sobolev"] = np.asarray("B_LE128_forward" in payload)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(out_path), **payload)
        sidecar = out_path.with_suffix(out_path.suffix + ".json")
        sidecar_report = {
            "out": str(out_path),
            "schema_version": SCHEMA_VERSION,
            "frames": n_frame,
            "point_count": int(ip_keys.shape[0]),
            "strain_field": str(payload.get("strain_field", "unknown")),
            "B_label_strain_field": str(payload.get("B_label_strain_field", "unknown")),
            "strain_label_key": str(payload.get("strain_label_key", "")),
            "B_label_key": str(payload.get("B_label_key", "")),
            "has_B_LE128_forward": bool("B_LE128_forward" in payload),
            "training_ready_sobolev": bool("B_LE128_forward" in payload),
            "audit_ref_ip_xyz_vs_abaqus_coord_max_abs": float(payload.get("audit_ref_ip_xyz_vs_abaqus_coord_max_abs", np.nan)),
            "audit_detJ_vs_IVOL_max_abs": float(payload.get("audit_detJ_vs_IVOL_max_abs", np.nan)),
            "require_ip_audit": bool(payload.get("require_ip_audit", False)),
            "skip_ivol_audit": bool(payload.get("skip_ivol_audit", False)),
            "ip_audit_scope": str(payload.get("ip_audit_scope", "")),
            "merge_q48_max_abs_diff": float(payload.get("merge_q48_max_abs_diff", np.nan)),
            "merge_LE128_base_max_abs_diff": float(payload.get("merge_LE128_base_max_abs_diff", np.nan)),
            "merge_ip_keys_match": bool(payload.get("merge_ip_keys_match", False)),
            "merge_strain_field": str(payload.get("merge_strain_field", "unknown")),
            "merge_B_label_strain_field": str(payload.get("merge_B_label_strain_field", "unknown")),
            "merge_strain_field_match": bool(payload.get("merge_strain_field_match", False)),
            "allow_merge_mismatch": bool(payload.get("allow_merge_mismatch", False)),
            "point_feature_dim": int(point_features.shape[-1]),
            "point_feature_names": point_feature_names.tolist(),
        }
        sidecar.write_text(json.dumps(sidecar_report, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
        print(json.dumps(sidecar_report, sort_keys=True, default=json_default))
    finally:
        odb.close()


if __name__ == "__main__":
    main()
