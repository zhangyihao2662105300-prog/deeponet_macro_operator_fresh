#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Probe which integration-point geometry fields can be exported from an Abaqus ODB.

Run with Abaqus Python, for example:

    abaqus python scripts/probe_abaqus_ip_export_support.py --odb path/to/job.odb --out probe.json

The script intentionally checks two routes:
1. Direct ODB field outputs such as COORD, LE, IVOL.
2. Derived geometry from ODB mesh nodes plus element/integration-point labels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from odbAccess import openOdb  # type: ignore
from abaqusConstants import INTEGRATION_POINT  # type: ignore


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


def safe_attrs(value: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in (
        "instance",
        "elementLabel",
        "nodeLabel",
        "integrationPoint",
        "position",
        "sectionPoint",
        "type",
        "data",
        "localCoordSystem",
        "conjugateData",
    ):
        try:
            item = getattr(value, name)
        except Exception:
            continue
        if name == "instance":
            try:
                out[name] = str(item.name)
            except Exception:
                out[name] = str(item)
        elif name == "data":
            out[name] = np.asarray(item).reshape(-1).tolist()
        elif name == "localCoordSystem":
            if item is None:
                out[name] = None
            else:
                out[name] = np.asarray(item).tolist()
        else:
            out[name] = str(item)
    return out


def first_step_last_frame(odb: Any) -> tuple[str, Any]:
    step_name = list(odb.steps.keys())[-1]
    step = odb.steps[step_name]
    return step_name, step.frames[-1]


def field_values(field: Any, *, integration_point_only: bool = False) -> list[Any]:
    if integration_point_only:
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


def find_instance_with_elements(odb: Any) -> Any:
    for inst in odb.rootAssembly.instances.values():
        if len(inst.elements) > 0:
            return inst
    raise RuntimeError("ODB has no assembly instance with elements")


def element_connectivity_xyz(instance: Any, element_label: int) -> np.ndarray:
    node_by_label = {int(node.label): np.asarray(node.coordinates, dtype=np.float64) for node in instance.nodes}
    for elem in instance.elements:
        if int(elem.label) == int(element_label):
            return np.asarray([node_by_label[int(label)] for label in elem.connectivity], dtype=np.float64)
    raise KeyError("element label not found: %s" % element_label)


def derive_from_mesh(instance: Any, values: list[Any]) -> dict[str, Any]:
    rows = []
    max_coord_diff = 0.0
    detj_values = []
    element_types = sorted({str(elem.type) for elem in instance.elements})
    for value in values:
        if not has_element_ip(value):
            continue
        elem = int(value.elementLabel)
        ip = int(value.integrationPoint)
        if ip < 1 or ip > 8:
            continue
        xyz_nodes = element_connectivity_xyz(instance, elem)
        gp = GAUSS_POINTS[ip - 1]
        n, dndr = css8_shape(*gp)
        xyz = n.dot(xyz_nodes)
        jmat = dndr.T.dot(xyz_nodes)
        detj = float(np.linalg.det(jmat))
        invj = np.linalg.inv(jmat)
        detj_values.append(detj)
        data = np.asarray(value.data, dtype=np.float64).reshape(-1)
        if data.size >= 3:
            max_coord_diff = max(max_coord_diff, float(np.max(np.abs(data[:3] - xyz))))
        frame = None
        try:
            u, _s, vh = np.linalg.svd(jmat, full_matrices=True)
            frame = u.dot(vh)
        except Exception:
            frame = None
        if len(rows) < 8:
            rows.append(
                {
                    "elementLabel": elem,
                    "integrationPoint": ip,
                    "ip_xi_local": list(gp),
                    "ip_xyz_from_mesh": xyz,
                    "ip_J_from_mesh": jmat,
                    "ip_invJ_from_mesh": invj,
                    "ip_detJ_from_mesh": detj,
                    "ip_frame_from_mesh_polar": frame,
                }
            )
    return {
        "supported": bool(rows),
        "element_types": element_types,
        "derived_fields": ["ip_xi", "ip_xyz", "ip_J", "ip_invJ", "ip_detJ", "ip_frame_polar"],
        "max_abs_coord_field_vs_mesh": max_coord_diff,
        "detJ_min": float(np.min(detj_values)) if detj_values else None,
        "detJ_max": float(np.max(detj_values)) if detj_values else None,
        "sample_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odb", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    odb_path = Path(args.odb).resolve()
    out_path = Path(args.out).resolve()
    odb = openOdb(path=str(odb_path), readOnly=True)
    try:
        step_name, frame = first_step_last_frame(odb)
        field_keys = sorted(str(k) for k in frame.fieldOutputs.keys())
        report: dict[str, Any] = {
            "odb": str(odb_path),
            "step": step_name,
            "frameValue": float(frame.frameValue),
            "field_output_keys": field_keys,
            "direct_support": {},
            "derived_support": None,
        }
        for key in ("COORD", "LE", "E", "S", "IVOL", "EVOL", "J", "JAC", "DG", "ORIENT"):
            if key not in frame.fieldOutputs:
                report["direct_support"][key] = {"present": False}
                continue
            field = frame.fieldOutputs[key]
            values = field_values(field)
            ip_values = [v for v in field_values(field, integration_point_only=True) if has_element_ip(v)]
            sample = safe_attrs(values[0]) if values else {}
            ip_sample = safe_attrs(ip_values[0]) if ip_values else {}
            report["direct_support"][key] = {
                "present": True,
                "value_count": int(len(values)),
                "integration_point_value_count": int(len(ip_values)),
                "sample": sample,
                "integration_point_sample": ip_sample,
            }
        if "COORD" in frame.fieldOutputs:
            instance = find_instance_with_elements(odb)
            coord_ip_values = [v for v in field_values(frame.fieldOutputs["COORD"], integration_point_only=True) if has_element_ip(v)]
            if not coord_ip_values and "LE" in frame.fieldOutputs:
                coord_ip_values = [v for v in field_values(frame.fieldOutputs["LE"], integration_point_only=True) if has_element_ip(v)]
            report["derived_support"] = derive_from_mesh(instance, coord_ip_values)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
        print(json.dumps({"out": str(out_path), "field_output_keys": field_keys}, sort_keys=True))
    finally:
        odb.close()


if __name__ == "__main__":
    main()
