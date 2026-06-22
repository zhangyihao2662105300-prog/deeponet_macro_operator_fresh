#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the v1.2 pilot fresh-Abaqus chain for one q48 boundary-control case.

This script is intentionally narrow.  It consumes a fresh v1.2 q48 frame plan,
builds a matching shape4 Abaqus base job and 48 forward perturbation jobs, then
packs the fresh finite-difference B labels for the current complete-compact
exporter.

It does not train a model, does not use legacy LE/B labels, and does not move
or create git-tracked data.  Old TRUE176 information is used only through the
q48 direction plan and the audited shape4/T-boundary bridge.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_ID = "case041"
DEFAULT_Q48_CSV = Path(
    r"D:\IS-FEM\outputs\query_point_v1_2_data_coverage_audit\fresh_boundary_control_plan\case041_q48_frames.csv"
)
DEFAULT_BOUNDARY96_CSV = Path(
    r"D:\IS-FEM\outputs\query_point_v1_2_data_coverage_audit\fresh_abaqus_boundary_inputs\case041_boundary96_frames.csv"
)
DEFAULT_BRIDGE_CONTRACT = Path(
    r"D:\IS-FEM\t176_cyl100_pilot6_cases001_020_full48"
    r"\sample_893060_cyl_lam100_tau006_theta24_case001_single_Axial_Force_+1"
    r"\boundary_contract\boundary_contract_arrays.npz"
)
DEFAULT_OUT_ROOT = Path(r"D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case041")
DEFAULT_OLD_SRC_ROOT = Path(r"D:\IS-FEM\NNSE_css8_push_tmp")
DEFAULT_EXPORT_LIB = Path(r"D:\IS-FEM\SRCv3.1 - 1\SRCv3.1\scripts\export_three_element_bfk_arrays.py")
DEFAULT_ABAQUS = Path(r"D:\Program Files\SIMULIA\Commands\abaqus.bat")


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def read_q48_frames(path: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    rows: list[dict[str, str]] = []
    q_rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"case_id", "frame_index", "alpha", "q_norm"} | {f"q{i}" for i in range(1, 49)}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            rows.append(row)
            q_rows.append([float(row[f"q{i}"]) for i in range(1, 49)])
    q48 = np.asarray(q_rows, dtype=np.float64)
    if q48.shape != (10, 48):
        raise ValueError(f"{path}: pilot case expects q48 shape (10,48), got {q48.shape}")
    return rows, q48


def read_boundary96_frames(path: Path) -> np.ndarray:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"frame_index", "boundary_node_index", "ux", "uy", "uz"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows.extend(reader)
    by_frame: dict[int, list[tuple[int, list[float]]]] = {}
    for row in rows:
        frame = int(row["frame_index"])
        node_index = int(row["boundary_node_index"])
        by_frame.setdefault(frame, []).append(
            (node_index, [float(row["ux"]), float(row["uy"]), float(row["uz"])])
        )
    frames = []
    for frame in sorted(by_frame):
        items = sorted(by_frame[frame], key=lambda item: item[0])
        if len(items) != 32:
            raise ValueError(f"{path}: frame {frame} has {len(items)} boundary nodes, expected 32")
        frames.append([v for _node_i, vec in items for v in vec])
    arr = np.asarray(frames, dtype=np.float64)
    if arr.shape != (10, 96):
        raise ValueError(f"{path}: expected boundary96 shape (10,96), got {arr.shape}")
    return arr


def parse_shape4(text: str) -> np.ndarray | None:
    raw = str(text).strip()
    if not raw:
        return None
    vals = [float(v) for v in raw.replace(";", ",").split(",") if v.strip()]
    if len(vals) != 4:
        raise ValueError("--shape4 must contain four comma-separated values")
    return np.asarray(vals, dtype=np.float64)


def load_bridge(path: Path, shape4_override: np.ndarray | None) -> dict[str, Any]:
    with np.load(str(path), allow_pickle=True) as z:
        required = {"T_boundary_96x48", "boundary_nodes", "keep_nodes", "shape4"}
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"{path}: missing bridge fields {sorted(missing)}")
        T = np.asarray(z["T_boundary_96x48"], dtype=np.float64).reshape(96, 48)
        shape4 = np.asarray(z["shape4"], dtype=np.float64).reshape(4)
        if shape4_override is not None:
            shape4 = np.asarray(shape4_override, dtype=np.float64).reshape(4)
        return {
            "path": str(path),
            "T_boundary_96x48": T,
            "boundary_nodes": np.asarray(z["boundary_nodes"], dtype=np.int64).reshape(32),
            "keep_nodes": np.asarray(z["keep_nodes"], dtype=np.int64).reshape(16),
            "shape4": shape4,
            "shape4_keys": np.asarray(z["shape4_keys"]).astype(str).reshape(-1).tolist()
            if "shape4_keys" in z.files
            else ["lambda", "tau", "chi", "mu"],
        }


def assert_linear_frames(q48: np.ndarray, tol: float) -> dict[str, Any]:
    q_final = np.asarray(q48[-1], dtype=np.float64).reshape(48)
    final_norm = float(np.linalg.norm(q_final))
    if not math.isfinite(final_norm) or final_norm <= 0.0:
        raise ValueError("final q48 frame has zero/invalid norm")
    factors = []
    max_abs = 0.0
    for i in range(q48.shape[0]):
        factor = float(i + 1) / float(q48.shape[0])
        pred = factor * q_final
        diff = float(np.max(np.abs(pred - q48[i])))
        max_abs = max(max_abs, diff)
        factors.append(factor)
    if max_abs > float(tol):
        raise ValueError(f"q48 frames are not linear in the final frame: max_abs={max_abs:.9g}, tol={tol:.9g}")
    return {"q48_final_norm": final_norm, "alpha_factors": factors, "linear_q48_max_abs_diff": max_abs}


def load_old_modules(old_src_root: Path):
    scripts_dir = old_src_root.resolve() / "scripts"
    if not scripts_dir.exists():
        raise FileNotFoundError(f"old TRUE176 script directory not found: {scripts_dir}")
    sys.path.insert(0, str(scripts_dir))
    import run_css8_shape4_nonzero_smoke as smoke  # type: ignore
    import run_css8_128_tower11_allframes_parallel_generation as gen  # type: ignore

    return smoke, gen


def export_status_ok(row: dict[str, Any]) -> bool:
    return str(row.get("export", {}).get("status")) in {"OK", "SKIPPED_EXISTING_NPZ"}


def run_jobs(jobs: list[dict[str, Any]], *, gen: Any, args: SimpleNamespace, logs: Path, workers: int) -> list[dict[str, Any]]:
    if int(workers) <= 1:
        return [gen._run_export_job(args=args, logs=logs, attempt=1, **job) for job in jobs]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futs = [pool.submit(gen._run_export_job, args=args, logs=logs, attempt=1, **job) for job in jobs]
        for fut in as_completed(futs):
            rows.append(fut.result())
    return rows


def repack_b_compact(native_sample: Path, out_path: Path) -> dict[str, Any]:
    with np.load(str(native_sample), allow_pickle=True) as z:
        required = {"q48_raw", "LE128_base", "B_LE128_forward", "ip_keys", "shape4"}
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"{native_sample}: missing B compact fields {sorted(missing)}")
        payload: dict[str, Any] = {
            "q48_raw": np.asarray(z["q48_raw"], dtype=np.float64),
            "LE128_base": np.asarray(z["LE128_base"], dtype=np.float64),
            "B_LE128_forward": np.asarray(z["B_LE128_forward"], dtype=np.float64),
            "ip_keys": np.asarray(z["ip_keys"], dtype=np.int64),
            "shape4": np.asarray(z["shape4"], dtype=np.float64),
            "strain_field": np.asarray("LE", dtype=object),
            "B_label_strain_field": np.asarray("LE", dtype=object),
            "strain_label_key": np.asarray("LE128_base", dtype=object),
            "B_label_key": np.asarray("B_LE128_forward", dtype=object),
            "sample_paths": np.asarray([str(native_sample)], dtype=object),
        }
        for key in (
            "delta",
            "frame_indices",
            "frame_values",
            "perturb_directions",
            "is_full_48_direction_sample",
            "T_boundary_96x48",
            "X_keep_ref",
            "X_boundary_ref",
            "keep_nodes",
            "boundary_nodes",
            "q_boundary_raw",
            "q_boundary_clean",
            "LE128_plus",
            "DLE128_forward",
            "RF_projected",
            "RF_projected_plus",
            "F_DLE_IVOL",
            "elastic_D",
        ):
            if key in z.files:
                payload[key] = z[key]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **payload)
    sidecar = {
        "out": str(out_path),
        "native_sample": str(native_sample),
        "field_shapes": {key: list(np.asarray(value).shape) for key, value in payload.items()},
        "strain_field": "LE",
        "B_label_strain_field": "LE",
        "uses_legacy_LE_B_as_labels": False,
    }
    write_json(out_path.with_suffix(out_path.suffix + ".json"), sidecar)
    return sidecar


def run_subprocess(cmd: list[str], cwd: Path, log_path: Path, timeout_s: int) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=int(timeout_s) if timeout_s and timeout_s > 0 else None,
            check=False,
        )
    return {"cmd": cmd, "cwd": str(cwd), "log": str(log_path), "returncode": int(proc.returncode)}


def build_and_optionally_run(args: argparse.Namespace) -> dict[str, Any]:
    case_id = str(args.case_id)
    q48_csv = Path(args.q48_frames_csv).resolve()
    boundary96_csv = Path(args.boundary96_frames_csv).resolve()
    bridge_path = Path(args.bridge_contract).resolve()
    out_root = Path(args.out_root).resolve()
    input_dir = out_root / "input"
    run_root = out_root / "abaqus_run"
    base_dir = run_root / "base"
    perturb_root = run_root / "perturb"
    b_dir = out_root / "b_compact"
    complete_dir = out_root / "complete"
    audit_dir = out_root / "audit"
    logs = run_root / "pipeline_logs"
    for path in (input_dir, base_dir, perturb_root, b_dir, complete_dir, audit_dir, logs):
        path.mkdir(parents=True, exist_ok=True)
    (run_root / "boundary_contract").mkdir(parents=True, exist_ok=True)

    rows, q48 = read_q48_frames(q48_csv)
    linear = assert_linear_frames(q48, float(args.linear_tol))
    bridge = load_bridge(bridge_path, parse_shape4(str(args.shape4)))
    boundary96_expected = q48 @ np.asarray(bridge["T_boundary_96x48"], dtype=np.float64).T
    boundary96_observed = read_boundary96_frames(boundary96_csv)
    bridge_diff = float(np.max(np.abs(boundary96_expected - boundary96_observed)))
    if bridge_diff > float(args.bridge_tol):
        raise ValueError(f"boundary96 CSV does not match T @ q48: max_abs={bridge_diff:.9g}")

    q48_final = np.asarray(q48[-1], dtype=np.float64).reshape(48)
    shape4 = np.asarray(bridge["shape4"], dtype=np.float64).reshape(4)
    smoke, gen = load_old_modules(Path(args.old_src_root))

    shutil.copy2(q48_csv, input_dir / f"{case_id}_q48_frames.csv")
    shutil.copy2(boundary96_csv, input_dir / f"{case_id}_boundary96_frames.csv")

    base_job = f"fresh_{case_id}"
    base_inp = base_dir / f"{base_job}.inp"
    info = smoke.write_shape4_displacement_inp(
        base_inp,
        shape4,
        q48_final,
        increments=int(args.increments),
        perturb_direction=None,
        delta=float(args.delta),
        nlgeom=str(args.nlgeom),
    )
    generated_q48 = np.asarray(info["q48_frames"], dtype=np.float64)[1:]
    generated_qb = np.asarray(info["q_boundary_frames"], dtype=np.float64)[1:]
    generated_q48_diff = float(np.max(np.abs(generated_q48 - q48)))
    generated_qb_diff = float(np.max(np.abs(generated_qb - boundary96_observed)))
    if generated_q48_diff > float(args.linear_tol):
        raise ValueError(f"generated Abaqus q48 frames do not match input plan: {generated_q48_diff:.9g}")
    if generated_qb_diff > float(args.bridge_tol):
        raise ValueError(f"generated Abaqus boundary96 frames do not match input plan: {generated_qb_diff:.9g}")

    np.save(run_root / "q48_final.npy", q48_final)
    np.save(run_root / "shape4.npy", shape4)
    np.savez(
        str(run_root / "boundary_contract" / "boundary_contract_arrays.npz"),
        shape4=shape4,
        shape4_keys=np.asarray(bridge["shape4_keys"]),
        path_name=np.asarray([case_id]),
        T_boundary_96x48=np.asarray(info["T_boundary_96x48"], dtype=np.float64),
        X_keep_ref=np.asarray(info["X_keep_ref"], dtype=np.float64),
        X_boundary_ref=np.asarray(info["X_boundary_ref"], dtype=np.float64),
        q48_frames=np.asarray(info["q48_frames"], dtype=np.float64),
        q_boundary_frames=np.asarray(info["q_boundary_frames"], dtype=np.float64),
        frame_times=np.asarray(info["frame_times"], dtype=np.float64),
        keep_nodes=np.asarray(info["keep_nodes"], dtype=np.int64),
        boundary_nodes=np.asarray(info["boundary_nodes"], dtype=np.int64),
        nlgeom=np.asarray([str(args.nlgeom).strip().upper()]),
    )

    directions = list(range(48))
    old_run_args = SimpleNamespace(
        abaqus=str(Path(args.abaqus).resolve()),
        abaqus_memory=str(args.abaqus_memory),
        src_root=str(Path(args.old_src_root).resolve()),
        export_lib=str(Path(args.export_lib).resolve()),
        job_timeout_s=int(args.job_timeout_s),
        export_timeout_s=int(args.export_timeout_s),
        frame_start=1,
        frame_end=int(args.increments),
        candidate=str(args.candidate),
        allow_partial_frames=False,
        skip_existing=bool(args.skip_existing),
        min_common_frames=int(args.increments),
        delta=float(args.delta),
        elastic_e=float(args.elastic_e),
        nu=float(args.nu),
        nlgeom=str(args.nlgeom).strip().upper(),
    )

    jobs: list[dict[str, Any]] = []
    for direction in directions:
        job_name = f"{base_job}_d{direction:02d}_plus"
        job_dir = perturb_root / job_name
        job_dir.mkdir(parents=True, exist_ok=True)
        inp = job_dir / f"{job_name}.inp"
        smoke.write_shape4_displacement_inp(
            inp,
            shape4,
            q48_final,
            increments=int(args.increments),
            perturb_direction=int(direction),
            delta=float(args.delta),
            nlgeom=str(args.nlgeom),
        )
        jobs.append(
            {
                "direction": int(direction),
                "job_name": job_name,
                "inp": inp,
                "odb": job_dir / f"{job_name}.odb",
                "npz": job_dir / f"{job_name}.npz",
            }
        )

    manifest: dict[str, Any] = {
        "case_id": case_id,
        "stage": "prepared",
        "q48_frames_csv": str(q48_csv),
        "boundary96_frames_csv": str(boundary96_csv),
        "bridge_contract": str(bridge_path),
        "shape4": shape4.tolist(),
        "shape4_source": "bridge_contract" if not str(args.shape4).strip() else "override_arg",
        "increments": int(args.increments),
        "delta": float(args.delta),
        "nlgeom": str(args.nlgeom).strip().upper(),
        "base_inp": str(base_inp),
        "base_odb": str(base_dir / f"{base_job}.odb"),
        "base_export_npz": str(run_root / "base_frames.npz"),
        "perturb_job_count": len(jobs),
        "b_compact": str(b_dir / f"{case_id}_sobolev_B_compact.npz"),
        "complete_compact": str(complete_dir / f"complete_{case_id}_training_ready.npz"),
        "checks": {
            **linear,
            "boundary96_Tq_max_abs_diff": bridge_diff,
            "generated_q48_max_abs_diff": generated_q48_diff,
            "generated_boundary96_max_abs_diff": generated_qb_diff,
            "uses_legacy_LE_B_as_labels": False,
        },
        "run_abaqus": bool(args.run_abaqus),
        "run_complete_export": bool(args.run_complete_export),
        "run_strict_audit": bool(args.run_strict_audit),
    }
    write_json(out_root / "pilot_case041_plan.json", manifest)
    if not bool(args.run_abaqus):
        manifest["stage"] = "dry_run_prepared"
        write_json(out_root / "pilot_case041_plan.json", manifest)
        return manifest

    base_res = gen._run_export_job(
        job_name=base_job,
        inp=base_inp,
        odb=base_dir / f"{base_job}.odb",
        npz=run_root / "base_frames.npz",
        args=old_run_args,
        logs=logs,
        attempt=1,
    )
    if not export_status_ok(base_res):
        manifest["stage"] = "base_failed"
        manifest["base_result"] = base_res
        write_json(out_root / "pilot_case041_plan.json", manifest)
        return manifest

    plus_results = run_jobs(jobs, gen=gen, args=old_run_args, logs=logs, workers=int(args.inner_workers))
    failed = [row for row in plus_results if not export_status_ok(row)]
    manifest["base_result"] = base_res
    manifest["perturb_results"] = plus_results
    if failed:
        manifest["stage"] = "perturb_failed"
        manifest["failed_perturb_count"] = len(failed)
        write_json(out_root / "pilot_case041_plan.json", manifest)
        return manifest

    native = smoke.build_shape4_sample(run_root, case_id, old_run_args, [{**job, "npz": str(job["npz"])} for job in jobs])
    native_sample = Path(native["native_sample_npz"]).resolve()
    b_compact = b_dir / f"{case_id}_sobolev_B_compact.npz"
    b_sidecar = repack_b_compact(native_sample, b_compact)
    manifest["native_sample"] = native
    manifest["b_compact_sidecar"] = b_sidecar
    manifest["stage"] = "b_compact_generated"
    write_json(out_root / "pilot_case041_plan.json", manifest)

    if not bool(args.run_complete_export):
        return manifest

    complete = complete_dir / f"complete_{case_id}_training_ready.npz"
    exporter = Path(args.complete_exporter).resolve()
    shape4_arg = ",".join("%.17g" % float(v) for v in shape4)
    complete_cmd = [
        str(Path(args.abaqus).resolve()),
        "python",
        str(exporter),
        "--odb",
        str(base_dir / f"{base_job}.odb"),
        "--out",
        str(complete),
        "--frames",
        ",".join(str(i) for i in range(1, int(args.increments) + 1)),
        "--shape4",
        shape4_arg,
        "--merge-compact",
        str(b_compact),
        "--strain-field",
        "LE",
        "--require-b",
        "--require-merge-ip-keys",
        "--require-ip-audit",
        "--merge-q-tol",
        str(args.merge_q_tol),
        "--merge-le-tol",
        str(args.merge_le_tol),
    ]
    complete_res = run_subprocess(
        complete_cmd,
        REPO_ROOT,
        logs / f"{case_id}.complete_export.log",
        int(args.complete_export_timeout_s),
    )
    manifest["complete_export_result"] = complete_res
    manifest["complete_compact_exists"] = complete.exists()
    if complete_res["returncode"] != 0 or not complete.exists():
        manifest["stage"] = "complete_export_failed"
        write_json(out_root / "pilot_case041_plan.json", manifest)
        return manifest
    manifest["stage"] = "complete_compact_generated"
    write_json(out_root / "pilot_case041_plan.json", manifest)

    if bool(args.run_strict_audit):
        audit_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "audit_query_point_data_coverage.py"),
            "--compact",
            str(complete),
            "--out-root",
            str(audit_dir),
            "--strict-v1-2",
        ]
        audit_res = run_subprocess(audit_cmd, REPO_ROOT, logs / f"{case_id}.strict_audit.log", int(args.audit_timeout_s))
        manifest["strict_audit_result"] = audit_res
        summary = audit_dir / "audit_summary.json"
        if summary.exists():
            manifest["strict_audit_summary"] = json.loads(summary.read_text(encoding="utf-8"))
        manifest["stage"] = "strict_audit_complete" if audit_res["returncode"] == 0 else "strict_audit_failed"
        write_json(out_root / "pilot_case041_plan.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--q48-frames-csv", default=str(DEFAULT_Q48_CSV))
    parser.add_argument("--boundary96-frames-csv", default=str(DEFAULT_BOUNDARY96_CSV))
    parser.add_argument("--bridge-contract", default=str(DEFAULT_BRIDGE_CONTRACT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--old-src-root", default=str(DEFAULT_OLD_SRC_ROOT))
    parser.add_argument("--export-lib", default=str(DEFAULT_EXPORT_LIB))
    parser.add_argument("--complete-exporter", default=str(REPO_ROOT / "scripts" / "export_abaqus_true176_complete_compact.py"))
    parser.add_argument("--abaqus", default=str(DEFAULT_ABAQUS))
    parser.add_argument("--shape4", default="", help="Optional comma-separated shape4 override. Defaults to bridge contract shape4.")
    parser.add_argument("--increments", type=int, default=10)
    parser.add_argument("--delta", type=float, default=1.0e-6)
    parser.add_argument("--nlgeom", default="YES", choices=("YES", "NO", "yes", "no"))
    parser.add_argument("--candidate", default="css8_le6_engineering")
    parser.add_argument("--elastic-e", type=float, default=210000.0)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--abaqus-memory", default="512mb")
    parser.add_argument("--job-timeout-s", type=int, default=3600)
    parser.add_argument("--export-timeout-s", type=int, default=1200)
    parser.add_argument("--complete-export-timeout-s", type=int, default=1200)
    parser.add_argument("--audit-timeout-s", type=int, default=300)
    parser.add_argument("--inner-workers", type=int, default=2)
    parser.add_argument("--merge-q-tol", default="1e-8")
    parser.add_argument("--merge-le-tol", default="1e-8")
    parser.add_argument("--bridge-tol", type=float, default=1.0e-12)
    parser.add_argument("--linear-tol", type=float, default=1.0e-12)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--run-abaqus", action="store_true")
    parser.add_argument("--run-complete-export", action="store_true")
    parser.add_argument("--run-strict-audit", action="store_true")
    args = parser.parse_args()
    result = build_and_optionally_run(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))
    return 0 if str(result.get("stage")) not in {"base_failed", "perturb_failed", "complete_export_failed", "strict_audit_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
