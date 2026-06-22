"""Generic integration-point feature loading.

The important rule is: training labels and trunk features must describe the same
physical integration point.  Therefore this module prefers point information
stored in the compact data itself instead of assuming that a shape4-generated
standard row map matches the Abaqus/real integration-point labels.

Accepted compact schemas, in priority order:

1. Prebuilt point features:

   point_features / ip_features / trunk_features: [N,128,F] or [128,F]

2. Raw point fields, which are concatenated into point_features:

   ip_xi       / xi128 / ip_natural_coords: [N,128,3] or [128,3]
   ip_xyz      / ip_coords / ip_coordinates: [N,128,3] or [128,3]
   ip_frame    / ip_frames: [N,128,3,3] or [128,3,3]
   ip_J        / ip_jacobian / J128 / jmat: [N,128,3,3] or [128,3,3]
   ip_invJ     / ip_inverse_jacobian / invJ128 / invj: [N,128,3,3] or [128,3,3]
   ip_detJ     / detJ128 / detj: [N,128] or [N,128,1] or [128] or [128,1]

The fallback shape4-generated features are still available, but only when the
caller explicitly asks for them.  That makes the general data contract safe by
default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .true176_data import (
    build_point_features_unique,
    canonical_scale_mode,
    length_scale_column,
    standard_css8_row_map,
    standard_ip_keys,
)

FEATURE_KEYS = ("point_features", "ip_features", "trunk_features")
FEATURE_NAME_KEYS = ("point_feature_names", "ip_feature_names", "trunk_feature_names")
FIELD_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ip_xi", ("ip_xi", "xi128", "ip_natural_coords", "natural_coords")),
    ("ip_xyz", ("ip_xyz", "ip_coords", "ip_coordinates", "integration_point_xyz", "gauss_xyz")),
    ("ip_frame", ("ip_frame", "ip_frames", "frames128")),
    ("ip_J", ("ip_J", "ip_jacobian", "J128", "jmat", "ip_jmat")),
    ("ip_invJ", ("ip_invJ", "ip_inverse_jacobian", "invJ128", "invj", "ip_invj")),
    ("ip_detJ", ("ip_detJ", "detJ128", "detj", "ip_detj")),
)
IP_KEY_CANDIDATES = ("ip_keys", "integration_point_keys", "point_keys", "abaqus_ip_keys")


def _find_key(z: np.lib.npyio.NpzFile, keys: Iterable[str]) -> str | None:
    files = set(z.files)
    for key in keys:
        if key in files:
            return key
    return None


def _decode_names(value: np.ndarray | Any) -> list[str]:
    arr = np.asarray(value)
    if arr.ndim == 0:
        text = str(arr.item())
        try:
            maybe = json.loads(text)
            if isinstance(maybe, list):
                return [str(v) for v in maybe]
        except Exception:
            return [text]
    return [str(v) for v in arr.reshape(-1).tolist()]


def _field_names(prefix: str, width: int) -> list[str]:
    if width == 1:
        return [prefix]
    if width == 3:
        return [f"{prefix}_{a}" for a in ("x", "y", "z")]
    if width == 9:
        return [f"{prefix}_{i}{j}" for i in range(3) for j in range(3)]
    return [f"{prefix}_{i}" for i in range(width)]


def _slice_point_array(arr: np.ndarray, *, rows: np.ndarray, target_ips: list[int], n_total: int, source_key: str) -> np.ndarray:
    vals = np.asarray(arr, dtype=np.float32)
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    ips = np.asarray(target_ips, dtype=np.int64).reshape(-1)

    if vals.ndim == 1 and vals.shape[0] == 128:
        vals = vals[:, None]
    if vals.ndim < 2:
        raise ValueError(f"{source_key}: expected at least point array rank 2, got shape {vals.shape}")

    if vals.shape[0] == int(n_total):
        picked = vals[rows]
    elif vals.shape[0] == 128:
        picked = np.broadcast_to(vals.reshape((1,) + vals.shape), (rows.size,) + vals.shape).copy()
    else:
        raise ValueError(
            f"{source_key}: first dimension must be frame count {n_total} or fixed point count 128; got {vals.shape}"
        )

    if picked.shape[1] != 128:
        raise ValueError(f"{source_key}: point dimension must be 128, got shape {picked.shape}")
    picked = picked[:, ips]
    if picked.ndim == 2:
        picked = picked[..., None]
    elif picked.ndim > 3:
        picked = picked.reshape(picked.shape[0], picked.shape[1], -1)
    return np.asarray(picked, dtype=np.float32)


def _slice_ip_keys(arr: np.ndarray, *, rows: np.ndarray, target_ips: list[int], n_total: int, source_key: str) -> np.ndarray:
    vals = np.asarray(arr)
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    ips = np.asarray(target_ips, dtype=np.int64).reshape(-1)
    if vals.ndim < 2:
        raise ValueError(f"{source_key}: expected ip key array with point dimension, got shape {vals.shape}")

    if vals.shape[0] == int(n_total):
        picked = vals[rows]
    elif vals.shape[0] == 128:
        picked = np.broadcast_to(vals.reshape((1,) + vals.shape), (rows.size,) + vals.shape).copy()
    else:
        raise ValueError(
            f"{source_key}: first dimension must be frame count {n_total} or fixed point count 128; got {vals.shape}"
        )
    if picked.shape[1] != 128:
        raise ValueError(f"{source_key}: point dimension must be 128, got shape {picked.shape}")
    return np.asarray(picked[:, ips])


def _ip_keys_have_standard_elem_labels(arr: np.ndarray) -> bool:
    vals = np.asarray(arr)
    if vals.ndim == 2 and vals.shape[0] == 128 and vals.shape[-1] >= 2:
        labels = vals[:, 0]
    elif vals.ndim == 3 and vals.shape[1] == 128 and vals.shape[-1] >= 2:
        labels = vals[:, :, 0].reshape(-1)
    else:
        return False
    unique_elem = np.unique(np.asarray(labels, dtype=np.int64))
    return bool(np.array_equal(unique_elem, np.arange(1, 17, dtype=np.int64)))


def _infer_frame_count(z: np.lib.npyio.NpzFile, fallback: int, source_key: str) -> int:
    if int(fallback) > 0:
        return int(fallback)
    for key in ("shape4", "q48_raw", "LE128_base", "le", "B_LE128_forward", "b"):
        if key in z.files:
            arr = np.asarray(z[key])
            if arr.ndim >= 1:
                return int(arr.shape[0])
    raise KeyError(f"{source_key}: cannot infer frame count")


def _id_features(target_ips: list[int], batch: int) -> tuple[np.ndarray, list[str]]:
    row = standard_css8_row_map()
    ips = np.asarray(target_ips, dtype=np.int64)
    sub = row[ips]
    elem = sub[:, 1]
    ip = sub[:, 2]
    ex = sub[:, 4]
    ey = sub[:, 5]
    elem_x = -1.0 + 2.0 * (ex + 0.5) / 4.0
    elem_y = -1.0 + 2.0 * (ey + 0.5) / 4.0
    feats = np.stack([(elem - 8.5) / 7.5, (ip - 4.5) / 3.5, elem_x, elem_y], axis=-1)
    return np.broadcast_to(feats.reshape(1, feats.shape[0], feats.shape[1]), (int(batch), feats.shape[0], feats.shape[1])).astype(np.float32), [
        "elem_index_norm",
        "ip_index_norm",
        "elem_x_center_id",
        "elem_y_center_id",
    ]


def _id_features_from_ip_keys(ip_keys: np.ndarray, *, standard_elem_labels: bool | None = None) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    keys = np.asarray(ip_keys, dtype=np.float32)
    if keys.ndim != 3 or keys.shape[-1] < 2:
        raise ValueError(f"ip_keys must have shape [N,P,>=2], got {keys.shape}")
    elem = keys[:, :, 0]
    ip = keys[:, :, 1]
    unique_elem = np.unique(elem.astype(np.int64))
    standard = bool(standard_elem_labels) if standard_elem_labels is not None else bool(
        np.array_equal(unique_elem, np.arange(1, 17, dtype=np.int64))
    )
    ip_norm = (ip - 4.5) / 3.5
    if standard:
        ex = np.mod(elem - 1.0, 4.0)
        ey = np.floor((elem - 1.0) / 4.0)
        elem_x = -1.0 + 2.0 * (ex + 0.5) / 4.0
        elem_y = -1.0 + 2.0 * (ey + 0.5) / 4.0
        feats = np.stack([(elem - 8.5) / 7.5, ip_norm, elem_x, elem_y], axis=-1)
        return feats.astype(np.float32), [
            "elem_index_norm",
            "ip_index_norm",
            "elem_x_center_id",
            "elem_y_center_id",
        ], {
            "id_feature_rule": "standard_true176_elem_label_4x4_spatial",
            "id_feature_guard_passed": True,
            "id_feature_element_labels": unique_elem.astype(np.int64).tolist(),
        }

    order = {int(label): i for i, label in enumerate(unique_elem.tolist())}
    rank = np.vectorize(lambda v: order[int(v)], otypes=[np.float32])(elem)
    denom = max(float(unique_elem.size - 1), 1.0)
    elem_rank_norm = -1.0 + 2.0 * rank / denom if unique_elem.size > 1 else np.zeros_like(rank, dtype=np.float32)
    feats = np.stack([elem_rank_norm, ip_norm], axis=-1)
    return feats.astype(np.float32), [
        "elem_rank_norm",
        "ip_index_norm",
    ], {
        "id_feature_rule": "nonstandard_elem_label_rank_only_no_4x4_spatial",
        "id_feature_guard_passed": False,
        "id_feature_element_labels": unique_elem.astype(np.int64).tolist(),
        "id_feature_warning": "element labels are not exactly 1..16; omitted 4x4 spatial ID features",
    }


def _feature_name_mismatch_message(path: Path, expected: list[str], got: list[str]) -> str:
    mismatch = next((i for i, (a, b) in enumerate(zip(expected, got)) if a != b), None)
    if mismatch is None and len(expected) != len(got):
        mismatch = min(len(expected), len(got))
    return (
        f"point feature names/order mismatch in {path}: expected {len(expected)} names, got {len(got)}. "
        f"first_mismatch_index={mismatch}; expected_first10={expected[:10]}; got_first10={got[:10]}"
    )


def _ip_keys_meta(ip_keys: np.ndarray | None) -> dict[str, Any]:
    if ip_keys is None:
        return {
            "point_feature_ip_keys": None,
            "point_feature_ip_keys_source": "not_provided",
            "point_feature_ip_keys_frame_count": 0,
            "point_feature_ip_keys_repeated_for_frames": False,
        }
    keys = np.asarray(ip_keys, dtype=np.int64)
    frame_count = int(keys.shape[0]) if keys.ndim >= 1 else 0
    if keys.ndim == 3 and frame_count > 0 and np.all(keys == keys[0:1]):
        return {
            "point_feature_ip_keys": keys[0].tolist(),
            "point_feature_ip_keys_source": "compact",
            "point_feature_ip_keys_shape": list(keys[0].shape),
            "point_feature_ip_keys_frame_count": frame_count,
            "point_feature_ip_keys_repeated_for_frames": True,
        }
    return {
        "point_feature_ip_keys": keys.tolist(),
        "point_feature_ip_keys_source": "compact",
        "point_feature_ip_keys_shape": list(keys.shape),
        "point_feature_ip_keys_frame_count": frame_count,
        "point_feature_ip_keys_repeated_for_frames": False,
    }


def _is_dimensionless_point_feature_name(name: str) -> bool:
    """Return True for point features that do not carry a physical length unit."""

    low = str(name).strip().lower()
    if not low:
        return False
    if "_hat" in low or low.endswith("hat"):
        return True
    if low in {
        "ip_xi",
        "xi128",
        "ip_natural_coords",
        "natural_coords",
        "local_r",
        "local_s",
        "local_t",
        "elem_index_norm",
        "ip_index_norm",
        "elem_x_center_id",
        "elem_y_center_id",
    }:
        return True
    if low.startswith(
        (
            "ip_xi_",
            "xi128_",
            "ip_natural_coords_",
            "natural_coords_",
            "xi_fine",
            "eta_fine",
            "zeta_fine",
            "local_",
            "sin_",
            "cos_",
            "ip_frame_",
            "ip_frames_",
            "frames128_",
            "frame_",
            "elem_index",
            "ip_index",
            "elem_x_",
            "elem_y_",
        )
    ):
        return True
    return False


def transform_point_features_for_scale(
    point: np.ndarray,
    feature_names: list[str],
    length_scale: np.ndarray,
    *,
    scale_mode: str,
    detj_scale_dim: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert physical point fields to dimensionless isoparametric features.

    This operates by feature name.  Raw-field names such as ``ip_xyz_*``,
    ``ip_J_*``, ``ip_invJ_*`` and ``ip_detJ`` are treated as physical fields.
    Existing ``*_hat`` features and natural coordinates are assumed already
    dimensionless.  In physical mode, anonymous prebuilt features are rejected
    when no scale-aware name is present, because the code cannot know which
    columns carry length units.
    """

    mode = canonical_scale_mode(scale_mode)
    arr = np.asarray(point, dtype=np.float32).copy()
    names = [str(v) for v in feature_names]
    if arr.shape[-1] != len(names):
        raise ValueError(f"point feature dimension {arr.shape[-1]} does not match feature_names length {len(names)}")
    h = length_scale_column(length_scale, arr.shape[0]).reshape(arr.shape[0], 1)
    if mode == "normalized":
        return arr, {
            "scale_mode": mode,
            "point_feature_scale_transform": "identity",
            "H_min": float(np.min(h)),
            "H_max": float(np.max(h)),
        }

    h_point = h.reshape(arr.shape[0], 1)
    transformed: list[dict[str, Any]] = []
    dimensionless: list[str] = []
    unrecognized: list[str] = []
    det_dim = int(detj_scale_dim)
    for i, name in enumerate(names):
        low = name.lower()
        if _is_dimensionless_point_feature_name(name):
            dimensionless.append(name)
            continue
        if low.startswith(("ip_xyz_", "ip_coords_", "ip_coordinates_", "integration_point_xyz_", "gauss_xyz_")):
            arr[:, :, i] = arr[:, :, i] / h_point
            transformed.append({"feature": name, "rule": "divide_by_H"})
        elif low.startswith(("ip_j_", "ip_jacobian_", "j128_", "jmat_", "ip_jmat_")):
            arr[:, :, i] = arr[:, :, i] / h_point
            transformed.append({"feature": name, "rule": "divide_by_H"})
        elif low.startswith(("ip_invj_", "ip_inverse_jacobian_", "invj128_", "invj_", "ip_invj_")):
            arr[:, :, i] = arr[:, :, i] * h_point
            transformed.append({"feature": name, "rule": "multiply_by_H"})
        elif low in {"ip_detj", "detj128", "detj", "ip_detj"} or low.startswith(("ip_detj_", "detj128_", "detj_")):
            arr[:, :, i] = arr[:, :, i] / (h_point ** det_dim)
            transformed.append({"feature": name, "rule": f"divide_by_H^{det_dim}"})
        elif low in {"log_abs_detj", "log_abs_ip_detj", "ip_log_abs_detj"}:
            arr[:, :, i] = arr[:, :, i] - float(det_dim) * np.log(h_point)
            transformed.append({"feature": name, "rule": f"subtract_{det_dim}_logH"})
        else:
            unrecognized.append(name)

    if unrecognized:
        sample = ", ".join(unrecognized[:8])
        raise ValueError(
            "scale_mode=physical found point feature names whose length units are unknown: "
            f"{sample}. "
            "For physical-size data, provide raw fields such as ip_xyz/ip_J/ip_invJ/ip_detJ, "
            "or provide point_feature_names with *_hat/dimensionless names if the prebuilt point_features "
            "are already nondimensionalized."
        )

    return arr.astype(np.float32), {
        "scale_mode": mode,
        "point_feature_scale_transform": "physical_to_dimensionless_by_feature_name",
        "detJ_scale_dim": det_dim,
        "transformed_feature_count": int(len(transformed)),
        "transformed_features": transformed[:64],
        "dimensionless_feature_count": int(len(dimensionless)),
        "unrecognized_feature_count": int(len(unrecognized)),
        "unrecognized_features": unrecognized[:64],
        "H_min": float(np.min(h)),
        "H_max": float(np.max(h)),
    }


def _shape4_audited_features(
    shape4: np.ndarray,
    *,
    target_ips: list[int],
    include_id_features: bool,
    source_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    point_all, meta = build_point_features_unique(shape4, include_id_features=bool(include_id_features))
    audit_note = (
        "Reconstructed from the TRUE176 shape4 CSS8 generation contract. "
        "This is valid only for compacts whose LE/B rows follow standard ip_keys "
        "[elem 1 IP1..IP8, ..., elem 16 IP1..IP8]. Run "
        "scripts/audit_true176_shape4_ip_contract.py on available sample NPZ files "
        "or generation sidecars before using it as a dimensionless TRUE176 point source."
    )
    return point_all[:, target_ips, :].astype(np.float32), {
        **meta,
        "point_feature_source": source_name,
        "point_feature_axis": "point_features[:, k, :] aligns with LE/B[:, k, ...] after target_ips selection",
        "point_feature_target_ips": [int(v) for v in target_ips],
        "point_feature_ip_keys": standard_ip_keys()[np.asarray(target_ips, dtype=np.int64)].astype(np.int64).tolist(),
        "point_feature_alignment": "shape4-reconstructed point_features[:, k, :] matches LE/B row target_ips[k] under the audited standard CSS8 Abaqus row order",
        "shape4_ip_contract": "standard_css8_elem_major_ip_major",
        "audit_required": True,
        "audit_note": audit_note,
    }


def load_point_features_from_compacts(
    *,
    compact_paths: list[str],
    source_index: np.ndarray,
    source_row: np.ndarray,
    shape4: np.ndarray,
    target_ips: list[int],
    source: str = "data",
    include_id_features: bool = True,
    allow_shape4_fallback: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return point/trunk features aligned with the loaded frame order.

    ``source='data'`` requires explicit point data in every compact.  ``source='auto'``
    uses explicit data when available and can fall back to shape4 only if
    ``allow_shape4_fallback`` is true.  ``source='shape4-audited'`` reconstructs
    the point geometry from the TRUE176 shape4 generation contract.  ``source='shape4'``
    remains a legacy compatibility route.
    """

    key = str(source).strip().lower()
    key = key.replace("_", "-")
    if key not in {"data", "auto", "shape4", "shape4-audited"}:
        raise ValueError("point feature source must be one of: data, auto, shape4-audited, shape4")

    target_ips = [int(v) for v in target_ips]
    source_index = np.asarray(source_index, dtype=np.int64).reshape(-1)
    source_row = np.asarray(source_row, dtype=np.int64).reshape(-1)
    shape4 = np.asarray(shape4, dtype=np.float32).reshape(-1, 4)

    if key == "shape4-audited":
        return _shape4_audited_features(
            shape4,
            target_ips=target_ips,
            include_id_features=bool(include_id_features),
            source_name="shape4_reconstructed_audited",
        )

    if key == "shape4":
        return _shape4_audited_features(
            shape4,
            target_ips=target_ips,
            include_id_features=bool(include_id_features),
            source_name="shape4_generated_legacy_explicit",
        )

    chunks: list[np.ndarray] = []
    names: list[str] | None = None
    sources_used: list[dict[str, Any]] = []
    missing: list[str] = []
    ip_key_chunks: list[np.ndarray] = []

    for compact_id, compact_text in enumerate(compact_paths):
        mask = source_index == int(compact_id)
        if not np.any(mask):
            continue
        rows = source_row[mask]
        path = Path(compact_text).resolve()
        with np.load(str(path), allow_pickle=True) as z:
            n_total = _infer_frame_count(z, 0, str(path))
            feature_key = _find_key(z, FEATURE_KEYS)
            if feature_key is not None:
                arr = _slice_point_array(z[feature_key], rows=rows, target_ips=target_ips, n_total=n_total, source_key=feature_key)
                name_key = _find_key(z, FEATURE_NAME_KEYS)
                cur_names = _decode_names(z[name_key]) if name_key is not None else _field_names(feature_key, arr.shape[-1])
                used = {"compact": str(path), "mode": "prebuilt", "key": feature_key, "feature_dim": int(arr.shape[-1])}
            else:
                fields: list[np.ndarray] = []
                cur_names = []
                used_keys: list[str] = []
                for prefix, candidates in FIELD_GROUPS:
                    found = _find_key(z, candidates)
                    if found is None:
                        continue
                    vals = _slice_point_array(z[found], rows=rows, target_ips=target_ips, n_total=n_total, source_key=found)
                    fields.append(vals)
                    cur_names.extend(_field_names(prefix, vals.shape[-1]))
                    used_keys.append(found)
                if not fields:
                    missing.append(str(path))
                    if key == "auto" and bool(allow_shape4_fallback):
                        # Fill this compact from shape4-generated features for backward compatibility only.
                        if "shape4" not in z.files:
                            raise ValueError(f"{path}: shape4 fallback requested but compact has no shape4 field")
                        local_shape = np.asarray(z["shape4"], dtype=np.float32)[rows].reshape(-1, 4)
                        point_local, meta_local = build_point_features_unique(local_shape, include_id_features=bool(include_id_features))
                        arr = point_local[:, target_ips, :].astype(np.float32)
                        cur_names = list(meta_local.get("feature_names", _field_names("shape4_generated", arr.shape[-1])))
                        used = {"compact": str(path), "mode": "shape4_generated_fallback", "feature_dim": int(arr.shape[-1])}
                    else:
                        continue
                else:
                    arr = np.concatenate(fields, axis=-1).astype(np.float32)
                    used = {"compact": str(path), "mode": "raw_fields", "keys": used_keys, "feature_dim": int(arr.shape[-1])}

            ip_key_arr = None
            ip_key = _find_key(z, IP_KEY_CANDIDATES)
            if ip_key is not None:
                standard_ip_key_labels = _ip_keys_have_standard_elem_labels(z[ip_key])
                ip_key_arr = _slice_ip_keys(z[ip_key], rows=rows, target_ips=target_ips, n_total=n_total, source_key=ip_key)
                ip_key_chunks.append(ip_key_arr)
                used["ip_key"] = ip_key

            if include_id_features and not (feature_key is None and used.get("mode") == "shape4_generated_fallback"):
                if ip_key_arr is not None:
                    ids, id_names, id_meta = _id_features_from_ip_keys(ip_key_arr, standard_elem_labels=standard_ip_key_labels)
                    used["id_feature_source"] = "ip_keys"
                    used.update(id_meta)
                else:
                    ids, id_names = _id_features(target_ips, arr.shape[0])
                    used["id_feature_source"] = "standard_target_ips"
                    used["id_feature_rule"] = "standard_target_ips_4x4_spatial"
                    used["id_feature_guard_passed"] = True
                arr = np.concatenate([arr, ids], axis=-1).astype(np.float32)
                cur_names = list(cur_names) + id_names
                used["appended_id_features"] = True

            if len(cur_names) != int(arr.shape[-1]):
                raise ValueError(
                    f"feature names for {path} have length {len(cur_names)}, "
                    f"but feature dimension is {arr.shape[-1]}"
                )

            if names is None:
                names = list(cur_names)
            elif names != list(cur_names):
                raise ValueError(_feature_name_mismatch_message(path, names, list(cur_names)))
            chunks.append(arr)
            sources_used.append(used)

    if missing and not (key == "auto" and bool(allow_shape4_fallback)):
        raise ValueError(
            "Missing explicit point features in one or more compact files: "
            + ", ".join(missing)
            + ". Add point_features/ip_features/trunk_features or raw fields such as ip_xi, ip_xyz, ip_J, ip_detJ."
        )

    if not chunks:
        message = (
            "No explicit point features were found in compact data. Add point_features/ip_features/trunk_features "
            "or raw fields such as ip_xi, ip_xyz, ip_J, ip_detJ. Do not rely on shape4-generated points unless "
            "you have audited that they match the Abaqus label coordinates."
        )
        if key == "auto" and bool(allow_shape4_fallback):
            point_all, meta = build_point_features_unique(shape4, include_id_features=bool(include_id_features))
            return point_all[:, target_ips, :].astype(np.float32), {**meta, "point_feature_source": "shape4_generated_global_fallback", "warning": message}
        raise ValueError(message)

    point = np.concatenate(chunks, axis=0).astype(np.float32)
    if point.shape[0] != shape4.shape[0]:
        raise ValueError(f"point feature frame count {point.shape[0]} does not match loaded frames {shape4.shape[0]}")
    ip_keys: np.ndarray | None = None
    if ip_key_chunks:
        if len(ip_key_chunks) != len(chunks):
            raise ValueError("ip_keys were present for only some point-feature chunks; provide them for every compact or none")
        ip_keys = np.concatenate(ip_key_chunks, axis=0)
        if ip_keys.shape[:2] != point.shape[:2]:
            raise ValueError(f"ip_keys {ip_keys.shape[:2]} do not align with point features {point.shape[:2]}")
    modes = {str(item.get("mode", "")) for item in sources_used}
    if modes == {"shape4_generated_fallback"}:
        point_feature_source = "shape4_generated_fallback"
    elif "shape4_generated_fallback" in modes:
        point_feature_source = "data_generic_with_shape4_fallback"
    else:
        point_feature_source = "data_generic"
    meta = {
        "point_feature_source": point_feature_source,
        "feature_names": names or _field_names("point_feature", point.shape[-1]),
        "feature_dim": int(point.shape[-1]),
        "point_count": int(point.shape[1]),
        "target_ips": target_ips,
        "point_feature_axis": "point_features[:, k, :] aligns with LE/B[:, k, ...] after target_ips selection",
        "point_feature_target_ips": [int(v) for v in target_ips],
        "point_feature_alignment": "data point_features were sliced from the same compact rows and target_ips order as LE/B labels",
        "sources_used": sources_used,
        "missing_point_data_compacts": missing,
        "allow_shape4_fallback": bool(allow_shape4_fallback),
    }
    meta.update(_ip_keys_meta(ip_keys))
    return point, meta
