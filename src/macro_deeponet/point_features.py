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

from .true176_data import build_point_features_unique, standard_css8_row_map

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
        "or generation sidecars before using it as a physical data source."
    )
    return point_all[:, target_ips, :].astype(np.float32), {
        **meta,
        "point_feature_source": source_name,
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

    for compact_id, compact_text in enumerate(compact_paths):
        mask = source_index == int(compact_id)
        if not np.any(mask):
            continue
        rows = source_row[mask]
        path = Path(compact_text).resolve()
        with np.load(str(path), allow_pickle=True) as z:
            n_total = int(z["shape4"].shape[0])
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

            if include_id_features and not (feature_key is None and used.get("mode") == "shape4_generated_fallback"):
                ids, id_names = _id_features(target_ips, arr.shape[0])
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
            elif len(names) != len(cur_names):
                raise ValueError(f"feature dimension/name mismatch in {path}: expected {len(names)}, got {len(cur_names)}")
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
        "sources_used": sources_used,
        "missing_point_data_compacts": missing,
        "allow_shape4_fallback": bool(allow_shape4_fallback),
    }
    return point, meta
