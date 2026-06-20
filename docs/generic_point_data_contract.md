# Generic real integration-point data contract

The general DeepONet route must not assume that points generated from `shape4`
and a fixed CSS8 row map are identical to the physical Abaqus/real integration
points attached to `LE128_base` and `B_LE128_forward`.

The safe contract is:

```text
Branch input:
  shape4:  [N, 4]
  q48_raw: [N, 48]

Trunk input, stored in compact data:
  point_features: [N, 128, F]

Labels:
  LE128_base:       [N, 128, 6]
  B_LE128_forward:  [N, 128, 6, 48]
```

For every frame `n` and point row `i`, the row

```text
point_features[n, i, :]
LE128_base[n, i, :]
B_LE128_forward[n, i, :, :]
```

must describe the same physical integration point.

## Preferred fields

The preferred compact schema is to store `point_features` directly.  If that is
not available, the generic loader can build it from raw fields:

```text
ip_xi       or xi128                 [N,128,3] or [128,3]
ip_xyz      or ip_coords             [N,128,3] or [128,3]
ip_frame    or ip_frames             [N,128,3,3] or [128,3,3]
ip_J        or ip_jacobian / J128    [N,128,3,3] or [128,3,3]
ip_invJ     or invJ128               [N,128,3,3] or [128,3,3]
ip_detJ     or detJ128               [N,128] / [N,128,1] / [128] / [128,1]
```

The loader concatenates all available fields and optionally appends normalized
`elem/ip` id features.

## Unsafe legacy fallback

`shape4`-generated point features are now an explicit fallback only:

```bash
--point-feature-source shape4
```

or:

```bash
--point-feature-source auto --allow-shape4-point-feature-fallback
```

Use this only after a coordinate/order audit proves:

```text
x_code(shape4, row_map[i]) == x_data[i]
```

for all 128 label rows.

## Data synchronization

Use the sync script to rewrite compact files into the generic schema:

```bash
PYTHONPATH=src python3 scripts/sync_true176_generic_point_compacts.py \
  --compact-list /path/to/compact_paths.txt \
  --out-root /path/to/true176_shape4_qraw_128ip_generic_points \
  --include-id-features
```

The script refuses to synthesize point features from `shape4` by default.  If the
old shape4 route has been audited and you deliberately want a compatibility copy:

```bash
PYTHONPATH=src python3 scripts/sync_true176_generic_point_compacts.py \
  --compact-list /path/to/compact_paths.txt \
  --out-root /path/to/legacy_shape4_point_features \
  --include-id-features \
  --legacy-shape4-fallback
```

The second command is not a physical-data correction; it only packages the old
assumption into a `point_features` field.
