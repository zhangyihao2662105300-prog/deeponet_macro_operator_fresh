# v2d Multi-Case v2 Compact Generation Plan

Date: 2026-06-23

## Purpose

v2c proved that one `case050` v2 compact can be read by a network, trained in a
tiny smoke, differentiated with respect to `q_useful`, and mapped back to raw
Abaqus B.

v2d does not train a model.  Its purpose is to extend the v2 coordinate compact
contract from one case to the current 10-case strict fresh pool.

The target chain for every case is:

```text
q48_raw -> q_useful
LE_Abaqus_global <-> LE_local_jacobian_frame
B_local_useful -> B_raw_hat
strict_v2_coordinate_pass = true
```

## Why Compact Generation Before Training

The next risk is not whether the one-case tiny model can overfit.  That already
worked in v2c.  The next risk is whether multiple fresh cases can be converted
with the same coordinate contract without silent q48 node-order, rigid-mode,
local-frame, or B-chain-rule drift.

Therefore v2d first builds and audits a multi-case v2 compact pool.  Formal v2
multi-case training should only happen after this pool passes strict-v2 audit.

## Input Pool

Source manifest:

```text
D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt
```

Cases:

```text
[19, 25, 31, 41, 43, 44, 45, 46, 49, 50]
```

The builder requires every source compact to contain:

```text
q48_raw
LE128_base
B_LE128_forward
ip_J
ip_xi
ip_invJ
ip_detJ
X_keep_ref
```

`X_keep_ref` is mandatory.  If it is absent, the case fails; q48 node order is
not guessed.

## Conversion Flow

For each source compact:

```text
1. Read q48_raw, LE128_base, B_LE128_forward, ip geometry fields, X_keep_ref.
2. Build rigid modes from X_keep_ref.
3. Build T_q_raw_to_useful [42,48].
4. Build q_useful [N,42].
5. Build B_useful_abq [N,128,6,42].
6. Build local_frame_Q from ip_J by Gram-Schmidt.
7. Build T_eps_from_abq / T_eps_to_abq [128,6,6].
8. Build LE128_local [N,128,6].
9. Build B_local_useful / B_standard_useful [N,128,6,42].
10. Run chain-rule audits.
11. Save v2b compact.
12. Run strict-v2 coordinate audit.
```

Generated output root:

```text
D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10
```

Strict audit output root:

```text
D:\IS-FEM\outputs\query_point_v2_coordinate_contract_audit\multi_case_v2b_min10
```

Generated list:

```text
D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt
```

Generated summary:

```text
D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_multi_case_summary.json
```

These are generated artifacts under `outputs` and are not committed.

## Commands

Dry-run:

```powershell
py -3 scripts\build_v2_multi_case_compacts.py `
  --compact-list D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10 `
  --audit-root D:\IS-FEM\outputs\query_point_v2_coordinate_contract_audit\multi_case_v2b_min10 `
  --strict `
  --dry-run
```

Generation:

```powershell
py -3 scripts\build_v2_multi_case_compacts.py `
  --compact-list D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10 `
  --audit-root D:\IS-FEM\outputs\query_point_v2_coordinate_contract_audit\multi_case_v2b_min10 `
  --strict
```

Markdown summary:

```powershell
py -3 scripts\summarize_v2_multi_case_contract.py `
  --summary D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_multi_case_summary.json
```

## Dry-Run Result

Dry-run passed:

```text
compact_count = 10
fail_count = 0
```

All 10 source compacts had the required fields, including `X_keep_ref`.

## Generation Result

Actual generation passed:

```text
compact_count = 10
pass_count = 10
fail_count = 0
```

All generated v2b compacts passed strict-v2 coordinate audit.

## Key Metric Ranges

| Metric | Min | Max |
|---|---:|---:|
| `q_useful_removed_rigid_rel` | `0.18971368414698137` | `0.7451366595918001` |
| `B_rigid_residual_rel` | `0.0001096826385160804` | `0.0012638931647260521` |
| `B_local_chain_rule_projected_rel` | `4.449125862533697e-16` | `4.619812329621079e-16` |
| `LE_local_roundtrip_rel` | `1.6410479539884723e-16` | `3.0740682511657065e-16` |

The larger `q_useful_removed_rigid_rel` values mean some fresh q48 samples
contain substantial rigid-mode content.  That is not a strict-v2 failure, but it
should be tracked in later v2 training interpretation.

## Per-Case Strict Audit Table

| case_id | strict_pass | q_removed_rel | B_rigid_residual_rel | B_local_chain_rule_projected_rel | B_local_chain_rule_raw_rel | LE_local_roundtrip_rel | local_frame_orthonormal_max |
|---|---|---:|---:|---:|---:|---:|---:|
| 19 | true | `0.193768` | `0.000154969` | `4.58871e-16` | `0.000154969` | `3.07407e-16` | `2.22045e-16` |
| 25 | true | `0.287773` | `0.000109683` | `4.61667e-16` | `0.000109683` | `2.91808e-16` | `2.22045e-16` |
| 31 | true | `0.189714` | `0.00126389` | `4.53994e-16` | `0.00126389` | `2.84021e-16` | `2.22045e-16` |
| 41 | true | `0.745137` | `0.000147499` | `4.488e-16` | `0.000147499` | `1.64105e-16` | `2.22045e-16` |
| 43 | true | `0.559347` | `0.000149533` | `4.51321e-16` | `0.000149533` | `1.84844e-16` | `2.22045e-16` |
| 44 | true | `0.360945` | `0.000147438` | `4.44913e-16` | `0.000147438` | `1.73454e-16` | `2.22045e-16` |
| 45 | true | `0.44903` | `0.000148166` | `4.61981e-16` | `0.000148166` | `1.77733e-16` | `2.22045e-16` |
| 46 | true | `0.602436` | `0.000147385` | `4.53719e-16` | `0.000147385` | `1.75553e-16` | `2.22045e-16` |
| 49 | true | `0.365956` | `0.000147042` | `4.47169e-16` | `0.000147042` | `1.76028e-16` | `2.22045e-16` |
| 50 | true | `0.335762` | `0.000147098` | `4.57374e-16` | `0.000147098` | `1.73419e-16` | `2.22045e-16` |

## Readiness

The 10-case v2b compact pool is ready for a future v2 formal tiny multi-case
training smoke.

This readiness means only:

```text
multi-case v2 compact generation and strict coordinate audit are complete.
```

It does not mean:

```text
v2 model generalization is proven.
```

## Boundaries

- No model training was performed.
- No formal model, loss, learning-rate, epoch, or split policy was changed.
- No tag was moved.
- No old TRUE176 `LE/B` labels were used as v2 labels.
- No `.npz`, `.odb`, `.pt`, `.pth`, checkpoint, or generated output artifact is
  committed.
