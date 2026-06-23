# v2b One-Case Local-Strain Pilot

Date: 2026-06-23

## Purpose

This pilot extends the v2a q-coordinate pilot by adding an explicit local
strain coordinate at each integration point.  It still does not train a model.

The checked chain is:

```text
LE_Abaqus_global <-> LE_local
B_local_useful -> B_raw_hat
```

where:

```text
B_raw_hat = T_eps_to_abq @ B_local_useful @ T_q_raw_to_useful
```

This isolates the strain-coordinate transform before any v2 training or
standard-coordinate model is introduced.

## Input

Input v2a compact:

```text
D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\case050\complete_case050_v2a_q_useful_pilot.npz
```

Output v2b compact:

```text
D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\case050\complete_case050_v2b_local_strain_pilot.npz
```

Both files are generated artifacts under `outputs` and are not committed.

## Local Frame

The local orthonormal basis is built from `ip_J` using Gram-Schmidt.

Current exporter metadata says:

```text
ip_J_convention = rows are physical derivatives with respect to local natural coordinates r,s,t
```

Therefore this pilot uses:

```text
local_frame_ip_J_axis_convention = rows
```

For each integration point:

```text
g1, g2, g3 = rows of ip_J
e1 = normalize(g1)
e2 = normalize(g2 - dot(g2,e1) e1)
e3 = cross(e1,e2)
if dot(e3,g3) < 0:
    e2 = -e2
    e3 = cross(e1,e2)
Q = [e1 e2 e3]
```

`Q` columns are local basis vectors represented in Abaqus/global coordinates.

Frame audit:

| Metric | Value |
|---|---:|
| `local_frame_Q_shape` | `[128,3,3]` |
| `local_frame_orthonormal_max` | `2.220446049250313e-16` |
| `local_frame_det_min` | `0.9999999999999999` |
| `local_frame_det_max` | `1.0000000000000002` |
| `local_frame_orientation_min` | `0.9995424705243147` |
| `local_frame_orientation_max` | `0.9995425972270762` |

## Voigt Convention

The pilot uses tensor shear components, not engineering gamma:

```text
strain_voigt_order = LE11_LE22_LE33_LE12_LE13_LE23
strain_shear_convention = tensor_shear_not_engineering_gamma
```

The transform matrices are generated numerically from basis tensors:

```text
E_local = Q.T @ E_abq @ Q
E_abq   = Q @ E_local @ Q.T
```

For each basis vector in 6-component Voigt form, the script converts to a
symmetric tensor, applies the tensor transform, and converts back to Voigt.  No
hand-written component expansion is used.

## Strain Transform Audit

Shapes:

```text
T_eps_from_abq: [128,6,6]
T_eps_to_abq:   [128,6,6]
LE128_local:    [10,128,6]
```

Roundtrip metrics:

| Metric | Value |
|---|---:|
| `T_eps_roundtrip_local_rel` | `5.874317750795854e-15` |
| `T_eps_roundtrip_abq_rel` | `6.1175772104987605e-15` |
| `LE_local_roundtrip_rel` | `1.7341868077270364e-16` |
| `LE_local_roundtrip_max_abs` | `4.336808689942018e-19` |

These are numerical-zero roundtrip errors for the local tensor-component
transform.

## B Chain-Rule Audit

Inputs:

```text
T_q_raw_to_useful: [42,48]
B_useful_abq:      [10,128,6,42]
B_LE128_forward:   [10,128,6,48]
```

Construct:

```text
B_local_useful = T_eps_from_abq @ B_useful_abq
B_standard_useful = B_local_useful
B_raw_hat = T_eps_to_abq @ B_local_useful @ T_q_raw_to_useful
```

`B_standard_useful` is an audit-compatible field name.  Its coordinate is:

```text
B_standard_useful_coordinate = local_jacobian_frame
```

It is not yet a covariant standard-coordinate strain label.

Audit metrics:

| Metric | Value |
|---|---:|
| `B_local_chain_rule_projected_rel` | `4.573742799247327e-16` |
| `B_local_chain_rule_projected_max_abs` | `1.5631940186722204e-13` |
| `B_local_chain_rule_raw_rel` | `0.00014709789545713997` |
| `B_local_chain_rule_raw_max_abs` | `0.01915484967056802` |

The projected chain-rule error is numerical zero.  The raw-space error matches
the v2a rigid-direction residual, as expected.

## Strict v2 Audit

Command:

```powershell
py -3 scripts\audit_v2_coordinate_consistent_contract.py `
  --compact D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\case050\complete_case050_v2b_local_strain_pilot.npz `
  --out-root D:\IS-FEM\outputs\query_point_v2_coordinate_contract_audit\case050_v2b_local_strain_pilot `
  --strict-v2
```

Result:

```text
strict_v2_coordinate_pass = true
strict_v2_coordinate_pass_count = 1
failure_counts = {}
```

The strict manifest recognizes:

```text
q_useful: [10,42]
T_q_raw_to_useful: [42,48]
B_standard_useful: [10,128,6,42]
T_eps_to_abq: [128,6,6]
B_label_q_coordinate = q_useful
B_label_output_coordinate = local_jacobian_frame
ip_xi/ip_J/ip_invJ/ip_detJ: ok
```

## Boundaries

- No model training was performed.
- No v1.3/v2a model, loss, learning rate, epoch, split policy, or tag was
  changed.
- No old TRUE176 `LE/B` labels were used as v2 labels.
- The output compact remains under `D:\IS-FEM\outputs\...` and is not committed
  to git.

## Current Conclusion

The v2b local-strain pilot passed for `case050`.

The following coordinate links are now closed for one case:

```text
q48_raw -> q_useful
LE_Abaqus_global <-> LE_local_jacobian_frame
B_local_useful -> B_raw_hat
```

This supports moving next to a v2c tiny oracle or tiny training smoke, still on
one case, before building any formal v2 training pool.
