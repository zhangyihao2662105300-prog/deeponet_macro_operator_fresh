# v2c One-Case Oracle / Tiny Training Smoke

Date: 2026-06-23

## Purpose

This smoke test checks whether the v2b coordinate contract can be read by a
network and differentiated through `q_useful`.

It is not formal training and it makes no generalization claim.  The only case
used is `case050`.

The checked chain is:

```text
q_useful + ip_xi -> LE_local_jacobian_frame
AD -> d(LE_local_jacobian_frame) / d(q_useful)
B_raw_hat_model = T_eps_to_abq @ AD_B_local_hat @ T_q_raw_to_useful
```

## Input Compact

Input:

```text
D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\case050\complete_case050_v2b_local_strain_pilot.npz
```

Important fields:

```text
q_useful:            [10,42]
LE128_local:         [10,128,6]
B_standard_useful:   [10,128,6,42]
T_q_raw_to_useful:   [42,48]
T_eps_to_abq:        [128,6,6]
B_LE128_forward:     [10,128,6,48]
ip_xi:               [128,3]
```

`B_standard_useful` is the v2b audit-compatible name.  Its actual coordinate is
`local_jacobian_frame`; this is not yet a full covariant standard-coordinate
strain formulation.

## Oracle Smoke

Command:

```powershell
py -3 scripts\audit_v2_one_case_oracle_smoke.py `
  --compact D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\case050\complete_case050_v2b_local_strain_pilot.npz `
  --out-root D:\IS-FEM\outputs\query_point_v2_tiny_smoke\case050_oracle
```

Output:

```text
D:\IS-FEM\outputs\query_point_v2_tiny_smoke\case050_oracle\oracle_summary.json
```

Key metrics:

| Metric | Value |
|---|---:|
| `LE_local_Bq_rel` | `0.04432110494813806` |
| `LE_local_Bq_cos` | `0.9990565149944766` |
| `LE_local_Bq_rms` | `0.000275044348489501` |
| `LE_local_target_rms` | `0.00027288740777478717` |
| `B_raw_hat_projected_rel` | `4.573742799247327e-16` |
| `B_raw_hat_raw_rel` | `0.00014709789545713997` |
| `B_rigid_residual_rel` | `0.00014709789545713607` |
| `zero_q_local_LE_rms_by_oracle` | `0.0` |

Interpretation:

- The chain-rule return to projected raw Abaqus B is numerical zero.
- The raw-space residual is the same rigid-direction residual seen in v2a/v2b.
- `LE_local_Bq_rel=0.0443` is a one-case linear B@q quality metric.  It is not a
  strict coordinate-contract failure because Abaqus `LE` can include nonlinear
  response over the finite load path.

## Tiny Training Smoke

Command:

```powershell
py -3 scripts\train_v2_one_case_tiny_smoke.py `
  --compact D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\case050\complete_case050_v2b_local_strain_pilot.npz `
  --out-root D:\IS-FEM\outputs\query_point_v2_tiny_smoke\case050_tiny_train `
  --steps 2000 `
  --seed 20260623
```

Output:

```text
D:\IS-FEM\outputs\query_point_v2_tiny_smoke\case050_tiny_train\tiny_train_summary.json
```

Script-local model:

```text
branch input: q_useful
trunk input:  ip_xi
output:       LE_local_jacobian_frame
AD target:    d(LE_local_jacobian_frame)/d(q_useful)
```

The model form is:

```text
LE_hat = B_prior_table(point) @ q_useful
       + R(q_useful, ip_xi)
       - R(0, ip_xi)
```

`B_prior_table(point)` is initialized from the one-case mean
`B_standard_useful`.  This is intentional for the smoke test and is not a formal
v2 architecture decision.

Key metrics:

| Metric | Value |
|---|---:|
| `best_step` | `1800` |
| `train_LE_local_rel` | `0.0070029981434345245` |
| `train_AD_B_local_rel` | `0.004019410815089941` |
| `train_AD_B_local_cos` | `0.9999919533729553` |
| `zero_q_LE_local_rms` | `0.0` |
| `B_model_raw_projected_rel` | `0.004044899716973305` |
| `B_model_raw_rel` | `0.004047574009746313` |

The initial metrics before training were:

| Metric | Value |
|---|---:|
| `train_LE_local_rel` | `0.13800807297229767` |
| `train_AD_B_local_rel` | `0.004735813941806555` |
| `B_model_raw_projected_rel` | `0.004784443881362677` |

## Current Conclusion

The v2c one-case chain is executable:

```text
q_useful + ip_xi
-> LE_local_jacobian_frame
-> AD_B_local_useful
-> B_raw_hat_model
```

The tiny model can overfit `case050` in local strain coordinates and its AD-B can
be mapped back to raw Abaqus B with low projected error.

This only proves a one-case training/AD smoke.  It does not prove multi-case v2
generalization, and it does not prove the final covariant standard-coordinate
strain formulation.

## Boundaries

- No formal model training was performed.
- No multi-case v2 pool was built.
- No v1.3/v2 training strategy, loss, learning rate, epoch policy, or tag was
  changed.
- No old TRUE176 `LE/B` labels were used as v2 labels.
- No output compact, checkpoint, `.npz`, `.odb`, `.pt`, or `.pth` artifact is
  committed.

## Next Step

The next step can be either:

```text
v2d one-case repeatability / normalization design
```

or:

```text
v2d multi-case v2 compact generation planning
```

Do not yet treat v2c as a formal training result.
