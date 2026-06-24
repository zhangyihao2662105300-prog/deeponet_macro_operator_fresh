# v2 Standard Operator Preprocessing Contract

Date: 2026-06-24

## Purpose

This step pauses v2 model tuning and extracts the coordinate preprocessing and
postprocessing rules from the neural-network route.

The result is a standard-operator data contract.  The network should not learn
rigid-motion cleanup, global/local strain conversion, or raw/useful B
projection.  Those transforms are explicit preprocessing and postprocessing
modules.

This is an engineering/data-contract audit, not a model-performance result.

## Raw Abaqus Space

The raw fresh-Abaqus compact side remains:

```text
q48_raw
LE_abq
B_abq = B_LE128_forward = dLE_abq / dq48_raw
```

`q48_raw` is the Abaqus-global 48-DOF boundary displacement coordinate.
`LE_abq` is the Abaqus-global strain tensor component label.  `B_abq` is the
raw Sobolev label with respect to `q48_raw`.

Old TRUE176 `LE/B` labels are not used as v2 labels.

## Preprocessing

The standard-operator compact is built from the raw compact by:

```text
q_useful = T_q_raw_to_useful @ q48_raw
LE_local = T_eps_from_abq @ LE_abq
B_local_useful = T_eps_from_abq @ B_abq @ T_q_raw_to_useful.T
```

With row-major arrays, the implementation uses:

```text
q_useful[n] = q48_raw[n] @ T_q_raw_to_useful.T
```

The current `T_q_raw_to_useful` removes the six rigid modes and maps the raw
48-DOF boundary displacement into a 42-DOF useful coordinate.

## Standard Operator

The neural network is only allowed to learn:

```text
LE_local = NN(q_useful, ip_xi)
```

The model-side derivative is:

```text
B_local_useful_hat = dLE_local_hat / dq_useful
```

This derivative is local-strain / useful-q.  It must not be directly compared
with raw Abaqus `B_LE128_forward`.

## Postprocessing

Model predictions return to Abaqus coordinates by:

```text
LE_abq_hat = T_eps_to_abq @ LE_local_hat
B_raw_hat = T_eps_to_abq @ B_local_useful_hat @ T_q_raw_to_useful
```

Only the postprocessed/projected raw-space quantity may be compared with the
raw Abaqus B label.

## Standard Compact Fields

Model-visible fields:

```text
q_useful: [N,42]
ip_xi: [128,3]
LE_local: [N,128,6]
B_local_useful: [N,128,6,42]
```

Audit/postprocess-only fields:

```text
q48_raw: [N,48]
LE_abq: [N,128,6]
B_LE128_forward: [N,128,6,48]
T_q_raw_to_useful: [42,48]
T_eps_to_abq: [128,6,6]
T_eps_from_abq: [128,6,6]
local_frame_Q: [128,3,3] when present
case_id
metadata
```

The contract metadata includes:

```text
standard_operator_contract_version = v2-standard-operator-001
model_input_coordinate = q_useful
model_point_coordinate = ip_xi_standard_or_parent
model_output_coordinate = local_jacobian_frame
model_output_quantity = LE_local
model_derivative = dLE_local/dq_useful
raw_q_coordinate = q48_raw_abaqus_global
raw_B_coordinate = dLE_abq_global/dq48_raw
network_must_not_learn_transforms = true
```

## Scripts

Build standard-operator compacts:

```powershell
py -3 scripts\build_v2_standard_operator_compacts.py `
  --compact-list D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v2_standard_operator\multi_case_min10 `
  --strict
```

Audit one standard compact:

```powershell
py -3 scripts\audit_v2_standard_operator_contract.py `
  --standard-compact D:\IS-FEM\outputs\query_point_v2_standard_operator\multi_case_min10\case050\case050_standard_operator.npz `
  --out-root D:\IS-FEM\outputs\query_point_v2_standard_operator_audit\multi_case_min10\case050 `
  --strict
```

## 10-Case Audit Result

Input cases:

```text
case019, case025, case031, case041, case043,
case044, case045, case046, case049, case050
```

Outputs:

```text
D:\IS-FEM\outputs\query_point_v2_standard_operator\multi_case_min10\standard_operator_compact_list.txt
D:\IS-FEM\outputs\query_point_v2_standard_operator\multi_case_min10\standard_operator_summary.json
```

Strict result:

```text
compact_count = 10
strict_pass_count = 10
strict_fail_count = 0
```

Metric ranges:

```text
q_useful_transform_rel:
  min = 0.0
  max = 0.0

LE_local_to_abq_roundtrip_rel:
  min = 1.6410479539884723e-16
  max = 3.0740682511657065e-16

B_raw_projected_rel:
  min = 4.462975556507417e-16
  max = 4.652329014360367e-16

B_rigid_residual_rel:
  min = 0.00010968263851608675
  max = 0.0012638931647260517
```

The rigid residual is recorded but is not a strict failure, because the useful
coordinate intentionally removes rigid modes.

## Boundaries

This step did not train a model.

This step did not change loss functions, learning rates, epochs, splits, tags,
or v2g/v2j/v2k conclusions.

This step did not use old TRUE176 `LE/B` as v2 labels.

Generated `.npz` standard compacts remain under `D:\IS-FEM\outputs\...` and
must not be committed.

## Next Step

Future v2 training scripts should read only model-visible fields:

```text
q_useful, ip_xi, LE_local, B_local_useful
```

The transform fields should be used only for preprocessing, postprocessing,
evaluation, and audit:

```text
T_q_raw_to_useful, T_eps_to_abq, T_eps_from_abq
```

The next modeling route should therefore be described as:

```text
(q_useful, ip_xi) -> LE_local
B_local_useful = autograd
postprocess to LE_abq and B_raw for evaluation
```
