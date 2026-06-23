# v2a One-Case q-Useful Pilot

Date: 2026-06-23

## Purpose

This pilot tests only the first coordinate-consistency step of the v2 route:

```text
q48_raw -> q_useful = T_q_raw_to_useful @ q48_raw
```

It does not train a model and does not create standard/local strain labels.
The output strain coordinate is intentionally kept as Abaqus global `LE`:

```text
strain_output_coordinate = abaqus_global
T_eps_to_abq = I_6
```

This isolates the q-coordinate question before introducing local/standard
strain tensor transforms.

## Input Case

Pilot case:

```text
case050
```

Source compact:

```text
D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case050\complete\complete_case050_training_ready.npz
```

`case050` was selected because the v1.3 LOO diagnostic showed a strong
data-side `Btrue @ q` oracle while still exposing mixed-direction value
generalization limits.  That makes it a good q-coordinate chain-rule pilot.

## q48 Node Coordinate Source

The q48 control-node reference coordinates are read from:

```text
X_keep_ref: [16,3]
```

The compact also stores `keep_nodes`, `X_keep`, `X_macro`, and
`T_boundary_96x48`, but this pilot uses `X_keep_ref` because it is already in
the 16-node q48 order:

```text
q[3*i + dof] maps to X_keep_ref[i, dof]
```

No q48 node order is guessed.

## T_q Construction

The script builds the six linearized rigid-body modes from `X_keep_ref`:

```text
translation x/y/z
rotation x/y/z with u_i = omega cross (X_i - centroid)
```

Let:

```text
R: [48,6]
```

be those rigid modes.  The pilot computes the orthogonal complement:

```text
N: [48,42]
T_q_raw_to_useful = N.T
q_useful = q48_raw @ T_q_raw_to_useful.T
P_useful = N @ N.T
```

This is a linear small-rotation projection.  It does not use per-frame
Kabsch/Procrustes fitting, because v2 needs an explicit derivative:

```text
dq_useful/dq48_raw = T_q_raw_to_useful
```

## q-Coordinate Audit

Output compact:

```text
D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\case050\complete_case050_v2a_q_useful_pilot.npz
```

Summary:

| Metric | Value |
|---|---:|
| `q48_shape` | `[10,48]` |
| `q_useful_shape` | `[10,42]` |
| `T_q_raw_to_useful_shape` | `[42,48]` |
| `rigid_mode_rank` | `6` |
| `rigid_annihilation_max` | `1.371671448566352e-16` |
| `useful_basis_orthonormal_max` | `4.440892098500626e-16` |
| `q_useful_reconstruction_rel` | `0.33576244788862675` |
| `q_useful_removed_rigid_rel` | `0.33576244788862675` |

`q_useful_reconstruction_rel` here is the fraction of raw q removed by the
useful projection:

```text
||q48_raw @ P_useful - q48_raw|| / ||q48_raw||
```

For `case050`, about 33.6 percent of the raw q vector lies in the removed
rigid-body subspace.  This is exactly the kind of signal the v2 q-coordinate
pilot is meant to expose.

## B Chain-Rule Audit

This v2a pilot keeps Abaqus global strain.  Therefore:

```text
T_eps_to_abq = I_6
B_raw = B_LE128_forward = d(LE_Abaqus)/d(q48_raw)
B_useful_abq = B_raw @ N
B_raw_hat = B_useful_abq @ T_q_raw_to_useful
```

Shapes:

```text
B_raw:        [10,128,6,48]
B_useful_abq: [10,128,6,42]
B_raw_hat:    [10,128,6,48]
```

Audit metrics:

| Metric | Value |
|---|---:|
| `B_chain_rule_projected_rel` | `4.100651493362974e-16` |
| `B_chain_rule_projected_max_abs` | `1.4210854715202004e-13` |
| `B_chain_rule_raw_rel` | `0.0001470978954571399` |
| `B_chain_rule_raw_max_abs` | `0.019154849670568908` |
| `B_rigid_residual_rel` | `0.00014709789545713607` |

Interpretation:

- `B_chain_rule_projected_rel` is near numerical zero, so the projected
  chain-rule implementation is correct.
- `B_chain_rule_raw_rel` and `B_rigid_residual_rel` are both about `1.47e-4`,
  meaning the raw Abaqus B label has only a tiny response in removed rigid
  directions.
- The q projection removes a meaningful raw-q rigid component, but it does not
  remove meaningful B/strain sensitivity.

## Strict v2 Audit

Command:

```powershell
py -3 scripts\audit_v2_coordinate_consistent_contract.py `
  --compact D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\case050\complete_case050_v2a_q_useful_pilot.npz `
  --out-root D:\IS-FEM\outputs\query_point_v2_coordinate_contract_audit\case050_v2a_q_useful_pilot `
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
B_useful_abq: [10,128,6,42]
T_eps_to_abq: [6,6]
ip_xi/ip_J/ip_invJ/ip_detJ: ok
```

## Boundaries

- No model training was performed.
- No v1.3 model, loss, learning rate, epoch, split policy, or tag was changed.
- No old TRUE176 `LE/B` labels were used as v2 labels.
- No standard/local strain labels were generated in this pilot.
- The generated pilot `.npz` remains under `D:\IS-FEM\outputs\...` and is not
  committed to git.

## Current Conclusion

The v2a q-coordinate pilot passed for `case050`.

This establishes the first v2 coordinate-consistency link:

```text
q48_raw -> q_useful -> B_useful_abq -> B_raw_hat
```

with an explicit linear `T_q_raw_to_useful` and a strict audit pass.

The next step is a v2b strain-coordinate pilot:

```text
abaqus_global LE -> local/standard strain coordinate
T_eps_to_abq / T_eps_from_abq
B_raw_hat = T_eps_to_abq @ B_standard_useful @ T_q_raw_to_useful
```

That should be done as a separate one-case audit before any v2 training.
