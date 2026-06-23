# v2e Multi-Case Tiny Training Smoke

Date: 2026-06-23

## Purpose

v2e checks whether the 10-case v2b compact pool can be loaded, split by case,
trained in local strain coordinates, differentiated with respect to
`q_useful`, and mapped back to raw Abaqus B.

This is a tiny smoke, not formal training.  It does not tune a production model
and does not prove v2 generalization.

## Input

v2b compact list:

```text
D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt
```

The list contains 10 strict-v2 compacts:

```text
[19, 25, 31, 41, 43, 44, 45, 46, 49, 50]
```

Each compact provides:

```text
q_useful:           [10,42]
LE128_local:        [10,128,6]
B_standard_useful:  [10,128,6,42]
ip_xi:              [128,3]
T_q_raw_to_useful:  [42,48]
T_eps_to_abq:       [128,6,6]
B_LE128_forward:    [10,128,6,48]
```

## Split

The smoke uses a non-overlapping case split:

```text
train_cases = [19,25,31,41,43,45,49,50]
val_cases   = [44,46]
validation_is_overlapping = false
```

This mirrors the v1.3 split_B-style held-out cases for easier comparison, but
the coordinate system and labels are v2b local-frame labels.

## Tiny Model

The script-local model does not use `case_id` as an input.

Coordinates:

```text
branch input = q_useful
trunk input  = ip_xi
output       = LE_local_jacobian_frame
AD target    = d(LE_local_jacobian_frame) / d(q_useful)
```

Model form:

```text
LE_hat = B_prior_table(point) @ q_useful
       + R(q_useful, ip_xi)
       - R(0, ip_xi)
```

`B_prior_table(point)` is initialized from the train-case mean
`B_standard_useful`.  This is a smoke-test choice, not a formal v2 architecture
decision.

Raw B back-projection:

```text
B_raw_hat_model = T_eps_to_abq @ AD_B_local_hat @ T_q_raw_to_useful
```

## Command

```powershell
py -3 scripts\train_v2_multi_case_tiny_smoke.py `
  --compact-list D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v2_tiny_smoke\multi_case_min10_smoke `
  --val-cases 44,46 `
  --steps 3000 `
  --seed 20260623
```

Outputs:

```text
D:\IS-FEM\outputs\query_point_v2_tiny_smoke\multi_case_min10_smoke\training_summary.json
D:\IS-FEM\outputs\query_point_v2_tiny_smoke\multi_case_min10_smoke\metrics_history.csv
D:\IS-FEM\outputs\query_point_v2_tiny_smoke\multi_case_min10_smoke\per_case_attribution.csv
D:\IS-FEM\outputs\query_point_v2_tiny_smoke\multi_case_min10_smoke\best_metrics.json
```

These generated outputs are not committed.

## Best Metrics

The best checkpoint by smoke selection score occurred at initialization:

```text
best_step = 0
```

Best metrics:

| Metric | Value |
|---|---:|
| `train_LE_local_rel` | `29.468563079833984` |
| `val_LE_local_rel` | `10.836355209350586` |
| `train_AD_B_local_rel` | `0.29527074098587036` |
| `val_AD_B_local_rel` | `0.1957888901233673` |
| `train_AD_B_local_cos` | `0.9554136395454407` |
| `val_AD_B_local_cos` | `0.9939731955528259` |
| `train_zero_q_LE_local_rms` | `0.0` |
| `val_zero_q_LE_local_rms` | `0.0` |
| `train_B_model_raw_projected_rel` | `0.2953280210494995` |
| `val_B_model_raw_projected_rel` | `0.1954159140586853` |
| `train_B_model_raw_rel` | `0.29532819986343384` |
| `val_B_model_raw_rel` | `0.19541595876216888` |

## Latest Metrics

Latest step:

```text
latest_step = 3000
```

Latest metrics:

| Metric | Value |
|---|---:|
| `train_LE_local_rel` | `1.1830968856811523` |
| `val_LE_local_rel` | `18.03921890258789` |
| `train_AD_B_local_rel` | `0.3041848838329315` |
| `val_AD_B_local_rel` | `0.2089325487613678` |
| `train_AD_B_local_cos` | `0.9526196718215942` |
| `val_AD_B_local_cos` | `0.9906823635101318` |
| `train_zero_q_LE_local_rms` | `0.0` |
| `val_zero_q_LE_local_rms` | `0.0` |
| `train_B_model_raw_projected_rel` | `0.30487608909606934` |
| `val_B_model_raw_projected_rel` | `0.2097579836845398` |
| `train_B_model_raw_rel` | `0.30487629771232605` |
| `val_B_model_raw_rel` | `0.209757998585701` |

## Per-Case Attribution

Per-case attribution is from the latest model.

| split | case_id | LE_local_rel | AD_B_local_rel | AD_B_local_cos | B_model_raw_projected_rel | B_model_raw_rel |
|---|---:|---:|---:|---:|---:|---:|
| train | 19 | `4.305357456207275` | `0.5448414087295532` | `0.951741635799408` | `0.5420086979866028` | `0.5420086979866028` |
| train | 25 | `11.2684326171875` | `0.545368492603302` | `0.9516934752464294` | `0.5424136519432068` | `0.5424136519432068` |
| train | 31 | `0.5123006105422974` | `0.6126246452331543` | `0.9229050874710083` | `0.6074854135513306` | `0.6074862480163574` |
| train | 41 | `18.292329788208008` | `0.20891904830932617` | `0.9906867146492004` | `0.20973972976207733` | `0.2097397893667221` |
| train | 43 | `30.633352279663086` | `0.20926477015018463` | `0.9906968474388123` | `0.20994360744953156` | `0.20994366705417633` |
| train | 45 | `13.374227523803711` | `0.20842064917087555` | `0.9907389879226685` | `0.20934054255485535` | `0.20934055745601654` |
| train | 49 | `22.259557723999023` | `0.2095365673303604` | `0.9906173944473267` | `0.2102586328983307` | `0.21025866270065308` |
| train | 50 | `37.028106689453125` | `0.20936231315135956` | `0.9907171726226807` | `0.2099982649087906` | `0.20999827980995178` |
| val | 44 | `16.429887771606445` | `0.20892377197742462` | `0.9906876087188721` | `0.20974159240722656` | `0.20974163711071014` |
| val | 46 | `18.913543701171875` | `0.20894134044647217` | `0.9906772375106812` | `0.20977436006069183` | `0.20977438986301422` |

## Current Interpretation

The v2e execution chain passes:

```text
multi-case v2b compact list
-> non-overlapping case split
-> q_useful + ip_xi -> LE_local_jacobian_frame
-> AD_B_local
-> raw-B backprojection
-> per-case attribution
```

However, this tiny no-case-id model does not prove case-level generalization.
The training trajectory reduces train `LE_local_rel` strongly:

```text
train_LE_local_rel: 29.47 -> 1.18
```

but validation `LE_local_rel` gets worse:

```text
val_LE_local_rel: 10.84 -> 18.04
```

The full AD-B metrics remain computable but do not improve:

```text
train_AD_B_local_rel: 0.295 -> 0.304
val_AD_B_local_rel:   0.196 -> 0.209
```

So the conservative conclusion is:

```text
v2e shows the multi-case v2b compact pool can be read, trained, differentiated
with respect to q_useful, and mapped back to raw B.  It does not establish v2
model performance or generalization.
```

## Readiness

This is ready for formal v2 training design, not for a performance claim.

The next step should be design work around:

```text
normalization of q_useful / LE_local / B_local
formal v2 architecture
whether geometry or case/load descriptors are needed
proper train/val objectives and checkpoint selection
```

It should not be ad hoc hyperparameter tuning of this smoke script.

## Boundaries

- No formal training was performed.
- No v1.3/v2 production model, loss, learning-rate, epoch, or split policy was
  changed.
- No tag was moved.
- No old TRUE176 `LE/B` labels were used as v2 labels.
- No `.npz`, `.odb`, `.pt`, `.pth`, checkpoint, or generated output artifact is
  committed.
