# v2g Formal Prototype Audit

Date: 2026-06-23

## Purpose

This audit implements the v2f formal prototype recommendation as a
script-local trainer.  It is a prototype audit, not a final formal training
result.

The goal is to check whether the v2 coordinate-consistent data chain can be
trained with:

```text
q_useful component normalization
LE_local component normalization
B_norm[a,k] = B_local[a,k] * q_std[k] / LE_std[a]
explicit q_amp descriptor
anchored zero-q structure
AD-B local evaluation
raw-B backprojection evaluation
```

It does not use old TRUE176 `LE/B` labels, does not write checkpoints, and does
not commit generated output files.

## Input And Split

Compact list:

```text
D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt
```

Cases:

```text
all_cases = [19,25,31,41,43,44,45,46,49,50]
train_cases = [19,25,31,41,43,45,49,50]
val_cases = [44,46]
validation_is_overlapping = false
compact_count = 10
```

Run:

```powershell
py -3 scripts\train_v2_formal_prototype.py `
  --compact-list D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v2_formal_prototype\split_B_min10 `
  --val-cases 44,46 `
  --steps 5000 `
  --seed 20260623 `
  --eval-every 250 `
  --use-q-amp
```

Outputs:

```text
D:\IS-FEM\outputs\query_point_v2_formal_prototype\split_B_min10\training_summary.json
D:\IS-FEM\outputs\query_point_v2_formal_prototype\split_B_min10\normalization_summary.json
D:\IS-FEM\outputs\query_point_v2_formal_prototype\split_B_min10\metrics_history.csv
D:\IS-FEM\outputs\query_point_v2_formal_prototype\split_B_min10\per_case_attribution.csv
```

## Train-Only Normalization

Normalization was computed from train cases only; validation cases 44 and 46
were excluded.

Key normalization stats:

```text
q_std_min = 0.00015828311158064476
q_std_max = 0.033928895258162534
q_std_ratio_before_floor = 214.35575102954596
q_std_floor_value = 1.866417625552965e-06
q_std_floored_count = 0

LE_std = [
  0.006498310714960098,
  0.0038321511819958687,
  0.0040618302300572395,
  0.004930346272885799,
  0.003132438752800226,
  0.003945972304791212
]
LE_std_ratio_before_floor = 2.074521237523037
LE_std_floor_value = 4.0039012470375455e-06
LE_std_floored_count = 0

q_amp_rms = 0.08326140530721668
B_norm_scale_min = 0.024357577421965322
B_norm_scale_max = 10.83146290488151
B_norm_scale_rms = 2.962409527222749
```

`LE_local` is scaled but not mean-shifted, so the anchored structure can keep
`q=0 -> LE=0`.

## Prototype Model

The script-local model uses:

```text
branch = q_useful_normed[42] + q_amp_scaled[1]
trunk = ip_xi[3]
use_q_amp = true
use_q_dir = false
use_geometry_features = false
gate_c = 0.1
```

The normalized output is:

```text
LE_norm_hat =
  B_prior_norm(point) @ q_normed
  + gate(q_amp) * (R(q_normed, q_amp, point) - R(0, 0, point))
```

The zero-q subtraction is exact in the model structure, and the audit verifies
`zero_q_LE_local_rms = 0.0`.

## Loss And Selection

Training loss:

```text
loss = LE_norm_mse + AD_B_norm_mse + 10 * zero_q_mse
```

Selection score:

```text
score =
  val_LE_local_rel
  + val_AD_B_local_rel
  + 0.5 * val_B_model_raw_projected_rel
  + 10 * zero_q_LE_norm_rms
```

The script writes metric JSON files for:

```text
best_combined
best_LE
best_B
best_raw_projected
latest
```

No checkpoint tensors are written.

## Best Combined Result

Best combined is also best LE:

```text
step = 250
score = 10.733413934707642

train_LE_local_rel = 28.91905975341797
train_LE_local_rmse = 0.13137225806713104
train_LE_target_rms = 0.004542756825685501
train_AD_B_local_rel = 0.3983873128890991
train_AD_B_local_cos = 0.9200356602668762
train_B_model_raw_projected_rel = 0.4033678472042084

val_LE_local_rel = 10.357998847961426
val_LE_local_rmse = 0.0011231731623411179
val_LE_target_rms = 0.000108435342554003
val_AD_B_local_rel = 0.24915096163749695
val_AD_B_local_cos = 0.9772958159446716
val_B_model_raw_projected_rel = 0.25252825021743774

zero_q_LE_local_rms = 0.0
```

## Latest Result

Latest checkpoint-equivalent metrics:

```text
step = 5000
score = 14.255964800715446

train_LE_local_rel = 28.909543991088867
train_LE_local_rmse = 0.1313290297985077
train_LE_target_rms = 0.004542756825685501
train_AD_B_local_rel = 0.41042250394821167
train_AD_B_local_cos = 0.9156504273414612
train_B_model_raw_projected_rel = 0.41631460189819336

val_LE_local_rel = 13.804399490356445
val_LE_local_rmse = 0.001496884855441749
val_LE_target_rms = 0.000108435342554003
val_AD_B_local_rel = 0.29874974489212036
val_AD_B_local_cos = 0.9589686989784241
val_B_model_raw_projected_rel = 0.3056311309337616

zero_q_LE_local_rms = 0.0
```

## Validation Per-Case Attribution

Validation cases are both low-amplitude cases:

```text
case044:
  LE_local_rel = 13.474361419677734
  LE_local_rmse = 0.0012535760179162025
  LE_target_rms = 9.303417027695104e-05
  AD_B_local_rel = 0.34123140573501587
  AD_B_local_cos = 0.9420009851455688
  B_raw_projected_rel = 0.35068729519844055
  q_amp_mean = 0.0010563157266005874
  q_amp_scaled_mean = 0.012686739675700665

case046:
  LE_local_rel = 13.993033409118652
  LE_local_rmse = 0.001705835689790547
  LE_target_rms = 0.00012190607230877504
  AD_B_local_rel = 0.24912229180335999
  AD_B_local_cos = 0.9768862724304199
  B_raw_projected_rel = 0.2526634931564331
  q_amp_mean = 0.0010737687116488814
  q_amp_scaled_mean = 0.012896356172859669
```

The absolute RMSE remains small, but because the held-out target RMS is very
small, the relative LE error remains large.

## Comparison With v2e

v2e latest baseline:

```text
train_LE_local_rel = 1.1830968856811523
val_LE_local_rel = 18.03921890258789
train_AD_B_local_rel = 0.3041848838329315
val_AD_B_local_rel = 0.2089325487613678
train_B_model_raw_projected_rel = 0.30487608909606934
val_B_model_raw_projected_rel = 0.2097579836845398
zero_q_LE_local_rms = 0.0
```

v2g latest relative to v2e latest:

```text
val_LE_local_rel_delta = -4.234819412231445
train_LE_local_rel_delta = +27.726447105407715
val_AD_B_local_rel_delta = +0.08981719613075256
val_B_raw_projected_rel_delta = +0.0958731472492218
```

The best v2g validation LE relative error is lower than v2e latest, but the
train LE relative error is still very high and AD-B is worse than the v2e
latest baseline.  Therefore this is not evidence that v2g model performance is
established.

## Current Judgment

Conservative conclusion:

```text
v2g formal prototype audit executes.
Train-only normalization, q_amp, anchored zero-q, AD-B local metrics, and raw-B
backprojection metrics are implemented and auditable.
Model effect is not established.
```

The run supports the engineering chain, but it does not yet solve the
low-amplitude held-out LE problem.  The remaining issue is likely still tied to
coverage, descriptors, architecture capacity, or the balance between the B
prior and value-field learning, rather than the v2 coordinate contract itself.

## Boundaries

- This is not a final formal training result.
- No old TRUE176 `LE/B` labels were used.
- No `.npz`, `.odb`, `.pt`, `.pth`, checkpoint, or output artifact is committed.
- `use_q_amp=true`.
- `use_q_dir=false`.
- `use_geometry_features=false`.
- Broader v2 training is not ready to be claimed as effective from this run
  alone.
