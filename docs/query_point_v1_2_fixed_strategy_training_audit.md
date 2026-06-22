# Query-Point v1.2 Fixed-Strategy Training Audit

Date: 2026-06-22

This note records the first v1.2 fixed-strategy training audit after the fresh
strict-pass compact pool reached the minimum coverage target.

## Scope

This audit answers one narrow question:

```text
After expanding the strict-pass fresh compact pool to 10 cases / 9 q-direction
clusters, does the frozen query-point strategy improve held-out LE/B behavior?
```

It is not a model-tuning run.

Constraints held fixed:

- No model-structure change.
- No loss change.
- No learning-rate strategy change.
- No epoch extension beyond the planned 50 epochs.
- No old TRUE176 `LE/B` labels were used as v1.2 training labels.
- No `.npz`, `.odb`, `.pt`, `.pth`, or other large generated outputs are
  committed to git.

## Pool

Frozen pool manifest:

```text
D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest
```

Source coverage audit:

```text
D:\IS-FEM\outputs\query_point_v1_2_data_coverage_audit\full_pool_after_second_fresh_batch
```

Pool summary:

```text
strict_v1_2_pass = true
compact_count = 10
case_count = 10
frame_count = 100
q_direction_cluster_count = 9
pairwise_abs_cos_min = 0.0279701647
pairwise_abs_cos_median = 0.6286590998
pairwise_abs_cos_max = 0.9937922950
strain_field = LE for all compacts
B_label_strain_field = LE for all compacts
```

Cases:

```text
case019
case025
case031
case041
case043
case044
case045
case046
case049
case050
```

## Split

Split plan:

```text
D:\IS-FEM\outputs\query_point_v1_2_training_audit\split_plan_main_80_20.json
```

Training cases:

```text
[19, 25, 31, 43, 44, 45, 46, 50]
```

Validation cases:

```text
[41, 49]
```

Validation split metadata from the trainer:

```text
validation_split_mode = case
validation_is_overlapping = false
train_frames = 80
val_frames = 20
```

Rationale:

- Both validation cases are fresh strict-pass compacts.
- `case041` is a low-LE fresh direction with nearest train abs-cosine
  `0.6417599947` to `case044`.
- `case049` is a mid-LE fresh direction with nearest train abs-cosine
  `0.8522134771` to `case045`.
- `case031` remains in training so the first 80/20 run is not dominated by
  high-LE scale extrapolation.

## Fixed Strategy

Output directory:

```text
D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_main_80_20
```

Training command used the current generic query-point trainer with:

```text
model_style = query-fe-linear-residual
point_feature_source = data
branch_feature_mode = xkeep-qraw
le_normalization = global-component
train_point_sample_count = 128
split_mode = case
allow_overlap_val = false
val_cases = 41,49
epochs = 50
b_baseline_warmstart_source = train-only
b_baseline_warmstart_steps = 500
j_loss_mode = physical
global_b_lr_scale = 0.05
point_b_lr_scale = 0.10
le_loss_weight = 1.0
```

The optimizer parameter groups were:

```text
other lr = 8.0e-05
global_b lr = 4.0e-06
point_b lr = 8.0e-06
```

Warm-start was train-only:

```text
warmstart_train_cases = [19, 25, 31, 43, 44, 45, 46, 50]
warmstart_val_cases_excluded = [41, 49]
warmstart_validation_is_overlapping = false
```

## Results

Best and latest checkpoints are both epoch 50.

```text
best_epoch = 50
best_score = 5.6694053020

train_LE_rel = 0.3710049271
val_LE_rel = 5.1376408458

train_AD_B_rel = 0.5343235027
val_AD_B_rel = 0.5317644562

train_AD_B_cos = 0.8530025687
val_AD_B_cos = 0.8755700366

train_phys_rand_dir_B_rel = 0.5360939388
val_phys_rand_dir_B_rel = 0.5349325279

b_prior_current_train_evalcols_B_rel = 0.5338542326
b_prior_current_val_evalcols_B_rel = 0.5306775207

train_AD_B_ip_rel_max = 4.0114861980
val_AD_B_ip_rel_max = 3.5443742903
val_AD_B_col_rel_max = 0.9385777742
```

Epoch trend:

```text
epoch 1:
  val_LE_rel = 12.7144820421
  val_AD_B_rel = 0.5337132008

epoch 10:
  val_LE_rel = 8.7524213359
  val_AD_B_rel = 0.5321552168

epoch 25:
  val_LE_rel = 7.2616524514
  val_AD_B_rel = 0.5311165897

epoch 50:
  val_LE_rel = 5.1376408458
  val_AD_B_rel = 0.5317644562
```

## Interpretation

The fixed-strategy 10-case audit completed cleanly with a non-overlapping
case-level validation split.

B behavior remains stable in the sense that:

- The train and validation raw-B metrics are close.
- The current B-prior metric and full-model AD-B metric are close.
- Residual AD terms did not blow up the B derivative field.

However, B accuracy did not match the 3-case formal split result:

```text
3-case val_AD_B_rel ~= 0.3716
10-case val_AD_B_rel ~= 0.5318
```

LE generalization is not established.  Although validation LE decreases during
training, the final validation value remains high:

```text
3-case val_LE_rel ~= 2.3336
10-case val_LE_rel ~= 5.1376
```

This run therefore supports the conservative conclusion:

```text
v1.2 fixed-strategy training audit ran cleanly, but this 80/20 split does not
yet show LE generalization improvement from the expanded 10-case pool.
```

The result should not trigger ad hoc tuning of model/loss/lr/epochs inside this
audit.  The next diagnostic step should compare additional fixed-strategy
splits or LOO cases to determine whether this failure is split-specific,
case-specific, or a broader residual/LE generalization issue.

