# Query-Point v1.2 Fixed-Strategy Split Diagnostic

Date: 2026-06-23

This note records the first split/case diagnostic after the v1.2 fixed-strategy
10-case audit failed to show LE generalization improvement.

The purpose is attribution, not tuning.

## Boundaries

- No model-structure change.
- No loss change.
- No learning-rate change.
- No epoch-count change.
- No old TRUE176 `LE/B` labels were used as v1.2 labels.
- No checkpoint, loss-history, NPZ, ODB, or other large generated outputs are
  committed to git.

Fixed strategy for all training runs:

```text
model_style = query-fe-linear-residual
point_feature_source = data
branch_feature_mode = xkeep-qraw
le_normalization = global-component
train_point_sample_count = 128
split_mode = case
allow_overlap_val = false
epochs = 50
b_baseline_warmstart_source = train-only
b_baseline_warmstart_steps = 500
j_loss_mode = physical
global_b_lr_scale = 0.05
point_b_lr_scale = 0.10
le_loss_weight = 1.0
```

## Attribution Script

Added:

```text
scripts/evaluate_v1_2_checkpoint_by_case.py
```

The script reconstructs the generic query-point trainer preprocessing from a
saved checkpoint and compact list, then evaluates selected case ids with the
existing trainer metrics.  It does not train or update weights.

Outputs:

```text
D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_main_80_20_case_attribution
D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_split_B_val44_46_case_attribution
```

## Split Plan

Generated:

```text
D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_split_diagnostic_plan
```

Planned splits:

```text
split_A completed:
  val_cases = [41, 49]

split_B run this round:
  val_cases = [44, 46]

split_C planned only:
  val_cases = [43, 50]

split_D planned only:
  val_cases = [25, 45]
```

Only split_B was run in this round.

## Split A

Output:

```text
D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_main_80_20
```

Split:

```text
train_cases = [19, 25, 31, 43, 44, 45, 46, 50]
val_cases = [41, 49]
validation_is_overlapping = false
```

Best and latest were both epoch 50:

```text
train_LE_rel = 0.3710
val_LE_rel = 5.1376
train_AD_B_rel = 0.5343
val_AD_B_rel = 0.5318
val_AD_B_cos = 0.8756
val_phys_rand_dir_B_rel = 0.5349
b_prior_current_val_evalcols_B_rel = 0.5307
val_AD_B_ip_rel_max = 3.5444
val_AD_B_col_rel_max = 0.9386
```

Per-case attribution:

```text
case041:
  LE_rel = 9.1422
  AD_B_rel = 0.5318
  phys_rand_dir_B_rel = 0.5330
  B_prior_rel = 0.5306
  AD_B_cos = 0.8754

case049:
  LE_rel = 4.0235
  AD_B_rel = 0.5317
  phys_rand_dir_B_rel = 0.5330
  B_prior_rel = 0.5307
  AD_B_cos = 0.8757
```

Split_A therefore was not caused by both validation cases behaving identically:
case041 was much worse for LE than case049, while B metrics were nearly the
same for both.

## Split B

Output:

```text
D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_split_B_val44_46
```

Split:

```text
train_cases = [19, 25, 31, 41, 43, 45, 49, 50]
val_cases = [44, 46]
validation_is_overlapping = false
```

Validation case context:

```text
case044:
  q_direction_cluster_id = 5
  LE_rms_mean = 8.3005e-05
  nearest_train_case = 45
  nearest_train_abs_cosine = 0.7265

case046:
  q_direction_cluster_id = 6
  LE_rms_mean = 1.0913e-04
  nearest_train_case = 19
  nearest_train_abs_cosine = 0.7670
```

Best and latest were both epoch 50:

```text
train_LE_rel = 0.3992
val_LE_rel = 8.5155
train_AD_B_rel = 0.5178
val_AD_B_rel = 0.5103
val_AD_B_cos = 0.8834
val_phys_rand_dir_B_rel = 0.5143
b_prior_current_val_evalcols_B_rel = 0.5094
val_AD_B_ip_rel_max = 3.3736
val_AD_B_col_rel_max = 1.0005
```

Per-case attribution:

```text
case044:
  LE_rel = 9.6526
  AD_B_rel = 0.5102
  phys_rand_dir_B_rel = 0.5132
  B_prior_rel = 0.5094
  AD_B_cos = 0.8835

case046:
  LE_rel = 7.7823
  AD_B_rel = 0.5104
  phys_rand_dir_B_rel = 0.5189
  B_prior_rel = 0.5094
  AD_B_cos = 0.8833
```

## Interpretation

Split_B did not rescue the result.  It was worse than split_A for LE:

```text
split_A val_LE_rel = 5.1376
split_B val_LE_rel = 8.5155
```

B stayed comparatively stable and remained aligned with the query B prior:

```text
split_A val_AD_B_rel = 0.5318
split_A B_prior_val = 0.5307

split_B val_AD_B_rel = 0.5103
split_B B_prior_val = 0.5094
```

This supports a stronger but still conservative diagnosis:

```text
The first v1.2 10-case failure is not just a case041/case049 split-specific
artifact.  Under the frozen strategy and current single-geometry 10-case pool,
LE case-level generalization appears systematically weak, while B remains
controlled by the query B prior.
```

This does not yet prove that the model architecture is bad.  It says the next
useful question is no longer "did split_A pick unlucky validation cases?" but:

```text
Why does the residual/LE prediction branch fail to generalize even when B is
kept stable?
```

Recommended next diagnostic options, still without ad hoc tuning:

- Run split_C or split_D only if more split evidence is needed.
- Inspect LE scale/normalization and residual branch behavior.
- Compare LE-only residual predictions against the B-prior derivative field.
- Consider whether the current single geometry and low-LE fresh cases are
  under-informative for LE values even though they constrain B.

