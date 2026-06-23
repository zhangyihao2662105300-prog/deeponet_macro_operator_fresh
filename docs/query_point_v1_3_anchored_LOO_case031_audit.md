# v1.3 Anchored LE Head LOO Case031 Audit

Date: 2026-06-23

## Purpose

Run the first risk-prioritized leave-one-case-out audit for v1.3 anchored LE.

`case031` was selected first because it is the high-amplitude / high-LE case.
Leaving it out asks whether the anchored head can extrapolate to a large
deformation state without using that case as a training anchor.

This audit only runs `LOO_case031`.  It does not run `LOO_case050` or full LOO.

## Boundary

No model, loss, learning-rate, epoch, data-pool, or tag changes were made.

No old TRUE176 `LE/B` labels were used.  The training data remains the strict
fresh compact pool listed in:

`D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt`

Large generated artifacts remain outside git.

## Split

- `validation_split_mode=case`
- `validation_is_overlapping=false`
- `train_cases=[19,25,41,43,44,45,46,49,50]`
- `val_cases=[31]`
- `train_frames=90`
- `val_frames=10`

Output directory:

`D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case031`

## Fixed Strategy

- `model_style=query-fe-linear-residual-anchored`
- `point_feature_source=data`
- `branch_feature_mode=xkeep-qraw`
- `le_normalization=global-component`
- `train_point_sample_count=128`
- `split_mode=case`
- `allow_overlap_val=false`
- `val_cases=31`
- `epochs=50`
- `b_baseline_warmstart_source=train-only`
- `b_baseline_warmstart_steps=500`
- `j_loss_mode=physical`
- `global_b_lr_scale=0.05`
- `point_b_lr_scale=0.10`
- `le_loss_weight=1.0`
- `baseline_j_loss_mode=norm-plus-physical`
- `baseline_jacobian_weight=1.0`
- `initial_jacobian_weight=1.0`

## Best / Latest Metrics

Best and latest are the same checkpoint in this run:

- `best_epoch=50`
- `latest_epoch=50`
- `best_score=1.6491811828050178`
- `train_LE_rel=0.4854010592101344`
- `val_LE_rel=0.9255415441048325`
- `train_AD_B_rel=0.6559284464681765`
- `val_AD_B_rel=0.7236396387001852`
- `train_AD_B_cos=0.7651519409711313`
- `val_AD_B_cos=0.7363247525021522`
- `train_phys_rand_dir_B_rel=0.6561240604036681`
- `val_phys_rand_dir_B_rel=0.7266812291314823`
- `b_prior_current_train_evalcols_B_rel=0.6554437944013998`
- `b_prior_current_val_evalcols_B_rel=0.7239878680693814`
- `train_AD_B_ip_rel_max=5.949945496582509`
- `val_AD_B_ip_rel_max=9.657011803672734`
- `val_AD_B_col_rel_max=1.1630645221848015`
- `train_zero_q_LE_pred_rms=2.3911419659618957e-13`
- `val_zero_q_LE_pred_rms=2.3911419659618957e-13`
- `train_zero_q_LE_pred_max_abs=5.547806138095357e-13`
- `val_zero_q_LE_pred_max_abs=5.547806138095357e-13`
- `train_zero_q_residual_rms=0.0`
- `val_zero_q_residual_rms=0.0`
- `train_Bprior_q_LE_rel=0.9071082461071032`
- `val_Bprior_q_LE_rel=0.9346750750874383`
- `train_model_minus_Bprior_offset_rms=0.000123498662124308`
- `val_model_minus_Bprior_offset_rms=0.0009073160436565322`

## Comparison With Two-Split Evidence

| Run | Held-out cases | `val_LE_rel` | `val_AD_B_rel` | `zero_q_LE_pred_rms` |
|---|---|---:|---:|---:|
| split_A best | `[41,49]` | 2.1796218548207182 | 0.5322959397919018 | 1.1484941768098402e-11 |
| split_B best | `[44,46]` | 1.5595633647976805 | 0.5125447167349023 | 1.157102562071606e-11 |
| LOO_case031 best | `[31]` | 0.9255415441048325 | 0.7236396387001852 | 2.3911419659618957e-13 |

`LOO_case031` does not collapse on held-out LE: `val_LE_rel` is below 1.0.
However, AD-B and B-prior metrics are substantially worse than split_A/split_B.
This means the high-amplitude case is learnable enough for the anchored LE
value prediction, but its B field is harder to cover when the high-amplitude
anchor is removed from training.

## Per-Case Attribution

Output:

`D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case031_case_attribution`

| Case | `LE_rel` | `AD_B_rel` | `B_prior_rel` | `zero_q_LE_pred_rms` | `Bprior_q_LE_rel` | `model_minus_Bprior_offset_rms` |
|---:|---:|---:|---:|---:|---:|---:|
| 31 | 0.9255415441048325 | 0.7236396387001852 | 0.7239878680693814 | 2.3911419659618957e-13 | 0.9346750750874383 | 0.0009073160436565322 |

## B@q Diagnostic

Output:

`D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case031_bq_anchor_oracle\LOO_case031_best`

| Case | `Btrue_q_LE_rel` | `Bprior_q_LE_rel` | `model_LE_rel` | `zero_q_pred_rms` | `current_offset_rms` | `true_offset_rms` |
|---:|---:|---:|---:|---:|---:|---:|
| 31 | 0.6522641115009261 | 0.967480304420319 | 0.9255415441048325 | 3.720241450096156e-13 | 0.0009074376473615923 | 0.008277004439016715 |

The `Btrue @ q` oracle is not near-perfect on `case031`:
`Btrue_q_LE_rel=0.6522641115009261`.  This differs from the earlier lower
amplitude held-out cases where `Btrue @ q` was much closer to `LE_true`.  The
result is consistent with stronger high-amplitude nonlinearity or state/path
effects in `case031`.

As before, B@q diagnostic values are evaluation-only.  If raw `q_coord` oracle
values differ from trainer/evaluate metrics, the trainer/evaluate metrics are
the primary audit source.

## Interpretation

What is established:

- `LOO_case031` completed with a clean non-overlapping case split.
- The zero-q LE ghost remains structurally fixed.
- The high-amplitude held-out case does not collapse in LE:
  `val_LE_rel=0.9255415441048325`.
- Best and latest are both epoch 50, so this run does not show the early-best
  degradation seen in split_A/split_B.

What is newly exposed:

- AD-B is significantly harder when `case031` is held out:
  `val_AD_B_rel=0.7236396387001852`, compared with about `0.53` and `0.51` in
  split_A/split_B.
- The B prior also degrades:
  `b_prior_current_val_evalcols_B_rel=0.7239878680693814`.
- `Btrue @ q` itself only gives `LE_rel=0.6522641115009261`, so high-amplitude
  value prediction is not a simple linear B@q reconstruction.

Current conclusion:

`LOO_case031` supports the v1.3 anchored head as a useful structural fix even
for the high-amplitude held-out case, because LE stays below 1.0 and zero-q
stays near zero.  It also flags a real remaining risk: high-amplitude B coverage
and state dependence are harder than in the two-split low/mid held-out cases.

Recommended next step:

- Run `LOO_case050` next with the same fixed strategy, before changing model,
  loss, learning rate, or epoch count.
