# v1.3 Anchored LE Head Two-Split Formal Summary

Date: 2026-06-23

## Stage Conclusion

The v1.3 anchored LE head is now supported by two formal non-overlapping
case-level splits:

- split_A: held out `case041` and `case049`
- split_B: held out `case044` and `case046`

Across both splits, the anchored head materially improves held-out `LE` relative
to the v1.2 fixed strategy, keeps AD-B stable, and structurally removes the
zero-q ghost LE field.

This is strong evidence that the anchored LE head is an effective direction.
It is not yet a full case-level generalization claim: leave-one-case-out is
still required, especially for high-amplitude and mixed-direction cases that
have not yet been held out.

## Audit Boundary

The v1.3 audits used the strict v1.2 fresh compact pool:

- `case019`
- `case025`
- `case031`
- `case041`
- `case043`
- `case044`
- `case045`
- `case046`
- `case049`
- `case050`

No old TRUE176 `LE/B` labels were used.

The fixed strategy was kept constant:

- `model_style=query-fe-linear-residual-anchored`
- `point_feature_source=data`
- `branch_feature_mode=xkeep-qraw`
- `le_normalization=global-component`
- `train_point_sample_count=128`
- `split_mode=case`
- `allow_overlap_val=false`
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

No loss, learning-rate, epoch-count, training-data, or tag changes were made in
these audits.  Large generated artifacts remain outside git.

## Split_A

Split:

- `train_cases=[19,25,31,43,44,45,46,50]`
- `val_cases=[41,49]`
- `validation_is_overlapping=false`
- `train_frames=80`
- `val_frames=20`

Outputs:

- v1.2 baseline:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_main_80_20`
- v1.3 anchored:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_A_val41_49`

| Metric | v1.2 split_A best/latest | v1.3 split_A best | v1.3 split_A latest |
|---|---:|---:|---:|
| `best_epoch` / `latest_epoch` | 50 / 50 | 10 | 50 |
| `val_LE_rel` | 5.137640845816213 | 2.1796218548207182 | 3.0012689077205144 |
| `train_LE_rel` | 0.37100492709620475 | 0.7879168866205835 | 0.3609466014204402 |
| `val_AD_B_rel` | 0.5317644562295766 | 0.5322959397919018 | 0.5329800420018063 |
| `train_AD_B_rel` | 0.5343235027465005 | 0.5389748535009075 | 0.5354624908732145 |
| `val_AD_B_cos` | 0.8755700365748692 | 0.8718914233087541 | 0.8738174330412968 |
| `b_prior_current_val_evalcols_B_rel` | 0.5306775207214273 | 0.532185230457769 | 0.5306402468885563 |
| `val_zero_q_LE_pred_rms` | prior diagnostic about 7.81e-04 | 1.1484941768098402e-11 | 1.1484941768098402e-11 |
| `val_model_minus_Bprior_offset_rms` | not recorded in trainer | 0.00010172704191546938 | 0.0003313721329316278 |

Interpretation:

- Held-out `LE` improves substantially.
- AD-B remains stable.
- The zero-q ghost is removed.
- Latest is worse than best, so split_A remains epoch-sensitive.

## Split_B

Split:

- `train_cases=[19,25,31,41,43,45,49,50]`
- `val_cases=[44,46]`
- `validation_is_overlapping=false`
- `train_frames=80`
- `val_frames=20`

Outputs:

- v1.2 baseline:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_split_B_val44_46`
- v1.3 anchored:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_B_val44_46`

| Metric | v1.2 split_B best/latest | v1.3 split_B best | v1.3 split_B latest |
|---|---:|---:|---:|
| `best_epoch` / `latest_epoch` | 50 / 50 | 1 | 50 |
| `val_LE_rel` | 8.515458607946384 | 1.5595633647976805 | 1.7711132238745921 |
| `train_LE_rel` | 0.3992176434086097 | 1.4833762049278454 | 0.4254284697637384 |
| `val_AD_B_rel` | 0.5103261083920874 | 0.5125447167349023 | 0.5114605897630667 |
| `train_AD_B_rel` | 0.517751748077844 | 0.5253695536442143 | 0.5192545011469342 |
| `val_AD_B_cos` | 0.8833993891072419 | 0.8790831126173732 | 0.8821396700473547 |
| `b_prior_current_val_evalcols_B_rel` | 0.5093552486612618 | 0.5125437997617942 | 0.509199943726343 |
| `val_zero_q_LE_pred_rms` | not recorded in v1.2 trainer | 1.157102562071606e-11 | 1.157102562071606e-11 |
| `val_model_minus_Bprior_offset_rms` | not recorded in trainer | 1.7732607370311359e-06 | 9.181341274153851e-05 |

Interpretation:

- Held-out `LE` improves substantially.
- AD-B remains stable.
- The zero-q ghost remains removed.
- Latest is slightly worse than best, so epoch sensitivity remains visible but
  is milder than split_A.

## Per-Case Attribution

| Split | Case | `LE_rel` | `AD_B_rel` | `zero_q_LE_pred_rms` | `Bprior_q_LE_rel` | `model_minus_Bprior_offset_rms` |
|---|---:|---:|---:|---:|---:|---:|
| split_A | 41 | 3.609960155721904 | 0.5321915350416156 | 1.1484941768098398e-11 | 3.8535600056709796 | 0.00010339966315563473 |
| split_A | 49 | 1.8090445751604918 | 0.5324002526523295 | 1.1484941768098398e-11 | 1.8501516802362072 | 0.00010005442067530404 |
| split_B | 44 | 1.2623462866926172 | 0.512531727633403 | 1.157102562071606e-11 | 1.2603167162795477 | 1.612453271090318e-06 |
| split_B | 46 | 1.7080236136717677 | 0.5125577066694196 | 1.157102562071606e-11 | 1.706468997850503 | 1.9340682029719545e-06 |

The worst of the four held-out cases is still `case041`, but it is improved
relative to the v1.2 split_A failure.  All four cases keep stable AD-B and
near-zero q=0 LE output.

## Conservative Interpretation

What is established:

- v1.3 fixes the q=0 ghost LE issue structurally.
- `LE` improves on both tested non-overlapping case splits.
- B supervision is not broken: `val_AD_B_rel`, `val_AD_B_cos`, and the query B
  prior metrics remain close to v1.2.
- The anchored head is now a credible v1.3 direction.

What is not established:

- Full case-level generalization is not proven.
- `case019`, `case025`, `case031`, `case043`, `case045`, and `case050` have not
  yet been evaluated as held-out validation cases under v1.3.
- Best epochs are early: split_A best is epoch 10 and split_B best is epoch 1.
  This means epoch sensitivity remains part of the current evidence.

## LOO Priority Plan

Current 10 cases:

`[19,25,31,41,43,44,45,46,49,50]`

Already held out in split_A/split_B:

`[41,49,44,46]`

Not yet held out:

`[19,25,31,43,45,50]`

Priority 1:

- `LOO_case031`
- Reason: `case031` is the high-amplitude / high-LE case.  Leaving it out
  tests whether v1.3 can extrapolate to a large-amplitude state without using
  that state as a training anchor.

Priority 2:

- `LOO_case050`
- Reason: `case050` is a mixed / larger fresh direction.  It tests non-low
  amplitude mixed-direction generalization.

Priority 3:

- `LOO_case043` or `LOO_case045`
- Reason: these cover medium fresh directions and add direction-generalization
  evidence without immediately paying the cost of full LOO.

Priority 4:

- Full 10-case LOO
- Reason: required for a final formal case-level generalization claim.

## Command Drafts

These commands are plans only.  They were not run in this summary step.

### LOO_case031

Validation:

- `val_cases=31`
- train cases implied by case split:
  `[19,25,41,43,44,45,46,49,50]`

```powershell
$env:PYTHONPATH='src'
$env:KMP_DUPLICATE_LIB_OK='TRUE'
py -3 -m macro_deeponet.train_true176_generic_sobolev `
  --compact-list 'D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt' `
  --out-dir 'D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case031' `
  --model-style query-fe-linear-residual-anchored `
  --point-feature-source data `
  --branch-feature-mode xkeep-qraw `
  --le-normalization global-component `
  --train-point-sample-count 128 `
  --split-mode case `
  --val-cases '31' `
  --epochs 50 `
  --b-baseline-warmstart-source train-only `
  --b-baseline-warmstart-steps 500 `
  --j-loss-mode physical `
  --global-b-lr-scale 0.05 `
  --point-b-lr-scale 0.10 `
  --le-loss-weight 1.0 `
  --baseline-j-loss-mode norm-plus-physical `
  --baseline-jacobian-weight 1.0 `
  --initial-jacobian-weight 1.0 `
  --jacobian-columns all `
  --eval-columns '0,1,2,3,4,5,6,7,8,9,10,11' `
  --batch-size 4 `
  --eval-batch-size 1 `
  --max-eval-frames 512 `
  --basis-dim 96 `
  --hidden-dim 384 `
  --branch-depth 5 `
  --trunk-depth 5 `
  --activation tanh `
  --fe-baseline-scale 1.0 `
  --lr 8.0e-5 `
  --lr-decay 0.9995 `
  --weight-decay 1.0e-5 `
  --grad-clip 10.0 `
  --eval-every 5 `
  --log-every 1 `
  --seed 20260620
```

### LOO_case050

Validation:

- `val_cases=50`
- train cases implied by case split:
  `[19,25,31,41,43,44,45,46,49]`

```powershell
$env:PYTHONPATH='src'
$env:KMP_DUPLICATE_LIB_OK='TRUE'
py -3 -m macro_deeponet.train_true176_generic_sobolev `
  --compact-list 'D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt' `
  --out-dir 'D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case050' `
  --model-style query-fe-linear-residual-anchored `
  --point-feature-source data `
  --branch-feature-mode xkeep-qraw `
  --le-normalization global-component `
  --train-point-sample-count 128 `
  --split-mode case `
  --val-cases '50' `
  --epochs 50 `
  --b-baseline-warmstart-source train-only `
  --b-baseline-warmstart-steps 500 `
  --j-loss-mode physical `
  --global-b-lr-scale 0.05 `
  --point-b-lr-scale 0.10 `
  --le-loss-weight 1.0 `
  --baseline-j-loss-mode norm-plus-physical `
  --baseline-jacobian-weight 1.0 `
  --initial-jacobian-weight 1.0 `
  --jacobian-columns all `
  --eval-columns '0,1,2,3,4,5,6,7,8,9,10,11' `
  --batch-size 4 `
  --eval-batch-size 1 `
  --max-eval-frames 512 `
  --basis-dim 96 `
  --hidden-dim 384 `
  --branch-depth 5 `
  --trunk-depth 5 `
  --activation tanh `
  --fe-baseline-scale 1.0 `
  --lr 8.0e-5 `
  --lr-decay 0.9995 `
  --weight-decay 1.0e-5 `
  --grad-clip 10.0 `
  --eval-every 5 `
  --log-every 1 `
  --seed 20260620
```
