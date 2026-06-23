# v1.3 Anchored LE Head Split_B Audit

Date: 2026-06-23

Purpose:

- Run the second fixed-strategy formal audit for the v1.3 anchored LE head.
- Compare directly against the v1.2 split_B result.
- Keep the only intended model change relative to v1.2 as
  `model_style=query-fe-linear-residual-anchored`.
- Change only the validation split relative to v1.3 split_A.
- Do not change loss, learning rate, epoch count, split policy, data pool, or
  training labels.

## Boundary

This is a split_B audit only.  It is not a leave-one-out audit.

No old TRUE176 `LE/B` labels were used.  The training pool is the strict v1.2
fresh compact pool:

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

The compact list used was:

`D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt`

All entered compacts were current `complete_case*_training_ready.npz` files,
with explicit `strain_field=LE`, `B_label_strain_field=LE`, and strict v1.2
data-contract checks already passed.

Large generated artifacts remain outside git.

## Fixed Strategy

The v1.3 split_B audit used the same fixed strategy as v1.3 split_A, changing
only `val_cases`:

- `model_style=query-fe-linear-residual-anchored`
- `point_feature_source=data`
- `branch_feature_mode=xkeep-qraw`
- `le_normalization=global-component`
- `train_point_sample_count=128`
- `split_mode=case`
- `allow_overlap_val=false`
- `val_cases=44,46`
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

Output directory:

`D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_B_val44_46`

## Split

- `validation_split_mode=case`
- `validation_is_overlapping=false`
- `train_cases=[19,25,31,41,43,45,49,50]`
- `val_cases=[44,46]`
- `train_frames=80`
- `val_frames=20`

## v1.2 Baseline

Baseline output:

`D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_split_B_val44_46`

v1.2 split_B best/latest:

- `train_LE_rel=0.3992176434086097`
- `val_LE_rel=8.515458607946384`
- `train_AD_B_rel=0.517751748077844`
- `val_AD_B_rel=0.5103261083920874`
- `val_AD_B_cos=0.8833993891072419`
- `b_prior_current_val_evalcols_B_rel=0.5093552486612618`

## v1.3 Result

Best checkpoint:

- `epoch=1`
- `score=2.072108081532583`
- `train_LE_rel=1.4833762049278454`
- `val_LE_rel=1.5595633647976805`
- `train_AD_B_rel=0.5253695536442143`
- `val_AD_B_rel=0.5125447167349023`
- `train_AD_B_cos=0.8530685106570757`
- `val_AD_B_cos=0.8790831126173732`
- `train_phys_rand_dir_B_rel=0.5235002475746036`
- `val_phys_rand_dir_B_rel=0.5103388758275206`
- `b_prior_current_train_evalcols_B_rel=0.5253683666040365`
- `b_prior_current_val_evalcols_B_rel=0.5125437997617942`
- `train_AD_B_ip_rel_max=4.371187815670929`
- `val_AD_B_ip_rel_max=3.8185485580892022`
- `val_AD_B_col_rel_max=0.7844640484737759`
- `train_zero_q_LE_pred_rms=1.157102562071606e-11`
- `val_zero_q_LE_pred_rms=1.157102562071606e-11`
- `train_zero_q_LE_pred_max_abs=2.614902738784508e-11`
- `val_zero_q_LE_pred_max_abs=2.614902738784508e-11`
- `train_zero_q_residual_rms=0.0`
- `val_zero_q_residual_rms=0.0`
- `train_Bprior_q_LE_rel=1.9669166608075834`
- `val_Bprior_q_LE_rel=1.4833928570650252`
- `train_model_minus_Bprior_offset_rms=2.1507669482353404e-05`
- `val_model_minus_Bprior_offset_rms=1.7732607370311359e-06`

Latest checkpoint:

- `epoch=50`
- `score=2.2825738136376588`
- `train_LE_rel=0.4254284697637384`
- `val_LE_rel=1.7711132238745921`
- `train_AD_B_rel=0.5192545011469342`
- `val_AD_B_rel=0.5114605897630667`
- `train_AD_B_cos=0.8591610678802043`
- `val_AD_B_cos=0.8821396700473547`
- `train_phys_rand_dir_B_rel=0.521875721461898`
- `val_phys_rand_dir_B_rel=0.5155716613597328`
- `b_prior_current_train_evalcols_B_rel=0.5178881106624103`
- `b_prior_current_val_evalcols_B_rel=0.509199943726343`
- `train_AD_B_ip_rel_max=4.038719666152137`
- `val_AD_B_ip_rel_max=3.4659318289659984`
- `val_AD_B_col_rel_max=0.7836263221857965`
- `train_zero_q_LE_pred_rms=1.157102562071606e-11`
- `val_zero_q_LE_pred_rms=1.157102562071606e-11`
- `train_zero_q_LE_pred_max_abs=2.614902738784508e-11`
- `val_zero_q_LE_pred_max_abs=2.614902738784508e-11`
- `train_zero_q_residual_rms=0.0`
- `val_zero_q_residual_rms=0.0`
- `train_Bprior_q_LE_rel=1.602284943964546`
- `val_Bprior_q_LE_rel=1.3981617323778495`
- `train_model_minus_Bprior_offset_rms=0.001039109275001205`
- `val_model_minus_Bprior_offset_rms=9.181341274153851e-05`

## v1.2 vs v1.3 Split_B

| Metric | v1.2 split_B | v1.3 anchored best | Interpretation |
|---|---:|---:|---|
| `val_LE_rel` | 8.515458607946384 | 1.5595633647976805 | improved on split_B |
| `val_AD_B_rel` | 0.5103261083920874 | 0.5125447167349023 | stable |
| `val_AD_B_cos` | 0.8833993891072419 | 0.8790831126173732 | stable |
| `b_prior_current_val_evalcols_B_rel` | 0.5093552486612618 | 0.5125437997617942 | stable |
| `zero_q_LE_pred_rms` | not recorded in v1.2 trainer | 1.157102562071606e-11 | structurally removed |

The best v1.3 split_B checkpoint improves `val_LE_rel` substantially while
preserving the B metrics.  The latest checkpoint remains much better than the
v1.2 split_B baseline, but is slightly worse than the best checkpoint.

## Per-Case Attribution

Output:

`D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_B_val44_46_case_attribution`

Best checkpoint per-case attribution:

| Case | `LE_rel` | `AD_B_rel` | `B_prior_rel` | `zero_q_LE_pred_rms` | `Bprior_q_LE_rel` | `model_minus_Bprior_offset_rms` |
|---:|---:|---:|---:|---:|---:|---:|
| 44 | 1.2623462866926172 | 0.512531727633403 | 0.5125308365182847 | 1.157102562071606e-11 | 1.2603167162795477 | 1.612453271090318e-06 |
| 46 | 1.7080236136717677 | 0.5125577066694196 | 0.5125567638373174 | 1.157102562071606e-11 | 1.706468997850503 | 1.9340682029719545e-06 |

Both held-out cases retain stable AD-B and near-zero zero-q behavior.

## B@q Diagnostic Note

Output:

`D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_B_bq_anchor_oracle\split_B_val44_46_best`

The B@q oracle script is evaluation-only.  It confirms near-zero q=0 outputs in
that diagnostic path.  Its `Bprior@q` numbers use the older raw `q_coord`
diagnostic style, so the primary v1.3 anchored `Bprior_q_LE_rel` values should
be taken from the trainer/evaluate metrics above.

## Conclusion

The v1.3 anchored LE head split_B audit completed.

What is established:

- The formal case split is clean: `validation_is_overlapping=false`.
- The zero-q LE ghost remains structurally removed.
- The split_B best `val_LE_rel` improves from `8.5155` to `1.5596`.
- AD-B remains stable: `val_AD_B_rel` changes from `0.5103` to `0.5125`.
- Together with split_A, the anchored model has now improved LE on two
  independent validation splits without breaking B.

What is not established:

- v1.3 broad performance is not fully proven until LOO or a formal v1.3
  summary is completed.
- The best checkpoint occurs very early on split_B, so epoch sensitivity should
  remain visible in the report.

Recommended next step:

- Either run leave-one-case-out with the same fixed strategy, or first write a
  compact formal v1.3 two-split summary.  Do not change loss, learning rate,
  epoch count, or architecture before that comparison is recorded.
