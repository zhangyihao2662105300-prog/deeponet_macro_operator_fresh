# v1.3 Anchored LE Head Split_A Audit

Date: 2026-06-23

Purpose:

- Run the first formal fixed-strategy audit for the v1.3 anchored LE head.
- Compare directly against the v1.2 split_A result.
- Change only `model_style` from `query-fe-linear-residual` to
  `query-fe-linear-residual-anchored`.
- Do not change loss, learning rate, epoch count, split, data pool, or training
  labels.

## Boundary

This is a split_A audit only.  It is not a split_B, LOO, or broad performance
claim.

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

The v1.3 audit used the same fixed strategy as v1.2 split_A except for the
model style:

- `model_style=query-fe-linear-residual-anchored`
- `point_feature_source=data`
- `branch_feature_mode=xkeep-qraw`
- `le_normalization=global-component`
- `train_point_sample_count=128`
- `split_mode=case`
- `allow_overlap_val=false`
- `val_cases=41,49`
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

`D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_A_val41_49`

## Split

- `validation_split_mode=case`
- `validation_is_overlapping=false`
- `train_cases=[19,25,31,43,44,45,46,50]`
- `val_cases=[41,49]`
- `train_frames=80`
- `val_frames=20`

## v1.2 Baseline

Baseline output:

`D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_main_80_20`

v1.2 split_A best/latest:

- `train_LE_rel=0.37100492709620475`
- `val_LE_rel=5.137640845816213`
- `train_AD_B_rel=0.5343235027465005`
- `val_AD_B_rel=0.5317644562295766`
- `val_AD_B_cos=0.8755700365748692`
- `b_prior_current_val_evalcols_B_rel=0.5306775207214273`
- prior v1.2 zero-q diagnostic: `zero_q_pred_rms ~= 7.81e-04`

## v1.3 Result

Best checkpoint:

- `epoch=10`
- `score=2.71191779461262`
- `train_LE_rel=0.7879168866205835`
- `val_LE_rel=2.1796218548207182`
- `train_AD_B_rel=0.5389748535009075`
- `val_AD_B_rel=0.5322959397919018`
- `train_AD_B_cos=0.846836356997155`
- `val_AD_B_cos=0.8718914233087541`
- `train_phys_rand_dir_B_rel=0.5382632234120508`
- `val_phys_rand_dir_B_rel=0.5317559154590024`
- `b_prior_current_train_evalcols_B_rel=0.5389720779955626`
- `b_prior_current_val_evalcols_B_rel=0.532185230457769`
- `train_AD_B_ip_rel_max=4.190292491830496`
- `val_AD_B_ip_rel_max=3.6808686821477834`
- `val_AD_B_col_rel_max=0.9017393721901116`
- `train_zero_q_LE_pred_rms=1.1484941768098398e-11`
- `val_zero_q_LE_pred_rms=1.1484941768098402e-11`
- `train_zero_q_LE_pred_max_abs=2.70666752899551e-11`
- `val_zero_q_LE_pred_max_abs=2.70666752899551e-11`
- `train_zero_q_residual_rms=0.0`
- `val_zero_q_residual_rms=0.0`
- `train_Bprior_q_LE_rel=1.4480483711929562`
- `val_Bprior_q_LE_rel=2.8518558429535927`
- `train_model_minus_Bprior_offset_rms=0.0010582778364344478`
- `val_model_minus_Bprior_offset_rms=0.00010172704191546938`

Latest checkpoint:

- `epoch=50`
- `score=3.5342489497223206`
- `train_LE_rel=0.3609466014204402`
- `val_LE_rel=3.0012689077205144`
- `train_AD_B_rel=0.5354624908732145`
- `val_AD_B_rel=0.5329800420018063`
- `train_AD_B_cos=0.8517631052796969`
- `val_AD_B_cos=0.8738174330412968`
- `train_phys_rand_dir_B_rel=0.5372316091218243`
- `val_phys_rand_dir_B_rel=0.5361412541537594`
- `b_prior_current_train_evalcols_B_rel=0.5338382718320831`
- `b_prior_current_val_evalcols_B_rel=0.5306402468885563`
- `train_AD_B_ip_rel_max=4.002757766552021`
- `val_AD_B_ip_rel_max=3.547008811624606`
- `val_AD_B_col_rel_max=0.825376288169154`
- `train_zero_q_LE_pred_rms=1.1484941768098398e-11`
- `val_zero_q_LE_pred_rms=1.1484941768098402e-11`
- `train_zero_q_LE_pred_max_abs=2.70666752899551e-11`
- `val_zero_q_LE_pred_max_abs=2.70666752899551e-11`
- `train_zero_q_residual_rms=0.0`
- `val_zero_q_residual_rms=0.0`
- `train_Bprior_q_LE_rel=1.306933164553619`
- `val_Bprior_q_LE_rel=2.6828986080522137`
- `train_model_minus_Bprior_offset_rms=0.0010395098096060433`
- `val_model_minus_Bprior_offset_rms=0.0003313721329316278`

## v1.2 vs v1.3 Split_A

| Metric | v1.2 split_A | v1.3 anchored best | Interpretation |
|---|---:|---:|---|
| `val_LE_rel` | 5.137640845816213 | 2.1796218548207182 | improved on split_A |
| `val_AD_B_rel` | 0.5317644562295766 | 0.5322959397919018 | stable |
| `val_AD_B_cos` | 0.8755700365748692 | 0.8718914233087541 | stable |
| `b_prior_current_val_evalcols_B_rel` | 0.5306775207214273 | 0.532185230457769 | stable |
| `zero_q_pred_rms` | about 7.81e-04 | 1.1484941768098402e-11 | structurally removed |

The best v1.3 checkpoint is substantially better than v1.2 split_A in
`val_LE_rel`, while preserving the B metrics.  The latest checkpoint is worse
than the best checkpoint, so the result is still epoch-sensitive.

## Per-Case Attribution

Output:

`D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_A_val41_49_case_attribution`

Best checkpoint per-case attribution:

| Case | `LE_rel` | `AD_B_rel` | `B_prior_rel` | `zero_q_LE_pred_rms` | `Bprior_q_LE_rel` | `model_minus_Bprior_offset_rms` |
|---:|---:|---:|---:|---:|---:|---:|
| 41 | 3.609960155721904 | 0.5321915350416156 | 0.5321483814179929 | 1.1484941768098398e-11 | 3.8535600056709796 | 0.00010339966315563473 |
| 49 | 1.8090445751604918 | 0.5324002526523295 | 0.532222051735648 | 1.1484941768098398e-11 | 1.8501516802362072 | 0.00010005442067530404 |

The held-out LE error is still much worse on `case041` than on `case049`, but
both cases keep stable AD-B and zero-q behavior.

## B@q Diagnostic Note

Output:

`D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_A_bq_anchor_oracle\split_A_val41_49_best`

The B@q oracle script is evaluation-only.  It confirms zero-q outputs are
exactly zero in that diagnostic path.  Its `Bprior@q` numbers use the older raw
`q_coord` diagnostic style, so the primary v1.3 anchored `Bprior_q_LE_rel`
values should be taken from the trainer/evaluate metrics above.

## Conclusion

The v1.3 anchored LE head split_A audit completed.

What is established:

- The formal case split is clean: `validation_is_overlapping=false`.
- The zero-q LE ghost is structurally removed.
- The split_A best `val_LE_rel` improves from `5.1376` to `2.1796`.
- AD-B remains stable: `val_AD_B_rel` changes only from `0.5318` to `0.5323`.
- The result is promising enough to run split_B next.

What is not established:

- v1.3 broad performance is not proven.
- split_B and LOO have not been run in this audit.
- Latest checkpoint degradation means best-checkpoint selection still matters.

Recommended next step:

- Run the same fixed strategy on split_B before changing loss, learning rate,
  epoch count, or architecture again.
