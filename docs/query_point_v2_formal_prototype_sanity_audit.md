# v2h Formal Prototype Sanity / Ablation Audit

Date: 2026-06-23

## Purpose

v2g passed as an engineering audit but did not pass as a model-learning result:

```text
best_combined step = 250
train_LE_local_rel = 28.91905975341797
val_LE_local_rel = 10.357998847961426

latest step = 5000
train_LE_local_rel = 28.909543991088867
val_LE_local_rel = 13.804399490356445
```

The key question for v2h is therefore not validation generalization.  It is:

```text
Why does the formal prototype fail to reduce train LE on the multi-case split?
```

This audit checks normalization roundtrip, B@q baselines, v2g initialization,
single-case overfit, B-prior freezing, and gate behavior.  It is diagnostic only
and is not a formal training result.

## Inputs

Compact list:

```text
D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt
```

v2g output:

```text
D:\IS-FEM\outputs\query_point_v2_formal_prototype\split_B_min10
```

Sanity output:

```text
D:\IS-FEM\outputs\query_point_v2_design_audit\v2g_sanity
```

Split:

```text
train_cases = [19,25,31,41,43,45,49,50]
val_cases = [44,46]
```

## Normalization Roundtrip

Normalization vectors are recomputed with the same train-only helper used by
v2g.  The v2g JSON stores scalar audit stats, not the full `q_std` vector.

Train:

```text
q_roundtrip_rel = 4.792623713739907e-17
LE_roundtrip_rel = 3.441183768746357e-17
B_roundtrip_rel = 3.787348081354616e-17
```

Validation:

```text
q_roundtrip_rel = 4.0597878073265377e-17
LE_roundtrip_rel = 3.354931535711588e-17
B_roundtrip_rel = 3.759929863285529e-17
```

Interpretation:

```text
normalization / inverse-normalization is numerically closed.
```

This does not look like a normalization roundtrip bug.

## Exact B@q Oracle

Using the true per-frame useful B:

```text
LE_hat = B_local_useful @ q_useful
```

Results:

```text
train_LE_local_rel = 0.6407233225614739
train_LE_local_rmse = 0.002910650128152691
train_LE_target_rms = 0.004542756825685501

val_LE_local_rel = 0.037104433513767326
val_LE_local_rmse = 4.023431656891624e-06
val_LE_target_rms = 0.000108435342554003
```

The exact B@q oracle is strong on the low-amplitude validation cases but not
perfect on the high-amplitude/mixed train pool.  This is consistent with the
v2f finding that `case031` dominates the B@q train-side error.

## Train-Mean B Prior Baseline

Using:

```text
B_mean_train(point) = mean_train_frames(B_local_useful)
LE_hat = B_mean_train(point) @ q_useful
```

Results:

```text
train_LE_local_rel = 29.466125499590156
train_AD_B_local_rel = 0.2952528280656775
train_AD_B_local_cos = 0.9554191580239643
train_B_raw_projected_rel = 0.29530885157242365

val_LE_local_rel = 10.826594009457496
val_AD_B_local_rel = 0.195877397799659
val_AD_B_local_cos = 0.9939725867408077
val_B_raw_projected_rel = 0.19550375115262664
```

This exactly explains the v2g initialization behavior: the mean B prior is good
as an average derivative field, but `B_mean @ q` is a poor value-field predictor
for the multi-case training pool.

## Normalized Train-Mean B Prior Baseline

The normalized baseline:

```text
B_norm_mean_train = mean_train(B_local * q_std / LE_std)
LE_norm_hat = B_norm_mean_train @ q_norm
LE_hat = LE_norm_hat * LE_std
```

matches the physical train-mean baseline:

```text
train_LE_local_rel = 29.466125499590152
val_LE_local_rel = 10.826594009457496
train_AD_B_local_rel = 0.2952528280656775
val_AD_B_local_rel = 0.195877397799659
```

Interpretation:

```text
The normalized B-prior implementation is equivalent to the physical B-prior
baseline.  The poor train LE is not caused by normalized-vs-physical mismatch.
```

## v2g Init / Best / Latest

v2g initialization:

```text
train_LE_local_rel = 29.466707229614258
val_LE_local_rel = 10.825510025024414
train_AD_B_local_rel = 0.29526641964912415
val_AD_B_local_rel = 0.19587081670761108
```

v2g best combined:

```text
step = 250
train_LE_local_rel = 28.91905975341797
val_LE_local_rel = 10.357998847961426
train_AD_B_local_rel = 0.3983873128890991
val_AD_B_local_rel = 0.24915096163749695
```

v2g latest:

```text
step = 5000
train_LE_local_rel = 28.909543991088867
val_LE_local_rel = 13.804399490356445
train_AD_B_local_rel = 0.41042250394821167
val_AD_B_local_rel = 0.29874974489212036
```

Interpretation:

```text
v2g starts as train-mean B prior plus zero residual.  Training only slightly
improves LE value metrics and degrades B metrics.  It does not learn the
multi-case value-field correction.
```

## Single-Case case050 Overfit

Run:

```powershell
py -3 scripts\train_v2_formal_prototype.py `
  --compact-list D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v2_design_audit\v2g_sanity\single_case050_overfit `
  --train-cases 50 `
  --val-cases 50 `
  --allow-overlap-val `
  --steps 2000 `
  --seed 20260623 `
  --eval-every 250 `
  --use-q-amp
```

Result:

```text
train_LE_local_rel = 0.03293348103761673
train_LE_local_rmse = 8.987132787297014e-06
train_LE_target_rms = 0.0002728874096646905
train_AD_B_local_rel = 0.0039431145414710045
train_AD_B_local_cos = 0.9999921321868896
train_B_raw_projected_rel = 0.003975565545260906
zero_q_LE_local_rms = 0.0
```

Interpretation:

```text
The v2g formal prototype can overfit one case.  The implementation is not dead,
and the normalization/model/AD path can reproduce a one-case value and B field.
```

## Freeze B Prior Ablation

Run uses the original split and:

```text
--freeze-b-prior-table
```

Best combined:

```text
step = 500
train_LE_local_rel = 28.97667121887207
val_LE_local_rel = 10.584187507629395
train_AD_B_local_rel = 0.5441113114356995
val_AD_B_local_rel = 0.2669954299926758
```

Latest:

```text
step = 1000
train_LE_local_rel = 28.9519100189209
val_LE_local_rel = 10.826315879821777
train_AD_B_local_rel = 0.42075225710868835
val_AD_B_local_rel = 0.1958789825439453
```

Interpretation:

```text
Freezing the train-mean B prior does not solve train LE.  Direct corruption of
the B-prior table is not the main failure source.
```

## Gate Ablation

Run uses the original split and:

```text
--gate-c 1e-9
```

This makes the residual gate approximately always open for nonzero q.

Best combined is initialization:

```text
step = 0
train_LE_local_rel = 29.466724395751953
val_LE_local_rel = 10.820159912109375
train_AD_B_local_rel = 0.2953197956085205
val_AD_B_local_rel = 0.19590897858142853
```

Latest:

```text
step = 1000
train_LE_local_rel = 28.90526008605957
val_LE_local_rel = 19.146818161010742
train_AD_B_local_rel = 0.8936592936515808
val_AD_B_local_rel = 0.8248840570449829
```

Interpretation:

```text
Opening the gate does not fix train LE and makes AD-B much worse.  The failure
is not simply caused by gate(q_amp) suppressing the residual.
```

## Current Diagnosis

v2h narrows the failure:

```text
normalization roundtrip bug: unlikely
B-prior normalized/physical mismatch: unlikely
B-prior table being directly destroyed: unlikely
gate suppressing residual: unlikely as the main cause
single-case implementation capacity: passes
multi-case formal split learning: still fails
```

The most likely issue is now:

```text
The v2g branch/trunk residual is not expressive or well-conditioned enough to
learn a case/state-dependent value-field correction on the mixed 8-case train
pool, starting from a train-mean B prior whose B@q value prediction is very poor
for high-amplitude/mixed train cases.
```

This is different from saying the coordinate contract is wrong.  The coordinate
contract still looks sound.

## Next Recommendation

Do not move to broader v2 training yet.

Next useful diagnostics:

1. Add per-case train attribution for v2g training history, especially
   high-amplitude `case031`.
2. Test a small case-conditioned or q-direction-aware residual descriptor
   prototype, still as an audit, not as final model policy.
3. Compare LE-only multi-case training against LE+B training to see whether
   AD-B loss prevents the residual from correcting the value field.
4. Consider a value-field anchor using exact or case-cluster B@q, because
   train-mean B@q is a poor value initializer for the multi-case pool.

## Boundaries

- No old TRUE176 `LE/B` labels were used.
- No `.npz`, `.odb`, `.pt`, `.pth`, checkpoint, or generated output is committed.
- This is not a formal model-effect result.
- v2g/v2h prove an auditable engineering and sanity chain, but not broader v2
  learning performance.
