# v2k Tangent / Training-Schedule Audit

Date: 2026-06-24

## Purpose

v2j showed that:

```text
B_prior(point) is not enough.
B_prior(point, state/regime) is a much better LE value anchor.
```

v2k asks the next narrower question:

```text
Once the state-dependent B prior exists, which AD-B tangent/loss path should be
supervised during training?
```

This is a prototype diagnostic only.  It is not formal training and does not
claim model performance.

## Inputs

Compact list:

```text
D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt
```

Split:

```text
train_cases = [19,25,31,41,43,45,49,50]
val_cases = [44,46]
validation_is_overlapping = false
```

Fixed v2k settings:

```text
b_prior_mode = amp_p75_cluster
detach_b_prior_regime_weight = true
use_q_amp = true
steps = 2000
eval_every = 250
seed = 20260623
warmup_steps = 500
```

Output root:

```text
D:\IS-FEM\outputs\query_point_v2_design_audit\v2k_tangent_schedule\
```

## New Options

`scripts/train_v2_formal_prototype.py` now supports:

```text
--b-loss-target-mode full_output
--b-loss-target-mode anchor_only
--b-loss-target-mode residual_only
--b-loss-target-mode detached_anchor_plus_residual

--training-schedule joint
--training-schedule anchor_then_residual
--training-schedule le_warmup_then_b
--warmup-steps 500
```

Definitions:

```text
full_output:
  B loss supervises d(full LE_hat)/dq.

anchor_only:
  B loss supervises the B-prior branch only.

residual_only:
  B loss supervises d(residual)/dq against B_target - B_prior_detached.

detached_anchor_plus_residual:
  B loss supervises B_prior_detached + d(residual)/dq.

joint:
  LE and B losses are active throughout training.

le_warmup_then_b:
  w_B = 0 for warmup_steps, then restored.

anchor_then_residual:
  freeze B-prior delta parameters and train residual against a detached anchor.
```

The default remains:

```text
b_loss_target_mode = full_output
training_schedule = joint
```

so previous behavior is preserved.

## Shared Anchor Baseline

All v2k runs start from the same p75-cluster state prior:

```text
anchor train_LE_local_rel = 4.77639102935791
anchor val_LE_local_rel = 5.700374126434326
anchor train_AD_B_local_rel = 0.22865082323551178
anchor val_AD_B_local_rel = 0.10279608517885208
anchor val_B_model_raw_projected_rel = 0.10271253436803818
zero_q_LE_local_rms = 0.0
```

This is the reference for judging whether training improves LE without damaging
AD-B/raw-B.

## A-E Results

| Run | B Loss Mode | Schedule | Best Step | Train LE | Val LE | Train AD-B | Val AD-B | Val AD-B Cos | Val Raw-B |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | full_output | joint | 1250 | 4.7013 | 5.4701 | 0.3173 | 0.1194 | 0.9963 | 0.1205 |
| B | anchor_only | joint | 500 | 4.6821 | 5.6596 | 0.2720 | 0.1122 | 0.9974 | 0.1129 |
| C | residual_only | joint | 2000 | 4.7011 | 5.6205 | 0.2893 | 0.1272 | 0.9952 | 0.1287 |
| D | full_output | le_warmup_then_b | 0 | 4.7768 | 5.6993 | 0.2287 | 0.1028 | 0.9986 | 0.1027 |
| E | detached_anchor_plus_residual | anchor_then_residual | 1750 | 4.6965 | 5.6547 | 0.3133 | 0.1312 | 0.9946 | 0.1331 |

Latest-step behavior:

```text
A latest:
  train_LE = 4.6922
  val_LE = 6.2750
  val_AD_B = 0.1315
  val_raw_B = 0.1335

B latest:
  train_LE = 4.6812
  val_LE = 6.4884
  val_AD_B = 0.1242
  val_raw_B = 0.1257

C latest:
  train_LE = 4.7011
  val_LE = 5.6205
  val_AD_B = 0.1272
  val_raw_B = 0.1287

D latest:
  train_LE = 4.6870
  val_LE = 6.3388
  val_AD_B = 0.1443
  val_raw_B = 0.1470

E latest:
  train_LE = 4.6985
  val_LE = 6.1754
  val_AD_B = 0.1315
  val_raw_B = 0.1334
```

## Per-Case Attribution

Representative final per-case rows:

```text
A full_output joint:
  case025 train LE = 43.7479, AD-B = 0.9652
  case031 train LE = 4.6124, AD-B = 0.1928
  case050 train LE = 28.8161, AD-B = 0.2532
  case044 val   LE = 6.9469, AD-B = 0.1321
  case046 val   LE = 5.8483, AD-B = 0.1309

B anchor_only joint:
  case025 train LE = 43.4530, AD-B = 0.7040
  case031 train LE = 4.6037, AD-B = 0.1913
  case050 train LE = 28.3476, AD-B = 0.3645
  case044 val   LE = 5.3305, AD-B = 0.1113
  case046 val   LE = 7.0760, AD-B = 0.1358
```

The persistent failures are not uniform.  The p75 anchor improves the gross
high-amplitude/global-anchor problem, but low-amplitude cases such as `case025`
and `case050` still have large relative LE errors.

## Current Judgment

v2k should be interpreted conservatively:

```text
No tested tangent/schedule solves the LE/AD-B tradeoff.
The p75 state-dependent anchor remains the strongest part of the route.
Training can make small LE improvements, but usually degrades AD-B/raw-B
relative to the anchor-only state.
```

Most stable AD-B:

```text
step 0 anchor-only baseline:
  val_AD_B_local_rel = 0.1028
  val_raw_B = 0.1027
```

Best nonzero-step AD-B/LE compromise:

```text
B_anchor_only_joint best_combined:
  step = 500
  train_LE_local_rel = 4.6821
  val_LE_local_rel = 5.6596
  train_AD_B_local_rel = 0.2720
  val_AD_B_local_rel = 0.1122
  val_B_model_raw_projected_rel = 0.1129
```

Best val LE:

```text
A_full_output_joint best_combined:
  val_LE_local_rel = 5.4701
  val_AD_B_local_rel = 0.1194
  val_raw_B = 0.1205
```

Warmup result:

```text
LE warmup does not improve the selected metric in this diagnostic; best remains
initialization, and latest AD-B/raw-B are worse.
```

## Recommendation

Do not broaden the residual network yet and do not claim a model effect.

The next step should be v2l:

```text
smoother / more informative state-dependent B prior
```

Possible directions:

```text
1. Replace hard p75 threshold with a smoother train-only q_amp gate.
2. Add q-direction or q-cluster features to the B prior itself, not only to the
   residual.
3. Keep anchor-only B supervision as the stable reference.
4. Delay full-output AD-B supervision until the anchor is less coarse.
```

The practical route after v2k is:

```text
B_prior(point, state/regime) stays.
anchor_only B supervision is the safest current tangent/loss reference.
full_output supervision can improve val LE slightly but costs AD-B/raw-B.
The coarse p75 anchor is still too coarse for a formal training claim.
```

## Boundaries

- This is prototype diagnostic, not formal training.
- No old TRUE176 `LE/B` labels were used.
- No `.npz`, `.odb`, `.pt`, `.pth`, checkpoint, loss-history, or generated
  output is committed.
