# v2j State-Dependent B Prior Audit

Date: 2026-06-24

## Purpose

v2i narrowed the v2 formal prototype failure from "the residual does not learn"
to a more specific issue:

```text
The point-only global train-mean B prior is a poor multi-case LE value anchor.
```

v2j tests a small state/regime-dependent B prior before changing residual
network size or making a formal performance claim.

This is a prototype / diagnostic audit only.

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

Output root:

```text
D:\IS-FEM\outputs\query_point_v2_design_audit\v2j_state_b_prior\
```

## Prior Modes

`scripts/train_v2_formal_prototype.py` now supports:

```text
--b-prior-mode global_mean
--b-prior-mode amp_median_cluster
--b-prior-mode amp_p75_cluster
--b-prior-mode amp_linear_interp
```

Definitions:

```text
global_mean:
  B_prior(point) = train mean B_norm(point)

amp_median_cluster:
  split train frames by q_amp median and use low/high B means

amp_p75_cluster:
  split train frames by q_amp p75 and use low/high B means

amp_linear_interp:
  interpolate between low/high B means using q_amp
```

The default remains:

```text
global_mean
```

so old behavior is preserved unless the new flag is used.

## Regime-Weight Detach

The diagnostic default is:

```text
--detach-b-prior-regime-weight
```

This lets the B prior act as a state-selected local value anchor without adding
the derivative of the regime selector into the AD-B target.  This is a
diagnostic choice, not a final tangent-consistency decision.

The full differentiable diagnostic is available as:

```text
--no-detach-b-prior-regime-weight
```

If a state-dependent B prior becomes the formal route, the tangent definition
must explicitly decide whether derivatives of the regime/interpolation weight
belong in the supervised AD-B.

## No-Training Anchor Audit

Script:

```text
scripts/audit_v2_state_dependent_b_prior.py
```

It evaluates anchor-only LE, AD-B-local, and raw-B backprojection metrics for
the prior modes without training and without writing checkpoints.

Global mean baseline:

```text
train_LE_local_rel = 29.46612548828125
val_LE_local_rel = 10.826595306396484
train_AD_B_local_rel = 0.29525285959243774
val_AD_B_local_rel = 0.19587740302085876
```

Median cluster:

```text
threshold = 0.003093380878729739
cluster sizes = 40 low / 40 high
train_LE_local_rel = 21.19461441040039
val_LE_local_rel = 5.700911998748779
train_AD_B_local_rel = 0.2749043405056
val_AD_B_local_rel = 0.10293348878622055
```

p75 cluster:

```text
threshold = 0.005826574499032849
cluster sizes = 60 low / 20 high
train_LE_local_rel = 4.77639102935791
val_LE_local_rel = 5.700374126434326
train_AD_B_local_rel = 0.22865082323551178
val_AD_B_local_rel = 0.10279608517885208
```

Linear interpolation, detached:

```text
threshold = 0.005826574499032849
cluster sizes = 60 low / 20 high
train_LE_local_rel = 5.709594249725342
val_LE_local_rel = 5.700374126434326
train_AD_B_local_rel = 0.2520321011543274
val_AD_B_local_rel = 0.10279608517885208
```

Linear interpolation, differentiable:

```text
train_LE_local_rel = 5.709594249725342
val_LE_local_rel = 5.700374126434326
train_AD_B_local_rel = 0.2519484758377075
val_AD_B_local_rel = 0.10279608517885208
```

Interpretation:

```text
The state/regime-dependent anchor strongly improves the no-training value
anchor relative to global_mean.  The p75 cluster reproduces the v2i p75
baseline and is the strongest train anchor in this diagnostic.
```

## Training Prototype Runs

All runs keep:

```text
--use-q-amp
--seed 20260623
--eval-every 250
zero-q anchor
AD-B local metrics
raw-B backprojection metrics
```

### Global Mean

Best combined:

```text
step = 250
train_LE_local_rel = 28.903976440429688
val_LE_local_rel = 10.439322471618652
train_AD_B_local_rel = 0.3928983211517334
val_AD_B_local_rel = 0.24318574368953705
val_AD_B_local_cos = 0.9792882800102234
val_B_model_raw_projected_rel = 0.24605792760849
zero_q_LE_local_rms = 0.0
```

This reproduces the v2i/v2g global-mean failure mode.

### p75 Cluster, Detached

Anchor metrics:

```text
train_LE_local_rel = 4.77639102935791
val_LE_local_rel = 5.700374126434326
train_AD_B_local_rel = 0.22865082323551178
val_AD_B_local_rel = 0.10279608517885208
```

Best combined:

```text
step = 1250
train_LE_local_rel = 4.701291084289551
val_LE_local_rel = 5.470116138458252
train_AD_B_local_rel = 0.31725096702575684
val_AD_B_local_rel = 0.11935402452945709
val_AD_B_local_cos = 0.9963414072990417
val_B_model_raw_projected_rel = 0.12048347294330597
zero_q_LE_local_rms = 0.0
```

Latest:

```text
step = 2000
train_LE_local_rel = 4.692205429077148
val_LE_local_rel = 6.275022506713867
train_AD_B_local_rel = 0.3029906153678894
val_AD_B_local_rel = 0.13147956132888794
val_B_model_raw_projected_rel = 0.1334630846977234
zero_q_LE_local_rms = 0.0
```

Interpretation:

```text
p75 clustering fixes the catastrophic initialization anchor and gives a small
additional training improvement in LE.  AD-B/raw-B degrade relative to the
anchor-only metrics, so this is not yet a finished training strategy.
```

### Linear Interpolation, Detached

Anchor metrics:

```text
train_LE_local_rel = 5.709594249725342
val_LE_local_rel = 5.700374126434326
train_AD_B_local_rel = 0.2520321011543274
val_AD_B_local_rel = 0.10279608517885208
```

Best combined remains initialization:

```text
step = 0
train_LE_local_rel = 5.710006237030029
val_LE_local_rel = 5.699275016784668
train_AD_B_local_rel = 0.25205448269844055
val_AD_B_local_rel = 0.1027916669845581
val_B_model_raw_projected_rel = 0.10270832479000092
zero_q_LE_local_rms = 0.0
```

Latest:

```text
step = 2000
train_LE_local_rel = 5.606312274932861
val_LE_local_rel = 9.569080352783203
train_AD_B_local_rel = 0.4365951418876648
val_AD_B_local_rel = 0.42422565817832947
val_B_model_raw_projected_rel = 0.4392179548740387
zero_q_LE_local_rms = 0.0
```

Interpretation:

```text
The interpolated anchor improves initialization, but training destabilizes
AD-B/raw-B in this short diagnostic.
```

### Linear Interpolation, Differentiable

Best LE:

```text
step = 250
train_LE_local_rel = 5.6188435554504395
val_LE_local_rel = 5.678556442260742
train_AD_B_local_rel = 0.26526346802711487
val_AD_B_local_rel = 0.11865215003490448
val_B_model_raw_projected_rel = 0.11953507363796234
zero_q_LE_local_rms = 0.0
```

Latest:

```text
step = 1000
train_LE_local_rel = 5.6100544929504395
val_LE_local_rel = 9.868256568908691
train_AD_B_local_rel = 0.4017329812049866
val_AD_B_local_rel = 0.2207990139722824
val_B_model_raw_projected_rel = 0.2277982085943222
zero_q_LE_local_rms = 0.0
```

Interpretation:

```text
The differentiable interpolation is viable as a diagnostic but does not solve
the training-stage AD-B/value tradeoff in this run.
```

## Per-Case Attribution

For the p75-cluster run, best combined still leaves difficult low-amplitude
train cases:

```text
case025: LE_local_rel = 43.747928619384766
case050: LE_local_rel = 28.81610870361328
case031: LE_local_rel = 4.612374305725098
case044: LE_local_rel = 6.946887493133545
case046: LE_local_rel = 5.848255634307861
```

This means the p75 anchor fixes the gross high-amplitude/global-anchor error,
but it is still too coarse for all cases.

## Current Judgment

v2j should be written conservatively:

```text
State/regime-dependent B prior is a much better value anchor than global_mean.
The p75 cluster anchor reproduces the v2i no-training baseline.
The training prototype starts from a much less catastrophic LE state.
Residual training gives only small LE improvement and can degrade AD-B/raw-B.
Model performance is not established.
```

The most useful result is not the final trained metric.  It is the diagnosis:

```text
The value anchor was indeed a primary blocker.
The next route should keep state-dependent B prior, but needs a cleaner
training/tangent treatment before claiming a model improvement.
```

## Next Recommendation

Do not simply widen the residual network yet.  The next step should refine the
state-dependent B prior and its tangent handling:

```text
1. keep p75 cluster as a hard low-risk reference
2. add an explicit train-only q_amp regime table / smoother monotone gate
3. separate anchor fit from residual fit or temporarily freeze the B-prior
   anchor during early residual training
4. evaluate whether the AD-B loss should supervise anchor-only B, full tangent,
   or a detached-regime tangent
```

The current v2j result supports moving from:

```text
B_prior(point)
```

to:

```text
B_prior(point, state/regime)
```

but not yet to a formal performance claim.

## Boundaries

- This is prototype / diagnostic, not formal training.
- No old TRUE176 `LE/B` labels were used.
- No `.npz`, `.odb`, `.pt`, `.pth`, checkpoint, loss-history, or generated
  output is committed.
