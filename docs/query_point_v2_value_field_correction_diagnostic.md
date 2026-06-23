# v2i Value-Field Correction Diagnostic

Date: 2026-06-24

## Purpose

v2h showed that the formal v2 prototype is not failing because of coordinate
contract, normalization roundtrip, B-prior normalized/physical mismatch, gate
suppression, or single-case capacity.  v2i narrows the remaining question:

```text
Why does the residual/value branch fail to learn the multi-case LE value-field
correction?
```

This is a diagnostic audit only.  It is not formal training and does not claim
model performance.

## Inputs

Compact list:

```text
D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt
```

Default split:

```text
train_cases = [19,25,31,41,43,45,49,50]
val_cases = [44,46]
```

Outputs:

```text
D:\IS-FEM\outputs\query_point_v2_design_audit\v2i_value_correction\
```

## v2h Already Excluded

The following are unlikely to be the main cause:

```text
normalization roundtrip bug
B-prior normalized/physical mismatch
B-prior table corruption
gate suppression as the main cause
single-case implementation incapacity
```

Single-case `case050` can overfit:

```text
LE_local_rel = 0.03293348103761673
AD_B_local_rel = 0.0039431145414710045
```

## LE-Only vs LE+B

LE-only run:

```text
out_root = D:\IS-FEM\outputs\query_point_v2_design_audit\v2i_value_correction\le_only
w_LE = 1
w_B = 0
w_zero = 10
```

Best combined:

```text
step = 250
train_LE_local_rel = 28.903217315673828
val_LE_local_rel = 10.226631164550781
train_AD_B_local_rel = 0.33925360441207886
val_AD_B_local_rel = 0.24254068732261658
```

Latest:

```text
step = 2000
train_LE_local_rel = 28.9032039642334
val_LE_local_rel = 10.24402141571045
train_AD_B_local_rel = 0.3411771357059479
val_AD_B_local_rel = 0.24326850473880768
```

LE+B run:

```text
out_root = D:\IS-FEM\outputs\query_point_v2_design_audit\v2i_value_correction\le_plus_b
w_LE = 1
w_B = 1
w_zero = 10
```

Best combined:

```text
step = 250
train_LE_local_rel = 28.91905975341797
val_LE_local_rel = 10.357998847961426
train_AD_B_local_rel = 0.3983873128890991
val_AD_B_local_rel = 0.24915096163749695
```

Interpretation:

```text
LE-only also fails to reduce train LE.  The issue is not simply AD-B loss
pressing down the residual value correction.
```

## Per-Case Training Attribution

The LE+B run wrote:

```text
D:\IS-FEM\outputs\query_point_v2_design_audit\v2i_value_correction\le_plus_b\per_case_history.csv
```

Selected examples:

```text
case031:
  step 0:
    LE_local_rel = 29.445392608642578
    B_mean_prior_LE_rel = 29.4448184967041
    Bq_oracle_rel = 0.6423731446266174
    residual_part_rms = 0.0003095715946983546
    B_prior_part_rms = 0.3757256269454956
    target_rms = 0.012815580703318119
  step 2000:
    LE_local_rel = 28.884708404541016
    B_mean_prior_LE_rel = 28.88783836364746
    residual_part_rms = 0.00568649685010314
    B_prior_part_rms = 0.368727445602417

case019:
  step 0:
    LE_local_rel = 37.37459182739258
    B_mean_prior_LE_rel = 37.37150192260742
    Bq_oracle_rel = 0.06301647424697876
  step 2000:
    LE_local_rel = 36.490623474121094
    B_mean_prior_LE_rel = 36.505428314208984
    residual_part_rms = 0.0020881269592791796

case050:
  step 0:
    LE_local_rel = 17.49992561340332
    B_mean_prior_LE_rel = 17.51706886291504
    Bq_oracle_rel = 0.044320594519376755
  step 2000:
    LE_local_rel = 17.42636489868164
    B_mean_prior_LE_rel = 20.278799057006836
    residual_part_rms = 0.0022045494988560677
```

Interpretation:

```text
The residual moves, but it does not become a large enough or correct
case/state-dependent value correction.  Predictions remain close to the poor
global train-mean B-prior value anchor.
```

## Remove case031

Run:

```text
train_cases = [19,25,41,43,45,49,50]
val_cases = [44,46]
```

Best combined:

```text
step = 250
train_LE_local_rel = 36.591793060302734
val_LE_local_rel = 7.851358890533447
train_AD_B_local_rel = 0.2906542718410492
val_AD_B_local_rel = 0.18610313534736633
```

Latest:

```text
step = 2000
train_LE_local_rel = 36.59855270385742
val_LE_local_rel = 8.906996726989746
train_AD_B_local_rel = 0.26411208510398865
val_AD_B_local_rel = 0.1820947527885437
```

Interpretation:

```text
Removing case031 improves held-out low-amplitude validation LE but does not fix
train LE.  case031 is important, but it is not the only source of value-anchor
failure.
```

## q_dir Descriptor

Run:

```text
--use-q-dir
```

Best combined:

```text
step = 250
train_LE_local_rel = 28.908042907714844
val_LE_local_rel = 10.261488914489746
train_AD_B_local_rel = 0.3572276830673218
val_AD_B_local_rel = 0.24071422219276428
```

Latest:

```text
step = 2000
train_LE_local_rel = 28.912994384765625
val_LE_local_rel = 12.533048629760742
train_AD_B_local_rel = 0.6289283633232117
val_AD_B_local_rel = 0.30773547291755676
```

Interpretation:

```text
Adding q_dir alone does not solve the value-field correction problem and can
hurt AD-B later in the run.
```

## Amplitude-Regime Descriptor

Run:

```text
--use-amp-regime-descriptor
```

This adds `q_amp_scaled^2` and `log1p(q_amp_scaled)` to the branch input.

Best combined:

```text
step = 250
train_LE_local_rel = 28.912452697753906
val_LE_local_rel = 10.579803466796875
train_AD_B_local_rel = 0.4507596790790558
val_AD_B_local_rel = 0.2403922826051712
```

Latest:

```text
step = 2000
train_LE_local_rel = 28.907602310180664
val_LE_local_rel = 11.634325981140137
train_AD_B_local_rel = 0.39266496896743774
val_AD_B_local_rel = 0.27040615677833557
```

Interpretation:

```text
A simple nonlinear amplitude descriptor does not solve the issue either.
```

## Value-Anchor Baselines

Global train-mean B prior from v2h:

```text
train_LE_local_rel = 29.466125499590156
val_LE_local_rel = 10.826594009457496
```

Clustered B prior by `q_amp` median:

```text
train_LE_local_rel = 21.19461440553013
val_LE_local_rel = 5.700911624442508
train_AD_B_local_rel = 0.27490432440403983
val_AD_B_local_rel = 0.10293348806464925
```

Clustered B prior by `q_amp` p75:

```text
train_LE_local_rel = 4.77639109241935
val_LE_local_rel = 5.700373146340969
train_AD_B_local_rel = 0.22865080859937914
val_AD_B_local_rel = 0.10279608559825179
```

Per-case train B-prior upper bound:

```text
train_LE_local_rel = 0.29293951890558007
train_AD_B_local_rel = 0.027328778282940253
train_B_raw_projected_rel = 0.02765599599942974
```

Per-case examples:

```text
case019: case_mean_B_LE_rel = 0.03035894761575067
case025: case_mean_B_LE_rel = 0.023026516778057612
case031: case_mean_B_LE_rel = 0.29369085332550365
case043: case_mean_B_LE_rel = 0.050092113454312644
case050: case_mean_B_LE_rel = 0.035844814999807566
```

Interpretation:

```text
The value-field problem is mostly an anchor/state-dependence problem.  A
point-only global mean B prior is a very poor value anchor.  Even a crude
amplitude-clustered B prior is much better, and a per-case B prior is a strong
upper bound.
```

## Current Diagnosis

v2i narrows the cause further:

```text
AD-B loss alone: not the main cause
case031 alone: not the main cause
q_dir alone: insufficient
simple q_amp nonlinear descriptor alone: insufficient
global point-only train-mean B prior: poor value anchor
state-dependent / regime-dependent B prior: strongly indicated
```

The most likely cause is:

```text
The formal prototype anchors LE to a point-only global mean B prior.  That prior
has decent average derivative metrics but a terrible B@q value prediction across
mixed cases.  The residual MLP is then asked to repair a huge case/state-specific
value error while preserving AD-B, and it fails.
```

## v2j Recommendation

Do not broaden v2 training yet.

The next prototype should change the value anchor/B prior, not merely make the
residual wider.  Recommended v2j directions:

1. Make `B_prior_norm` state-dependent:

```text
B_prior_norm(point, q_amp, q_dir or q_cluster)
```

2. Start with an auditable low-risk version:

```text
two-cluster or amplitude-regime B prior
linear interpolation between low/high B priors by q_amp
```

3. Keep the anchored zero-q structure and raw-B backprojection metrics.

4. Re-run the same diagnostic split before claiming any model improvement.

## Boundaries

- This is diagnostic, not formal training.
- No old TRUE176 `LE/B` labels were used.
- No `.npz`, `.odb`, `.pt`, `.pth`, checkpoint, or generated output is
  committed.
