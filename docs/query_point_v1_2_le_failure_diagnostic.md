# Query-Point v1.2 LE Failure Diagnostic

Date: 2026-06-23

This note decomposes the high validation `LE_rel` observed in the v1.2 fixed
strategy audits.  It is an evaluation-only diagnostic: no model, loss,
learning-rate, epoch, or data-contract setting was changed.

## Inputs

Split_A checkpoint:

```text
D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_main_80_20\best.pt
```

Split_B checkpoint:

```text
D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_split_B_val44_46\best.pt
```

Shared compact list:

```text
D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt
```

Diagnostic output:

```text
D:\IS-FEM\outputs\query_point_v1_2_training_audit\le_failure_diagnostic
```

## Script

Added:

```text
scripts/diagnose_v1_2_le_failure.py
```

The script reconstructs the generic query-point preprocessing from the saved
checkpoint, runs forward prediction for selected cases, and writes:

```text
le_case_summary.csv
le_component_summary.csv
le_worst_ip_summary.csv
le_frame_trend_all_cases.csv
le_frame_trend_case###.csv
le_offset_summary.csv
le_offset_by_case.csv
le_failure_diagnostic_summary.json
```

It also computes train-only affine LE oracle and B-prior offset diagnostics.  It
does not train or update weights.

## Main Finding

The high `LE_rel` is not just a relative-metric artifact from tiny validation
strain magnitudes.  The model predicts LE values whose RMS is about 4-10 times
larger than the target RMS, and a zero prediction is better than the model for
all checked validation cases.

```text
case041: target_rms=9.33e-05, pred_rms=8.60e-04, LE_rel=9.14
case049: target_rms=2.21e-04, pred_rms=9.04e-04, LE_rel=4.02
case044: target_rms=9.36e-05, pred_rms=9.18e-04, LE_rel=9.65
case046: target_rms=1.23e-04, pred_rms=9.64e-04, LE_rel=7.78
```

Zero baseline:

```text
zero_pred_LE_rel = 1.0 for all cases
```

So the model is worse than predicting zero on these held-out cases.

## Scale And Bias

Scale-only correction reduces LE relative error close to 1, but not below 1:

```text
case041: optimal_scale=0.0137, scaled_LE_rel=0.9920
case049: optimal_scale=0.0473, scaled_LE_rel=0.9810
case044: optimal_scale=0.0205, scaled_LE_rel=0.9796
case046: optimal_scale=0.0141, scaled_LE_rel=0.9939
```

Bias-only correction barely helps:

```text
case041: bias_corrected_LE_rel=9.1269
case049: bias_corrected_LE_rel=4.0113
case044: bias_corrected_LE_rel=9.5471
case046: bias_corrected_LE_rel=7.6996
```

Scale-plus-bias also stays near 1:

```text
case041: scale_bias_corrected_LE_rel=0.9914
case049: scale_bias_corrected_LE_rel=0.9810
case044: scale_bias_corrected_LE_rel=0.9794
case046: scale_bias_corrected_LE_rel=0.9893
```

Interpretation:

```text
The LE failure is dominated by an overlarge predicted value field, but it is not
only a scalar calibration or constant-offset problem.  The held-out LE spatial /
component pattern is weak as well.
```

## Components And IPs

Worst average components:

```text
split_A:
  LE11 mean_rel ~= 13.58
  LE23 mean_rel ~= 8.80
  LE22 mean_rel ~= 8.69

split_B:
  LE11 mean_rel ~= 17.98
  LE23 mean_rel ~= 14.08
  LE22 mean_rel ~= 9.88
```

Worst IPs are localized but severe.  Examples:

```text
split_A case041:
  worst IP_LE_rel = 43.56 at ip_local=37

split_B case044:
  worst IP_LE_rel = 40.99 at ip_local=11
```

The local worst-component relative values can be very large because some
component targets are near zero at those IPs.  Still, the aggregate LE prediction
is also too large, so this is not purely a single-IP reporting artifact.

## Frame / Alpha Trend

LE relative error is worst at small alpha and improves with larger alpha, but it
remains high even at alpha near 1:

```text
case041:
  alpha=0.1 -> LE_rel=52.05
  alpha=1.0 -> LE_rel=6.37

case049:
  alpha=0.1 -> LE_rel=22.16
  alpha=1.0 -> LE_rel=2.87

case044:
  alpha=0.1 -> LE_rel=59.65
  alpha=1.0 -> LE_rel=6.06

case046:
  alpha=0.1 -> LE_rel=45.77
  alpha=1.0 -> LE_rel=5.14
```

This supports two points:

- Low target RMS amplifies relative errors at small alpha.
- The model is still wrong at high alpha, so low-alpha scaling alone does not
  explain the failure.

## Affine Oracle

The train-only affine LE oracle is not usable on held-out cases in its
unregularized least-squares form:

```text
split_A affine_design_rank = 41 / 49
split_A affine_design_condition ~= 1.51e20

split_B affine_design_rank = 41 / 49
split_B affine_design_condition ~= 1.37e20
```

Held-out affine oracle LE relative errors are enormous:

```text
case041: affine_oracle_LE_rel ~= 4.79e6
case049: affine_oracle_LE_rel ~= 9.99e6
case044: affine_oracle_LE_rel ~= 1.98e6
case046: affine_oracle_LE_rel ~= 2.06e6
```

Interpretation:

```text
The current 8-train-case q design is numerically rank-deficient / ill-conditioned
for an unconstrained affine value-field fit.  This is evidence that the current
case directions do not provide a stable value-field extrapolation basis, even
though they are enough to keep the query B prior controlled.
```

This is not a recommendation to use the affine oracle as a model; it is a
diagnostic showing that value-field extrapolation is fragile.

## Offset / Anchor

True offset:

```text
offset_true = LE_true - B_true @ q48
```

Validation true offsets are tiny:

```text
case041 offset_true_rms = 3.73e-06
case049 offset_true_rms = 1.14e-05
case044 offset_true_rms = 3.56e-06
case046 offset_true_rms = 2.87e-06
```

Predicted offset using the query B prior:

```text
offset_pred = LE_pred - B_prior @ q48
```

is much larger:

```text
case041 offset_pred_rms = 7.96e-04
case049 offset_pred_rms = 7.87e-04
case044 offset_pred_rms = 8.97e-04
case046 offset_pred_rms = 8.99e-04
```

Offset relative errors:

```text
case041 pred_offset_vs_true_offset_rel = 213.1
case049 pred_offset_vs_true_offset_rel = 68.9
case044 pred_offset_vs_true_offset_rel = 252.0
case046 pred_offset_vs_true_offset_rel = 313.2
```

This is the clearest signal:

```text
B prior is stable, but the LE value anchor/integration constant learned by the
network is wrong on held-out cases.
```

Offset by case also shows that training contains a high-amplitude case:

```text
case031:
  LE_rms = 1.27e-02
  offset_true_rms = 8.28e-03
  q_norm_mean = 0.2119
```

while most fresh validation cases have:

```text
LE_rms ~= 9e-05 to 2e-04
offset_true_rms ~= 3e-06 to 1e-05
q_norm_mean ~= 0.0011 to 0.0032
```

This scale imbalance may be pulling the learned LE value anchor away from the
low-amplitude held-out cases.

## Current Diagnosis

The most likely current explanation is:

```text
The query B prior learns a stable derivative field, but the residual/LE value
branch predicts a nonzero LE anchor/value field of roughly 8e-04 to 1e-03 RMS on
held-out low-LE cases.  The true held-out LE values are much smaller, so the
model is worse than zero prediction.  This is not fixed by bias correction, and
scale correction only brings the result back to about zero-baseline quality,
which means the held-out value-field pattern is also weak.
```

So the problem is not primarily:

```text
bad B labels
residual AD exploding B
legacy TRUE176 label contamination
split_A bad luck only
```

It is more likely:

```text
LE value anchor / residual value-field generalization failure under the frozen
strategy and current single-geometry, low-LE-heavy fresh validation cases.
```

## Next Decisions

Do not continue by simply increasing epochs or tuning loss inside v1.2.

Useful next diagnostics or design changes should target:

- LE anchor / zero-displacement consistency.
- A value-field baseline tied more directly to `B @ q`.
- Residual branch amplitude control for low-alpha cases.
- LE normalization or case/scale weighting, especially with the high-amplitude
  case031 mixed with much lower-amplitude fresh cases.
- More informative mid/high-alpha fresh cases if the goal is to reduce
  low-target-RMS dominance.

