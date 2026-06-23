# v1.2 B@q Anchor Oracle Diagnostic

This note records the evaluation-only B@q anchor oracle diagnostic for the
query-point Abaqus route.  No model was trained, no training hyperparameters
were changed, and no old TRUE176 `LE/B` labels were used as v1.2 labels.

## Purpose

The preceding LE failure diagnostic showed that the fixed-strategy v1.2
checkpoints predict an overlarge held-out LE field on low-LE validation cases
while the query B prior remains stable.  This diagnostic asks a narrower
question:

```text
Does B_prior @ q48 provide a better LE value anchor than the current learned
LE head?
```

It also checks the data-side oracle:

```text
Does B_true @ q48 reconstruct LE_true for the fresh strict-pass compacts?
```

## Inputs

Compact list:

```text
D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt
```

Split_A checkpoint:

```text
D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_main_80_20\best.pt
```

Split_B checkpoint:

```text
D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_split_B_val44_46\best.pt
```

Diagnostic outputs:

```text
D:\IS-FEM\outputs\query_point_v1_2_training_audit\bq_anchor_oracle_diagnostic\split_A_val41_49
D:\IS-FEM\outputs\query_point_v1_2_training_audit\bq_anchor_oracle_diagnostic\split_B_val44_46
```

The script writes:

```text
bq_anchor_case_summary.csv
bq_anchor_frame_summary.csv
zero_q_anchor_summary.csv
residual_offset_summary.csv
bq_anchor_diagnostic_summary.json
```

## Results

| split | case | model_LE_rel | Btrue_q_LE_rel | Bprior_q_LE_rel | zero_pred_LE_rel | zero_q_pred_rms | current_offset_rms |
|---|---:|---:|---:|---:|---:|---:|---:|
| split_A | 41 | 9.1422 | 0.0400 | 3.4515 | 1.0000 | 7.813e-04 | 7.961e-04 |
| split_A | 49 | 4.0235 | 0.0517 | 1.6303 | 1.0000 | 7.813e-04 | 7.866e-04 |
| split_B | 44 | 9.6526 | 0.0380 | 1.3224 | 1.0000 | 9.021e-04 | 8.965e-04 |
| split_B | 46 | 7.7823 | 0.0233 | 1.5006 | 1.0000 | 9.021e-04 | 8.994e-04 |

Additional anchor values:

| split | case | target_rms | model_pred_rms | Bprior_q_pred_rms | true_offset_rms | Bprior_offset_rms | model/Bprior rel ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| split_A | 41 | 9.330e-05 | 8.597e-04 | 3.190e-04 | 3.734e-06 | 3.220e-04 | 2.6488 |
| split_A | 49 | 2.208e-04 | 9.045e-04 | 3.761e-04 | 1.142e-05 | 3.600e-04 | 2.4679 |
| split_B | 44 | 9.364e-05 | 9.180e-04 | 1.320e-04 | 3.558e-06 | 1.238e-04 | 7.2994 |
| split_B | 46 | 1.231e-04 | 9.639e-04 | 1.456e-04 | 2.872e-06 | 1.848e-04 | 5.1862 |

## Interpretation

`B_true @ q48` reconstructs `LE_true` well on all checked held-out cases:

```text
Btrue_q_LE_rel = 0.0233 to 0.0517
true_offset_rms = 2.87e-06 to 1.14e-05
```

This supports the fresh compact data contract: the loaded `q48_raw`, `LE`, and
`B` fields are mutually consistent for these cases.

`B_prior @ q48` is consistently better than the current model LE prediction:

```text
model_LE_rel / Bprior_q_LE_rel = 2.47 to 7.30
```

This strengthens the diagnosis that the learned LE value head is not anchored to
the stable query B prior.  The model adds a large value/residual offset:

```text
current_offset_rms = 7.87e-04 to 8.99e-04
```

That offset is close to the previous LE failure diagnostic's predicted offset
scale and far larger than the true `LE_true - B_true @ q48` offset.

However, `B_prior @ q48` is still not a solved LE predictor:

```text
Bprior_q_LE_rel = 1.32 to 3.45
zero_pred_LE_rel = 1.0
```

So the direct B-prior value anchor is much better than the current LE head, but
it remains worse than predicting zero on these low-LE held-out cases.  This
means the next design should not simply replace the LE head with `B_prior @ q`.
It should anchor the value field to B while also enforcing a stable zero-q and
low-alpha residual behavior.

The q=0 check is also decisive:

```text
zero_q_pred_rms = 7.81e-04 for split_A
zero_q_pred_rms = 9.02e-04 for split_B
```

Those values are close to the full-model error scale.  The current checkpoint
therefore carries a large nonzero LE field even when the raw q slice is set to
zero.  This points to a wrong zero-displacement/value anchor, not just a
directional B issue.

## Current Conclusion

Conservative route-level conclusion:

```text
Fresh q/LE/B consistency is strong.
B_prior @ q48 is a substantially better held-out value anchor than the current
LE head.
But B_prior @ q48 alone is not sufficient on low-LE validation cases.
The main failure is now localized to the LE value anchor / residual value field,
especially a nonzero q=0 output.
```

Potential next design direction, not implemented in this diagnostic:

```text
LE_hat = B_prior(point) @ q + residual(q, point)
residual(q=0, point) = 0
low-alpha residual is gated or scaled so it cannot create a large ghost LE field
```

## Validation Boundary

- No training was performed.
- No model structure, loss, learning-rate, or epoch setting was changed.
- Old TRUE176 `LE/B` values were not used as v1.2 labels.
- No `.npz`, `.odb`, `.pt`, `.pth`, checkpoint, or large output artifact should
  be committed.
