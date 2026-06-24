# Query-Point Abaqus Route Changelog

This file records the route-level history for the Abaqus real-integration-point
DeepONet/query-point training line.  Keep it updated whenever this route changes.

## v3 B-head-only capacity diagnosis - 2026-06-24 - Direct B is learnable

Purpose:

- Follow up the radial tangent-head diagnosis by turning off every coupled
  objective except the explicit tangent head.
- Ask only whether the current inputs can directly fit
  `B_local_useful_stack_hat`:

```text
input:
  q_useful_hat
  geometry_global_hat
  trunk_features_hat / query point

loss:
  ||B_hat(s, d, geometry, point) - B_local_useful_stack_hat||^2
```

- Keep this as a capacity diagnostic only.  `B_local_useful_stack_hat` is still
  the derivative label from Abaqus/exported v3 compact; this gate simply does
  not require `LE_hat` and `AD(LE_hat, q)` to be consistent.

Implementation:

- `scripts/train_v3_single_frame_radial_operator_smoke.py` now skips disabled
  losses instead of computing them with zero weight.
- The B-only runs used:

```text
--le-weight 0
--b-weight 0
--direct-b-head-weight 1.0
--radial-weight 0
```

One-frame B-only result:

```text
case031 frame09
steps = 3000
hidden = 256
sample_count = 1

B_hat_direct_rel = 0.1303068548
B_hat_direct_cos = 0.9916398525
AD_B_rel = 0.1303039491
B_model_raw_rel = 0.1310344338
```

One-case B-only result:

```text
case031, all 10 frames
steps = 3000
hidden = 256
sample_count = 10

B_hat_direct_rel = 0.1692510098
B_hat_direct_cos = 0.9856898189
AD_B_rel = 0.1692342162
B_model_raw_rel = 0.1698791385
```

10-case training-pool B-only result:

```text
cases = [19,25,31,41,43,44,45,46,49,50]
steps = 3000
hidden = 256
sample_count = 100

B_hat_direct_rel = 0.1821686924
B_hat_direct_cos = 0.9841534495
AD_B_rel = 0.1818507612
B_model_raw_rel = 0.1842970997

case031:
  B_hat_direct_rel = 0.2065032274
  AD_B_rel = 0.2063035220
  B_model_raw_rel = 0.2076769918
```

Interpretation:

- The direct tangent head is learnable under direct supervision.  It can fit
  one frame, one case, and the 10-case training pool to about `0.13-0.18`
  relative error with high cosine.
- This rules out the strongest version of "the B-head cannot represent the
  tangent from the current inputs."
- The earlier coupled radial run failed because the joint `LE/V`, `B_hat`,
  AD-B, and radial consistency objective did not preserve this direct B
  capacity.

Updated next-step hypothesis:

```text
The problem is not B-head capacity by itself.
The next blocker is V/B coupling:
  how to make LE_hat values and B_hat tangents consistent
  without destroying the directly learnable tangent field.
```

Boundary:

- This is not a final model and not a held-out generalization result.
- `LE_hat` is intentionally not trained in this diagnostic, so `LE_rel` is not
  meaningful here.
- No checkpoint or large data artifact is committed.

## v3 single-frame radial learning diagnosis - 2026-06-24 - B-head is the bottleneck

Purpose:

- Follow up the single-frame radial smoke, which ran but did not close the
  10-case training pool.
- Diagnose whether the failure is caused by the autograd/radial construction
  or by the direct tangent head itself.
- Keep this as an overfit diagnostic only: no held-out split, no data expansion,
  no checkpoint, and no final model claim.

Added / changed:

- Extended `scripts/train_v3_single_frame_radial_operator_smoke.py` with:
  `--filter-case`, `--filter-frame`, `B_hat_direct_rel`,
  `B_hat_direct_cos`, and optional `--direct-b-head-weight`.
- The new direct tangent metrics compare the model's explicit
  `B_hat(s,d,geometry,point)` head against `B_local_useful_stack_hat`, before
  using autograd to form `AD-B`.

One-frame overfit command:

```powershell
py -3 scripts\train_v3_single_frame_radial_operator_smoke.py `
  --compact-list D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\single_frame_radial_learning_diagnosis_case031_frame09 `
  --filter-case 31 `
  --filter-frame 9 `
  --steps 2000 `
  --eval-every 500 `
  --hidden 128 `
  --lr 3e-4 `
  --sample-batch 1 `
  --le-point-batch 128 `
  --ad-point-batch 16 `
  --eval-point-batch 16 `
  --le-weight 1.0 `
  --b-weight 0.1 `
  --direct-b-head-weight 1.0 `
  --radial-weight 0.1
```

One-frame result:

```text
sample_count = 1
filter_case = [31]
filter_frame = [9]

train_LE_local_stack_rel = 0.0308176912
train_AD_B_local_useful_hat_rel = 0.6214236617
train_AD_B_local_useful_hat_cos = 0.7985625863
B_hat_direct_rel = 0.6214291453
B_hat_direct_cos = 0.7985562086
B_model_raw_rel = 0.6221962571
```

One-case / 10-frame command:

```powershell
py -3 scripts\train_v3_single_frame_radial_operator_smoke.py `
  --compact-list D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\single_frame_radial_learning_diagnosis_case031_10frame `
  --filter-case 31 `
  --steps 2000 `
  --eval-every 500 `
  --hidden 128 `
  --lr 3e-4 `
  --sample-batch 4 `
  --le-point-batch 128 `
  --ad-point-batch 16 `
  --eval-point-batch 16 `
  --le-weight 1.0 `
  --b-weight 0.1 `
  --direct-b-head-weight 1.0 `
  --radial-weight 0.1
```

One-case result:

```text
sample_count = 10
filter_case = [31]

train_LE_local_stack_rel = 0.1983168423
train_AD_B_local_useful_hat_rel = 0.5744041800
train_AD_B_local_useful_hat_cos = 0.8311612010
B_hat_direct_rel = 0.5744073987
B_hat_direct_cos = 0.8311581612
B_model_raw_rel = 0.5756050348
```

Interpretation:

- One-frame `LE` value overfit is possible (`LE_rel` about `0.031`), so the
  value path is not the immediate blocker.
- In both diagnostics, `B_hat_direct_rel` and `AD_B_rel` are essentially equal.
  That means the autograd/radial construction is not adding a separate large
  error; the explicit `B_hat` tangent head itself is not learning the target
  tangent well enough.
- The one-case path improves `B` slightly but leaves value worse than one-frame,
  so the current naive single-frame radial network still does not close even a
  one-case amplitude path.

Boundary:

- This is still not a held-out result.
- No old true176 labels are used.
- No large data artifact or checkpoint is committed.
- The next modeling step should focus on the tangent/B representation, not on
  debugging raw-B back projection or adding more data.

## v3 single-frame radial operator smoke - 2026-06-24 - Deployable interface runs, learning not closed

Purpose:

- Move one step away from the successful `hybrid-cubic` per-case oracle.
- Test the intended deployment interface on the current 10-case training pool:

```text
input:
  one current q_useful_hat frame
  geometry_global_hat
  trunk_features_hat / query point

output:
  LE_local_stack

AD target:
  dLE_local_stack / dq_useful_hat
```

- Keep this as a training-pool smoke only: no held-out split, no checkpoint,
  and no final performance claim.

Added:

- `scripts/train_v3_single_frame_radial_operator_smoke.py`

Single-frame radial form:

```text
s = ||q_useful_hat||
d = stop_gradient(q_useful_hat / (s + eps))
q_perp = q_useful_hat - s * d

LE_hat(q, point) = V_hat(s, d, geometry, point)
                 + B_hat(s, d, geometry, point) @ q_perp
```

The radial consistency loss enforces:

```text
dV_hat/ds ~= B_hat @ d
```

so that autograd at the current single frame can recover both the radial and
transverse tangent contributions.

Command:

```powershell
py -3 scripts\train_v3_single_frame_radial_operator_smoke.py `
  --compact-list D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\single_frame_radial_operator_smoke_10case `
  --steps 600 `
  --eval-every 200 `
  --hidden 128 `
  --lr 1e-4 `
  --sample-batch 8 `
  --le-point-batch 128 `
  --ad-point-batch 8 `
  --eval-point-batch 16 `
  --le-weight 1.0 `
  --b-weight 0.1 `
  --radial-weight 0.1
```

Result after 600 steps:

```text
overall:
  train_LE_local_stack_rel = 1.0012407303
  train_AD_B_local_useful_hat_rel = 0.9635614157
  train_AD_B_local_useful_hat_cos = 0.3446145356
  B_model_raw_rel = 0.9650352597

case031:
  LE_rel = 1.0011492968
  AD_B_rel = 0.9606473446
  B_model_raw_rel = 0.9617090225
```

Interpretation:

- The single-frame/radial code path is executable end to end:
  loader, normalized `q_useful_hat`, geometry/trunk inputs, `LE_local_stack`
  output, autograd `AD-B`, and raw-B back projection all run.
- The model does not yet reproduce the current training pool.  `LE_rel` stays
  near `1.0`, while `AD_B_rel` only improves mildly from `1.0` to about `0.96`.
- This is therefore a useful negative smoke result: the final single-frame
  interface is now testable, but the current naive radial network has not
  replaced the per-case `hybrid-cubic` oracle.

Boundary:

- No `case_id`, `q_mean(case)`, or multi-frame case anchor is used as an input.
- No old true176 labels are used.
- No checkpoint or large data artifact is written to the repository.
- This is not a held-out generalization result and not a final v3 model.

## v3 CSS8 hybrid-cubic anchor prototype - 2026-06-24 - Value and tangent close together

Purpose:

- Combine the value accuracy of `affine-quadratic` with the derivative accuracy
  of `tangent-cubic`.
- Verify, on the current 10-case training pool, whether a single differentiable
  anchor can keep both `LE_local_stack` and `AD-B` low.
- Keep this as a training-pool objective gate only: no held-out split, no
  checkpoint, and no generalization claim.

Added / changed:

- Added read-only hybrid oracle:
  `scripts/audit_v3_hybrid_anchor_oracle.py`
- Extended `scripts/train_v3_css8_formal_objective_prototype.py` with:
  `--anchor-mode hybrid-cubic`

Hybrid anchor:

```text
s = <q_useful_hat - q_mean(case), q_dir(case)>
q_perp = q_useful_hat - q_mean(case) - s * q_dir(case)

V(s) = affine-quadratic value path
B(s) = tangent-cubic tangent path

LE_hybrid(q) = V(s) + B(s) @ q_perp
```

Important implementation detail:

```text
V(s) must depend only on scalar s, not on full q.
Otherwise autograd adds an extra transverse B_mean term and AD-B is wrong.
```

Oracle command:

```powershell
py -3 scripts\audit_v3_hybrid_anchor_oracle.py `
  --compact-list D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_compact_list.txt `
  --focus-case 31 `
  --tangent-degree 3 `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\hybrid_anchor_oracle_10case
```

Oracle result:

```text
pooled affine-quadratic LE_rel = 0.0098883569
pooled tangent-poly AD_B_rel = 0.0021637615
pooled hybrid LE_rel = 0.0098883781
pooled hybrid AD_B_rel = 0.0021643945

case031:
  hybrid LE_rel = 0.0099147606
  hybrid AD_B_rel = 0.0103414782
  hybrid AD_B_cos = 0.9999465255
```

Autograd prototype command:

```powershell
py -3 scripts\train_v3_css8_formal_objective_prototype.py `
  --compact-list D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\formal_objective_prototype_10case_hybrid_cubic `
  --steps 1600 `
  --eval-every 400 `
  --hidden 128 `
  --lr 1e-5 `
  --case-batch 4 `
  --frame-batch 10 `
  --le-point-batch 128 `
  --ad-point-batch 8 `
  --eval-point-batch 16 `
  --anchor-mode hybrid-cubic `
  --tangent-anchor-degree 3
```

Prototype result:

```text
train_LE_local_stack_rel = 0.0098776286
train_AD_B_local_useful_hat_rel = 0.0021653324
train_AD_B_local_useful_hat_cos = 0.9999976754
B_model_raw_rel = 0.0022233343

case031:
  LE_rel = 0.0099039581
  AD_B_rel = 0.0103416536
  B_model_raw_rel = 0.0104684727

non-case031 latest:
  LE_rel roughly 0.000091 to 0.001091
  AD_B_rel roughly 0.000077 to 0.000203
```

Interpretation:

- The v3 training-pool objective gate now closes both value and tangent on the
  current 10-case pool.
- `case031` is no longer a value or AD-B outlier under the hybrid anchor:
  it remains the hardest case, but its metrics are now in the intended range.
- This validates the coordinate-consistent v3 contract plus scalar-path
  amplitude/tangent anchor design on the training pool.

Boundary:

- The hybrid coefficients are still fitted per case from the current training
  pool.  This is not a held-out generalization result.
- The next step is to replace these per-case closed-form coefficients with a
  model-predictable / geometry-and-direction-conditioned anchor, or run a very
  cautious leave-one-case diagnostic that explicitly acknowledges this
  per-case-oracle limitation.

## v3 CSS8 tangent-amplitude oracle - 2026-06-24 - B(s) explains case031 tangent

Purpose:

- Follow up the affine-quadratic value-anchor result.
- Test whether the remaining `case031` AD-B error is caused by using a constant
  `B_mean(case,point)` tangent on a high-amplitude scalar path.
- Keep this as a training-pool tangent/objective diagnostic: no held-out split,
  no checkpoint, and no generalization claim.

Added / changed:

- Added read-only tangent oracle:
  `scripts/audit_v3_tangent_amplitude_oracle.py`
- Extended `scripts/train_v3_css8_formal_objective_prototype.py` with:
  `--anchor-mode tangent-cubic`
- Added `--tangent-anchor-degree`, default `3`.

Tangent path model:

```text
s = <q_useful_hat - q_mean(case), q_dir(case)>
q_perp = q_useful_hat - q_mean(case) - s * q_dir(case)

B_anchor(s) = B0(case,point)
            + B1(case,point) * s
            + B2(case,point) * s^2
            + B3(case,point) * s^3

LE_path(s) = C(case,point)
           + integral( B_anchor(s) @ q_dir(case), ds )

LE_anchor(q) = LE_path(s) + B_anchor(s) @ q_perp
```

On scalar paths, `q_perp ~= 0`, and autograd sees:

```text
dLE_anchor/dq_useful_hat = B_anchor(s)
```

Oracle command:

```powershell
py -3 scripts\audit_v3_tangent_amplitude_oracle.py `
  --compact-list D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_compact_list.txt `
  --focus-case 31 `
  --max-degree 3 `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\tangent_amplitude_oracle_10case
```

Key oracle result:

```text
case031:
  q_perp_rel_max = 1.6638293738e-07
  Bmean_AD_B_rel = 0.1120560463
  B_poly_deg1_AD_B_rel = 0.0310015948
  B_poly_deg2_AD_B_rel = 0.0173812051
  B_poly_deg3_AD_B_rel = 0.0103404544
  B_poly_deg3_integrated_LE_rel = 0.0178418577
  secant_Bproj_rel_mean = 0.0321930186

non-case031:
  B_poly_deg3_AD_B_rel is roughly 2e-5 to 1.85e-4
```

Anchor-only tangent-cubic prototype:

```powershell
py -3 scripts\train_v3_css8_formal_objective_prototype.py `
  --compact-list D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\formal_objective_prototype_10case_tangent_cubic_anchor_only `
  --steps 0 `
  --eval-every 400 `
  --hidden 128 `
  --lr 1e-5 `
  --case-batch 4 `
  --frame-batch 10 `
  --le-point-batch 128 `
  --ad-point-batch 8 `
  --eval-point-batch 16 `
  --anchor-mode tangent-cubic `
  --tangent-anchor-degree 3
```

Anchor-only result:

```text
train_LE_local_stack_rel = 0.0178092010
train_AD_B_local_useful_hat_rel = 0.0021637613
train_AD_B_local_useful_hat_cos = 0.9999976158
B_model_raw_rel = 0.0022217832

case031:
  LE_rel = 0.0178418718
  AD_B_rel = 0.0103404541
```

1600-step tangent-cubic prototype:

```text
train_LE_local_stack_rel = 0.0174125843
train_AD_B_local_useful_hat_rel = 0.0022670364
train_AD_B_local_useful_hat_cos = 0.9999972582
B_model_raw_rel = 0.0023269854

case031:
  LE_rel = 0.0174451172
  AD_B_rel = 0.0103776203
```

Comparison to affine-quadratic value anchor:

```text
affine-quadratic:
  pooled LE_rel ~= 0.00988
  pooled AD_B_rel ~= 0.02368
  case031 LE_rel ~= 0.00991
  case031 AD_B_rel ~= 0.11205

tangent-cubic:
  pooled LE_rel ~= 0.01741
  pooled AD_B_rel ~= 0.00227
  case031 LE_rel ~= 0.01745
  case031 AD_B_rel ~= 0.01038
```

Interpretation:

- The remaining `case031` AD-B error is not random.  It is largely explained by
  tangent variation along scalar amplitude `s`.
- `B(s)` cubic fitting reduces `case031` AD-B from about `0.112` to about
  `0.010`.
- The tangent-cubic anchor trades value accuracy for derivative accuracy:
  LE is worse than affine-quadratic, while AD-B is much better.
- Residual training did not remove this tradeoff at the current settings.

Current v3 objective state:

```text
affine-quadratic = value-accurate anchor
tangent-cubic    = derivative-accurate anchor
```

Next recommended gate:

- Build a hybrid or constrained objective that keeps the affine-quadratic value
  closure while adding tangent-cubic `B(s)` supervision/regularization.
- Do not run held-out split claims until the training-pool anchor can keep both:

```text
LE_rel near 0.01
AD_B_rel near 0.002--0.01
```

## v3 CSS8 affine-quadratic anchor prototype - 2026-06-24 - Value map closes on 10-case pool

Purpose:

- Follow up the `case031` amplitude/nonlinearity audit.
- Test whether the remaining high-amplitude `case031` LE error is primarily a
  value-anchor issue.
- Keep this as a training-pool objective prototype only: no held-out split, no
  checkpoint, and no generalization claim.

Added / changed:

- Added read-only oracle:
  `scripts/audit_v3_affine_quadratic_anchor_oracle.py`
- Extended `scripts/train_v3_css8_formal_objective_prototype.py` with:
  `--anchor-mode affine-quadratic`
- Added `--steps 0` support so the closed-form anchor can be evaluated without
  residual training.

Anchor:

```text
s = <q_useful_hat - q_mean(case), q_dir(case)>

LE_hat =
  LE_mean(case,point)
  + B_mean(case,point) @ (q_useful_hat - q_mean(case))
  + C0(case,point)
  + C1(case,point) * s
  + C2(case,point) * s^2
  + R(q, geometry, trunk)
  - R(0, geometry, trunk)
```

The `C0/C1/C2` coefficients are fitted in closed form per case from the current
training pool.  They are part of this prototype anchor, not a held-out
generalization result.

Oracle command:

```powershell
py -3 scripts\audit_v3_affine_quadratic_anchor_oracle.py `
  --compact-list D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_compact_list.txt `
  --focus-case 31 `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\affine_quadratic_anchor_oracle_10case
```

Oracle result:

```text
pooled linear B_mean @ q LE_rel = 0.2930525351
pooled affine LE_rel = 0.0973463091
pooled affine-quadratic LE_rel = 0.0098883569

pooled affine AD_B_rel = 0.0236806057
pooled affine-quadratic AD_B_rel = 0.0236802581

case031:
  affine LE_rel = 0.0976020099
  affine-quadratic LE_rel = 0.0099147393
  affine AD_B_rel = 0.1120560463
  affine-quadratic AD_B_rel = 0.1120542138
```

Anchor-only prototype command:

```powershell
py -3 scripts\train_v3_css8_formal_objective_prototype.py `
  --compact-list D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\formal_objective_prototype_10case_affine_quadratic_anchor_only `
  --steps 0 `
  --eval-every 400 `
  --hidden 128 `
  --lr 1e-5 `
  --case-batch 4 `
  --frame-batch 10 `
  --le-point-batch 128 `
  --ad-point-batch 8 `
  --eval-point-batch 16 `
  --anchor-mode affine-quadratic
```

Anchor-only result:

```text
train_LE_local_stack_rel = 0.0098883007
train_AD_B_local_useful_hat_rel = 0.0236802567
train_AD_B_local_useful_hat_cos = 0.9997195601
B_model_raw_rel = 0.0240008552

case031:
  LE_rel = 0.0099146822
  AD_B_rel = 0.1120542139
```

1600-step prototype result:

```text
train_LE_local_stack_rel = 0.0098798191
train_AD_B_local_useful_hat_rel = 0.0236802325
train_AD_B_local_useful_hat_cos = 0.9997195005
B_model_raw_rel = 0.0240008291

case031:
  LE_rel = 0.0099061737
  AD_B_rel = 0.1120538861

non-case031 latest LE_rel range:
  about 0.000059 to 0.000453
```

Interpretation:

- The `case031` value-map problem is largely closed by a scalar
  amplitude-aware anchor.
- The full `C0 + C1*s + C2*s^2` correction is required.  A bare `C2*s^2`
  correction was too weak (`case031 LE_rel` only reached about `0.0752`).
- The anchor does not hide an AD-B regression: pooled AD-B stays at about
  `0.02368`, and `case031` AD-B stays at about `0.11205`.
- Residual training adds little after this anchor; the closed-form anchor
  already explains the current 10-case value field.

Remaining blocker:

- `case031` still has high AD-B error (`~0.112`) even when its LE value error is
  low.  The next issue is therefore q-dependent B / tangent modeling, not LE
  value reconstruction.

Next recommended gate:

- Add a tangent-aware audit/prototype for the affine-quadratic anchor, e.g.
  inspect whether `B_local_useful_stack_hat` is also a scalar-amplitude function
  on `case031`, and prototype `B(s,point)` or derivative-consistent cubic value
  anchors before held-out split claims.

## v3 CSS8 case031 nonlinearity audit - 2026-06-24 - Amplitude path isolated

Purpose:

- Pause before held-out splits or data expansion.
- Audit why `case031` remains difficult after the v3 formal objective
  prototype.
- Determine whether the remaining `case031` error is a compact/label issue,
  q-direction abnormality, large-amplitude nonlinear path, or value-anchor
  design issue.
- Keep this as a read-only data/objective diagnostic: no training, no split,
  no checkpoint, and no model-performance claim.

Added:

- `scripts/audit_v3_case031_nonlinearity.py`

Command:

```powershell
py -3 scripts\audit_v3_case031_nonlinearity.py `
  --compact-list D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_compact_list.txt `
  --focus-case 31 `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\case031_nonlinearity_audit
```

Outputs:

```text
D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\case031_nonlinearity_audit\v3_case031_nonlinearity_summary.json
D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\case031_nonlinearity_audit\v3_case031_nonlinearity_case_summary.csv
D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\case031_nonlinearity_audit\v3_case031_nonlinearity_frame_manifest.csv
```

Key result:

```text
case_count = 10
focus_case = 31

case031 ranks:
  q_norm_mean_desc = 1
  LE_rms_mean_desc = 1
  Bmean_at_q_LE_rel_desc = 1
  affine_Bmean_LE_rel_desc = 1

case031 q_norm_mean = 0.2080905983
pool q_norm_mean_median = 0.0025180054
case031 q_norm_mean / pool median = 82.6410454331

case031 LE_rms_mean = 0.0128052995
pool LE_rms_mean_median = 0.0001958019
case031 LE_rms_mean / pool median = 65.3992597910

case_internal_q_direction_cos_min = 0.9999999999999998
case_internal_q_direction_cos_mean = 1.0

B_mean(case,point) @ q:
  case031 LE_rel = 0.2938258832

LE_mean + B_mean @ (q - q_mean):
  case031 LE_rel = 0.0976020099

quadratic amplitude correction over affine:
  case031 LE_rel = 0.0099147393

secant_Bmean_rel_mean = 0.3333519672
secant_Bmean_rel_max = 0.7711158134
```

Interpretation:

- `case031` is not merely a random hard case.  It is the largest-amplitude and
  largest-LE case in the 10-case v3 pool by a large margin.
- Its q path is essentially a scalar amplitude path:
  all frames share the same direction, with varying amplitude.
- The affine anchor already removes a large offset error
  (`0.2938 -> 0.0976`), but a scalar quadratic correction reduces the same
  value reconstruction problem to about `0.0099`.
- This isolates the remaining issue as amplitude-dependent value-map
  nonlinearity / anchor design, not v3 compact contract failure, not old
  TRUE176 label leakage, and not q-direction inconsistency.

Next recommended gate:

- Prototype a conservative amplitude-aware value anchor before held-out split
  claims, for example a case-local scalar correction along the case q direction:

```text
s = <q_useful_hat - q_mean, q_dir_case>
LE_hat =
  LE_mean(case,point)
  + B_mean(case,point) @ (q_useful_hat - q_mean(case))
  + C0(case,point)
  + C1(case,point) * s
  + C2(case,point) * s^2
  + residual(q, geometry, trunk)
  - residual(q_mean or zero anchor)
```

- Keep per-case/relative LE and B scaling and explicit AD-B supervision.
- Do not expand the dataset or judge held-out generalization until the current
  10-case training pool is stable with this stronger value anchor.

## v3 CSS8 formal objective prototype - 2026-06-24 - Affine anchor and per-case scaling

Purpose:

- Move from diagnostic smoke scripts toward a formal v3 objective prototype.
- Test whether the 10-case training pool can be stably overfit using:
  train-only normalization, per-case relative LE/B scaling, AD-B supervision,
  and a stronger value anchor.
- Keep this as a prototype only: no held-out split, no checkpoint, no formal
  generalization claim.

Added:

- `scripts/train_v3_css8_formal_objective_prototype.py`

Objective prototype:

```text
LE_hat =
  LE_mean(case,point)
  + B_mean(case,point) @ (q_useful_hat - q_mean(case))
  + R(q_useful_hat, geometry_global_hat, trunk_features_hat)
  - R(0, geometry_global_hat, trunk_features_hat)
```

The objective uses per-case/relative scaling for both LE and B losses:

```text
LE loss scale = rms(LE_local_stack for that case)
B  loss scale = rms(B_local_useful_stack_hat for that case)
```

Affine-anchor command:

```powershell
py -3 scripts\train_v3_css8_formal_objective_prototype.py `
  --compact-list D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\formal_objective_prototype_10case_affine `
  --steps 1600 `
  --eval-every 400 `
  --hidden 128 `
  --lr 1e-5 `
  --case-batch 4 `
  --frame-batch 10 `
  --le-point-batch 128 `
  --ad-point-batch 8 `
  --eval-point-batch 16 `
  --anchor-mode affine
```

Affine-anchor result with frozen B anchor:

```text
initial train_LE_local_stack_rel = 0.0973462090
initial train_AD_B_local_useful_hat_rel = 0.0236806050

latest step = 1600
train_LE_local_stack_rel = 0.0958199650
train_AD_B_local_useful_hat_rel = 0.0237608757
train_AD_B_local_useful_hat_cos = 0.9997175932
B_model_raw_rel = 0.0240830462

case031:
  train_LE_local_stack_rel = 0.0960717276
  train_AD_B_local_useful_hat_rel = 0.1120867655
```

Comparison with the previous value anchor:

```text
B_mean(case,point) @ q_useful_hat:
  case031 LE_rel ~= 0.2938258832

LE_mean + B_mean @ (q - q_mean):
  case031 LE_rel ~= 0.0976 initially
```

Thus the affine anchor fixes a large part of the case031 value error before
residual training begins.

Trainable-B-anchor command:

```powershell
py -3 scripts\train_v3_css8_formal_objective_prototype.py `
  --compact-list D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\formal_objective_prototype_10case_affine_trainableB `
  --steps 1600 `
  --eval-every 400 `
  --hidden 128 `
  --lr 1e-5 `
  --case-batch 4 `
  --frame-batch 10 `
  --le-point-batch 128 `
  --ad-point-batch 8 `
  --eval-point-batch 16 `
  --anchor-mode affine `
  --no-freeze-b-anchor
```

Trainable-B-anchor result:

```text
latest step = 1600
train_LE_local_stack_rel = 0.0934077054
train_AD_B_local_useful_hat_rel = 0.0237150602
train_AD_B_local_useful_hat_cos = 0.9997186661
B_model_raw_rel = 0.0240362827

case031:
  train_LE_local_stack_rel = 0.0936552733
  train_AD_B_local_useful_hat_rel = 0.1120703891

9/10 non-case031 cases:
  train_LE_local_stack_rel ranges roughly 0.0019--0.0091
  train_AD_B_local_useful_hat_rel remains roughly 0.0017--0.0118
```

Interpretation:

- The formal-objective prototype confirms the next objective direction:
  per-case/relative scaling plus affine value anchor.
- Low-amplitude cases are no longer sacrificed.
- A trainable B anchor improves the 9 non-case031 cases substantially while
  keeping AD-B stable.
- `case031` remains a hard case even with affine anchor and trainable B anchor:
  its LE improves from ~0.294 to ~0.094, but does not fall to the 0.001--0.01
  range of the other cases.
- The next issue is therefore not compact data coverage.  It is `case031`
  value-map nonlinearity / q-dependent anchor / amplitude-path behavior.

Next recommended gate:

- Add a q-dependent or small nonlinear value anchor for the v3 formal objective,
  or run a focused case031 amplitude/path audit before held-out splits.
- Do not judge v3 held-out generalization until the 10-case training pool is
  stable under the formal objective.

## v3 CSS8 overfit diagnosis - 2026-06-24 - Case scale and value-anchor audit

Purpose:

- Pause before adding more data or running held-out splits.
- Diagnose why the 10-case v3 training-set smoke showed uneven
  `LE_local_stack` overfit.
- Separate three possible causes:
  data/contract issue, value-anchor weakness, and loss-scale imbalance.

Added:

- `scripts/diagnose_v3_css8_overfit_cases.py`
- `--loss-scale-mode {global,per-case}` diagnostic option in
  `scripts/train_v3_css8_multi_case_overfit_smoke.py`.

Per-case oracle diagnostic command:

```powershell
py -3 scripts\diagnose_v3_css8_overfit_cases.py `
  --compact-list D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_compact_list.txt `
  --smoke-case-history D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_overfit_smoke_10case_lr1e5\v3_multi_case_overfit_case_history.csv `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\overfit_case_diagnosis_10case
```

Initial diagnostic result:

```text
Bmean_at_q_LE_rel_min = 0.0230911508
Bmean_at_q_LE_rel_median = 0.0365142219
Bmean_at_q_LE_rel_max = 0.2938258832
Bmean_anchor_bad_case_ids = [31]
smoke_degraded_from_Bmean_anchor_case_ids =
  [19,25,41,43,44,45,46,49,50]
```

Interpretation:

- For 9/10 cases, the closed-form value anchor
  `B_mean(case,point) @ q_useful_hat` already explains `LE_local_stack` to
  roughly 2--5 percent relative error.
- `case031` is the only clear Bmean/value-anchor hard case
  (`Bmean_at_q_LE_rel ~= 0.294`).
- The previous global-scale 10-case residual smoke degraded many low-amplitude
  cases from a good anchor to very poor relative LE.  That points to loss-scale
  imbalance/residual optimization, not a v3 compact contract failure.

Per-case loss-scale capacity smoke:

```powershell
py -3 scripts\train_v3_css8_multi_case_overfit_smoke.py `
  --compact-list D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_overfit_smoke_10case_percase_scale `
  --steps 1200 `
  --eval-every 300 `
  --hidden 128 `
  --lr 1e-5 `
  --case-batch 4 `
  --frame-batch 10 `
  --le-point-batch 128 `
  --ad-point-batch 8 `
  --eval-point-batch 16 `
  --loss-scale-mode per-case
```

Per-case loss-scale result:

```text
smoke_final_LE_rel_min = 0.0208749175
smoke_final_LE_rel_median = 0.0290177781
smoke_final_LE_rel_max = 0.2798568606
smoke_degraded_from_Bmean_anchor_case_ids = []

case031 remains the worst:
  Bmean_at_q_LE_rel = 0.2938258832
  smoke_final_LE_rel = 0.2798568606
  smoke_final_AD_B_rel = 0.1122235432
```

Representative latest per-case values:

```text
case019: LE_rel=0.0308, AD_B_rel=0.0134
case025: LE_rel=0.0209, AD_B_rel=0.0072
case031: LE_rel=0.2799, AD_B_rel=0.1122
case041: LE_rel=0.0339, AD_B_rel=0.0056
case043: LE_rel=0.0300, AD_B_rel=0.0066
case044: LE_rel=0.0336, AD_B_rel=0.0056
case045: LE_rel=0.0275, AD_B_rel=0.0059
case046: LE_rel=0.0226, AD_B_rel=0.0056
case049: LE_rel=0.0242, AD_B_rel=0.0062
case050: LE_rel=0.0280, AD_B_rel=0.0070
```

Current conclusion:

- The v3 10-case training-set issue is not primarily "need more data".
- The v3 compact contract and multi-case loader/AD path are usable.
- Most cases can be held near the value-anchor oracle when the diagnostic loss
  uses per-case scaling.
- The global-scale smoke failed because low-amplitude cases were sacrificed by
  the residual/shared loss scale.
- `case031` is a separate hard case where the simple `B_mean @ q` value anchor
  is intrinsically weak.

Next recommended gate:

- Do not start held-out split yet.
- Do not add data just to fix this symptom.
- First move the formal v3 training objective toward train-only per-case or
  relative LE scaling, and add a better value anchor for cases like `case031`
  before interpreting split performance.

## v3 CSS8 multi-case smoke - 2026-06-24 - Loader, shape, and AD gate

Purpose:

- Move from the one-case v3 loader/overfit smoke to a multi-case smoke on the
  same standard-operator compact contract.
- Verify that multiple v3 CSS8 compacts can be loaded together, normalized over
  the selected smoke pool, batched by case/frame/point, and differentiated with
  respect to `q_useful_hat`.
- Keep this as a loader/contract/AD smoke, not formal training and not a
  held-out generalization result.

Added:

- `scripts/train_v3_css8_multi_case_overfit_smoke.py`

Contract fix found by the multi-case gate:

- The first 10-case attempt failed before training because
  `geometry_global_hat` was not cross-case shape-consistent:

```text
case019/case025/case031 geometry_global_hat = [154]
case041+ geometry_global_hat = [157]
```

- Root cause: older compacts stored constant `shape4` as `[4]`, while newer
  compacts stored frame-aligned `shape4` as `[N,4]`.  The v3 builder used
  `shape4[0]`, which turned old `[4]` values into a scalar and dropped three
  geometry components.
- `scripts/build_v3_css8_standard_operator_compacts.py` now normalizes both
  encodings to one fixed 4-vector before constructing `geometry_global_hat`.
- `scripts/audit_v3_css8_standard_operator_compact.py` now reports cross-case
  shape consistency for key model-visible tensors and fails strict audit if
  they are inconsistent.

Rebuilt/audited compact result:

```text
compact_count = 10
strict_pass_count = 10
strict_fail_count = 0
strict_pass = true

cross_case_shape_consistent:
  q_useful_hat = true
  geometry_global_hat = true
  trunk_features_hat = true
  LE_local_stack = true
  B_local_useful_stack_hat = true

geometry_global_hat shape = [157] for all 10 cases
```

Smoke model form:

```text
LE_hat =
  B_prior(case, point) @ q_useful_hat
  + R(q_useful_hat, geometry_global_hat, trunk_features_hat)
  - R(0, geometry_global_hat, trunk_features_hat)
```

`B_prior(case, point)` is initialized from each case's mean
`B_local_useful_stack_hat` and frozen by default.  This is a smoke-test anchor,
not a final model architecture.

Three-case command:

```powershell
py -3 scripts\train_v3_css8_multi_case_overfit_smoke.py `
  --compact-list D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_compact_list.txt `
  --case-ids 041,045,050 `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_overfit_smoke_3case_lr1e5 `
  --steps 800 `
  --eval-every 200 `
  --hidden 96 `
  --lr 1e-5 `
  --case-batch 3 `
  --frame-batch 10 `
  --le-point-batch 128 `
  --ad-point-batch 8 `
  --eval-point-batch 16
```

Three-case result:

```text
case_ids = [41,45,50]
case_count = 3
frame_count_per_case = 10
point_count = 128
q_useful_hat_dim = 42
geometry_global_hat_dim = 157
trunk_features_hat_dim = 48
b_prior_trainable = false

initial train_LE_local_stack_rel = 0.0374424942
initial train_AD_B_local_useful_hat_rel = 0.0026580677
initial train_AD_B_local_useful_hat_cos = 0.9999965429

best/latest step = 800
train_LE_local_stack_rel = 0.0256929491
train_AD_B_local_useful_hat_rel = 0.0035731499
train_AD_B_local_useful_hat_cos = 0.9999935627
B_model_raw_rel = 0.0036204634
```

Ten-case command:

```powershell
py -3 scripts\train_v3_css8_multi_case_overfit_smoke.py `
  --compact-list D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_overfit_smoke_10case_lr1e5 `
  --steps 800 `
  --eval-every 200 `
  --hidden 128 `
  --lr 1e-5 `
  --case-batch 4 `
  --frame-batch 10 `
  --le-point-batch 128 `
  --ad-point-batch 8 `
  --eval-point-batch 16
```

Ten-case result:

```text
case_ids = [19,25,31,41,43,44,45,46,49,50]
case_count = 10
frame_count_per_case = 10
point_count = 128
q_useful_hat_dim = 42
geometry_global_hat_dim = 157
trunk_features_hat_dim = 48
b_prior_trainable = false

initial train_LE_local_stack_rel = 0.2930527627
initial train_AD_B_local_useful_hat_rel = 0.0236806050
initial train_AD_B_local_useful_hat_cos = 0.9997195601

best/latest step = 800
train_LE_local_stack_rel = 0.2310481668
train_AD_B_local_useful_hat_rel = 0.0303847101
train_AD_B_local_useful_hat_cos = 0.9995383620
B_model_raw_rel = 0.0308490749
```

Interpretation:

- The multi-case v3 loader/contract/AD gate is now executable on both a 3-case
  subset and the full 10-case v3 compact pool.
- The gate caught and fixed a real compact contract issue:
  `geometry_global_hat` must be dimension-stable across cases.
- The 3-case subset shows stable residual overfit without breaking AD-B.
- The 10-case run loads and differentiates the full pool, but LE overfit remains
  uneven by case.  That is a modeling/optimization signal for a later step, not
  a compact-contract failure.
- No formal split, checkpoint, tag move, or generated `.npz/.pt/.pth` artifact
  is part of this commit.
- No old TRUE176 `LE/B` labels were used as v3 labels.

## v3 CSS8 one-case smoke - 2026-06-24 - Loader and AD overfit gate

Purpose:

- Verify that the new v3 CSS8 standard-operator compact can be loaded by a
  script-local model using the intended model-visible fields:

```text
q_useful_hat + geometry_global_hat + trunk_features_hat -> LE_local_stack
```

- Verify that autograd can differentiate the model output with respect to
  `q_useful_hat` and compare against `B_local_useful_stack_hat`.
- Keep this as a one-case loader/overfit smoke, not formal training and not a
  multi-case generalization claim.

Added:

- `scripts/train_v3_css8_one_case_overfit_smoke.py`

Smoke model form:

```text
LE_hat =
  B_prior(point) @ q_useful_hat
  + R(q_useful_hat, geometry_global_hat, trunk_features_hat)
  - R(0, geometry_global_hat, trunk_features_hat)
```

`B_prior(point)` is initialized from the one-case mean
`B_local_useful_stack_hat`.  This is a smoke-test anchor so the gate checks the
v3 compact loader, normalization, AD path, and raw-B backprojection rather than
asking a tiny MLP to rediscover the finite-element tangent from scratch.

Command:

```powershell
py -3 scripts\train_v3_css8_one_case_overfit_smoke.py `
  --compact D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\case050\case050_v3_css8_standard_operator.npz `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\one_case_overfit_smoke_case050 `
  --steps 800 `
  --eval-every 200 `
  --hidden 96 `
  --ad-point-batch 16 `
  --eval-point-batch 16
```

Result:

```text
case_id = 50
frame_count = 10
point_count = 128
q_useful_hat_dim = 42
geometry_global_hat_dim = 157
trunk_features_hat_dim = 48

initial train_LE_local_stack_rel = 0.0361400433
initial train_AD_B_local_useful_hat_rel = 0.0037798325
initial train_AD_B_local_useful_hat_cos = 0.9999928474
initial B_model_raw_rel = 0.0037848030

best/latest step = 800
train_LE_local_stack_rel = 0.0062978775
train_AD_B_local_useful_hat_rel = 0.0078215189
train_AD_B_local_useful_hat_cos = 0.9999694824
B_model_raw_projected_rel = 0.0079457052
B_model_raw_rel = 0.0079470677
```

Output:

```text
D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\one_case_overfit_smoke_case050\v3_one_case_overfit_summary.json
D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\one_case_overfit_smoke_case050\v3_one_case_overfit_history.csv
```

Interpretation:

- The v3 one-case loader/contract/AD smoke passes on case050.
- The script reads `q_useful_hat`, `geometry_global_hat`,
  `trunk_features_hat`, `LE_local_stack`, and `B_local_useful_stack_hat`.
- The AD output is mapped back by
  `B_raw_hat = T_eps_to_abq_stack @ AD_B_local_hat @ T_q_raw_to_useful_hat`.
- No checkpoint is written.
- No formal model structure, split, loss, learning-rate, or epoch policy was
  changed.
- No old TRUE176 `LE/B` labels were used as v3 labels.
- No generated `.npz`, `.pt`, `.pth`, or other large data artifact was
  committed.

## v3 CSS8 standard-operator compact builder - 2026-06-24 - Contract implementation

Purpose:

- Implement the v3 CSS8 standard-operator preprocessing/postprocessing chain.
- Convert fresh/v2b Abaqus compacts into model-visible standard fields:

```text
q_useful_hat
geometry_global_hat
trunk_features_hat = ip_macro_xi + ip_local_rst + local_geometry_features_hat
  -> LE_local_stack
```

- Keep raw Abaqus fields and transforms as audit/postprocess fields, not as
  quantities the network must learn.

Added:

- `scripts/v3_css8_standard_operator_common.py`
- `scripts/build_v3_css8_standard_operator_compacts.py`
- `scripts/audit_v3_css8_standard_operator_compact.py`

Core contract:

```text
q_useful_hat = (T_q_raw_to_useful @ q48_raw) / L_ref
B_local_useful_stack_hat =
  L_ref * T_eps_from_abq_stack @ B_LE128_forward @ T_q_raw_to_useful.T
B_raw_hat =
  T_eps_to_abq_stack @ B_local_useful_stack_hat @ (T_q_raw_to_useful / L_ref)
```

Commands:

```powershell
py -3 scripts\build_v3_css8_standard_operator_compacts.py `
  --compact-list D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10 `
  --strict

py -3 scripts\audit_v3_css8_standard_operator_compact.py `
  --compact-list D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10_audit `
  --strict
```

Result:

```text
compact_count = 10
strict_pass_count = 10
strict_fail_count = 0
strict_pass = true
```

Independent audit metric ranges:

```text
q_useful_hat_transform_rel = 0.0 to 0.0
LE_local_stack_to_abq_roundtrip_rel = 1.4773780403428246e-16 to 1.777626317589404e-16
B_raw_projected_rel = 4.325447894744243e-16 to 4.648154783275927e-16
B_rigid_residual_rel = 0.00010968263851608603 to 0.0012638931647260504
Q_stack_orthonormal_max = 1.1102230246251565e-16 to 2.220446049250313e-16
Q_stack_e3_dot_g_t_min = 0.9999999999999998 to 0.9999999999999999
ip_J_hat_scaling_rel = 0.0 to 0.0
ip_invJ_hat_scaling_rel = 0.0 to 0.0
ip_detJ_hat_scaling_rel = 0.0 to 0.0
```

Interpretation:

- The v3 CSS8 compact data contract is now implemented and audited.
- This is still not a training/model-performance result.
- No model structure, loss, learning rate, epoch, or split was changed.
- No old TRUE176 `LE/B` labels were used as v3 labels.
- No generated `.npz` output files were committed.

## v3 CSS8 contract clarification - 2026-06-24 - Curved-shell standard domain

Purpose:

- Clarify the standard-operator domain for the actual CSS8 continuum-shell /
  solid-shell route.
- Remove ambiguity around whether a curved macro element should be interpreted
  as one flattened HEX8 block.
- Add an audit for CSS8 midsurface/director geometry, stack-direction local
  frame, and macro-patch trunk coordinates.

Added:

- `scripts/audit_css8_curved_shell_standard_operator_contract.py`
- `docs/query_point_v3_css8_curved_shell_standard_operator_contract.md`
- Route documentation update in
  `docs/query_point_v2_coordinate_consistent_route.md`.

Command:

```powershell
py -3 scripts\audit_css8_curved_shell_standard_operator_contract.py `
  --compact-list D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt `
  --out D:\IS-FEM\outputs\query_point_v2_standard_operator\css8_curved_shell_standard_operator_contract.json `
  --strict
```

Result:

```text
compact_count = 10
strict_css8_geometry_pass_count = 10
strict_css8_geometry_fail_count = 0
all_strict_css8_geometry_pass = true
```

Key decisions:

```text
macro_shape = piecewise_4x4_css8_parent_domain_not_single_hex8
current_ip_xi_role = local_css8_parent_r_s_t
required_trunk_coordinate = ip_macro_xi_or_equivalent_subcell_id_plus_local_r_s_t
final_local_frame = stack_director_frame_with_local_3_parallel_to_g_t
network_output_coordinate = LE_local_in_css8_stack_director_frame
```

Direct answer to the CSS8 mapping question:

```text
The macro element maps to a 4x4 piecewise CSS8 parent patch.
It does not map to one globally flattened HEX8.
For a curved solid-shell patch, each integration point carries its own
Q_stack(point), with local-3 aligned to dX/dt.
```

Measured contract gap:

```text
ip_macro_xi_not_stored_in_current_v2_standard_compacts = true
current_v2_local_frame_is_surface_normal_not_exact_stack_director = true
current_Q_rel_to_stack_Q_min = 0.015430432383667826
current_Q_rel_to_stack_Q_max = 0.024697166064443367
```

Interpretation:

- The CSS8 geometry interpretation is now explicit and audited.
- The current v2 compacts remain valid diagnostics.
- They are not the final CSS8 standard-operator compact because they lack
  `ip_macro_xi` and use the older surface-normal local frame.
- No training was run and no model-performance claim is made.

## v2l - 2026-06-24 - Standard-operator preprocessing contract

Purpose:

- Pause v2 model tuning after the v2k tangent/schedule diagnostic.
- Extract q-coordinate, strain-coordinate, and B-coordinate transforms from
  the neural-network route into an explicit preprocessing/postprocessing
  contract.
- Redefine the network-visible operator as:

```text
(q_useful, ip_xi) -> LE_local
B_local_useful = autograd(dLE_local/dq_useful)
```

Contract:

```text
Raw Abaqus:
  q48_raw, LE_abq, B_abq

Preprocess:
  q_useful = T_q_raw_to_useful @ q48_raw
  LE_local = T_eps_from_abq @ LE_abq
  B_local_useful = T_eps_from_abq @ B_abq @ T_q_raw_to_useful.T

Postprocess:
  LE_abq_hat = T_eps_to_abq @ LE_local_hat
  B_raw_hat = T_eps_to_abq @ B_local_useful_hat @ T_q_raw_to_useful
```

Added:

- `scripts/build_v2_standard_operator_compacts.py`
- `scripts/audit_v2_standard_operator_contract.py`
- `docs/query_point_v2_standard_operator_preprocessing_contract.md`

Input:

```text
D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt
```

Output:

```text
D:\IS-FEM\outputs\query_point_v2_standard_operator\multi_case_min10\standard_operator_compact_list.txt
D:\IS-FEM\outputs\query_point_v2_standard_operator\multi_case_min10\standard_operator_summary.json
```

Strict 10-case result:

```text
compact_count = 10
strict_pass_count = 10
strict_fail_count = 0
q_useful_transform_rel = 0.0 to 0.0
LE_local_to_abq_roundtrip_rel = 1.6410479539884723e-16 to 3.0740682511657065e-16
B_raw_projected_rel = 4.462975556507417e-16 to 4.652329014360367e-16
B_rigid_residual_rel = 0.00010968263851608675 to 0.0012638931647260517
```

Interpretation:

- The standard-operator preprocessing/postprocessing contract closes for all
  10 v2b fresh cases.
- This is a data/engineering contract result, not a model-performance result.
- The rigid residual is recorded but not used as a strict failure because the
  useful q-coordinate intentionally removes rigid modes.

Validation boundary:

- No training was run.
- No model structure, loss, learning rate, epoch, or split policy was changed.
- No old TRUE176 `LE/B` labels were used as v2 labels.
- No tag was moved.
- Generated `.npz` outputs remain outside git and must not be committed.

Follow-up random-geometry audit:

- Added `scripts/test_random_geometry_isoparametric_scaling.py`.
- Purpose: verify random positive-orientation Hex8 geometries under
  translation, rotation, scaling, and small distortion, rather than only fixed
  10-case fresh compacts.
- Report:

```text
D:\IS-FEM\outputs\query_point_v2_standard_operator\random_geometry_isoparametric_invariance.json
```

Result:

```text
passed = true
trials = 64
total_points = 512
detJ_min = 0.29031834735368334
ip_xi_invariance_abs = 5.384581669432009e-14
rotation_scale_J_abs = 2.6645352591003757e-15
rotation_scale_invJ_abs = 1.687538997430238e-14
rotation_scale_detJ_abs = 3.410605131648481e-13
B_scaling_abs = 1.1102230246251565e-15
B_scaling_rel = 8.403417513265492e-16
LE_scaled_similarity_abs = 1.7763568394002505e-15
T_eps_roundtrip_abq_rel = 7.168741415637943e-15
T_eps_roundtrip_local_rel = 7.060577084968476e-15
LE_local_to_abq_roundtrip_rel = 3.2594517417223477e-16
B_raw_projected_rel = 5.627036520025794e-16
B_rigid_residual_rel = 0.35443953847154996
```

Interpretation:

- The random-geometry isoparametric/scaling contract passes.
- `B_rigid_residual_rel` is informational only; the strict projected raw-B
  closure is `B_raw_projected_rel`.
- This strengthens the preprocessing/postprocessing contract but still does
  not establish model performance.

## v2k - 2026-06-24 - Tangent / training-schedule diagnostic

Purpose:

- Diagnose how AD-B should be supervised after v2j introduced a
  state/regime-dependent p75-cluster B prior.
- Compare full-output, anchor-only, residual-only, detached-anchor residual,
  and LE-warmup schedules.
- Keep this as a prototype diagnostic, not a formal model-effect claim.

Included:

- `scripts/diagnose_v2_tangent_schedule.py`
- `scripts/train_v2_formal_prototype.py` options:
  `--b-loss-target-mode`, `--training-schedule`, and `--warmup-steps`.
- `docs/query_point_v2_tangent_schedule_audit.md`
- Route documentation update in
  `docs/query_point_v2_coordinate_consistent_route.md`.

Fixed v2k setup:

```text
b_prior_mode = amp_p75_cluster
detach_b_prior_regime_weight = true
use_q_amp = true
steps = 2000
eval_every = 250
seed = 20260623
warmup_steps = 500
```

Shared anchor baseline:

- `train_LE_local_rel=4.77639102935791`
- `val_LE_local_rel=5.700374126434326`
- `train_AD_B_local_rel=0.22865082323551178`
- `val_AD_B_local_rel=0.10279608517885208`
- `val_B_model_raw_projected_rel=0.10271253436803818`

Diagnostic combinations:

- A `full_output + joint`, best combined:
  - step `1250`
  - `train_LE_local_rel=4.701291084289551`
  - `val_LE_local_rel=5.470116138458252`
  - `train_AD_B_local_rel=0.31725096702575684`
  - `val_AD_B_local_rel=0.11935402452945709`
  - `val_AD_B_local_cos=0.9963414072990417`
  - `val_B_model_raw_projected_rel=0.12048347294330597`
- B `anchor_only + joint`, best combined:
  - step `500`
  - `train_LE_local_rel=4.682076454162598`
  - `val_LE_local_rel=5.659628391265869`
  - `train_AD_B_local_rel=0.2719678580760956`
  - `val_AD_B_local_rel=0.11224554479122162`
  - `val_AD_B_local_cos=0.997394323348999`
  - `val_B_model_raw_projected_rel=0.11289863288402557`
- C `residual_only + joint`, best combined:
  - step `2000`
  - `train_LE_local_rel=4.70106315612793`
  - `val_LE_local_rel=5.620501518249512`
  - `val_AD_B_local_rel=0.12718208134174347`
  - `val_B_model_raw_projected_rel=0.12866534292697906`
- D `full_output + le_warmup_then_b`, best combined:
  - step `0`
  - `train_LE_local_rel=4.776750564575195`
  - `val_LE_local_rel=5.699275016784668`
  - `val_AD_B_local_rel=0.1027916669845581`
  - latest degrades to `val_AD_B_local_rel=0.14432035386562347`
- E `detached_anchor_plus_residual + anchor_then_residual`, best combined:
  - step `1750`
  - `train_LE_local_rel=4.696530818939209`
  - `val_LE_local_rel=5.65471887588501`
  - `val_AD_B_local_rel=0.13117122650146484`
  - `val_B_model_raw_projected_rel=0.13306614756584167`

Current interpretation:

```text
v2k diagnostic completed.
No tested tangent/schedule solves the LE/AD-B tradeoff.
The p75 state-dependent anchor remains the strongest route component.
Anchor-only B supervision is the safest current tangent/loss reference.
Full-output supervision gives the best val LE in this audit, but costs more
AD-B/raw-B accuracy.
LE warmup does not help in this diagnostic.
Model performance is not established.
```

Recommended next step:

```text
v2l should improve the state-dependent B prior itself: smoother q_amp gate,
q-direction/q-cluster-aware prior, or richer state descriptor in the B-prior
branch.  Do not simply widen the residual network yet.
```

Validation boundary:

- No old TRUE176 `LE/B` labels were used as v2 labels.
- No tag was moved.
- No `.npz`, `.odb`, `.pt`, `.pth`, checkpoint, loss-history, or generated
  output file is committed.

## v2j - 2026-06-24 - State-dependent B prior prototype audit

Purpose:

- Test the v2i diagnosis that the point-only global train-mean B prior is a
  poor LE value anchor.
- Add simple train-only state/regime-dependent B priors before widening the
  residual model or making a formal training claim.
- Preserve zero-q anchor, AD-B local metrics, and raw-B backprojection metrics.

Included:

- `scripts/audit_v2_state_dependent_b_prior.py`
- `scripts/train_v2_formal_prototype.py` options:
  `--b-prior-mode`, `--detach-b-prior-regime-weight`, and
  `--no-detach-b-prior-regime-weight`.
- `docs/query_point_v2_state_dependent_b_prior_audit.md`
- Route documentation update in
  `docs/query_point_v2_coordinate_consistent_route.md`.

Prior modes:

```text
global_mean
amp_median_cluster
amp_p75_cluster
amp_linear_interp
```

The default remains `global_mean`, preserving old behavior.

No-training anchor audit:

- `global_mean`:
  - `train_LE_local_rel=29.46612548828125`
  - `val_LE_local_rel=10.826595306396484`
  - `train_AD_B_local_rel=0.29525285959243774`
  - `val_AD_B_local_rel=0.19587740302085876`
- `amp_median_cluster`:
  - threshold `0.003093380878729739`
  - cluster sizes `40 low / 40 high`
  - `train_LE_local_rel=21.19461441040039`
  - `val_LE_local_rel=5.700911998748779`
- `amp_p75_cluster`:
  - threshold `0.005826574499032849`
  - cluster sizes `60 low / 20 high`
  - `train_LE_local_rel=4.77639102935791`
  - `val_LE_local_rel=5.700374126434326`
  - `train_AD_B_local_rel=0.22865082323551178`
  - `val_AD_B_local_rel=0.10279608517885208`
- `amp_linear_interp_detached`:
  - `train_LE_local_rel=5.709594249725342`
  - `val_LE_local_rel=5.700374126434326`

Training diagnostic, best run:

- `amp_p75_cluster`, detached regime weight, best combined at step `1250`:
  - `train_LE_local_rel=4.701291084289551`
  - `val_LE_local_rel=5.470116138458252`
  - `train_AD_B_local_rel=0.31725096702575684`
  - `val_AD_B_local_rel=0.11935402452945709`
  - `val_AD_B_local_cos=0.9963414072990417`
  - `val_B_model_raw_projected_rel=0.12048347294330597`
  - `zero_q_LE_local_rms=0.0`

Current interpretation:

```text
v2j prototype diagnostic completed.
State/regime-dependent B prior clearly improves the LE value anchor.
The p75 cluster reproduces the v2i no-training p75 baseline.
The training prototype starts from train_LE_rel≈4.78 instead of ≈29.47.
Residual training gives only small LE improvement and can degrade AD-B/raw-B.
Model performance is not established.
```

Next recommendation:

```text
Keep B_prior(point, state/regime) on the v2 route, but run another targeted
audit on regime-gate smoothness, residual/anchor training schedule, and whether
AD-B supervision should use anchor-only, detached-regime, or full-tangent
derivatives.
```

Validation boundary:

- No old TRUE176 `LE/B` labels were used as v2 labels.
- No tag was moved.
- No `.npz`, `.odb`, `.pt`, `.pth`, checkpoint, loss-history, or generated
  output file is committed.

## v2i - 2026-06-24 - Value-field correction diagnostics

Purpose:

- Diagnose why the v2 formal prototype does not learn multi-case LE
  value-field correction after v2h excluded normalization, B-prior coordinate
  mismatch, B-prior table corruption, gate suppression as the main cause, and
  single-case capacity failure.
- Keep this as a targeted diagnostic stage, not formal performance training.

Included:

- `scripts/diagnose_v2_value_field_correction.py`
- Diagnostic extensions in `scripts/train_v2_formal_prototype.py`:
  `--write-per-case-history`, `--use-amp-regime-descriptor`, decomposed
  `components()`, and per-case Bq/B-prior/residual attribution fields.
- `docs/query_point_v2_value_field_correction_diagnostic.md`
- Route documentation update in
  `docs/query_point_v2_coordinate_consistent_route.md`.

Diagnostic outputs:

```text
D:\IS-FEM\outputs\query_point_v2_design_audit\v2i_value_correction\
```

Key training-diagnostic results:

- LE-only best:
  - `train_LE_local_rel=28.903217315673828`
  - `val_LE_local_rel=10.226631164550781`
  - Interpretation: removing AD-B loss does not make the value residual learn.
- LE+B best:
  - `train_LE_local_rel=28.91905975341797`
  - `val_LE_local_rel=10.357998847961426`
  - Interpretation: AD-B loss alone is not the main blocker.
- Remove `case031` best:
  - `train_LE_local_rel=36.591793060302734`
  - `val_LE_local_rel=7.851358890533447`
  - Interpretation: removing the high-amplitude case improves held-out
    low-amplitude validation LE, but does not fix train LE.
- `--use-q-dir` best:
  - `train_LE_local_rel=28.908042907714844`
  - `val_LE_local_rel=10.261488914489746`
  - Interpretation: direction descriptors alone are insufficient.
- `--use-amp-regime-descriptor` best:
  - `train_LE_local_rel=28.912452697753906`
  - `val_LE_local_rel=10.579803466796875`
  - Interpretation: simple nonlinear amplitude descriptors alone are
    insufficient.

Value-anchor baselines:

- Global train-mean B prior:
  - `train_LE_local_rel=29.466125499590156`
  - `val_LE_local_rel=10.826594009457496`
- Clustered B prior by `q_amp` median:
  - `train_LE_local_rel=21.19461440553013`
  - `val_LE_local_rel=5.700911624442508`
  - `train_AD_B_local_rel=0.27490432440403983`
  - `val_AD_B_local_rel=0.10293348806464925`
- Clustered B prior by `q_amp` p75:
  - `train_LE_local_rel=4.77639109241935`
  - `val_LE_local_rel=5.700373146340969`
  - `train_AD_B_local_rel=0.22865080859937914`
  - `val_AD_B_local_rel=0.10279608559825179`
- Per-case train B-prior upper bound:
  - `train_LE_local_rel=0.29293951890558007`
  - `train_AD_B_local_rel=0.027328778282940253`
  - `train_B_raw_projected_rel=0.02765599599942974`

Current interpretation:

```text
v2i engineering diagnostic completed.
Model performance is not established.
The most likely blocker is not the v2 coordinate chain or normalized B metrics,
but the point-only global train-mean B prior used as a value anchor.  It is a
reasonable average derivative baseline but a poor multi-case B@q value anchor.
The residual branch is then asked to repair a large case/state-dependent value
error while preserving AD-B, and it does not.
```

Recommended next step:

```text
v2j should change the value anchor / B-prior mechanism, starting with an
auditable state- or regime-dependent B_prior_norm(point, q_amp, q_dir or
q_cluster), such as a two-cluster or amplitude-interpolated B prior.
```

Validation boundary:

- No old TRUE176 `LE/B` labels were used as v2 labels.
- No tag was moved.
- No `.npz`, `.odb`, `.pt`, `.pth`, checkpoint, loss-history, or generated
  output file is committed.

## v2h - 2026-06-23 - Formal prototype sanity / ablation audit

Purpose:

- Diagnose why v2g formal prototype execution succeeds but multi-case train LE
  remains high.
- Check normalization roundtrip, B@q baselines, v2g initialization,
  single-case overfit, B-prior freeze, and gate behavior.
- Keep this as a sanity/ablation audit, not formal training.

Included:

- `scripts/audit_v2_formal_prototype_sanity.py`
- Diagnostic switches in `scripts/train_v2_formal_prototype.py`:
  `--train-cases`, `--allow-overlap-val`, and `--freeze-b-prior-table`.
- `docs/query_point_v2_formal_prototype_sanity_audit.md`
- Route documentation update in
  `docs/query_point_v2_coordinate_consistent_route.md`.

No-training sanity output:

```text
D:\IS-FEM\outputs\query_point_v2_design_audit\v2g_sanity\sanity_summary.json
```

Key sanity results:

- Normalization roundtrip closes:
  - `train_q_roundtrip_rel=4.792623713739907e-17`
  - `train_LE_roundtrip_rel=3.441183768746357e-17`
  - `train_B_roundtrip_rel=3.787348081354616e-17`
  - `val_q_roundtrip_rel=4.0597878073265377e-17`
  - `val_LE_roundtrip_rel=3.354931535711588e-17`
  - `val_B_roundtrip_rel=3.759929863285529e-17`
- Exact B@q oracle:
  - `train_LE_local_rel=0.6407233225614739`
  - `val_LE_local_rel=0.037104433513767326`
- Train-mean B prior:
  - `train_LE_local_rel=29.466125499590156`
  - `val_LE_local_rel=10.826594009457496`
  - `train_AD_B_local_rel=0.2952528280656775`
  - `val_AD_B_local_rel=0.195877397799659`
- Normalized train-mean B prior matches physical train-mean B prior.
- v2g initialization matches the train-mean B-prior baseline.

Single-case overfit:

- Run: `case050` train/val overlap diagnostic, 2000 steps.
- Result:
  - `train_LE_local_rel=0.03293348103761673`
  - `train_AD_B_local_rel=0.0039431145414710045`
  - `train_AD_B_local_cos=0.9999921321868896`
  - `zero_q_LE_local_rms=0.0`

Light ablations:

- Freeze-B-prior best:
  - `train_LE_local_rel=28.97667121887207`
  - `val_LE_local_rel=10.584187507629395`
  - `train_AD_B_local_rel=0.5441113114356995`
  - `val_AD_B_local_rel=0.2669954299926758`
- Gate-open latest (`--gate-c 1e-9`):
  - `train_LE_local_rel=28.90526008605957`
  - `val_LE_local_rel=19.146818161010742`
  - `train_AD_B_local_rel=0.8936592936515808`
  - `val_AD_B_local_rel=0.8248840570449829`

Current interpretation:

```text
normalization bug: unlikely
B-prior normalized/physical mismatch: unlikely
B-prior table corruption: unlikely
gate suppression: unlikely as the main cause
single-case implementation capacity: passes
multi-case formal split learning: still fails
```

The narrowed issue is that the v2g residual/value branch is not yet learning a
case/state-dependent correction on the mixed multi-case train pool.  Broader v2
training is still not ready.

Validation boundary:

- No old TRUE176 `LE/B` labels were used as v2 labels.
- No tag was moved.
- No `.npz`, `.odb`, `.pt`, `.pth`, checkpoint, loss-history, or generated
  output file is committed.

## v2g - 2026-06-23 - Formal v2 prototype audit

Commit:

- This section is introduced by the `Add v2 formal prototype audit` commit.

Purpose:

- Implement the v2f formal normalization/model design as a script-local
  prototype audit.
- Keep the same 10-case v2b pool and split as v2e:
  `train_cases=[19,25,31,41,43,45,49,50]`,
  `val_cases=[44,46]`, `validation_is_overlapping=false`.
- Add train-only normalization artifacts, explicit `q_amp`, anchored zero-q,
  AD-B local evaluation, and raw-B backprojection evaluation.

Included:

- `scripts/train_v2_formal_prototype.py`
- `docs/query_point_v2_formal_prototype_audit.md`
- Route documentation update in
  `docs/query_point_v2_coordinate_consistent_route.md`.

Run:

```powershell
py -3 scripts\train_v2_formal_prototype.py `
  --compact-list D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v2_formal_prototype\split_B_min10 `
  --val-cases 44,46 `
  --steps 5000 `
  --seed 20260623 `
  --eval-every 250 `
  --use-q-amp
```

Key normalization facts:

- `normalization_train_only=true`
- `q_std_ratio_before_floor=214.35575102954596`
- `q_std_floored_count=0`
- `LE_std=[0.006498310714960098, 0.0038321511819958687,
  0.0040618302300572395, 0.004930346272885799,
  0.003132438752800226, 0.003945972304791212]`
- `LE_std_floored_count=0`
- `q_amp_rms=0.08326140530721668`
- `use_q_amp=true`
- `use_q_dir=false`

Best combined result:

- Step: `250`
- `train_LE_local_rel=28.91905975341797`
- `val_LE_local_rel=10.357998847961426`
- `val_LE_local_rmse=0.0011231731623411179`
- `val_LE_target_rms=0.000108435342554003`
- `train_AD_B_local_rel=0.3983873128890991`
- `val_AD_B_local_rel=0.24915096163749695`
- `val_AD_B_local_cos=0.9772958159446716`
- `val_B_model_raw_projected_rel=0.25252825021743774`
- `zero_q_LE_local_rms=0.0`

Latest result:

- Step: `5000`
- `train_LE_local_rel=28.909543991088867`
- `val_LE_local_rel=13.804399490356445`
- `train_AD_B_local_rel=0.41042250394821167`
- `val_AD_B_local_rel=0.29874974489212036`
- `val_AD_B_local_cos=0.9589686989784241`
- `val_B_model_raw_projected_rel=0.3056311309337616`
- `zero_q_LE_local_rms=0.0`

Current interpretation:

- The v2g formal prototype execution chain is now auditable:
  train-only normalization, `q_amp`, anchored zero-q, AD-B local metrics, and
  raw-B backprojection all run on the 10-case v2b pool.
- This is not evidence that model performance is established.
- Relative to v2e latest, best v2g reduces held-out LE relative error, but the
  train LE relative error remains high and AD-B/raw-projected B metrics degrade
  relative to v2e latest.
- The conservative conclusion is:

```text
v2g formal prototype audit executes;
model effect is not established.
```

Validation boundary:

- No old TRUE176 `LE/B` labels were used as v2 labels.
- No tag was moved.
- No `.npz`, `.odb`, `.pt`, `.pth`, checkpoint, loss-history, or generated
  output file is committed.
- Generated metrics remain under:
  `D:\IS-FEM\outputs\query_point_v2_formal_prototype\split_B_min10`.

## v1 - 2026-06-22 - First query-point Abaqus route baseline

Planned marker:

- Branch: `query-point-abaqus-route`
- Tag: `query-point-abaqus-v1`
- Base commit: `9bab6fc Add FE linear residual DeepONet baseline`

Purpose:

- Treat Abaqus-exported real integration points as the source of truth.
- Remove the fixed 128-point model-structure assumption from the new DeepONet route.
- Align every point feature row with the same row in `LE` and `B`.

Included:

- `QueryFELinearResidualDeepONet` for arbitrary `P` query points.
- Generic Sobolev trainer support for `--model-style query-fe-linear-residual`.
- Complete compact exporter for Abaqus ODB data, including `ip_keys`, `ip_xi`,
  `ip_xyz`, `ip_J`, `ip_invJ`, `ip_detJ`, `ip_frame`, and `point_features`.
- Explicit point-feature loader that prefers compact-stored point data over
  shape4 reconstruction.
- `ip_keys`-based ID features when compact keys are available.
- Optional `--le-normalization global-component` for arbitrary query-point
  inference compatibility.
- Optional `--train-point-sample-count` so query-point training can use random
  point subsets while evaluation stays on the full loaded point table.
- Smoke tests for complete compact loading, query-point training, `ip_keys`
  alignment, and dynamic point counts.

Validation:

- `py -3 -m pytest tests/smoke_test.py -q`
- `py -3 -m compileall src/macro_deeponet scripts`

Known gaps:

- `B_LE128_forward` still requires finite-difference or perturbed-ODB merge data;
  a single Abaqus ODB does not directly provide `dLE/dq48`.
- Production batch export and merge orchestration still needs a wrapper.
- Arbitrary query-point inference needs a polished entry point that builds and
  normalizes point features, then denormalizes predicted `LE`.
- Dataset split should be controlled at case/geometry level for fair MLP vs
  DeepONet comparison.

Change-note rule for future route work:

- Add one dated section here for every meaningful route change.
- Mention the commit hash, user-facing behavior change, validation command, and
  any remaining risk.

## workflow-baseline - 2026-06-22 - Route tracking toolchain

Purpose:

- Add a fixed workflow for continuing the query-point Abaqus route without
  losing track of data-contract risks.

Included:

- GitHub Actions smoke workflow for `compileall` and `pytest`.
- GitHub issue template for `v1.1 hard guards`.
- Pull request review template focused on data trustworthiness.
- ChatGPT Project instructions for long-running route review.
- Route workflow document with version plan and review gates.

Validation:

- `py -3 -m pytest tests/smoke_test.py -q`
- `py -3 -m compileall src/macro_deeponet scripts`

Known gaps:

- The `v1.1 hard guards` issue checklist is now documented, but the guards still
  need to be implemented in code.

## v1.1 - 2026-06-22 - Hard guards for data trustworthiness

Planned marker:

- Branch: `query-point-abaqus-route`
- Tag: `query-point-abaqus-v1.1`

Purpose:

- Stop bad real-data compacts from training silently when labels, point rows, or
  validation splits are inconsistent.

Included:

- Strict validation split helper with explicit `case`, `geometry`, `frame`, and
  `overlap-debug` modes.
- Generic trainer records validation split metadata in config, checkpoints, and
  training summaries.
- Point feature loader requires feature names and order to match across all
  compacts.
- `ip_keys` ID feature guard: standard TRUE176 `1..16` labels can use 4x4
  spatial IDs; nonstandard labels fall back to rank-only IDs.
- Abaqus complete compact exporter can fail on merge `q48_raw`, `LE128_base`,
  and `ip_keys` mismatch.
- Abaqus complete compact exporter can fail on IP geometry audit mismatch.
- Query-point Abaqus launcher defaults to the new route and does not enable ID
  features by default.

Validation:

- `py -3 -m pytest tests/smoke_test.py -q`
- `py -3 -m compileall src/macro_deeponet scripts`

Known gaps:

- This route still trains on the Abaqus 128-IP table and selected subsets of
  that table; full arbitrary query-point label generation remains future work.

## v1.1 follow-up - 2026-06-22 - Mark legacy split helper

Purpose:

- Make the remaining old split helper visibly legacy/debug-only so future
  scripts do not mistake its overlapping validation fallback for formal route
  validation.

Included:

- `split_indices()` now documents its historical overlapping fallback and emits
  a runtime warning when called.
- Smoke coverage checks that the legacy warning remains present.

Validation:

- `py -3 -m pytest tests/smoke_test.py -q`
- `py -3 -m compileall src/macro_deeponet scripts`

Known gaps:

- The old fixed-128-IP trainer still calls `split_indices()` for compatibility;
  formal query-point/Abaqus training should keep using `split_indices_with_meta()`.

## v1.1 follow-up - 2026-06-22 - One-case complete compact smoke audit

Purpose:

- Exercise the hard guards on a real Abaqus ODB plus matching Sobolev B compact
  before attempting multi-case query-point training.

Included:

- Exporter `--shape4` and `--shape4-json` now accept a single 4-vector and
  broadcast it across all exported frames.
- Point-feature metadata compresses repeated per-frame `ip_keys` to one `[P,3]`
  table with an explicit repeat marker, avoiding huge config/checkpoint JSON
  when all frames share the same integration-point order.

Audit result:

- Generated one training-ready complete compact from
  `sample_894100.../base/t176_s4_plus_894100_base.odb` with `--strain-field E`
  and merged `css8_shape4_nonzero_sample_894100.npz`.
- Merge/audit guards passed:
  `merge_q48_max_abs_diff=0`,
  `merge_LE128_base_max_abs_diff=0`,
  `merge_ip_keys_match=true`,
  `audit_ref_ip_xyz_vs_abaqus_coord_max_abs=4.76837158203125e-07`,
  `audit_detJ_vs_IVOL_max_abs=7.639755494892597e-10`.
- Formal `case` split correctly failed because only one case/geometry was
  available.
- A 3-epoch `overlap-debug` trainer smoke ran on 8 target IPs and recorded
  `validation_is_overlapping=true`, worst-IP metrics, and random-direction B
  metrics.

Validation:

- `py -3 -m pytest tests/smoke_test.py -q`
- `py -3 -m compileall src/macro_deeponet scripts`

Known gaps:

- This is not a formal validation run. It uses one case and `overlap-debug`;
  next training audit still needs at least two real training-ready complete
  compacts/cases for case-level or geometry-level split.

## v1.1 follow-up - 2026-06-22 - Strain-field consistency guard

Purpose:

- Prevent future training batches from silently mixing Abaqus `E` and `LE`
  labels while the arrays are still named `LE128_base` / `B_LE128_forward` for
  route compatibility.

Included:

- Complete compact exporter now writes `strain_field`, `B_label_strain_field`,
  `strain_label_key`, and `B_label_key`.
- Merge guard rejects explicit `strain_field` or `B_label_strain_field`
  mismatch between the current ODB export and merged Sobolev B compact, unless
  `--allow-merge-mismatch` is explicitly used for debug.
- Compact loader rejects mixed explicit strain fields across multiple compact
  files and records `strain_meta`.
- Generic trainer records `strain_meta` in config, checkpoints, and training
  summaries.
- Legacy compacts without strain metadata remain loadable as `unknown`, but
  future training-ready compacts should declare this field explicitly.

Validation:

- `py -3 -m pytest tests/smoke_test.py -q`
- `py -3 -m compileall src/macro_deeponet scripts`

Known gaps:

- Existing old compacts that lack `strain_field` cannot be retroactively proven
  to be `E` or `LE`; they remain marked `unknown` and should not be mixed into
  formal training without audit notes.

## v1.1 follow-up - 2026-06-22 - Reference-frame IP audit

Purpose:

- Keep the IP geometry audit strict while avoiding false failures when Abaqus
  `COORD` in loaded frames reports deformed integration-point coordinates.

Included:

- Complete compact exporter now uses frame 0 `COORD` / `IVOL` as the reference
  geometry audit source when available.
- Selected training/eval frame `COORD` values are stored separately as
  `ip_xyz_abaqus_coord_selected_frames` and summarized by
  `audit_selected_frame_coord_vs_reference_max_abs`.
- `ip_xyz_abaqus_coord` / `ip_IVOL_abaqus` remain the blocking audit fields and
  now explicitly carry reference-frame scope metadata.

Validation:

- `py -3 -m pytest tests/smoke_test.py -q`
- `py -3 -m compileall src/macro_deeponet scripts`

Known gaps:

- This still assumes the TRUE176/CSS8 reference-frame `IVOL` is directly
  comparable to `detJ`; other element families may need quadrature weights.

## v1.1 follow-up - 2026-06-22 - Constant shape4 loader support

Purpose:

- Let the generic query-point trainer consume complete compacts where a constant
  geometry is stored once as `shape4[4]` or `shape4[1,4]`, matching the exporter
  behavior.

Included:

- Compact frame count inference now prioritizes frame-aligned arrays
  (`q48_raw`, `LE128_base`, `B_LE128_forward`) before reading `shape4`.
- `load_one_compact()` broadcasts constant `shape4` metadata to the selected
  frame count, while still requiring `q48_raw`, `LE128_base`, and
  `B_LE128_forward` to remain frame-aligned.
- Smoke coverage verifies that `shape4[4]` is accepted and broadcast safely.

Validation:

- `py -3 -m pytest tests/smoke_test.py -q` -> `40 passed`
- `py -3 -m compileall src/macro_deeponet scripts`

## v1.1 follow-up - 2026-06-22 - Three-case formal small training audit

Purpose:

- Move from one-case `overlap-debug` smoke to a real non-overlapping
  case-level query-point trainer run.

Inputs:

- `complete_case019_training_ready.npz`
- `complete_case025_training_ready.npz`
- `complete_case031_training_ready.npz`
- All three declare `strain_field=LE`, `B_label_strain_field=LE`,
  `merge_q48_max_abs_diff=0`, `merge_LE128_base_max_abs_diff=0`,
  `merge_ip_keys_match=true`, `audit_ref_ip_xyz_vs_abaqus_coord_max_abs`
  about `2.38e-7`, and `audit_detJ_vs_IVOL_max_abs` about `4.29e-10`.

Trainer:

- `model_style=query-fe-linear-residual`
- `point_feature_source=data`
- `branch_feature_mode=xkeep-qraw`
- `le_normalization=global-component`
- `train_point_sample_count=64`
- `split_mode=case`, `val_fraction=0.34`, `allow_overlap_val=false`
- `epochs=50`

Audit result:

- `validation_split_mode=case`
- `validation_is_overlapping=false`
- `train_cases=[25,31]`, `val_cases=[19]`
- `train_frames=20`, `val_frames=10`
- `point_feature_source=data_generic`, `feature_dim=35`
- `supports_dynamic_points=true`
- `strain_meta.strain_field=LE`

Best observed checkpoint:

- Epoch `25`, `score=2.3568`
- `train_LE_rel=0.8248`, `val_LE_rel=1.3879`
- `train_AD_B_rel=0.9668`, `val_AD_B_rel=0.9689`
- `train_rand_dir_B_rel=0.9872`, `val_rand_dir_B_rel=0.9867`

Latest epoch:

- Epoch `50`, `score=5.3967`
- `train_LE_rel=0.8435`, `val_LE_rel=4.0148`
- `train_AD_B_rel=1.4325`, `val_AD_B_rel=1.3820`

Conclusion:

- Formal small training pipeline audit passed: real complete compacts, explicit
  strain metadata, data point features, dynamic query model, and non-overlap
  case split all worked end to end.
- Model performance is not established. With only three load cases and one
  validation case, validation metrics are poor and unstable; this run should not
  be used as evidence that the architecture trains well.

Known gaps:

- This is case-level load extrapolation within one geometry, not geometry
  generalization.
- More real training-ready compacts are needed before judging convergence or
  comparing architecture quality.

## v1.1 follow-up - 2026-06-22 - Query B baseline warm-start diagnostics

Purpose:

- Address the diagnostic gap where fixed-128 per-IP B baselines fit Sobolev B
  labels well, but the query-point model's point-conditioned B baseline learns
  too slowly from a zero-last `point_b_net`.

Included:

- Added optional `query-fe-linear-residual` warm-start arguments:
  `--b-baseline-warmstart-steps`, `--b-baseline-warmstart-lr`,
  `--b-baseline-warmstart-weight-decay`, and
  `--b-baseline-warmstart-source=train-only`.
- Warm-start uses only `train_idx` frames:
  `B_mean_train[ip,6,48] = mean_train B_norm[frame,ip,6,48]`,
  initializes `global_b_norm` from the IP mean, and trains `point_b_net` on the
  train-only residual `B_mean_train - B_global`.
- Validation cases are recorded as excluded in `warmstart_meta`; they are not
  used to form the B prior.
- Config, checkpoints, partial/final summaries, and per-epoch rows record
  warm-start and point-B diagnostics, including
  `b_prior_before_train_rel`, `b_prior_after_train_rel`,
  `b_prior_after_train_cos`, `global_b_prior_rms`,
  `point_b_correction_rms`, and `point_b_correction_norm_ratio`.
- Baseline diagnostics now include both normalized-J metrics and physical/raw-B
  metrics, including eval-column subsets, so warm-start can be compared against
  `AD_B_rel` without mixing metric scales.
- CSV loss history now includes the point-B correction diagnostics.

Validation:

- `py -3 -m pytest tests/smoke_test.py -q` -> `41 passed`
- `py -3 -m compileall src/macro_deeponet scripts`
- Real 3-case overlap-debug, 1-epoch diagnostic:
  `b_prior_after_train_rel = 0.5929` in normalized-J space, while
  `b_prior_after_train_evalcols_B_rel = 0.3579` matches
  `train_AD_B_rel = 0.3579` in physical/raw-B space. This confirms the new
  raw-B baseline metrics are directly comparable with `AD_B_rel`.

Known gaps:

- Warm-start is a model-diagnostic and training aid, not a replacement for
  wider load-direction coverage.
- It is currently restricted to `train-only`; other sources are intentionally
  rejected to avoid validation leakage.

## v1.1 follow-up - 2026-06-22 - Query warm-start main-training controls

Purpose:

- Diagnose why full Sobolev main training can degrade a useful query B prior
  after warm-start.

Included:

- Added `--le-loss-weight` so B-only main-training diagnostics can set LE loss
  weight to zero without changing the training loop by hand.
- Added `--freeze-b-baseline-after-warmstart` to freeze `global_b_norm` and
  `point_b_net` after train-only warm-start, isolating whether the main
  optimizer is directly corrupting the B baseline.
- Added `--global-b-lr-scale` and `--point-b-lr-scale` optimizer parameter
  groups to test smaller baseline learning rates during full training.
- Config and summaries now record `main_train_control`, including optimizer
  parameter counts and group learning rates.

Intended diagnostics:

- Warm-start + frozen B baseline: if the baseline metric stays good but full
  model `AD_B_rel` worsens, the residual AD path is the likely source.
- Warm-start + B-only main train (`--le-loss-weight 0`): if B degrades despite
  no LE loss, inspect AD-B loss/optimizer settings before increasing epochs.
- Warm-start + reduced baseline learning rates: tests whether preserving the
  prior while training residual improves stability.

Real 3-case overlap-debug diagnostics:

- All runs below are diagnostic only: they use overlapping train/validation
  frames and must not be interpreted as case-level generalization.
- Warm-start baseline before main training:
  `b_prior_after_train_evalcols_B_rel = 0.3579`.
- Experiment A, frozen B baseline with full LE + AD loss:
  the baseline stayed fixed at `b_prior_current_train_evalcols_B_rel = 0.3579`,
  but full-model `train_AD_B_rel` worsened from `0.3586` at epoch 1 to
  `15.2793` at epoch 20. This rules out direct corruption of
  `global_b_norm/point_b_net` as the only problem and points to the residual
  AD path as a major failure mode.
- Experiment B, B-only main train with normalized-J loss:
  normalized `train_AD_B_norm_rel` improved from about `0.5913` to `0.4356`,
  while physical/raw `train_AD_B_rel` worsened from `0.9217` to `1.8168`.
  Optimizing normalized-J alone can move the model in a direction that looks
  better in normalized space but worse in physical B space.
- Experiment B2, B-only main train with physical-J loss:
  raw `train_AD_B_rel` stabilized near `0.49-0.50`, better than normalized-J
  B-only but still worse than the warm-start prior. The loss scale is safer,
  but the main optimizer can still erode the prior.
- Experiment C, full LE + physical-J loss with small baseline learning rates
  (`global_b_lr_scale = 0.05`, `point_b_lr_scale = 0.10`):
  the B prior was preserved (`b_prior_current_train_evalcols_B_rel = 0.3571`
  at epoch 20), full-model `train_AD_B_rel = 0.3574`,
  `val_AD_B_rel = 0.3657`, and LE improved to `train_LE_rel = 0.3028`,
  `val_LE_rel = 0.2514`.

Current interpretation:

- The useful query B prior can be preserved during main training, but it needs
  physical/raw-B-aware J loss and conservative baseline parameter updates.
- The remaining route question is not whether B labels or AD-B are wired, but
  how to schedule residual and baseline updates so LE improves without
  injecting destructive q-derivatives.

## v1.1 follow-up - 2026-06-22 - Formal case split warm-start audit

Purpose:

- Move the best overlap-debug training strategy back to a real non-overlapping
  case-level validation split.
- Add current train/validation B-prior metrics to the generic trainer summary
  and CSV history so future runs can distinguish whether the query B prior
  itself generalizes to the held-out case, or whether residual AD terms dominate
  the total model derivative.

Code changes:

- `query-fe-linear-residual` training now logs both
  `b_prior_current_train_*` and `b_prior_current_val_*`.
- The legacy DeepONet CSV writer includes the validation B-prior fields so
  `loss_history.csv` retains the same diagnostics as the JSON summaries.

Formal 3-case audit:

- Output directory:
  `outputs/query_point_v1_1_formal_small_audit/trainer_case_split_3compact_warmstart_physical_smallBLR_ep50`
- Inputs:
  `complete_case019_training_ready.npz`,
  `complete_case025_training_ready.npz`,
  `complete_case031_training_ready.npz`
- Split:
  `split_mode=case`, `allow_overlap_val=false`,
  `train_cases=[25,31]`, `val_cases=[19]`,
  `train_frames=20`, `val_frames=10`
- Model/training:
  `model_style=query-fe-linear-residual`,
  `point_feature_source=data`,
  `branch_feature_mode=xkeep-qraw`,
  `le_normalization=global-component`,
  `train_point_sample_count=128`,
  train-only B warm-start for 500 steps,
  `j_loss_mode=physical`,
  `global_b_lr_scale=0.05`,
  `point_b_lr_scale=0.10`,
  `le_loss_weight=1.0`,
  `epochs=50`

Warm-start and latest metrics:

- Train-only warm-start, before main training:
  `b_prior_before_train_evalcols_B_rel = 0.9781`,
  `b_prior_after_train_evalcols_B_rel = 0.4313`,
  `b_prior_after_train_evalcols_B_cos = 0.9022`
- Latest and best checkpoint are both epoch 50, `score = 2.7052`
- Current B prior:
  `b_prior_current_train_evalcols_B_rel = 0.3756`,
  `b_prior_current_val_evalcols_B_rel = 0.3694`
- Full model B:
  `train_AD_B_rel = 0.3768`, `train_AD_B_cos = 0.9317`,
  `val_AD_B_rel = 0.3716`, `val_AD_B_cos = 0.9339`
- Physical random-direction B:
  `train_phys_rand_dir_B_rel = 0.3773`,
  `val_phys_rand_dir_B_rel = 0.3708`
- LE:
  `train_LE_rel = 0.2253`, `val_LE_rel = 2.3336`
- Worst aggregate errors:
  `train_AD_B_ip_rel_max = 2.4735`,
  `val_AD_B_ip_rel_max = 2.5196`,
  `val_AD_B_col_rel_max = 1.8485`

Epoch trend:

- Epoch 1: `val_AD_B_rel=0.4385`, `val_LE_rel=7.0280`
- Epoch 10: `val_AD_B_rel=0.3726`, `val_LE_rel=5.9601`
- Epoch 25: `val_AD_B_rel=0.3735`, `val_LE_rel=4.0671`
- Epoch 50: `val_AD_B_rel=0.3716`, `val_LE_rel=2.3336`

Current interpretation:

- The formal case split is clean and non-overlapping.
- The query B strategy is now credible under a held-out case: train-only
  warm-start plus physical-J and conservative B-baseline learning rates keeps
  validation physical/raw `AD_B_rel` near `0.37`.
- LE held-out case generalization is still not established. It improves from a
  very poor initial value, but `val_LE_rel = 2.3336` remains too high.
- With only three load cases and one held-out case, the next scientific step is
  adding independent cases/load directions, not extending this 3-case run to a
  long training schedule.

Validation:

- `py -3 -m pytest tests/smoke_test.py -q` -> `42 passed`
- `py -3 -m compileall src/macro_deeponet scripts`
- `git diff --check`

## v1.2 start - 2026-06-22 - Data-coverage audit tooling

Purpose:

- Start `v1.2 data-coverage audit`.
- Keep model/training strategy fixed and first test whether poor held-out `LE`
  prediction is caused by insufficient independent case/load-direction
  coverage.
- Provide a reproducible data ledger before generating or training on
  `10-20` real training-ready complete compacts.

Included:

- Added `scripts/audit_query_point_data_coverage.py`.
- The script reads complete compact NPZ files and writes:
  `compact_manifest.csv`, `compact_manifest.json`,
  `q_direction_cosine_matrix.csv`, `q_direction_abs_cosine_matrix.csv`,
  `le_rms_distribution.csv`, `audit_summary.json`,
  `train_80_20_summary.json`, and `loo_summary.json`.
- Each compact row records case id, frame count, strain metadata, q norm
  statistics, representative q direction, internal q-direction cosine
  statistics, q-direction nearest-existing coverage, q-direction cluster id,
  `LE_rms`, `B_rms`, merge guards, strain guard, and reference-frame IP audit
  fields.
- Strict mode requires the v1.1 hard-guard metadata:
  `training_ready_sobolev`, explicit non-unknown matching strain fields,
  merge q/LE tolerances, `merge_ip_keys_match`,
  `merge_strain_field_match`, reference COORD audit, and TRUE176/CSS8
  `detJ` vs `IVOL` audit.
- The generated 80/20 and leave-one-case-out files are coverage/split plans
  with metric fields set to `null`; they should be filled only after the
  corresponding formal training runs complete.
- Updated workflow version boundaries: `v1.2` is now data-coverage audit, and
  ID/inference polish is shifted to `v1.3`.

Current v1.2 boundary:

- Do not prioritize model architecture changes in v1.2.
- First expand the real complete compact pool and record q-direction coverage,
  LE scale coverage, and clean case-level splits.
- Keep the first v1.2 training configuration fixed:
  train-only query-B warm-start, physical-J loss,
  `global_b_lr_scale=0.05`, `point_b_lr_scale=0.10`,
  `train_point_sample_count=128`, `split_mode=case`,
  `allow_overlap_val=false`.

Validation:

- Added smoke coverage for manifest/matrix/split-plan generation and strict
  hard-guard failure on missing audit metadata.
- Real 3-case probe using the existing case019/case025/case031 complete
  compacts passed strict v1.2 audit and produced two q-direction clusters at
  `abs(cos) >= 0.95`; pairwise abs-cosine ranged from `0.8967` to `0.9938`.
  This supports the current interpretation that the 3-case pool is still too
  thin in independent load directions.

## v1.2 follow-up - 2026-06-22 - Legacy TRUE176 boundary-control candidate audit

Purpose:

- Use old TRUE176 compact data as a candidate boundary-displacement direction
  library without treating it as formal v1.2 query-point/Abaqus training data.
- Select independent `q48_raw` directions that should be reapplied as fresh
  boundary-displacement controls, then regenerated through Abaqus and the
  current exporter as strict v1.2 complete compacts.

Included:

- Added `scripts/select_legacy176_reexport_candidates.py`.
- The script reads legacy `q48_raw`, `LE128_base`/`le`, optional B metadata,
  and `sample_paths`; it does not train a model and does not require old
  compacts to pass strict v1.2.
- The script compares legacy representative q directions against the current
  strict-pass pool, clusters old candidates by absolute cosine, and writes:
  `legacy176_candidate_manifest.csv`,
  `legacy176_candidate_manifest.json`,
  `legacy176_to_current_abs_cosine.csv`,
  `legacy176_candidate_cluster_summary.json`, and
  `legacy176_reexport_plan.md`.
- The script also writes `legacy176_boundary_control_plan.json`, which contains
  the selected unit `q_direction_48` vectors for future
  `q_new = alpha * q_direction_48` displacement-control runs.
- Added `docs/query_point_v1_2_legacy176_reexport_plan.md` to document that
  old TRUE176 compacts are boundary-control direction sources only and must not
  be added directly to the formal v1.2 pool.

Audit result:

- Current strict-pass pool remains case019/case025/case031:
  `strict_v1_2_pass=true`, `compact_count=3`, `case_count=3`,
  `frame_count=30`, `q_direction_cluster_count=2`,
  `pairwise_abs_cos_min=0.8966673549`,
  `pairwise_abs_cos_median=0.9078195202`,
  `pairwise_abs_cos_max=0.9937922950`.
- Legacy candidate audit scanned 6 old compact sources and found 733 candidate
  samples in 17 q-direction clusters.
- 150 candidates met `max_abs_cos_to_current < 0.95`; 100 of those met the
  stronger `< 0.90` threshold.
- The top 12 recommended re-export candidates are assigned tentative new case
  ids case040-case051.  Their `max_abs_cos_to_current` range is
  `0.226673` to `0.862846`, so they are all less redundant than the current
  case019/case031 near-duplicate direction.
- The audit was rerun with `--compute-b-rms`; recommended candidate `B_rms_mean`
  values were recorded in the manifest and re-export plan.

Known gaps:

- The old TRUE176 compacts still lack the current strict v1.2 metadata
  (`point_features`, IP geometry/Jacobian tables, `ip_keys`, explicit strain
  fields, merge-audit fields, and reference-frame IP audit fields).  They remain
  candidate-direction data only.
- The selected candidates' local `sample_path` entries do not currently resolve
  to ready-to-use current v1.2 samples: top-12 `base_odb_exists=false` and
  `perturb_dir_exists=false`.
- Formal pool entry still requires regenerating a fresh base ODB from
  `q_new = alpha * q_direction_48`, generating a matching Sobolev B compact,
  running the current complete-compact exporter, and passing strict v1.2.

## v1.2 follow-up - 2026-06-22 - Fresh boundary-control case plan

Purpose:

- Convert selected legacy `q_direction_48` candidates into explicit fresh
  Abaqus boundary-displacement control plans for case040-case051.
- Keep v1.2 focused on data generation coverage, not model/loss/training
  changes.

Included:

- Added `scripts/build_v1_2_boundary_control_cases.py`.
- The script reads `legacy176_boundary_control_plan.json` and writes:
  `fresh_boundary_control_plan.json`, `fresh_boundary_control_plan.csv`,
  per-case `caseXXX_q48_frames.csv`, and
  `abaqus_boundary_control_command_templates.md`.
- Each selected case gets 10 frames with
  `alpha_factors = 0.1, 0.2, ..., 1.0` and
  `q48_frame = alpha_factor * q_norm_seed_mean * q_direction_48`.
- The script validates that every frame satisfies `||q48_frame|| = alpha` and
  keeps the same direction as the selected `q_direction_48`.
- Added `docs/query_point_v1_2_fresh_boundary_control_cases.md` to document the
  q48 order and fresh Abaqus/export/strict-audit workflow.

Boundary condition note:

- q48 is the current 16 keep-node x 3 DOF control vector.  Existing generated
  Abaqus `.inp` examples often apply displacements on 32 boundary nodes / 96
  DOF through a boundary-contract bridge.  If the Abaqus generator expects 96
  DOF, it must explicitly apply the same `q48 -> q_boundary[32,3]` bridge; q48
  must not be silently reinterpreted as a 96-DOF vector.

Known gaps:

- This commit does not run Abaqus and does not generate ODB/B/complete compact
  files.
- A project-specific q48-to-Abaqus boundary-control generator, shape4 values,
  and matching Sobolev B compact generation are still required before any
  case040-case051 compact can enter the formal v1.2 pool.

## v1.2 follow-up - 2026-06-22 - Abaqus boundary bridge requirements

Purpose:

- Resolve the bridge between generated `case040-case051` q48 frame plans and
  Abaqus boundary-node displacement inputs before running fresh ODB generation.

Included:

- Added `scripts/build_v1_2_abaqus_boundary_inputs.py`.
- The script applies an explicit bridge contract:
  `q_boundary96 = T_boundary_96x48 @ q48`.
- The bridge is read from `boundary_contract_arrays.npz`, which provides
  `T_boundary_96x48`, `keep_nodes`, and `boundary_nodes`.
- The script writes per-case `case###_boundary96_frames.csv`,
  `case###_boundary_nodes.json`, and `case###_abaqus_bc_commands.txt`, plus a
  summary JSON/CSV under the output directory.
- Added `docs/query_point_v1_2_abaqus_boundary_bridge_requirements.md` to record
  the bridge source, node orders, pilot sequence, exporter template, and strict
  audit template.

Bridge audit:

- Local boundary-contract files contain a fixed `T_boundary_96x48` with
  `boundary_nodes = [1,2,3,4,5,6,10,11,15,16,20,21,22,23,24,25,26,27,28,29,30,31,35,36,40,41,45,46,47,48,49,50]`.
- The checked contracts share the same T matrix and node orders.
- For a representative contract,
  `q48_frames @ T_boundary_96x48.T` reproduces stored
  `q_boundary_frames` with `max_abs_diff=0`.

Known gaps:

- The generated Abaqus BC command files are templates only.  A fresh Abaqus
  base model/input generator, shape4 selection, step/increment convention, field
  output requests, and matching Sobolev B compact generation are still required.
- Old TRUE176 `LE/B` values remain scale/provenance hints only and are not used
  as v1.2 labels.

## v1.2 follow-up - 2026-06-22 - Pilot fresh Abaqus case041 chain

Purpose:

- Run one representative fresh Abaqus pilot case before attempting any batch
  case040-case051 generation.
- Verify the end-to-end chain:
  `case041 q48/boundary96 plan -> fresh Abaqus ODB -> matching Sobolev B compact
  -> current complete compact exporter -> strict v1.2 coverage audit`.

Included:

- Added `scripts/run_v1_2_pilot_case041_fresh_abaqus.py`.
- The script is intentionally narrow and defaults to case041 only.
- It validates that the generated q48 frames are linear 0.1-1.0 scale factors
  of the final q48 vector and that
  `T_boundary_96x48 @ q48 == boundary96` before running Abaqus.
- It uses the audited shape4/T-boundary bridge and existing fresh Abaqus
  base-plus-48-forward-perturbation generator to build a matching Sobolev B
  compact.
- It repacks the fresh B compact with explicit
  `strain_field=LE` and `B_label_strain_field=LE`, then calls the current
  `export_abaqus_true176_complete_compact.py` with `--merge-compact`,
  `--require-b`, `--require-merge-ip-keys`, and `--require-ip-audit`.

Pilot result:

- Fresh base job and 48 forward perturbation jobs completed successfully for
  case041.
- Generated:
  `D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case041\abaqus_run\base\fresh_case041.odb`.
- Generated:
  `D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case041\b_compact\case041_sobolev_B_compact.npz`.
- Generated:
  `D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case041\complete\complete_case041_training_ready.npz`.
- Single-case strict audit passed:
  `strict_v1_2_pass=true`, `compact_count=1`, `case_count=1`,
  `frame_count=10`, `q_direction_cluster_count=1`.
- Full-pool audit with case019/case025/case031 plus fresh case041 passed:
  `strict_v1_2_pass=true`, `compact_count=4`, `case_count=4`,
  `frame_count=40`, `q_direction_cluster_count=3`,
  `pairwise_abs_cos_min=0.3343678087`,
  `pairwise_abs_cos_median=0.6789307736`,
  `pairwise_abs_cos_max=0.9937922950`.
- case041 adds a new q-direction cluster.  Its nearest existing direction is
  case031 with `max_abs_cos_to_current=0.4611941923`.

Known gaps:

- This is still a data-generation/contract pilot, not a training result.
- Large generated Abaqus/NPZ outputs remain outside git and must not be
  committed.
- Old TRUE176 `LE/B` values were not used as training labels; only the selected
  boundary-displacement direction and audited bridge were used to produce fresh
  Abaqus data.

## v1.2 follow-up - 2026-06-22 - Extend fresh Abaqus pilots to case043 and case050

Purpose:

- Extend the validated case041 pilot path to two more selected fresh
  boundary-control directions without running the full case040-case051 batch.
- Keep this as a data-generation and coverage-audit step only; no training or
  model/loss/learning-rate changes were made.

Included:

- Generalized `scripts/run_v1_2_pilot_case041_fresh_abaqus.py` so its manifest
  file is `pilot_{case_id}_plan.json` instead of the case041-specific
  `pilot_case041_plan.json`.
- Reused the same audited bridge, fresh Abaqus base-plus-48-forward-perturbation
  chain, B compact repack, current complete compact exporter, and strict v1.2
  data coverage audit for case043 and case050.

Pilot result:

- case043 generated:
  `D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case043\abaqus_run\base\fresh_case043.odb`,
  `D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case043\b_compact\case043_sobolev_B_compact.npz`,
  and
  `D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case043\complete\complete_case043_training_ready.npz`.
- case050 generated:
  `D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case050\abaqus_run\base\fresh_case050.odb`,
  `D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case050\b_compact\case050_sobolev_B_compact.npz`,
  and
  `D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case050\complete\complete_case050_training_ready.npz`.
- Both single-case strict audits passed with `strict_v1_2_pass=true`.
- Full-pool audit with case019/case025/case031/case041/case043/case050 passed:
  `strict_v1_2_pass=true`, `compact_count=6`, `case_count=6`,
  `frame_count=60`, `q_direction_cluster_count=5`,
  `pairwise_abs_cos_min=0.1808575415`,
  `pairwise_abs_cos_median=0.5679210886`,
  `pairwise_abs_cos_max=0.9937922950`.
- case043 adds q-direction cluster 3 with
  `max_abs_cos_to_current=0.5679210886`, nearest existing case019,
  `LE_rms_mean=0.0001830914007`, and `B_rms_mean=9.1729672619`.
- case050 adds q-direction cluster 4 with
  `max_abs_cos_to_current=0.8533500881`, nearest existing case019,
  `LE_rms_mean=0.0002458211478`, and `B_rms_mean=9.1749346929`.

Known gaps:

- The pool is improved but still below the v1.2 target of 10-20 strict-pass
  complete compacts and 6-10 q-direction clusters.
- The original case019/case031 near-duplicate remains in the pool, so
  `pairwise_abs_cos_max` is still `0.9937922950`.
- Large generated Abaqus/NPZ outputs remain outside git and must not be
  committed.
- Old TRUE176 `LE/B` values were not used as training labels.

## v1.2 follow-up - 2026-06-22 - Second fresh Abaqus batch reaches coverage target

Purpose:

- Expand the v1.2 fresh strict-pass compact pool beyond the first three fresh
  pilots while still avoiding training/model/loss changes.
- Re-rank the remaining case040-case051 candidates against the current 6-case
  strict-pass pool before selecting the next cases.

Selection:

- Selected case044, case046, case049, and case045.
- All selected candidates had `max_abs_cos_to_current_pool < 0.95`, were
  expected to add q-direction clusters, and had non-tiny amplitude/LE scale
  hints.
- Deferred case040 even though its direction was independent because its
  expected LE scale was extremely low (`LE_rms_hint_mean=2.8848e-06`,
  `q_norm_max=6.6611e-05`).
- Deferred case048 because it was near-duplicate with existing case050
  (`max_abs_cos_to_current_pool=0.9551`).
- Deferred case051 because it was close to case045/case049 in the candidate
  set.  case042 and case047 remain usable follow-up candidates.

Pilot result:

- case044, case046, case049, and case045 each generated a fresh Abaqus base
  ODB, matching Sobolev B compact, and complete training-ready compact.
- All four single-case strict v1.2 audits passed.
- Full-pool audit with case019/case025/case031/case041/case043/case050 plus
  this second batch passed:
  `strict_v1_2_pass=true`, `compact_count=10`, `case_count=10`,
  `frame_count=100`, `q_direction_cluster_count=9`,
  `pairwise_abs_cos_min=0.0279701647`,
  `pairwise_abs_cos_median=0.6286590998`,
  `pairwise_abs_cos_max=0.9937922950`.
- case044 adds q-direction cluster 5 with
  `max_abs_cos_to_current=0.6653788417`, nearest case031,
  `LE_rms_mean=0.00008300474246`, and `B_rms_mean=9.1698764680`.
- case046 adds q-direction cluster 6 with
  `max_abs_cos_to_current=0.7669577541`, nearest case019,
  `LE_rms_mean=0.0001091349404`, and `B_rms_mean=9.1698297192`.
- case049 adds q-direction cluster 7 with
  `max_abs_cos_to_current=0.8364611150`, nearest case031,
  `LE_rms_mean=0.0001956932680`, and `B_rms_mean=9.1729725926`.
- case045 adds q-direction cluster 8 with
  `max_abs_cos_to_current=0.8522134771`, nearest case049,
  `LE_rms_mean=0.0001667079867`, and `B_rms_mean=9.1675470365`.

Current v1.2 status:

- The data pool now meets the minimum coverage target:
  10 strict-pass complete compacts and 9 q-direction clusters.
- The pool still contains the original case019/case031 near-duplicate, so
  `pairwise_abs_cos_max` remains `0.9937922950`.
- This is a data coverage milestone only.  No training was run, no model/loss
  settings changed, and old TRUE176 `LE/B` values were not used as labels.
- Large Abaqus/NPZ outputs remain outside git and must not be committed.

## v1.2 follow-up - 2026-06-22 - Fixed-strategy 10-case training audit

Purpose:

- Start the first v1.2 fixed-strategy training audit after the fresh
  strict-pass compact pool reached the minimum coverage target.
- Keep the v1.2 strategy frozen and test whether the expanded 10-case /
  9-cluster pool improves held-out LE/B behavior relative to the 3-case formal
  split.
- Do not tune the model, loss, learning rates, or epoch count in response to
  this run.

Pool:

- Source coverage audit:
  `D:\IS-FEM\outputs\query_point_v1_2_data_coverage_audit\full_pool_after_second_fresh_batch`
- Frozen training manifest:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest`
- Pool cases:
  `case019`, `case025`, `case031`, `case041`, `case043`,
  `case044`, `case045`, `case046`, `case049`, `case050`.
- Coverage:
  `strict_v1_2_pass=true`, `compact_count=10`, `case_count=10`,
  `frame_count=100`, `q_direction_cluster_count=9`,
  `pairwise_abs_cos_min=0.0279701647`,
  `pairwise_abs_cos_median=0.6286590998`,
  `pairwise_abs_cos_max=0.9937922950`.
- All compacts declare `strain_field=LE` and
  `B_label_strain_field=LE`; no unknown strain metadata and no legacy TRUE176
  label compact were mixed into the pool.

Split:

- Split plan:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\split_plan_main_80_20.json`
- Train cases:
  `[19, 25, 31, 43, 44, 45, 46, 50]`
- Validation cases:
  `[41, 49]`
- Trainer split metadata:
  `validation_split_mode=case`, `validation_is_overlapping=false`,
  `train_frames=80`, `val_frames=20`.
- Validation case notes:
  `case041` is a low-LE fresh direction with nearest-train abs-cosine
  `0.6417599947` to `case044`.
  `case049` is a mid-LE fresh direction with nearest-train abs-cosine
  `0.8522134771` to `case045`.

Fixed strategy:

- Output directory:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_main_80_20`
- Trainer:
  `macro_deeponet.train_true176_generic_sobolev`
- Fixed settings:
  `model_style=query-fe-linear-residual`,
  `point_feature_source=data`,
  `branch_feature_mode=xkeep-qraw`,
  `le_normalization=global-component`,
  `train_point_sample_count=128`,
  `split_mode=case`,
  `allow_overlap_val=false`,
  `epochs=50`,
  `b_baseline_warmstart_source=train-only`,
  `b_baseline_warmstart_steps=500`,
  `j_loss_mode=physical`,
  `global_b_lr_scale=0.05`,
  `point_b_lr_scale=0.10`,
  `le_loss_weight=1.0`.
- Warm-start was train-only:
  `warmstart_train_cases=[19,25,31,43,44,45,46,50]`,
  `warmstart_val_cases_excluded=[41,49]`,
  `warmstart_validation_is_overlapping=false`.

Result:

- Best and latest checkpoints are both epoch 50.
- Best/latest metrics:
  `best_score=5.6694053020`,
  `train_LE_rel=0.3710049271`,
  `val_LE_rel=5.1376408458`,
  `train_AD_B_rel=0.5343235027`,
  `val_AD_B_rel=0.5317644562`,
  `train_AD_B_cos=0.8530025687`,
  `val_AD_B_cos=0.8755700366`,
  `train_phys_rand_dir_B_rel=0.5360939388`,
  `val_phys_rand_dir_B_rel=0.5349325279`,
  `b_prior_current_train_evalcols_B_rel=0.5338542326`,
  `b_prior_current_val_evalcols_B_rel=0.5306775207`,
  `train_AD_B_ip_rel_max=4.0114861980`,
  `val_AD_B_ip_rel_max=3.5443742903`,
  `val_AD_B_col_rel_max=0.9385777742`.
- Epoch trend:
  `val_LE_rel` decreased from `12.7145` at epoch 1 to `5.1376` at epoch 50,
  while `val_AD_B_rel` stayed near `0.532`.

Interpretation:

- The fixed-strategy 10-case run completed with a clean non-overlapping
  case-level split.
- The B prior and full AD-B metrics remain closely aligned, so the residual AD
  path did not destroy the B field in this split.
- B did not match the 3-case formal split quality:
  `val_AD_B_rel` is about `0.5318` here versus about `0.3716` in the 3-case
  split.
- LE generalization is not established:
  `val_LE_rel=5.1376`, worse than the previous 3-case audit value
  `val_LE_rel=2.3336`.
- Conservative conclusion:
  the v1.2 fixed-strategy training audit ran cleanly, but this main 80/20 split
  does not show LE generalization improvement from the expanded 10-case pool.
  The next diagnostic step should be additional fixed-strategy split or
  leave-one-case-out audits, not ad hoc model/loss/lr/epoch tuning.

Validation boundary:

- No model structure, loss, learning-rate strategy, or epoch schedule was
  changed for this audit.
- Old TRUE176 `LE/B` values were not used as v1.2 labels.
- Large generated training artifacts and checkpoints remain outside git and
  must not be committed.

## v1.2 follow-up - 2026-06-23 - Fixed-strategy split diagnostic

Purpose:

- Diagnose whether the poor first 10-case fixed-strategy result was specific
  to the split_A validation cases (`case041`, `case049`) or reflects a broader
  fixed-strategy LE generalization issue.
- Keep the training strategy frozen: no model, loss, learning-rate, or epoch
  changes.
- Add a lightweight checkpoint attribution helper that evaluates selected
  cases from an existing checkpoint without training.

Included:

- Added `scripts/evaluate_v1_2_checkpoint_by_case.py`.
- The script reconstructs the generic query-point preprocessing from a saved
  checkpoint and compact list, then calls the existing trainer evaluation
  metrics for selected case ids.
- Generated split diagnostic plans under:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_split_diagnostic_plan`.
- Planned splits:
  split_A `[41,49]` completed, split_B `[44,46]` run this round, split_C
  `[43,50]` planned only, split_D `[25,45]` planned only.

Split_A attribution:

- Output:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_main_80_20_case_attribution`
- Per-case metrics:
  `case041`: `LE_rel=9.1422`, `AD_B_rel=0.5318`,
  `phys_rand_dir_B_rel=0.5330`, `B_prior_rel=0.5306`,
  `AD_B_cos=0.8754`.
  `case049`: `LE_rel=4.0235`, `AD_B_rel=0.5317`,
  `phys_rand_dir_B_rel=0.5330`, `B_prior_rel=0.5307`,
  `AD_B_cos=0.8757`.
- Interpretation: split_A's LE failure was worse on case041 than case049, while
  B metrics were nearly identical across the two validation cases.

Split_B:

- Output:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_split_B_val44_46`
- Split:
  `train_cases=[19,25,31,41,43,45,49,50]`,
  `val_cases=[44,46]`,
  `validation_is_overlapping=false`.
- Fixed strategy matched split_A exactly:
  `model_style=query-fe-linear-residual`,
  `point_feature_source=data`,
  `branch_feature_mode=xkeep-qraw`,
  `le_normalization=global-component`,
  `train_point_sample_count=128`,
  `split_mode=case`,
  `allow_overlap_val=false`,
  `epochs=50`,
  `b_baseline_warmstart_source=train-only`,
  `b_baseline_warmstart_steps=500`,
  `j_loss_mode=physical`,
  `global_b_lr_scale=0.05`,
  `point_b_lr_scale=0.10`,
  `le_loss_weight=1.0`.
- Best and latest checkpoints are both epoch 50:
  `train_LE_rel=0.3992`,
  `val_LE_rel=8.5155`,
  `train_AD_B_rel=0.5178`,
  `val_AD_B_rel=0.5103`,
  `val_AD_B_cos=0.8834`,
  `val_phys_rand_dir_B_rel=0.5143`,
  `b_prior_current_val_evalcols_B_rel=0.5094`,
  `val_AD_B_ip_rel_max=3.3736`,
  `val_AD_B_col_rel_max=1.0005`.

Split_B attribution:

- Output:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_split_B_val44_46_case_attribution`
- Per-case metrics:
  `case044`: `LE_rel=9.6526`, `AD_B_rel=0.5102`,
  `phys_rand_dir_B_rel=0.5132`, `B_prior_rel=0.5094`,
  `AD_B_cos=0.8835`.
  `case046`: `LE_rel=7.7823`, `AD_B_rel=0.5104`,
  `phys_rand_dir_B_rel=0.5189`, `B_prior_rel=0.5094`,
  `AD_B_cos=0.8833`.

Current interpretation:

- split_B did not rescue LE generalization; `val_LE_rel=8.5155` is worse than
  split_A's `val_LE_rel=5.1376`.
- B remains stable and aligned with the B prior in both splits:
  split_A `val_AD_B_rel=0.5318` vs `B_prior_val=0.5307`;
  split_B `val_AD_B_rel=0.5103` vs `B_prior_val=0.5094`.
- The conservative diagnosis is now stronger:
  the v1.2 10-case LE failure is not just a case041/case049 split-specific
  artifact.  Under the frozen strategy and current single-geometry 10-case
  pool, LE case-level generalization appears systematically weak while B
  remains controlled by the query B prior.
- This still does not prove the model architecture is bad.  It means the next
  useful diagnostic question is why the residual/LE prediction branch fails to
  generalize when the B prior remains stable.

Validation boundary:

- No old TRUE176 `LE/B` labels were used as v1.2 labels.
- No model, loss, learning-rate, or epoch settings were changed.
- Large generated checkpoints, NPZ/ODB files, and loss histories remain outside
  git and must not be committed.

## v1.2 follow-up - 2026-06-23 - LE failure decomposition

Purpose:

- Decompose why fixed-strategy split_A and split_B have high held-out `LE_rel`
  while `AD_B` remains stable and aligned with the query B prior.
- Keep this as an evaluation-only diagnostic: no training, no model/loss/lr
  changes, and no epoch changes.

Included:

- Added `scripts/diagnose_v1_2_le_failure.py`.
- The script reconstructs the generic query-point preprocessing from a saved
  checkpoint, runs forward LE prediction for selected cases, and writes:
  `le_case_summary.csv`, `le_component_summary.csv`,
  `le_worst_ip_summary.csv`, per-case frame trend CSVs,
  `le_offset_summary.csv`, `le_offset_by_case.csv`, and
  `le_failure_diagnostic_summary.json`.
- It also computes zero-prediction and train-mean baselines, scale/bias
  corrections, train-only affine LE oracle diagnostics, and B-prior offset
  diagnostics.

Outputs:

- Split_A diagnostic:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\le_failure_diagnostic\split_A_val41_49`
- Split_B diagnostic:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\le_failure_diagnostic\split_B_val44_46`

Key LE scale result:

- The model predicts held-out LE values that are much larger than the true
  target RMS:
  `case041 target_rms=9.33e-05, pred_rms=8.60e-04, LE_rel=9.14`;
  `case049 target_rms=2.21e-04, pred_rms=9.04e-04, LE_rel=4.02`;
  `case044 target_rms=9.36e-05, pred_rms=9.18e-04, LE_rel=9.65`;
  `case046 target_rms=1.23e-04, pred_rms=9.64e-04, LE_rel=7.78`.
- `zero_pred_LE_rel=1.0` for all checked validation cases, so the model is
  worse than predicting zero on these held-out cases.
- Scale-only correction improves each case only to about zero-baseline quality:
  `scaled_LE_rel` is about `0.98-0.99`.
- Bias-only correction barely helps, so the failure is not a simple constant
  offset.

Component/IP localization:

- Worst average components are dominated by `LE11`, `LE23`, and `LE22`.
- Worst local IP errors are severe:
  split_A `case041` reaches `IP_LE_rel=43.56`;
  split_B `case044` reaches `IP_LE_rel=40.99`.
- Low component targets amplify some local relative errors, but the aggregate
  predicted LE RMS is also too large, so this is not only a reporting artifact.

Frame/alpha trend:

- Low-alpha frames are worst, but high-alpha frames remain poor:
  `case041 alpha=0.1 -> LE_rel=52.05`, `alpha=1.0 -> LE_rel=6.37`;
  `case044 alpha=0.1 -> LE_rel=59.65`, `alpha=1.0 -> LE_rel=6.06`.
- Thus low target RMS amplifies the metric, but does not fully explain the
  failure.

Affine oracle:

- The train-only affine LE oracle is numerically ill-conditioned:
  split_A rank `41/49`, condition about `1.51e20`;
  split_B rank `41/49`, condition about `1.37e20`.
- Its held-out errors are enormous, so the current 8-train-case q design does
  not provide a stable unconstrained affine value-field extrapolation basis.

Offset / anchor result:

- True offset `LE_true - B_true @ q48` is tiny for the validation cases:
  `offset_true_rms` ranges from about `2.87e-06` to `1.14e-05`.
- Predicted offset `LE_pred - B_prior @ q48` is much larger:
  about `7.9e-04` to `9.0e-04`.
- Offset relative errors are large:
  `case041=213.1`, `case049=68.9`, `case044=252.0`, `case046=313.2`.
- This is the clearest signal: B is stable, but the LE value anchor/integration
  constant learned by the network is wrong on held-out cases.

Current interpretation:

- The high `LE_rel` is partly amplified by low validation target RMS, but the
  model is genuinely predicting an overlarge held-out LE field and is worse
  than the zero baseline.
- The issue is not bad B labels, residual AD destroying B, legacy TRUE176 label
  contamination, or split_A bad luck only.
- The most likely current bottleneck is LE value-anchor / residual value-field
  generalization under the frozen strategy and current single-geometry,
  low-LE-heavy fresh validation cases.

Validation boundary:

- No new model was trained for this decomposition.
- No old TRUE176 `LE/B` labels were used as v1.2 labels.
- No model, loss, learning-rate, or epoch settings were changed.
- Large generated artifacts remain outside git and must not be committed.

## v1.2 follow-up - 2026-06-23 - B@q anchor oracle diagnostic

Purpose:

- Test whether the stable query B prior provides a better held-out LE value
  anchor than the current learned LE head.
- Check the data-side oracle `B_true @ q48` against `LE_true`.
- Keep this as an evaluation-only diagnostic: no training, no model/loss/lr
  changes, and no epoch changes.

Included:

- Added `scripts/diagnose_v1_2_bq_anchor_oracle.py`.
- The script reconstructs the checkpoint preprocessing, evaluates existing
  checkpoints on selected held-out cases, and writes:
  `bq_anchor_case_summary.csv`, `bq_anchor_frame_summary.csv`,
  `zero_q_anchor_summary.csv`, `residual_offset_summary.csv`, and
  `bq_anchor_diagnostic_summary.json`.

Outputs:

- Split_A diagnostic:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\bq_anchor_oracle_diagnostic\split_A_val41_49`
- Split_B diagnostic:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\bq_anchor_oracle_diagnostic\split_B_val44_46`

Key results:

- `B_true @ q48` reconstructs `LE_true` well:
  `Btrue_q_LE_rel=0.0400` for `case041`, `0.0517` for `case049`,
  `0.0380` for `case044`, and `0.0233` for `case046`.
- `B_prior @ q48` is substantially better than the current model LE head, but
  still worse than zero on these low-LE held-out cases:
  `Bprior_q_LE_rel=3.4515` vs `model_LE_rel=9.1422` for `case041`;
  `1.6303` vs `4.0235` for `case049`;
  `1.3224` vs `9.6526` for `case044`;
  `1.5006` vs `7.7823` for `case046`.
- The q=0 model output is large and near the observed error scale:
  `zero_q_pred_rms=7.81e-04` for split_A and `9.02e-04` for split_B.
- The full-model residual/value offset remains large:
  `current_offset_rms=7.87e-04` to `8.99e-04`, compared with
  `true_offset_rms=2.87e-06` to `1.14e-05`.

Current interpretation:

- The fresh strict-pass compact `q/LE/B` contract is strong: the data-side
  `B_true @ q48` oracle nearly reconstructs LE.
- The learned LE value head is not using the stable query B prior as a reliable
  value anchor.  `B_prior @ q48` beats the current LE prediction by about
  `2.47x` to `7.30x` in relative error.
- `B_prior @ q48` alone is not a solved LE predictor for these low-LE held-out
  cases because it remains worse than the zero baseline.
- The next design question is therefore a value-anchor question, not another
  data-contract or training-schedule question: consider an anchored form such as
  `LE_hat = B_prior @ q + residual` with explicit zero-q and low-alpha residual
  constraints.

Validation boundary:

- No new model was trained for this diagnostic.
- No old TRUE176 `LE/B` labels were used as v1.2 labels.
- No model, loss, learning-rate, or epoch settings were changed.
- Large generated artifacts remain outside git and must not be committed.

## v1.3 prototype - 2026-06-23 - Anchored LE head

Purpose:

- Move from v1.2 diagnostics to a structural LE value-anchor prototype.
- Keep the old `query-fe-linear-residual` model style unchanged.
- Add an opt-in anchored model style that enforces raw `q=0 -> LE=0`.

Included:

- Added `QueryFEAnchoredLinearResidualDeepONet`.
- Added `--model-style query-fe-linear-residual-anchored`.
- The anchored model computes:
  `LE_hat_norm = LE_zero_norm + B_prior_norm(point) @ (q_norm - q0_norm)
  + gate(||q_raw||) * (R_raw(q, point) - R_raw(q0, point))`.
- Added evaluation metrics:
  `zero_q_LE_pred_rms`, `zero_q_LE_pred_max_abs`,
  `zero_q_residual_rms`, `Bprior_q_LE_rel`, and
  `model_minus_Bprior_offset_rms`.
- Added smoke coverage for forward shape, raw zero-q anchor,
  residual zero-subtraction, AD-B, old model style compatibility, and a tiny
  1-epoch anchored training smoke.
- Added `docs/query_point_v1_3_anchored_le_head_design.md`.

Current interpretation:

- v1.3 is not a hyperparameter tuning step.  It is motivated by v1.2 evidence
  that the LE value branch has a wrong zero/value anchor while the fresh
  `q/LE/B` data contract and query B prior remain credible.
- This prototype only establishes the optional structure and smoke behavior.
  It does not claim improved formal case-level performance.

Validation boundary:

- No formal 50-epoch v1.3 training audit was run in this commit.
- No old TRUE176 `LE/B` labels were used as v1.2/v1.3 labels.
- No tag was moved.
- Large generated artifacts remain outside git and must not be committed.

## v1.3 audit - 2026-06-23 - Anchored LE head split_A

Purpose:

- Run the first fixed-strategy formal audit of the v1.3 anchored LE head.
- Compare directly against the v1.2 split_A baseline.
- Keep the only intended change as:
  `model_style=query-fe-linear-residual-anchored`.
- Do not change loss, learning rate, epoch count, split, or training data.

Inputs:

- Compact pool:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt`
- Cases:
  `case019`, `case025`, `case031`, `case041`, `case043`,
  `case044`, `case045`, `case046`, `case049`, `case050`.
- Validation split:
  `train_cases=[19,25,31,43,44,45,46,50]`,
  `val_cases=[41,49]`.
- `validation_is_overlapping=false`, `train_frames=80`, `val_frames=20`.

Outputs:

- Training:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_A_val41_49`
- Case attribution:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_A_val41_49_case_attribution`
- B@q diagnostic:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_A_bq_anchor_oracle\split_A_val41_49_best`
- Comparison:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_A_comparison`
- Documentation:
  `docs/query_point_v1_3_anchored_split_A_audit.md`

Key split_A comparison:

- v1.2 split_A best/latest:
  `val_LE_rel=5.137640845816213`,
  `val_AD_B_rel=0.5317644562295766`,
  `b_prior_current_val_evalcols_B_rel=0.5306775207214273`,
  prior zero-q diagnostic `zero_q_pred_rms ~= 7.81e-04`.
- v1.3 anchored best checkpoint, epoch 10:
  `val_LE_rel=2.1796218548207182`,
  `val_AD_B_rel=0.5322959397919018`,
  `val_AD_B_cos=0.8718914233087541`,
  `b_prior_current_val_evalcols_B_rel=0.532185230457769`,
  `val_zero_q_LE_pred_rms=1.1484941768098402e-11`,
  `val_model_minus_Bprior_offset_rms=0.00010172704191546938`.
- v1.3 anchored latest checkpoint, epoch 50:
  `val_LE_rel=3.0012689077205144`,
  `val_AD_B_rel=0.5329800420018063`,
  `val_zero_q_LE_pred_rms=1.1484941768098402e-11`,
  `val_model_minus_Bprior_offset_rms=0.0003313721329316278`.

Per-case best attribution:

- `case041`: `LE_rel=3.609960155721904`,
  `AD_B_rel=0.5321915350416156`,
  `B_prior_rel=0.5321483814179929`,
  `zero_q_LE_pred_rms=1.1484941768098398e-11`,
  `Bprior_q_LE_rel=3.8535600056709796`,
  `model_minus_Bprior_offset_rms=0.00010339966315563473`.
- `case049`: `LE_rel=1.8090445751604918`,
  `AD_B_rel=0.5324002526523295`,
  `B_prior_rel=0.532222051735648`,
  `zero_q_LE_pred_rms=1.1484941768098398e-11`,
  `Bprior_q_LE_rel=1.8501516802362072`,
  `model_minus_Bprior_offset_rms=0.00010005442067530404`.

Current interpretation:

- The zero-q ghost LE field is structurally removed in split_A.
- The best split_A `val_LE_rel` improves materially relative to v1.2, while
  AD-B remains stable.
- The latest checkpoint is worse than the best checkpoint, so the result is
  still epoch-sensitive and should be judged by best checkpoint for this audit.
- This is enough evidence to run split_B next with the same fixed strategy.
- It is not yet evidence that v1.3 broadly solves LE generalization; split_B
  and LOO remain required.

Validation boundary:

- No model, loss, learning-rate, epoch, or split changes were made beyond the
  already committed opt-in anchored model style.
- No old TRUE176 `LE/B` labels were used as v1.2/v1.3 labels.
- No tag was moved.
- Large generated artifacts remain outside git and must not be committed.

## v1.3 audit - 2026-06-23 - Anchored LE head split_B

Purpose:

- Run the second fixed-strategy formal audit of the v1.3 anchored LE head.
- Compare directly against the v1.2 split_B baseline.
- Keep the only intended model change relative to v1.2 as:
  `model_style=query-fe-linear-residual-anchored`.
- Change only `val_cases` relative to v1.3 split_A; do not change loss,
  learning rate, epoch count, split policy, or data pool.

Inputs:

- Compact pool:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt`
- Cases:
  `case019`, `case025`, `case031`, `case041`, `case043`,
  `case044`, `case045`, `case046`, `case049`, `case050`.
- Validation split:
  `train_cases=[19,25,31,41,43,45,49,50]`,
  `val_cases=[44,46]`.
- `validation_is_overlapping=false`, `train_frames=80`, `val_frames=20`.

Outputs:

- Training:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_B_val44_46`
- Case attribution:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_B_val44_46_case_attribution`
- B@q diagnostic:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_B_bq_anchor_oracle\split_B_val44_46_best`
- Comparison:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_B_comparison`
- Documentation:
  `docs/query_point_v1_3_anchored_split_B_audit.md`

Key split_B comparison:

- v1.2 split_B best/latest:
  `val_LE_rel=8.515458607946384`,
  `val_AD_B_rel=0.5103261083920874`,
  `val_AD_B_cos=0.8833993891072419`,
  `b_prior_current_val_evalcols_B_rel=0.5093552486612618`.
- v1.3 anchored best checkpoint, epoch 1:
  `val_LE_rel=1.5595633647976805`,
  `val_AD_B_rel=0.5125447167349023`,
  `val_AD_B_cos=0.8790831126173732`,
  `b_prior_current_val_evalcols_B_rel=0.5125437997617942`,
  `val_zero_q_LE_pred_rms=1.157102562071606e-11`,
  `val_model_minus_Bprior_offset_rms=1.7732607370311359e-06`.
- v1.3 anchored latest checkpoint, epoch 50:
  `val_LE_rel=1.7711132238745921`,
  `val_AD_B_rel=0.5114605897630667`,
  `val_zero_q_LE_pred_rms=1.157102562071606e-11`,
  `val_model_minus_Bprior_offset_rms=9.181341274153851e-05`.

Per-case best attribution:

- `case044`: `LE_rel=1.2623462866926172`,
  `AD_B_rel=0.512531727633403`,
  `B_prior_rel=0.5125308365182847`,
  `zero_q_LE_pred_rms=1.157102562071606e-11`,
  `Bprior_q_LE_rel=1.2603167162795477`,
  `model_minus_Bprior_offset_rms=1.612453271090318e-06`.
- `case046`: `LE_rel=1.7080236136717677`,
  `AD_B_rel=0.5125577066694196`,
  `B_prior_rel=0.5125567638373174`,
  `zero_q_LE_pred_rms=1.157102562071606e-11`,
  `Bprior_q_LE_rel=1.706468997850503`,
  `model_minus_Bprior_offset_rms=1.9340682029719545e-06`.

Current interpretation:

- The zero-q ghost LE field remains structurally removed in split_B.
- The best split_B `val_LE_rel` improves materially relative to v1.2, while
  AD-B remains stable.
- The latest checkpoint remains much better than v1.2 split_B, but is slightly
  worse than best, so epoch sensitivity remains visible.
- Together with split_A, v1.3 anchored head now shows LE improvement on two
  independent validation splits without breaking B.
- Broad performance is still not fully proven; LOO or a formal v1.3 summary is
  the next audit step before any new tuning.

Validation boundary:

- No model, loss, learning-rate, epoch, split-policy, or data-pool changes were
  made beyond the already committed opt-in anchored model style and the requested
  split_B validation cases.
- No old TRUE176 `LE/B` labels were used as v1.2/v1.3 labels.
- No tag was moved.
- Large generated artifacts remain outside git and must not be committed.

## v1.3 summary - 2026-06-23 - Two-split formal evidence

Purpose:

- Freeze the v1.3 evidence from split_A and split_B before running additional
  LOO audits.
- Record the current interpretation conservatively: two non-overlapping case
  splits improved, but full case-level generalization is not yet proven.
- Define a risk-prioritized LOO plan instead of immediately running full LOO or
  changing hyperparameters.

Included:

- Added `docs/query_point_v1_3_two_split_formal_summary.md`.
- The summary records the fixed strategy, split_A/split_B metrics, per-case
  attribution, and LOO priority plan.

Two-split result:

- split_A:
  `train_cases=[19,25,31,43,44,45,46,50]`,
  `val_cases=[41,49]`,
  `validation_is_overlapping=false`.
  v1.2 `val_LE_rel=5.137640845816213`;
  v1.3 best `val_LE_rel=2.1796218548207182`,
  `val_AD_B_rel=0.5322959397919018`,
  `val_zero_q_LE_pred_rms=1.1484941768098402e-11`.
- split_B:
  `train_cases=[19,25,31,41,43,45,49,50]`,
  `val_cases=[44,46]`,
  `validation_is_overlapping=false`.
  v1.2 `val_LE_rel=8.515458607946384`;
  v1.3 best `val_LE_rel=1.5595633647976805`,
  `val_AD_B_rel=0.5125447167349023`,
  `val_zero_q_LE_pred_rms=1.157102562071606e-11`.

Per-case v1.3 attribution:

- split_A `case041`: `LE_rel=3.609960155721904`,
  `AD_B_rel=0.5321915350416156`,
  `zero_q_LE_pred_rms=1.1484941768098398e-11`,
  `Bprior_q_LE_rel=3.8535600056709796`.
- split_A `case049`: `LE_rel=1.8090445751604918`,
  `AD_B_rel=0.5324002526523295`,
  `zero_q_LE_pred_rms=1.1484941768098398e-11`,
  `Bprior_q_LE_rel=1.8501516802362072`.
- split_B `case044`: `LE_rel=1.2623462866926172`,
  `AD_B_rel=0.512531727633403`,
  `zero_q_LE_pred_rms=1.157102562071606e-11`,
  `Bprior_q_LE_rel=1.2603167162795477`.
- split_B `case046`: `LE_rel=1.7080236136717677`,
  `AD_B_rel=0.5125577066694196`,
  `zero_q_LE_pred_rms=1.157102562071606e-11`,
  `Bprior_q_LE_rel=1.706468997850503`.

Current interpretation:

- v1.3 anchored head materially improves held-out `LE` on split_A and split_B.
- The zero-q ghost LE field is structurally removed.
- AD-B is not broken: validation AD-B metrics remain close to v1.2.
- Best epochs are early (`split_A` epoch 10, `split_B` epoch 1), so epoch
  sensitivity remains part of the evidence.
- Full case-level generalization is not yet proven because `case019`,
  `case025`, `case031`, `case043`, `case045`, and `case050` have not yet been
  held out under v1.3.

LOO priority plan:

- Priority 1: `LOO_case031`, because `case031` is the high-amplitude / high-LE
  case and tests whether the anchored head can extrapolate without that training
  anchor.
- Priority 2: `LOO_case050`, because it is a mixed / larger fresh direction and
  tests non-low-amplitude mixed-direction generalization.
- Priority 3: `LOO_case043` or `LOO_case045`, to add medium fresh-direction
  evidence.
- Priority 4: full 10-case LOO for a final formal generalization claim.

Validation boundary:

- This step did not run additional training.
- No model, loss, learning-rate, epoch, split-policy, or data-pool changes were
  made.
- No old TRUE176 `LE/B` labels were used as v1.2/v1.3 labels.
- No tag was moved.
- Large generated artifacts remain outside git and must not be committed.

## v1.3 audit - 2026-06-23 - Anchored LOO case031

Purpose:

- Run the first risk-prioritized LOO audit after the two-split summary.
- Hold out `case031`, the high-amplitude / high-LE case, to test whether v1.3
  can extrapolate without that case as a training anchor.
- Keep the fixed v1.3 strategy unchanged.

Inputs:

- Compact pool:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt`
- Validation split:
  `train_cases=[19,25,41,43,44,45,46,49,50]`,
  `val_cases=[31]`.
- `validation_is_overlapping=false`, `train_frames=90`, `val_frames=10`.

Outputs:

- Training:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case031`
- Case attribution:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case031_case_attribution`
- B@q diagnostic:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case031_bq_anchor_oracle\LOO_case031_best`
- Comparison:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case031_comparison`
- Documentation:
  `docs/query_point_v1_3_anchored_LOO_case031_audit.md`

Key result:

- Best and latest are both epoch 50:
  `val_LE_rel=0.9255415441048325`,
  `train_LE_rel=0.4854010592101344`,
  `val_AD_B_rel=0.7236396387001852`,
  `val_AD_B_cos=0.7363247525021522`,
  `b_prior_current_val_evalcols_B_rel=0.7239878680693814`,
  `val_zero_q_LE_pred_rms=2.3911419659618957e-13`,
  `val_Bprior_q_LE_rel=0.9346750750874383`,
  `val_model_minus_Bprior_offset_rms=0.0009073160436565322`.
- Per-case attribution for `case031`:
  `LE_rel=0.9255415441048325`,
  `AD_B_rel=0.7236396387001852`,
  `B_prior_rel=0.7239878680693814`,
  `zero_q_LE_pred_rms=2.3911419659618957e-13`,
  `Bprior_q_LE_rel=0.9346750750874383`,
  `model_minus_Bprior_offset_rms=0.0009073160436565322`.
- B@q diagnostic:
  `Btrue_q_LE_rel=0.6522641115009261`,
  `Bprior_q_LE_rel=0.967480304420319`,
  `model_LE_rel=0.9255415441048325`,
  `zero_q_pred_rms=3.720241450096156e-13`,
  `current_offset_rms=0.0009074376473615923`,
  `true_offset_rms=0.008277004439016715`.

Current interpretation:

- `LOO_case031` does not collapse on held-out high-amplitude LE:
  `val_LE_rel` is below 1.0.
- The zero-q ghost remains structurally removed.
- AD-B and the B prior are substantially harder than in split_A/split_B:
  `val_AD_B_rel=0.7236` versus about `0.53` and `0.51` in split_A/split_B.
- `Btrue @ q` is not a near-perfect value oracle on `case031`
  (`Btrue_q_LE_rel=0.6523`), which is consistent with stronger high-amplitude
  nonlinearity or state/path effects.
- This supports the anchored head as a useful structural fix while exposing a
  remaining high-amplitude B coverage / state-dependence risk.
- Recommended next audit: run `LOO_case050` with the same fixed strategy before
  changing model, loss, learning rate, or epoch count.

Validation boundary:

- No model, loss, learning-rate, epoch, split-policy, or data-pool changes were
  made.
- No old TRUE176 `LE/B` labels were used as v1.2/v1.3 labels.
- No tag was moved.
- Large generated artifacts remain outside git and must not be committed.

## v2 coordinate audit - 2026-06-23 - One-case oracle / tiny training smoke

Purpose:

- Extend the `case050` v2b local-strain pilot into a minimal v2c smoke.
- Check whether the v2 coordinate contract can be read by a script-local
  network, differentiated with respect to `q_useful`, and mapped back to raw
  Abaqus B.
- Keep this as a one-case smoke, not formal training and not a multi-case
  generalization result.

Inputs:

- v2b compact:
  `D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\case050\complete_case050_v2b_local_strain_pilot.npz`
- Oracle output:
  `D:\IS-FEM\outputs\query_point_v2_tiny_smoke\case050_oracle\oracle_summary.json`
- Tiny-train output:
  `D:\IS-FEM\outputs\query_point_v2_tiny_smoke\case050_tiny_train\tiny_train_summary.json`
- Documentation:
  `docs/query_point_v2_one_case_tiny_smoke.md`

Oracle result:

- `LE_local_Bq_rel=0.04432110494813806`
- `LE_local_Bq_cos=0.9990565149944766`
- `B_raw_hat_projected_rel=4.573742799247327e-16`
- `B_raw_hat_raw_rel=0.00014709789545713997`
- `B_rigid_residual_rel=0.00014709789545713607`
- `zero_q_local_LE_rms_by_oracle=0.0`

Tiny model:

- Branch input: `q_useful`.
- Trunk input: `ip_xi`.
- Output: `LE_local_jacobian_frame`.
- AD target: `B_standard_useful`, whose actual coordinate is
  `local_jacobian_frame`.
- Raw back-projection:
  `B_raw_hat_model = T_eps_to_abq @ AD_B_local_hat @ T_q_raw_to_useful`.
- Model form:
  `LE_hat = B_prior_table(point) @ q_useful + R(q_useful,ip_xi) - R(0,ip_xi)`.
- `B_prior_table(point)` is initialized from one-case mean `B_standard_useful`.
  This is a smoke-test choice, not a formal architecture decision.

Tiny training result:

- `best_step=1800`
- `train_LE_local_rel=0.0070029981434345245`
- `train_AD_B_local_rel=0.004019410815089941`
- `train_AD_B_local_cos=0.9999919533729553`
- `zero_q_LE_local_rms=0.0`
- `B_model_raw_projected_rel=0.004044899716973305`
- `B_model_raw_rel=0.004047574009746313`

Current interpretation:

- The v2c one-case chain is executable:
  `q_useful + ip_xi -> LE_local_jacobian_frame -> AD_B_local_useful -> B_raw_hat_model`.
- The tiny model can overfit `case050` in local strain coordinates and keep a
  low AD-B/raw-backprojection error.
- This does not prove formal v2 performance, multi-case generalization, or a
  full covariant standard-coordinate strain formulation.

Validation boundary:

- No formal model training was performed.
- No v1.3/v2 production model, loss, learning-rate, epoch policy, split policy,
  or tag was changed.
- No old TRUE176 `LE/B` labels were used as v2 labels.
- No `.npz`, `.odb`, `.pt`, `.pth`, checkpoint, or output artifact is committed.

## v2 coordinate audit - 2026-06-23 - Multi-case v2b compact generation

Purpose:

- Move from one-case v2c smoke to multi-case v2 compact generation planning and
  execution.
- Convert the current 10-case strict fresh pool into v2b local-strain compacts.
- Verify every generated compact with strict-v2 coordinate audit before any
  formal v2 multi-case training.

Inputs:

- Source manifest:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt`
- Cases:
  `[19,25,31,41,43,44,45,46,49,50]`
- Output root:
  `D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10`
- Audit root:
  `D:\IS-FEM\outputs\query_point_v2_coordinate_contract_audit\multi_case_v2b_min10`
- Documentation:
  `docs/query_point_v2_multi_case_compact_generation_plan.md`

Implementation:

- Added `scripts/build_v2_multi_case_compacts.py`.
- Added `scripts/summarize_v2_multi_case_contract.py`.
- The builder reuses the one-case v2a and v2b payload builders instead of
  changing the formal training code.
- `X_keep_ref` is required for q48 control-node coordinates.  Missing
  `X_keep_ref` fails the case; q48 node order is not guessed.

Result:

- Dry-run passed:
  `compact_count=10`, `fail_count=0`.
- Actual generation passed:
  `compact_count=10`, `pass_count=10`, `fail_count=0`.
- v2b compact list:
  `D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt`
- Multi-case summary:
  `D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_multi_case_summary.json`

Key metric ranges:

- `q_useful_removed_rigid_rel`: `0.18971368414698137 .. 0.7451366595918001`
- `B_rigid_residual_rel`: `0.0001096826385160804 .. 0.0012638931647260521`
- `B_local_chain_rule_projected_rel`: max `4.619812329621079e-16`
- `LE_local_roundtrip_rel`: max `3.0740682511657065e-16`

Current interpretation:

- The 10-case v2b compact generation route is established.
- The strict coordinate chain passes for every case:
  `q48_raw -> q_useful`, `LE_Abaqus_global <-> LE_local_jacobian_frame`, and
  `B_local_useful -> B_raw_hat`.
- Some cases have large removed rigid content in q48, especially `case041`
  (`q_useful_removed_rigid_rel=0.7451`).  This is not a strict-v2 failure, but
  should be tracked in future v2 training interpretation.
- This prepares v2 formal tiny multi-case training smoke; it does not prove v2
  model performance or generalization.

Validation boundary:

- No model training was performed.
- No formal model, loss, learning-rate, epoch, split policy, or tag was changed.
- No old TRUE176 `LE/B` labels were used as v2 labels.
- Generated `.npz` compacts and audit outputs remain under `D:\IS-FEM\outputs`
  and are not committed.

## v2 coordinate audit - 2026-06-23 - Multi-case tiny training smoke

Purpose:

- Run a lightweight v2e smoke on the 10-case v2b compact pool.
- Verify multi-case v2 compact loading, case split, local LE supervision,
  AD-B-local supervision, raw-B backprojection metrics, and per-case attribution.
- Keep this as a script-local smoke, not formal v2 training.

Inputs:

- v2b compact list:
  `D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt`
- Output:
  `D:\IS-FEM\outputs\query_point_v2_tiny_smoke\multi_case_min10_smoke`
- Documentation:
  `docs/query_point_v2_multi_case_tiny_smoke.md`

Split:

- `train_cases=[19,25,31,41,43,45,49,50]`
- `val_cases=[44,46]`
- `validation_is_overlapping=false`

Tiny model:

- Branch input: `q_useful`.
- Trunk input: `ip_xi`.
- Output: `LE_local_jacobian_frame`.
- AD target: `d(LE_local_jacobian_frame)/d(q_useful)`.
- Raw backprojection:
  `B_raw_hat_model = T_eps_to_abq @ AD_B_local_hat @ T_q_raw_to_useful`.
- `case_id` is not used as input.
- `B_prior_table(point)` is initialized from train-case mean `B_standard_useful`.
  This is a smoke-test choice, not a formal architecture decision.

Best metrics:

- `best_step=0`
- `train_LE_local_rel=29.468563079833984`
- `val_LE_local_rel=10.836355209350586`
- `train_AD_B_local_rel=0.29527074098587036`
- `val_AD_B_local_rel=0.1957888901233673`
- `train_AD_B_local_cos=0.9554136395454407`
- `val_AD_B_local_cos=0.9939731955528259`
- `train_B_model_raw_projected_rel=0.2953280210494995`
- `val_B_model_raw_projected_rel=0.1954159140586853`
- `zero_q_LE_local_rms=0.0`

Latest metrics:

- `latest_step=3000`
- `train_LE_local_rel=1.1830968856811523`
- `val_LE_local_rel=18.03921890258789`
- `train_AD_B_local_rel=0.3041848838329315`
- `val_AD_B_local_rel=0.2089325487613678`
- `train_AD_B_local_cos=0.9526196718215942`
- `val_AD_B_local_cos=0.9906823635101318`
- `train_B_model_raw_projected_rel=0.30487608909606934`
- `val_B_model_raw_projected_rel=0.2097579836845398`
- `train_B_model_raw_rel=0.30487629771232605`
- `val_B_model_raw_rel=0.209757998585701`
- `zero_q_LE_local_rms=0.0`

Per-case attribution at latest step:

- Val `case044`:
  `LE_local_rel=16.429887771606445`,
  `AD_B_local_rel=0.20892377197742462`,
  `AD_B_local_cos=0.9906876087188721`,
  `B_model_raw_projected_rel=0.20974159240722656`.
- Val `case046`:
  `LE_local_rel=18.913543701171875`,
  `AD_B_local_rel=0.20894134044647217`,
  `AD_B_local_cos=0.9906772375106812`,
  `B_model_raw_projected_rel=0.20977436006069183`.

Current interpretation:

- The v2e execution chain passes:
  multi-case v2b compact list -> non-overlapping case split ->
  `q_useful + ip_xi -> LE_local_jacobian_frame` -> AD-B-local -> raw-B
  backprojection -> per-case attribution.
- The tiny no-case-id model reduces train LE strongly:
  `train_LE_local_rel: 29.47 -> 1.18`.
- Validation LE worsens:
  `val_LE_local_rel: 10.84 -> 18.04`.
- Full AD-B metrics remain computable but do not improve:
  `train_AD_B_local_rel: 0.295 -> 0.304`,
  `val_AD_B_local_rel: 0.196 -> 0.209`.
- Therefore v2e proves execution readiness, not v2 performance or
  generalization.

Recommended next step:

- Move to formal v2 model and normalization design:
  `q_useful/LE_local/B_local` normalization, architecture choice, possible
  geometry/load descriptors, and objective/checkpoint design.
- Do not treat this smoke as a reason for ad hoc hyperparameter tuning.

Validation boundary:

- No formal model training was performed.
- No v1.3/v2 production model, loss, learning-rate, epoch, split policy, or tag
  was changed.
- No old TRUE176 `LE/B` labels were used as v2 labels.
- Generated metrics remain under `D:\IS-FEM\outputs` and are not committed.

## v2 design audit - 2026-06-23 - Formal normalization and model design

Purpose:

- Analyze the 10-case v2b compact pool after v2e showed the engineering chain
  passes but tiny no-case-id generalization is not established.
- Identify q/LE/B scale mismatches before implementing a formal v2 model.
- Define the recommended normalization, architecture, and checkpoint scheme.

Inputs:

- v2b compact list:
  `D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt`
- Stats output:
  `D:\IS-FEM\outputs\query_point_v2_design_audit\multi_case_min10_stats\stats_summary.json`
- Documentation:
  `docs/query_point_v2_formal_model_normalization_design.md`

Key q statistics:

- `q_useful_rms=0.011491464788205397`
- `q_norm_mean=0.024060026482276273`
- `q_norm_max=0.3783465445747462`
- `q component std ratio=214.9691629443162`
- `q covariance rank estimate=10`
- `q covariance condition estimate=3705605606.4196844`
- `train q_norm_mean / val q_norm_mean=27.98834726646046`

Key LE statistics:

- `LE_local_rms=0.004063454592726386`
- `LE component scale ratio=2.0776393847256514`
- `train LE_local_rms / val LE_local_rms=41.8936927649456`

Key B statistics:

- `B_local_rms=8.95431355354351`
- `B component scale ratio=5.914964821491512`
- `B q-column scale ratio=9.299907348162048`
- `train B_local_rms / val B_local_rms=0.8704529505657129`

B@q oracle:

- `global LE_local_Bq_rel=0.6406777204317293`
- `global LE_local_Bq_cos=0.9222083945148773`
- `case044 LE_local_Bq_rel=0.049291182462898236`
- `case046 LE_local_Bq_rel=0.02763180576935294`
- `case031 LE_local_Bq_rel=0.6423729454305975`
- `Bq_rel_vs_q_norm_mean=0.9981798243696219`
- `Bq_rel_vs_LE_local_rms=0.9984726534079255`

Current interpretation:

- v2e validation cases are much lower amplitude than the training distribution:
  train/val q scale differs by about 28x and LE scale by about 42x.
- B scale is similar between train and val, and train-mean B prior is reasonably
  close to val B (`~0.196` relative), so val LE failure is not primarily a B
  coordinate-contract failure.
- The 42-dimensional `q_useful` space is under-covered by the current 10 cases:
  covariance rank estimate is 10.
- B@q oracle is strong on low-amplitude cases and weak on high-amplitude
  `case031`, indicating amplitude/nonlinearity effects.

Recommended normalization:

- Use a minimal A+C hybrid:
  q component standardization, LE six-component standardization, induced B
  normalization `B_norm[a,k] = B_local[a,k] * q_std[k] / LE_std[a]`, and an
  explicit `q_amp` descriptor.
- Optionally add `q_dir` with a zero-q guard.
- Do not rely only on pure case-amplitude normalization; keep amplitude
  available to the model.

Recommended architecture:

- Branch: `q_useful_normed + q_amp (+ optional q_dir)`.
- Trunk: `ip_xi (+ optional ip_detJ / ip_J / local-frame geometry descriptors)`.
- Output: `LE_local_jacobian_frame`.
- Structure:
  `LE_hat = B_prior_local(point,geom) @ q_useful + gate(q_amp) * (R(q,point,geom)-R(0,point,geom))`.
- Preserve `q_useful=0 => LE_local=0`.
- Evaluate AD-B-local and raw-B backprojection as before.

Recommended checkpoint score:

```text
score =
  1.0 * val_LE_local_rel
+ 1.0 * val_AD_B_local_rel
+ 0.5 * val_B_model_raw_projected_rel
+ 10.0 * zero_q_rms_norm
```

Also record `best_LE`, `best_B`, `best_raw_projected`, `best_combined`, and
`latest`.

Next step:

- v2g formal prototype implementation:
  train-only normalization artifact, explicit `q_amp`, anchored zero-q model,
  normalized LE/B losses, and raw-B backprojection evaluation.

Validation boundary:

- No formal model training was performed.
- No v1.3/v2 production model, loss, learning-rate, epoch, split policy, or tag
  was changed.
- No old TRUE176 `LE/B` labels were used as v2 labels.
- Generated stats remain under `D:\IS-FEM\outputs` and are not committed.

## v2 planning - 2026-06-23 - Coordinate-consistent route contract

Purpose:

- Freeze the conceptual transition from v1.3 raw-q diagnostics to a v2
  coordinate-consistent finite-element operator route.
- Record that the final route should learn a standard/local macro-element
  operator, not only a raw Abaqus-coordinate black-box map.
- Add a read-only contract audit that identifies which current/future compacts
  contain the fields needed for v2 chain-rule-consistent B supervision.

Key route change:

```text
q48_raw
  -> q_useful = T_q_raw_to_useful @ q48_raw
  -> NN(q_useful, xi_standard, geometry_features)
  -> strain_standard_or_local
  -> LE_Abaqus
```

The v2 B comparison must use:

```text
B_raw_hat = T_eps_to_abq @ d(strain_standard)/d(q_useful) @ T_q_raw_to_useful
```

and may not directly compare `d(strain_standard)/d(q_useful)` against the
current raw Abaqus/Sobolev label:

```text
B_LE128_forward = d(LE_Abaqus)/d(q48_raw)
```

Added:

- `docs/query_point_v2_coordinate_consistent_route.md`
- `scripts/audit_v2_coordinate_consistent_contract.py`
- v2 gate wording in `docs/query_point_abaqus_workflow.md`

Probe result on the current 10-case strict fresh pool:

```powershell
py -3 scripts\audit_v2_coordinate_consistent_contract.py `
  --compact-list D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v2_coordinate_contract_audit\v1_2_min10_probe
```

- `compact_count=10`
- `strict_v2_coordinate_pass=false`
- `strict_v2_coordinate_pass_count=0`
- The existing v1.x compacts already have valid standard/IP geometry fields:
  `ip_xi_ok=true`, `ip_J_ok=true`, `ip_invJ_ok=true`, `ip_detJ_ok=true`.
- All 10 fail the new v2 coordinate-contract fields:
  `missing_q_useful`, `missing_T_q_raw_to_useful`,
  `missing_strain_output_coordinate`, `missing_q_useful_coordinate`,
  `missing_B_label_q_coordinate`, `missing_B_label_output_coordinate`,
  `missing_T_eps_to_abq_or_B_standard_useful`, and
  `missing_B_chain_rule_metadata`.
- `--strict-v2` was checked on one current compact and failed nonzero as
  expected.

Current interpretation:

- v1.3 remains useful diagnostic evidence: strict compacts, case splits,
  anchored zero-q behavior, and AD-B stability are not discarded.
- v1.3 is not the final clean operator contract because the branch coordinate,
  strain coordinate, and B-label coordinate are still raw-coordinate coupled.
- The next scientific step is a one-case v2 coordinate-chain audit, not another
  long v1.3 training run.

Validation boundary:

- No model, loss, learning-rate, epoch, split-policy, or tag changes were made.
- No old TRUE176 `LE/B` labels were used as v2 labels.
- The new audit script is read-only and does not train or transform labels.
- Existing v1.x compacts are expected to fail strict v2 until `q_useful`,
  `T_q_raw_to_useful`, strain-coordinate metadata, and chain-rule metadata are
  generated.

## v2a pilot - 2026-06-23 - One-case q-coordinate consistency

Purpose:

- Build a one-case q-coordinate pilot for the v2 route without training.
- Isolate the branch-coordinate transform before attempting standard/local
  strain labels.
- Use `case050` because prior v1.3 diagnostics showed strong data-side
  `Btrue @ q` while still exposing value-field generalization limits.

Input:

- Source compact:
  `D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case050\complete\complete_case050_training_ready.npz`
- q48 node coordinate source:
  `X_keep_ref [16,3]`

Added:

- `scripts/build_v2_q_useful_pilot_compact.py`
- `docs/query_point_v2_one_case_q_useful_pilot.md`
- The v2 audit now recognizes `B_useful_abq` as the v2a useful-B field name.

Generated output, not committed:

- `D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\case050\complete_case050_v2a_q_useful_pilot.npz`
- `D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\case050\complete_case050_v2a_q_useful_pilot.summary.json`

Method:

```text
R = six rigid modes from X_keep_ref:
    translations x/y/z
    small rotations x/y/z with u_i = omega cross (X_i - centroid)
N = orthogonal complement of R, shape [48,42]
T_q_raw_to_useful = N.T
q_useful = q48_raw @ T_q_raw_to_useful.T
P_useful = N @ N.T
```

This pilot keeps output strain in Abaqus global coordinates:

```text
strain_output_coordinate = abaqus_global
T_eps_to_abq = I_6
B_useful_abq = B_LE128_forward @ N
B_raw_hat = B_useful_abq @ T_q_raw_to_useful
```

Key result:

- `q48_shape=[10,48]`
- `q_useful_shape=[10,42]`
- `T_q_raw_to_useful_shape=[42,48]`
- `B_useful_abq_shape=[10,128,6,42]`
- `rigid_mode_rank=6`
- `rigid_annihilation_max=1.371671448566352e-16`
- `useful_basis_orthonormal_max=4.440892098500626e-16`
- `q_useful_reconstruction_rel=0.33576244788862675`
- `q_useful_removed_rigid_rel=0.33576244788862675`
- `B_chain_rule_projected_rel=4.100651493362974e-16`
- `B_chain_rule_projected_max_abs=1.4210854715202004e-13`
- `B_chain_rule_raw_rel=0.0001470978954571399`
- `B_chain_rule_raw_max_abs=0.019154849670568908`
- `B_rigid_residual_rel=0.00014709789545713607`

Strict v2 audit:

```powershell
py -3 scripts\audit_v2_coordinate_consistent_contract.py `
  --compact D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\case050\complete_case050_v2a_q_useful_pilot.npz `
  --out-root D:\IS-FEM\outputs\query_point_v2_coordinate_contract_audit\case050_v2a_q_useful_pilot `
  --strict-v2
```

Result:

- `strict_v2_coordinate_pass=true`
- `strict_v2_coordinate_pass_count=1`
- `failure_counts={}`

Current interpretation:

- The v2a q-coordinate chain closes for `case050`.
- About 33.6 percent of raw q is removed as rigid-body content, which confirms
  that raw-q contains non-useful motion.
- The raw Abaqus B response in removed rigid directions is tiny
  (`B_rigid_residual_rel=1.47e-4`), so the useful-q projection does not remove
  meaningful strain sensitivity for this case.
- The projected B chain rule is exact to numerical precision.
- The next step should be a separate v2b strain-coordinate/local-coordinate
  pilot, not immediate v2 training.

Validation boundary:

- No model, loss, learning-rate, epoch, split-policy, or tag changes were made.
- No old TRUE176 `LE/B` labels were used as v2 labels.
- No standard/local strain labels were generated; this is Abaqus-global-strain
  q-coordinate pilot only.
- Generated `.npz` output remains under `outputs` and was not committed.

## v2b pilot - 2026-06-23 - One-case local strain coordinate

Purpose:

- Extend the v2a `case050` q-coordinate pilot with a local strain coordinate.
- Verify `LE_Abaqus_global <-> LE_local` and
  `B_local_useful -> B_raw_hat` roundtrips.
- Keep the scope to one case and do no training.

Input:

- v2a compact:
  `D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\case050\complete_case050_v2a_q_useful_pilot.npz`

Generated output, not committed:

- `D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\case050\complete_case050_v2b_local_strain_pilot.npz`
- `D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\case050\complete_case050_v2b_local_strain_pilot.summary.json`

Added:

- `scripts/build_v2_local_strain_pilot_compact.py`
- `docs/query_point_v2_one_case_local_strain_pilot.md`
- v2 route wording for `local_jacobian_frame`.
- The v2 audit now recognizes `B_standard_useful` / `B_local_useful` before
  `B_useful_abq`, so v2b manifests report the local useful-B field.

Method:

```text
local_frame_Q from ip_J Gram-Schmidt
local_frame_ip_J_axis_convention = rows
strain_voigt_order = LE11_LE22_LE33_LE12_LE13_LE23
strain_shear_convention = tensor_shear_not_engineering_gamma

E_local = Q.T @ E_abq @ Q
E_abq   = Q @ E_local @ Q.T

B_local_useful = T_eps_from_abq @ B_useful_abq
B_raw_hat = T_eps_to_abq @ B_local_useful @ T_q_raw_to_useful
```

Key result:

- `local_frame_Q_shape=[128,3,3]`
- `local_frame_orthonormal_max=2.220446049250313e-16`
- `local_frame_det_min=0.9999999999999999`
- `local_frame_det_max=1.0000000000000002`
- `local_frame_orientation_min=0.9995424705243147`
- `local_frame_orientation_max=0.9995425972270762`
- `T_eps_to_abq_shape=[128,6,6]`
- `T_eps_from_abq_shape=[128,6,6]`
- `T_eps_roundtrip_local_rel=5.874317750795854e-15`
- `T_eps_roundtrip_abq_rel=6.1175772104987605e-15`
- `LE_local_roundtrip_rel=1.7341868077270364e-16`
- `LE_local_roundtrip_max_abs=4.336808689942018e-19`
- `B_local_chain_rule_projected_rel=4.573742799247327e-16`
- `B_local_chain_rule_projected_max_abs=1.5631940186722204e-13`
- `B_local_chain_rule_raw_rel=0.00014709789545713997`
- `B_local_chain_rule_raw_max_abs=0.01915484967056802`

Strict v2 audit:

```powershell
py -3 scripts\audit_v2_coordinate_consistent_contract.py `
  --compact D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\case050\complete_case050_v2b_local_strain_pilot.npz `
  --out-root D:\IS-FEM\outputs\query_point_v2_coordinate_contract_audit\case050_v2b_local_strain_pilot `
  --strict-v2
```

Result:

- `strict_v2_coordinate_pass=true`
- `strict_v2_coordinate_pass_count=1`
- `failure_counts={}`
- Manifest useful-B field:
  `B_standard_useful [10,128,6,42]`
- `B_label_q_coordinate=q_useful`
- `B_label_output_coordinate=local_jacobian_frame`

Current interpretation:

- The v2b local strain coordinate chain closes for `case050`.
- `LE_Abaqus_global <-> LE_local_jacobian_frame` is lossless to numerical
  precision.
- `B_local_useful` maps back to projected raw Abaqus B to numerical precision.
- The remaining raw-space B difference is the same rigid-direction residual
  already measured in v2a.
- This is still a local Jacobian-frame tensor-component pilot, not yet a full
  covariant standard-coordinate strain formulation.
- Next step: v2c one-case tiny oracle or tiny training smoke, still before any
  formal v2 training pool.

Validation boundary:

- No model, loss, learning-rate, epoch, split-policy, or tag changes were made.
- No old TRUE176 `LE/B` labels were used as v2 labels.
- Generated `.npz` output remains under `outputs` and was not committed.

## v1.3 audit - 2026-06-23 - Anchored LOO case043

Purpose:

- Run the third risk-prioritized LOO audit after `LOO_case031` and
  `LOO_case050`.
- Hold out `case043`, a medium fresh direction, to test whether the anchored
  strategy remains stable between the high-amplitude and mixed/larger checks.
- Keep the fixed v1.3 strategy unchanged.

Inputs:

- Compact pool:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt`
- Validation split:
  `train_cases=[19,25,31,41,44,45,46,49,50]`,
  `val_cases=[43]`.
- `validation_is_overlapping=false`, `train_frames=90`, `val_frames=10`.

Outputs:

- Training:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case043`
- Case attribution:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case043_case_attribution`
- B@q diagnostic:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case043_bq_anchor_oracle\LOO_case043_best`
- Comparison:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case043_comparison`
- Documentation:
  `docs/query_point_v1_3_anchored_LOO_case043_audit.md`

Key result:

- Best checkpoint, epoch 10:
  `val_LE_rel=1.8380485229166579`,
  `train_LE_rel=0.74541650372748`,
  `val_AD_B_rel=0.5036482686290428`,
  `val_AD_B_cos=0.8830602940469523`,
  `b_prior_current_val_evalcols_B_rel=0.503501472353476`,
  `val_zero_q_LE_pred_rms=3.863962753320447e-12`,
  `val_Bprior_q_LE_rel=1.9000499821060757`,
  `val_model_minus_Bprior_offset_rms=0.00016322065296508765`.
- Latest checkpoint, epoch 50:
  `val_LE_rel=1.9656412310345146`,
  `val_AD_B_rel=0.5016945264798282`,
  `b_prior_current_val_evalcols_B_rel=0.5007423836646588`,
  `val_zero_q_LE_pred_rms=3.863962753320447e-12`.
- Per-case attribution for `case043`:
  `LE_rel=1.8380485229166579`,
  `AD_B_rel=0.5036482686290428`,
  `B_prior_rel=0.503501472353476`,
  `zero_q_LE_pred_rms=3.863962753320447e-12`,
  `Bprior_q_LE_rel=1.9000499821060757`,
  `model_minus_Bprior_offset_rms=0.00016322065296508765`.
- B@q diagnostic:
  `Btrue_q_LE_rel=0.10034757778215131`,
  `Bprior_q_LE_rel=1.8931336235552196`,
  `model_LE_rel=1.8380485229166579`,
  `zero_q_pred_rms=0.0`,
  `current_offset_rms=0.00021721550419227364`,
  `true_offset_rms=2.075807588384023e-05`.

Current interpretation:

- `LOO_case043` keeps the zero-q ghost structurally removed.
- AD-B and the query B prior remain stable:
  `val_AD_B_rel=0.5036`, `b_prior_current_val_evalcols_B_rel=0.5035`.
- `val_LE_rel=1.8380` remains high, so medium fresh-direction LE value
  generalization is not solved.
- `Btrue @ q` is strong on `case043` (`Btrue_q_LE_rel=0.1003`), so the data-side
  q/LE/B contract is not the bottleneck; the remaining issue is value-field
  generalization.
- Best is epoch 10 and latest is worse, so epoch sensitivity remains visible.
- Recommended next audit: run `LOO_case045` with the same fixed strategy, or
  write a partial LOO summary before deciding whether to expand data or revisit
  the value branch.

Validation boundary:

- No model, loss, learning-rate, epoch, split-policy, or data-pool changes were
  made.
- No old TRUE176 `LE/B` labels were used as v1.2/v1.3 labels.
- No tag was moved.
- Large generated artifacts remain outside git and must not be committed.

## v1.3 audit - 2026-06-23 - Anchored LOO case050

Purpose:

- Run the second risk-prioritized LOO audit after `LOO_case031`.
- Hold out `case050`, a mixed / larger fresh direction, to test mixed-direction
  held-out generalization.
- Keep the fixed v1.3 strategy unchanged.

Inputs:

- Compact pool:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt`
- Validation split:
  `train_cases=[19,25,31,41,43,44,45,46,49]`,
  `val_cases=[50]`.
- `validation_is_overlapping=false`, `train_frames=90`, `val_frames=10`.

Outputs:

- Training:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case050`
- Case attribution:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case050_case_attribution`
- B@q diagnostic:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case050_bq_anchor_oracle\LOO_case050_best`
- Comparison:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case050_comparison`
- Documentation:
  `docs/query_point_v1_3_anchored_LOO_case050_audit.md`

Key result:

- Best checkpoint, epoch 1:
  `val_LE_rel=1.5294014338724993`,
  `train_LE_rel=1.6161069592630004`,
  `val_AD_B_rel=0.517000373997325`,
  `val_AD_B_cos=0.8751722384624953`,
  `b_prior_current_val_evalcols_B_rel=0.5169978402891399`,
  `val_zero_q_LE_pred_rms=1.1676963344145722e-11`,
  `val_Bprior_q_LE_rel=1.5326505786455733`,
  `val_model_minus_Bprior_offset_rms=3.92915279653482e-06`.
- Latest checkpoint, epoch 50:
  `val_LE_rel=1.7229821122212494`,
  `val_AD_B_rel=0.5145903038519773`,
  `b_prior_current_val_evalcols_B_rel=0.5137761146056249`,
  `val_zero_q_LE_pred_rms=1.1676963344145722e-11`.
- Per-case attribution for `case050`:
  `LE_rel=1.5294014338724993`,
  `AD_B_rel=0.517000373997325`,
  `B_prior_rel=0.5169978402891399`,
  `zero_q_LE_pred_rms=1.1676963344145722e-11`,
  `Bprior_q_LE_rel=1.5326505786455733`,
  `model_minus_Bprior_offset_rms=3.92915279653482e-06`.
- B@q diagnostic:
  `Btrue_q_LE_rel=0.06644355125974967`,
  `Bprior_q_LE_rel=1.5310648298695695`,
  `model_LE_rel=1.5294014338724993`,
  `zero_q_pred_rms=0.0`,
  `current_offset_rms=4.327757044449705e-06`,
  `true_offset_rms=1.843403091453214e-05`.

Current interpretation:

- `LOO_case050` does not collapse on the held-out mixed / larger fresh direction.
- The zero-q ghost remains structurally removed.
- AD-B and the query B prior remain stable and close to split_B:
  `val_AD_B_rel=0.5170`, `b_prior_current_val_evalcols_B_rel=0.5170`.
- `val_LE_rel=1.5294` is comparable to split_B and better than split_A, but it
  is still above 1.0, so mixed-direction value generalization is not solved.
- `Btrue @ q` is strong on `case050` (`Btrue_q_LE_rel=0.0664`), so the data-side
  q/LE/B contract is not the bottleneck; the remaining issue is value-field
  generalization.
- Best is early (epoch 1) and latest is worse, so epoch sensitivity remains
  visible.
- Recommended next audit: run `LOO_case043` or `LOO_case045` before full LOO.

Validation boundary:

- No model, loss, learning-rate, epoch, split-policy, or data-pool changes were
  made.
- No old TRUE176 `LE/B` labels were used as v1.2/v1.3 labels.
- No tag was moved.
- Large generated artifacts remain outside git and must not be committed.
