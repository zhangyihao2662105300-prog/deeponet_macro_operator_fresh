# Current Status

Role: Current Status Archivist

Status date: 2026-06-25

This document summarizes the current project state. It does not run code, train
a network, change the model, or modify audit scripts.

## Current Mainline

The active route is:

```text
Macro16 v4 boundary route
```

Mainline contract:

- 16 boundary control nodes.
- 48 boundary displacement DOFs.
- Model displacement input: `q48_def_hat`.
- Model geometry input: `X16_hat`.
- Model output: integration-point strain `LE`.
- `B` is obtained by automatic differentiation of `LE` with respect to
  `q48_def_hat`.
- Standard current integration rule: fixed 128 parent integration points.

TRUE176 / CSS8 128-IP is the teacher and audit baseline, not the final model
interface.

The current priority is mechanical assembly and geometry generality, not
network training.

## Standard Element

The standard Macro16 element is a fixed parent element:

- parent domain `(r, s, t) in [-1, 1]^3`;
- 8 Q8 serendipity surface nodes on the lower surface;
- matching 8 Q8 serendipity surface nodes on the upper surface;
- linear interpolation through thickness;
- fixed parent nodes and fixed parent integration-point coordinates;
- current standard point rule: 128 parent points.

The standard element does not change between flat, cylindrical, conical,
thickness-varying, double-curvature, or distorted shell geometries. Only the
physical mapping from `X16_raw` changes.

## Real Geometry

Real geometry is represented by the 16 physical boundary nodes:

```text
X16_raw
```

For each parent coordinate `(r, s, t)`, the physical position is obtained from
the Macro16 isoparametric map:

```text
x(r, s, t) = sum_i N_i(r, s, t) X16_raw_i
```

The model-visible geometry is:

```text
X16_hat = (X16_raw - X_center) / L_ref
```

The route must not expose fine-grid internal nodes, TRUE176 internal nodes, or
`X_macro` to the Macro16 model. Wind-turbine shell geometry families must be
represented by different valid `X16_raw` values with positive `detJ` at the
fixed parent integration points.

## Gate Status

| Gate | Current Status | Evidence / Note |
|---|---|---|
| Gate 01 data contract | PASS | `reports/macro16_source128_data_audit.md`; loader uses `q48_def_hat` and `B_macro_qdef`; input remains 48D. |
| Gate 02 teacher closure | FAIL / not cleared | `LE128_base` and `B_LE128_forward` are credible, and teacher force closes with selected-frame IVOL, but full teacher stiffness closure remains unresolved. |
| Gate 03 Macro16 mechanics | split status | Selected-frame force closure passes on the current 10-case set; material-only K is diagnostic; full tangent closure remains incomplete. |
| Gate 04 geometry generality | incomplete | Regular, light, and medium selected-frame force closure pass; cylindrical, conical, thickness-varying, and mild double-curvature shell families remain. |
| Gate 05 IP reduction | not started | Blocked until Gate 04 mainline passes; planned order is `128 -> 96 -> 64 -> 32 -> 18`. |
| Gate 06 training | blocked | Blocked by unresolved Gate 02/03/04/05 conditions. |

Training is not allowed yet.

## Verified Conclusions

Current verified or adopted conclusions:

- Macro16 v4 boundary route is the active route.
- The model input remains 48D; it must not be reduced to 42D.
- `q48_def_hat` and `X16_hat` are the training inputs.
- The model output is `LE`.
- `B` is obtained from automatic differentiation of `LE`.
- The current trusted compact family is regenerated Macro16 source128 compact
  data.
- `LE128_base` and `B_LE128_forward` are credible teacher labels; B closes
  against finite-difference LE perturbations.
- `LE_macro` and `B_macro_qraw` copy the source labels correctly for the
  localized case031 diagnosis.
- Selected-frame physical volume is the canonical audit volume for Abaqus RF
  force closure.
- `integration_weight_phys` remains the reference/standard Macro16 physical
  volume and must not be silently redefined.
- Material-only `B^T D B dV` is a diagnostic stiffness, not full Abaqus
  finite-difference tangent closure.
- The 18-point route is a known failed historical route and remains a failure
  control for future Gate 05 planning.

## Pending Issues That Are Not Data Errors

The following items are unresolved, but they are not evidence that the source
data is bad:

- Gate 02 full teacher stiffness closure is unresolved because material-only K
  is not the full Abaqus perturbed-reaction tangent.
- Gate 03 full tangent closure is incomplete because full consistent tangent
  terms have not been implemented or waived.
- The `case031` stiffness residual is a tangent-definition mismatch involving
  missing terms such as `dV/dq`, `dB/dq`, and geometric/stress stiffness.
- Material-only stiffness may remain above the old full-FD `0.02` threshold on
  large-response cases; this is now a named diagnostic, not proof of data
  corruption.
- Strong non-inverted distortion data currently duplicates the medium geometry;
  this is a non-blocking robustness-data issue, not a hard mainline Gate 04
  blocker.
- `B_macro_qdef` uses the current small-rotation linear projection
  approximation; strict nonlinear Kabsch Jacobian remains a future audit topic.
- Full consistent tangent K is still an unresolved route decision or
  implementation task.

These issues should not be routed into network training as if a network could
repair a definition mismatch.

## Current Gate 04 Wind-Turbine Shell Geometry Task

Gate 04 is the active mainline task before reduction or training.

Already audited:

- regular geometry selected-frame force closure;
- light distortion selected-frame force closure;
- medium distortion selected-frame force closure.

Still required for the main wind-turbine shell geometry route:

- cylindrical shell geometry;
- conical shell geometry;
- thickness-varying shell geometry;
- mild double-curvature shell geometry.

For each required geometry family, Gate 04 must verify:

- valid `X16_raw`;
- positive `detJ` at all fixed 128 parent integration points;
- unchanged 128 parent point rule;
- no `X_macro`, fine-grid internal nodes, or TRUE176 internal nodes as model
  input;
- selected-frame force closure against Abaqus RF;
- material-only K reported separately as diagnostic;
- no full tangent closure claim unless a real full-tangent candidate is
  audited.

The key question is whether the same fixed Macro16 parent element, mapped only
by `X16_raw`, preserves force closure across wind-turbine shell real
geometries.

## Next Tasks

Recommended next tasks, in order:

1. Continue Gate 04 mainline geometry audit on cylindrical, conical,
   thickness-varying, and mild double-curvature shell families.
2. For each new geometry family, rebuild or collect source128 compacts with
   valid `X16_raw` and selected-frame physical volume evidence.
3. Rerun data contract and selected-frame force closure checks for the Gate 04
   families.
4. Keep material-only K diagnostics separate from full tangent claims.
5. Decide whether the full consistent tangent requirement is implemented or
   explicitly waived with evidence.
6. Only after Gate 04 mainline passes, start Gate 05 IP reduction planning
   execution.
7. Keep Gate 06 training blocked until Gate 01-05 conditions are satisfied or
   formally waived where allowed.

## Currently Forbidden

Do not do the following now:

- Do not train the network.
- Do not start Gate 05 integration-point reduction.
- Do not change the model architecture.
- Do not change `q48` ordering.
- Do not change `LE` component ordering.
- Do not switch to active42.
- Do not expose `X_macro` to the Macro16 model.
- Do not expose fine-grid internal nodes or TRUE176 internal nodes to the
  Macro16 model.
- Do not change the model-visible input from 48D to 42D.
- Do not change the standard 128-point rule.
- Do not treat training loss as sufficient validation.
- Do not use material-only K as the full Abaqus FD tangent `0.02` gate.
- Do not silently redefine `integration_weight_phys` as selected-frame volume.
- Do not route volume/tangent definition mismatches into network training.

Current route remains:

```text
Macro16 v4 boundary route, standard 128 integration points
```
