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

The current priority is limited LE/B training plus mandatory post-training
mechanical audit. Gate 17 shows that a state-dependent B baseline is the
current promising direction, but it is still a prototype. Full tangent closure
remains pending.

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
| Gate 02 teacher closure | limited release basis | `LE128_base` and `B_LE128_forward` are credible, and teacher force closes with selected-frame IVOL; full teacher stiffness closure remains unresolved but does not block limited LE/B training. |
| Gate 03 Macro16 mechanics | split status | Selected-frame force closure passes on the current 10-case set; material-only K is diagnostic; full tangent closure remains incomplete. |
| Gate 04 geometry generality | PASS for main-route required families | Regular, light, medium, cylindrical, conical, thickness-varying, and mild double-curvature selected-frame force closure pass; repaired strong non-inverted robustness data also passes. |
| Gate 05 IP reduction | keep 128 active | `reports/05_ip_reduction_audit.md`; 96/64/32 candidates fail selected-frame force closure, 18 remains failure control. |
| Gate 06 training | LIMITED RELEASE | `docs/TRAINING_RELEASE_DECISION.md`; training may start only for LE/B learning, with mandatory selected-frame force closure after training. |
| Gate 16 q dependent B diagnosis | diagnosis PASS, large training release FAIL | `reports/16_q_dependent_b_diagnosis.md`; B varies with `q48_def_hat`; point-only B prior is insufficient; formal large training remains blocked. |
| Gate 17 state B prototype | prototype PASS, production release FAIL | `reports/17_state_b_baseline_prototype.md`; state-dependent B baseline reduces force rel to `0.0919`, below the prototype `0.10` target but above final `0.02`. |
| Gate 18 state B migration plan | PLAN PASS | `reports/18_state_b_migration_plan.md`; next step is guarded implementation of a new state B model variant, not formal large training. |
| Gate 19 state B main model smoke | implementation smoke PASS, numeric reproduction pending | `reports/19_state_b_main_model_smoke.md`; `le0-state-b` model variant, checkpoint path, and force-audit loading are implemented; Linux case031 first6 numeric reproduction still pending. |
| Gate 20 state B reproduction diagnosis | failure localized | `reports/20_state_b_reproduction_failure_diagnosis.md`; Gate 19 numeric reproduction failed because the migrated state B factorization was weaker than the Gate 17 prototype. Default state B kind was corrected to `point_q_rank`. |
| Gate 21 point q rank rerun | FAIL | `reports/21_state_b_point_q_rank_rerun.md`; Linux `point_q_rank` rerun still fails. Best model force rel is about `0.59` to `0.62`, while teacher force rel remains `0.01279`. Large training remains blocked. |
| Gate 22 state B equivalence diagnosis | diagnosis PASS, training release FAIL | `reports/22_state_b_equivalence_diagnosis.md`; main `le0-state-b` is not equivalent to Gate 17. The main path lacks fixed `static_b[128,6,48]`, and residual AD terms mix into B. Next step is a fixed-128 direct-B state baseline, not larger training. |
| Gate 23 fixed128 direct B training | FAIL, close but not released | `reports/23_fixed128_direct_b_training.md`; Linux rank 4/8/12 fixed128 direct-B training completed. Best force rel is `0.1073723868`, above the `0.10` prototype threshold; teacher force rel remains `0.0127911083`. Large training remains blocked. |

Training is allowed only within the limited LE/B release boundary.

## Verified Conclusions

Current verified or adopted conclusions:

- Macro16 v4 boundary route is the active route.
- The model input remains 48D; it must not be reduced to 42D.
- `q48_def_hat` and `X16_hat` are the training inputs.
- The model output is `LE`.
- `B` is obtained from automatic differentiation of `LE`.
- The current trusted compact family is regenerated Macro16 source128 compact
  data.
- Gate 06 has a limited training release for LE/B learning only.
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
- Current 96/64/32 reduced integration-point candidates fail Gate 05; keep 128
  points active.
- Gate 16 shows `B_macro_qdef` changes with `q48_def_hat`; a point-only
  `B_base(point)` prior has about `0.08` to `0.09` irreducible error on the
  case031 six-frame diagnosis.
- A linear q correction reduces B error to about `0.043` on the same diagnosis,
  so q dependent B is a real missing expression, not an ordering or data
  corruption issue.
- Gate 17 state-dependent B baseline prototype reduces case031 six-frame force
  rel from about `0.2107` to `0.0919` and B rel from about `0.0834` to
  `0.0474`.
- Gate 17 proves the direction but does not release the production model.
- Gate 18 records the safe migration plan: add a separate state-dependent B
  model variant, keep default LE0 route intact, and require a six-frame
  reproduction before any larger training.
- Gate 19 implements the separate `le0-state-b` model variant and verifies
  local smoke tests, while leaving the default `le0` route unchanged.
- Gate 20 shows the first `le0-state-b` numeric rerun failed, but the failure is
  localized to an under-expressive state B factorization rather than data,
  checkpoint loading, q ordering, LE ordering, or the 128 point rule.
- Gate 21 shows the corrected `point_q_rank` rerun also failed to reproduce the
  Gate 17 prototype. Teacher force still closes, but trained model force rel
  remains about `0.59` to `0.62`, so large training is still blocked.
- Gate 22 shows the reason: the main `le0-state-b` trainer is not equivalent to
  Gate 17. Gate 17 uses a fixed `static_b[128,6,48]` direct-B baseline, while
  the main Query model uses `global_b_norm[6,48]` plus point nets and residual
  AD contributions.
- Gate 23 implements a fixed-128 direct-B training path and runs Linux rank
  4/8/12 training. Train B improves to about `0.048`, but validation B remains
  about `0.107`, and best selected-frame force rel is `0.1073723868`, so it is
  still not released.

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
- `B_macro_qdef` uses the current small-rotation linear projection
  approximation; strict nonlinear Kabsch Jacobian remains a future audit topic.
- Full consistent tangent K is still an unresolved route decision or
  implementation task.
- The new `Macro16BoundaryDeepONetWithLE0StateB` implementation has only passed
  local smoke tests and failed the corrected Linux `point_q_rank` rerun. Gate 22
  localizes the mismatch to the trainer objective and B baseline structure, so
  it does not yet meet the final `0.02` force closure target.
- The fixed-128 direct-B variant is closer to Gate 17 but still fails the
  prototype force target. Gate 24 must align Gate 17 loss weighting,
  initialization, and checkpoint selection before any larger training.

These issues must not be hidden inside network training as if a network could
repair a definition mismatch. They do not block the limited LE/B training
release, but they still block any full tangent or solver-readiness claim.

## Current Gate 04 Wind-Turbine Shell Geometry Task

Gate 04 main-route required families now pass. Repaired strong non-inverted
distortion also passes as a non-blocking robustness boundary test.

Already audited:

- regular geometry selected-frame force closure;
- light distortion selected-frame force closure;
- medium distortion selected-frame force closure;
- cylindrical shell geometry selected-frame force closure;
- conical shell geometry selected-frame force closure;
- thickness-varying shell geometry selected-frame force closure;
- mild double-curvature shell geometry selected-frame force closure;
- repaired strong non-inverted robustness selected-frame effective force
  closure.

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

## Gate 06 Limited Training Release

The active release is:

```text
LE/B training only
```

Allowed:

- train `LE`;
- supervise `B = dLE / d(q48_def_hat)` through autograd;
- keep `q48_def_hat`, `X16_hat`, and 128 integration points.

Required after training:

- rerun selected-frame recovery-force closure;
- report material-only K as diagnostic only;
- do not claim full tangent closure unless a real full-tangent candidate is
  implemented and audited.

## Next Tasks

Recommended next tasks, in order:

1. Keep material-only K diagnostics separate from full tangent claims.
2. Implement Gate 23 fixed-128 direct-B state baseline as a separate model
   variant.
3. Keep migration staged behind tests and six-frame reproduction.
4. After any migrated prototype training, rerun selected-frame force closure before
   claiming mechanical usability.
5. Decide later whether the full consistent tangent requirement is implemented
   or explicitly waived with evidence.

## Currently Forbidden

Do not do the following now:

- Do not train outside the limited LE/B release boundary.
- Do not start a new integration-point reduction route as part of training.
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
- Do not claim full tangent closure from LE/B training.

Current route remains:

```text
Macro16 v4 boundary route, standard 128 integration points, limited LE/B
training release
```
