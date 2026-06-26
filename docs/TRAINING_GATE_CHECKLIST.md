# Training Gate Checklist

Role: Training Gate Checklist Agent

Status: limited training release decision recorded

Gate 16 update: q dependent B diagnosis is complete. The limited training
release remains diagnostic only; formal large training is not released.

Gate 17 update: state-dependent B baseline prototype passed the `0.10` force
target, but production release is still not allowed.

Gate 18 update: state-dependent B migration plan is complete. The next allowed
step is a guarded implementation and six-frame reproduction, not formal large
training.

Gate 19 update: the guarded `le0-state-b` implementation smoke is complete.
Linux six-frame numeric reproduction is still required before larger training.

Gate 20 update: the first Linux reproduction failed. The cause was localized to
an under-expressive state B factorization, then corrected to `point_q_rank`.

Gate 21 update: the corrected `point_q_rank` Linux rerun also failed. Teacher
force still closes, but model force remains about `0.59` to `0.62`. Larger
training is still not released.

Gate 22 update: the equivalence diagnosis passed as diagnosis and failed as a
training release. Main `le0-state-b` is not equivalent to Gate 17 because the
main Query path lacks fixed `static_b[128,6,48]` and residual AD terms mix into
B. Larger training is still not released.

Gate 31 update: corrected 16-case qdef force residual training failed
selected-frame force closure. Wind shell cases `70` to `73` are a real
generalization failure, and `case060` is a near-zero-force special sample.
Do not expand training before Gate 32 diagnosis.

Gate 33 update: Gate 32/33 localize the wind shell failure to branch
normalization, mainly `X16_hat` geometry columns with near-zero train std. Do
not train again before a normalization replay.

Gate 34 update: normalization replay passed with `X16_hat` std floor `0.05`.
The next allowed training is only a small Linux test with an explicit guarded
normalization option, followed by selected-frame force audit.

Gate 35 update: guarded `--branch-x16-std-floor 0.05` implementation passed
tests, but the Linux small training failed force audit. Do not continue the
same OOD split training; next step is wind-shell in-split diagnosis.

Gate 36 update: wind-shell in-split training also failed force audit. Cases
`70,71,72` were included in training and case `73` was held out, but model
force closure still failed while teacher force remained good. Do not expand
training before a Gate 37 geometry and force-error diagnosis.

Gate 37 update: network IO contract diagnosis found no q/LE/B loader mismatch
and no 128-point ordering mismatch, but found severe trunk `point_features`
normalization explosion on `case073`. Do not train again before Gate 38
point-feature normalization replay.

This checklist defines what must be true before network training is allowed.
It does not start training, change the model, or modify the data contract.

## Gate Requirements Before Training

Training is now released only for the limited LE/B learning task recorded in:

```text
docs/TRAINING_RELEASE_DECISION.md
```

This is not a full solver-readiness release.

Required before training:

1. Gate 01 data contract audit passes.
2. Gate 02 TRUE176/CSS8 teacher LE/B finite-difference and selected-frame force
   evidence is sufficient for limited LE/B training. Full teacher tangent
   remains pending and is not a blocker for this limited release.
3. Gate 03 Macro16 source128 selected-frame force closure passes. Full tangent
   remains pending and is not a blocker for this limited release.
4. Gate 04 mainline geometry generality audit passes.
5. Gate 05 integration-point reduction audit records that 96/64/32 candidates
   fail and the project keeps the current 128-point rule.
6. Rigid preprocessing remains passed.

Current status:

- Gate 01: passed.
- Gate 02: cleared only for LE/B training; full teacher stiffness closure is
  still unresolved.
- Gate 03: selected-frame force closure passes; full tangent closure remains
  incomplete but does not block this limited training release.
- Gate 04: mainline geometry generality passes; repaired strong non-inverted
  robustness data also passes.
- Gate 05: reduced candidates fail; keep 128 points active.
- Gate 06: LIMITED RELEASE for LE/B training only.
- Gate 16: q dependent B diagnosis passed, but large training release failed.
  `B_macro_qdef` changes with `q48_def_hat`, while the current point-only B
  prior does not directly express that state dependence.
- Gate 17: state-dependent B baseline prototype passed its small force target,
  but the main Macro16 model has not yet been migrated and force is still above
  the final `0.02` target.
- Gate 18: migration plan passed; it permits only a guarded state B model
  implementation and small reproduction in Gate 19.
- Gate 19: implementation smoke passed; numeric reproduction on Linux remains
  pending and formal large training is still not released.
- Gate 20: initial Linux reproduction failed because the first state B
  factorization was too weak.
- Gate 21: corrected `point_q_rank` rerun failed; any larger training remains
  blocked.
- Gate 22: equivalence diagnosis completed; next step is a fixed-128 direct-B
  state baseline variant, not larger training.
- Gate 31: corrected 16-case force residual audit failed; next step is wind
  shell scale and distribution diagnosis, not larger training.
- Gate 33: branch outlier diagnosis passed; next step is normalization replay,
  not larger training.
- Gate 34: normalization replay passed; next step is guarded implementation and
  small Linux training, not larger training.
- Gate 35: guarded std-floor training failed force audit; next step is
  wind-shell in-split small diagnosis, not larger training.
- Gate 36: wind-shell in-split training failed selected-frame force audit; next
  step is case073 geometry and error diagnosis, not larger training.
- Gate 37: q/LE/B/128-point loading checks passed, but point-feature
  normalization fails; next step is point-feature std-floor replay, not
  training.

Do not treat this as a full tangent or solver-readiness release.

## Pending Issues That Are Not Direct Data Errors

The following items are pending mechanics or route decisions. They should not
be treated as proof that the data is corrupt:

- Gate 02 teacher `LE128_base` and `B_LE128_forward` are credible; B closes
  against finite-difference LE perturbations.
- Teacher and Macro16 force closure use selected-frame physical volume for
  Abaqus RF comparison.
- `integration_weight_phys` remains the reference/standard Macro16 physical
  volume; selected-frame physical volume is a separate audit convention.
- Material-only stiffness `B^T D B dV` is a diagnostic and is not the full
  Abaqus RF finite-difference tangent.
- The `case031` stiffness residual is a tangent-definition mismatch involving
  missing full-tangent terms such as `dV/dq`, `dB/dq`, and geometric/stress
  stiffness.
- Repaired strong non-inverted distortion no longer duplicates the medium
  geometry and passes as a non-blocking robustness boundary check.
- `B_macro_qdef` uses the current small-rotation linear projection
  approximation; strict nonlinear Kabsch Jacobian remains a future audit topic.
- Gate 16 shows the current `B_base(point)` prior has residual q dependent B
  error. The next safe step is a small state-dependent B baseline prototype, not
  formal large training.
- Gate 17 shows the state-dependent B baseline direction is useful. The next
  step is a migration plan and guarded implementation, not formal large
  training.
- Gate 18 defines that guarded implementation route, including a separate model
  variant and detach-state default.
- Gate 19 adds the separate `le0-state-b` model variant. This is still a
  limited implementation smoke until Linux force closure reproduces the target.
- Gate 20 confirms the first failed rerun was caused by weaker B factorization,
  not by source data corruption or changed q48 and LE ordering.
- Gate 21 confirms the corrected `point_q_rank` form still does not reproduce
  Gate 17. The next step is an equivalence diagnosis between the main trainer
  and the Gate 17 prototype, not larger training.
- Gate 22 confirms that the main trainer and Gate 17 are not equivalent. The
  mismatch is in B baseline structure and objective coupling, not data,
  standardization, q48 order, LE order, or the 128 point rule.
- Gate 31 confirms that the force residual route cannot be scaled directly from
  two first6 cases to 16 cases. This is not evidence of q48 ordering, LE
  ordering, or 128 point rule corruption.
- Gate 33 confirms the largest wind shell outlier is in `X16_hat` geometry
  standardization, not in `q48_def_hat`, `LE`, `B`, or the 128 point rule.
- Gate 34 confirms `X16_hat` std floor `0.05` fixes the input-scale explosion
  in replay, but it does not prove trained force closure.
- Gate 35 confirms the normalization fix alone is insufficient. The failure is
  now more likely an out-of-distribution geometry-family split or model/loss
  issue.
- Gate 36 confirms that the failure is not only a fully out-of-distribution
  wind-shell split. Even with cases `70,71,72` in training, held-out `case073`
  fails severely, and training cases remain above the force gate.
- Gate 37 confirms that the current immediate problem is network input
  standardization, not label loading. The trunk feature `ip_invJ_hat_10` reaches
  about `134374` normalized magnitude on `case073` because train std is about
  `8.08e-08`.

These issues no longer block the limited LE/B training release, but they remain
unresolved mechanics items and are not the same as q48 ordering corruption, LE
component corruption, or bad source compact copying.

## Training Data Contract

The active model route is Macro16 v4 boundary route.

Training inputs:

```text
q48_def_hat
X16_hat
```

Training output:

```text
LE
```

The model-visible displacement remains 48-dimensional.

The model predicts integration-point strain. The strain-displacement matrix
`B` is obtained by automatic differentiation of `LE` with respect to the model
displacement input:

```text
B = dLE / d(q48_def_hat)
```

The current default model remains:

```text
Macro16BoundaryDeepONetWithLE0
```

The guarded state B candidate is:

```text
Macro16BoundaryDeepONetWithLE0StateB
```

It is selected only by:

```text
--model-style le0-state-b
```

## Prohibited Training Routes

Do not use or introduce:

- active42 as a main route;
- `X_macro` as a network input;
- fine-grid internal nodes as network inputs;
- TRUE176/CSS8 internal nodes as network inputs;
- any model route that changes the model-visible displacement input from 48D
  to 42D;
- any route that changes q48 ordering;
- any route that changes LE component ordering;
- any route that treats training loss alone as validation.

The model input must remain:

```text
48D q48_def_hat + X16_hat
```

## Post-Training Required Audits

Training completion is not the final validation.

After any future training run, the trained model must be audited again for
mechanical closure.

Required post-training checks:

- strain label error;
- B/autograd consistency;
- selected-frame recovery-force closure;
- material-only K diagnostic, reported separately;
- any full tangent check only if a real full-tangent candidate is implemented
  or explicitly required by the active gate decision.

At minimum, training after completion must rerun the force closure audit using
the canonical selected-frame physical volume convention.

Passing training loss is not sufficient to pass the route.

## Release Boundary

Allowed:

- Train the current Macro16 model for `LE`.
- Supervise `B` through autograd with respect to `q48_def_hat`.
- Use 128 integration points.

Not allowed by this checklist:

- Claim full tangent closure.
- Claim solver-ready macro element status before selected-frame force closure
  passes after training.
- Treat material-only K as the full tangent gate.
- Change model architecture, q48 order, LE order, or integration rule.
- Start formal large training before the state-dependent B baseline is migrated,
  tested, and force-audited below the active gate target.
- Treat the Gate 18 migration plan as proof that the main model already works.
- Treat the Gate 19 smoke as proof that case031 force closure has passed.
- Treat the Gate 21 rerun as permission for larger training.
- Treat Gate 30 two-case success as permission for larger 16-case or full-data
  training.
- Ignore Gate 31 failure and continue training before Gate 32 diagnosis.
- Continue training before Gate 34 resolves or tests branch normalization.
- Treat Gate 34 replay as a trained model pass.
- Repeat the same Gate 35 OOD split training and hope more epochs fix it.
- Expand training after Gate 36 without first diagnosing `case073` geometry,
  force magnitude, and feature distribution.
- Train after Gate 37 without first replaying a guarded point-feature
  normalization floor.
