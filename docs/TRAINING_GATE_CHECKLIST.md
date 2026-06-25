# Training Gate Checklist

Role: Training Gate Checklist Agent

Status: limited training release decision recorded

Gate 16 update: q dependent B diagnosis is complete. The limited training
release remains diagnostic only; formal large training is not released.

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
- Start formal large training before the Gate 17 state-dependent B baseline
  prototype is diagnosed.
