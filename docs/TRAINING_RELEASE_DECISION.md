# Training Release Decision

Decision date: 2026-06-25

Role: Gate 06 Training Release Decision Agent

Status: limited training release

This document is a decision record only. It does not train a network, change the
model, change `q48` ordering, change `LE` component ordering, or change the
128-point integration rule.

## 1. Decision

Training is released for the limited Macro16 v4 LE/B learning task.

Released scope:

1. Train the current Macro16 model to predict integration-point `LE`.
2. Supervise `B` through automatic differentiation:

```text
B = dLE / d(q48_def_hat)
```

3. Use the active Macro16 source128 data contract.
4. Keep the active 128-point integration rule.

Not released:

1. Do not claim full tangent closure from this training.
2. Do not claim solver-ready macro-element status from training loss alone.
3. Do not use material-only K as a full Abaqus tangent gate.
4. Do not change model architecture as part of this decision.

## 2. Basis

The release is based on the current gate evidence:

1. Gate 01 data contract passes.
2. Teacher `LE128_base` and `B_LE128_forward` are credible and finite-difference
   consistent.
3. Macro16 source128 selected-frame recovery-force closure passes.
4. Gate 04 required geometry families pass selected-frame force closure.
5. Gate 05 reduced candidates fail, so the active route keeps 128 points.
6. Rigid preprocessing remains part of the active contract.

## 3. Full Tangent Decision

Full tangent closure remains pending, but it is not a blocker for this limited
LE/B training release.

Reason:

1. The current training target is `LE` and `B`, not full Abaqus tangent `K`.
2. The unresolved full tangent terms are mechanics-assembly terms beyond
   material-only `B^T D B dV`.
3. Missing `dV/dq`, `dB/dq`, geometric stiffness, and stress stiffness should
   not be hidden inside network training.

Therefore:

```text
full tangent pending != LE/B training blocked
```

But:

```text
full tangent pending == no full tangent claim after training
```

## 4. Material-Only K Policy

Material-only stiffness remains diagnostic only:

```text
K_material = integral B^T D B dV
```

It must be reported after training, but it is not the full Abaqus tangent and
must not be used as the full tangent pass/fail gate.

## 5. Required Post-Training Audits

After any released training run, the trained model must be audited before any
macro-element usability claim.

Required:

1. `LE` error.
2. `B` error from autograd.
3. `B` consistency with the active q-coordinate.
4. Selected-frame recovery-force closure.
5. Material-only K diagnostic.
6. Worst-case and per-geometry force errors.

The minimum mechanical post-training gate is:

```text
selected-frame force closure
```

Training loss alone is not enough.

## 6. Active Training Contract

Inputs:

```text
q48_def_hat
X16_hat
standard 128 parent integration-point features
```

Output:

```text
LE
```

B source:

```text
autograd dLE / d(q48_def_hat)
```

Default model:

```text
Macro16BoundaryDeepONetWithLE0
```

Forbidden:

1. No active42 route.
2. No `X_macro` input.
3. No fine-grid internal geometry input.
4. No `q48` reordering.
5. No `LE` component reordering.
6. No 18-point route revival.
7. No reduced 96/64/32 rule adoption from the failed Gate 05 candidates.

## 7. Decision Summary

Gate 06 status:

```text
LIMITED RELEASE
```

Meaning:

1. Training may start for LE/B learning.
2. Full tangent remains pending.
3. Material-only K remains diagnostic.
4. Selected-frame force closure is mandatory after training.
5. The trained network is not accepted until the post-training force audit
   passes.
