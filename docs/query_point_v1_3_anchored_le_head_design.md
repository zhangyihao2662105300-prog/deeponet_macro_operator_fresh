# v1.3 Anchored LE Head Design

This document closes the v1.2 diagnostic phase and defines the first v1.3
prototype.  v1.3 is a structural value-anchor change, not another epoch,
learning-rate, or loss-weight tuning pass.

## Why v1.3 Is Needed

The v1.2 data-coverage and failure diagnostics showed that the current problem
is not a data-contract failure:

- Fresh strict-pass compacts preserve the current query-point data contract.
- `B_true @ q48` reconstructs `LE_true` well, so the loaded `q48_raw`, `LE`, and
  `B` labels are mutually consistent.
- The query B prior remains stable under the fixed strategy.
- The learned LE value head produces a large nonzero field at raw `q=0`.

The strongest evidence is the B@q anchor diagnostic:

| case | model_LE_rel | Btrue_q_LE_rel | Bprior_q_LE_rel |
|---:|---:|---:|---:|
| 41 | 9.1422 | 0.0400 | 3.4515 |
| 49 | 4.0235 | 0.0517 | 1.6303 |
| 44 | 9.6526 | 0.0380 | 1.3224 |
| 46 | 7.7823 | 0.0233 | 1.5006 |

The q=0 check is decisive:

```text
split_A q=0 pred RMS = 7.81e-04
split_B q=0 pred RMS = 9.02e-04
```

Those q=0 outputs are already near the observed held-out error scale.  The
model is carrying a ghost LE field even when the boundary displacement input is
zero.

## Design Goal

The v1.3 LE head should use the stable B prior as the value anchor while
forbidding an unconstrained residual branch from creating a nonzero raw-q-zero
LE field.

The target structure is:

```text
LE_hat(q, x) = B_prior(x) @ q + R(q, x)
R(0, x) = 0
LE_hat(0, x) = 0
```

Low-alpha residuals should also be controlled so they cannot dominate low-LE
validation cases with a large ghost field.

## Prototype Formula

The first prototype is a zero-subtracted residual head:

```text
raw_R(q, x)  = residual_net(q, x)
raw_R0(x)   = residual_net(q0, x)
R(q, x)     = gate(||q_raw||) * [raw_R(q, x) - raw_R0(x)]
LE_hat_norm = LE_zero_norm + B_prior_norm(x) @ (q_norm - q0_norm) + R(q, x)
```

Here `q0_norm` is the standardized branch q slice for raw `q=0`, and
`LE_zero_norm` is the standardized raw `LE=0` value.  This keeps all additions
inside normalized LE space.  The model does not mix raw LE and normalized LE.

The optional gate is:

```text
gate = ||q_raw|| / (||q_raw|| + q0)
```

For the first prototype `q0` is provided by `--anchored-residual-gate-q0`.  A
zero value disables gate damping but still keeps the zero-subtracted residual
and raw zero anchor.

## B Prior Training

The first v1.3 prototype keeps the v1.2 query-B training strategy:

- train-only B baseline warm-start remains available;
- `global_b_lr_scale` and `point_b_lr_scale` remain available;
- `B_prior_norm(point)` stays the source of the linear value anchor;
- the old `query-fe-linear-residual` model behavior remains unchanged.

The new behavior is opt-in through:

```text
--model-style query-fe-linear-residual-anchored
```

## Loss Boundary

The first prototype does not add a new training objective.  It keeps:

- LE loss;
- Sobolev/physical J loss;
- train-only B warm-start;
- B-prior parameter groups.

The implementation adds audit metrics for comparison with v1.2:

```text
zero_q_LE_pred_rms
zero_q_LE_pred_max_abs
zero_q_residual_rms
Bprior_q_LE_rel
model_minus_Bprior_offset_rms
```

Complex zero-q consistency losses or residual-amplitude penalties should be
considered only after this structural prototype has a clean smoke run.

## Implementation Summary

Added model class:

```text
QueryFEAnchoredLinearResidualDeepONet
```

Added model style:

```text
query-fe-linear-residual-anchored
```

The legacy model style remains:

```text
query-fe-linear-residual
```

and is not changed.

## Validation Boundary

The v1.3 prototype is only a structural smoke step.  It does not claim improved
formal case-level performance until a later fixed-strategy audit is explicitly
run.

Current smoke requirements:

- anchored model forward shape is correct;
- raw `q=0` gives raw `LE=0` to numerical tolerance;
- zero-subtracted residual is zero at `q=0`;
- AD-B can be computed;
- a 1-epoch tiny training smoke runs;
- old model style still builds and trains through existing tests.
