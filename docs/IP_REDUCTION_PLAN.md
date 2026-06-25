# Gate 05 IP Reduction Plan

Role: Gate 05 IP Reduction Planning Agent

Status: planning only

This document is a pre-audit plan for future integration-point reduction. It
does not start Gate 05, does not run any audit, does not train a network, does
not change the model, and does not change the current standard 128-point
integration rule.

## Entry Condition

Gate 05 may start only after the Gate 04 mainline passes.

The Gate 04 mainline means the fixed Macro16 parent element has passed the
required real-geometry generality route, including:

- regular geometry;
- light distortion;
- medium distortion;
- cylindrical shell geometry;
- conical shell geometry;
- thickness-varying shell geometry;
- mild double-curvature shell geometry.

Strong non-inverted distortion remains a useful robustness boundary test, but
it is not the hard Gate 04 mainline entry condition for Gate 05.

Until Gate 04 mainline passes, integration-point reduction is blocked.

## Comparison Order

The reduction comparison order is fixed:

```text
128 -> 96 -> 64 -> 32 -> 18
```

Interpretation:

- `128` is the current baseline and must be evaluated first.
- `96`, `64`, and `32` are candidate reduced rules.
- `18` is a failure control, not the expected production route.

The 128-point rule remains the active current route unless and until a future
Gate 05 decision explicitly adopts a reduced rule.

## Audit Quantity

Gate 05 must evaluate mechanical closure, not only strain labels.

Required primary quantity:

```text
F_int = integral B^T sigma dV
```

The force closure audit must use the selected-frame physical volume convention
for comparison against Abaqus projected reaction force.

Canonical force audit volume:

```text
selected-frame physical volume
```

In source compact terms, this corresponds to:

```text
ip_IVOL_abaqus_selected_frames
```

If carried in Macro16 compacts later, the preferred explicit field name is:

```text
integration_weight_phys_selected
```

The existing `integration_weight_phys` field remains the reference/standard
Macro16 physical volume and must not be silently redefined.

## Stiffness Policy

Material-only stiffness

```text
K_material = integral B^T D B dV
```

is diagnostic only for Gate 05 planning.

It should be reported for every integration-point rule, but it is not the
current pass/fail gate for reduction.

Full consistent tangent closure is not a current Gate 05 reduction threshold.
The full tangent route remains unresolved until the project implements or
explicitly waives the missing terms identified in the Gate 03 tangent audits,
including `dV/dq`, `dB/dq`, and geometric/stress stiffness when applicable.

Gate 05 must therefore not reject or accept an integration-point rule solely
from material-only K against the full Abaqus FD tangent threshold.

## Required Audits Per Rule

Every integration-point rule in the comparison order must rerun the data
contract audit.

The data contract audit must confirm at least:

- point count and parent-coordinate table for that rule;
- `X16_raw` / `X16_hat` contract;
- `q48_raw` / `q48_def_hat` contract;
- `LE_macro` shape and component order;
- `B_macro_qraw` and `B_macro_qdef` shapes and q-coordinate meaning;
- physical and dimensionless weight fields are not mixed;
- no fine-grid internal nodes, `X_macro`, or active42 route are exposed to the
  Macro16 model.

Every integration-point rule must also rerun the selected-frame force closure
audit.

The force closure audit must report:

- compact count and frame count;
- geometry families covered;
- force mean relative error;
- force max relative error;
- worst case and frame if available;
- selected-frame physical-volume source;
- material-only K diagnostic metrics separately from force pass/fail.

## Pass Standard

The planned Gate 05 pass standard is:

```text
selected-frame force max relative error < 0.02
```

This threshold applies to force closure only.

It does not apply to:

- LE strain error alone;
- material-only stiffness as a full tangent proxy;
- full consistent tangent K, which is not the current reduction gate.

A rule cannot pass Gate 05 by showing low LE error alone. Force closure is
required.

## Planned Report

The future Gate 05 audit report target is:

```text
reports/05_ip_reduction_audit.md
```

That future report must include:

- explicit statement that Gate 04 mainline passed before Gate 05 started;
- data paths and compact counts for every point rule;
- commands used in the actual audit;
- per-rule data contract results;
- per-rule selected-frame force closure results;
- per-rule material-only K diagnostics;
- pass/fail decision for each rule;
- final recommendation: keep 128 or adopt a reduced rule.

## Current Decision

No integration-point reduction is started by this plan.

Current route remains:

```text
Macro16 v4 boundary route, standard 128 integration points
```

Gate 05 remains blocked until Gate 04 mainline passes.
