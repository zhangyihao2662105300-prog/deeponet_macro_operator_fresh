# Gate Decisions

Decision date: 2026-06-25

Role: Gate Decision Archivist

## Scope

This document records the current Gate 02 and Gate 03 mechanics decisions from:

- `reports/02_teacher_closure_audit.md`
- `reports/03a_volume_convention_audit.md`
- `reports/03b_consistent_tangent_audit.md`
- `reports/03_macro16_force_stiffness_reaudit.md`

No network training, model change, q48 ordering change, LE ordering change, or
128-point integration-rule change is implied by these decisions.

## Decision Summary

| ID | Decision | Status |
|---|---|---|
| D01 | Use selected-frame physical volume as the canonical audit volume for force closure against Abaqus RF. | adopted |
| D02 | Keep `integration_weight_phys` as the reference/standard Macro16 physical volume. | adopted |
| D03 | Do not apply the full Abaqus FD tangent `0.02` threshold to material-only `B^T D B dV`. | adopted |
| D04 | Treat current Gate 02/03 failures as definition mismatches, not bad data or network failures. | adopted |
| D05 | Audit scripts expose explicit volume and tangent modes. | adopted |

## D01: Canonical Audit Volume

Selected-frame physical volume is adopted as the canonical audit volume when
comparing assembled recovery force against Abaqus projected reaction force.

Reason:

- Gate 02 teacher force closure with selected-frame IVOL passes the current
  `0.02` max-error criterion on the 10-case set; max force rel is
  `0.00858572515082126`.
- Gate 03a shows that, for the blocker `case031`, changing only the volume
  convention reduces force rel from about `0.03578` to about `0.00859`.
- Inferred Abaqus volume follows the same trend and is acceptable as a
  cross-check or fallback when selected-frame IVOL is unavailable.

Canonical audit volume:

```text
ip_IVOL_abaqus_selected_frames
```

Fallback/cross-check volume:

```text
IVOL128_inferred_from_DLE
```

This decision applies to force closure against Abaqus RF. It does not by itself
close the full tangent stiffness gate.

## D02: `integration_weight_phys` Contract

`integration_weight_phys` remains the Macro16 reference/standard physical
volume:

```text
integration_weight_phys = integration_weight_hat * L_ref^3
```

It must not be silently redefined to mean selected-frame Abaqus IVOL.

Required naming for future compact/audit work:

- `integration_weight_phys`: current reference/standard Macro16 volume.
- `integration_weight_phys_selected`: selected-frame physical volume for
  canonical RF audit, when carried in Macro16 compacts.
- `integration_weight_phys_inferred`: inferred Abaqus physical volume, when
  carried as a fallback or diagnostic field.

If compact regeneration or extension is performed later, the new fields should
be added explicitly. The training loader does not need to change unless a future
training loss actually consumes physical volume.

## D03: Material-Only K Threshold

Material-only stiffness

```text
K_material = integral B^T D B dV
```

is retained as an assembly diagnostic, but it is no longer the candidate for
the full Abaqus finite-difference tangent `0.02` max-error gate.

Reason:

- Gate 03b shows the current audit compares symmetric material-only K against
  `K_fd = (RF_projected_plus - RF_projected) / delta`.
- For `case031`, the Abaqus step uses `nlgeom=YES`, so this FD reference is a
  full nonlinear projected reaction-force tangent.
- With selected-frame/inferred volume, material-only K still has about
  `0.032` relative error against Abaqus FD.
- Missing full-tangent terms include `dV/dq`, `dB/dq`, and geometric/stress
  stiffness.

Decision:

```text
The 0.02 Abaqus FD tangent threshold applies only to a full-tangent candidate,
not to material-only B^T D B dV.
```

Material-only K can still have its own diagnostic threshold, but that threshold
must be named as material-only and must not be described as full Abaqus tangent
closure.

## D04: Gate 02/03 Failure Interpretation

The current Gate 02 and Gate 03 failures are definition mismatches, not evidence
of bad data.

Evidence:

- `LE128_base` is credible and merge-consistent in Gate 02.
- `B_LE128_forward` closes against `(LE128_plus - LE128_base) / delta` with max
  relative error `2.5944000095411714e-08`.
- `LE_macro` equals source `LE128_base` for `case031`.
- `B_macro_qraw` equals source `B_LE128_forward` for `case031`.
- Teacher and Macro16 force closure improve to below the force threshold when
  selected-frame volume is used.
- The remaining stiffness failure is localized to using material-only K against
  a full Abaqus RF finite-difference tangent.

Gate interpretation after this decision:

| Gate | Interpretation |
|---|---|
| Gate 02 | Teacher labels and B finite differences are valid; teacher force closes with selected-frame IVOL; full teacher stiffness remains unresolved because material-only K is not full Abaqus FD tangent. |
| Gate 03 | Macro16 source128 data copy/order is valid; selected-frame force closure passes on the current 10-case set; remaining full-tangent closure is a tangent-definition mismatch. |

These failures should not be routed to network training.

## D05: Explicit Audit Modes

The audit script now makes the physical and tangent definitions explicit. No
current Gate 03 audit should infer these silently from field presence.

Implemented force/volume mode:

```text
--volume-mode reference
--volume-mode selected-frame
--volume-mode inferred
```

Implemented stiffness/tangent target mode:

```text
--tangent-mode material-only
--tangent-mode full-fd-reference
```

Future full-tangent work may add an explicitly named
`full-consistent-candidate` mode or equivalent feature flags.

Recommended reference declaration:

```text
--force-reference abaqus-rf-projected
--stiffness-reference abaqus-rf-forward-fd
```

Recommended full-tangent feature flags or equivalent named candidate modes:

```text
--include-dVdq
--include-dBdq
--include-geometric-stiffness
--include-stress-stiffness
```

Required output metadata:

- volume mode used;
- source field used for physical volume;
- stiffness mode used;
- stiffness reference used;
- whether `dV/dq`, `dB/dq`, geometric stiffness, and stress stiffness were
  included;
- whether the reported stiffness threshold is material-only or full tangent.

## Gate Status After Decision

Gate 02 remains FAIL for full teacher stiffness closure until a full-tangent
candidate closes against Abaqus FD or the project explicitly waives that
requirement with evidence.

Gate 03 is split after `reports/03_macro16_force_stiffness_reaudit.md`:

1. force closure with selected-frame physical volume: PASS on the current
   10-case set;
2. material-only stiffness: retained as a diagnostic, not full tangent closure;
3. full tangent closure: incomplete until a full consistent tangent candidate
   closes against Abaqus FD or the project explicitly waives that requirement
   with evidence.

Training remains blocked by the gate workflow.

## Next Actions

1. Extend or regenerate compacts to carry selected-frame and inferred physical
   volume fields with explicit names.
2. Rerun Gate 02 force closure with selected-frame volume when teacher data is
   regenerated or expanded.
3. Decide whether the solver route requires full consistent tangent K or accepts
   material-only K as a named quasi-Newton approximation.
4. Do not use network training to absorb the volume/tangent definition mismatch.
