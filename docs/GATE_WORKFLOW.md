# Gate Workflow

The project advances through gates. Agents do not decide theory by intuition;
they check whether each gate has a report, whether the report satisfies the
thresholds, and whether the theory status was updated honestly.

## Four Layers

1. Theory baseline: `docs/THEORY_BASELINE.md`.
2. Numerical audit matrix: `docs/THEORY_AUDIT_MATRIX.md`.
3. Task gates: this file and `tasks/GATE_INDEX.md`.
4. Conclusion archive: reports under `reports/` plus status updates in
   `docs/PROJECT_BRIEF.md` and `docs/THEORY_AUDIT_MATRIX.md`.

## Gate 01: Data Contract Audit

Goal:

- Confirm fields, shapes, scales, rigid preprocessing, and source128 point
  table.

Required checks:

- `X16_raw`, `X16_hat`.
- `q48_raw`, `q48_def_hat`.
- `B_macro_qdef`, `B_macro_qraw`.
- `integration_weight_hat`, `integration_weight_phys`.
- Loader defaults to `q48_def_hat` and `B_macro_qdef`.
- Input remains 48D.

Current result:

- Gate result: PASS.
- Evidence: `reports/macro16_source128_data_audit.md`.

## Gate 02: Teacher Closure Audit

Goal:

- Confirm the TRUE176/CSS8 teacher system closes before using it as a source for
  Macro16.

Required checks:

- Teacher strain label credibility.
- Teacher B finite-difference closure.
- Teacher force closure.
- Teacher stiffness closure.

Current result:

- Gate result: FAIL on full teacher stiffness closure.
- Evidence: `reports/02_teacher_closure_audit.md`.
- `LE128_base` credibility passes.
- `B_LE128_forward` finite-difference LE closure passes with max relative error
  `2.5944000095411714e-08`.
- Teacher recovery force closes with selected-frame IVOL; max relative error is
  `0.00858572515082126`.
- Teacher material-only stiffness does not close against Abaqus
  perturbed-reaction stiffness on the full 10-case set; selected-frame IVOL max
  stiffness relative error is `0.03196211043109838`, worst case `case031`.

## Gate 03: Macro16 Source128 Mechanics Audit

Goal:

- Confirm standard Macro16 source128 recovery force and stiffness closure.

Required checks:

- Recovery-force mean relative error.
- Recovery-force max relative error.
- Stiffness mean relative error.
- Stiffness max relative error.
- Worst case.
- Worst DOF or matrix rows/columns when available.
- Energy consistency when available.
- Stiffness symmetry.

Current result:

- Gate result: FAIL current `0.02` max-error closure threshold.
- Evidence: `reports/macro16_force_stiffness_audit.md` and
  `reports/macro16_case031_mechanics_diagnosis.md`.
- Worst case: `case031`.
- Force max relative error: `0.03578126940126121`.
- Stiffness max relative error: `0.04256563396475579`.
- Stiffness symmetry passes.
- Failure status: localized, not resolved.
- Localized diagnosis: `case031` is a large-response mechanics case; compact
  `LE_macro` and `B_macro_qraw` match the source fields exactly, and excluding
  `case031` the remaining 9 cases pass the `0.02` max-error gate.
- Current blockers: choose the active large-response volume convention, and
  account for missing consistent tangent terms beyond material-only
  `B^T D B dV`, including `dV/dq`, `dB/dq`, and geometric/stress stiffness
  when applicable.

## Gate 04: Generality Audit

Goal:

- Confirm the route is not only valid on the current 10-case set.

Geometry families:

- Regular geometry.
- Light distortion.
- Medium distortion.
- Strong non-inverted distortion.
- Cylindrical shell.
- Conical shell.
- Thickness-varying shell.
- Mild double-curvature shell.

Current result:

- Gate result: not complete.

## Gate 05: Integration-Point Reduction Audit

Goal:

- Determine whether fewer integration points can preserve mechanical closure.

Order:

- 128 points.
- 96 points.
- 64 points.
- 32 points.
- 18 points as a failure control.

Rule:

- Every reduction must rerun force and stiffness audits.
- Do not decide from strain error alone.

Current result:

- Gate result: not complete.
- 18-point route is a known failed historical route.

## Gate 06: Training Gate

Goal:

- Decide whether network training is allowed.

Training is blocked until:

- Gate 01 passes.
- Gate 02 passes or is explicitly waived with evidence.
- Gate 03 passes.
- Gate 04 passes.
- Gate 05 passes or the project explicitly decides to keep 128 points.
- Rigid preprocessing remains passed.

Current result:

- Gate result: blocked.
- Reason: Gate 02 fails full teacher stiffness closure, Gate 03 fails the
  current max-error threshold, and Gate 04/05 are incomplete.

## Required Report Template

Each gate report must include:

1. Task goal.
2. Data used.
3. Scripts and commands used.
4. Key metrics.
5. Pass standard.
6. Actual result.
7. Gate result.
8. Unresolved issues.
9. Recommended next action.

## Canonical Report Targets

- Gate 01: `reports/01_dataset_audit.md` or current evidence
  `reports/macro16_source128_data_audit.md`.
- Gate 02: `reports/02_teacher_closure_audit.md`.
- Gate 03: `reports/03_macro16_force_stiffness_audit.md` or current evidence
  `reports/macro16_force_stiffness_audit.md`.
- Gate 04: `reports/04_distortion_generality_audit.md`.
- Gate 05: `reports/05_ip_reduction_audit.md`.
- Gate 06: `reports/06_training_gate.md`.

## Current Next Action

Decide the Gate 03 mechanics route before training or reduction:

- Record whether Gate 03 should close against reference/standard physical
  volume, selected-frame Abaqus IVOL, or inferred volume.
- Split stiffness closure into material-only stiffness and the full consistent
  tangent expected by the Abaqus finite-difference response.
- Treat missing `dV/dq`, `dB/dq`, and geometric/stress stiffness terms as the
  current Gate 03 blocker.
- Keep the 128-point rule unchanged until this mechanics mismatch is resolved.
