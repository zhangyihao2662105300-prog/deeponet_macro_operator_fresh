# Agent Instructions

Read these files before doing project work:

1. `docs/PROJECT_BRIEF.md`
2. `docs/THEORY_BASELINE.md`
3. `docs/THEORY_AUDIT_MATRIX.md`
4. `docs/GATE_WORKFLOW.md`

They are the current route, theory, and gate baselines for this repository.

## Current Route

- The active route is Macro16 v4 boundary route.
- TRUE176 / CSS8 128-IP is teacher data and audit baseline, not the final model
  interface.
- The trusted compact type is regenerated Macro16 source128 compact.
- The current priority is generality audit and mechanical assembly, not network
  training.

## Hard Constraints

- Do not change `q48` ordering.
- Do not change `LE` component ordering.
- Do not use active42 as a main route.
- Do not expose CSS8 internal nodes, `X_macro`, or fine-mesh geometry to the
  Macro16 model.
- Do not reduce the input from 48 dimensions to 42 dimensions.
- Do not change the standard 128 integration-point rule unless explicitly
  asked to run a reduction study.
- Do not mix `integration_weight_hat` and `integration_weight_phys`.
- Do not treat training loss as sufficient validation; recovery force and
  stiffness audits are required.

## Default Model/Data Contract

- Model: `Macro16BoundaryDeepONetWithLE0`.
- Geometry input: `X16_hat`.
- Displacement input: `q48_def_hat`.
- Label: `LE_macro`.
- B supervision: `B_macro_qdef`.
- B prediction: automatic differentiation of `LE` with respect to
  `q48_def_hat`.
- Integration rule: standard 128 integration points.

## Current Gates

Use the canonical gate order in `docs/GATE_WORKFLOW.md` and
`tasks/GATE_INDEX.md`:

1. Gate 01: data contract audit.
2. Gate 02: TRUE176/CSS8 teacher closure audit.
3. Gate 03: Macro16 source128 force/stiffness audit.
4. Gate 04: distortion and geometry generality audit.
5. Gate 05: integration-point reduction audit.
6. Gate 06: training gate.

Training is blocked until Gate 01 passes, Gate 02 is passed or explicitly
waived with evidence, Gate 03 passes, Gate 04 passes, and Gate 05 either passes
or the project explicitly decides to keep 128 integration points.

Current gate state:

- Gate 01: PASS, evidence `reports/macro16_source128_data_audit.md`.
- Gate 02: FAIL on full teacher stiffness closure, evidence
  `reports/02_teacher_closure_audit.md`. Teacher `LE128_base` and
  `B_LE128_forward` pass label/B finite-difference checks, and teacher force
  closes with selected-frame IVOL, but material-only stiffness does not close
  against Abaqus perturbed-reaction stiffness on `case031`.
- Gate 03: FAIL current max-error threshold, evidence
  `reports/macro16_force_stiffness_audit.md` and
  `reports/macro16_case031_mechanics_diagnosis.md`. The failure is localized
  to the large-response `case031` volume/tangent mismatch.
- Gate 04: not complete.
- Gate 05: not complete.
- Gate 06: blocked.

Do not say "now train the network" while any required gate is incomplete or
failed.

## Reporting

Every agent task should write a concise report under `reports/` or update an
existing report. Reports must state:

- data path and compact count used,
- exact command(s) run,
- pass/fail status,
- force and stiffness errors when applicable,
- unresolved risks,
- recommended next action.

Every gate that changes theory status must update:

- `docs/PROJECT_BRIEF.md`
- `docs/THEORY_AUDIT_MATRIX.md`

Theory status values are limited to:

- 已验证
- 待验证
- 待决策
- 近似
- 旧路线

Never mark a known approximation as 已验证.
