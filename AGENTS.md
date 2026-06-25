# Agent Instructions

Read `docs/PROJECT_BRIEF.md` before doing project work. It is the current route
baseline for this repository.

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

Before training a network, complete these gates:

1. Confirm source128 compact fields and dimensions.
2. Run rigid preprocessing audit.
3. Run recovery-force audit.
4. Run stiffness audit.
5. Repeat the above on regular, light-distortion, medium-distortion, and strong
   non-inverted geometries.

## Reporting

Every agent task should write a concise report under `reports/` or update an
existing report. Reports must state:

- data path and compact count used,
- exact command(s) run,
- pass/fail status,
- force and stiffness errors when applicable,
- unresolved risks,
- recommended next action.
