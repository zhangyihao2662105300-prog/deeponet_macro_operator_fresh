# Task 01: Macro16 Source128 Data Audit

Role: Dataset Agent.

## Goal

Confirm that regenerated Macro16 source128 compacts obey the current data
contract in `docs/PROJECT_BRIEF.md`.

## Scope

Audit compact fields, shapes, coordinate conventions, rigid preprocessing
fields, integration weights, and 128-IP row contract. Do not train a network.

## Required Checks

- `X16_raw`, `X_center`, `L_ref`, `X16_hat`.
- `q48_raw`, `q48_hat`, `q48_rigid_raw`, `q48_def_raw`, `q48_def_hat`.
- `LE_macro`, `B_macro_qraw`, `B_macro_qdef`.
- `integration_weight_hat`, `integration_weight_phys`.
- `rigid_rotation_R`, `rigid_translation_t`, `rigid_projection_P`.
- Input remains 48-dimensional.
- Loader defaults to `q48_def_hat` and `B_macro_qdef`.

## Output

Write `reports/macro16_source128_data_audit.md`.

The report must include compact paths, frame counts, pass/fail status, failed
fields if any, and whether the data is suitable for force/stiffness audit.
