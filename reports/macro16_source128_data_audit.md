# Macro16 Source128 Data Audit

Role: Dataset Agent

## Scope

This audit checks regenerated Macro16 source128 compact data only. It does not
train a network, change network structure, change q48/LE ordering, or change
the standard 128 integration-point rule.

## Data

Compact list:

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\macro16_source128_teacher_compact_list.txt`

Compacts checked:

1. `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\0000_case019_macro16_source128_teacher.npz`
2. `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\0001_case025_macro16_source128_teacher.npz`
3. `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\0002_case031_macro16_source128_teacher.npz`
4. `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\0003_case041_macro16_source128_teacher.npz`
5. `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\0004_case043_macro16_source128_teacher.npz`
6. `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\0005_case044_macro16_source128_teacher.npz`
7. `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\0006_case045_macro16_source128_teacher.npz`
8. `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\0007_case046_macro16_source128_teacher.npz`
9. `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\0008_case049_macro16_source128_teacher.npz`
10. `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\0009_case050_macro16_source128_teacher.npz`

Compact count: 10

Frame count: 100

## Commands

```powershell
$env:PYTHONPATH='src;scripts'; py scripts\audit_macro16_rigid_preprocessing.py --compact-list runs\macro16_source128_rigid_preprocess_audit_real_10case\macro16_source128_teacher_compact_list.txt --out runs\macro16_source128_rigid_preprocess_audit_real_10case\source128_data_contract_audit.json --strict
```

```powershell
$env:PYTHONPATH='src;scripts'; py scripts\audit_macro16_force_stiffness.py --compact-list runs\macro16_source128_rigid_preprocess_audit_real_10case\macro16_source128_teacher_compact_list.txt --out runs\macro16_source128_rigid_preprocess_audit_real_10case\force_stiffness_audit.json --volume-mode reference --tangent-mode material-only
```

## Result

Status: PASS

Data-contract audit JSON:

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\source128_data_contract_audit.json`

Summary:

- `strict_pass`: true
- `compact_count`: 10
- `passed_count`: 10
- `missing_required_field_count`: 0
- Failed fields: none
- Synthetic rigid cases: passed

## Contract Checks

Required fields are present:

- `X16_raw`, `X_center`, `L_ref`, `X16_hat`
- `q48_raw`, `q48_hat`, `q48_rigid_raw`, `q48_def_raw`, `q48_def_hat`
- `LE_macro`
- `B_macro_qraw`, `B_macro_qhat`, `B_macro_qdef_raw`, `B_macro_qdef`
- `integration_weight_hat`, `integration_weight_phys`
- `rigid_rotation_R`, `rigid_translation_t`, `rigid_projection_P`
- `macro16_point_xi`

Representative shapes:

- `X16_raw`: `[10, 16, 3]`
- `X_center`: `[10, 3]`
- `L_ref`: `[10, 1]`
- `X16_hat`: `[10, 16, 3]`
- `q48_raw`: `[10, 48]`
- `q48_def_hat`: `[10, 48]`
- `LE_macro`: `[10, 128, 6]`
- `B_macro_qraw`: `[10, 128, 6, 48]`
- `B_macro_qdef`: `[10, 128, 6, 48]`

The model-visible loader defaults are correct:

- `model_visible_q`: `q48_def_hat`
- `model_visible_B`: `B_macro_qdef`
- q input remains 48D
- no 42D input is produced

The source128 row contract is correct:

- `point_count`: 128
- `macro16_point_xi` matches `macro16_source128_point_table`
- max parent-coordinate difference: `2.7211718323094658e-08`

Scale and rigid checks:

- `q48_raw = q48_rigid_raw + q48_def_raw` max relative error: `2.7477666005717422e-08`
- `q48_def_hat * L_ref = q48_def_raw` max relative error: `0.0`
- `B_macro_qdef = B_macro_qdef_raw * L_ref` max relative error: `0.0`
- `integration_weight_phys = integration_weight_hat * L_ref^3` max relative error: `0.0`
- `X_center` max absolute difference from recomputed value: `8.940696716308594e-08`
- `L_ref` max absolute difference from recomputed value: `0.0`
- `X16_hat` max absolute difference from recomputed value: `0.0`
- `rigid_rotation_R` orthogonality max absolute error: `5.921472190362209e-08`
- `q48_rigid_raw` reconstructed from `R,t` max absolute error: `1.12718169020809e-07`

## Force/Stiffness Audit Suitability

These compacts are suitable for force/stiffness audit. The required physical
assembly fields are present:

- `LE_macro`
- `B_macro_qraw`
- `integration_weight_phys`

The force/stiffness audit script ran on the same compact list with
`--volume-mode reference --tangent-mode material-only` and completed
successfully. This is the explicit spelling of the previous reference-volume
material-only audit.

Force/stiffness context from the existing audit output:

- mean recovery-force relative error: `0.00973330739371948`
- max recovery-force relative error: `0.03578126940126121`
- mean stiffness relative error: `0.010125161902503614`
- max stiffness relative error: `0.04256563396475579`

## Unresolved Risks

- The checked compacts cover the current regenerated 10-case / 100-frame set;
  medium-distortion and strong non-inverted distortion compacts are not covered
  by this report.
- `B_macro_qdef` still uses the current small-rotation linear projection
  approximation. This report does not claim a strict nonlinear Kabsch Jacobian.
- Force/stiffness max errors remain above `0.02` on this set; that belongs to
  the mechanical audit/generality gate, not this data-contract gate.

## Recommended Next Action

Proceed to the force/stiffness audit gate on the same regenerated source128
compact list, then repeat this data-contract audit for medium-distortion and
strong non-inverted geometry compacts once they are regenerated.
