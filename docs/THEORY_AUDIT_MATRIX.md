# Theory Audit Matrix

This matrix maps theory assumptions to numerical audits. The `Theory Status`
column uses only:

- 已验证
- 待验证
- 待决策
- 近似
- 旧路线

## Current Matrix

| ID | Theory Assumption | Audit Method | Pass Standard | Current Evidence | Theory Status |
|---|---|---|---|---|---|
| T01 | Standard element parent coordinates are fixed. | Check 16 parent nodes and 128 integration-point parent coordinates across compacts. | Parent coordinates do not change between cases. | `reports/macro16_source128_data_audit.md` | 已验证 |
| T02 | The standard source128 point table is unchanged. | Compare `macro16_point_xi` against the canonical source128 point table. | Max parent-coordinate difference is near zero. | `reports/macro16_source128_data_audit.md` | 已验证 |
| T03 | True geometry is represented by `X16_raw` and `X16_hat`, not by `X_macro`. | Check model-visible compact fields and loader contract. | No fine-grid geometry is exposed to the Macro16 model. | `docs/PROJECT_BRIEF.md`, `reports/macro16_source128_data_audit.md` | 已验证 |
| T04 | `X16_hat` is consistent with `X16_raw`, `X_center`, and `L_ref`. | Recompute centering/scaling and compare. | Reconstruction error is near zero. | `reports/macro16_source128_data_audit.md` | 已验证 |
| T05 | Physical and dimensionless integration weights are scale-consistent. | Check `integration_weight_phys = integration_weight_hat * L_ref^3`. | Scale relative error is near zero. | `reports/macro16_source128_data_audit.md` | 已验证 |
| T06 | Valid geometries require positive Jacobian determinant. | Check all integration-point Jacobian determinants for each geometry. | No negative or near-zero determinant. | Needed in distortion audit. | 待验证 |
| T07 | Rigid translation does not produce deformation displacement. | Feed pure translation and check `q48_def_raw`. | Rigid-removed displacement is near zero. | `reports/macro16_source128_data_audit.md` | 已验证 |
| T08 | Rigid rotation does not produce deformation displacement. | Feed pure rotation and check `q48_def_raw`. | Rigid-removed displacement is near zero. | `reports/macro16_source128_data_audit.md` | 已验证 |
| T09 | The active model input remains 48D. | Check loader output and compact fields. | No 42D active input path. | `reports/macro16_source128_data_audit.md` | 已验证 |
| T10 | `B_macro_qdef` is scale-consistent with `q48_def_hat`. | Check `B_macro_qdef = B_macro_qdef_raw * L_ref`. | Scale relative error is near zero. | `reports/macro16_source128_data_audit.md` | 已验证 |
| T11 | `B_macro_qraw` is the copied source derivative used by physical assembly, but material-only assembly is not yet a complete tangent closure proof. | Compare compact `B_macro_qraw` against source `B_LE128_forward`, then assemble force/stiffness with physical weights. | Field copy matches source and Gate 03 mechanics closure passes. | `reports/macro16_case031_mechanics_diagnosis.md` shows `B_macro_qraw` equals source exactly, but Gate 03 still fails on tangent closure. | 待验证 |
| T12 | Teacher `B_LE128_forward` labels represent finite-difference derivatives of `LE128_base`. | Compare `(LE128_plus - LE128_base) / delta` with `B_LE128_forward`. | B-FD error below threshold. | `reports/02_teacher_closure_audit.md` max rel `2.5944000095411714e-08`. | 已验证 |
| T13 | The rigid-chain derivative for `B_macro_qdef` is exact. | Compare the current linear projection against a strict nonlinear Kabsch Jacobian. | Difference below threshold across required cases. | Current route uses small-rotation projection. | 近似 |
| T14 | Teacher TRUE176/CSS8 strain labels are self-consistent. | Audit finite values, merge guards, point-coordinate audits, and B finite-difference closure. | LE labels are finite, merge-consistent, and B-FD closes. | `reports/02_teacher_closure_audit.md` verifies `LE128_base` and `B_LE128_forward` label consistency. | 已验证 |
| T15 | Macro16 source128 recovery force closes. | Assemble recovery force from `B_macro_qraw`, stress, and `integration_weight_phys`. | Mean and max force errors below gate thresholds. | `reports/macro16_force_stiffness_audit.md` has max error `0.03578126940126121`, above `0.02`; `reports/macro16_case031_mechanics_diagnosis.md` localizes the failure to large-response `case031`, with all other 9 cases below `0.02`. | 待验证 |
| T16 | Macro16 source128 stiffness closes. | Assemble stiffness and compare against perturbed-force finite-difference stiffness. | Mean and max stiffness errors below gate thresholds. | `reports/macro16_force_stiffness_audit.md` has max error `0.04256563396475579`, above `0.02`; `reports/macro16_case031_mechanics_diagnosis.md` localizes the failure to large-response `case031`, with all other 9 cases below `0.02`. | 待验证 |
| T17 | Macro16 stiffness is symmetric for the current material/assembly path. | Measure macro stiffness symmetry relative error. | Symmetry relative error below threshold. | `reports/macro16_force_stiffness_audit.md` max symmetry rel `1.4366806903130055e-16`. | 已验证 |
| T18 | The active volume convention for large-response mechanics closure is fixed. | Compare reference/standard, selected-frame Abaqus IVOL, and inferred-volume assembly on the Gate 03 blocker. | The project selects one convention and records what physical response it is meant to close against. | `reports/macro16_case031_mechanics_diagnosis.md` shows selected-frame or inferred volume reduces force rel from about `0.036` to about `0.0086`, but the route decision is not made. | 待决策 |
| T19 | Material-only stiffness `B^T D B dV` is an approximate audit stiffness, not the complete Abaqus tangent under large deformation. | Compare material-only stiffness with perturbed-force finite-difference tangent under large-response cases. | Tangent error below Gate 03 thresholds, or the missing terms are explicitly added. | `reports/macro16_case031_mechanics_diagnosis.md` shows material-only stiffness remains above gate even after volume update. | 近似 |
| T20 | Consistent tangent terms required by Gate 03 are included. | Audit or derive contributions from `dV/dq`, `dB/dq`, and geometric/stress stiffness when comparing against Abaqus finite-difference tangent. | Gate 03 stiffness max error below `0.02` on the full case set. | Missing consistent tangent terms are the localized Gate 03 blocker in `reports/macro16_case031_mechanics_diagnosis.md`. | 待验证 |
| T21 | Teacher TRUE176/CSS8 recovery force closes against Abaqus reaction force with selected-frame IVOL. | Assemble `B_LE128_forward^T D LE128_base dV_selected` and compare to `RF_projected`. | Mean and max force errors below gate threshold. | `reports/02_teacher_closure_audit.md` max rel `0.00858572515082126`. | 已验证 |
| T22 | Teacher TRUE176/CSS8 material-only stiffness closes against Abaqus perturbed-reaction stiffness. | Assemble `B_LE128_forward^T D B_LE128_forward dV` and compare to `RF_projected_plus` finite differences. | Mean and max stiffness errors below gate threshold. | `reports/02_teacher_closure_audit.md` max rel `0.03196211043109838` with selected-frame IVOL, worst case `case031`. | 待验证 |
| T23 | The 18-point Macro16 route is adequate. | Compare 18-point force and stiffness closure. | Same closure threshold as active route. | Existing results show high mean errors. | 旧路线 |
| T24 | Generality holds beyond the current 10-case set. | Repeat data and mechanics gates on regular, light, medium, strong non-inverted, cylinder, cone, thickness variation, and mild double-curvature shells. | Each geometry family passes force and stiffness thresholds. | No full generality report yet. | 待验证 |
| T25 | Integration-point reduction can preserve mechanics closure. | Audit 128, 96, 64, 32, and 18-point rules. | Reduced rules pass force and stiffness gates. | Not started; 18-point route is a failure control. | 待验证 |

## Status Update Rule

When a gate report changes an assumption, update this matrix in the same change:

- Change only the affected rows.
- Put the evidence report path in `Current Evidence`.
- Keep approximations marked as `近似` until exact numerical evidence exists.
- Keep failed or incomplete closure claims as `待验证`, even if the mean error is good.
