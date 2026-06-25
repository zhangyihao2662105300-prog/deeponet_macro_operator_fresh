# Theory Baseline

This file records the finite-element theory baseline for the current Macro16 v4
route. Every statement has exactly one theory status:

- 已验证
- 待验证
- 待决策
- 近似
- 旧路线

Do not mark an approximation as 已验证 unless a numerical audit report proves
the exact claim.

## 1. Standard Element

| Statement | Status | Evidence |
|---|---|---|
| The active target element is the Macro16 v4 boundary-route standard macro element. | 已验证 | `docs/PROJECT_BRIEF.md` |
| The element has fixed parent-coordinate definitions. | 已验证 | `reports/macro16_source128_data_audit.md` |
| The 16 Macro16 parent nodes are fixed. | 已验证 | `reports/macro16_source128_data_audit.md` |
| The shape-function definition is treated as fixed for the current route. | 待验证 | Needs explicit shape-function audit report. |
| The standard 128 integration-point rule is fixed for the current source128 route. | 已验证 | `reports/macro16_source128_data_audit.md` |
| True geometry is obtained from the 16 physical node coordinates, not from fine-grid geometry. | 已验证 | `docs/PROJECT_BRIEF.md` |
| The 18-point Macro16 integration route is not the active route. | 旧路线 | `docs/PROJECT_BRIEF.md` |

## 2. Geometry Mapping

| Statement | Status | Evidence |
|---|---|---|
| The geometry input starts from `X16_raw`. | 已验证 | `reports/macro16_source128_data_audit.md` |
| `X_center` is computed from the 16 nodes. | 已验证 | `reports/macro16_source128_data_audit.md` |
| `L_ref` is the reference length used for geometry and displacement scaling. | 已验证 | `reports/macro16_source128_data_audit.md` |
| `X16_hat` is produced by centering and scaling `X16_raw`. | 已验证 | `reports/macro16_source128_data_audit.md` |
| Integration-point positions are obtained by isoparametric mapping from `X16_hat`. | 待验证 | Needs explicit map reproduction audit report. |
| Integration weights are obtained from the Jacobian and scale relation. | 已验证 | `reports/macro16_source128_data_audit.md` |
| The Jacobian determinant must remain positive for valid non-inverted geometries. | 已验证 | `reports/04_distortion_generality_audit.md` verifies positive detJ for audited Gate 04 required geometries; true strong non-inverted geometry remains a non-blocking robustness item. |

## 3. Displacement And Rigid Motion

| Statement | Status | Evidence |
|---|---|---|
| `q48_raw` is the total 48-dimensional boundary displacement. | 已验证 | `reports/macro16_source128_data_audit.md` |
| Total displacement contains translation, rotation, and true deformation. | 已验证 | `reports/macro16_source128_data_audit.md` |
| Global translation should not produce strain. | 已验证 | `reports/macro16_source128_data_audit.md` |
| Global rotation should not produce strain. | 已验证 | `reports/macro16_source128_data_audit.md` |
| Preprocessing removes fitted global translation and rotation. | 已验证 | `reports/macro16_source128_data_audit.md` |
| The model-visible displacement is `q48_def_hat`. | 已验证 | `reports/macro16_source128_data_audit.md` |
| The model input remains 48-dimensional. | 已验证 | `reports/macro16_source128_data_audit.md` |
| The active route must not reduce the input to 42 dimensions. | 已验证 | `docs/PROJECT_BRIEF.md` |

## 4. Strain-Displacement Matrix

| Statement | Status | Evidence |
|---|---|---|
| The network is intended to output integration-point strain. | 已验证 | `docs/PROJECT_BRIEF.md` |
| The model-predicted `B` is obtained from the derivative of strain with respect to displacement. | 已验证 | `docs/PROJECT_BRIEF.md` |
| Training uses `B_macro_qdef` as the B supervision target. | 已验证 | `reports/macro16_source128_data_audit.md` |
| Physical recovery-force and stiffness audits use `B_macro_qraw`. | 已验证 | `reports/macro16_force_stiffness_audit.md` |
| The current rigid-chain backpropagation for `B_macro_qdef` uses a small-rotation linear projection. | 近似 | `docs/PROJECT_BRIEF.md` |
| A strict nonlinear Kabsch Jacobian is not yet part of the verified route. | 待验证 | No Kabsch-Jacobian audit report yet. |
| `B` finite-difference consistency for all required coordinates is not yet fully closed in the gate workflow. | 待验证 | Needs teacher and Macro16 B-FD closure reports. |

## 5. Recovery Force And Stiffness

| Statement | Status | Evidence |
|---|---|---|
| TRUE176/CSS8 teacher `LE128_base` is a credible exported strain label on the current 10-case set. | 已验证 | `reports/02_teacher_closure_audit.md` |
| TRUE176/CSS8 teacher `B_LE128_forward` closes against finite-difference `LE128_plus`. | 已验证 | `reports/02_teacher_closure_audit.md` |
| TRUE176/CSS8 teacher recovery force closes against `RF_projected` when selected-frame IVOL is used. | 已验证 | `reports/02_teacher_closure_audit.md` |
| TRUE176/CSS8 teacher material-only stiffness closes against Abaqus perturbed-reaction stiffness on the full current set. | 待验证 | `reports/02_teacher_closure_audit.md` fails this subcheck; selected-frame IVOL max stiffness rel is `0.03196211043109838`, worst case `case031`. |
| Recovery force is assembled by integrating stress and `B`. | 待验证 | Macro16 audit exists but current max-error gate failed; `case031` failure is localized in `reports/macro16_case031_mechanics_diagnosis.md`. |
| The active physical-volume convention for large-response Abaqus RF audit is selected-frame physical volume. | 已验证 | `docs/GATE_DECISIONS.md`, `reports/03a_volume_convention_audit.md`, and `reports/02_teacher_closure_audit.md`; selected-frame volume closes teacher force with max rel `0.00858572515082126` and reduces `case031` Macro16 force rel to about `0.0086`. |
| Material-only stiffness `B^T D B dV` is an audit approximation and must not be described as the complete Abaqus tangent under large deformation. | 近似 | `reports/macro16_case031_mechanics_diagnosis.md` shows stiffness remains about `0.032` even with selected-frame or inferred volume. |
| Complete consistent tangent closure requires nonlinear terms beyond material-only stiffness, including `dV/dq`, `dB/dq`, and geometric/stress stiffness when applicable. | 待验证 | Missing terms are the current Gate 03 blocker identified in `reports/macro16_case031_mechanics_diagnosis.md`. |
| Physical assembly audits must declare the physical-volume mode explicitly; `integration_weight_phys` remains the reference/standard Macro16 volume. | 已验证 | `docs/GATE_DECISIONS.md` |
| Dimensionless `integration_weight_hat` must not be used as physical volume weight. | 已验证 | `reports/macro16_force_stiffness_audit.md` |
| Recovery force must close against teacher/reference reaction force. | 待验证 | `reports/macro16_force_stiffness_audit.md` failed current max-error gate; excluding `case031`, the remaining 9 cases pass with max rel `0.009089404785732353`. |
| Stiffness must close against perturbed-force finite-difference stiffness. | 待验证 | `reports/macro16_force_stiffness_audit.md` failed current max-error gate; excluding `case031`, the remaining 9 cases pass with max rel `0.007424585935534821`. |
| Macro stiffness symmetry is numerically closed on the current 10-case set. | 已验证 | `reports/macro16_force_stiffness_audit.md` |
