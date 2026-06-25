# IS-FEM Project Brief

This file is the current project baseline for agent work. When old scripts,
old reports, or historical folders conflict with this file, treat this file as
the current route unless a newer dated project brief says otherwise.

## 1. Final Goal

Theory status: 已验证

Build a standard macro element that can be used inside a solver.

The goal is not merely to train a network. The macro element must output
integration-point strain, provide a strain-displacement matrix, and support
assembly of recovery force and stiffness.

## 2. Current Main Route

Theory status: 已验证

The current main route is the Macro16 v4 boundary route:

- 16 boundary control nodes.
- 48 boundary displacement DOFs.
- Standard 128 integration points.
- Do not input fine-mesh internal nodes.
- Do not input fine-mesh geometry.

TRUE176 / CSS8 128-IP is not the final model route. It is the teacher system and
audit baseline.

Relationship:

- TRUE176 / CSS8 128-IP is the teacher system.
- Macro16 v4 is the target standard-element system.

Historical routes:

- active42 is 旧路线.
- The 18-point Macro16 integration route is 旧路线 for the current workflow.
- Routes that only train a network without recovery-force and stiffness audits
  are abandoned.

## 3. Current Trusted Dataset

Theory status: 已验证

The currently trusted data is regenerated Macro16 source128 compact data.

Data type:

- `npz` compact files.

Current verified range:

- 10 compacts.
- 100 frames.
- Covers regular geometry and lightly distorted geometry.

Theory status: 待验证

- Medium-distortion geometry is not complete.
- Strong-distortion geometry is not complete.
- Generality has not fully passed.
- The E-drive path mentioned in older context is not formally confirmed in this
  project brief. Treat it as a historical path to confirm, not as the unique
  current main data path.

## 4. Main Sample Fields

Geometry fields:

- `X16_raw`: Abaqus original 16-node physical coordinates. CONFIRMED.
- `X_center`: center of the 16 nodes. CONFIRMED.
- `L_ref`: reference length. CONFIRMED.
- `X16_hat`: centered and scaled geometry input. CONFIRMED.

Displacement fields:

- `q48_raw`: original 48-dimensional boundary displacement. CONFIRMED.
- `q48_hat`: original displacement divided by reference length. CONFIRMED.
- `q48_rigid_raw`: fitted rigid displacement. CONFIRMED.
- `q48_def_raw`: deformation displacement after removing global translation and
  rotation. CONFIRMED.
- `q48_def_hat`: rigid-removed deformation displacement divided by reference
  length. CONFIRMED.

Label fields:

- `LE_macro`: standard integration-point strain labels. CONFIRMED.
- `B_macro_qraw`: derivative of strain with respect to original physical
  displacement. CONFIRMED.
- `B_macro_qdef`: derivative of strain with respect to rigid-removed normalized
  displacement. CONFIRMED.

Weight fields:

- `integration_weight_hat`: dimensionless integration weight. CONFIRMED.
- `integration_weight_phys`: physical volume weight. CONFIRMED.

Rigid fields:

- `rigid_rotation_R`: fitted global rotation. CONFIRMED.
- `rigid_translation_t`: fitted global translation. CONFIRMED.
- `rigid_projection_P`: small-rotation linear rigid projection matrix.
  CONFIRMED.

## 5. Core Variable Definitions

- `q48_raw`: 16 boundary nodes times 3 DOFs, for 48 original displacement DOFs.
  CONFIRMED.
- `shape4`: not confirmed as part of the current Macro16 main route in this
  brief. UNCERTAIN.
- `X_keep`: old-route or teacher-data boundary-node coordinate concept. The
  current Macro16 route prefers `X16_raw` and `X16_hat`. OLD.
- `LE128_base`: TRUE176 / CSS8 teacher-system 128-IP strain label. CONFIRMED.
- `B_LE128_forward`: TRUE176 / CSS8 teacher-system derivative of strain with
  respect to original 48-dimensional boundary displacement. CONFIRMED.
- `ip_feature`: standard integration-point features generated from parent
  coordinates and the `X16_hat` isoparametric map. CONFIRMED.
- `F_int`: recovery force assembled from strain, stress, strain-displacement
  matrix, and physical integration weights. CONFIRMED for the assembly path;
  selected-frame physical volume is the canonical audit volume for Abaqus RF
  closure.
- `K_material`: material-only stiffness assembled as `B^T D B dV`.
  CONFIRMED as an audit quantity, but not the complete Abaqus tangent under
  large deformation.
- `K`: complete tangent stiffness. UNCERTAIN until the consistent tangent terms
  required by Gate 03 are included or explicitly waived.

## 6. Current Model Setting

Theory status: 已验证

Inputs:

- `q48_def_hat`.
- `X16_hat`.
- Standard 128 integration-point features.

Output:

- Integration-point strain `LE`.

Source of `B`:

- Automatic differentiation of strain with respect to `q48_def_hat`.

Loss terms:

- Strain error.
- `B` error.
- Rigid constraint loss.

Current default model:

- `Macro16BoundaryDeepONetWithLE0`.

Notes:

- Rigid motion is removed in preprocessing.
- The model input remains 48-dimensional, not 42-dimensional.

## 7. Existing Results

Theory status: 已验证

The 18-point Macro16 route failed.

18-point result:

- Mean recovery-force error: 0.355.
- Mean stiffness error: 0.326.

The standard 128-point Macro16 route now has explicit Gate 03 audit modes. With
selected-frame Abaqus IVOL, recovery force closes on the current 10-case set.
The remaining stiffness issue is not a data-copy or network-training issue; it
is the difference between material-only `B^T D B dV` and the full Abaqus RF
finite-difference tangent for large-response `case031`.

Standard 128-point result:

- Selected-frame recovery-force mean error: 0.0012727832304676163.
- Selected-frame recovery-force max error: 0.00858572515082126.
- Selected-frame material-only stiffness diagnostic mean error:
  0.0036642982829359003.
- Selected-frame material-only stiffness diagnostic max error:
  0.031962110431098374.
- Full-fd-reference tangent reference mean/max error: 0.0 / 0.0 by
  construction; this validates the reference path, not a Macro16 full tangent
  implementation.
- Worst case: `case031`.
- Gate result: split status; force closure PASS, material-only stiffness is a
  diagnostic residual, and full tangent closure remains incomplete.
- Failure status: localized tangent-definition mismatch, not resolved as a full
  consistent tangent.

Rigid preprocessing audit passed.

Audit result:

- 73 tests passed.
- Loader defaults to `q48_def_hat`.
- Loader defaults to `B_macro_qdef`.
- Input remains 48-dimensional.
- After pure translation and pure rotation, the rigid-removed displacement is
  close to zero.

Theory status: 待验证

- Full tangent closure has not been implemented or waived.
- The active large-response force-audit volume convention is selected-frame
  physical volume. The existing `integration_weight_phys` field remains the
  reference/standard Macro16 volume and must not be silently redefined.
- Material-only stiffness `B^T D B dV` is not the complete Abaqus
  finite-difference tangent under large deformation.
- Material-only stiffness remains above the old full-FD `0.02` max-error
  threshold on `case031`; this is now treated as a named diagnostic, not as full
  tangent closure failure by itself.
- Missing consistent tangent terms, including `dV/dq`, `dB/dq`, and
  geometric/stress stiffness when applicable, block Gate 03.
- Medium-distortion and strong-distortion cases have not been verified.
- `B_macro_qdef` is currently based on a small-rotation linear projection
  approximation, not a strict nonlinear Kabsch Jacobian.

## 8. Current Highest-Priority Questions

1. CONFIRMED: Whether standard 128-point closure remains stable across more
   geometries.
2. CONFIRMED: Whether recovery-force and stiffness errors amplify under
   medium-distortion and strong-distortion geometries.
3. UNCERTAIN: Whether the current small-rotation linear projection used for
   `B_macro_qdef` is sufficient for training.
4. UNCERTAIN: Whether a strict Kabsch Jacobian is needed.
5. UNCERTAIN: Whether the rule can later be reduced from 128 points to 64 or 32
   points while preserving force and stiffness closure.

## 9. Do Not Change Without Explicit Approval

Status: CONFIRMED

1. Do not change `q48` ordering.
2. Do not change `LE` component ordering.
3. Do not mix in active42.
4. Do not expose CSS8 internal nodes to the Macro16 model.
5. Do not input `X_macro` to the network.
6. Do not input fine-mesh geometry to the network.
7. Do not change the input from 48 dimensions to 42 dimensions.
8. Do not change the standard 128 integration-point rule.
9. Do not judge progress by training error alone; recovery force and stiffness
   must be audited.
10. Do not mix physical weights with dimensionless weights.

## 10. Gate Status Ledger

Status values in theory documents are limited to `已验证`, `待验证`, `待决策`,
`近似`, and `旧路线`. Gate result may be PASS, FAIL, incomplete, or blocked.

| Gate | Purpose | Gate Result | Evidence | Theory Status |
|---|---|---|---|---|
| 01 | Data contract audit | PASS | `reports/macro16_source128_data_audit.md` | 已验证 |
| 02 | TRUE176/CSS8 teacher closure audit | FAIL on full teacher stiffness closure | `reports/02_teacher_closure_audit.md` | 待验证 |
| 03 | Macro16 source128 force/stiffness audit | split: force closure PASS; material-only stiffness diagnostic residual; full tangent incomplete | `reports/03_macro16_force_stiffness_reaudit.md`, `reports/macro16_case031_mechanics_diagnosis.md` | 待验证 |
| 04 | Distortion/general geometry audit | incomplete | no canonical report yet | 待验证 |
| 05 | Integration-point reduction audit | incomplete | no canonical report yet | 待验证 |
| 06 | Training gate | blocked | blocked by Gate 02/03/04/05 | 待验证 |

Current mechanics gate evidence:

- Gate 02 teacher label evidence:
  `LE128_base` is finite and merge-consistent, `B_LE128_forward` closes against
  `LE128_plus` finite differences with max relative error
  `2.5944000095411714e-08`, and teacher recovery force closes with
  selected-frame IVOL with max relative error `0.00858572515082126`.
- Gate 02 teacher stiffness blocker: material-only `B^T D B dV` does not close
  against Abaqus perturbed-reaction stiffness on the full set; selected-frame
  IVOL max stiffness relative error is `0.03196211043109838`, worst case
  `case031`.
- Macro16 source128 force closure with selected-frame IVOL:
  mean relative error `0.0012727832304676163`, max relative error
  `0.00858572515082126`; force closure PASS on the current 10-case set.
- Material-only stiffness diagnostic with selected-frame IVOL:
  mean relative error `0.0036642982829359003`, max relative error
  `0.031962110431098374`; worst case `case031`.
- Full-fd-reference tangent reference:
  mean/max relative error `0.0` / `0.0` by construction; this verifies the
  Abaqus FD reference branch but is not a Macro16 full consistent tangent.
- Worst case for force and material-only stiffness: `case031`.
- Material-only stiffness symmetry max relative error:
  `1.3549217800954415e-16`.
- Abaqus FD active tangent asymmetry max relative error:
  `0.024048688993506127`.
- Diagnosis: `case031` is a large-response volume/tangent mismatch. `LE_macro`
  and `B_macro_qraw` match source fields exactly. Selected-frame volume closes
  force; missing full consistent tangent terms still block full tangent
  closure.

## 11. Current Work Priority

Theory status: 已验证

The current priority is data audit and mechanical assembly, not network
training.

Next steps:

1. Define or audit the consistent tangent terms needed beyond material-only
   `B^T D B dV`.
2. Resolve or explicitly waive Gate 02 full teacher stiffness closure with
   consistent-tangent evidence.
3. Generate medium-distortion geometries.
4. Generate strong but non-inverted distortion geometries.
5. Rebuild source128 compacts.
6. Rerun rigid preprocessing audit.
7. Rerun recovery-force and stiffness audit.
8. If 128 points stably closes, run the generality audit.
9. Only after 128-point mechanics and generality gates pass, consider 96/64/32
   point reduction.
10. Train the network only after the gate workflow permits it.

## Three Core Sentences

1. The current main route is Macro16 v4; TRUE176 / CSS8 128-IP is teacher data
   and audit baseline.
2. The current trusted data is regenerated Macro16 source128 compact data, not
   18-point data.
3. The current priority is generality audit and mechanical assembly, not network
   training.
