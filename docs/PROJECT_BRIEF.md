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

The standard 128-point Macro16 route has good mean force/stiffness errors but
does not pass the current `0.02` max-error mechanics gate. The `case031`
diagnosis has localized the failure; it is not a compact copy/reorder error and
not a network-training issue.

Standard 128-point result:

- Mean recovery-force error: 0.00973.
- Max recovery-force error: 0.03578.
- Mean stiffness error: 0.01013.
- Max stiffness error: 0.04257.
- Worst case: `case031`.
- Gate result: FAIL current max-error threshold.
- Failure status: localized, not resolved.

Rigid preprocessing audit passed.

Audit result:

- 73 tests passed.
- Loader defaults to `q48_def_hat`.
- Loader defaults to `B_macro_qdef`.
- Input remains 48-dimensional.
- After pure translation and pure rotation, the rigid-removed displacement is
  close to zero.

Theory status: 待验证

- Maximum error has not been fully reduced below 0.02.
- The active large-response force-audit volume convention is selected-frame
  physical volume. The existing `integration_weight_phys` field remains the
  reference/standard Macro16 volume and must not be silently redefined.
- Material-only stiffness `B^T D B dV` is not the complete Abaqus
  finite-difference tangent under large deformation.
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
| 03 | Macro16 source128 force/stiffness audit | FAIL current max-error gate; failure localized | `reports/macro16_force_stiffness_audit.md`, `reports/macro16_case031_mechanics_diagnosis.md` | 待验证 |
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
- Recovery-force mean relative error: `0.00973330739371948`.
- Recovery-force max relative error: `0.03578126940126121`.
- Stiffness mean relative error: `0.010125161902503614`.
- Stiffness max relative error: `0.04256563396475579`.
- Worst case: `case031`.
- Stiffness symmetry max relative error: `1.4366806903130055e-16`.
- Diagnosis: `case031` is a large-response volume/tangent mismatch. `LE_macro`
  and `B_macro_qraw` match source fields exactly. Excluding `case031`, the
  remaining 9 cases pass the `0.02` max-error gate with force max
  `0.009089404785732353` and stiffness max `0.007424585935534821`.

## 11. Current Work Priority

Theory status: 已验证

The current priority is data audit and mechanical assembly, not network
training.

Next steps:

1. Implement explicit Gate 03 physical-volume audit modes using selected-frame
   physical volume as the canonical Abaqus RF closure mode.
2. Define or audit the consistent tangent terms needed beyond material-only
   `B^T D B dV`.
3. Resolve or explicitly waive Gate 02 full teacher stiffness closure with
   consistent-tangent evidence.
4. Generate medium-distortion geometries.
5. Generate strong but non-inverted distortion geometries.
6. Rebuild source128 compacts.
7. Rerun rigid preprocessing audit.
8. Rerun recovery-force and stiffness audit.
9. If 128 points stably closes, run the generality audit.
10. Only after 128-point mechanics and generality gates pass, consider 96/64/32
   point reduction.
11. Train the network only after the gate workflow permits it.

## Three Core Sentences

1. The current main route is Macro16 v4; TRUE176 / CSS8 128-IP is teacher data
   and audit baseline.
2. The current trusted data is regenerated Macro16 source128 compact data, not
   18-point data.
3. The current priority is generality audit and mechanical assembly, not network
   training.
