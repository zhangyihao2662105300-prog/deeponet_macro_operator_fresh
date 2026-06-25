# IS-FEM Project Brief

This file is the current project baseline for agent work. When old scripts,
old reports, or historical folders conflict with this file, treat this file as
the current route unless a newer dated project brief says otherwise.

## 1. Final Goal

Status: CONFIRMED

Build a standard macro element that can be used inside a solver.

The goal is not merely to train a network. The macro element must output
integration-point strain, provide a strain-displacement matrix, and support
assembly of recovery force and stiffness.

## 2. Current Main Route

Status: CONFIRMED

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

- active42 is OLD.
- The 18-point Macro16 integration route has temporarily failed.
- Routes that only train a network without recovery-force and stiffness audits
  are abandoned.

## 3. Current Trusted Dataset

Status: CONFIRMED

The currently trusted data is regenerated Macro16 source128 compact data.

Data type:

- `npz` compact files.

Current verified range:

- 10 compacts.
- 100 frames.
- Covers regular geometry and lightly distorted geometry.

Status: UNCERTAIN

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
  matrix, and physical integration weights. CONFIRMED.
- `K`: stiffness assembled from strain-displacement matrix, material matrix, and
  physical integration weights. CONFIRMED.

## 6. Current Model Setting

Status: CONFIRMED

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

Status: CONFIRMED

The 18-point Macro16 route failed.

18-point result:

- Mean recovery-force error: 0.355.
- Mean stiffness error: 0.326.

The standard 128-point Macro16 route basically closes.

Standard 128-point result:

- Mean recovery-force error: 0.00973.
- Max recovery-force error: 0.03578.
- Mean stiffness error: 0.01013.
- Max stiffness error: 0.04257.

Rigid preprocessing audit passed.

Audit result:

- 73 tests passed.
- Loader defaults to `q48_def_hat`.
- Loader defaults to `B_macro_qdef`.
- Input remains 48-dimensional.
- After pure translation and pure rotation, the rigid-removed displacement is
  close to zero.

Status: UNCERTAIN

- Maximum error has not been fully reduced below 0.02.
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

## 10. Current Work Priority

Status: CONFIRMED

The current priority is data audit and mechanical assembly, not network
training.

Next steps:

1. Continue standard 128-point Macro16 generality validation.
2. Generate medium-distortion geometries.
3. Generate strong but non-inverted distortion geometries.
4. Rebuild source128 compacts.
5. Run rigid preprocessing audit.
6. Run recovery-force and stiffness audit.
7. If 128 points remain stably closed, then consider 64-point and 32-point
   reduction.
8. Train the network only after the above gates pass.

## Three Core Sentences

1. The current main route is Macro16 v4; TRUE176 / CSS8 128-IP is teacher data
   and audit baseline.
2. The current trusted data is regenerated Macro16 source128 compact data, not
   18-point data.
3. The current priority is generality audit and mechanical assembly, not network
   training.
