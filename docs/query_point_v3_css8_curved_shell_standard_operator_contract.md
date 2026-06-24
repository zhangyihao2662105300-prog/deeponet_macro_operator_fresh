# v3 CSS8 Curved-Shell Standard-Operator Contract

Date: 2026-06-24

## Purpose

This document sharpens the v2 standard-operator route for the actual element
used in this project:

```text
CSS8 / 8-node continuum-shell / solid-shell subelements
4 x 4 subelement macro patch
128 integration-point rows
```

The goal is to remove ambiguity around one central question:

```text
What is the standard domain for a curved CSS8 macro element?
```

The answer used by this route is:

```text
The standard domain is a piecewise 4x4 CSS8 parent domain, not one single
flattened HEX8 block.
```

For this project, the curved macro element is therefore interpreted as:

```text
macro standard patch:
  (xi_macro, eta_macro) in [-1,1]^2, split into a 4x4 grid

each subcell:
  local CSS8 parent coordinate (r,s,t) in [-1,1]^3

through-thickness coordinate:
  t is the CSS8 stack coordinate from bottom face to top face
```

There is no single global shell normal after this mapping.  Each integration
point carries its own local stack frame.  The standard-domain point tells the
network where the integration point lives on the 4x4 patch; the physical
geometry fields tell it how that standard point is embedded in the curved
Abaqus geometry.

Each CSS8 subelement still has the usual local parent coordinates:

```text
(r, s, t) in [-1, 1]^3
```

The macro patch additionally needs a macro-level coordinate:

```text
(xi_macro, eta_macro, zeta_macro)
```

or an equivalent pair:

```text
subcell_id + local_css8_parent_r_s_t
```

External reference anchors:

- Abaqus continuum shell documentation distinguishes continuum shells from
  conventional shells and treats them as 3D-discretized shell elements with
  shell-like thickness behavior:
  <https://docs.software.vt.edu/abaqusv2024/English/?show=SIMACAEELMRefMap%2Fsimaelm-c-shelloverview.htm>
- Abaqus shell element guidance notes that continuum shell elements such as
  SC8R allow thickness change and are suitable for finite membrane strain /
  large-rotation shell applications:
  <https://docs.software.vt.edu/abaqusv2024/English/SIMACAEELMRefMap/simaelm-c-shellelem.htm>
- Dassault/SIMULIA shell-element guidance describes SC8R continuum-shell
  orientation by bottom and top element faces: nodes 1-4 form the bottom face,
  nodes 5-8 form the top face, and the default stack/thickness direction runs
  from bottom to top:
  <https://3dswym.3dexperience.3ds.com/post/3dexperience-edu-students/abaqus-fundamentals-for-3dexperience-simulia-shell-elements_NAjtDlPVQAmPY_3_rLe55g>
- Abaqus material-orientation documentation exposes continuum-shell stacking
  direction choices, including element isoparametric direction 3 and the
  material-orientation normal:
  <https://docs.software.vt.edu/abaqusv2024/English/SIMACAECAERefMap/simacae-t-prpassignmatorient.htm>
- Degenerated / Reissner-Mindlin shell formulations commonly describe shell
  geometry by a midsurface plus a director/thickness direction, which is the
  interpretation used here for the CSS8 standard operator:
  <https://2021.help.altair.com/2021/hwsolvers/rad/topics/solvers/rad/theory_element_general_degenerated_4_node_shell_formulations_r.htm>

## Non-Negotiable Contract Decisions

### 1. Current `ip_xi` Is Local CSS8 Parent Coordinate

Current v2b compacts store:

```text
ip_xi = [r, s, t]
```

where each component only takes the two Gauss values:

```text
-1/sqrt(3), +1/sqrt(3)
```

Therefore current `ip_xi` is not enough to identify the 4x4 macro-patch
position.  It repeats for every CSS8 subelement.

Final CSS8 standard-operator compacts must add:

```text
ip_macro_xi: [128, 3]
```

where the first two coordinates distinguish the 4x4 subelement locations and
the third coordinate is the thickness/stack coordinate.  The existing row map
already gives the required values:

```text
standard_css8_row_map()[:, 9:12]
```

The 128 rows have:

```text
unique xi_macro count  = 8
unique eta_macro count = 8
unique zeta count      = 2
```

The explicit mapping from one local CSS8 point to a macro-patch coordinate is:

```text
cell_width = 2 / 4
xi_macro  = -1 + cell_width * (ex + 0.5 * (r + 1))
eta_macro = -1 + cell_width * (ey + 0.5 * (s + 1))
zeta_macro = t
```

where `(ex, ey)` is the 4x4 subcell index.  This is equivalent to
`standard_css8_row_map()[:, 9:12]`.

### 2. CSS8 Geometry Is Midsurface + Stack Director

For each CSS8 subelement, the isoparametric map can be written exactly as:

```text
x(r,s,t) = x_mid(r,s) + 0.5 * t * director(r,s)
```

with:

```text
x_mid(r,s)  = bilinear interpolation of midsurface nodes
director(r,s) = bilinear interpolation of top - bottom node vectors
```

This is the correct interpretation for the current curved solid-shell route:

```text
(r,s) = local in-plane coordinates
t     = through-thickness / stack coordinate
```

The macro element should therefore not be described as a single physical HEX8
that has been globally flattened.  It is a curved 4x4 CSS8 patch over a standard
piecewise parent domain.

### 3. Final Local Strain Frame Must Be Stack-Director Based

For CSS8 solid-shell interpretation, the final local frame should be:

```text
e3 = normalize(g_t)
```

where:

```text
g_r = dX/dr
g_s = dX/ds
g_t = dX/dt
```

Then construct a right-handed orthonormal frame:

```text
e1 = normalized projection of g_r onto the plane normal to e3
e2 = e3 x e1
Q_stack = [e1 e2 e3]
```

The neural operator output coordinate should be:

```text
LE_local in Q_stack
```

The key point is that `Q_stack` is integration-point dependent.  A curved shell
patch is not made standard by forcing all normals to a common direction.  It is
made standard by using the same `(xi_macro, eta_macro, t)` coordinates and
recording the local physical stack frame at each point.

The v2 local frame was:

```text
surface-normal Gram-Schmidt frame oriented by g_t
```

That frame is close for the current thin curved shapes, but it is not the final
CSS8 contract because local-3 is not exactly the stack direction.

### 4. Network Input Must Distinguish Macro Position

The final model-visible trunk fields should be:

```text
ip_macro_xi: [128,3]
ip_local_rst: [128,3]          # optional, for subelement-local detail
geometry descriptors:
  detJ
  thickness = 2 ||g_t||
  Q_stack or compact orientation features
  curvature/director-gradient descriptors when available
```

Implemented v3 compact model-visible fields:

```text
q_useful_hat: [N,42]
geometry_global_hat: [G]
ip_macro_xi: [128,3]
ip_local_rst: [128,3]
local_geometry_features_hat: [128,F]
trunk_features_hat = [ip_macro_xi, ip_local_rst, local_geometry_features_hat]
LE_local_stack: [N,128,6]
B_local_useful_stack_hat: [N,128,6,42]
```

The branch displacement is dimensionless:

```text
q_useful_hat = (T_q_raw_to_useful @ q48_raw) / L_ref
```

The corresponding B label uses the same scale:

```text
B_local_useful_stack_hat =
  L_ref * T_eps_from_abq_stack @ B_LE128_forward @ T_q_raw_to_useful.T
```

The raw-B backprojection is therefore:

```text
B_raw_hat =
  T_eps_to_abq_stack @ B_local_useful_stack_hat @ (T_q_raw_to_useful / L_ref)
```

Audit/postprocess fields:

```text
q48_raw
LE_abq
B_LE128_forward
T_q_raw_to_useful
T_eps_to_abq_stack
T_eps_from_abq_stack
Q_stack
ip_J
ip_detJ
ip_macro_xi
ip_local_rst
```

### 5. B Chain Rule Remains Explicit

The neural network learns:

```text
LE_local_stack = NN(q_useful, ip_macro_xi, geometry_features)
B_local_useful_stack = dLE_local_stack / dq_useful
```

Evaluation returns to Abaqus raw coordinates by:

```text
LE_abq_hat = T_eps_to_abq_stack @ LE_local_stack_hat
B_raw_hat  = T_eps_to_abq_stack @ B_local_useful_stack_hat @ T_q_raw_to_useful
```

Do not directly compare local-stack AD-B with raw Abaqus `B_LE128_forward`.

This contract assumes the stack frame is built from the reference geometry and
is independent of the boundary displacement variable used by autograd.  If a
future route uses a current-configuration frame `Q_stack(q)`, the B chain rule
must include the additional derivative of the strain-basis transform with
respect to `q`.  That is not part of the current v3 contract and must not be
silently mixed with the audited reference-frame route.

## Critical Validation Gates

Before any CSS8 v3 training, a compact must pass these checks:

```text
1. ip_xi equals local CSS8 Gauss (r,s,t)
2. ip_macro_xi exists and distinguishes all 4x4 macro-patch positions
3. x(r,s,t) = x_mid(r,s) + 0.5 * t * director(r,s)
4. Q_stack is orthonormal and right-handed
5. Q_stack local-3 is parallel to g_t = dX/dt
6. det(ip_J) matches ip_detJ
7. LE_abq <-> LE_local_stack roundtrip is near machine precision
8. B_local_useful_stack -> B_raw roundtrip uses the explicit T_q chain rule
9. no old TRUE176 LE/B labels are used as v3 labels
```

If a compact lacks `ip_macro_xi` or stores only the older surface-normal frame,
it can still be a diagnostic artifact, but it is not a final CSS8 standard
operator compact.

## Audit Script

The contract is audited by:

```powershell
py -3 scripts\audit_css8_curved_shell_standard_operator_contract.py `
  --compact-list D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt `
  --out D:\IS-FEM\outputs\query_point_v2_standard_operator\css8_curved_shell_standard_operator_contract.json `
  --strict
```

The audit checks:

```text
ip_xi matches local CSS8 Gauss r,s,t
ip_macro_xi is required for final trunk coordinate
x(r,s,t) = x_mid(r,s) + 0.5 * t * director(r,s)
stack frame Q_stack is right-handed and orthonormal
Q_stack local-3 is parallel to g_t = dX/dt
det(ip_J) matches stored ip_detJ
T_eps_from_abq_stack / T_eps_to_abq_stack roundtrip
current v2 local frame gap relative to Q_stack
```

## 10-Case Result

Input:

```text
D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt
```

Output:

```text
D:\IS-FEM\outputs\query_point_v2_standard_operator\css8_curved_shell_standard_operator_contract.json
```

Result:

```text
compact_count = 10
strict_css8_geometry_pass_count = 10
strict_css8_geometry_fail_count = 0
all_strict_css8_geometry_pass = true
```

Key values:

```text
ip_xi_current_meaning = local_css8_parent_r_s_t
ip_macro_xi_required_for_trunk = true
ip_macro_xi_currently_stored = false
ip_macro_xi_unique_counts = xi:8, eta:8, zeta:2
point_features_include_macro_xi = false

final_recommended_local_frame = css8_stack_director_frame
existing_v2_local_frame = surface_normal_gram_schmidt_oriented_by_g_t
current_Q_rel_to_stack_Q_min = 0.015430432383667826
current_Q_rel_to_stack_Q_max = 0.024697166064443367

stack_e3_dot_g_t_unit_min >= 0.9999999999999998
surface_e3_dot_g_t_unit_min >= 0.9995424705243146
midsurface_director_identity_max_abs <= 4.440892098500626e-16
T_eps_stack_roundtrip_abq_rel <= 2.0009698173194833e-15
T_eps_stack_roundtrip_local_rel <= 2.0006489588449314e-15
```

Interpretation:

```text
CSS8 geometry interpretation is now clear and audited.
The current 10 v2b cases satisfy the CSS8 midsurface/director geometry checks.
The current v2 local frame is close, but the final contract should rebuild
local strain labels in Q_stack.
Current standard compacts lack ip_macro_xi, so they are not the final CSS8
standard-operator compact.
```

This is still not a model-performance result.

## Implemented v3 Compact Builder and Audit

Implemented lightweight utilities:

```text
scripts/v3_css8_standard_operator_common.py
scripts/build_v3_css8_standard_operator_compacts.py
scripts/audit_v3_css8_standard_operator_compact.py
```

Builder command:

```powershell
py -3 scripts\build_v3_css8_standard_operator_compacts.py `
  --compact-list D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10 `
  --strict
```

Independent audit command:

```powershell
py -3 scripts\audit_v3_css8_standard_operator_compact.py `
  --compact-list D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10_audit `
  --strict
```

Output:

```text
D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_compact_list.txt
D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_summary.json
D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10_audit\v3_css8_standard_operator_audit_summary.json
```

10-case result:

```text
compact_count = 10
strict_pass_count = 10
strict_fail_count = 0
strict_pass = true
```

Metric ranges from the independent audit:

```text
q_useful_hat_transform_rel = 0.0 to 0.0
LE_local_stack_to_abq_roundtrip_rel = 1.4773780403428246e-16 to 1.777626317589404e-16
B_raw_projected_rel = 4.325447894744243e-16 to 4.648154783275927e-16
B_rigid_residual_rel = 0.00010968263851608603 to 0.0012638931647260504
Q_stack_orthonormal_max = 1.1102230246251565e-16 to 2.220446049250313e-16
Q_stack_e3_dot_g_t_min = 0.9999999999999998 to 0.9999999999999999
ip_J_hat_scaling_rel = 0.0 to 0.0
ip_invJ_hat_scaling_rel = 0.0 to 0.0
ip_detJ_hat_scaling_rel = 0.0 to 0.0
ip_detJ_vs_det_ip_J_rel = 3.477257433075717e-08 to 4.098070681275398e-08
```

Interpretation:

```text
The v3 CSS8 data contract is now implemented and audited on 10 fresh cases.
This establishes the preprocessing/postprocessing chain, not model performance.
No training was run.
```

Only after this compact contract passes should v3 training be considered.

## One-Case Loader / AD Smoke

After the 10-case compact audit, a one-case loader/overfit smoke was run on
`case050` to check that the model-visible v3 fields are directly consumable:

```text
q_useful_hat + geometry_global_hat + trunk_features_hat -> LE_local_stack
```

The smoke script is:

```text
scripts/train_v3_css8_one_case_overfit_smoke.py
```

It uses a script-local anchored model:

```text
LE_hat =
  B_prior(point) @ q_useful_hat
  + R(q_useful_hat, geometry_global_hat, trunk_features_hat)
  - R(0, geometry_global_hat, trunk_features_hat)
```

and differentiates the result with respect to `q_useful_hat`:

```text
AD_B_local_hat = dLE_local_stack_hat / dq_useful_hat
B_raw_hat = T_eps_to_abq_stack @ AD_B_local_hat @ T_q_raw_to_useful_hat
```

The case050 smoke reached:

```text
train_LE_local_stack_rel = 0.0062978775
train_AD_B_local_useful_hat_rel = 0.0078215189
train_AD_B_local_useful_hat_cos = 0.9999694824
B_model_raw_rel = 0.0079470677
```

This passes the v3 loader/contract/autograd smoke gate.  It is not formal
training, not a multi-case split result, and not a model-performance claim.

## Multi-Case Loader / AD Smoke

The next gate was a multi-case loader/overfit smoke.  Its first useful result
was a contract issue, not a training result: older compacts stored constant
`shape4` as `[4]`, while newer compacts stored it as `[N,4]`.  The original v3
builder treated `shape4[0]` as the compact-level geometry descriptor, which
accidentally turned old `[4]` values into a scalar.  This made
`geometry_global_hat` cross-case inconsistent:

```text
old cases: geometry_global_hat = [154]
new cases: geometry_global_hat = [157]
```

The builder now normalizes both shape encodings to one fixed 4-vector before
constructing `geometry_global_hat`, and the audit reports/fails cross-case
shape inconsistency for key model-visible fields.

After rebuilding the v3 compacts, strict audit reports:

```text
compact_count = 10
strict_pass_count = 10
strict_fail_count = 0
strict_pass = true

q_useful_hat = [10,42]
geometry_global_hat = [157]
trunk_features_hat = [128,48]
LE_local_stack = [10,128,6]
B_local_useful_stack_hat = [10,128,6,42]
```

The multi-case smoke script is:

```text
scripts/train_v3_css8_multi_case_overfit_smoke.py
```

It uses a script-local case-indexed B-prior anchor:

```text
LE_hat =
  B_prior(case, point) @ q_useful_hat
  + R(q_useful_hat, geometry_global_hat, trunk_features_hat)
  - R(0, geometry_global_hat, trunk_features_hat)
```

The 3-case smoke on cases `[41,45,50]` reached:

```text
train_LE_local_stack_rel = 0.0256929491
train_AD_B_local_useful_hat_rel = 0.0035731499
train_AD_B_local_useful_hat_cos = 0.9999935627
B_model_raw_rel = 0.0036204634
```

The full 10-case smoke runs end-to-end and differentiates the complete pool, but
the LE overfit remains uneven by case:

```text
train_LE_local_stack_rel = 0.2310481668
train_AD_B_local_useful_hat_rel = 0.0303847101
train_AD_B_local_useful_hat_cos = 0.9995383620
B_model_raw_rel = 0.0308490749
```

This passes the multi-case loader/shape/autograd gate.  It does not prove v3
model performance or held-out generalization.

## Overfit Diagnosis

The next diagnostic asked whether the 10-case training-set issue means that more
data is needed immediately.  The answer is no: the first issue is loss scaling
and value anchoring inside the existing 10-case pool.

The diagnostic script is:

```text
scripts/diagnose_v3_css8_overfit_cases.py
```

It computes per-case scale metrics and closed-form value anchors, including:

```text
B_mean(case, point) @ q_useful_hat -> LE_local_stack
```

On the current 10-case pool:

```text
Bmean_at_q_LE_rel_min = 0.0230911508
Bmean_at_q_LE_rel_median = 0.0365142219
Bmean_at_q_LE_rel_max = 0.2938258832
Bmean_anchor_bad_case_ids = [31]
```

Thus, for 9/10 cases, the simple case-wise Bmean value anchor already explains
`LE_local_stack` to about 2--5 percent relative error.  The main outlier is
`case031`.

The original 10-case global-scale residual smoke degraded many low-amplitude
cases away from this good anchor.  A diagnostic rerun with per-case loss scaling
kept all non-case031 cases near the anchor:

```text
smoke_final_LE_rel_min = 0.0208749175
smoke_final_LE_rel_median = 0.0290177781
smoke_final_LE_rel_max = 0.2798568606
smoke_degraded_from_Bmean_anchor_case_ids = []
```

Latest per-case values under per-case scaling:

```text
case019: LE_rel=0.0308, AD_B_rel=0.0134
case025: LE_rel=0.0209, AD_B_rel=0.0072
case031: LE_rel=0.2799, AD_B_rel=0.1122
case041: LE_rel=0.0339, AD_B_rel=0.0056
case043: LE_rel=0.0300, AD_B_rel=0.0066
case044: LE_rel=0.0336, AD_B_rel=0.0056
case045: LE_rel=0.0275, AD_B_rel=0.0059
case046: LE_rel=0.0226, AD_B_rel=0.0056
case049: LE_rel=0.0242, AD_B_rel=0.0062
case050: LE_rel=0.0280, AD_B_rel=0.0070
```

Interpretation:

```text
The v3 compact contract is not the blocker.
The next blocker is formal training objective design:
  per-case / relative LE scaling is needed;
  case031 needs a stronger value anchor than B_mean(case,point) @ q.
```

Therefore the next gate should refine the formal v3 objective/value anchor on
the existing 10-case training pool before adding more data or judging held-out
generalization.

## Formal Objective Prototype

The refined objective prototype uses an affine case-wise value anchor:

```text
LE_hat =
  LE_mean(case,point)
  + B_mean(case,point) @ (q_useful_hat - q_mean(case))
  + R(q_useful_hat, geometry_global_hat, trunk_features_hat)
  - R(0, geometry_global_hat, trunk_features_hat)
```

and per-case relative scaling:

```text
LE loss scale = rms(LE_local_stack for that case)
B  loss scale = rms(B_local_useful_stack_hat for that case)
```

The prototype script is:

```text
scripts/train_v3_css8_formal_objective_prototype.py
```

With frozen B anchor on the current 10-case training pool:

```text
train_LE_local_stack_rel = 0.0958199650
train_AD_B_local_useful_hat_rel = 0.0237608757
train_AD_B_local_useful_hat_cos = 0.9997175932
B_model_raw_rel = 0.0240830462

case031 LE_rel = 0.0960717276
case031 AD_B_rel = 0.1120867655
```

With trainable B anchor:

```text
train_LE_local_stack_rel = 0.0934077054
train_AD_B_local_useful_hat_rel = 0.0237150602
train_AD_B_local_useful_hat_cos = 0.9997186661
B_model_raw_rel = 0.0240362827

case031 LE_rel = 0.0936552733
case031 AD_B_rel = 0.1120703891
```

The important interpretation is case-wise:

```text
9/10 non-case031 cases:
  LE_rel roughly 0.0019--0.0091 under trainable B anchor
  AD_B_rel remains low

case031:
  B_mean @ q gave LE_rel about 0.294
  affine anchor improves this to about 0.096
  trainable B anchor only improves it to about 0.094
```

This means the v3 formal objective should keep:

```text
per-case / relative scaling
affine value anchor
explicit AD-B supervision
```

but `case031` still needs a stronger q-dependent/nonlinear value anchor or a
focused amplitude-path audit before held-out split results should be trusted.

## Case031 Amplitude / Nonlinearity Audit

The focused read-only audit script is:

```text
scripts/audit_v3_case031_nonlinearity.py
```

Command:

```powershell
py -3 scripts\audit_v3_case031_nonlinearity.py `
  --compact-list D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\multi_case_min10\v3_css8_standard_operator_compact_list.txt `
  --focus-case 31 `
  --out-root D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\case031_nonlinearity_audit
```

Outputs:

```text
D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\case031_nonlinearity_audit\v3_case031_nonlinearity_summary.json
D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\case031_nonlinearity_audit\v3_case031_nonlinearity_case_summary.csv
D:\IS-FEM\outputs\query_point_v3_css8_standard_operator\case031_nonlinearity_audit\v3_case031_nonlinearity_frame_manifest.csv
```

Result:

```text
case_count = 10
focus_case = 31

case031 q_norm_mean = 0.2080905983
pool q_norm_mean_median = 0.0025180054
case031 q_norm_mean / pool median = 82.6410454331

case031 LE_rms_mean = 0.0128052995
pool LE_rms_mean_median = 0.0001958019
case031 LE_rms_mean / pool median = 65.3992597910

case_internal_q_direction_cos_min = 0.9999999999999998
case_internal_q_direction_cos_mean = 1.0

B_mean(case,point) @ q LE_rel = 0.2938258832
affine B_mean anchor LE_rel = 0.0976020099
quadratic correction over affine LE_rel = 0.0099147393

secant_Bmean_rel_mean = 0.3333519672
secant_Bmean_rel_max = 0.7711158134
```

Interpretation:

```text
case031 is a high-amplitude scalar path, not a q-direction-inconsistent case.
The v3 compact contract is not the blocker.
The simple affine value anchor is still underpowered for case031.
A scalar quadratic amplitude correction almost closes the value reconstruction
gap on this path.
```

This is not a training result.  It is a gate that updates the formal-objective
design requirement:

```text
v3 formal objective should keep:
  per-case / relative LE scaling
  affine value anchor
  explicit AD-B supervision

and should next test:
  amplitude-aware or q-dependent value anchor
  before held-out generalization claims
```

## Affine-Quadratic Value Anchor Prototype

The next gate tested the amplitude-aware anchor implied by the case031 audit.
The read-only oracle is:

```text
scripts/audit_v3_affine_quadratic_anchor_oracle.py
```

The formal objective prototype now supports:

```text
--anchor-mode affine-quadratic
```

and can run anchor-only evaluation with:

```text
--steps 0
```

The prototype anchor is:

```text
s = <q_useful_hat - q_mean(case), q_dir(case)>

LE_hat =
  LE_mean(case,point)
  + B_mean(case,point) @ (q_useful_hat - q_mean(case))
  + C0(case,point)
  + C1(case,point) * s
  + C2(case,point) * s^2
  + R(q, geometry, trunk)
  - R(0, geometry, trunk)
```

The `C0/C1/C2` terms are fitted in closed form per case from the current
10-case training pool.  They are a prototype value anchor, not a held-out
generalization mechanism.

Oracle result:

```text
pooled affine LE_rel = 0.0973463091
pooled affine-quadratic LE_rel = 0.0098883569

pooled affine AD_B_rel = 0.0236806057
pooled affine-quadratic AD_B_rel = 0.0236802581

case031:
  affine LE_rel = 0.0976020099
  affine-quadratic LE_rel = 0.0099147393
  affine AD_B_rel = 0.1120560463
  affine-quadratic AD_B_rel = 0.1120542138
```

Formal prototype result after 1600 steps:

```text
train_LE_local_stack_rel = 0.0098798191
train_AD_B_local_useful_hat_rel = 0.0236802325
train_AD_B_local_useful_hat_cos = 0.9997195005
B_model_raw_rel = 0.0240008291

case031:
  LE_rel = 0.0099061737
  AD_B_rel = 0.1120538861

non-case031 latest LE_rel range:
  about 0.000059 to 0.000453
```

Interpretation:

```text
The v3 LE value-map issue on the 10-case pool is largely closed by an
amplitude-aware scalar anchor.

The remaining case031 problem is no longer LE value reconstruction.
It is the tangent / AD-B side: case031 AD_B_rel remains about 0.112.
```

Therefore the next v3 gate should not be a held-out split yet.  It should first
test tangent-aware value/derivative consistency, for example:

```text
B(s, point) audit on case031
derivative-consistent cubic value anchor
or q-dependent B anchor tied to the scalar amplitude path
```

## Tangent-Amplitude Oracle

The tangent-aware gate adds:

```text
scripts/audit_v3_tangent_amplitude_oracle.py
```

and the formal objective prototype now supports:

```text
--anchor-mode tangent-cubic
--tangent-anchor-degree 3
```

The tangent path model is:

```text
s = <q_useful_hat - q_mean(case), q_dir(case)>
q_perp = q_useful_hat - q_mean(case) - s * q_dir(case)

B_anchor(s) = B0 + B1*s + B2*s^2 + B3*s^3

LE_path(s) =
  C + integral( B_anchor(s) @ q_dir(case), ds )

LE_anchor(q) =
  LE_path(s) + B_anchor(s) @ q_perp
```

For scalar amplitude paths, `q_perp` is nearly zero and autograd returns:

```text
dLE_anchor/dq_useful_hat = B_anchor(s)
```

Oracle result on the current 10-case pool:

```text
case031:
  q_perp_rel_max = 1.6638293738e-07
  Bmean_AD_B_rel = 0.1120560463
  B_poly_deg1_AD_B_rel = 0.0310015948
  B_poly_deg2_AD_B_rel = 0.0173812051
  B_poly_deg3_AD_B_rel = 0.0103404544
  B_poly_deg3_integrated_LE_rel = 0.0178418577
```

The `tangent-cubic` anchor-only prototype gives:

```text
train_LE_local_stack_rel = 0.0178092010
train_AD_B_local_useful_hat_rel = 0.0021637613
B_model_raw_rel = 0.0022217832

case031:
  LE_rel = 0.0178418718
  AD_B_rel = 0.0103404541
```

After 1600 steps:

```text
train_LE_local_stack_rel = 0.0174125843
train_AD_B_local_useful_hat_rel = 0.0022670364
B_model_raw_rel = 0.0023269854

case031:
  LE_rel = 0.0174451172
  AD_B_rel = 0.0103776203
```

Current interpretation:

```text
affine-quadratic anchor:
  better LE value closure
  weaker AD-B closure

tangent-cubic anchor:
  better AD-B closure
  weaker LE value closure
```

This confirms that `case031` needs amplitude-aware tangent modeling.  It also
shows that the next objective should combine the value accuracy of
affine-quadratic with the tangent accuracy of tangent-cubic, instead of treating
either anchor as the final formal objective.
