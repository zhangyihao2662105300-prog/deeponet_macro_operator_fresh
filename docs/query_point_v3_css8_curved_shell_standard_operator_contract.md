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
