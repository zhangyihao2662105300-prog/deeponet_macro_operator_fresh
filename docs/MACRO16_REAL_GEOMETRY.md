# Macro16 Standard Parent Element And Real Geometry

This document explains the relationship between the standard Macro16 parent
element and real wind-turbine shell geometry. It is documentation only. It does
not change the model, training route, audit scripts, or the standard 128
integration-point rule.

## 1. Standard Macro16 Parent Element

The active Macro16 route is a standard boundary-control macro element:

- 16 boundary control nodes;
- 48 boundary displacement DOFs;
- one fixed parent domain `(r, s, t) in [-1, 1]^3`;
- 8 serendipity surface nodes on the lower surface;
- the matching 8 serendipity surface nodes on the upper surface;
- standard 128 integration points in fixed parent coordinates.

The parent element is fixed. Its parent nodes, shape functions, and
integration-point coordinates do not change from one real geometry to another.

The Macro16 shape is:

```text
surface Q8 serendipity interpolation in (r, s)
linear interpolation through thickness in t
```

The lower surface uses `t = -1`. The upper surface uses `t = +1`.

## 2. Sixteen Parent Node Coordinates

The canonical Macro16 parent-node order is:

| Macro16 node | r | s | t | Surface role |
|---:|---:|---:|---:|---|
| 0 | -1 | -1 | -1 | lower corner |
| 1 | 0 | -1 | -1 | lower midside |
| 2 | 1 | -1 | -1 | lower corner |
| 3 | 1 | 0 | -1 | lower midside |
| 4 | 1 | 1 | -1 | lower corner |
| 5 | 0 | 1 | -1 | lower midside |
| 6 | -1 | 1 | -1 | lower corner |
| 7 | -1 | 0 | -1 | lower midside |
| 8 | -1 | -1 | 1 | upper corner |
| 9 | 0 | -1 | 1 | upper midside |
| 10 | 1 | -1 | 1 | upper corner |
| 11 | 1 | 0 | 1 | upper midside |
| 12 | 1 | 1 | 1 | upper corner |
| 13 | 0 | 1 | 1 | upper midside |
| 14 | -1 | 1 | 1 | upper corner |
| 15 | -1 | 0 | 1 | upper midside |

This order is the order used by `X16_raw`, `X16_hat`, `q48_raw`,
`q48_def_hat`, and the Macro16 geometry map.

## 3. Fixed 128 Integration Points

The source128 rule is a fixed parent-domain rule. It is not generated from
fine-grid internal nodes.

The current rule is:

```text
4 x 4 in the (r, s) parent plane
2 x 2 x 2 Gauss points per in-plane subcell and through thickness
total = 4 * 4 * 8 = 128 parent points
```

Every integration point has a fixed parent coordinate:

```text
xi_p = (r_p, s_p, t_p)
```

These 128 parent coordinates are the same for a flat plate, cylindrical shell,
conical shell, thickness-varying shell, double-curvature shell, and distorted
shell. Only the physical coordinates obtained from the geometry map change.

## 4. Real Geometry Comes From X16_raw

Real physical geometry is represented by the 16 physical boundary nodes:

```text
X16_raw = [X_0, X_1, ..., X_15]
X_i in R^3
```

For a parent coordinate `xi = (r, s, t)`, the physical point is obtained by the
Macro16 isoparametric map:

```text
x(r, s, t) = sum_{i=0}^{15} N_i(r, s, t) X_i
```

Here:

- `N_i(r, s, t)` are the fixed Macro16 parent shape functions;
- `X_i` are the real physical coordinates from `X16_raw`;
- `x(r, s, t)` is the real physical position of that parent point.

The through-thickness split is:

```text
N_i(r, s, t)      = 0.5 * (1 - t) * S_i(r, s),  i = 0..7
N_{i+8}(r, s, t) = 0.5 * (1 + t) * S_i(r, s),  i = 0..7
```

where `S_i(r, s)` are the lower/upper surface Q8 serendipity shape functions in
the fixed surface-node order.

The physical Jacobian is:

```text
J(r, s, t) = dx / d(r, s, t)
```

For a valid Macro16 real geometry, `detJ` must be positive at every integration
point. Negative or near-zero `detJ` means the mapped element is inverted or
degenerate and must not be accepted as a valid training or audit geometry.

## 5. Geometry Scaling And Network Input

The compact stores both raw and normalized geometry:

```text
X_center = mean(X16_raw)
L_ref    = characteristic reference length
X16_hat  = (X16_raw - X_center) / L_ref
```

The model-visible geometry input is `X16_hat`, not Abaqus global `X16_raw`.
This keeps geometry scale consistent with the normalized displacement input
`q48_def_hat`.

The physical geometry still comes from the same 16 nodes. In physical form:

```text
x_raw(r, s, t) = sum_i N_i(r, s, t) X16_raw_i
```

In normalized feature form:

```text
x_hat(r, s, t) = sum_i N_i(r, s, t) X16_hat_i
```

The normalized map is used to build model features and dimensionless
integration weights. Physical-volume quantities are obtained by applying the
scale relation with `L_ref`.

## 6. Wind-Turbine Shell Geometries

The same standard Macro16 parent element can represent different wind-turbine
shell element geometries through different `X16_raw` values.

Examples:

- cylindrical shell element: `X16_raw` nodes lie on two nearby cylindrical
  surfaces;
- conical shell element: `X16_raw` nodes lie on two nearby conical surfaces;
- thickness-varying shell element: lower and upper `X16_raw` surfaces have
  spatially varying separation;
- double-curvature shell element: `X16_raw` surfaces curve in both in-plane
  directions;
- distorted shell element: `X16_raw` is perturbed from the regular shell shape,
  while still preserving positive `detJ`.

These are not different parent elements. They are different physical mappings
of the same parent element:

```text
same parent nodes
same shape functions
same 128 parent integration points
different X16_raw
different x(r, s, t), J, detJ, local frame, and physical weights
```

## 7. What The Macro16 Route Does Not Input

The active Macro16 route must not expose teacher or fine-grid geometry to the
model.

Do not input:

- fine-grid internal CSS8 nodes;
- TRUE176 internal nodes;
- `X_macro` from old routes;
- any geometry field that bypasses the 16-node Macro16 standard map.

The model geometry contract is:

```text
model geometry input = X16_hat
real physical geometry source = X16_raw
integration-point geometry = isoparametric map from X16_raw or X16_hat
```

This is why cylindrical, conical, thickness-varying, double-curvature, and
distorted wind-turbine shell elements must all be expressed by their own
`X16_raw`, not by adding internal geometry nodes to the model input.

## 8. Gate 04 Requirement

Gate 04 is the distortion and real-geometry generality audit. It must verify
that the Macro16 source128 route is not only valid on the current regular and
lightly distorted set.

Gate 04 should test at least:

- regular geometry;
- light distortion;
- medium distortion;
- cylindrical shell geometry;
- conical shell geometry;
- thickness-varying shell geometry;
- mild double-curvature shell geometry.

Strong non-inverted distortion is useful as a robustness boundary test, but it
is not the main wind-turbine shell geometry gate. It should not block the main
route if regular, light, medium, and typical wind-turbine shell families pass.

For each geometry family, Gate 04 must check:

- `X16_raw` produces positive `detJ` at all 128 fixed parent integration
  points;
- the 128 parent point rule is unchanged;
- no fine-grid internal nodes or `X_macro` are used as model input;
- force closure is rerun with the canonical selected-frame physical-volume
  convention for Abaqus RF comparison;
- material-only stiffness remains separately reported as a diagnostic;
- full tangent closure is not claimed unless a real full-tangent candidate is
  audited.

The key Gate 04 question is:

```text
Can the same fixed Macro16 parent element, mapped only by X16_raw, preserve
force closure across wind-turbine shell real geometries?
```

Until Gate 04 passes, the route should not be treated as geometry-general, and
training remains blocked by the gate workflow.
