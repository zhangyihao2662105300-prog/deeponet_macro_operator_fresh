# Task 03: Macro16 Distortion Generality

Role: Data + Mechanics Agent.

## Goal

Extend the current Macro16 source128 audit beyond regular and lightly distorted
geometries.

## Scope

Generate or select medium-distortion and typical wind-turbine shell geometries,
rebuild source128 compacts, then rerun data, rigid preprocessing, force, and
stiffness audits.

## Required Checks

- Geometry is distorted or curved but not inverted.
- Jacobian determinant remains valid at all integration points.
- Rigid preprocessing still removes pure translation and pure rotation.
- Force and stiffness closure errors do not amplify beyond acceptable limits.
- Required main-route geometry families include regular, light distortion,
  medium distortion, cylindrical shell, conical shell, thickness-varying shell,
  and mild double-curvature shell.
- Strong non-inverted distortion is a non-blocking robustness boundary test.

## Output

Write `reports/macro16_distortion_generality.md`.

The report must separate regular, light, medium, and typical wind-turbine shell
geometry results. Strong non-inverted distortion should be reported separately
as a robustness boundary test when available.
