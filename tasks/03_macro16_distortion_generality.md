# Task 03: Macro16 Distortion Generality

Role: Data + Mechanics Agent.

## Goal

Extend the current Macro16 source128 audit beyond regular and lightly distorted
geometries.

## Scope

Generate or select medium-distortion and strong non-inverted geometries, rebuild
source128 compacts, then rerun data, rigid preprocessing, force, and stiffness
audits.

## Required Checks

- Geometry is distorted but not inverted.
- Jacobian determinant remains valid at all integration points.
- Rigid preprocessing still removes pure translation and pure rotation.
- Force and stiffness closure errors do not amplify beyond acceptable limits.

## Output

Write `reports/macro16_distortion_generality.md`.

The report must separate regular, light, medium, and strong non-inverted
geometry results.
