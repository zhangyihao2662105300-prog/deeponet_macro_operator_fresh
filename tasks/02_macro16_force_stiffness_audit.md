# Task 02: Macro16 Force And Stiffness Audit

Role: Mechanics Agent.

## Goal

Audit whether the standard 128-point Macro16 source128 route closes recovery
force and stiffness.

## Scope

Use trusted Macro16 source128 compacts. Do not train or tune the neural network.

## Required Checks

- Recovery force mean relative error.
- Recovery force max relative error.
- Stiffness mean relative error.
- Stiffness max relative error.
- Stiffness symmetry error when available.
- Physical integration weights are used for physical assembly.
- Dimensionless weights are not mixed into physical assembly.

## Current Reference Results

Standard 128-point Macro16 currently has:

- Mean recovery-force error: 0.00973.
- Max recovery-force error: 0.03578.
- Mean stiffness error: 0.01013.
- Max stiffness error: 0.04257.

## Output

Write `reports/macro16_force_stiffness_audit.md`.

The report must state whether the route passes the current closure gate and what
case or geometry causes the worst error.
