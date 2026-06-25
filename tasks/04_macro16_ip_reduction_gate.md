# Task 04: Macro16 IP Reduction Gate

Role: Mechanics Agent.

## Goal

Only after the 128-point route is stable, evaluate whether 64-point or 32-point
rules can preserve force and stiffness closure.

## Entry Criteria

Do not run this task until standard 128-point Macro16 passes regular, light,
medium, and required wind-turbine shell geometry audits. Strong non-inverted
distortion is a non-blocking robustness boundary test and does not block this
task by itself.

## Required Checks

- 128-point baseline errors.
- 64-point force and stiffness errors.
- 32-point force and stiffness errors.
- Worst-case geometry and load cases.
- Whether reduction preserves mechanical closure, not just strain fit.

## Output

Write `reports/macro16_ip_reduction_gate.md`.

The report must clearly recommend keep 128, allow 64, or allow 32.
