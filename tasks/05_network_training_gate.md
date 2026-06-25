# Task 05: Network Training Gate

Role: NN Agent.

## Goal

Train only after data and mechanics gates pass.

## Entry Criteria

Training is allowed only after:

- source128 compact audit passes,
- rigid preprocessing audit passes,
- recovery-force audit passes,
- stiffness audit passes,
- medium and required wind-turbine shell geometry generality audits pass or are
  explicitly waived. Strong non-inverted distortion is a non-blocking robustness
  boundary test.

## Default Contract

- Model: `Macro16BoundaryDeepONetWithLE0`.
- Input: `q48_def_hat`, `X16_hat`, standard 128-IP features.
- Output: `LE`.
- AD derivative: `dLE/dq48_def_hat`.
- B target: `B_macro_qdef`.
- Input dimension remains 48.

## Required Reporting

Report:

- train/validation split,
- LE error,
- B error,
- rigid constraint loss,
- worst IP,
- worst q48 column,
- downstream force and stiffness audit using network outputs if available.

## Output

Write `reports/macro16_network_training_gate.md`.
