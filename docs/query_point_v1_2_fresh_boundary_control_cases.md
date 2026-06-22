# v1.2 Fresh Boundary-Control Cases

This document describes the fresh Abaqus boundary-displacement control plan for
the selected legacy TRUE176 directions.

## Rule

The old TRUE176 data remains a boundary-displacement direction library only.
It is not a label source.

```text
q_dir = q48_raw / ||q48_raw||
q_new = alpha * q_dir
```

The old `LE` and `B` values are not v1.2 training labels.  They are used only as
scale hints when choosing conservative first-pass amplitudes.  A new case can
enter the formal v1.2 pool only after:

```text
fresh Abaqus boundary-displacement run
matching Sobolev B compact for the same q48 frames
complete_case*_training_ready.npz from the current exporter
strict_v1_2_pass = true
strain_field = LE
B_label_strain_field = LE
```

## Generated Plan

Use:

```powershell
py -3 scripts/build_v1_2_boundary_control_cases.py `
  --boundary-control-plan D:\IS-FEM\outputs\query_point_v1_2_data_coverage_audit\legacy176_candidate_audit\legacy176_boundary_control_plan.json `
  --out-root D:\IS-FEM\outputs\query_point_v1_2_data_coverage_audit\fresh_boundary_control_plan `
  --case-start 40 `
  --case-end 51 `
  --alpha-factors "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" `
  --alpha-scale 1.0
```

Outputs:

```text
fresh_boundary_control_plan.json
fresh_boundary_control_plan.csv
case040_q48_frames.csv
...
case051_q48_frames.csv
abaqus_boundary_control_command_templates.md
```

Each case has 10 conservative amplitude frames:

```text
alpha = alpha_factor * alpha_mid
alpha_mid = q_norm_seed_mean
alpha_factor = 0.1, 0.2, ..., 1.0
```

The generated q48 frame satisfies:

```text
||q48_frame|| = alpha
q48_frame / ||q48_frame|| = q_direction_48
```

## q48 Order

The current q48 contract is 16 control nodes times 3 displacement DOF:

```text
keep_node_ids = [1, 3, 5, 11, 15, 21, 23, 25, 26, 28, 30, 36, 40, 46, 48, 50]
q[3*i + 0] -> keep_node_ids[i], dof 1
q[3*i + 1] -> keep_node_ids[i], dof 2
q[3*i + 2] -> keep_node_ids[i], dof 3
```

Existing Abaqus `.inp` examples in the workspace often apply displacement
amplitudes on 32 boundary nodes / 96 DOF.  That is not the same object as q48.
If the fresh Abaqus generator expects 96 boundary DOF, use the existing boundary
contract / bridge for the same shape to map:

```text
q48 -> q_boundary[32,3]
```

Do not silently reinterpret q48 as a 96-DOF boundary vector.

## Abaqus Template

After a fresh ODB and matching Sobolev B compact are generated for a case:

```powershell
abaqus python scripts/export_abaqus_true176_complete_compact.py `
  --odb <fresh_case_job>.odb `
  --out <complete_caseXXX_training_ready.npz> `
  --merge-compact <matching_LE_B_compact.npz> `
  --require-b `
  --require-ip-audit `
  --require-merge-ip-keys `
  --strain-field LE `
  --merge-q-tol 1e-8 `
  --merge-le-tol 1e-8
```

Then rerun the strict v1.2 coverage audit.  Do not train until the new complete
compact passes strict v1.2.

