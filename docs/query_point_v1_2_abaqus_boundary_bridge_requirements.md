# v1.2 Abaqus Boundary Bridge Requirements

This note records the bridge from the v1.2 `q48` frame plans to Abaqus boundary
displacement inputs.

## Current Status

The bridge was found locally in existing TRUE176 boundary-contract files:

```text
boundary_contract/boundary_contract_arrays.npz
```

Required arrays:

```text
T_boundary_96x48: [96,48]
keep_nodes: [16]
boundary_nodes: [32]
```

Observed node orders:

```text
keep_nodes = [1, 3, 5, 11, 15, 21, 23, 25, 26, 28, 30, 36, 40, 46, 48, 50]
boundary_nodes = [1, 2, 3, 4, 5, 6, 10, 11, 15, 16, 20, 21, 22, 23, 24, 25,
                  26, 27, 28, 29, 30, 31, 35, 36, 40, 41, 45, 46, 47, 48, 49, 50]
```

The checked contracts share the same `T_boundary_96x48`, `keep_nodes`, and
`boundary_nodes`.  For a representative contract, the audit relation was exact:

```text
q_boundary_frames = q48_frames @ T_boundary_96x48.T
max_abs_diff = 0
```

The external UEL contract also documents the same route:

```text
q48_raw -> T_boundary_96x48 q48_raw -> q_boundary[32,3]
```

For the present fresh Abaqus input step, the generated boundary input is the raw
96-DOF boundary displacement:

```text
q_boundary96 = T_boundary_96x48 @ q48
```

Rigid-body cleanup is part of the neural-network preprocessing contract, not a
reason to reinterpret q48 as 96 DOF.

## Generated Boundary Inputs

Generate boundary tables with:

```powershell
py -3 scripts/build_v1_2_abaqus_boundary_inputs.py `
  --q48-plan-root D:\IS-FEM\outputs\query_point_v1_2_data_coverage_audit\fresh_boundary_control_plan `
  --out-root D:\IS-FEM\outputs\query_point_v1_2_data_coverage_audit\fresh_abaqus_boundary_inputs `
  --bridge-contract D:\IS-FEM\t176_cyl100_pilot6_cases001_020_full48\sample_893060_cyl_lam100_tau006_theta24_case001_single_Axial_Force_+1\boundary_contract\boundary_contract_arrays.npz `
  --case-start 40 `
  --case-end 51
```

Outputs:

```text
fresh_abaqus_boundary_inputs_summary.json
fresh_abaqus_boundary_inputs_summary.csv
case###_boundary96_frames.csv
case###_boundary_nodes.json
case###_abaqus_bc_commands.txt
```

The CSV output uses rows:

```text
case_id, frame_index, alpha, q_norm, boundary_node_index, abaqus_node_label, ux, uy, uz
```

The Abaqus boundary command files are templates only.  They must be inserted
into a matching fresh Abaqus model/step with the same node labels and geometry
contract.

## Pilot Order

Do not run all cases first.  Suggested pilot order:

```text
1. case041
2. case043
3. case050
4. remaining case040-case051
```

Reason:

```text
case040 has a very small alpha range and may produce weak strain.
case041 and case043 are more useful mid-amplitude fresh-direction pilots.
case050 checks a mixed axial/shear direction at a larger alpha range.
```

## Remaining Requirements Before Formal v1.2 Pool Entry

For each pilot case:

```text
q48_frames.csv
-> boundary96_frames.csv / Abaqus BC template
-> fresh Abaqus ODB
-> matching Sobolev B compact for the same q48 frames
-> complete_case###_training_ready.npz
-> strict_v1_2_pass = true
```

Still required from the Abaqus generation side:

```text
base model / inp path for the fresh geometry
shape4 or shape4-json used for that fresh model
step/increment convention for the 10 q48 frames
confirmed field output requests for U, LE, COORD, IVOL
matching finite-difference/Sobolev B compact generation command
```

## Exporter Template

Use the actual exporter argument names:

```powershell
abaqus python scripts/export_abaqus_true176_complete_compact.py `
  --odb <fresh_case###.odb> `
  --out <complete_case###_training_ready.npz> `
  --shape4 <shape4-values> `
  --merge-compact <matching_sobolev_B_compact.npz> `
  --strain-field LE `
  --require-b `
  --require-merge-ip-keys `
  --require-ip-audit `
  --merge-q-tol 1e-8 `
  --merge-le-tol 1e-8
```

## Strict v1.2 Audit Template

```powershell
py -3 scripts/audit_query_point_data_coverage.py `
  --compact-glob "D:\path\to\complete_case*_training_ready.npz" `
  --out-root D:\IS-FEM\outputs\query_point_v1_2_data_coverage_audit\full_pool_after_fresh_cases `
  --strict-v1-2
```

## Non-Negotiable Boundary

Old TRUE176 `LE/B` values are not v1.2 labels.  The only accepted labels for the
formal pool come from fresh Abaqus runs, matching Sobolev B compacts, current
complete-compact export, and strict v1.2 audit.

