# v1.2 Legacy TRUE176 Boundary-Control Regeneration Plan

Current stage: `v1.2 data-coverage audit`.

This document records how the old TRUE176 compact pool should be used in the
query-point/Abaqus route.  The old TRUE176 compacts are useful, but they are not
formal v1.2 training data.

## Rule

Old TRUE176 compact files must not be added directly to the formal v1.2
query-point/Abaqus training pool.

They may only be used as a candidate boundary-displacement/load-direction
library for selecting independent `q48_raw` directions.  The intended workflow
is:

```text
legacy q48_raw
-> q_dir = q48_raw / ||q48_raw||
-> q_new = alpha * q_dir
-> fresh Abaqus displacement-control run
-> current exporter + matching Sobolev B merge
-> complete_case*_training_ready.npz
-> strict_v1_2_pass
```

A candidate can enter the formal pool only after this fresh generation path
produces a current complete compact and passes the strict v1.2 audit:

```text
complete_case*_training_ready.npz
strict_v1_2_pass = true
strain_field = LE
B_label_strain_field = LE
training_ready_sobolev = true
```

The reason is that the old compacts do not carry the current route's required
hard-guard metadata, including point features, IP geometry/Jacobian tables,
`ip_keys`, explicit strain metadata, merge-audit fields, and reference-frame IP
audit fields.  Their old `LE/B` arrays are not trusted as v1.2 labels; they are
used only as scale hints when choosing new boundary displacement amplitudes.

## Current Strict-Pass Pool

Audit directory:

```text
D:\IS-FEM\outputs\query_point_v1_2_data_coverage_audit\full_pool_strict_pass
```

Current strict-pass complete compacts:

| case | q-direction cluster | LE_rms_mean | B_rms_mean |
| --- | ---: | ---: | ---: |
| case019 | 0 | 6.928620e-04 | 5.499105 |
| case025 | 1 | 1.137612e-04 | 5.497626 |
| case031 | 0 | 1.089809e-02 | 5.520879 |

Strict audit summary:

```text
strict_v1_2_pass = true
compact_count = 3
case_count = 3
frame_count = 30
q_direction_cluster_count = 2
pairwise_abs_cos_min = 0.8966673549
pairwise_abs_cos_median = 0.9078195202
pairwise_abs_cos_max = 0.9937922950
```

`case019` and `case031` are near-duplicate directions at
`abs(cos)=0.9937922950`, so the current pool is still too thin for v1.2.

## Legacy Candidate Audit

Script:

```text
scripts/select_legacy176_reexport_candidates.py
```

Command used:

```powershell
py -3 scripts/select_legacy176_reexport_candidates.py `
  --legacy-compact-glob "E:\true176_shape4_qraw_128ip_training_data_20260620\compacts\*\true176_shape4_template_plus_smoke_compact.npz" `
  --legacy-compact D:\IS-FEM\true176_merge_results_only_20260616\true176_cylinder_mild_theta15_plus48_compact_1760.npz `
  --current-manifest D:\IS-FEM\outputs\query_point_v1_2_data_coverage_audit\full_pool_strict_pass\compact_manifest.json `
  --out-root D:\IS-FEM\outputs\query_point_v1_2_data_coverage_audit\legacy176_candidate_audit `
  --top-count 12 `
  --new-case-start 40 `
  --compute-b-rms
```

Output directory:

```text
D:\IS-FEM\outputs\query_point_v1_2_data_coverage_audit\legacy176_candidate_audit
```

The audit produced:

```text
legacy176_candidate_manifest.csv
legacy176_candidate_manifest.json
legacy176_to_current_abs_cosine.csv
legacy176_candidate_cluster_summary.json
legacy176_reexport_plan.md
legacy176_boundary_control_plan.json
```

Summary:

```text
legacy_compact_count = 6
legacy_candidate_count = 733
legacy_q_direction_cluster_count = 17
recommended_candidate_count = 12
strong threshold = max_abs_cos_to_current < 0.90
candidate threshold = max_abs_cos_to_current < 0.95
compute_b_rms = true
```

Across all legacy candidates, 150 samples met the re-export recommendation
threshold: 100 strong candidates and 50 secondary candidates.  The top-12 list
below prioritizes directions with low `max_abs_cos_to_current`; most of these
are low-`LE_rms` samples, so medium-`LE_rms` backups are listed separately.

## Top Re-Export Candidates

| new case | legacy index | legacy case | direction type | LE_rms_mean | B_rms_mean | max abs cos to current | nearest current case |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| case040 | 674 | 1 | axial | 2.884773e-06 | 9.169774 | 0.226673 | 25 |
| case041 | 356 | 62 | bending | 1.503261e-04 | 9.169914 | 0.461194 | 31 |
| case042 | 102 | 103 | unknown | 1.539562e-04 | 5.497624 | 0.475236 | 31 |
| case043 | 112 | 113 | unknown | 3.586691e-04 | 5.501378 | 0.567921 | 19 |
| case044 | 476 | 102 | mixed:axial+bending | 1.508794e-04 | 9.170039 | 0.665379 | 31 |
| case045 | 127 | 128 | unknown | 2.554445e-04 | 5.494971 | 0.710425 | 31 |
| case046 | 702 | 9 | bending | 1.381921e-04 | 3.941842 | 0.766958 | 19 |
| case047 | 178 | 26 | mixed:axial+bending | 1.708498e-04 | 9.142733 | 0.771914 | 25 |
| case048 | 116 | 117 | unknown | 3.518223e-04 | 5.501471 | 0.772681 | 19 |
| case049 | 244 | 38 | mixed:shear+bending | 3.565561e-04 | 9.175514 | 0.836461 | 31 |
| case050 | 363 | 76 | mixed:axial+shear | 4.484425e-04 | 9.178932 | 0.853350 | 19 |
| case051 | 364 | 78 | mixed:axial+shear+bending | 3.585123e-04 | 9.175613 | 0.862846 | 31 |

These are candidates for re-export only.  They are not training data yet.
Use their `q_direction_48` values from `legacy176_boundary_control_plan.json`
as the boundary-displacement directions for new Abaqus runs.

## Medium-LE Backup Candidates

The strongest direction candidates mostly fall in the low-`LE_rms` bin.  If the
next data batch needs better `LE_rms` scale coverage, consider these secondary
medium-scale candidates after the top direction candidates:

| legacy index | legacy case | direction type | LE_rms_mean | B_rms_mean | max abs cos to current | nearest current case |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 203 | 36 | shear | 7.681796e-04 | 3.945937 | 0.822648 | 19 |
| 323 | 76 | mixed:axial+shear | 8.344067e-04 | 3.950522 | 0.861656 | 19 |
| 326 | 82 | mixed:axial+shear+bending | 7.563856e-04 | 3.949679 | 0.872095 | 31 |
| 206 | 42 | mixed:shear+bending | 7.456229e-04 | 3.949660 | 0.887741 | 31 |
| 444 | 118 | mixed:shear+bending | 1.031882e-03 | 3.950510 | 0.887914 | 19 |

No high-`LE_rms` legacy samples met the current `max_abs_cos_to_current < 0.95`
recommendation threshold in this audit.  If high-scale cases are needed, they
will likely require either accepting more direction similarity or generating new
load cases directly.

## Boundary-Control Generation Status

The repository has the current complete compact exporter:

```text
scripts/export_abaqus_true176_complete_compact.py
```

Fresh complete compact export template after a new displacement-controlled ODB
and matching B compact have been generated:

```powershell
abaqus python scripts/export_abaqus_true176_complete_compact.py `
  --odb <base.odb> `
  --out <complete_caseXXX_training_ready.npz> `
  --merge-compact <matching_LE_B_compact.npz> `
  --require-b `
  --require-ip-audit `
  --require-merge-ip-keys `
  --strain-field LE `
  --shape4-json <shape4-json-or-path> `
  --merge-q-tol 1e-8 `
  --merge-le-tol 1e-8
```

The current local machine has the completed case019/case025/case031 ODB and B
compact chain.  For the selected legacy directions, however, the important next
step is not to recover old labels.  It is to reapply each selected `q_dir` as a
new boundary-displacement control:

```text
q_new = alpha * q_dir
```

Then run Abaqus and regenerate the current complete compact.  The old
`sample_path` entries can still be helpful for shape/request provenance, but
they do not provide a complete current v1.2 training sample.  For the top-12
candidates:

```text
base_odb_exists = false
perturb_dir_exists = false
```

The available `sample_path` values usually point to old sample NPZ files or
sample roots, not to a base ODB plus the matching Sobolev B compact needed by
the exporter.

Before adding any recommended case to the formal pool, locate or regenerate:

```text
fresh base ODB path from q_new boundary displacement
matching Sobolev B compact path for the same q_new frames
Abaqus command/environment
shape4 argument or shape4-json
frame mapping / selected frames
perturb data used to generate B compact
```

Do not train on the legacy compact while these items are missing.

## Next Step

Use the top-12 candidates as a prioritized boundary-displacement direction list
for generating new Abaqus cases.  Pick several `alpha` amplitudes per direction
to cover low/medium/high `LE_rms` as needed.  After each fresh export, run the
strict coverage audit again and only add the new complete compact to the formal
v1.2 pool if `strict_v1_2_pass=true`.
