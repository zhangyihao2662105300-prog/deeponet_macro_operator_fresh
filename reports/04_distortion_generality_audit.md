# Gate 04 Distortion Generality Audit

Date: 2026-06-25

Role: Gate 04 Generality Agent

## 1. Task Goal

推进 Macro16 source128 泛用性审计。

本轮不训练网络，不改模型，不改 128 点积分规则。Gate 03 完整刚度切线暂时标记为待处理，不阻塞本任务。Material-only stiffness 只记录为诊断量，不作为 Gate 04 阻塞条件。

## 2. Data Used

1. Base 10-case Macro16 source128 compacts

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\macro16_source128_teacher_compact_list.txt`

2. Distorted 42 compact set

`D:\IS-FEM\outputs\macro16_source128_generality_distorted_tasks_local\postprocess\macro16_source128_x16weights\macro16_source128_teacher_compact_list.txt`

3. Repaired strong non-inverted `case061` compact set

Complete compact list:

`D:\IS-FEM\outputs\macro16_source128_generality_case061_repair\strong_case061_repaired_complete_compact_list.txt`

Rebuilt Macro16 source128 compact list:

`D:\IS-FEM\outputs\macro16_source128_generality_case061_repair\postprocess\macro16_source128_x16weights\macro16_source128_teacher_compact_list.txt`

This repaired set contains 21 Macro16 source128 compacts and 210 frames.

4. Gate 04 working directory

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate04_generality`

5. Wind-turbine shell geometry audit directory

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate04_wind_shell_generality`

Generated Macro16 source128 compact list:

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate04_wind_shell_generality\macro16_source128\macro16_source128_teacher_compact_list.txt`

This wind-shell set contains 4 Macro16 source128 compacts and 40 total base
frames. The selected-frame force audit reports 1 selected frame per compact.

## 3. Commands Used

Geometry inventory and class lists:

```powershell
$env:PYTHONPATH='src'
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
py -3 - <geometry inventory script>
```

Data contract audits:

```powershell
py -3 scripts\audit_macro16_rigid_preprocessing.py --compact-list runs\gate04_generality\regular_compact_list.txt --out runs\gate04_generality\regular_data_contract_audit.json --skip-synthetic --strict
py -3 scripts\audit_macro16_rigid_preprocessing.py --compact-list runs\gate04_generality\lightly_distorted_compact_list.txt --out runs\gate04_generality\lightly_distorted_data_contract_audit.json --skip-synthetic --strict
py -3 scripts\audit_macro16_rigid_preprocessing.py --compact-list runs\gate04_generality\moderately_distorted_compact_list.txt --out runs\gate04_generality\moderately_distorted_data_contract_audit.json --skip-synthetic --strict
py -3 scripts\audit_macro16_rigid_preprocessing.py --compact-list runs\gate04_generality\strong_non_flipped_intended_invalid_compact_list.txt --out runs\gate04_generality\strong_non_flipped_intended_invalid_data_contract_audit.json --skip-synthetic --strict
py -3 scripts\audit_macro16_rigid_preprocessing.py --compact-list D:\IS-FEM\outputs\macro16_source128_generality_case061_repair\postprocess\macro16_source128_x16weights\macro16_source128_teacher_compact_list.txt --out runs\gate04_generality\strong_non_flipped_repaired_data_contract_audit.json --skip-synthetic --strict
```

Selected-frame volume force audits:

```powershell
py -3 scripts\audit_macro16_force_stiffness.py --compact-list runs\gate04_generality\regular_compact_list.txt --out runs\gate04_generality\regular_selected_material_only_force_audit.json --volume-mode selected-frame --tangent-mode material-only --max-force-rel 0.02 --max-macro-stiffness-symmetry-rel 1e-10 --workers 8
py -3 scripts\audit_macro16_force_stiffness.py --compact-list runs\gate04_generality\lightly_distorted_compact_list.txt --out runs\gate04_generality\lightly_distorted_selected_material_only_force_audit.json --volume-mode selected-frame --tangent-mode material-only --max-force-rel 0.02 --max-macro-stiffness-symmetry-rel 1e-10 --workers 8
py -3 scripts\audit_macro16_force_stiffness.py --compact-list runs\gate04_generality\moderately_distorted_compact_list.txt --out runs\gate04_generality\moderately_distorted_selected_material_only_force_audit.json --volume-mode selected-frame --tangent-mode material-only --max-force-rel 0.02 --max-macro-stiffness-symmetry-rel 1e-10 --workers 8
py -3 scripts\audit_macro16_force_stiffness.py --compact-list runs\gate04_generality\strong_non_flipped_intended_invalid_compact_list.txt --out runs\gate04_generality\strong_non_flipped_intended_invalid_selected_material_only_force_audit.json --volume-mode selected-frame --tangent-mode material-only --max-force-rel 0.02 --max-macro-stiffness-symmetry-rel 1e-10 --workers 8
py -3 scripts\audit_macro16_force_stiffness.py --compact-list runs\gate04_generality\strong_non_flipped_repaired_effective_force_compact_list.txt --out runs\gate04_generality\strong_non_flipped_repaired_effective_selected_material_only_force_audit.json --volume-mode selected-frame --tangent-mode material-only --max-force-rel 0.02 --max-macro-stiffness-symmetry-rel 1e-10 --workers 8 --strict
```

Strong non-inverted repair generation:

```powershell
py -3 scripts\run_v1_2_pilot_case041_fresh_abaqus.py --case-id <case061_id> --q48-frames-csv <case061_task>\q48_frames.csv --boundary96-frames-csv <case061_task>\boundary96_frames.csv --bridge-contract <case061_task>\boundary_contract_arrays.npz --out-root D:\IS-FEM\outputs\macro16_source128_generality_case061_repair\strongly_distorted_not_flipped\<case061_id>\fresh_run --shape4 1,0.02,1.3,0.2 --increments 10 --delta 1e-6 --inner-workers 4 --run-abaqus --run-complete-export --run-strict-audit
py -3 scripts\build_macro16_source128_teacher.py --compact-list D:\IS-FEM\outputs\macro16_source128_generality_case061_repair\strong_case061_repaired_complete_compact_list.txt --out-root D:\IS-FEM\outputs\macro16_source128_generality_case061_repair\postprocess\macro16_source128_x16weights --source-node-order auto --strain-coordinate-mode global-to-macro-local --volume-weight-mode macro16-x16 --workers 4
```

Wind-turbine shell generation and audit:

```powershell
$env:PYTHONPATH='src;scripts'
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
py -3 scripts\run_gate04_wind_shell_generality_audit.py --out-root runs\gate04_wind_shell_generality --inner-workers 4 --post-workers 8 --skip-existing
```

Output summary:

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate04_wind_shell_generality\gate04_wind_shell_summary.json`

## 4. Class Definition

规则和轻微畸变来自 base 10-case 数据，按标准 128 点处的 Macro16 几何畸变分数分类。

中等畸变来自 `case060`。

强畸变但不翻转来自 repaired `case061`。旧 `case061` 数据无效，因为旧 21 个
fresh-run plan 全部把 `shape4` 写成 `[1.0, 0.02, 1.3, 0.0]`，导致 complete
compact 和 Macro16 compact 与 `case060` 重复。repaired `case061` 重新运行
Abaqus 和 complete export，显式使用 `--shape4 1,0.02,1.3,0.2`。

After route correction, strong non-inverted distortion is treated as a
non-blocking robustness boundary test, not a hard Gate 04 main-route
requirement. The required main-route Gate 04 families are regular, light
distortion, medium distortion, and typical wind-turbine shell geometries such
as cylindrical, conical, thickness-varying, and mild double-curvature shells.

## 5. Required Metrics

Force closure 使用 selected-frame physical volume，即 `ip_IVOL_abaqus_selected_frames`。

`q_zero` 和刚体工况的参考反力接近 0，relative force 会被分母放大。因此下表的 force 采用有效非刚体工况统计。完整 JSON 里保留了全量统计。

| Class | Geometry Count | Frame Count | detJ Positive | Data Contract | Force Mean | Force Max | Material-only K Mean | Material-only K Max | Enter Next Class |
|---|---:|---:|---|---|---:|---:|---:|---:|---|
| Regular | 3 | 30 | yes, min `0.001245609` | PASS | `0.003506506` | `0.008585725` | `0.011443767` | `0.031962110` | yes |
| Light distortion | 7 | 70 | yes, min `0.000744175` | PASS | `0.000315474` | `0.000575427` | `0.000330240` | `0.000472802` | yes |
| Medium distortion | 21 | 210 | yes, min `0.002315427` | PASS | `0.000137256` | `0.000248711` | `0.001706392` | `0.007058560` | yes |
| Strong non-inverted robustness, repaired | 21 | 210 | yes, min `0.001723677` | PASS | `0.000096867` | `0.000233469` | `0.002260105` | `0.006567508` | yes, robustness boundary passes |

## 6. Strong Geometry Repair

The strong non-inverted task list intended `shape4 = [1.0, 0.02, 1.3, 0.2]`.

The old generated complete compact and Macro16 compact contained:

```text
shape4 = [1.0, 0.02, 1.3, 0.0]
```

and `X16` matched `case060`. The root cause was the old fresh-run command:
all 21 old `case061_*` plan files used `--shape4 1,0.02,1.3,0`.

The repaired run regenerated all 21 `case061` complete compacts with:

```text
shape4 = [1.0, 0.02, 1.3, 0.2]
```

Repair checks:

- `case061` no longer duplicates `case060`: repaired `X16` differs from
  `case060` by max absolute difference `0.115132153`.
- detJ is positive on all 210 repaired frames and all 128 fixed parent points:
  min `0.0017236767153505214`, max `0.002591064959109894`.
- Data contract audit passes on 21/21 repaired Macro16 source128 compacts.
- Full relative force statistics still include `q_zero` and rigid translation
  cases with reference reaction RMS around `1e-12`, so that all-case relative
  max is not used as the force gate.
- Effective nonzero-reaction force audit uses 17 compacts and passes the
  `0.02` force gate: mean `0.00009686685671651643`, max
  `0.00023346870978522232`.
- Worst repaired effective force case is
  `case061_11_random_48_small_pos_002`, with force rel
  `0.00023346870978522232`.
- Material-only K remains diagnostic only: mean `0.002260104975094106`, max
  `0.006567508247031507`.

## 7. Wind-Turbine Shell Geometry Audit

The wind-turbine shell audit generated four teacher CSS8 patches, converted
them to Macro16 source128 compacts, and ran the strict data-contract audit plus
selected-frame volume force audit. The script did not train a network, did not
modify model code, and did not change the fixed 128-point rule.

| Shell Family | Geometry Count | Frame Count | Selected Force Frames | detJ Positive | Data Contract | Force Mean | Force Max | Material-only K Mean | Material-only K Max | Enter Next Class |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|
| Cylindrical shell | 1 | 10 | 1 | yes, min `0.002461802` | PASS | `0.000026252` | `0.000026252` | `0.000087750` | `0.000087750` | yes |
| Conical shell | 1 | 10 | 1 | yes, min `0.002022906` | PASS | `0.000027973` | `0.000027973` | `0.000109798` | `0.000109798` | yes |
| Thickness-varying shell | 1 | 10 | 1 | yes, min `0.002254853` | PASS | `0.000025519` | `0.000025519` | `0.000090403` | `0.000090403` | yes |
| Mild double-curvature shell | 1 | 10 | 1 | yes, min `0.002482064` | PASS | `0.000025423` | `0.000025423` | `0.000087221` | `0.000087221` | yes |

Pass standard:

1. detJ must be positive.
2. Data contract must pass.
3. Selected-frame force max must be below `0.02`.
4. Material-only K is recorded as a diagnostic only.

All four wind-turbine shell families pass the required Gate 04 selected-frame
force standard. The largest wind-shell force max is `0.000027973`, far below
`0.02`.

## 8. Gate Result

Gate 04 required-family result: PASS for the current main route.

Passed:

1. Regular geometry data contract and selected-frame force closure.
2. Light distortion data contract and selected-frame force closure.
3. Medium distortion data contract and selected-frame force closure.
4. Cylindrical shell data contract and selected-frame force closure.
5. Conical shell data contract and selected-frame force closure.
6. Thickness-varying shell data contract and selected-frame force closure.
7. Mild double-curvature shell data contract and selected-frame force closure.
8. Strong non-inverted repaired robustness geometry data contract and
   selected-frame force closure.
9. detJ is positive for all audited required and repaired robustness
   geometries.

Not a blocker:

1. Material-only stiffness remains a diagnostic only. The full Gate 03 tangent is still pending and not resolved here.

## 9. Unresolved Issues

1. Material-only stiffness is not a complete Abaqus tangent. The full Gate 03 consistent tangent remains unresolved.
2. Gate 05 integration-point reduction is still not started.
3. Broader production geometry sampling can still expand beyond the audited
   representative wind-shell families.

## 10. Recommended Next Action

1. Keep Gate 04 main-route required families marked as passed.
2. Treat repaired strong non-inverted distortion as a passed robustness
   boundary check, not as a network-training signal.
3. Return to Gate 03 consistent tangent decision or Gate 05 only when the project explicitly chooses that route.
4. Do not train the network yet because Gate 02 full teacher stiffness, Gate 03 full tangent, and Gate 05 still block Gate 06.
