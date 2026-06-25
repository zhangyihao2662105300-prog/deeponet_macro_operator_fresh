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

3. Gate 04 working directory

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate04_generality`

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
```

Selected-frame volume force audits:

```powershell
py -3 scripts\audit_macro16_force_stiffness.py --compact-list runs\gate04_generality\regular_compact_list.txt --out runs\gate04_generality\regular_selected_material_only_force_audit.json --volume-mode selected-frame --tangent-mode material-only --max-force-rel 0.02 --max-macro-stiffness-symmetry-rel 1e-10 --workers 8
py -3 scripts\audit_macro16_force_stiffness.py --compact-list runs\gate04_generality\lightly_distorted_compact_list.txt --out runs\gate04_generality\lightly_distorted_selected_material_only_force_audit.json --volume-mode selected-frame --tangent-mode material-only --max-force-rel 0.02 --max-macro-stiffness-symmetry-rel 1e-10 --workers 8
py -3 scripts\audit_macro16_force_stiffness.py --compact-list runs\gate04_generality\moderately_distorted_compact_list.txt --out runs\gate04_generality\moderately_distorted_selected_material_only_force_audit.json --volume-mode selected-frame --tangent-mode material-only --max-force-rel 0.02 --max-macro-stiffness-symmetry-rel 1e-10 --workers 8
py -3 scripts\audit_macro16_force_stiffness.py --compact-list runs\gate04_generality\strong_non_flipped_intended_invalid_compact_list.txt --out runs\gate04_generality\strong_non_flipped_intended_invalid_selected_material_only_force_audit.json --volume-mode selected-frame --tangent-mode material-only --max-force-rel 0.02 --max-macro-stiffness-symmetry-rel 1e-10 --workers 8
```

## 4. Class Definition

规则和轻微畸变来自 base 10-case 数据，按标准 128 点处的 Macro16 几何畸变分数分类。

中等畸变来自 `case060`。

强畸变但不翻转本应来自 `case061`，但实际 compact 中的 `shape4` 和 `X16` 与 `case060` 一致，因此本轮只能记为“强畸变意图数据无效”。

## 5. Required Metrics

Force closure 使用 selected-frame physical volume，即 `ip_IVOL_abaqus_selected_frames`。

`q_zero` 和刚体工况的参考反力接近 0，relative force 会被分母放大。因此下表的 force 采用有效非刚体工况统计。完整 JSON 里保留了全量统计。

| Class | Geometry Count | Frame Count | detJ Positive | Data Contract | Force Mean | Force Max | Material-only K Mean | Material-only K Max | Enter Next Class |
|---|---:|---:|---|---|---:|---:|---:|---:|---|
| Regular | 3 | 30 | yes, min `0.001245609` | PASS | `0.003506506` | `0.008585725` | `0.011443767` | `0.031962110` | yes |
| Light distortion | 7 | 70 | yes, min `0.000744175` | PASS | `0.000315474` | `0.000575427` | `0.000330240` | `0.000472802` | yes |
| Medium distortion | 21 | 210 | yes, min `0.002315427` | PASS | `0.000137256` | `0.000248711` | `0.001706392` | `0.007058560` | yes, but strong data must be regenerated |
| Strong non-inverted | 21 | 210 | yes, min `0.002315427` | PASS | `0.000111752` | `0.000218634` | `0.001906917` | `0.006931154` | no |

## 6. Strong Geometry Finding

The strong non-inverted task list intended `shape4 = [1.0, 0.02, 1.3, 0.2]`.

But the actual complete compact and Macro16 compact contain:

```text
shape4 = [1.0, 0.02, 1.3, 0.0]
```

and `X16` matches `case060`.

Therefore the force and material-only K numbers in the strong row are numerically useful as a repeated medium-distortion check, but they do not verify strong non-inverted geometry.

## 7. Gate Result

Gate 04 result: incomplete.

Passed so far:

1. Regular geometry data contract and selected-frame force closure.
2. Light distortion data contract and selected-frame force closure.
3. Medium distortion data contract and selected-frame force closure.
4. detJ is positive for all audited effective geometries.

Not passed yet:

1. Strong non-inverted geometry has not been verified because the generated compact is not actually strong-distorted.
2. Cylindrical, conical, thickness-varying, and mild double-curvature families remain outside this report.

## 8. Unresolved Issues

1. Regenerate true strong non-inverted geometry and verify that `shape4`, `X16`, and detJ match the intended class before force audit.
2. Material-only stiffness remains a diagnostic only. The full Gate 03 tangent is still pending and not resolved here.
3. The current medium and strong-intended distorted sets are generated from cylindrical shape4-style data; broader shape families remain to be audited.

## 9. Recommended Next Action

1. Fix strong-distortion generation so the complete compact preserves `shape4 = [1.0, 0.02, 1.3, 0.2]` and the corresponding `X16`.
2. Rebuild Macro16 source128 compacts for the corrected strong non-inverted set.
3. Rerun Gate 04 data contract and selected-frame force closure.
4. Only after true strong non-inverted geometry passes, proceed to additional geometry families.
