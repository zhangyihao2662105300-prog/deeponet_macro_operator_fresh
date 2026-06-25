# Gate 06 Training Data Readiness

Date: 2026-06-25

Role: Training Data Readiness Agent

## 1. Task Goal

训练前确认 Macro16 source128 训练数据是否完整可用。

本轮不训练网络，不改模型，不改 `q48` 顺序，不改 `LE` 顺序，不改 128 点积分规则。

## 2. Data Used

- regular: `D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate04_generality\regular_compact_list.txt`
- lightly_distorted: `D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate04_generality\lightly_distorted_compact_list.txt`
- moderately_distorted: `D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate04_generality\moderately_distorted_compact_list.txt`
- strong_non_flipped_repaired: `D:\IS-FEM\outputs\macro16_source128_generality_case061_repair\postprocess\macro16_source128_x16weights\macro16_source128_teacher_compact_list.txt`
- cylindrical_shell: `D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate04_wind_shell_generality\audits\cylindrical_shell\cylindrical_shell_macro16_source128_compact_list.txt`
- conical_shell: `D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate04_wind_shell_generality\audits\conical_shell\conical_shell_macro16_source128_compact_list.txt`
- thickness_varying_shell: `D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate04_wind_shell_generality\audits\thickness_varying_shell\thickness_varying_shell_macro16_source128_compact_list.txt`
- mild_double_curvature_shell: `D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate04_wind_shell_generality\audits\mild_double_curvature_shell\mild_double_curvature_shell_macro16_source128_compact_list.txt`

## 3. Commands Used

```powershell
py -3 -m py_compile scripts\audit_macro16_training_data_readiness.py
py -3 scripts\audit_macro16_training_data_readiness.py --strict
```

JSON detail:

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate06_training_data_readiness\training_data_readiness.json`

## 4. Overall Result

- Status: PASS
- Compact count: `56`
- Frame count: `560`
- Case count: `16`
- Geometry hash count: `8`
- Required categories present: PASS
- Required fields present: PASS
- Standard 128 point rule match: PASS
- No `X_macro` or CSS8 fine-grid geometry exposed to model: PASS

## 5. Category Coverage

| Category | Compact Count | Frame Count | Case IDs | Geometry Count | Field Contract | 128 Rule | Loader Ready |
|---|---:|---:|---|---:|---|---|---|
| regular | 3 | 30 | `[19, 25, 31]` | 1 | PASS | PASS | PASS |
| lightly_distorted | 7 | 70 | `[41, 43, 44, 45, 46, 49, 50]` | 1 | PASS | PASS | PASS |
| moderately_distorted | 21 | 210 | `[60]` | 1 | PASS | PASS | PASS |
| strong_non_flipped_repaired | 21 | 210 | `[61]` | 1 | PASS | PASS | PASS |
| cylindrical_shell | 1 | 10 | `[70]` | 1 | PASS | PASS | PASS |
| conical_shell | 1 | 10 | `[71]` | 1 | PASS | PASS | PASS |
| thickness_varying_shell | 1 | 10 | `[72]` | 1 | PASS | PASS | PASS |
| mild_double_curvature_shell | 1 | 10 | `[73]` | 1 | PASS | PASS | PASS |

## 6. Required Field Checks

Required model/data fields:

1. `q48_def_hat`: present, shape `[N,48]`.
2. `X16_hat`: present, shape `[N,16,3]`.
3. `LE_macro`: present, shape `[N,128,6]`.
4. `B_macro_qdef`: present, shape `[N,128,6,48]`.
5. `macro16_point_xi`: present and equal to `macro16_source128_point_table()`.
6. `case_id`: present for case-isolated split.

Result:

- Required fields: PASS
- 128 point rule: PASS

## 7. Train Val Split Readiness

- Case split possible: PASS
- Geometry/category holdout possible: PASS
- Recommended split: `case/category isolated`
- Recommended validation cases: `[70, 71, 72, 73]`
- Frame-random split: not recommended.

说明：loader 支持 `val_cases` 做 case 隔离。若要严格几何泛化，应按类别或几何哈希选择验证集，而不是只做随机 frame split。

## 8. Model Feed Readiness

The data can be loaded by the current Macro16 training loader with:

```text
model = Macro16BoundaryDeepONetWithLE0
q input = q48_def_hat
geometry input = X16_hat
label = LE_macro
B target = B_macro_qdef
integration rule = standard source128
```

Result: PASS

## 9. Decision

Training data readiness: PASS for limited LE/B training.

可以直接喂给 `Macro16BoundaryDeepONetWithLE0` 做 limited LE/B training。

边界：这不是 full tangent 通过，也不是 solver-ready 结论。训练后仍必须做 selected-frame force closure。Material-only K 只作为 diagnostic。

## 10. Unresolved Risks

1. `B_macro_qdef` 仍使用当前小转角线性刚体投影近似。
2. Full tangent 仍待处理，不由本报告释放。
3. Strong non-inverted repaired data可作为鲁棒性覆盖，但训练 split 应避免把同一几何的近似重复 frame 同时放入 train 和 val 来冒充泛化。

## 11. Recommended Next Action

1. 若开始 limited training，使用 128 点 source128 compact。
2. 使用显式 `val_cases` 做 case 或类别隔离。
3. 训练后必须重新跑 selected-frame force closure。

