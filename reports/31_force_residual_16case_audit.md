# Gate 31 Force Residual 16 Case Audit

日期：2026-06-26

## 1. 目标

把 Gate 30 的 two-case first6 force residual 结果扩展到 16 case 小集合。

本轮目标是验证当前最佳设置是否能跨规则、畸变和风机壳几何保持 selected-frame force closure。

不训练大模型。

不改模型结构。

不改 `q48` 顺序。

不改 `LE` 顺序。

不改 128 点规则。

## 2. 重要更正

第一轮 Gate 31 训练用了旧 compact。

旧 10-case compact 缺少：

```text
q48_def_hat
B_macro_qdef
rigid_projection_P
```

因此第一轮旧 compact 训练结果作废，不能作为 Gate 31 有效结论。

有效结论只基于后续从 source compact 重新生成的 qdef source128 compact。

## 3. 有效数据

Linux 机器目录：

```text
/home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo
```

有效 compact 生成脚本：

```text
scripts/build_macro16_source128_teacher.py
```

elastic_D 富集脚本：

```text
scripts/enrich_macro16_compact_elastic_d.py
```

有效 enriched list：

```text
runs/gate31_force_residual_16case_qdef/enriched_16case_compact_list.txt
```

16 case：

```text
19 25 31 41 43 44 45 46 49 50 60 61 70 71 72 73
```

验证集：

```text
70 71 72 73
```

## 4. 训练设置

沿用 Gate 30 最佳设置。

```text
model le0-fixed128-state-b
state_b_rank 12
direct_state_b_loss_weight 5.0
force_residual_loss_weight 0.1 或 0.05
le_loss_weight 1.0
jacobian_loss_weight 0.0
batch_size 16
epochs 3000
no B prior warmstart
```

对照训练：

```text
f01_lr8e5
f01_lr4e5
f005_lr8e5
```

## 5. 训练指标

| run | train LE rel | train AD_B rel | val LE rel | val AD_B rel |
|---|---:|---:|---:|---:|
| f01_lr8e5 | 0.1354864684 | 0.1892318696 | 29.3649670335 | 2.2703607369 |
| f01_lr4e5 | 0.1529806635 | 0.2254616972 | 32.2863710948 | 2.0425101637 |
| f005_lr8e5 | 0.1266386517 | 0.1879824157 | 29.1711167937 | 2.3216505914 |

训练集下降，但风机壳验证集 `70 71 72 73` 明显崩溃。

## 6. 整体 Force Audit

体积口径：

```text
selected-frame
```

| run | model force rel | teacher force rel |
|---|---:|---:|
| f01_lr8e5 | 2.8591430992 | 0.0016118213 |
| f01_lr4e5 | 3.1723748486 | 0.0016118213 |
| f005_lr8e5 | 3.0138454652 | 0.0016118213 |

三个 run 都未通过 Gate 31。

## 7. 逐 Case Force Audit

逐 case audit 已对 `f01_lr8e5` 完成。

| case | model force rel | teacher force rel | 结论 |
|---|---:|---:|---|
| 19 | 2.8591430992 | 0.0016118213 | FAIL |
| 25 | 20.0507191984 | 0.0003280678 | FAIL |
| 31 | 1.0278027900 | 0.0103085390 | FAIL |
| 41 | 11.1842734558 | 0.0002256038 | FAIL |
| 43 | 3.3226841590 | 0.0002986425 | FAIL |
| 44 | 10.8819204469 | 0.0002114396 | FAIL |
| 45 | 5.0073809332 | 0.0002860184 | FAIL |
| 46 | 7.6560285462 | 0.0001620829 | FAIL |
| 49 | 4.0124820698 | 0.0004404087 | FAIL |
| 50 | 2.9229902602 | 0.0005755133 | FAIL |
| 60 | 8.447255763109e12 | 0.8126680805 | INVALID RELATIVE FORCE GATE SAMPLE |
| 61 | 2.7505839423 | 0.0000183974 | FAIL |
| 70 | 56.7939266227 | 0.0000256137 | FAIL |
| 71 | 48.2835347727 | 0.0000265392 | FAIL |
| 72 | 62.0157398467 | 0.0000249434 | FAIL |
| 73 | 93.8086133163 | 0.0000241213 | FAIL |

## 8. Gate 31 结论

```text
FAIL
```

原因：

1. 16 case 混合训练没有保持 Gate 30 的 two-case first6 force closure。
2. 训练集指标下降，但 selected-frame force audit 失败。
3. 风机壳验证 cases `70 71 72 73` 的 teacher force closure 很好，但模型 force closure 极差。
4. `case060` 是 q_zero 或近零力工况，teacher relative force 也不正常，不应作为普通 relative force gate 样本。

## 9. 诊断判断

当前问题不是：

```text
q48 顺序错误
LE 顺序错误
128 点规则错误
teacher force audit 路径错误
```

当前问题更可能是：

```text
跨几何族分布差异
风机壳 70 到 73 的尺度或分布离群
relative force 在近零力样本上的门槛定义不适合
当前模型和 loss 不能直接从 first6 扩展到 16 case
```

## 10. 下一步

进入 Gate 32。

不要继续扩大训练。

不要改模型结构。

先做风机壳泛化失败诊断。

Gate 32 需要统计：

1. train cases 和 wind-shell val cases 的 `q48_def_hat` 范数。
2. `LE_macro` 范数。
3. `B_macro_qdef` 范数。
4. teacher assembled force 范数。
5. branch norm 标准化后风机壳是否离群。
6. `case070` 到 `case073` 是否因为 q 或 force 太小导致 relative error 爆炸。

建议报告：

```text
reports/32_wind_shell_generalization_diagnosis.md
```
