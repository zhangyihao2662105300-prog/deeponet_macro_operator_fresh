# Gate 09 B Loss Scale Diagnosis

Date: 2026-06-25

Role: B Loss Scale Diagnosis Agent

## 1. Task Goal

诊断当前 `AD_B` loss 的标准化口径，解释为什么 `LE` 能下降但 `AD_B` 不能稳定拟合。

本轮只做 loss 和数据尺度诊断。

不训练大模型，不改模型结构，不改 `q48` 顺序，不改 `LE` 顺序，不改 128 点积分规则。

## 2. Data Used

诊断数据：

1. `runs\gate08_training_overfit_sanity\overfit_case019_first6_compact_list.txt`
2. `runs\gate08_training_overfit_sanity\overfit_case031_first6_compact_list.txt`
3. `runs\gate07_training_smoke\macro16_training_smoke_compact_list.txt`

其中：

1. case019 和 case031 是 Gate 08 overfit 失败样本
2. Gate 07 smoke 数据作为多几何对照
3. 输入仍为 `q48_def_hat` 和 `X16_hat`
4. 标签仍为 `LE_macro` 和 `B_macro_qdef`
5. 积分点仍为标准 source128

## 3. Commands Used

```powershell
py -3 -m py_compile scripts\diagnose_macro16_b_loss_scale.py
```

```powershell
py -3 scripts\diagnose_macro16_b_loss_scale.py --compact-list runs\gate08_training_overfit_sanity\overfit_case019_first6_compact_list.txt --val-cases 1902 --out runs\gate09_b_loss_scale_diagnosis\case019_first6_b_loss_scale.json
```

```powershell
py -3 scripts\diagnose_macro16_b_loss_scale.py --compact-list runs\gate08_training_overfit_sanity\overfit_case031_first6_compact_list.txt --val-cases 3102 --out runs\gate09_b_loss_scale_diagnosis\case031_first6_b_loss_scale.json
```

```powershell
py -3 scripts\diagnose_macro16_b_loss_scale.py --compact-list runs\gate07_training_smoke\macro16_training_smoke_compact_list.txt --val-cases 70,71,72,73 --max-frames-per-compact 2 --out runs\gate09_b_loss_scale_diagnosis\gate07_smoke_first2_b_loss_scale.json
```

## 4. Current B Normalization

训练器当前使用：

```text
branch_raw = [q48_def_hat, X16_hat.flatten, L_ref]
j_norm_target = B_macro_qdef * q_std / LE_std
```

反算检查：

```text
B_recovered = j_norm_target * LE_std / q_std
```

结果：

| Data | B recovery rel | B recovery max abs |
|---|---:|---:|
| case019 first6 | 5.21e-17 | 1.42e-14 |
| case031 first6 | 4.97e-17 | 1.42e-14 |
| Gate07 first2 | 5.11e-17 | 2.84e-14 |

判断：

1. 转换公式本身是闭合的。
2. 问题不是 `B_macro_qdef` 被转换错。
3. 问题是 `j_norm_target` 的优化尺度和物理 `B_macro_qdef` 误差不一致。

## 5. Key Metrics

| Data | B rms | B max | J rms | J max | q std ratio | LE std ratio | scale ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| case019 first6 | 5.50 | 72.76 | 284.40 | 17633.29 | 219.30 | 4146.39 | 909285.71 |
| case031 first6 | 5.50 | 73.00 | 40.60 | 2589.83 | 4270.74 | 494.74 | 2112898.86 |
| Gate07 first2 | 4.54 | 119.06 | 15.57 | 870.70 | 263.09 | 37.76 | 9934.54 |

这里 `scale ratio` 是 `q_std / LE_std` 的最大最小比。

判断：

1. 物理 `B_macro_qdef` 尺度相对正常。
2. `j_norm_target` 被 `q_std / LE_std` 放大后动态范围极大。
3. case019 的 `J max` 达到 `17633.29`，远大于物理 `B max = 72.76`。
4. case031 的 `J max` 达到 `2589.83`，远大于物理 `B max = 73.00`。

## 6. Energy Concentration

| Data | J top 0.1 percent | J top 1 percent | J top 5 percent |
|---|---:|---:|---:|
| case019 first6 | 0.8695 | 0.9899 | 0.9995 |
| case031 first6 | 0.5675 | 0.9106 | 0.9924 |
| Gate07 first2 | 0.4950 | 0.8664 | 0.9860 |

判断：

1. case019 中，`j_norm_target` 前 1 percent 张量项占 98.99 percent 能量。
2. case031 中，前 1 percent 占 91.06 percent 能量。
3. Gate07 多几何对照中，前 1 percent 仍占 86.64 percent 能量。
4. 当前 `J_norm_mse` 会优先拟合极少数高权重点。
5. 这解释了为什么 B-only 能降低 `J_norm_mse`，但物理 `AD_B_rel` 反而不好。

## 7. Group Bias

`j_norm_target` 与物理 `B_macro_qdef` 的能量分布差异：

| Data | column shift | component shift | IP shift |
|---|---:|---:|---:|
| case019 first6 | 0.7585 | 0.4876 | 0.6882 |
| case031 first6 | 0.8331 | 0.3988 | 0.5697 |
| Gate07 first2 | 0.8413 | 0.3878 | 0.4586 |

数值为 half L1 distance，越大表示 `J_norm` 口径越偏离物理 `B` 口径。

典型现象：

1. case031 物理 B 的列能量较均匀，但 `j_norm_target` 主要集中在列 `12,36,30,6,0`。
2. Gate07 对照中，`j_norm_target` 也主要集中在列 `12,36,6,0,30`。
3. 分量和积分点也有明显重排。
4. 当前 loss 不是在均衡学习 48 列物理 B。

## 8. Low Scale Source

case031 最小 `q_std`：

```text
column 39: 3.50e-6
column 15: 3.50e-6
column 27: 5.22e-6
column 3 : 5.22e-6
```

case031 最小 `LE_std`：

```text
ip 59 component 4: 1.75e-5
ip 64 component 4: 2.18e-5
```

case019 更极端：

```text
LE_std min = 1.33e-7
LE_std ratio = 4146.39
```

判断：

1. 极小 `LE_std` 会把对应 `j_norm_target` 放大。
2. 极窄 q 覆盖会让 `q_std` 统计不稳定。
3. 两者组合后，`q_std / LE_std` 的点状比例会主导 loss。

## 9. Result

Gate 09 B loss scale diagnosis: FAIL for current B loss scaling.

结论：

1. `B_macro_qdef` 到 `j_norm_target` 的公式闭合。
2. 当前 `J_norm_mse` 不等价于物理 `B_macro_qdef` 拟合。
3. 当前 B loss 被极端 `q_std / LE_std` 尺度主导。
4. 当前 B loss 会过度关注少数列，少数分量，少数积分点。
5. 这能解释 Gate 08 中 `LE` 能下降但 `AD_B_rel` 不稳定下降。

## 10. Recommended B Loss Scheme

建议下一步不改模型结构，只改 B loss 口径。

主方案：

```text
先用 autograd 得到 J_pred_norm
再转换回物理 qdef 坐标
B_pred = J_pred_norm * LE_std_eff / q_std_eff
然后直接监督 B_macro_qdef
```

loss：

```text
loss_B = balanced_mean(((B_pred - B_macro_qdef) / B_scale)^2)
```

其中：

1. `B_scale` 用 `B_macro_qdef` 的 robust RMS。
2. `B_scale` 至少按 q 列和 LE 分量分组。
3. `B_scale` 必须有全局 floor，避免小样本零尺度。
4. 每列先平均，再对 48 列平均。
5. 每个 LE 分量先平均，再对 6 分量平均。
6. 必要时再加积分点分组平衡。

建议保留的评价指标：

1. physical `AD_B_rel`
2. column-balanced `B_rel`
3. component-balanced `B_rel`
4. selected-frame force closure

不建议继续使用：

1. 单独用当前 `J_norm_mse` 选择最优模型。
2. 让极小 `LE_std` 和窄 q 覆盖决定全部 B loss 权重。

## 11. Next Action

1. 新增一个训练参数开关，例如 `--jacobian-loss-scale physical-balanced`。
2. 默认先保持旧口径，避免破坏历史结果。
3. 用 Gate 08 的 case019 和 case031 极小 overfit 重新跑。
4. 判断标准先看：
   1. `train_LE_rel`
   2. physical `train_AD_B_rel`
   3. column-balanced B error
   4. selected-frame force closure
5. 小数据仍不能过拟合时，再查 `B_macro_qdef` 的刚体投影近似和 AD 路径容量。

当前不释放大训练。
