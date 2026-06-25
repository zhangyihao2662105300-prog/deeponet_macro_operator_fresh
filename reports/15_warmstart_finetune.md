# Gate 15 Warmstart Fine Tune Audit

日期：2026-06-25

角色：Macro16 Warmstart Fine Tune Agent

## 1. 目标

在 Gate 14 已验证的 B prior warmstart 基础上，做小规模 Linux GPU fine tune。

本轮目标：

1. 比较 frozen warmstart，unfreeze warmstart，residual fine tune
2. 继续只训练 `LE` 和 `B`
3. `B` 仍由 `LE` 对 `q48_def_hat` 自动微分得到
4. 训练后做 selected-frame force closure
5. 判断是否允许进入正式大训练

本轮禁止：

1. 不做正式大训练
2. 不改模型主结构
3. 不改 `q48` 顺序
4. 不改 `LE` 顺序
5. 不改 128 点积分规则
6. 不声明 full tangent 已解决

## 2. 数据

compact list：

```text
runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt
```

数据范围：

1. case031 first6 split
2. train case `3101`
3. val case `3102`
4. train frame `4`
5. val frame `2`
6. 积分点 `128`
7. 输入 `q48_def_hat` 和 `X16_hat`
8. 输出 `LE`
9. B 标签 `B_macro_qdef`

## 3. Linux 训练环境

训练机：

```text
lab-gpu-ts
```

硬件：

```text
3 x NVIDIA GeForce RTX 4090
24 GB each
CPU 64
```

本轮使用：

1. 3 张 GPU 并行跑 3 个小实验
2. 单实验单 GPU
3. `num_workers = 8`
4. `pin_memory = true`
5. `jacobian_columns = all`
6. `jacobian_columns_per_batch = 48`
7. batch size `6`

说明：

数据只有 6 frame，多 GPU DDP 对单个实验不划算。本轮用 3 张卡并行跑不同设置，能更充分利用 Linux 训练机。

## 4. Point B Prior

本轮重新拟合 point B prior：

```text
train B rel = 0.0624246872
val B rel = 0.1299495436
```

checkpoint：

```text
runs/gate15_warmstart_finetune/point_b_prior.pt
```

结论：

B prior 可正常加载到 `global_b_norm` 和 `point_b_net`。

## 5. 第一轮 Fine Tune

| run | residual | B prior | B weight | train LE | train AD_B | val LE | val AD_B |
|---|---:|---|---:|---:|---:|---:|---:|
| frozen_le_b_200 | 0.0 | frozen | 0.1 | 0.0756105403 | 0.0624246872 | 0.1415187567 | 0.1299495274 |
| unfreeze_low_lr_200 | 0.0 | unfreeze | 0.1 | 0.0505151643 | 0.0970587892 | 0.1510118869 | 0.1500159696 |
| unfreeze_bweight1_200 | 0.0 | unfreeze | 1.0 | 0.0520859350 | 0.0696503380 | 0.1526884803 | 0.1333612506 |

第一轮结论：

1. frozen warmstart 保住了 B prior
2. unfreeze 能改善 train LE
3. unfreeze 没有改善 val AD_B
4. 只解冻 B prior 不足以释放训练

## 6. 第二轮 Residual Fine Tune

第二轮打开 residual 路径，检查 q-dependent residual 是否能补上 B。

| run | residual | B prior | B weight | train LE | train AD_B | val LE | val AD_B |
|---|---:|---|---:|---:|---:|---:|---:|
| residual_frozen_b1_300 | 1.0 | frozen | 1.0 | 0.0494495792 | 0.0589181534 | 0.1168038515 | 0.1284064042 |
| residual_unfreeze_b1_300 | 1.0 | unfreeze | 1.0 | 0.0526382817 | 0.0651789221 | 0.1000574069 | 0.1280735274 |
| residual_unfreeze_b2_300 | 1.0 | unfreeze | 2.0 | 0.0542125434 | 0.0614029108 | 0.1048228425 | 0.1256288921 |

第二轮结论：

1. residual 能改善 val LE
2. residual 对 val AD_B 只有小幅改善
3. 最好 val AD_B 是 `0.1256288921`
4. 仍未达到继续大训练的准入水平

## 7. Force Closure

Linux checkpoint 拉回 Windows 后审计。

原因：

compact 中 `source_compact` 是 Windows 路径，Linux 上无法直接读取老师源文件。

审计命令口径：

```text
selected-frame physical volume
case-list 3101,3102
max-frames 6
batch-size 6
```

| run | LE rel | B qdef rel | model force rel | teacher force rel |
|---|---:|---:|---:|---:|
| frozen_le_b_200 | 0.1252528037 | 0.0907799105 | 0.2521781464 | 0.0127911083 |
| residual_frozen_b1_300 | 0.1012358845 | 0.0884560225 | 0.2118769211 | 0.0127911083 |
| residual_unfreeze_b1_300 | 0.0884239257 | 0.0911758313 | 0.2083507750 | 0.0127911083 |
| residual_unfreeze_b2_300 | 0.0924675761 | 0.0882467229 | 0.2087366919 | 0.0127911083 |
| unfreeze_bweight1_200 | 0.1306720586 | 0.0957945380 | 0.2602905598 | 0.0127911083 |
| unfreeze_low_lr_200 | 0.1291152226 | 0.1174568295 | 0.2579721949 | 0.0127911083 |

最好 force：

```text
residual_unfreeze_b1_300
model force rel = 0.2083507750
```

对比 Gate 14：

```text
Gate 14 best force rel = 0.2560688360
Gate 15 best force rel = 0.2083507750
```

结论：

1. Gate 15 有改善
2. 改善幅度不够
3. teacher 同口径仍为 `0.0127911083`
4. 模型 force closure 仍未接近 `0.02`

## 8. 原理判断

当前失败不是数据契约坏，也不是 autograd B 路径断。

更可能的问题：

1. 当前 point B prior 主要表达 point feature 到 B 的平均映射
2. `B` 实际还随 `q48_def_hat` 变化
3. residual 路径能改善 LE，但没有充分学到 q-dependent B
4. 6 frame 极小数据无法证明泛化
5. force 装配对 LE 和 B 同时敏感，B rel 约 `0.088` 仍会放大到 force rel 约 `0.208`

因此，Gate 15 不能释放正式大训练。

## 9. Gate 15 结论

状态：

```text
FAIL for large training release
PASS for diagnostic fine tune completion
```

已证明：

1. warmstart fine tune 链路能跑通
2. Linux GPU 训练可用
3. residual 能改善 LE
4. force rel 从约 `0.256` 改善到约 `0.208`

未证明：

1. 模型可用于求解器
2. trained force closure 达到 `0.02`
3. 当前结构能稳定学习 q-dependent B
4. 可以启动正式大训练

## 10. 下一步

推荐 Gate 16：

```text
Macro16 Q-Dependent B Residual Diagnosis Agent
```

目标：

1. 诊断 `B_macro_qdef` 随 `q48_def_hat` 的变化幅度
2. 分离 point-only B 误差和 q-dependent B 误差
3. 检查 residual 路径对 `dLE/dq` 的有效秩
4. 判断是训练设置问题，还是当前结构表达 q-dependent B 的能力不足
5. 只做诊断和小实验，不启动正式大训练

建议通过标准：

1. train AD_B rel 小于 `0.05`
2. val AD_B rel 明显低于 `0.10`
3. force rel 先低于 `0.10`

在 Gate 16 之前，仍不允许正式大训练。
