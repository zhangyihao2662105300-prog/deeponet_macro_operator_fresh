# Gate 21 State B Point Q Rank Rerun

日期：2026-06-26

角色：Macro16 State B Point Q Rank Rerun Agent

## 1. 目标

验证 Gate 20 修正后的 `point_q_rank` state B 形式，能否在 Linux 训练机复现 Gate 17 小样本原型结果。

本轮只做：

1. Linux 小样本训练
2. 三组 rank 扫描
3. Windows selected-frame force closure 审计
4. 记录结论

本轮不做：

1. 不正式大训练
2. 不改 `q48` 顺序
3. 不改 `LE` 顺序
4. 不改 128 点积分规则
5. 不声明 solver ready

## 2. 数据和环境

训练机：

```text
lab-gpu-ts
```

GPU：

```text
3 x NVIDIA GeForce RTX 4090
```

训练目录：

```text
/home/ydh/IS-FEM/gate20_state_b_point_q_rank/repo
```

本地回收目录：

```text
runs/gate20_state_b
```

训练数据：

```text
runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt
```

实际数据：

```text
runs/gate08_training_overfit_sanity/case031_first6_overfit_case_split_macro16_source128_teacher.npz
```

训练帧：

```text
4
```

验证帧：

```text
2
```

## 3. 训练设置

共同参数：

```text
--model-style le0-state-b
--state-b-kind point_q_rank
--detach-state-b
--epochs 3000
--batch-size 6
--eval-batch-size 6
--num-workers 8
--pin-memory
--hidden-dim 256
--branch-depth 4
--trunk-depth 4
--le-loss-weight 1.0
--jacobian-loss-weight 5.0
--rigid-loss-weight 0.0
--jacobian-loss-scale physical-balanced
--jacobian-columns all
--jacobian-columns-per-batch 48
--eval-columns all
--val-cases 3102
--lr 1.0e-3
--lr-decay 0.9995
--cuda
```

rank 扫描：

```text
4
8
12
```

## 4. 通过标准

Gate 20 rerun 目标：

```text
train B rel < 0.03
all B rel < 0.05
model force rel < 0.10
teacher force rel < 0.02
```

## 5. 训练结果

| run | best epoch | train LE | train B | val LE | val B |
|---|---:|---:|---:|---:|---:|
| rank4 | 3000 | 0.0449918070 | 0.5528287132 | 0.4522338397 | 0.5631727346 |
| rank8 | 3000 | 0.0442601492 | 0.5371192334 | 0.5193480190 | 0.5520089874 |
| rank12 | 1700 | 0.0628008674 | 0.6038300931 | 0.4786688203 | 0.6105186769 |

latest 3000：

| run | train LE | train B | val LE | val B |
|---|---:|---:|---:|---:|
| rank4 | 0.0449918070 | 0.5528287132 | 0.4522338397 | 0.5631727346 |
| rank8 | 0.0442601492 | 0.5371192334 | 0.5193480190 | 0.5520089874 |
| rank12 | 0.0611081097 | 0.5363588348 | 0.5499551227 | 0.5544649119 |

## 6. Force Audit

Windows selected-frame force audit 使用 best checkpoint。

| run | audit LE rel | audit B rel | model force rel | teacher force rel |
|---|---:|---:|---:|---:|
| rank4 | 0.3783769350 | 0.5563093839 | 0.5944244775 | 0.0127911083 |
| rank8 | 0.4342825403 | 0.5421442045 | 0.6198139264 | 0.0127911083 |
| rank12 | 0.4011279631 | 0.6060751151 | 0.6095472149 | 0.0127911083 |

审计输出：

```text
runs/gate20_state_b/rank4/force_audit_best.json
runs/gate20_state_b/rank8/force_audit_best.json
runs/gate20_state_b/rank12/force_audit_best.json
```

## 7. 结论

Gate 21 结果：

```text
FAIL
```

原因：

1. teacher force 仍然闭合，`0.0127911083 < 0.02`
2. 数据和 selected-frame force 审计链路没有坏
3. `point_q_rank` 训练后 B rel 仍约 `0.54` 到 `0.61`
4. 模型 force rel 仍约 `0.59` 到 `0.62`
5. 没有复现 Gate 17 原型的 `B rel ≈ 0.0474` 和 `force rel ≈ 0.0919`

所以：

```text
不允许进入大训练
不允许声明 state B production route 通过
不允许声明 trained Macro16 solver ready
```

## 8. 当前判断

本轮失败不是：

1. q48 顺序错误
2. LE 顺序错误
3. 128 点规则错误
4. teacher 数据错误
5. selected-frame volume 审计错误

更可能的问题：

1. 主训练器的 `LE0 + B_state dq + residual` 组合仍没有复现 Gate 17 原型的直接 B 拟合口径
2. `B = dLE / d(q48_def_hat)` 的 autograd 约束和 Gate 17 direct B baseline 仍有训练口径差异
3. `detach_state_b` 后的 state B 只通过线性项表达一阶导，可能仍被 residual 或 LE 目标干扰
4. `q_norm - q0_norm` 与 Gate 17 直接使用 `q48_def_hat` 的坐标口径仍需核对

## 9. 下一步

下一步不是大训练。

建议 Gate 22：

1. 做训练器和 Gate 17 原型的逐项等价诊断
2. 固定同一 batch，同一 checkpoint 初始化，同一 q 坐标
3. 比较 direct B loss 和 autograd B loss
4. 先冻结 LE residual，只训练 state B 路径
5. 检查 `q_norm - q0_norm` 是否应改为直接 `q_norm`

只有 Gate 22 找到差异并重新把 force rel 压到 `0.10` 以下，才允许继续扩大样本。
