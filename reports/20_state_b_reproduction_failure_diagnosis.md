# Gate 20 State B Reproduction Failure Diagnosis

日期：2026-06-25

角色：Macro16 State B Reproduction Diagnosis Agent

## 1. 目标

复现 Gate 17 的 case031 first6 state B 小实验。

本轮做了：

1. 在 Linux 训练机运行三组 `le0-state-b`
2. 每组使用一张 RTX 4090
3. rank 为 4，8，12
4. 训练 3000 epoch
5. 回到 Windows 做 selected-frame force audit

## 2. 训练环境

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
/home/ydh/IS-FEM/gate19_state_b_main_model/repo
```

训练数据：

```text
runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt
```

## 3. 第一次启动问题

第一次启动继承了训练器默认：

```text
rigid_loss_weight = 0.1
```

现象：

1. 只有 rank4 先跑起来
2. loss 被 rigid loss 主导
3. 与 Gate 17 原型训练口径不一致

处理：

1. 停止错误任务
2. 清理错误输出
3. 用脚本重新启动三组训练
4. 显式加入 `--rigid-loss-weight 0.0`

## 4. 正式 Gate 19 三组结果

共同参数：

```text
epochs 3000
hidden_dim 256
branch_depth 4
trunk_depth 4
LE weight 1
B weight 5
rigid loss 0
lr 1e-3
lr decay 0.9995
```

结果：

| run | best epoch | train LE | train B | val LE | val B |
|---|---:|---:|---:|---:|---:|
| rank4 | 1200 | 0.0736663962 | 0.6350954600 | 0.5728182058 | 0.6413728732 |
| rank8 | 1500 | 0.0847779351 | 0.5953465281 | 0.5170658104 | 0.6109269269 |
| rank12 | 200 | 0.2236332579 | 0.7923653959 | 0.4518980687 | 0.7957530428 |

latest 3000：

| run | train LE | train B | val LE | val B |
|---|---:|---:|---:|---:|
| rank4 | 0.0440077015 | 0.5346682061 | 0.8377563029 | 0.5562479166 |
| rank8 | 0.0517478497 | 0.5217131966 | 0.8814619198 | 0.5532234232 |
| rank12 | 0.0532510902 | 0.5275497725 | 0.8782632943 | 0.5400591728 |

结论：

```text
Gate 19 numeric reproduction FAIL
```

## 5. Force audit

Windows selected-frame force audit 使用 best checkpoint。

| run | LE rel | B rel | force rel | teacher force rel |
|---|---:|---:|---:|---:|
| rank4 | 0.4799560371 | 0.6372016456 | 0.5978557317 | 0.0127911083 |
| rank8 | 0.4342076407 | 0.6006019315 | 0.6024681885 | 0.0127911083 |
| rank12 | 0.3968571169 | 0.7934999103 | 0.7512728299 | 0.0127911083 |

结论：

1. teacher force 仍然闭合
2. 数据没有坏
3. checkpoint 加载没有坏
4. 迁移模型没有复现 Gate 17 原型表达能力

## 6. 原理诊断

Gate 17 原型的 state B 形式是：

```text
B_state = B_static(point) + B_point_delta(point) + U(point, strain, q, rank) C(q_state, geometry, rank)
```

也就是每个：

```text
point
strain component
q column
```

都有自己的 rank basis。

Gate 19 初版迁移模型实际写成：

```text
B_state = B_static(point) + B_point_delta(point) + U(point, strain, rank) V(q_state, geometry, rank, q)
```

这个形式把 q 列和 point basis 分离了，表达能力弱很多。

这解释了：

1. LE 能下降
2. B 长期停在约 0.52 到 0.64
3. force rel 停在约 0.60
4. 无法复现 Gate 17 的 B rel 约 0.047 和 force rel 约 0.092

## 7. 修正

已修正 `Macro16BoundaryDeepONetWithLE0StateB` 默认 state B kind：

```text
point_q_rank
```

修正后默认形式为：

```text
B_delta = basis(point, strain, q, rank) coeff(q_state, geometry, rank)
```

保留旧形式：

```text
low_rank_uv
```

作为兼容选项，但不作为默认。

## 8. 本地验证

已运行：

```text
PYTHONPATH=src;scripts py -3 -m pytest tests/smoke_test.py -q -k "state_b or macro16_training or b_prior_checkpoint"
```

结果：

```text
6 passed
```

## 9. 当前状态

Gate 19 初版数值复现：

```text
FAIL
```

修正后的 Gate 20 代码状态：

```text
ready for Linux rerun
```

仍然不允许：

1. 不正式大训练
2. 不声明 force closure 已通过
3. 不声明 solver ready
4. 不改 `q48` 顺序
5. 不改 `LE` 顺序
6. 不改 128 点规则

## 10. 下一步

下一步运行 Gate 20 rerun：

1. 同步修正后的代码到 Linux
2. 用 `point_q_rank` 默认形式重跑 rank4，rank8，rank12
3. 如果 B rel 接近 Gate 17 原型，再做 force audit
4. 如果仍失败，继续比较归一化和 q 坐标差异
