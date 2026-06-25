# Gate 16 Q Dependent B Diagnosis

日期：2026-06-25

角色：Macro16 Q Dependent B Residual Diagnosis Agent

## 1. 目标

诊断 Gate 15 force closure 失败是否来自 `B_macro_qdef` 的 q 相关性没有被当前模型充分表达。

本轮约束：

1. 不启动正式大训练
2. 不改模型主结构
3. 不改 `q48` 顺序
4. 不改 `LE` 顺序
5. 不改 128 点积分规则
6. full tangent 仍为待处理

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
7. point feature dim `50`
8. 输入 `q48_def_hat` 和 `X16_hat`
9. B 标签 `B_macro_qdef`

## 3. 新增脚本

新增只读诊断脚本：

```text
scripts/diagnose_macro16_q_dependent_b.py
```

脚本功能：

1. 统计同一几何下 `B_macro_qdef` 随 `q48_def_hat` 的变化
2. 计算 point only B baseline 误差
3. 拟合线性 q 修正，判断 q 相关 B 是否可被低阶状态项解释
4. 分解 checkpoint 中 point baseline 和 residual AD-B 的贡献
5. 不训练，不改数据，不改模型

本轮诊断输出：

```text
runs/gate16_q_dependent_b/q_dependent_b_diagnosis_full.json
```

## 4. q 相关 B 诊断

同一几何下共有 6 frame。

关键结果：

| 项目 | rel |
|---|---:|
| 同一几何 constant B mean error | 0.0799931444 |
| point only mean B error | 0.0907799089 |
| 线性 q 修正 train error | 0.0188597020 |
| 线性 q 修正 val error | 0.0699423050 |
| 线性 q 修正 all error | 0.0432747493 |

解释：

1. 同一几何下 B 不是常量
2. point only B prior 有约 `0.08` 到 `0.09` 的不可消除误差
3. 简单线性 q 修正能把 all error 从约 `0.091` 降到约 `0.043`
4. 所以 q dependent B 是真实存在的缺项

## 5. Checkpoint 分解

本轮分解 3 个 checkpoint：

1. Gate 15 residual fine tune
2. Gate 16 residual only B loss
3. Gate 16 residual LE plus B loss

| checkpoint | full AD-B all | full AD-B train | full AD-B val | residual vs needed all | residual norm fraction all |
|---|---:|---:|---:|---:|---:|
| gate15_residual | 0.0911758315 | 0.0651789318 | 0.1280735360 | 0.9343241571 | 0.0325958016 |
| gate16_bonly | 0.0789451668 | 0.0509543992 | 0.1160552546 | 0.8696325689 | 0.0451042604 |
| gate16_leb | 0.0834366597 | 0.0536939514 | 0.1227971184 | 0.9191092969 | 0.0337848135 |

解释：

1. Gate 16 B-only residual 能改善 AD-B
2. 但 residual 实际贡献仍很小
3. residual 对需要补偿的 `B_target - B_baseline` 只学到一小部分
4. 当前结构不是完全不能学，而是学习 q dependent B 的效率不足

## 6. Linux 小实验

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

本轮使用 3 张 GPU 并行跑 3 个小实验。

共同设置：

1. 单实验单 GPU
2. batch size `6`
3. `num_workers = 8`
4. `pin_memory = true`
5. `jacobian_columns = all`
6. `jacobian_columns_per_batch = 48`
7. 冻结 B prior
8. residual scale `1.0`
9. 只跑 6 frame 小实验

结果：

| run | train LE | train AD-B | val LE | val AD-B |
|---|---:|---:|---:|---:|
| residual_only_b1_400 | 0.1071604497 | 0.0512034671 | 0.1794831316 | 0.1166248046 |
| residual_only_b5_400 | 0.1080986354 | 0.0509543926 | 0.1805023302 | 0.1160552302 |
| residual_le_b5_400 | 0.0402486713 | 0.0536939378 | 0.1097874606 | 0.1227970994 |

结论：

1. 加强 B loss 可以把 train AD-B 接近 `0.05`
2. val AD-B 仍停在约 `0.116`
3. 加 LE loss 能改善 LE，但没有改善 B
4. 不能释放正式大训练

## 7. Force Closure

Windows 上做 selected-frame force closure 审计。

原因：

compact 中 `source_compact` 是 Windows 路径，Linux 上不能直接读源老师文件。

| run | LE rel | B rel | model force rel | teacher force rel |
|---|---:|---:|---:|---:|
| residual_only_b1_400 | 0.1610212281 | 0.0793321843 | 0.3460638053 | 0.0127911083 |
| residual_only_b5_400 | 0.1620021088 | 0.0789451666 | 0.3506583861 | 0.0127911083 |
| residual_le_b5_400 | 0.0942947227 | 0.0834366597 | 0.2106850647 | 0.0127911083 |

解释：

1. B-only residual 虽然改善 AD-B，但 LE 变差，force 更差
2. LE+B 小实验 force 约 `0.211`
3. Gate 15 最好 force 约 `0.208`
4. 当前结构和训练方式没有突破 force closure

## 8. 原理判断

当前模型主结构是：

```text
LE = LE0(point) + B_base(point) q + residual(q, geometry, point)
```

其中：

```text
B_base = global_b_norm + point_b_net(point)
```

所以：

1. `B_base` 不看当前 `q48_def_hat`
2. q dependent B 只能通过 `d residual / d q` 表达
3. 但当前 residual 是 DeepONet 低秩 branch trunk 形式
4. 在 6 frame 小实验中，它只学到很小一部分 `B_target - B_base`
5. 这解释了为什么 LE 可以下降，但 AD-B 和 force closure 不够

## 9. Gate 16 结论

状态：

```text
PASS for diagnosis
FAIL for large training release
```

已证明：

1. `B_macro_qdef` 在同一几何下随 `q48_def_hat` 变化
2. point only B prior 存在约 `0.08` 到 `0.09` 的不可消除误差
3. 线性 q 修正可显著降低 B 误差
4. 当前 residual 能改善 B，但不足以通过 force closure

未证明：

1. 当前结构可通过继续加 epoch 解决
2. 训练后 force closure 可达到 `0.02`
3. 可以启动正式大训练

## 10. 下一步

推荐 Gate 17：

```text
Macro16 State Dependent B Baseline Prototype Agent
```

目标：

1. 不改 q48 顺序
2. 不改 LE 顺序
3. 不改 128 点规则
4. 在小实验脚本中测试 state dependent B baseline
5. 形式参考旧 v3 中的 `B_state(point, q_state, geometry)`
6. q_state 默认 detach，避免把 baseline 变成二阶切线
7. 只做 prototype，不替换主模型
8. 先要求 6 frame overfit 达到 train AD-B 小于 `0.03`
9. force rel 先低于 `0.10`

建议原因：

当前证据说明问题不只是 loss 权重，而是 `B_base(point)` 缺少 state dependent 表达。
