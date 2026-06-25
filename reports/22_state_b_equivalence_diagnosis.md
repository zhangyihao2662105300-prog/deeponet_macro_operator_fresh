# Gate 22 State B Equivalence Diagnosis

日期：2026-06-26

角色：Macro16 State B Equivalence Diagnosis Agent

## 1. 目标

解释为什么主训练器 `le0-state-b` 没有复现 Gate 17 原型。

本轮只做诊断：

1. 读取 Gate 17 原型 checkpoint
2. 读取 Gate 21 主训练器 checkpoint
3. 对同一 case031 first6 数据做等价性检查
4. 不启动正式训练
5. 不改模型
6. 不改 `q48` 顺序
7. 不改 `LE` 顺序
8. 不改 128 点积分规则

## 2. 新增脚本

```text
scripts/diagnose_macro16_state_b_equivalence.py
```

输出：

```text
runs/gate22_state_b_equivalence/state_b_equivalence.json
```

脚本检查：

1. checkpoint 标准化统计是否一致
2. `q_norm` 是否能还原 `q48_def_hat`
3. 主训练器显式 `state_b` 与 autograd `AD_B` 是否一致
4. residual 对 AD-B 的贡献是否明显
5. Gate 17 direct B checkpoint 是否仍能在同一数据上复现

## 3. 数据

compact list：

```text
runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt
```

数据：

```text
case031 first6
train case 3101
val case 3102
train frames 4
val frames 2
all frames 6
```

## 4. 对照 checkpoint

Gate 17 原型：

```text
runs/gate17_state_b_prototype/state_rank8_detach_3000/best.pt
```

Gate 21 主训练器：

```text
runs/gate20_state_b/rank4/best.pt
runs/gate20_state_b/rank8/best.pt
runs/gate20_state_b/rank12/best.pt
```

## 5. Gate 17 复现结果

同一数据，同一 split 下，Gate 17 原型仍然复现：

| split | LE rel | direct B rel |
|---|---:|---:|
| train | 0.0221970526 | 0.0237174312 |
| val | 0.0926408039 | 0.0747412453 |
| all | 0.0783039178 | 0.0473557015 |

说明：

1. Gate 17 checkpoint 没坏
2. 数据没坏
3. direct B 路径确实能达到 `all B rel ≈ 0.047`

## 6. Gate 21 主训练器诊断

主训练器 best checkpoint 在同一数据上的结果：

| run | all LE rel | all AD_B rel | all explicit state_B rel | AD_B vs explicit state_B rel |
|---|---:|---:|---:|---:|
| rank4 | 0.3783770762 | 0.5563093833 | 0.8428741243 | 0.6345682580 |
| rank8 | 0.4342824207 | 0.5421441986 | 0.8358013294 | 0.6614620176 |
| rank12 | 0.4011280039 | 0.6060751144 | 0.6993502701 | 0.4032707988 |

关键观察：

1. 主训练器的 `AD_B` 不是显式 `state_b`
2. `AD_B` 明显优于显式 `state_b`
3. 说明 residual 的导数正在参与 B 拟合
4. 这和 Gate 17 的 direct B 训练口径不同

## 7. q 坐标检查

`q_norm` 还原到 `q48_def_hat` 的误差：

| split | rel |
|---|---:|
| train | 约 `1.10e-08` |
| val | 约 `2.88e-09` |
| all | 约 `6.77e-09` |

结论：

```text
q 坐标没有错
```

`q_norm - q0_norm` 和直接 `q_norm` 的线性项不同，这是预期现象。

原因：

```text
q_norm - q0_norm = q48_def_hat / q_std
```

所以它是主训练器在标准化 LE 坐标中表达物理 `B q` 的正确形式，不是失败原因。

## 8. 标准化检查

主训练器 checkpoint 的以下统计量与当前数据完全一致：

```text
branch_mean
branch_std
point_mean
point_std
le_mean
le_std
```

结论：

```text
标准化统计没有错
```

## 9. 结构差异

Gate 17 原型使用固定 128 点表：

```text
static_b[128,6,48]
point_b_net(point)
state_b_point_net(point) -> [128,6,48,rank]
state_coeff_net(q,geom) -> [rank]
```

主训练器 `le0-state-b` 使用动态点 Query 结构：

```text
global_b_norm[6,48]
point_b_net(point)
state_b_point_net(point) -> [point,6,48,rank]
state_b_coeff_net(branch) -> [rank]
residual(branch,point)
```

最关键差异：

```text
Gate 17 有 static_b[128,6,48]
主训练器只有 global_b_norm[6,48]
```

这会把 128 个积分点的静态 B 差异压成一个全局均值，点位差异必须重新从 `point_b_net` 学。

## 10. 失败原因

Gate 22 判断：

```text
主训练器和 Gate 17 原型不等价
```

主要原因：

1. Gate 17 直接监督物理 `B_state`
2. 主训练器监督的是 `AD_B = dLE / d(q48_def_hat)`
3. 主训练器输出里还有 residual 路径
4. residual 的导数参与了 AD-B
5. 主训练器的显式 `state_b` 没有被直接约束到 Gate 17 的 B 水平
6. 主训练器没有 Gate 17 的固定 128 点 `static_b[128,6,48]` 表

不是以下问题：

1. 不是数据坏
2. 不是 q 坐标错
3. 不是标准化错
4. 不是 `q48` 顺序错
5. 不是 `LE` 顺序错
6. 不是 128 点规则错

## 11. 是否需要 direct B warmstart

结论：

```text
需要
```

但 warmstart 不应只加载 `global_b_norm + point_b_net`。

原因：

1. Gate 17 的成功依赖固定 128 点 direct B prior
2. 当前主训练器的动态点 `global_b_norm` 会丢掉固定点表
3. 仅靠 residual AD-B 去补 B，会导致显式 state B 和 AD-B 分离

更合理的下一步是：

```text
实现 fixed 128 direct-B state baseline 变体
```

最小形式：

```text
LE_norm = LE0_norm(point)
        + B_state_norm(point, q, geom) @ (q_norm - q0_norm)
```

并允许：

```text
static_b_norm[128,6,48]
residual_scale = 0
direct B loss
```

先复现 Gate 17，再决定是否重新接 residual。

## 12. Gate 22 结论

状态：

```text
PASS as diagnosis
FAIL as training release
```

允许下一步：

1. 设计 fixed 128 direct-B state baseline 变体
2. 或把 Gate 17 原型迁移成可审计主模型变体
3. 只做 case031 first6 小复现
4. 训练必须在 Linux 机器上运行

不允许：

1. 不允许大训练
2. 不允许声明 `le0-state-b` 已通过
3. 不允许声明 solver ready
4. 不允许把 Gate 21 失败继续当调参问题

## 13. 下一步任务

建议 Gate 23：

```text
Macro16 Fixed 128 State B Baseline Agent
```

目标：

1. 新增独立模型变体，不替换默认模型
2. 保留 `q48_def_hat` 48D
3. 保留 `X16_hat`
4. 保留 128 点规则
5. 加入 fixed 128 `static_b_norm[128,6,48]`
6. 先设置 `residual_scale = 0`
7. 先用 direct B loss 或严格等价 AD-B loss
8. 在 Linux 上复现 case031 first6
9. 门槛先回到 Gate 17：`all B rel < 0.05`，force rel `< 0.10`
