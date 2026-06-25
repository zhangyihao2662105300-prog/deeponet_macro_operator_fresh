# Gate 17 State Dependent B Baseline Prototype

日期：2026-06-25

角色：Macro16 State Dependent B Baseline Prototype Agent

## 1. 目标

验证 Gate 16 的判断：

```text
当前 point only B_base(point) 不足，需要 state dependent B baseline
```

本轮只做原型，不替换主模型。

约束：

1. 不改 `q48` 顺序
2. 不改 `LE` 顺序
3. 不改 128 点积分规则
4. 不改生产训练器主结构
5. 不启动正式大训练
6. 训练只在 Linux GPU 电脑运行

## 2. 新增脚本

新增：

```text
scripts/train_macro16_state_b_baseline_prototype.py
```

脚本作用：

1. 训练 state dependent B baseline 原型
2. 保存 prototype checkpoint
3. 在 Windows 上做 selected-frame force closure
4. 不接入 `Macro16BoundaryDeepONetWithLE0`
5. 不改变生产模型

## 3. 原型形式

原型使用：

```text
LE = LE0(point) + B_state(point, q_state, geometry) q48_def_hat
```

其中：

```text
B_state = B_static(point) + B_point_delta(point) + U(point) V(q_state, geometry)
```

默认：

```text
q_state detach
```

原因：

1. 先让 baseline 代表局部切线
2. 避免把 `dB_state / dq` 当成额外二阶项混进 AD-B
3. 只验证表达能力，不改变 full tangent 定义

## 4. 数据

compact list：

```text
runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt
```

数据：

1. case031 first6 split
2. train case `3101`
3. val case `3102`
4. train frame `4`
5. val frame `2`
6. 积分点 `128`
7. 输入 `q48_def_hat`
8. 几何 `X16_hat`
9. 标签 `LE_macro`
10. 标签 `B_macro_qdef`

## 5. Linux 训练环境

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

本轮用法：

1. 3 张 GPU 并行跑 3 个小实验
2. 单实验单 GPU
3. 只跑 6 frame 小实验
4. 不做 DDP
5. 不做正式大训练

## 6. Linux 小实验

共同设置：

1. steps `3000`
2. hidden dim `256`
3. depth `4`
4. LE weight `1`
5. B weight `5`
6. lr `1e-3`
7. lr decay `0.9995`

结果：

| run | state rank | detach | best step | train LE | train B | val LE | val B | all LE | all B |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| state_rank4_detach_3000 | 4 | true | 1000 | 0.0203054051 | 0.0225893587 | 0.1048207439 | 0.0756159174 | 0.0882244589 | 0.0474528730 |
| state_rank8_detach_3000 | 8 | true | 1000 | 0.0221973582 | 0.0237174317 | 0.0926411863 | 0.0747412433 | 0.0783043002 | 0.0473557002 |
| state_rank8_nodetach_3000 | 8 | false | 1000 | 0.0221973582 | 0.0237174317 | 0.0926411863 | 0.0747412433 | 0.0783043002 | 0.0473557002 |

结论：

1. train B 已低于 `0.03`
2. all B 约 `0.047`
3. val B 约 `0.075`
4. state rank 8 优于 state rank 4
5. detach 和 no detach 在本小数据上结果相同

## 7. Force Closure

Windows 上做 selected-frame force closure。

原因：

compact 中 `source_compact` 是 Windows 路径，Linux 上不能直接读取源老师文件。

| run | LE rel | B rel | model force rel | teacher force rel |
|---|---:|---:|---:|---:|
| state_rank4_detach_3000 | 0.0882249309 | 0.0474528738 | 0.1079819746 | 0.0127911083 |
| state_rank8_detach_3000 | 0.0783039178 | 0.0473557015 | 0.0919219356 | 0.0127911083 |
| state_rank8_nodetach_3000 | 0.0783039178 | 0.0473557015 | 0.0919219356 | 0.0127911083 |

结果：

```text
Gate 17 prototype force target 0.10: PASS
```

但仍未达到：

```text
final force target 0.02
```

## 8. 与 Gate 16 对比

Gate 16 最好：

```text
force rel = 0.2106850647
B rel = 0.0834366597
```

Gate 17 最好：

```text
force rel = 0.0919219356
B rel = 0.0473557015
```

改善：

1. force rel 降低约 `56%`
2. B rel 降低约 `43%`
3. 证明 state dependent B baseline 是正确方向

## 9. Gate 17 结论

状态：

```text
PASS for prototype target
FAIL for production release
```

已证明：

1. state dependent B baseline 能显著改善 B
2. state dependent B baseline 能把 force rel 压到 `0.10` 以下
3. Gate 16 的原理判断成立

未证明：

1. 主模型已经可用
2. 正式训练可以启动
3. force closure 已达到 `0.02`
4. full tangent 已解决

## 10. 下一步

推荐 Gate 18：

```text
Macro16 State B Migration Plan Agent
```

目标：

1. 规划如何把 state dependent B baseline 迁移进 Macro16 主模型
2. 保持 `q48_def_hat` 输入 48D
3. 保持 `X16_hat`
4. 保持 128 点规则
5. 保持 `B = dLE / d(q48_def_hat)` 的审计口径
6. 明确 detach state baseline 的 tangent 含义
7. 先写迁移计划和测试清单
8. 不直接大训练

建议门槛：

1. 迁移后先复现 Gate 17 的 6 frame prototype 结果
2. force rel 仍需低于 `0.10`
3. 再继续压到 `0.02`
4. 通过后才考虑更大训练
