# Gate 18 State Dependent B Migration Plan

日期：2026-06-25

角色：Macro16 State B Migration Plan Agent

## 1. 目标

把 Gate 17 验证有效的 state dependent B baseline，规划为可迁移到 Macro16 主训练器的安全路线。

本轮只写迁移计划，不实现主模型，不训练，不改 128 点积分规则。

保持不变：

1. `q48_def_hat` 仍是 48D 输入
2. `X16_hat` 仍是几何输入
3. `LE_macro` 仍是输出标签
4. `B = dLE / d(q48_def_hat)` 仍是审计口径
5. 128 个标准父坐标积分点不变
6. 不暴露 `X_macro`
7. 不暴露细网格内部节点

## 2. Gate 17 证据

Gate 17 原型：

```text
LE = LE0(point) + B_state(point, q_state, geometry) q48_def_hat
```

```text
B_state = B_static(point) + B_point_delta(point) + U(point) V(q_state, geometry)
```

最好结果：

1. train LE rel `0.0221973582`
2. train B rel `0.0237174317`
3. val LE rel `0.0926411863`
4. val B rel `0.0747412433`
5. all LE rel `0.0783043002`
6. all B rel `0.0473557002`
7. model force rel `0.0919219356`
8. teacher force rel `0.0127911083`

判断：

1. state dependent B 方向成立
2. point only B prior 不足
3. force rel 已低于阶段目标 `0.10`
4. force rel 仍高于最终目标 `0.02`
5. 不能释放正式大训练

## 3. 当前主模型阻塞点

当前默认模型：

```text
Macro16BoundaryDeepONetWithLE0
```

当前主结构：

```text
LE_norm = LE0_norm(point)
        + B_base_norm(point) (q_norm - q0_norm)
        + residual_offset_norm(q_norm, point)
```

问题：

1. `_linear_b_norm(point_norm)` 只显式依赖 point
2. `B_macro_qdef` 已被 Gate 16 证明随 `q48_def_hat` 变化
3. 现有 residual 可以通过导数间接表达 q dependent B，但小 overfit 不稳定
4. Gate 17 表明，把 q dependent B 放进 baseline 更直接

## 4. 迁移设计

建议新增模型变体，不直接替换默认模型：

```text
Macro16BoundaryDeepONetWithLE0StateB
```

建议形式：

```text
LE_norm = LE0_norm(point)
        + B_state_norm(point, q_state, geometry) (q_norm - q0_norm)
        + residual_offset_norm(q_norm, point)
```

其中：

```text
B_state_norm = global_b_norm
             + baseline_scale point_b_net(point)
             + state_b_scale U(point) V(q_state, geometry)
```

输入拆分：

1. `q_norm` 来自 `q48_def_hat`
2. `geometry` 来自 branch 中的 `X16_hat` 和 `L_ref` 相关标准化输入
3. `point` 来自固定 128 积分点特征

## 5. detach state baseline 含义

默认：

```text
q_state = detach(q_norm)
geometry_state = detach(geometry_norm)
```

含义：

1. `B_state` 表示当前状态附近的局部割线或局部切线 baseline
2. AD-B 主要看到 `B_state @ q` 对 q 的一阶导数
3. 暂时不把 `dB_state / dq` 项混进训练目标
4. 这不等于 full Abaqus tangent
5. full tangent 仍是待处理问题

需要保留开关：

```text
detach_state_b = true or false
```

默认使用 `true`。

## 6. 需要新增的模型参数

建议新增：

1. `state_b_enabled`
2. `state_b_rank`
3. `state_b_scale`
4. `detach_state_b`
5. `state_b_kind`
6. `state_b_point_net`
7. `state_b_coeff_net`

推荐默认：

```text
state_b_enabled = false
state_b_rank = 8
state_b_scale = 1.0
detach_state_b = true
state_b_kind = low_rank_uv
```

## 7. 训练脚本改动点

文件：

```text
src/macro_deeponet/train_macro16_boundary_sobolev.py
```

建议改动：

1. `--model-style` 增加 `le0-state-b`
2. 增加 `--state-b-rank`
3. 增加 `--state-b-scale`
4. 增加 `--detach-state-b`
5. checkpoint 写入 state B 参数
6. `best.pt` 和 `latest.pt` 保存新模型 style
7. force audit 加载 checkpoint 时识别新模型

## 8. warmstart 策略

已有 point B prior warmstart 继续有效：

1. `global_b_norm` 继续加载
2. `point_b_net` 继续加载
3. state B 新增网络零初始化或小初始化
4. 先冻结或低学习率保护 point prior
5. 再训练 state B low rank correction

优先策略：

```text
global_b_norm lr scale = 0.1
point_b_net lr scale = 0.1
state_b_net lr scale = 1.0
```

## 9. 测试清单

Gate 19 实施时至少新增测试：

1. 新模型 forward 输出形状为 `[B,128,6]`
2. 新模型 AD-B 输出形状为 `[B,128,6,48]`
3. `detach_state_b=true` 时 AD-B 不包含 state net 对 q 的二阶路径
4. `detach_state_b=false` 时计算图可反传
5. checkpoint 保存和加载新模型
6. 旧 `le0` checkpoint 路径不受影响
7. point B prior warmstart 可加载到新模型
8. `q_dim` 仍为 48
9. `point_count` 仍为 128
10. smoke test 通过

## 10. Gate 19 通过门槛

Gate 19 只允许小实验，不允许正式大训练。

通过标准：

1. 测试通过
2. case031 first6 可复现 Gate 17 水平
3. train B rel 小于 `0.03`
4. all B rel 小于 `0.05`
5. selected-frame force rel 小于 `0.10`
6. teacher force rel 保持低于 `0.02`
7. 不改 `q48` 顺序
8. 不改 `LE` 顺序
9. 不改 128 点规则

失败处理：

1. 如果 force rel 大于 `0.10`，不进入大训练
2. 如果 AD-B 形状或口径不一致，回滚实现
3. 如果旧模型加载路径破坏，回滚实现

## 11. 下一步任务

下一步 Gate 19：

```text
Macro16 State B Main Model Implementation Agent
```

只做：

1. 新增 gated state B 模型变体
2. 接入训练脚本选项
3. 接入 checkpoint 加载
4. 接入 force audit 加载
5. 跑 case031 first6 小实验
6. 输出 `reports/19_state_b_main_model_smoke.md`

不做：

1. 不正式大训练
2. 不改默认生产路线
3. 不改 128 点规则
4. 不声称 solver ready
