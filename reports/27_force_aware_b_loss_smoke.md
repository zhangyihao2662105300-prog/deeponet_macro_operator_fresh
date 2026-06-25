# Gate 27 Force Aware B Loss Smoke

日期：2026-06-26

角色：Macro16 Force Aware B Loss Smoke Agent

## 1. 目标

实现并验证 Gate 26 提出的第一版 force aware B loss。

本轮允许：

1. 改训练 loss
2. 跑 Linux 小规模训练
3. 跑 force audit

本轮禁止：

1. 改模型结构
2. 改 `q48` 顺序
3. 改 `LE` 顺序
4. 改 128 点规则
5. 放行大训练

## 2. 代码改动

提交：

```text
0f2c564 Add Macro16 force aware B loss
```

新增参数：

```text
--force-aware-b-loss-weight
--force-aware-b-weight-mode strain-volume
--force-aware-b-weight-min
--force-aware-b-weight-max
```

默认：

```text
force-aware-b-loss-weight = 0.0
```

所以默认训练不受影响。

## 3. Loss 公式

第一版使用 strain-volume proxy：

```text
w = abs(LE_true) * integration_weight_hat
w = clamp(w / mean(w), 0.1, 10.0)
loss = mean(w * ((B_pred - B_true) / B_scale)^2)
```

其中：

```text
B_pred = explicit _state_b_norm * LE_std / q_std
B_true = B_macro_qdef
```

## 4. 本地验证

测试：

```text
PYTHONPATH=src;scripts py -3 -m pytest tests\smoke_test.py -q
```

结果：

```text
78 passed
```

覆盖：

1. force aware B loss 公式
2. fixed128 训练循环
3. summary 记录 `force_aware_b_loss_weight`

## 5. Linux 训练

训练机：

```text
lab-gpu-ts
```

GPU：

```text
3 x NVIDIA RTX 4090
```

输出：

```text
runs/gate27_force_aware_b_loss
```

三组实验：

| run | direct B weight | force aware weight | LE weight |
|---|---:|---:|---:|
| fa1_le1 | 5 | 1 | 1 |
| fa3_le1 | 5 | 3 | 1 |
| fa1_le0.5 | 5 | 1 | 0.5 |

共同参数：

```text
model-style le0-fixed128-state-b
state-b-rank 12
residual-scale 0
jacobian-loss-weight 0
epochs 3000
val-cases 3102
```

## 6. 训练指标

| run | checkpoint | epoch | train LE rel | train B rel | val LE rel | val B rel |
|---|---|---:|---:|---:|---:|---:|
| fa1_le1 | best | 200 | 0.0856343361 | 0.0585398356 | 0.1496580192 | 0.1202659622 |
| fa1_le1 | latest | 3000 | 0.0475354282 | 0.0488131974 | 0.2132529705 | 0.1081663896 |
| fa3_le1 | best | 2800 | 0.0546144706 | 0.0477760683 | 0.1703769834 | 0.1029996316 |
| fa3_le1 | latest | 3000 | 0.0536596012 | 0.0475823174 | 0.1733975082 | 0.1028764811 |
| fa1_le0.5 | best/latest | 3000 | 0.0564665895 | 0.0466110271 | 0.1429534255 | 0.1016758578 |

## 7. Force Audit

| run | checkpoint | audit LE rel | audit B rel | model force rel | teacher force rel |
|---|---|---:|---:|---:|---:|
| fa1_le1 | best | 0.1335424011 | 0.0847933544 | 0.1484515496 | 0.0127911083 |
| fa1_le1 | latest | 0.1799552154 | 0.0741731945 | 0.1149584190 | 0.0127911083 |
| fa3_le1 | best | 0.1453876601 | 0.0706979217 | 0.1251610874 | 0.0127911083 |
| fa3_le1 | latest | 0.1477495302 | 0.0705192216 | 0.1258299097 | 0.0127911083 |
| fa1_le0.5 | best/latest | 0.1233308717 | 0.0695030333 | 0.1246353032 | 0.0127911083 |

最佳 force：

```text
fa1_le1 latest
force rel 0.1149584190
```

Gate 23 best：

```text
force rel 0.1073731865
```

结论：

```text
Gate 27 未超过 Gate 23
```

## 8. Gate 27 结论

状态：

```text
FAIL
```

原因：

```text
best force rel 0.1149584190 > 0.10
best force rel 也差于 Gate 23 best 0.1073731865
```

所以不能放行大训练。

## 9. 原理判断

第一版 force aware B loss 没有成功。

原因判断：

```text
abs(LE_true) * integration_weight_hat 只是 strain-volume proxy
它没有真正包含 material D，也没有直接看 B_error * stress * dV 对 force 的方向贡献
```

实际表现：

1. B rel 有改善
2. force rel 没有改善
3. teacher force 仍稳定闭合
4. 问题仍是 loss 与 force functional 不一致

## 10. 下一步

下一步不建议继续调这个 proxy 权重。

建议 Gate 28：

```text
Force residual loss implementation plan or diagnosis
```

两条可选路线：

1. 实现真正的 force residual loss
2. 先写诊断，比较 `strain-volume` 权重和 Gate 25 实际 force error component 是否一致

更推荐：

```text
先实现 force residual loss
```

公式：

```text
F_pred = sum(B_pred * D * LE_pred * dV_hat)
F_teacher = sum(B_true * D * LE_true * dV_hat)
loss_force = mean(((F_pred - F_teacher) / F_scale)^2)
```

第一版如果没有 `D`：

```text
先用 identity D 做诊断，不作为最终训练方案
```

正式版需要把 `elastic_D` 纳入训练数据。

## 11. 当前状态

Gate 27 证明：

```text
默认关闭的 force aware B loss 代码可用
但 strain-volume proxy 不足以改善 force closure
```

大训练继续 blocked。
