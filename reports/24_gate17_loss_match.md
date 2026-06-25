# Gate 24 Gate17 Loss Match

日期：2026-06-26

角色：Macro16 Gate17 Loss Match Agent

## 1. 目标

复刻 Gate 17 的训练口径，检查是否能让 fixed128 direct-B 主训练器通过 prototype force gate。

本轮只做小范围 Linux 扫参：

1. `rank = 12`
2. `direct-state-b-loss-weight = 10,20`
3. `le-loss-weight = 0.1,0.5`
4. `residual-scale = 0`
5. `jacobian-loss-weight = 0`
6. 训练后必须跑 selected-frame force audit

不改 `q48` 顺序。

不改 `LE` 顺序。

不改 128 点积分规则。

## 2. 数据

数据：

```text
runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt
```

split：

```text
train case 3101
val case 3102
all frames 6
```

Linux 输出：

```text
runs/gate24_gate17_loss_match
```

## 3. Linux 训练

训练机：

```text
lab-gpu-ts
```

GPU：

```text
3 x NVIDIA RTX 4090
```

任务：

```text
rank12_b10_le0.1
rank12_b20_le0.1
rank12_b20_le0.5
```

训练均完成 3000 epoch。

## 4. 训练指标

| run | checkpoint | epoch | train LE rel | train B rel | val LE rel | val B rel |
|---|---|---:|---:|---:|---:|---:|
| rank12_b10_le0.1 | best | 2300 | 0.0600462241 | 0.0448450161 | 0.1220671431 | 0.0992981426 |
| rank12_b10_le0.1 | latest | 3000 | 0.0579390754 | 0.0440998991 | 0.1253484647 | 0.0981635942 |
| rank12_b20_le0.1 | best | 2000 | 0.0659681694 | 0.0451694991 | 0.1218110427 | 0.0995643663 |
| rank12_b20_le0.1 | latest | 3000 | 0.0591063331 | 0.0435058189 | 0.1267541135 | 0.0973323717 |
| rank12_b20_le0.5 | best | 2200 | 0.0692202942 | 0.0474531949 | 0.1166273411 | 0.1035273102 |
| rank12_b20_le0.5 | latest | 3000 | 0.0635006827 | 0.0449403330 | 0.1228746791 | 0.0995254226 |

对比 Gate 23：

```text
Gate 23 best val B rel 0.1070583245
Gate 24 best val B rel 0.0973323717
```

所以 B 指标确实变好。

## 5. Force Audit

审计参数：

```text
--source-path-map D:\IS-FEM=/home/ydh/IS-FEM
--case-list 3101,3102
--max-frames 6
```

| run | checkpoint | audit LE rel | audit B rel | model force rel | teacher force rel |
|---|---|---:|---:|---:|---:|
| rank12_b10_le0.1 | best | 0.1071378601 | 0.0677191482 | 0.1976783740 | 0.0127911083 |
| rank12_b10_le0.1 | latest | 0.1094032190 | 0.0667752510 | 0.2071816720 | 0.0127911083 |
| rank12_b20_le0.1 | best | 0.1079864589 | 0.0681834340 | 0.1958358811 | 0.0127911083 |
| rank12_b20_le0.1 | latest | 0.1107131293 | 0.0662269453 | 0.2104053318 | 0.0127911083 |
| rank12_b20_le0.5 | best | 0.1045597537 | 0.0710433881 | 0.1781248909 | 0.0127911083 |
| rank12_b20_le0.5 | latest | 0.1083776020 | 0.0677531367 | 0.1979610689 | 0.0127911083 |

最佳 force：

```text
rank12_b20_le0.5 best
model force rel 0.1781248909
```

## 6. 结论

Gate 24 状态：

```text
FAIL
```

原因：

```text
best force rel 0.1781248909 > 0.10
```

这轮不能放行大训练。

## 7. 原理判断

本轮说明：

1. 单纯加大 direct B loss 可以降低 B rel
2. B rel 降低没有带来 force rel 降低
3. force rel 从 Gate 23 最好 `0.1073723868` 反而变差到 `0.1781248909`
4. teacher force 仍然是 `0.0127911083`，审计链路没有坏

所以问题不是：

1. 不是 source compact 坏
2. 不是 selected-frame volume 坏
3. 不是 q48 顺序坏
4. 不是 LE 顺序坏
5. 不是 128 点规则坏

更可能是：

```text
当前 B loss 的全局相对误差指标和 force functional 不一致
```

也就是说，某些积分点，分量，方向的 B 误差对 force 很敏感，但 balanced B loss 没有优先压它们。

## 8. 下一步

下一步任务是 Gate 25：

```text
force-aware error diagnosis
```

建议先不继续盲目训练。

先诊断：

1. 每个积分点对 force error 的贡献
2. 每个 strain component 对 force error 的贡献
3. 每个 q dof 对 force error 的贡献
4. 模型 force 幅值偏低还是方向偏差
5. Gate 17 原型与 Gate 24 checkpoint 在 force integrand 上的差异

如果诊断确认只是 loss weighting 问题，再做：

```text
force-weighted B loss
```

而不是继续提高 direct B loss 权重。

## 9. 验证

Linux：

```text
rank12_b10_le0.1 done
rank12_b20_le0.1 done
rank12_b20_le0.5 done
force audit done
teacher force rel 0.0127911083
```
