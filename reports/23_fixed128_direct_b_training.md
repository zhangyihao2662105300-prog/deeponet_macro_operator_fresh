# Gate 23 Fixed128 Direct B Training

日期：2026-06-26

角色：Macro16 Fixed128 Direct B Training Agent

## 1. 目标

验证一个更接近 Gate 17 原型的主训练器变体：

1. 使用 `le0-fixed128-state-b`
2. 保留固定 128 点 `static_b_norm[128,6,48]`
3. 关闭 residual AD 路径
4. 用 direct physical B loss 直接约束显式 `_state_b_norm`
5. Linux 三张 4090 并行扫 `rank 4,8,12`
6. 训练后必须跑 selected-frame force closure

本轮不改 `q48` 顺序，不改 `LE` 顺序，不改 128 点积分规则。

## 2. 代码改动

提交：

```text
e10d883 Add Macro16 fixed128 direct B training
d4fce5b Map trained force audit source paths
```

新增或扩展：

```text
Macro16BoundaryDeepONetWithLE0Fixed128StateB
--model-style le0-fixed128-state-b
--direct-state-b-loss-weight
--source-path-map
```

`--direct-state-b-loss-weight` 的目标是：

```text
显式 _state_b_norm -> 物理 B_macro_qdef
```

当 `--jacobian-loss-weight 0.0` 时，训练不再为 B loss 计算 AD Jacobian。

## 3. 数据

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

Linux 源数据路径映射：

```text
D:\IS-FEM=/home/ydh/IS-FEM
```

## 4. Linux 训练命令

训练机：

```text
lab-gpu-ts
```

GPU：

```text
3 x NVIDIA RTX 4090
```

输出目录：

```text
runs/gate23_fixed128_direct_b
```

启动脚本：

```text
runs/gate23_fixed128_direct_b/run_gate23_direct_b.sh
```

核心参数：

```text
--model-style le0-fixed128-state-b
--residual-scale 0.0
--detach-state-b
--jacobian-loss-weight 0.0
--direct-state-b-loss-weight 5.0
--jacobian-columns all
--jacobian-columns-per-batch 48
--epochs 3000
--batch-size 6
--hidden-dim 256
--branch-depth 4
--trunk-depth 4
--val-cases 3102
```

## 5. 训练结果

| rank | checkpoint | epoch | train LE rel | train B rel | val LE rel | val B rel |
|---:|---|---:|---:|---:|---:|---:|
| 4 | best/latest | 3000 | 0.0534461782 | 0.0487380434 | 0.1423037185 | 0.1078751764 |
| 8 | best | 300 | 0.0830776039 | 0.0573022791 | 0.1517913442 | 0.1224093674 |
| 8 | latest | 3000 | 0.0512258766 | 0.0494159189 | 0.1678382296 | 0.1106910416 |
| 12 | best/latest | 3000 | 0.0462821349 | 0.0483478774 | 0.1616963410 | 0.1070583245 |

观察：

1. train B 已经降到约 `0.048` 到 `0.049`
2. val B 仍约 `0.107`
3. 说明 fixed128 direct B 能拟合训练帧，但泛化到 val frame 仍不够

## 6. Force Audit

审计命令使用：

```text
scripts/audit_macro16_trained_force_closure.py
--source-path-map D:\IS-FEM=/home/ydh/IS-FEM
```

| rank | checkpoint | audit LE rel | audit B rel | model force rel | teacher force rel |
|---:|---|---:|---:|---:|---:|
| 4 | best/latest | 0.1223956866 | 0.0732312962 | 0.1190965849 | 0.0127911083 |
| 8 | best | 0.1347266173 | 0.0843158579 | 0.1521049184 | 0.0127911083 |
| 8 | latest | 0.1429351591 | 0.0757327513 | 0.1330706113 | 0.0127911083 |
| 12 | best/latest | 0.1373809397 | 0.0731878236 | 0.1073723868 | 0.0127911083 |

当前最佳：

```text
rank12 best/latest
model force rel 0.1073723868
teacher force rel 0.0127911083
```

## 7. Gate 23 结论

Gate 23 当前状态：

```text
FAIL
```

原因：

```text
force rel 0.1073723868 > 0.10
```

不能释放大训练。

不能声称模型力闭合通过。

## 8. 原理判断

本轮修正证明：

1. fixed `static_b_norm[128,6,48]` 方向是对的
2. direct physical B loss 明显优于前面主训练器 AD 混合路径
3. teacher force 仍然闭合，所以不是数据或审计坏掉
4. `q48` 顺序，`LE` 顺序，128 点规则没有发现问题

但仍存在差距：

1. train B 可到 `0.048`
2. val B 卡在约 `0.107`
3. force rel 最好约 `0.107`
4. 与 Gate 17 原型 `all B rel 0.047` 和 force rel `0.0919` 仍未完全复现

更可能原因：

```text
当前 loss 权重和 checkpoint selection 仍没有完全复刻 Gate 17
```

次要可能：

```text
LE loss 与 direct B loss 互相牵制
```

## 9. 下一步

下一步任务是 Gate 24：

```text
复刻 Gate 17 训练口径
```

建议只做小范围扫参：

1. `direct-state-b-loss-weight 10,20`
2. `le-loss-weight 0.1,0.5,1.0`
3. `rank 12` 优先
4. checkpoint selection 增加 force audit 或 all B rel 优先
5. 目标仍是 force rel `< 0.10`

如果 Gate 24 仍不能过：

```text
回到 Gate 17 原型代码，逐项对齐 optimizer，loss scale，初始化，selection
```

## 10. 验证

本地验证：

```text
PYTHONPATH=src;scripts py -3 -m pytest tests\smoke_test.py -q
77 passed
```

Linux 验证：

```text
rank 4,8,12 训练完成
best/latest force audit 完成
teacher force rel 0.0127911083
```
