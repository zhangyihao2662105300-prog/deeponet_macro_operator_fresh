# Gate 30 Force Residual Warmstart Generality

日期：2026-06-26

## 1. 目标

验证 Gate 29 的 force residual loss 是否只对 `case031 first6` 有效。

本轮只做小样本泛化和 warmstart 对照。

不做大训练。

不改模型结构。

不改 `q48` 顺序。

不改 `LE` 顺序。

不改 128 点规则。

## 2. 数据

Linux 机器：

```text
lab-gpu-ts
```

远端目录：

```text
/home/ydh/IS-FEM/gate30_force_residual_warmstart/repo
```

数据：

```text
case019 first6
case031 first6
```

每个 case：

```text
1 compact
6 frames
train 4 frames
val 2 frames
128 integration points
```

elastic_D 富集：

```text
runs/gate30_force_residual_warmstart/enrich_elastic_d_summary.json
```

富集结果：

```text
case019 elastic_D shape 6 x 6 x 6
case031 elastic_D shape 6 x 6 x 6
source key elastic_D
```

## 3. 训练设置

主设置来自 Gate 29 best `f01`：

```text
model le0-fixed128-state-b
state_b_rank 12
direct_state_b_loss_weight 5.0
force_residual_loss_weight 0.1
le_loss_weight 1.0
jacobian_loss_weight 0.0
epochs 3000
batch_size 6
num_workers 4
pin_memory true
```

额外对照：

```text
force_residual_loss_weight 0.05
B prior warmstart
```

## 4. B Prior 预训练

命令类型：

```text
python3 scripts/fit_macro16_point_b_prior.py
```

结果：

| case | train B rel | val B rel |
|---|---:|---:|
| case019 | 0.0044910896 | 0.0122487627 |
| case031 | 0.0624246872 | 0.1299495311 |

结论：

```text
case019 的 point B prior 拟合很好
case031 的 point B prior 拟合一般
```

但后续训练显示，B prior warmstart 不一定改善 force closure。

## 5. 训练结果

| run | best epoch | val LE rel | val AD_B rel |
|---|---:|---:|---:|
| case019 f01 no prior | 100 | 0.0185985676 | 0.0119859135 |
| case019 f01 prior | 3000 | 0.1542114100 | 0.0311125475 |
| case019 f005 no prior | 100 | 0.0166732966 | 0.0119737007 |
| case031 f01 no prior | 3000 | 0.1590740954 | 0.1092934815 |
| case031 f01 prior | 100 | 0.1151441349 | 0.1411496080 |
| case031 f005 no prior | 200 | 0.1482879802 | 0.1226960276 |

## 6. Selected Frame Force Audit

命令类型：

```text
python3 scripts/audit_macro16_trained_force_closure.py
```

体积口径：

```text
selected-frame
```

source path map：

```text
D:\IS-FEM=/home/ydh/IS-FEM
```

结果：

| run | model force rel | teacher force rel | LE rel | B rel | result |
|---|---:|---:|---:|---:|---|
| case019 f01 no prior | 0.0175615121 | 0.0016164669 | 0.0172713393 | 0.0078743755 | PASS |
| case019 f01 prior | 0.1047005215 | 0.0016164669 | 0.1535161204 | 0.0271243737 | FAIL |
| case019 f005 no prior | 0.0177242518 | 0.0016164669 | 0.0156857745 | 0.0078738001 | PASS |
| case031 f01 no prior | 0.0882157802 | 0.0127911083 | 0.1352254904 | 0.0746432692 | PASS |
| case031 f01 prior | 0.2043860133 | 0.0127911083 | 0.1053777026 | 0.1061656943 | FAIL |
| case031 f005 no prior | 0.1417496103 | 0.0127911083 | 0.1320591161 | 0.0863616108 | FAIL |

## 7. 结论

Gate 30 小样本泛化：

```text
PASS
```

通过证据：

```text
case019 f01 no prior force rel 0.0175615121 < 0.10
case031 f01 no prior force rel 0.0882157802 < 0.10
```

这说明 Gate 29 的改善不是只对 `case031 first6` 偶然成立。

## 8. 原理诊断

1. `f01 no prior` 是当前最稳设置。

2. B prior warmstart 没有改善 force closure。

3. case019 的 B prior 本身很好，但 warmstart 后 force 反而从 `0.01756` 变到 `0.10470`。

4. 这说明当前 warmstart 加载到 `global_b_norm + point_b_net` 后，和后续 `state_b`、LE loss、force residual loss 的联合优化存在不协调。

5. case031 prior 后 LE rel 变好，但 B rel 和 force rel 变差，说明只降低 LE 或 point prior B 不能保证 force functional 闭合。

6. 当前训练准入更应依赖 selected-frame force audit，而不是单看 B prior 或 val LE。

## 9. 当前最佳设置

建议固定：

```text
model le0-fixed128-state-b
state_b_rank 12
direct_state_b_loss_weight 5.0
force_residual_loss_weight 0.1
le_loss_weight 1.0
no B prior warmstart
```

## 10. 下一步

进入 Gate 31。

目标：

从两个 first6 case 扩展到更多 case。

建议任务：

1. 使用当前最佳设置
2. 不使用 B prior warmstart
3. 训练 16 case 小集合
4. case split 必须按 case 隔离
5. 训练后跑 selected-frame force audit
6. 报告每个 case 的 force rel

Gate 31 暂定通过标准：

```text
overall force rel < 0.10
每个 case force rel 尽量 < 0.10
teacher force rel 保持正常
```

如果 Gate 31 失败：

```text
按最坏 case 诊断 force residual 项
不要直接扩大训练规模
```
