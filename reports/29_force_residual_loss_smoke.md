# Gate 29 Force Residual Loss Implementation

日期：2026-06-26

## 1. 目标

实现第一版 Macro16 force residual loss。

本轮只做代码准备和本地 smoke。

不训练大模型。

不改模型结构。

不改 `q48` 顺序。

不改 `LE` 顺序。

不改 128 点积分规则。

## 2. 实现内容

新增训练数据字段：

```text
elastic_D
```

训练 dataset 现在返回：

```text
branch
point
LE
J
integration_weight_hat
elastic_D
```

新增 loss：

```text
F_pred = sum B_pred * D * LE_pred * dV_hat
F_teacher = sum B_true * D * LE_true * dV_hat
loss = mean(((F_pred - F_teacher) / F_scale)^2)
```

第一版 `dV` 使用：

```text
integration_weight_hat
```

注意：

这只是训练引导。

最终 force audit 仍使用 selected-frame volume。

## 3. elastic_D 来源

loader 读取顺序：

```text
elastic_D
material_D
identity_fallback
```

如果启用：

```text
--force-residual-loss-weight > 0
```

则不允许 `identity_fallback`。

也就是说，真正使用 force residual loss 时，compact 必须带真实 `elastic_D` 或 `material_D`。

## 4. 新增脚本

```text
scripts/enrich_macro16_compact_elastic_d.py
```

用途：

从 Macro16 compact 的 `source_compact` 找到 source compact。

从 source compact 复制 `elastic_D` 或 `material_D`。

写入 Macro16 compact。

不改 `q48`。

不改 `LE`。

不改 `B`。

不改 128 点规则。

## 5. 新增参数

```text
--force-residual-loss-weight
--force-residual-volume-mode hat
--force-residual-scale-mode component-rms
--force-residual-scale-floor-rel
--force-residual-scale-floor-abs
```

默认关闭：

```text
--force-residual-loss-weight 0.0
```

## 6. 本地验证

已运行：

```text
PYTHONPATH=src;scripts py -3 -m py_compile src\macro_deeponet\train_macro16_boundary_sobolev.py scripts\enrich_macro16_compact_elastic_d.py
```

结果：

```text
PASS
```

已运行：

```text
PYTHONPATH=src;scripts py -3 -m pytest tests\smoke_test.py -q -k "elastic_d or force_residual or fixed128_state_b"
```

结果：

```text
5 passed
```

新增测试覆盖：

1. loader 可读取 compact 自带 `elastic_D`
2. force residual loss 在 teacher force 一致时为 0
3. force residual loss 在 LE 扰动后大于 0
4. fixed128 state B 训练 smoke 可启用 force residual loss
5. enrichment 脚本可从 source compact 复制 `elastic_D`

## 7. Linux 训练

Linux 机器：

```text
lab-gpu-ts
```

远端目录：

```text
/home/ydh/IS-FEM/gate29_force_residual/repo
```

数据：

```text
runs/gate29_force_residual_loss/enriched_case031_first6_compact_list.txt
```

elastic_D 富集报告：

```text
runs/gate29_force_residual_loss/enrich_elastic_d_summary.json
```

富集结果：

```text
1 compact
6 frames
elastic_D shape 6 x 6 x 6
source key elastic_D
```

三卡训练：

```text
f01 GPU 0 force weight 0.1 LE weight 1.0
f05 GPU 1 force weight 0.5 LE weight 1.0
f10 GPU 2 force weight 1.0 LE weight 0.5
```

共同设置：

```text
model le0-fixed128-state-b
state_b_rank 12
direct_state_b_loss_weight 5.0
jacobian_loss_weight 0.0
epochs 3000
batch_size 6
num_workers 4
pin_memory true
```

## 8. 训练结果

| run | best epoch | train LE rel | train AD_B rel | val LE rel | val AD_B rel |
|---|---:|---:|---:|---:|---:|
| f01 | 3000 | 0.0462323164 | 0.0490282800 | 0.1590740954 | 0.1092934815 |
| f05 | 200 | 0.0800320987 | 0.0630040397 | 0.1746738984 | 0.1276345614 |
| f10 | 100 | 0.0882521778 | 0.0659928900 | 0.1559539688 | 0.1359003293 |

## 9. Selected Frame Force Audit

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
| f01 | 0.0882157802 | 0.0127911083 | 0.1352254904 | 0.0746432692 | PASS |
| f05 | 0.1412967153 | 0.0127911083 | 0.1523413314 | 0.0892691982 | FAIL |
| f10 | 0.1939571690 | 0.0127911083 | 0.1389705778 | 0.0963234912 | FAIL |

Gate 23 best：

```text
force rel 0.1073731865
```

Gate 29 best：

```text
force rel 0.0882157802
```

## 10. 当前状态

代码实现：

```text
PASS
```

Linux 训练：

```text
PASS
```

Gate 29 结果：

```text
PASS prototype threshold
```

通过原因：

```text
f01 force rel 0.0882157802 < 0.10
f01 better than Gate 23 best 0.1073731865
```

限制：

```text
只是在 case031 first6 小样本上通过
不是大规模训练释放
不是 full tangent closure
```

## 11. 下一步

进入 Gate 30。

建议任务：

1. 诊断为什么 f01 通过但 f05 和 f10 变差
2. 固定 f01 作为当前最佳超参数
3. 加入 B prior warmstart 对照
4. 在 case019 first6 和 case031 first6 都跑 force residual 小训练
5. 如果两 case 都过 `0.10`，再扩到 16 case 小训练

Gate 30 暂定通过标准：

```text
case019 first6 force rel < 0.10
case031 first6 force rel < 0.10
两者 teacher force rel 仍正常
```
