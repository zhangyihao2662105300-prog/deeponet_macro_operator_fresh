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

## 7. 当前状态

代码实现：

```text
PASS
```

Linux 正式训练：

```text
待运行
```

Gate 29 结果：

```text
IMPLEMENTATION READY
```

不是训练通过。

## 8. 下一步

在 Linux 机器上：

1. 同步本提交
2. enrich case031 first6 compact 的 `elastic_D`
3. 跑三组 force residual 小训练
4. 对 best checkpoint 跑 selected-frame force audit
5. 和 Gate 23 best `0.1073731865` 比较

通过标准：

```text
force rel < 0.10
```

并且最好优于 Gate 23 best。
