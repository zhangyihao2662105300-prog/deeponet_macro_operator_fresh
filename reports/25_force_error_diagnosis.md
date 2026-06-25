# Gate 25 Force Error Diagnosis

日期：2026-06-26

角色：Macro16 Force Error Diagnosis Agent

## 1. 目标

解释为什么 Gate 24 的 `B rel` 下降，但 selected-frame force rel 变差。

本轮只做诊断：

1. 不训练
2. 不改模型
3. 不改 `q48` 顺序
4. 不改 `LE` 顺序
5. 不改 128 点规则

## 2. 新增脚本

```text
scripts/diagnose_macro16_force_error_contributions.py
```

输出：

```text
runs/gate25_force_error_diagnosis.json
```

脚本分解：

1. 积分点 force error 贡献
2. strain component force error 贡献
3. q dof force error 贡献
4. frame force error 贡献
5. force 幅值比例
6. force 方向余弦

## 3. 对比对象

Gate 23 best：

```text
runs/gate23_fixed128_direct_b/rank12/best.pt
```

Gate 24 best：

```text
runs/gate24_gate17_loss_match/rank12_b20_le0.5/best.pt
```

数据：

```text
runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt
```

frames：

```text
case 3101,3102
6 frames
```

## 4. 总体结果

| checkpoint | LE rel | B rel | force rel | teacher force rel |
|---|---:|---:|---:|---:|
| Gate 23 best | 0.1373813802 | 0.0731878231 | 0.1073731865 | 0.0127911083 |
| Gate 24 best | 0.1045596145 | 0.0710433862 | 0.1781247428 | 0.0127911083 |

关键事实：

```text
Gate 24 的 LE rel 和 B rel 都更低
但 force rel 更高
```

差值：

```text
B rel delta = -0.0021444369
LE rel delta = -0.0328217657
force rel delta = +0.0707515562
```

结论：

```text
全局 B rel 不是 force closure 的可靠排序指标
```

## 5. Force 幅值和方向

Gate 23：

```text
force norm ratio mean 1.0065976956
force norm ratio min 0.9824257991
force norm ratio max 1.0233872713
force cos mean 0.9967029873
force cos min 0.9905664231
```

Gate 24：

```text
force norm ratio mean 0.9746743606
force norm ratio min 0.8446932449
force norm ratio max 1.0292717917
force cos mean 0.9917252634
force cos min 0.9763622288
```

判断：

```text
Gate 24 主要不是单纯方向错
而是后几帧 force 幅值明显偏低
```

最差 frame 的幅值比例：

```text
Gate 23 frame 5 ratio 0.9824257991
Gate 24 frame 5 ratio 0.8446932449
```

## 6. Frame 贡献

Gate 23 最大 frame：

```text
frame 5
fraction 0.8640846959
```

Gate 24 最大 frame：

```text
frame 5
fraction 0.8688277408
```

结论：

```text
force error 主要集中在最后一帧
也就是 q 振幅最大的 frame
```

这和之前 case031 诊断一致：

```text
随 q 振幅增大误差变差
```

## 7. Strain Component 贡献

Gate 23 最大贡献：

| component | name | fraction |
|---:|---|---:|
| 4 | G13 | 0.5273913792 |
| 0 | E11 | 0.2955502958 |
| 2 | E33 | 0.0679545179 |

Gate 24 最大贡献：

| component | name | fraction |
|---:|---|---:|
| 0 | E11 | 0.9134486801 |
| 2 | E33 | 0.0385449008 |
| 1 | E22 | 0.0212113652 |

关键变化：

```text
Gate 23 是 G13 主导
Gate 24 变成 E11 绝对主导
```

解释：

```text
加大 B loss 后，整体 B rel 下降
但 E11 对 force 的敏感误差被放大或没有被正确压住
```

## 8. q DOF 贡献

Gate 23 最大 q dof：

| dof | fraction |
|---:|---:|
| 5 | 0.2288229210 |
| 29 | 0.2255314191 |
| 27 | 0.1205605027 |

Gate 24 最大 q dof：

| dof | fraction |
|---:|---:|
| 39 | 0.1346341682 |
| 27 | 0.1155450522 |
| 21 | 0.1027761880 |

结论：

```text
Gate 24 的 force error 不是均匀扩散
而是换了一组主导自由度
```

这说明 loss 权重改变了误差分布，但没有对准 force-sensitive dof。

## 9. 积分点贡献

Gate 23 最大积分点：

| point | fraction |
|---:|---:|
| 123 | 0.0607897608 |
| 8 | 0.0576426919 |
| 99 | 0.0534314842 |

Gate 24 最大积分点：

| point | fraction |
|---:|---:|
| 126 | 0.0406112469 |
| 0 | 0.0339121991 |
| 5 | 0.0335213942 |

结论：

```text
积分点贡献相对分散
不像 component 和 frame 那样单点决定
```

所以优先处理：

```text
frame amplitude
strain component
q dof
```

而不是只按积分点重加权。

## 10. 原理结论

Gate 25 判断：

```text
B loss 的全局 balanced rel 与 force functional 不一致
```

具体表现：

1. Gate 24 降低了全局 B rel
2. Gate 24 同时让 force 幅值在大 q frame 偏低
3. Gate 24 force error 由 E11 主导
4. Gate 24 误差集中在少数 q dof

因此，不应该继续盲目提高 direct B loss。

## 11. 下一步

下一步建议：

```text
Gate 26 Force Weighted B Loss Plan
```

不要直接大训练。

先实现或规划：

1. 用 `|stress| * dV` 给 B loss 加权
2. 或用 `B_error * stress * dV` 直接构造 force-aware B loss
3. 对 E11 增加强约束
4. 对最后一帧或大 q 振幅 frame 增加强约束
5. checkpoint selection 加入 force rel，而不是只看 `val_LE_rel + val_B_rel`

最小下一步：

```text
先写 force weighted B loss 方案
再做小规模 rank12 对照训练
```

## 12. 验证

命令：

```text
PYTHONPATH=src;scripts py -3 scripts\diagnose_macro16_force_error_contributions.py ...
```

结果：

```text
runs/gate25_force_error_diagnosis.json
```

没有训练。

没有改模型。

没有改 128 点规则。
