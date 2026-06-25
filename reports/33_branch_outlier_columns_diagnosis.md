# Gate 33 Branch Outlier Columns Diagnosis

日期：2026-06-26

## 1. 目标

解释 Gate 32 发现的 wind shell branch 标准化离群。

本任务只做诊断。

不训练网络。

不改模型。

不改 `q48` 顺序。

不改 `LE` 顺序。

不改 128 点规则。

## 2. 使用脚本

```text
scripts/diagnose_macro16_branch_outlier_columns.py
```

Linux 运行目录：

```text
/home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo
```

命令：

```bash
python3 scripts/diagnose_macro16_branch_outlier_columns.py \
  --compact-list runs/gate31_force_residual_16case_qdef/enriched_16case_compact_list.txt \
  --checkpoint runs/gate31_force_residual_16case_qdef/train/f01_lr8e5/best.pt \
  --out runs/gate31_force_residual_16case_qdef/audits/gate33_branch_outlier_columns.json
```

## 3. Branch 定义

当前训练 branch：

```text
q48_def_hat 48列
X16_hat 48列
L_ref 1列
总计 97列
```

标准化来自 Gate 31 `f01_lr8e5` checkpoint。

## 4. 关键结果

最大离群组：

```text
X16_hat_z
```

最大离群列：

```text
X01_z
```

最大离群值：

```text
abs z = 1525076.1154592873
```

对应样本：

```text
case073
frame 130
raw X01_z = -0.015250761061906815
mean = 0.0
std = 9.99999993922529e-09
```

同类异常列：

| case | column | name | group | raw | std | abs z |
|---|---:|---|---|---:|---:|---:|
| 073 | 53 | X01_z | X16_hat_z | -0.0152507611 | 1.0e-08 | 1525076.115 |
| 073 | 65 | X05_z | X16_hat_z | -0.0152507611 | 1.0e-08 | 1525076.115 |
| 073 | 77 | X09_z | X16_hat_z | 0.0046457732 | 1.0e-08 | 464577.324 |
| 073 | 89 | X13_z | X16_hat_z | 0.0046457732 | 1.0e-08 | 464577.324 |
| 073 | 81 | X11_x | X16_hat_x | 0.4985102117 | 0.0089504495 | 57.018 |
| 073 | 93 | X15_x | X16_hat_x | -0.4985102117 | 0.0089504495 | 54.375 |

对照：

```text
q48_def_hat 最大 abs z = 5.623005387233668
```

所以 Gate 32 的百万级 branch 离群不是位移输入导致，而是几何输入标准化导致。

## 5. 逐 Case 最大离群

| case | max group | max column | abs z | branch norm max |
|---|---|---|---:|---:|
| 019 | q48_def_hat | q39 | 1.784 | 6.226 |
| 025 | q48_def_hat | q21 | 2.295 | 7.142 |
| 031 | q48_def_hat | q43 | 5.623 | 33.043 |
| 041 | q48_def_hat | q09 | 1.457 | 5.161 |
| 043 | q48_def_hat | q03 | 3.177 | 10.384 |
| 044 | q48_def_hat | q09 | 1.498 | 5.167 |
| 045 | q48_def_hat | q27 | 2.666 | 8.873 |
| 046 | q48_def_hat | q09 | 1.315 | 4.200 |
| 049 | q48_def_hat | q39 | 1.940 | 6.101 |
| 050 | q48_def_hat | q47 | 3.133 | 9.892 |
| 060 | X16_hat_z | X12_z | 3.276 | 10.689 |
| 061 | X16_hat_y | X04_y | 3.317 | 19.579 |
| 070 | X16_hat_x | X04_x | 0.797 | 3.099 |
| 071 | X16_hat_y | X04_y | 4.241 | 18.175 |
| 072 | X16_hat_x | X11_x | 1.027 | 3.388 |
| 073 | X16_hat_z | X01_z | 1525076.115 | 2254634.899 |

## 6. 原理解释

Gate 31 的 train cases 主要是规则或畸变补丁。

某些 `X16_hat_z` 列在训练集中几乎恒定。

checkpoint 里这些列的标准差被压到约：

```text
1.0e-08
```

当 wind shell `case073` 的同一列出现真实非零曲率或坐标旋转时，标准化结果变成：

```text
z = raw / tiny_std
```

因此得到百万级 branch 输入。

这会把网络推到训练分布外，解释 Gate 31 中 wind shell force closure 失败。

## 7. 结论

Gate 33 结论：

```text
diagnosis PASS
主要离群来自 X16_hat 几何列
不是 q48_def_hat
不是 LE
不是 B 标签
不是 128 点规则
```

当前不能继续扩大训练。

## 8. 下一步

进入 Gate 34。

目标：

修正 branch normalization 方案。

候选方向：

1. 对 `X16_hat` 使用固定物理尺度，不用 train split 的 per-column std。
2. 对 geometry branch std 设置下限，例如 `0.05` 或 `0.1`。
3. 分组标准化 `q48_def_hat`，`X16_hat`，`L_ref`，避免几何恒定列 std 过小。
4. 先做 normalization-only replay，不训练，检查 wind branch max 是否回到合理范围。
5. 再做小训练，不直接扩大数据规模。

建议 Gate 34 先写方案和只读 normalization replay 脚本。
