# Gate 36 Wind Shell In Split Training

日期：2026-06-26

## 1. 目标

验证 Gate 35 之后的判断。

问题是：

```text
wind shell 失败是否只是完全 OOD split 导致
```

本轮把 wind shell case `70,71,72` 放入训练分布，保留 `73` 做验证。

约束：

1. 不改 `q48` 顺序
2. 不改 `LE` 顺序
3. 不改 128 点规则
4. 不改模型结构
5. 不训练全量正式模型
6. 用 selected-frame force closure 判断结果

## 2. Linux 训练

机器：

```text
lab-gpu-ts
```

运行目录：

```text
/home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo
```

输出目录：

```text
runs/gate36_wind_shell_in_split/train/f01_floor005_train70_72_val73
```

数据：

```text
runs/gate31_force_residual_16case_qdef/enriched_16case_compact_list.txt
16 cases
160 frames
```

关键参数：

```text
model-style le0-fixed128-state-b
state-b-rank 12
direct-state-b-loss-weight 5.0
force-residual-loss-weight 0.1
branch-x16-std-floor 0.05
val-cases 73
val-fraction 0.0
epochs 3000
batch-size 16
num-workers 4
pin-memory true
CUDA_VISIBLE_DEVICES 0
```

std floor 元数据：

```text
branch_x16_std_floor = 0.05
x16_columns_changed = 48
x16_std_min_before = 9.99999993922529e-09
x16_std_min_after = 0.05000000074505806
q_std_unchanged = true
l_ref_std_unchanged = true
```

## 3. 训练指标

best checkpoint：

```text
epoch = 1
train_LE_rel = 0.6879238017441861
train_AD_B_rel = 0.30117798914208155
val_LE_rel = 6.216842124075486
val_AD_B_rel = 2.4815666456200183
```

latest checkpoint：

```text
epoch = 3000
train_LE_rel = 0.12984803131693504
train_AD_B_rel = 0.19196307691605177
val_LE_rel = 17.32762860801707
val_AD_B_rel = 5.824046980294429
```

训练集下降。

验证 case `73` 明显恶化。

训练 loss 不能作为通过依据。

## 4. Selected Frame Force Audit

审计脚本：

```text
scripts/audit_macro16_trained_force_closure.py
```

审计口径：

```text
volume_mode = selected-frame
force_coordinate = qdef
model_visible_q = q48_def_hat
model_visible_B = B_macro_qdef
point_count = 128
```

overall 16 case：

| checkpoint | model force rel | teacher force rel | LE rel | B rel |
|---|---:|---:|---:|---:|
| best | 0.853668281648636 | 0.008568855675169518 | 0.6954600516821234 | 0.49496781226483694 |
| latest | 0.472340736565529 | 0.008568855675169518 | 0.16220337789387235 | 0.663929225872648 |

wind shell 分 case，best：

| case | model force rel | teacher force rel | LE rel | B rel |
|---|---:|---:|---:|---:|
| 70 | 31.805762831914723 | 2.560913513133724e-05 | 10.258515921914695 | 1.2504548022311257 |
| 71 | 26.5876217805224 | 2.6548040880994235e-05 | 8.545347546149083 | 1.1166516949189167 |
| 72 | 34.67033336145459 | 2.4948734705572523e-05 | 10.378752743069128 | 1.4474653536547188 |
| 73 | 53.40411713064857 | 2.4123080275059657e-05 | 6.216842131294531 | 2.4688479238150327 |

wind shell 分 case，latest：

| case | model force rel | teacher force rel | LE rel | B rel |
|---|---:|---:|---:|---:|
| 70 | 1.150269541449442 | 2.560913513133724e-05 | 1.17323677292635 | 0.8244457474591985 |
| 71 | 0.8096015173533533 | 2.6548040880994235e-05 | 0.7679622305316057 | 0.6450275469142943 |
| 72 | 1.4293494020412825 | 2.4948734705572523e-05 | 1.1488929227329217 | 0.9463748978909772 |
| 73 | 251.13256074370318 | 2.4123080275059657e-05 | 17.327624869089206 | 5.931577140789179 |

## 5. 结论

Gate 36：

```text
training completed
force audit FAIL
```

老师 force closure 仍然通过。

模型 force closure 失败。

`case70,71,72` 进入训练分布后，latest 在这些 case 上比 best 明显改善，但仍远高于 `0.02` 门槛。

保留的 wind shell case `73` 在 latest 上严重恶化。

因此不能说 Gate 31 的 wind shell 失败只是完全 OOD split 导致。

更准确的判断是：

```text
当前 le0-fixed128-state-b 加 force residual 小训练
即使见过部分 wind shell
仍不能稳定学习 wind shell 几何族
```

## 6. 不允许的结论

不能释放大训练。

不能把 latest checkpoint 当可用模型。

不能用训练集下降替代 force audit。

不能改 `q48`、`LE` 或 128 点规则解释失败。

不能说老师数据坏。

## 7. 下一步

进入 Gate 37。

建议任务：

1. 诊断 case `73` 和 `70,71,72` 的几何差异。
2. 检查 `X16_hat`、`point_features`、`L_ref`、曲率、厚度、坐标列分布。
3. 对比 train wind shell 和 val wind shell 的 branch/trunk 激活。
4. 检查 force error 是否集中在近零参考力导致的相对误差放大。
5. 决定下一步是 geometry split 修正，几何特征增强，还是模型容量问题。

暂时不要继续扩大训练轮数。
