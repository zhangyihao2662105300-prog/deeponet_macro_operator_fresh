# Gate 35 Branch X16 Std Floor Training

日期：2026-06-26

## 1. 目标

把 Gate 34 的 normalization replay 方案接入训练。

本轮只做小训练和 force audit。

不改 `q48` 顺序。

不改 `LE` 顺序。

不改 128 点规则。

不扩大数据规模。

## 2. 实现

新增训练参数：

```text
--branch-x16-std-floor
```

默认：

```text
0.0
```

默认保持旧行为。

本轮显式使用：

```text
--branch-x16-std-floor 0.05
```

实现约束：

```text
只改 X16_hat 48 个几何列的 branch std 下限
q48_def_hat std 不变
L_ref std 不变
```

checkpoint 记录：

```text
branch_x16_std_floor = 0.05
x16_columns_changed = 47
x16_std_min_before = 9.99999993922529e-09
x16_std_min_after = 0.05000000074505806
q_std_unchanged = true
l_ref_std_unchanged = true
```

## 3. 本地验证

```text
py_compile PASS
smoke_test PASS
82 passed
```

## 4. Linux 训练

Linux 机器：

```text
lab-gpu-ts
```

GPU：

```text
NVIDIA RTX 4090
```

训练目录：

```text
/home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo
```

输出：

```text
runs/gate35_branch_x16_floor/train/f01_floor005_lr8e5_complete
```

设置：

```text
model le0-fixed128-state-b
state_b_rank 12
direct_state_b_loss_weight 5.0
force_residual_loss_weight 0.1
jacobian_loss_weight 0.0
epochs 3000
batch_size 16
val_cases 70,71,72,73
branch_x16_std_floor 0.05
```

## 5. 训练指标

best checkpoint：

```text
best_epoch = 1
train_LE_rel = 0.90101546319115
train_AD_B_rel = 0.26497121664292267
val_LE_rel = 10.261582054091793
val_AD_B_rel = 1.9032424399904309
```

latest checkpoint：

```text
latest_epoch = 3000
train_LE_rel = 0.14234780728786875
train_AD_B_rel = 0.19578871277372156
val_LE_rel = 40.31566920876957
val_AD_B_rel = 3.617301198104712
```

训练集下降。

风机壳验证集仍然崩溃。

## 6. Selected Frame Force Audit

overall best：

```text
model force rel = 0.9909391049353954
teacher force rel = 0.008568855675169518
LE rel = 0.9092818434200872
B rel = 0.5264460489269744
```

overall latest：

```text
model force rel = 3.8008025209932095
teacher force rel = 0.008568855675169518
LE rel = 0.3521295258471789
B rel = 0.8790089189758244
```

wind shell best：

| case | model force rel | teacher force rel | LE rel | B rel |
|---|---:|---:|---:|---:|
| 70 | 46.16988370481955 | 2.560913513133724e-05 | 12.918033639169613 | 1.5651662126614236 |
| 71 | 39.30080008717659 | 2.6548040880994235e-05 | 10.791869994520543 | 1.406008945397978 |
| 72 | 50.59679788424623 | 2.4948734705572523e-05 | 13.106678514462175 | 1.7884986585442948 |
| 73 | 76.32909191653128 | 2.4123080275059657e-05 | 7.6798618247822015 | 2.760930974791687 |

wind shell latest：

| case | model force rel | teacher force rel | LE rel | B rel |
|---|---:|---:|---:|---:|
| 70 | 18.491706475229567 | 2.560913513133724e-05 | 13.299531965826018 | 1.3698962758251236 |
| 71 | 33.94768275046621 | 2.6548040880994235e-05 | 18.99095037630248 | 1.9773758540159088 |
| 72 | 31.12248583748845 | 2.4948734705572523e-05 | 19.253762872358113 | 2.121452216131517 |
| 73 | 2028.1803702824302 | 2.4123080275059657e-05 | 55.71553304461297 | 7.564029990330541 |

## 7. Gate 35 结论

```text
implementation PASS
training FAIL
force audit FAIL
```

`X16_hat std floor 0.05` 修复了输入标准化爆炸，但没有修复 wind shell 泛化。

这说明 Gate 31 的失败不只是标准化尺度问题。

当前模型仍无法从非 wind shell train cases 外推到 wind shell validation cases。

## 8. 原理判断

当前证据指向：

```text
wind shell 几何族本身需要进入训练分布
或者需要改变几何表达和 split 策略
```

不支持继续只调训练轮数。

不支持继续扩大同一 split 的训练。

不支持把 latest checkpoint 当可用模型。

## 9. 下一步

进入 Gate 36。

建议任务：

1. 做 wind shell in split 小训练。
2. 训练集加入一部分 wind shell case。
3. 保留另一个 wind shell case 做验证。
4. 仍使用 `--branch-x16-std-floor 0.05`。
5. 训练后跑 selected-frame force audit。

建议 split：

```text
train wind shell 70,71,72
val wind shell 73
```

对照目标：

```text
判断失败来自 wind shell 完全 OOD
还是模型即使见过 wind shell 也学不住
```
