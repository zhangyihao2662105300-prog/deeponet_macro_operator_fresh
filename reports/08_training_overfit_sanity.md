# Gate 08 Training Overfit Sanity

Date: 2026-06-25

Role: Macro16 Overfit Sanity Agent

## 1. Task Goal

确认 `Macro16BoundaryDeepONetWithLE0` 能否在极小数据上把 `LE` 和 autograd `B` 拟合下来。

本轮不改模型结构，不改 `q48` 顺序，不改 `LE` 顺序，不改 128 点积分规则。

## 2. Data Used

极小 overfit compact：

1. `runs\gate08_training_overfit_sanity\case019_first6_overfit_case_split_macro16_source128_teacher.npz`
2. `runs\gate08_training_overfit_sanity\case031_first6_overfit_case_split_macro16_source128_teacher.npz`

数据设置：

1. 每个 compact 取 6 个 frame
2. case019 split 为 train `1901`，val `1902`
3. case031 split 为 train `3101`，val `3102`
4. 输入为 `q48_def_hat` 和 `X16_hat`
5. 标签为 `LE_macro` 和 `B_macro_qdef`
6. 积分点为标准 source128

## 3. Commands Used

代表性命令：

```powershell
py -3 -m macro_deeponet.train_macro16_boundary_sobolev --compact-list runs\gate08_training_overfit_sanity\overfit_case031_first6_compact_list.txt --out-dir runs\gate08_training_overfit_sanity\train_overfit_case031_first6_v1 --model-style le0 --epochs 300 --batch-size 4 --eval-batch-size 2 --basis-dim 64 --hidden-dim 128 --branch-depth 3 --trunk-depth 3 --jacobian-columns all --jacobian-columns-per-batch 48 --eval-columns all --le-loss-weight 1.0 --jacobian-loss-weight 0.01 --rigid-loss-weight 0.0 --lr 2.0e-4 --lr-decay 0.999 --val-cases 3102 --eval-every 50 --cuda
```

```powershell
py -3 -m macro_deeponet.train_macro16_boundary_sobolev --compact-list runs\gate08_training_overfit_sanity\overfit_case031_first6_compact_list.txt --out-dir runs\gate08_training_overfit_sanity\train_overfit_case031_first6_bonly --model-style le0 --epochs 300 --batch-size 4 --eval-batch-size 2 --basis-dim 64 --hidden-dim 128 --branch-depth 3 --trunk-depth 3 --jacobian-columns all --jacobian-columns-per-batch 48 --eval-columns all --le-loss-weight 0.0 --jacobian-loss-weight 1.0 --rigid-loss-weight 0.0 --lr 1.0e-3 --lr-decay 0.999 --val-cases 3102 --eval-every 50 --cuda
```

```powershell
py -3 scripts\diagnose_macro16_overfit_adb.py --compact-list runs\gate08_training_overfit_sanity\overfit_case031_first6_compact_list.txt --out runs\gate08_training_overfit_sanity\diagnose_case031_first6_adb_grad.json --val-cases 3102 --batch-size 4 --basis-dim 64 --hidden-dim 128 --branch-depth 3 --trunk-depth 3 --cuda
```

```powershell
py -3 scripts\audit_macro16_trained_force_closure.py --checkpoint runs\gate08_training_overfit_sanity\train_overfit_case031_first6_v1\best.pt --compact-list runs\gate08_training_overfit_sanity\overfit_case031_first6_compact_list.txt --case-list 3101,3102 --max-frames 6 --batch-size 1 --out runs\gate08_training_overfit_sanity\trained_force_closure_case031_v1.json --cuda
```

## 4. Training Results

| Run | Best Epoch | Train LE rel | Train AD B rel | Val LE rel | Val AD B rel | Result |
|---|---:|---:|---:|---:|---:|---|
| case019 v1 | 20 | 0.6372 | 1.8287 | 0.6401 | 1.8283 | FAIL |
| case019 global prior | 300 | 0.9490 | 1.8303 | 0.9746 | 1.8299 | FAIL |
| case019 pointB lowJ | 500 | 0.2435 | 1.8311 | 0.2751 | 1.8310 | FAIL |
| case031 v1 | 100 | 0.4551 | 1.3090 | 0.4452 | 1.2964 | FAIL |
| case031 B only | 1 | 1.1074 | 1.1126 | 1.0812 | 1.1182 | FAIL |

补充观察：

1. case019 的 `LE` 可以降到 `0.2435`，但 `AD_B_rel` 保持约 `1.83`。
2. case031 的 `LE` 在 latest 可降到 train `0.2552`，但 `AD_B_rel` 又升到 `1.8643`。
3. 只训练 B 时，归一化 `J_loss` 从约 `1617.6` 降到约 `1065.4`，但物理 `AD_B_rel` 恶化到 train `5.7857`。

## 5. AD B Diagnostic

诊断输出：

`runs\gate08_training_overfit_sanity\diagnose_case031_first6_adb_grad.json`

关键数值：

1. 初始 `J_norm_mse = 1617.5582`
2. 初始 `LE_norm_mse = 6.0905`
3. 初始 batch `AD_B_rel = 1.0852`
4. `B_rms = 5.5086`
5. `B_abs_max = 72.9954`
6. `j_norm_target_rms = 39.6420`
7. `j_norm_target_abs_max = 2589.8301`
8. `q_std_min = 3.4979e-6`
9. `LE_std_min = 1.7512e-5`

AD B loss 梯度：

1. `point_b_net` grad sum `1.6746`
2. `branch` grad sum `0.0240`
3. `static_b_norm` grad sum `8.80e-8`
4. `trunk` grad sum `0`
5. `LE0` grad sum `0`

判断：

1. AD B 梯度不是完全断路。
2. B 监督主要进入 `point_b_net` 和少量 `branch`。
3. `static_b_norm` 基线几乎不被 AD B loss 推动。
4. 当前 B 目标标准化尺度很硬，`j_norm_target` 最大值达到约 `2589.8`。
5. 只 B loss 能降低归一化 J loss，但不能降低物理 B rel，说明损失口径和物理 B 误差不一致。

## 6. Force Closure After Training

| Checkpoint | Frame Count | LE rel | B qdef hat rel | Model Force rel | Teacher Force rel | Result |
|---|---:|---:|---:|---:|---:|---|
| case019 pointB lowJ best | 6 | 0.2651 | 1.8311 | 1.5405 | 0.00162 | FAIL |
| case031 v1 best | 6 | 0.4482 | 1.3048 | 4.7773 | 0.01279 | FAIL |

判断：

1. Teacher 同口径 selected-frame force closure 仍然通过。
2. 模型预测的 `LE` 和 autograd `B` 装配力不闭合。
3. 当前失败不是 source128 teacher force closure 问题。

## 7. Result

Gate 08 overfit sanity: FAIL

原因：

1. 极小数据上没有把 `LE_rel` 降到明显过拟合区间。
2. 更关键的是 `AD_B_rel` 基本没有稳定下降。
3. 训练后 selected-frame force closure 明显失败。
4. AD B 梯度存在，但当前损失标准化和物理 B 拟合之间不一致。

## 8. Unresolved Issues

1. `B_macro_qdef` 的归一化目标存在很大动态范围。
2. `q_std_min` 和 `LE_std_min` 很小，可能放大部分 AD B 目标。
3. 当前 `AD_B_norm_mse` 下降不等价于物理 `AD_B_rel` 下降。
4. `static_b_norm` 初始化后几乎不被 AD B loss 更新。
5. residual 的 q 导数可能难以在极小数据下学习 frame-dependent B。

## 9. Recommended Next Action

1. 先不要做大训练。
2. 先检查 B loss 标准化口径。
3. 加一个物理 B rel 或 RMS diagnostic loss，不改变模型结构，只用于诊断。
4. 对比 forward AD 和 reverse AD 的数值一致性。
5. 单独审计 `j_norm_target = B * q_std / LE_std` 的逐列和逐积分点分布。
6. 若仍不行，再检查 `B_macro_qdef` 的刚体投影近似是否放大了部分列。
