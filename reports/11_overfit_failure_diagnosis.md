# Gate 11 Overfit Failure Diagnosis

日期：2026-06-25

角色：Macro16 Overfit Failure Diagnosis Agent

## 1. 目标

诊断 `physical-balanced` B loss 下，为什么极小数据仍不能把 `AD_B_rel` 压下来。

本轮只做诊断和小实验：

1. 不做大训练
2. 不改模型结构
3. 不改数据契约
4. 不改 `q48` 顺序
5. 不改 `LE` 顺序
6. 不改 128 点积分规则
7. 不释放训练准入

## 2. 数据

compact list：

```text
runs\gate08_training_overfit_sanity\overfit_case031_first6_compact_list.txt
```

实际 compact：

```text
runs\gate08_training_overfit_sanity\case031_first6_overfit_case_split_macro16_source128_teacher.npz
```

数据规模：

1. case：`3101` 训练，`3102` 验证
2. frame：6
3. 训练 frame：4
4. 验证 frame：2
5. 积分点：128
6. 模型可见 q：`q48_def_hat`
7. 模型可见 B：`B_macro_qdef`

## 3. 命令

基础诊断：

```powershell
$env:PYTHONPATH='src'
py -3 scripts\diagnose_macro16_overfit_failure.py --compact-list runs\gate08_training_overfit_sanity\overfit_case031_first6_compact_list.txt --val-cases 3102 --checkpoint runs\gate10_b_loss_refactor\case031_first6_physical_balanced\best.pt --out runs\gate11_overfit_failure_diagnosis\case031_first6_initial_and_gate10_checkpoint.json
```

小实验矩阵：

```powershell
$env:PYTHONPATH='src'
py -3 -m macro_deeponet.train_macro16_boundary_sobolev --compact-list runs\gate08_training_overfit_sanity\overfit_case031_first6_compact_list.txt --out-dir runs\gate11_overfit_failure_diagnosis\<run_name> --model-style le0 --epochs 200 --batch-size 4 --eval-batch-size 2 --basis-dim 64 --hidden-dim 128 --branch-depth 3 --trunk-depth 3 --jacobian-columns all --jacobian-columns-per-batch 48 --eval-columns all --jacobian-loss-scale physical-balanced --rigid-loss-weight 0.0 --lr 1.0e-3 --lr-decay 0.999 --weight-decay 0.0 --val-cases 3102 --eval-every 50
```

点级 B prior 诊断：

```powershell
$env:PYTHONPATH='src'
py -3 scripts\fit_macro16_point_b_prior.py --compact-list runs\gate08_training_overfit_sanity\overfit_case031_first6_compact_list.txt --val-cases 3102 --epochs 1000 --hidden-dim 256 --depth 4 --lr 1.0e-3 --out runs\gate11_overfit_failure_diagnosis\point_b_prior_case031_first6.json --cuda
```

B-only 500 epoch 小诊断：

```powershell
$env:PYTHONPATH='src'
py -3 -m macro_deeponet.train_macro16_boundary_sobolev --compact-list runs\gate08_training_overfit_sanity\overfit_case031_first6_compact_list.txt --out-dir runs\gate11_overfit_failure_diagnosis\b_only_point_resid0_500_cuda --model-style le0 --epochs 500 --batch-size 4 --eval-batch-size 2 --basis-dim 64 --hidden-dim 128 --branch-depth 3 --trunk-depth 3 --jacobian-columns all --jacobian-columns-per-batch 48 --eval-columns all --le-loss-weight 0.0 --jacobian-loss-weight 1.0 --jacobian-loss-scale physical-balanced --rigid-loss-weight 0.0 --residual-scale 0.0 --lr 1.0e-3 --lr-decay 0.999 --weight-decay 0.0 --val-cases 3102 --eval-every 100 --cuda
```

## 4. 数据诊断

`q48_def_hat` 训练集去均值后极低秩：

| 指标 | 数值 |
|---|---:|
| centered rank | 3 |
| 第 1 奇异值能量占比 | 0.9999999333 |
| condition nonzero | 1.9288e6 |

判断：

1. 6 frame 基本只覆盖 48 维 q 空间中的 1 条方向
2. 这个数据不能证明模型能学习完整 48 列 B
3. 也不能支撑真正训练，只能做局部 overfit 诊断

B 标签跨 frame 变化很小：

| 指标 | 数值 |
|---|---:|
| `constant_B_mean_rel` | 0.06242 |
| `B_rms` | 5.50414 |
| `B_abs_max` | 72.99537 |

判断：

1. 对固定 case031 first6，B 场接近 point-dependent constant
2. 理论上 point-wise B prior 应该能明显拟合
3. 如果完整模型拟合不好，优先怀疑 B baseline 参数化或优化路径

`LE0_star` 仍明显存在：

| 指标 | 数值 |
|---|---:|
| `LE0_star train rel to LE` | 1.13023 |
| `LE0_star val rel to LE` | 1.27974 |

判断：

1. `LE0` 结构仍是必要的
2. 但 `LE0` 对 B loss 梯度为 0，不是本轮 AD_B 压不下来的直接原因

## 5. 梯度诊断

physical B loss 初始梯度：

| 参数组 | grad norm sum |
|---|---:|
| `point_b_net` | 53.22724 |
| `static_b_norm` | 21.64867 |
| `branch` | 0.65860 |
| `le0` | 0.0 |
| `trunk` | 0.0 |

判断：

1. B loss 没有断路
2. 梯度主要进入 `point_b_net` 和 `static_b_norm`
3. `LE0` 不参与 q 导数，符合设计
4. 在 B-only 且 `residual_scale=0` 时，`trunk` 不参与是正常现象

Gate 10 checkpoint 分解：

| 指标 | 数值 |
|---|---:|
| full B rel | 1.00323 |
| linear baseline B rel | 1.00610 |
| residual B norm over target | 0.06825 |

判断：

1. Gate 10 模型的 B 主要仍是 baseline 路径
2. residual 对 B 的贡献很小
3. 问题不是 residual 主导，也不是 LE0 主导

## 6. 小实验矩阵

| 实验 | epoch | train LE rel | train AD_B rel | val LE rel | val AD_B rel |
|---|---:|---:|---:|---:|---:|
| B-only，point baseline，residual 0 | 200 | 1.03397 | 0.67463 | 0.99583 | 0.67523 |
| B-only，point baseline，residual 1 | 200 | 0.92597 | 0.65922 | 0.89553 | 0.65981 |
| B-only，global baseline only | 200 | 0.96509 | 0.97743 | 0.94656 | 0.97879 |
| B-only，大容量 | 200 | 1.10540 | 0.64528 | 1.06284 | 0.64583 |
| LE-only | 200 | 0.22176 | 1.70421 | 0.25719 | 1.52306 |
| LE+B，B weight 1 | 200 | 0.23210 | 0.88585 | 0.24709 | 0.88847 |
| LE+B，B weight 10 | 200 | 0.24648 | 0.77334 | 0.30688 | 0.77521 |
| B-only，point baseline，residual 0，500 epoch | 500 | 0.74540 | 0.57130 | 0.72738 | 0.57264 |

判断：

1. B-only 明显优于 LE+B
2. LE-only 会严重破坏 AD_B
3. B weight 从 1 到 10 有帮助，但不够
4. point baseline 必须打开
5. global baseline only 基本失败
6. 单纯增大容量帮助有限
7. residual 开关影响很小
8. 500 epoch 仍只能到 `AD_B_rel ≈ 0.571`，不是 200 epoch 太短这一个原因

## 7. Point B Prior 诊断

直接用 point MLP 拟合 `B_macro_qdef`：

| 指标 | 数值 |
|---|---:|
| epoch | 1000 |
| train B rel | 0.10969 |
| val B rel | 0.15284 |
| train J rel | 0.18604 |
| val J rel | 0.39414 |

判断：

1. 点特征到 B 场不是完全不可学
2. 直接 point MLP 明显优于完整 DeepONet 的 B-only 路径
3. 当前完整模型的 `static_b_norm + point_b_net` 路径没有充分学到这个 point-wise B prior
4. 问题更集中在 B baseline 初始化，尺度，优化口径，或参数化，而不是数据 B 标签坏掉

## 8. 结论

Gate 11 结论：FAIL，但原因已收窄。

1. `physical-balanced` loss 比旧 `j-norm` 好，但仍不能让极小数据真正 overfit
2. AD_B 梯度没有断路
3. `LE0` 不污染 B 导数，不是直接原因
4. residual 不是主因
5. 模型容量不是单独主因
6. `q48_def_hat` 极低秩是数据覆盖问题，但它不能解释 B-only point prior 能明显拟合而完整模型拟合不到位
7. 当前最可疑的是完整模型里的 B baseline 路径：
   `static_b_norm + point_b_net(point)` 的初始化，尺度和优化方式

一句话：

**B 标签和 AD 路径没有坏；极小 overfit 失败主要是 B baseline 路径没有学到接近常量的 point-wise B 场，同时当前 q 数据极低秩，不能作为训练准入证据。**

## 9. 下一步建议

优先级 1：

1. 不做大训练
2. 先做 B baseline 专项诊断
3. 对 `point_b_net` 使用 point prior 预拟合初始化
4. 再接回 `Macro16BoundaryDeepONetWithLE0` 做 B-only sanity
5. 判断 `AD_B_rel` 是否能从 `0.57` 降到接近 point prior 的 `0.11`

优先级 2：

1. 检查 `j_norm_target` 中 `q_std / LE_std` 是否让某些列优化过弱
2. 尝试直接在模型内部以物理 B 初始化 `static_b_norm`
3. 记录 `B_pred_phys` 分列 rel，而不是只看整体 rel
4. 对大 B 列和小 B 列分别统计误差

优先级 3：

1. 补多方向 q 小数据
2. 每个 case 至少覆盖正负单自由度扰动和组合扰动
3. 再做单 case overfit

仍然不能做的事：

1. 不能释放大训练
2. 不能把 material-only K 当 full tangent 门槛
3. 不能绕过 selected-frame force closure
4. 不能改 q48 顺序
5. 不能改 LE 顺序
6. 不能改 128 点规则

## 10. 产物

新增脚本：

```text
scripts\diagnose_macro16_overfit_failure.py
scripts\fit_macro16_point_b_prior.py
```

新增结果：

```text
runs\gate11_overfit_failure_diagnosis\case031_first6_initial_and_gate10_checkpoint.json
runs\gate11_overfit_failure_diagnosis\point_b_prior_case031_first6.json
runs\gate11_overfit_failure_diagnosis\b_only_point_resid0_500_cuda
```

报告：

```text
reports\11_overfit_failure_diagnosis.md
```

## 11. 验证

编译检查：

```powershell
py -3 -m py_compile scripts\diagnose_macro16_overfit_failure.py scripts\fit_macro16_point_b_prior.py
```

smoke 测试：

```powershell
$env:PYTHONPATH='src'
py -3 -m pytest tests\smoke_test.py -q
```

结果：

```text
70 passed
```
