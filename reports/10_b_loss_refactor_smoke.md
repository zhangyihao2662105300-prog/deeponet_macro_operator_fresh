# Gate 10 B Loss Refactor Smoke

Date: 2026-06-25

Role: B Loss Refactor Agent

## 1. Task Goal

实现物理 `B_macro_qdef` 坐标下的 balanced B loss。

本轮不改模型结构，不改数据契约，不改 `q48` 顺序，不改 `LE` 顺序，不改 128 点积分规则。

## 2. Code Changes

Updated:

1. `src\macro_deeponet\train_macro16_boundary_sobolev.py`
2. `tests\smoke_test.py`

新增训练参数：

```text
--jacobian-loss-scale physical-balanced
--jacobian-loss-scale j-norm
--physical-b-loss-floor-rel
--physical-b-loss-floor-abs
```

当前默认：

```text
jacobian_loss_scale = physical-balanced
```

旧口径保留：

```text
jacobian_loss_scale = j-norm
```

## 3. Loss Formula

旧口径：

```text
loss = mse(J_pred_norm, J_target_norm)
```

其中：

```text
J_target_norm = B_macro_qdef * q_std / LE_std
```

新口径：

```text
B_pred = J_pred_norm * LE_std / q_std
B_target = J_target_norm * LE_std / q_std
loss = balanced_mean(((B_pred - B_target) / B_scale)^2)
```

`B_scale` 来自训练集 `B_macro_qdef` 的 component by column RMS，并带 floor。

## 4. Tests

命令：

```powershell
py -3 -m py_compile src\macro_deeponet\train_macro16_boundary_sobolev.py scripts\diagnose_macro16_b_loss_scale.py
```

```powershell
$env:PYTHONPATH='src'
py -3 -m pytest tests\smoke_test.py -q
```

结果：

```text
70 passed
```

新增测试覆盖：

1. `J_norm -> B_macro_qdef` 反算公式
2. physical balanced B loss 在预测等于目标时为 0
3. 默认 Macro16 training smoke 使用 `physical-balanced`
4. 旧 `j-norm` loss 仍可显式启用

## 5. Tiny Overfit Smoke

数据：

```text
runs\gate08_training_overfit_sanity\overfit_case031_first6_compact_list.txt
```

设置：

1. 6 frames
2. train case `3101`
3. val case `3102`
4. 80 epochs
5. all 48 AD columns
6. `Macro16BoundaryDeepONetWithLE0`

新 loss 命令：

```powershell
$env:PYTHONPATH='src'
py -3 -m macro_deeponet.train_macro16_boundary_sobolev --compact-list runs\gate08_training_overfit_sanity\overfit_case031_first6_compact_list.txt --out-dir runs\gate10_b_loss_refactor\case031_first6_physical_balanced --model-style le0 --epochs 80 --batch-size 4 --eval-batch-size 2 --basis-dim 64 --hidden-dim 128 --branch-depth 3 --trunk-depth 3 --jacobian-columns all --jacobian-columns-per-batch 48 --eval-columns all --le-loss-weight 1.0 --jacobian-loss-weight 0.1 --jacobian-loss-scale physical-balanced --rigid-loss-weight 0.0 --lr 5.0e-4 --lr-decay 0.999 --weight-decay 0.0 --val-cases 3102 --eval-every 20
```

旧 loss 对照命令：

```powershell
$env:PYTHONPATH='src'
py -3 -m macro_deeponet.train_macro16_boundary_sobolev --compact-list runs\gate08_training_overfit_sanity\overfit_case031_first6_compact_list.txt --out-dir runs\gate10_b_loss_refactor\case031_first6_jnorm_legacy --model-style le0 --epochs 80 --batch-size 4 --eval-batch-size 2 --basis-dim 64 --hidden-dim 128 --branch-depth 3 --trunk-depth 3 --jacobian-columns all --jacobian-columns-per-batch 48 --eval-columns all --le-loss-weight 1.0 --jacobian-loss-weight 0.1 --jacobian-loss-scale j-norm --rigid-loss-weight 0.0 --lr 5.0e-4 --lr-decay 0.999 --weight-decay 0.0 --val-cases 3102 --eval-every 20
```

## 6. Training Metrics

| Loss scale | Epoch | Train LE rel | Train AD B rel | Val LE rel | Val AD B rel |
|---|---:|---:|---:|---:|---:|
| physical-balanced | 80 | 0.4112 | 1.0032 | 0.4117 | 1.0065 |
| j-norm best | 60 | 0.4610 | 1.2856 | 0.4533 | 1.2876 |
| j-norm latest | 80 | 0.4120 | 1.3651 | 0.4253 | 1.3486 |

判断：

1. 新 loss 明显优于旧 `j-norm` 口径。
2. 新 loss 下物理 `AD_B_rel` 从初始约 `1.10` 降到约 `1.00`。
3. 旧 `j-norm` 在同等设置下物理 `AD_B_rel` 恶化到约 `1.35`。
4. 新 loss 改善了方向，但还没有达到真正 overfit。

## 7. Force Closure Smoke

命令：

```powershell
$env:PYTHONPATH='src;scripts'
py -3 scripts\audit_macro16_trained_force_closure.py --checkpoint runs\gate10_b_loss_refactor\case031_first6_physical_balanced\best.pt --compact-list runs\gate08_training_overfit_sanity\overfit_case031_first6_compact_list.txt --case-list 3101,3102 --max-frames 6 --batch-size 1 --out runs\gate10_b_loss_refactor\force_closure_case031_physical_balanced.json
```

```powershell
$env:PYTHONPATH='src;scripts'
py -3 scripts\audit_macro16_trained_force_closure.py --checkpoint runs\gate10_b_loss_refactor\case031_first6_jnorm_legacy\best.pt --compact-list runs\gate08_training_overfit_sanity\overfit_case031_first6_compact_list.txt --case-list 3101,3102 --max-frames 6 --batch-size 1 --out runs\gate10_b_loss_refactor\force_closure_case031_jnorm_legacy.json
```

结果：

| Loss scale | Model LE rel | Model B rel | Model force rel | Teacher force rel |
|---|---:|---:|---:|---:|
| physical-balanced | 0.4115 | 1.0043 | 1.3476 | 0.01279 |
| j-norm | 0.4557 | 1.2863 | 4.9703 | 0.01279 |

判断：

1. Teacher 同口径 force closure 仍通过。
2. 新 loss 的模型 force rel 明显低于旧 `j-norm`。
3. 但模型 force closure 仍失败。

## 8. Result

Gate 10 B loss refactor smoke: IMPLEMENTED and SMOKE PASS.

但训练收敛状态仍是 FAIL。

结论：

1. physical balanced B loss 已实现。
2. 旧 `j-norm` loss 已保留。
3. 公式测试通过。
4. 完整 smoke tests 通过。
5. 极小 overfit 下新 loss 明显改善物理 B 和 force，但没有真正过拟合。
6. 当前不能释放大训练。

## 9. Next Action

1. 保留 `physical-balanced` 作为 Macro16 训练默认 B loss。
2. 下一步不要直接大训练。
3. 先继续做小数据 overfit：
   1. 提高 `jacobian-loss-weight`
   2. 尝试 B-only physical-balanced
   3. 尝试更长 epoch
   4. 检查 `point_b_net` 和 branch 梯度
4. 如果 physical-balanced B-only 仍不能过拟合，再查：
   1. `B_macro_qdef` 小转角刚体投影近似
   2. AD_B 路径容量
   3. `static_b_norm` 和 point baseline 初始化方式
