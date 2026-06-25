# Gate 13 B Prior Warmstart Implementation

日期：2026-06-25

角色：Macro16 B Prior Warmstart Implementation Agent

## 1. 目标

实现 point B prior warmstart 加载。

本轮约束：

1. 不做大训练
2. 不改模型主结构
3. 不改 `q48` 顺序
4. 不改 `LE` 顺序
5. 不改 128 点积分规则

## 2. 改动

文件 1：

```text
scripts/fit_macro16_point_b_prior.py
```

新增能力：

1. `--checkpoint-out`
2. `--baseline-scale`
3. 输出可加载 checkpoint
4. checkpoint 保存 `global_b_norm`
5. checkpoint 保存 `point_b_net_state`
6. checkpoint 明确目标坐标是 `J_norm`
7. checkpoint 记录 train case 和 excluded val case

checkpoint 格式：

```text
b_prior_checkpoint_version = macro16-point-b-prior-v1
b_prior_kind = global_plus_point
target_coordinate = J_norm
global_b_norm = [6,48]
point_b_net_state = MLP state dict
baseline_scale
norms
split
best
```

第 14 步修正：

```text
fit_coordinate = B_macro_qdef
target_coordinate = J_norm
checkpoint_conversion = final_linear_head_refit_to_J_norm_delta_least_squares
```

原因：

直接在 `J_norm` 坐标拟合会受到 `LE_std / q_std` 极端列尺度影响，导致 direct point prior 从 Gate 11 的 `train B rel ≈ 0.11` 退化到约 `0.61`。修正后先在物理 `B_macro_qdef` 坐标拟合，再把最后一层重拟合为可加载的 `J_norm` checkpoint。

文件 2：

```text
src/macro_deeponet/train_macro16_boundary_sobolev.py
```

新增能力：

1. `--b-prior-warmstart-checkpoint`
2. `--freeze-b-prior-after-warmstart`
3. `--global-b-lr-scale`
4. `--point-b-lr-scale`
5. 加载 checkpoint 到 `global_b_norm`
6. 加载 checkpoint 到 `point_b_net`
7. 校验 `point_dim`
8. 校验 `q_dim = 48`
9. 校验 `strain_dim = 6`
10. 校验标准化统计量一致
11. 记录 `b_prior_warmstart_meta`
12. 记录 `optimizer_meta`

文件 3：

```text
tests/smoke_test.py
```

新增 smoke：

```text
test_macro16_point_b_prior_checkpoint_loads_into_training_model
```

覆盖：

1. point prior 脚本能输出 checkpoint
2. checkpoint 目标坐标是 `J_norm`
3. `global_b_norm` 形状是 `[6,48]`
4. train case 和 val case 分离
5. 训练器能加载 checkpoint
6. checkpoint 被加载到 `global_b_norm`
7. `point_b_net` 权重能加载
8. 可冻结 B prior

## 3. 不变项

保持不变：

1. `Macro16BoundaryDeepONetWithLE0` 主结构
2. `q48` 顺序
3. `LE` 顺序
4. 128 点积分规则
5. 数据契约
6. `B_macro_qdef` 口径
7. `LE_macro` 口径

## 4. 使用方式

先拟合 point B prior：

```powershell
$env:PYTHONPATH='src'
py -3 scripts\fit_macro16_point_b_prior.py `
  --compact-list runs\gate08_training_overfit_sanity\overfit_case031_first6_compact_list.txt `
  --val-cases 3102 `
  --epochs 1000 `
  --hidden-dim 256 `
  --depth 4 `
  --lr 1.0e-3 `
  --out runs\gate13_b_prior_warmstart\point_b_prior.json `
  --checkpoint-out runs\gate13_b_prior_warmstart\point_b_prior.pt `
  --cuda
```

再加载到 Macro16 训练器：

```powershell
$env:PYTHONPATH='src'
py -3 -m macro_deeponet.train_macro16_boundary_sobolev `
  --compact-list runs\gate08_training_overfit_sanity\overfit_case031_first6_compact_list.txt `
  --out-dir runs\gate13_b_prior_warmstart\case031_first6_bonly `
  --model-style le0 `
  --b-prior-warmstart-checkpoint runs\gate13_b_prior_warmstart\point_b_prior.pt `
  --global-b-lr-scale 0.1 `
  --point-b-lr-scale 0.1 `
  --le-loss-weight 0.0 `
  --jacobian-loss-weight 1.0 `
  --jacobian-loss-scale physical-balanced `
  --residual-scale 0.0 `
  --rigid-loss-weight 0.0 `
  --jacobian-columns all `
  --jacobian-columns-per-batch 48 `
  --eval-columns all `
  --val-cases 3102 `
  --cuda
```

Linux 上建议：

1. 小 overfit 用单 GPU
2. `jacobian-columns-per-batch` 优先 48
3. batch size 按显存调大
4. 多 case 小训练再考虑并行

## 5. 验证

已做局部验证：

```powershell
py -3 -m py_compile scripts\fit_macro16_point_b_prior.py src\macro_deeponet\train_macro16_boundary_sobolev.py tests\smoke_test.py
```

```powershell
$env:PYTHONPATH='src'
py -3 -m pytest tests\smoke_test.py::test_macro16_point_b_prior_checkpoint_loads_into_training_model -q
```

结果：

```text
1 passed
```

全量 smoke 按用户要求执行：

```powershell
$env:PYTHONPATH='src'
py -3 -m pytest tests\smoke_test.py -q
```

结果：

```text
71 passed
```

## 6. 下一步

1. 跑完整 smoke
2. 若通过，提交并推送
3. 后续才能做 B-only 极小 overfit
4. 仍不能直接大训练

## 7. 结论

point B prior warmstart 加载链路已实现。

当前路线变成：

```text
fit point B prior
输出 global_b_norm + point_b_net checkpoint
Macro16 训练器加载 checkpoint
B-only sanity
LE+B 小 overfit
selected-frame force closure
```

一句话：

**现在已经可以把 direct point B prior 作为 warmstart 接入 `Macro16BoundaryDeepONetWithLE0`，但下一步仍只能做小 overfit 验证，不能直接大训练。**
