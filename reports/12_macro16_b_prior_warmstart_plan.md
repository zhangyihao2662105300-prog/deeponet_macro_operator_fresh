# Gate 12 Macro16 B Prior Warmstart Plan

日期：2026-06-25

角色：Macro16 B Prior Warmstart Agent

## 1. 任务边界

本报告只制定方案。

1. 不做大训练
2. 不改模型主结构
3. 不改 `q48` 顺序
4. 不改 `LE` 顺序
5. 不改 128 点积分规则
6. 不提交

目标：

1. 先让完整模型里的 B prior 学到接近直接 point B prior 的水平
2. 再接极小数据 B-only 和 `LE+B` overfit
3. 训练仍只针对 `LE_macro` 和 `B_macro_qdef`
4. 训练后必须做 selected-frame force closure

## 2. 已读依据

1. `AGENTS.md`
2. `docs/CURRENT_STATUS.md`
3. `reports/11_overfit_failure_diagnosis.md`
4. `reports/10_b_loss_refactor_smoke.md`
5. `D:\IS-FEM\deeponet_2000epoch_remote_snapshot_20260624\code\README.md`
6. `D:\IS-FEM\deeponet_2000epoch_remote_snapshot_20260624\run\config.json`
7. `src/macro_deeponet/models.py`
8. `src/macro_deeponet/train_macro16_boundary_sobolev.py`
9. `scripts/fit_macro16_point_b_prior.py`

按 `AGENTS.md` 补读：

1. `docs/PROJECT_BRIEF.md`
2. `docs/THEORY_BASELINE.md`
3. `docs/THEORY_AUDIT_MATRIX.md`
4. `docs/GATE_WORKFLOW.md`

## 3. 当前事实

Gate 11 关键结果：

| 对象 | train B rel |
|---|---:|
| 完整模型 B-only 500 epoch | 0.57130 |
| 直接 point B prior | 0.10969 |

判断：

1. 数据标签没有坏
2. AD 求 B 没有坏
3. point feature 到 B 场可以学
4. 完整模型没有先学好 point-wise B prior
5. 当前不能大训练

## 4. 为什么完整模型 B prior 学不过直接 point prior

原因 1：当前 `Macro16BoundaryDeepONetWithLE0` 的静态 B 基线不是固定 128 点表。

当前 active 类继承的是动态点 Query FE 结构，实际 B prior 是：

```text
B_base_norm(point) = global_b_norm + point_b_net(point)
```

不是旧 fixed-IP 类里的：

```text
static_b_norm[128,6,48] + point_b_net(point)
```

训练器现在虽然用 `skip_init = mean_train(J_norm_target)` 得到 `[128,6,48]`，但传入 active Query 模型后会被平均成：

```text
global_b_norm[6,48]
```

所以点位差异几乎全要从 zero-last 的 `point_b_net` 重新学。直接 point prior 没有这个损失。

原因 2：直接 point prior 的目标更干净。

直接 point prior 训练的是：

```text
point_features -> B_macro_qdef
```

完整模型训练的是：

```text
LE = LE0(point) + B_base(point) @ (q - q0) + residual(q, point)
```

因此：

```text
dLE/dq = B_base(point) + d residual/dq
```

B loss 会牵动 baseline 和 residual 两条路径，优化更绕。

原因 3：`point_b_net` 是 zero-last 初始化。

初始时：

```text
point_b_net(point) = 0
```

如果一开始就同时训 LE、B、residual，`point_b_net` 会慢，B prior 还可能被 residual 或 LE 目标带偏。

原因 4：当前 q 数据极低秩。

Gate 11 中 case031 first6 的 `q48_def_hat` 去均值后：

```text
centered rank = 3
first singular energy share = 0.9999999333
```

这不足以靠 LE loss 约束完整 48 列 B。

原因 5：参数组没有被单独保护。

当前优化器是：

```text
AdamW(model.parameters())
```

`global_b_norm`、`point_b_net`、LE0、branch、trunk、residual 用同一套学习率。旧 2000 轮路线说明，B prior 需要先学好，再用冻结或低学习率保护。

## 5. 是否需要先单独预训练 point_b_net

结论：需要。

理由：

1. 直接 point prior 能到 `train B rel = 0.10969`
2. 完整模型 B-only 500 epoch 只能到 `train AD_B_rel = 0.57130`
3. 差距说明完整训练不能自然学好 point-wise B prior
4. 单独预训练 `point_b_net` 是最小干预
5. 不需要改模型结构

预训练原则：

1. 只用 train case
2. 不用 val case
3. 不改 compact
4. 不改 128 点规则
5. 不改 q 和 LE 顺序
6. 目标在 `J_norm` 坐标中拟合
7. 评估同时输出 physical B rel

## 6. warmstart 目标定义

训练器内部目标：

```text
J_norm_target = B_macro_qdef * q_std / LE_std
```

active 模型实际 B prior：

```text
B_base_norm(point) = global_b_norm + baseline_scale * point_b_net(point)
```

推荐 warmstart：

```text
global_b_norm = mean_train_all_points(J_norm_target)
point_b_net(point) ~= (J_norm_target - global_b_norm) / baseline_scale
```

loss 推荐仍用 physical-balanced 口径：

```text
B_pred = J_pred_norm * LE_std / q_std
B_target = J_norm_target * LE_std / q_std
loss = balanced_mse(B_pred, B_target)
```

这样参数仍在模型的 `J_norm` 坐标中，评价口径仍是物理 `B_macro_qdef`。

## 7. static_b_norm 和 point_b_net 怎么加载

必须先说明命名：

1. 旧 fixed-IP FE-linear residual 模型有 `static_b_norm`
2. 当前 active `Macro16BoundaryDeepONetWithLE0` 没有 `static_b_norm`
3. 当前等价静态项是 `global_b_norm`
4. 用户说的 `static_b_norm` 在本轮应理解为 B 静态基线

当前 active 模型加载方式：

```python
with torch.no_grad():
    model.global_b_norm.copy_(global_b_norm_tensor)
model.point_b_net.load_state_dict(point_b_net_state)
```

如果以后切回 fixed-IP 模型，才使用：

```python
with torch.no_grad():
    model.static_b_norm.copy_(static_b_norm_tensor)
model.point_b_net.load_state_dict(point_b_net_state)
```

checkpoint 建议保存：

```text
b_prior_kind = global_plus_point
global_b_norm
point_b_net_state
point_mean
point_std
branch_mean
branch_std
le_mean
le_std
q_std
train_case_ids
val_case_ids_excluded
compact_paths
target = J_norm_target
eval_B_rel_train
eval_B_rel_val
```

不建议把 direct physical point prior checkpoint 直接硬塞进完整模型，除非先转换：

```text
B_phys -> J_norm = B_phys * q_std / LE_std
```

## 8. 最小代码改动

最小改动只放在训练脚本和测试，不改 `models.py`。

文件 1：

```text
src/macro_deeponet/train_macro16_boundary_sobolev.py
```

新增参数：

```text
--b-prior-warmstart-steps
--b-prior-warmstart-lr
--b-prior-warmstart-weight-decay
--b-prior-warmstart-source train-only
--b-prior-warmstart-eval-every
--freeze-b-prior-after-warmstart
--global-b-lr-scale
--point-b-lr-scale
```

新增函数：

```text
warmstart_macro16_b_prior(...)
```

函数职责：

1. 只读取 `train_idx`
2. 计算 `J_norm_target`
3. 初始化 `global_b_norm`
4. 只训练 `point_b_net`
5. 用 physical-balanced B loss
6. 记录 warmstart 前后 train B rel 和 val B rel
7. 写入 `config.json` 和 `training_summary.json`

主训练优化器改为参数组：

```text
group 1 normal params lr = lr
group 2 global_b_norm lr = lr * global_b_lr_scale
group 3 point_b_net lr = lr * point_b_lr_scale
```

通过 warmstart 后推荐：

```text
global_b_lr_scale = 0.0 到 0.1
point_b_lr_scale = 0.0 到 0.2
```

文件 2：

```text
tests/smoke_test.py
```

新增测试：

1. warmstart 只用 train case
2. val case 不参与 warmstart loss
3. `global_b_norm` 形状是 `[6,48]`
4. `point_b_net` 输出形状是 `[B,P,6,48]`
5. `J_norm -> B_macro_qdef` 公式不变
6. 128 点规则不变
7. q48 和 LE 顺序不变

可选文件：

```text
scripts/fit_macro16_point_b_prior.py
```

只建议增加 `--save-checkpoint`，用于保存 point prior 初始化。但第一版更推荐训练器内 warmstart，减少标准化错配。

## 9. warmstart 后 overfit 通过标准

第一阶段：B prior warmstart。

| 指标 | 标准 |
|---|---:|
| train B prior rel | 小于 0.15 |
| val B prior rel | 小于 0.25 |
| 是否接近 direct point prior | 是 |
| val case 是否未参与 warmstart | 是 |

第二阶段：完整模型 B-only sanity。

设置：

```text
LE loss = 0
B loss = physical-balanced
residual_scale = 0
all 48 AD columns
rigid loss = 0
```

通过标准：

| 指标 | 标准 |
|---|---:|
| train AD_B_rel | 小于 0.20 |
| val AD_B_rel | 小于 0.30 |
| B prior 是否未被冲坏 | 是 |

第三阶段：完整 `LE+B` 极小 overfit。

设置：

```text
LE loss > 0
B loss = physical-balanced
B prior 低学习率或冻结
all 48 AD columns
```

通过标准：

| 指标 | 标准 |
|---|---:|
| train LE_rel | 小于 0.10 |
| train AD_B_rel | 小于 0.25 |
| val AD_B_rel | 不明显劣化 |
| selected-frame force closure | 必须记录 |

如果第二阶段不过，不进入 `LE+B`。

## 10. 如果通过，下一步训练命令怎么写

说明：

1. 以下命令是下一步实现 warmstart 参数后的建议命令
2. 本轮不执行
3. 小 overfit 优先用 Linux 单 GPU
4. 多 GPU不适合 6 frame tiny overfit，开销大于收益
5. 后续多 case 小训练再考虑 DDP 或多进程分 case

Linux 环境建议：

```bash
cd /home/ydh/桌面/zhangyihao/deeponet_macro_operator_fresh
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
```

第 1 个命令：B prior warmstart 加 B-only sanity。

```bash
python3 -m macro_deeponet.train_macro16_boundary_sobolev \
  --compact-list runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt \
  --out-dir runs/gate12_b_prior_warmstart/case031_first6_bonly \
  --model-style le0 \
  --cuda \
  --epochs 300 \
  --batch-size 4 \
  --eval-batch-size 2 \
  --basis-dim 64 \
  --hidden-dim 128 \
  --branch-depth 3 \
  --trunk-depth 3 \
  --b-prior-warmstart-steps 1000 \
  --b-prior-warmstart-lr 1.0e-3 \
  --b-prior-warmstart-weight-decay 0.0 \
  --b-prior-warmstart-source train-only \
  --global-b-lr-scale 0.1 \
  --point-b-lr-scale 0.1 \
  --le-loss-weight 0.0 \
  --jacobian-loss-weight 1.0 \
  --jacobian-loss-scale physical-balanced \
  --residual-scale 0.0 \
  --rigid-loss-weight 0.0 \
  --jacobian-columns all \
  --jacobian-columns-per-batch 48 \
  --eval-columns all \
  --lr 1.0e-3 \
  --lr-decay 0.999 \
  --weight-decay 0.0 \
  --val-cases 3102 \
  --eval-every 50
```

第 2 个命令：如果 B-only 通过，再跑 `LE+B` 极小 overfit。

```bash
python3 -m macro_deeponet.train_macro16_boundary_sobolev \
  --compact-list runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt \
  --out-dir runs/gate12_b_prior_warmstart/case031_first6_leb \
  --model-style le0 \
  --cuda \
  --epochs 500 \
  --batch-size 4 \
  --eval-batch-size 2 \
  --basis-dim 64 \
  --hidden-dim 128 \
  --branch-depth 3 \
  --trunk-depth 3 \
  --b-prior-warmstart-steps 1000 \
  --b-prior-warmstart-lr 1.0e-3 \
  --b-prior-warmstart-weight-decay 0.0 \
  --b-prior-warmstart-source train-only \
  --global-b-lr-scale 0.05 \
  --point-b-lr-scale 0.10 \
  --le-loss-weight 1.0 \
  --jacobian-loss-weight 1.0 \
  --jacobian-loss-scale physical-balanced \
  --residual-scale 1.0 \
  --rigid-loss-weight 0.0 \
  --jacobian-columns all \
  --jacobian-columns-per-batch 48 \
  --eval-columns all \
  --lr 5.0e-4 \
  --lr-decay 0.999 \
  --weight-decay 0.0 \
  --val-cases 3102 \
  --eval-every 50
```

第 3 个命令：训练后必须跑 selected-frame force closure。

```bash
PYTHONPATH=src:scripts python3 scripts/audit_macro16_trained_force_closure.py \
  --checkpoint runs/gate12_b_prior_warmstart/case031_first6_leb/best.pt \
  --compact-list runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt \
  --case-list 3101,3102 \
  --max-frames 6 \
  --batch-size 1 \
  --out runs/gate12_b_prior_warmstart/force_closure_case031_first6_leb.json
```

多 case 小训练通过后，Linux 性能策略：

1. 优先单机 GPU
2. batch size 尽量填满显存
3. `jacobian-columns-per-batch` 优先 48
4. 显存不够再降到 24 或 16
5. 当前 Macro16 训练器没有 DDP 参数，不把 tiny overfit 硬改成 DDP
6. 真正多 case 阶段再新增 Linux DDP launcher

## 11. 下一步执行顺序

1. 在训练器内实现 train-only B prior warmstart
2. 加参数组学习率保护
3. 加 smoke 测试
4. 跑 case031 first6 B-only sanity
5. 达到 `train AD_B_rel < 0.20` 后再跑 `LE+B`
6. 跑 selected-frame force closure
7. 仍不做大训练

## 12. 最终判断

当前不应该继续直接训练完整模型。

最小合理路线是：

1. 先 warmstart `global_b_norm + point_b_net`
2. 让完整模型的 B prior 接近 direct point prior
3. 再做 B-only sanity
4. 再做 `LE+B` 极小 overfit
5. 最后做 selected-frame force closure

一句话：

**现在的问题不是 B 标签，也不是 AD 路径，而是完整模型的 B prior 没有先学到 point-wise B 场；下一步应先做 train-only B prior warmstart，再进入极小 overfit。**
