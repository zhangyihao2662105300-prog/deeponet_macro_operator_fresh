# Gate 12 Macro16 B Prior Warmstart Plan

日期：2026-06-25

角色：Macro16 B Prior Warmstart Agent

## 1. 目标

只制定 `Macro16BoundaryDeepONetWithLE0` 的 B prior warmstart 方案。

本轮不做大训练，不改模型结构，不改 `q48` 顺序，不改 `LE` 顺序，不改 128 点积分规则。

目标是：

1. 先让 `point_b_net` 学到接近直接 point B prior 的效果
2. 再接完整 `LE + B` 小规模 overfit
3. 训练仍只针对 `LE_macro` 和 `B_macro_qdef`
4. 训练后仍必须做 selected-frame force closure

## 2. 已读依据

1. `reports\11_overfit_failure_diagnosis.md`
2. 旧 2000 轮 `fe-linear-residual` 架构
3. 当前 `Macro16BoundaryDeepONetWithLE0`
4. `docs\query_point_abaqus_route_changelog.md` 里 query B warmstart 记录

## 3. 当前问题

Gate 11 结果：

| 对象 | train B rel |
|---|---:|
| 完整模型 B-only 500 epoch | 0.57130 |
| 直接 point B prior | 0.10969 |

结论：

1. 数据标签没有坏
2. AD 求 B 没有坏
3. point feature 到 B 场不是不可学
4. 当前完整模型没有先学好 point-wise B prior

## 4. 为什么完整模型 B prior 学不过直接 point prior

原因 1：优化目标不一样。

直接 point prior 训练的是：

```text
point_features -> B_macro_qdef
```

完整模型训练的是：

```text
LE = LE0(point) + B_base(point) @ (q - q0) + residual(q, point)
```

`B` 来自：

```text
dLE/dq = B_base(point) + d residual / dq
```

所以完整模型的 B loss 会同时影响 baseline 和 residual 路径，优化更绕。

原因 2：`point_b_net` 是 zero-last 初始化。

当前 `point_b_net` 初始输出为 0，主要先靠 `static_b_norm` 的训练均值。
如果训练一开始就同时有 LE loss、AD B loss、residual 路径，`point_b_net` 很容易学得慢。

原因 3：`static_b_norm + point_b_net` 的尺度在 normalized J 空间。

训练器内部保存的是：

```text
J_norm = B_macro_qdef * q_std / LE_std
```

但评价关心的是 physical B rel。
即使使用 physical-balanced loss，参数本身仍在 J_norm 坐标中更新，容易出现部分列有效学习不足。

原因 4：当前 q 数据极低秩。

case031 first6 的 `q48_def_hat` 去均值后几乎是一维：

```text
first singular energy share = 0.9999999333
```

这不足以靠 LE loss 约束 48 维 B。
因此必须先独立约束 B prior。

原因 5：旧经验已经说明主训练会冲掉 B prior。

旧 query FE-linear residual 路线显示：

1. warmstart 能改善 B prior
2. 主训练如果不保护 baseline，会让 B 变差
3. 需要 physical/raw-B-aware loss
4. 需要冻结或降低 baseline 学习率

## 5. 是否需要先单独预训练 point_b_net

结论：需要。

理由：

1. 直接 point prior 已能到 `train B rel = 0.10969`
2. 完整模型 B-only 500 epoch 只能到 `train AD_B_rel = 0.57130`
3. 差距说明完整训练环节没有自然学好 point-wise B prior
4. warmstart 是最小干预，不改变模型结构

推荐顺序：

1. 先计算训练集 `J_norm_target`
2. 用训练集均值初始化 `static_b_norm`
3. 训练 `point_b_net(point)` 拟合剩余项
4. warmstart 只用训练 case，不用验证 case
5. warmstart 后记录 train 和 val B prior rel

## 6. warmstart 目标定义

当前模型里：

```text
B_base_norm(point) = static_b_norm + point_b_net(point)
```

warmstart 应该拟合：

```text
B_base_norm(point) ~= J_norm_target
```

其中：

```text
J_norm_target = B_macro_qdef * q_std / LE_std
```

推荐分解：

```text
static_b_norm = mean_train(J_norm_target)
point_b_net(point) = mean_train(J_norm_target) - static_b_norm
```

对固定 128 点当前模型，`static_b_norm` 可以保持 `[128, 6, 48]`。

如果后面要支持 query points，再用旧 query 路线的：

```text
global_b_norm + point_b_net(point)
```

但本轮 Macro16 source128 不需要改 128 点规则。

## 7. static_b_norm 和 point_b_net 怎么加载

推荐两种加载方式。

方式 1：训练器内 warmstart。

在 `train_macro16_boundary_sobolev.py` 建模后执行：

```text
1. 用 train_idx 计算 j_norm_target_train
2. static_b_norm.copy_(mean_train(j_norm_target_train))
3. 冻结除 point_b_net 外所有参数
4. 训练 point_b_net 拟合 residual_target
5. 解冻后进入主训练
```

优点：

1. 不需要额外 checkpoint 格式
2. 不需要改变模型结构
3. 不容易出现标准化不一致

方式 2：单独 warmstart checkpoint。

新增脚本输出：

```text
b_prior_warmstart.pt
```

其中保存：

```text
static_b_norm
point_b_net.state_dict
branch_mean
branch_std
point_mean
point_std
le_mean
le_std
q_std
compact list hash
train case ids
val case ids excluded
```

主训练加载时：

```text
model.static_b_norm.copy_(checkpoint["static_b_norm"])
model.point_b_net.load_state_dict(checkpoint["point_b_net"])
```

建议先用方式 1。
方式 2 后面再做可复现大流程。

## 8. 最小代码改动

只改训练脚本，不改模型结构。

新增参数：

```text
--b-prior-warmstart-steps
--b-prior-warmstart-lr
--b-prior-warmstart-weight-decay
--b-prior-warmstart-source train-only
--freeze-b-prior-after-warmstart
--static-b-lr-scale
--point-b-lr-scale
```

新增函数：

```text
warmstart_macro16_b_prior(...)
```

函数职责：

1. 只使用 train_idx
2. 计算 warmstart 前 B prior rel
3. 复制 `static_b_norm`
4. 只训练 `point_b_net`
5. 计算 warmstart 后 B prior rel
6. 记录 val cases excluded
7. 写入 `training_summary.json`

新增主训练控制：

1. 可冻结 `static_b_norm`
2. 可冻结 `point_b_net`
3. 可降低 `static_b_norm` 学习率
4. 可降低 `point_b_net` 学习率

新增测试：

1. warmstart 不使用 val case
2. warmstart 后 `static_b_norm` 形状仍为 `[128, 6, 48]`
3. warmstart 不改变 q48 顺序
4. warmstart 不改变 LE 顺序
5. warmstart 不改变 128 点规则
6. warmstart 后 B prior rel 下降

不需要改：

1. `Macro16BoundaryDeepONetWithLE0`
2. compact 数据契约
3. `q48_def_hat`
4. `B_macro_qdef`
5. 128 点 point table

## 9. Linux 性能使用方案

训练时使用 Linux 电脑，优先利用 GPU。

单机单 GPU 小诊断：

```bash
export PYTHONPATH=src
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
python3 -m macro_deeponet.train_macro16_boundary_sobolev \
  --compact-list runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt \
  --out-dir runs/gate12_b_prior_warmstart/case031_first6 \
  --model-style le0 \
  --cuda \
  --epochs 200 \
  --batch-size 4 \
  --eval-batch-size 2 \
  --jacobian-columns all \
  --jacobian-columns-per-batch 48 \
  --eval-columns all \
  --b-prior-warmstart-steps 1000 \
  --b-prior-warmstart-lr 1.0e-3 \
  --le-loss-weight 0.0 \
  --jacobian-loss-weight 1.0 \
  --jacobian-loss-scale physical-balanced \
  --residual-scale 0.0 \
  --rigid-loss-weight 0.0 \
  --val-cases 3102
```

多 GPU 后续方案：

1. warmstart 本身数据很小，不优先用 DDP
2. 多 case 后再开 DDP
3. 每张卡固定不同 case batch
4. `jacobian-columns-per-batch` 优先设为 48
5. 如果显存不足，再降到 24 或 16

性能原则：

1. 小 overfit 先用单 GPU，避免 DDP 开销
2. 多几何多 case 再用 DDP
3. 全 48 列 AD 优先，避免列采样掩盖 B 问题
4. batch 尽量填满显存，但不能改变 case split

## 10. warmstart 后 overfit 通过标准

第一阶段：B prior warmstart 通过。

固定 case031 first6：

| 指标 | 标准 |
|---|---:|
| `b_prior_after_train_B_rel` | 小于 0.15 |
| `b_prior_after_val_B_rel` | 小于 0.25 |
| train prior 接近 direct point prior | 是 |
| val cases 是否未参与 warmstart | 是 |

第二阶段：完整模型 B-only sanity 通过。

设置：

```text
LE loss = 0
B loss = physical-balanced
residual_scale = 0
all 48 AD columns
```

通过标准：

| 指标 | 标准 |
|---|---:|
| train AD_B_rel | 小于 0.20 |
| val AD_B_rel | 小于 0.30 |
| B prior rel 不被主训练冲坏 | 是 |

第三阶段：完整 `LE + B` 小 overfit 通过。

设置：

```text
LE loss > 0
B loss = physical-balanced
B baseline lr scale <= 0.10
必要时 freeze B prior 前 50 到 100 epoch
```

通过标准：

| 指标 | 标准 |
|---|---:|
| train LE_rel | 小于 0.10 |
| train AD_B_rel | 小于 0.25 |
| val AD_B_rel | 不明显劣化 |
| selected-frame force closure | 必须记录 |

如果第二阶段不过：

1. 不进入 LE+B
2. 检查 warmstart target 是否是 `J_norm_target`
3. 检查 `q_std / LE_std` 反算 B 的公式
4. 检查 `static_b_norm` 是否被主训练覆盖

## 11. 下一步执行顺序

1. 给 `train_macro16_boundary_sobolev.py` 加 train-only B prior warmstart
2. 加 warmstart 元数据和测试
3. 跑 case031 first6 B prior warmstart
4. 跑 B-only sanity
5. 若 `AD_B_rel < 0.20`，再跑 `LE+B` 小 overfit
6. 跑 selected-frame trained force closure 小审计
7. 仍不做大训练

## 12. 最终判断

当前不应该继续直接训练完整模型。

应先做：

```text
point_b_net train-only warmstart
static_b_norm 正确初始化
B prior 保护
B-only overfit
LE+B 小 overfit
force closure 小审计
```

一句话：

**下一步不是加大训练，而是把 Macro16 的 B prior warmstart 接入训练器，让完整模型先达到 direct point prior 附近的 B 拟合能力。**
