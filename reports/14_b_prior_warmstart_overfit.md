# Gate 14 B Prior Warmstart Overfit Validation

日期：2026-06-25

角色：Macro16 Warmstart Overfit Validation Agent

## 1. 目标

验证 B prior warmstart 是否解决极小 overfit 中的 AD_B 学习失败。

本轮约束：

1. 不启动正式大训练
2. 不改模型主结构
3. 不改 q48 顺序
4. 不改 LE 顺序
5. 不改 128 点积分规则
6. 训练在 Linux GPU 电脑执行
7. 每一步留报告和提交痕迹

## 2. 数据

compact list：

```text
runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt
```

数据：

1. case031 first6 split
2. train case 3101
3. val case 3102
4. train frame 4
5. val frame 2
6. 积分点 128
7. q 输入 q48_def_hat
8. B 标签 B_macro_qdef

## 3. Linux 训练环境

训练机：

```text
lab-gpu-ts
```

硬件：

```text
3 x NVIDIA GeForce RTX 4090
24 GB each
CPU 64
```

本轮使用：

1. 单 GPU
2. CUDA
3. batch size 6
4. jacobian columns all
5. jacobian columns per batch 48
6. num workers 8
7. pin memory

说明：

极小 overfit 只有 6 frame，多 GPU 通信开销大于收益，所以不用 DDP。

## 4. 发现的问题

第一次运行 point B prior checkpoint 时结果异常：

```text
train B rel 0.61299
val B rel 0.61447
```

这和 Gate 11 的 direct point B prior 结果不一致。

原因：

1. Gate 11 直接在物理 B 坐标拟合 B_macro_qdef
2. Gate 13 为了 checkpoint 可加载，改成直接拟合 J_norm
3. J_norm 到 B 的换算含 LE_std 和 q_std
4. 当前 q_std 最小约 3.5e-6
5. LE_std / q_std 最大约 2476
6. 直接学 J_norm 会被列尺度放大影响
7. 所以 point prior 学习能力被人为压坏

修正：

1. point prior 仍然先在物理 B 坐标拟合
2. 保存 checkpoint 前只重拟合最后一层到 J_norm delta
3. checkpoint 仍然输出 target_coordinate = J_norm
4. 训练器仍然加载到 global_b_norm 和 point_b_net

这不改变模型结构。

## 5. 代码修正

修改文件：

```text
scripts/fit_macro16_point_b_prior.py
```

核心修正：

```text
fit coordinate = B_macro_qdef
checkpoint coordinate = J_norm
conversion = final linear head refit to J_norm delta
```

另外补充：

```text
src/macro_deeponet/train_macro16_boundary_sobolev.py
```

1. checkpoint 中 args 转成字符串安全格式
2. 新增 num_workers
3. 新增 pin_memory
4. summary 记录 dataloader_meta

```text
scripts/audit_macro16_trained_force_closure.py
```

1. 兼容 Linux checkpoint 在 Windows 上加载
2. 解决 PosixPath 反序列化问题

## 6. Point B Prior 结果

命令：

```bash
cd /home/ydh/IS-FEM/gate14_macro16_warmstart_overfit/repo
export PYTHONPATH=src
CUDA_VISIBLE_DEVICES=0 python3 scripts/fit_macro16_point_b_prior.py \
  --compact-list runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt \
  --val-cases 3102 \
  --epochs 1000 \
  --eval-every 100 \
  --hidden-dim 256 \
  --depth 4 \
  --lr 1.0e-3 \
  --out runs/gate14_b_prior_warmstart_overfit/point_b_prior.json \
  --checkpoint-out runs/gate14_b_prior_warmstart_overfit/point_b_prior.pt \
  --cuda
```

结果：

| 项目 | train | val |
|---|---:|---:|
| physical B fit rel | 0.1097309741 | 0.1528332939 |
| loadable checkpoint B rel | 0.0624246872 | 0.1299495436 |
| loadable checkpoint J rel | 0.1486711425 | 0.3711900394 |

结论：

B prior checkpoint 已经可用。

## 7. Warmstart B-only 加载验证

命令：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m macro_deeponet.train_macro16_boundary_sobolev \
  --compact-list runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt \
  --out-dir runs/gate14_b_prior_warmstart_overfit/b_only_loaded_prior_frozen \
  --model-style le0 \
  --b-prior-warmstart-checkpoint runs/gate14_b_prior_warmstart_overfit/point_b_prior.pt \
  --freeze-b-prior-after-warmstart \
  --global-b-lr-scale 0.0 \
  --point-b-lr-scale 0.0 \
  --epochs 1 \
  --batch-size 6 \
  --eval-batch-size 6 \
  --basis-dim 64 \
  --hidden-dim 256 \
  --branch-depth 4 \
  --trunk-depth 4 \
  --jacobian-columns all \
  --jacobian-columns-per-batch 48 \
  --eval-columns all \
  --le-loss-weight 0.0 \
  --jacobian-loss-weight 1.0 \
  --jacobian-loss-scale physical-balanced \
  --residual-scale 0.0 \
  --rigid-loss-weight 0.0 \
  --lr 1.0e-4 \
  --lr-decay 1.0 \
  --weight-decay 0.0 \
  --val-cases 3102 \
  --eval-every 1 \
  --cuda
```

结果：

```text
train AD_B_rel 0.0624246872
val AD_B_rel 0.1299495274
train LE_rel 0.1072727094
val LE_rel 0.1852555459
```

结论：

warmstart 加载后，训练器内 autograd B 与 checkpoint B prior 一致。

## 8. LE+B Overfit 对照

Linux 命令，warmstart：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m macro_deeponet.train_macro16_boundary_sobolev \
  --compact-list runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt \
  --out-dir runs/gate14_b_prior_warmstart_overfit/le_b_warmstart_frozen_prior_80_workers8 \
  --model-style le0 \
  --b-prior-warmstart-checkpoint runs/gate14_b_prior_warmstart_overfit/point_b_prior.pt \
  --freeze-b-prior-after-warmstart \
  --global-b-lr-scale 0.0 \
  --point-b-lr-scale 0.0 \
  --epochs 80 \
  --batch-size 6 \
  --eval-batch-size 6 \
  --num-workers 8 \
  --pin-memory \
  --basis-dim 64 \
  --hidden-dim 256 \
  --branch-depth 4 \
  --trunk-depth 4 \
  --jacobian-columns all \
  --jacobian-columns-per-batch 48 \
  --eval-columns all \
  --le-loss-weight 1.0 \
  --jacobian-loss-weight 0.1 \
  --jacobian-loss-scale physical-balanced \
  --residual-scale 0.0 \
  --rigid-loss-weight 0.0 \
  --lr 5.0e-4 \
  --lr-decay 0.999 \
  --weight-decay 0.0 \
  --val-cases 3102 \
  --eval-every 20 \
  --cuda
```

Linux 命令，no warmstart：

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m macro_deeponet.train_macro16_boundary_sobolev \
  --compact-list runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt \
  --out-dir runs/gate14_b_prior_warmstart_overfit/le_b_no_warmstart_80_workers8 \
  --model-style le0 \
  --epochs 80 \
  --batch-size 6 \
  --eval-batch-size 6 \
  --num-workers 8 \
  --pin-memory \
  --basis-dim 64 \
  --hidden-dim 256 \
  --branch-depth 4 \
  --trunk-depth 4 \
  --jacobian-columns all \
  --jacobian-columns-per-batch 48 \
  --eval-columns all \
  --le-loss-weight 1.0 \
  --jacobian-loss-weight 0.1 \
  --jacobian-loss-scale physical-balanced \
  --residual-scale 0.0 \
  --rigid-loss-weight 0.0 \
  --lr 5.0e-4 \
  --lr-decay 0.999 \
  --weight-decay 0.0 \
  --val-cases 3102 \
  --eval-every 20 \
  --cuda
```

结果：

| 项目 | warmstart | no warmstart |
|---|---:|---:|
| train LE rel | 0.1331754586 | 0.3637975709 |
| train AD_B rel | 0.0624246872 | 0.9937736268 |
| val LE rel | 0.1476313324 | 0.3970121292 |
| val AD_B rel | 0.1299495274 | 0.9958339778 |
| dataloader workers | 8 | 8 |
| pin memory | true | true |

结论：

warmstart 明显解决了极小 overfit 的 B 学习失败。

## 9. Force Closure 审计

Linux checkpoint 拉回 Windows 后审计。

原因：

compact 中 `source_compact` 是 Windows 路径，Linux 没有该源文件。

Windows 命令：

```powershell
$env:PYTHONPATH='src;scripts'
py -3 scripts\audit_macro16_trained_force_closure.py `
  --checkpoint runs\gate14_b_prior_warmstart_overfit\le_b_warmstart_frozen_prior_80_workers8\best.pt `
  --compact-list runs\gate08_training_overfit_sanity\overfit_case031_first6_compact_list.txt `
  --case-list 3101,3102 `
  --max-frames 6 `
  --batch-size 6 `
  --out runs\gate14_b_prior_warmstart_overfit\force_closure_warmstart_frozen_prior_80_workers8_windows.json
```

结果：

| 项目 | warmstart | no warmstart | teacher |
|---|---:|---:|---:|
| LE rel | 0.1433988972 | 0.3872505835 | NA |
| B qdef hat rel | 0.0907799105 | 0.9944631271 | NA |
| force rel | 0.2560688360 | 1.2685236399 | 0.0127911083 |

结论：

1. warmstart force closure 明显改善
2. 但 force rel 仍为 0.256，远高于 0.02
3. 不能释放大训练
4. 下一步应先让 LE 和 B 同时进一步下降

## 10. 当前结论

Gate 14 状态：

```text
PASS for B prior warmstart validation
FAIL for trained force closure release
```

已经证明：

1. B 标签没坏
2. AD 路径没坏
3. warmstart 能把 train AD_B_rel 从约 0.994 降到约 0.062
4. warmstart 能把 force rel 从约 1.269 降到约 0.256

还没有证明：

1. 网络可用于求解器
2. 训练后 force closure 达到 0.02
3. 可以启动正式大训练

## 11. 下一步

推荐 Gate 15：

```text
Macro16 Warmstart Fine Tune Agent
```

目标：

1. 在 warmstart 基础上解冻 B prior
2. 同时训练 LE 和 B
3. 尝试把 train LE rel 和 train AD_B rel 都压到 0.05 以下
4. force rel 目标先压到 0.10 以下
5. 仍然只做小规模 Linux GPU 训练

禁止：

1. 不做正式大训练
2. 不改模型主结构
3. 不改 q48 顺序
4. 不改 LE 顺序
5. 不改 128 点积分规则
