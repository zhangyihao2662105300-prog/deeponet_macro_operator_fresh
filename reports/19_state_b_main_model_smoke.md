# Gate 19 State B Main Model Smoke

日期：2026-06-25

角色：Macro16 State B Main Model Implementation Agent

## 1. 目标

把 Gate 18 计划中的 state dependent B baseline 迁移为主训练器可选模型变体。

本轮只做实现和 smoke。

不做：

1. 不正式大训练
2. 不替换默认模型
3. 不改 `q48` 顺序
4. 不改 `LE` 顺序
5. 不改 128 点积分规则
6. 不声明 solver ready

## 2. 新增模型

新增：

```text
Macro16BoundaryDeepONetWithLE0StateB
```

文件：

```text
src/macro_deeponet/models.py
```

模型形式：

```text
LE_norm = LE0_norm(point)
        + B_state_norm(point, q_state, geometry) (q_norm - q0_norm)
        + residual_offset_norm(q_norm, point)
```

其中：

```text
B_state_norm = global_b_norm
             + baseline_scale point_b_net(point)
             + state_b_scale U(point) V(q_state, geometry)
```

## 3. detach state B

默认：

```text
detach_state_b = true
```

含义：

1. state B 是当前状态下的局部 baseline
2. AD-B 主要读取 `B_state @ q` 的一阶导数
3. 暂时不把 `dB_state / dq` 当 full tangent 项
4. full tangent 仍然待处理

## 4. 训练器接入

文件：

```text
src/macro_deeponet/train_macro16_boundary_sobolev.py
```

新增参数：

```text
--model-style le0-state-b
--state-b-rank
--state-b-scale
--state-b-lr-scale
--state-b-kind
--detach-state-b / --no-detach-state-b
--state-b-random-init
```

默认模型仍是：

```text
le0
```

所以旧训练路径不变。

## 5. checkpoint 和 force audit

checkpoint 已记录：

1. `model_style`
2. `state_b`
3. `model_state`
4. `args`
5. normalization

force audit 已能加载：

```text
macro16-boundary-deeponet-with-le0-state-b
```

文件：

```text
scripts/audit_macro16_trained_force_closure.py
```

## 6. point B prior warmstart

已保持兼容。

`global_b_norm` 和 `point_b_net` 仍可从 point B prior checkpoint 加载。

state B 新增网络独立训练。

## 7. 本地验证

编译检查：

```text
py -3 -m py_compile src/macro_deeponet/models.py src/macro_deeponet/train_macro16_boundary_sobolev.py scripts/audit_macro16_trained_force_closure.py tests/smoke_test.py
```

相关 smoke：

```text
PYTHONPATH=src;scripts py -3 -m pytest tests/smoke_test.py -q -k "state_b or macro16_training or b_prior_checkpoint"
```

结果：

```text
6 passed
```

全量 smoke：

```text
PYTHONPATH=src;scripts py -3 -m pytest tests/smoke_test.py -q
```

结果：

```text
73 passed
```

## 8. 当前结论

Gate 19 实现 smoke 通过。

已证明：

1. 新模型 forward 可用
2. 新模型 AD-B 可用
3. detach 和 no detach 路径不同
4. 训练脚本可保存 `le0-state-b` checkpoint
5. force audit 可加载 `le0-state-b` checkpoint
6. 旧默认 `le0` 路径未被替换

未证明：

1. case031 first6 已复现 Gate 17 数值
2. force rel 已低于 `0.10`
3. force rel 已低于最终 `0.02`
4. full tangent 已解决
5. 可以正式大训练

## 9. Linux 小实验命令

训练只能在 Linux GPU 电脑运行。

路径：

```text
/home/ydh/IS-FEM/gate15_macro16_warmstart_finetune/repo
```

命令：

```bash
cd /home/ydh/IS-FEM/gate15_macro16_warmstart_finetune/repo
git pull origin query-point-abaqus-route
export PYTHONPATH=src
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

CUDA_VISIBLE_DEVICES=0 python -m macro_deeponet.train_macro16_boundary_sobolev \
  --compact-list runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt \
  --out-dir runs/gate19_state_b/rank4 \
  --model-style le0-state-b \
  --state-b-rank 4 \
  --detach-state-b \
  --epochs 3000 \
  --batch-size 6 \
  --eval-batch-size 6 \
  --num-workers 8 \
  --pin-memory \
  --hidden-dim 256 \
  --branch-depth 4 \
  --trunk-depth 4 \
  --le-loss-weight 1.0 \
  --jacobian-loss-weight 5.0 \
  --jacobian-loss-scale physical-balanced \
  --jacobian-columns all \
  --jacobian-columns-per-batch 48 \
  --eval-columns all \
  --val-cases 3102 \
  --eval-every 100 \
  --lr 1.0e-3 \
  --lr-decay 0.9995 \
  --cuda &

CUDA_VISIBLE_DEVICES=1 python -m macro_deeponet.train_macro16_boundary_sobolev \
  --compact-list runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt \
  --out-dir runs/gate19_state_b/rank8 \
  --model-style le0-state-b \
  --state-b-rank 8 \
  --detach-state-b \
  --epochs 3000 \
  --batch-size 6 \
  --eval-batch-size 6 \
  --num-workers 8 \
  --pin-memory \
  --hidden-dim 256 \
  --branch-depth 4 \
  --trunk-depth 4 \
  --le-loss-weight 1.0 \
  --jacobian-loss-weight 5.0 \
  --jacobian-loss-scale physical-balanced \
  --jacobian-columns all \
  --jacobian-columns-per-batch 48 \
  --eval-columns all \
  --val-cases 3102 \
  --eval-every 100 \
  --lr 1.0e-3 \
  --lr-decay 0.9995 \
  --cuda &

CUDA_VISIBLE_DEVICES=2 python -m macro_deeponet.train_macro16_boundary_sobolev \
  --compact-list runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt \
  --out-dir runs/gate19_state_b/rank12 \
  --model-style le0-state-b \
  --state-b-rank 12 \
  --detach-state-b \
  --epochs 3000 \
  --batch-size 6 \
  --eval-batch-size 6 \
  --num-workers 8 \
  --pin-memory \
  --hidden-dim 256 \
  --branch-depth 4 \
  --trunk-depth 4 \
  --le-loss-weight 1.0 \
  --jacobian-loss-weight 5.0 \
  --jacobian-loss-scale physical-balanced \
  --jacobian-columns all \
  --jacobian-columns-per-batch 48 \
  --eval-columns all \
  --val-cases 3102 \
  --eval-every 100 \
  --lr 1.0e-3 \
  --lr-decay 0.9995 \
  --cuda &

wait
```

## 10. Gate 19 数值门槛

Linux 小实验通过标准：

1. train B rel 小于 `0.03`
2. all B rel 小于 `0.05`
3. selected-frame force rel 小于 `0.10`
4. teacher force rel 小于 `0.02`
5. 不改 `q48` 顺序
6. 不改 `LE` 顺序
7. 不改 128 点规则

如果没有达到，继续诊断，不进入正式大训练。
