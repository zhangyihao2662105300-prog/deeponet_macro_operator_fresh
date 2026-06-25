# Gate 28 Force Residual Loss Plan

日期：2026-06-26

角色：Macro16 Force Residual Loss Plan Agent

## 1. 目标

设计真正对准 force functional 的训练 loss。

本轮只写方案：

1. 不训练
2. 不改模型结构
3. 不改 `q48` 顺序
4. 不改 `LE` 顺序
5. 不改 128 点规则

## 2. 背景

Gate 27 结论：

```text
strain-volume proxy force-aware B loss 可运行
但 best force rel 0.1149584190
仍差于 Gate 23 best 0.1073731865
```

原因：

```text
abs(LE_true) * integration_weight_hat 没有真正表示 B * stress * dV 对 force 的贡献
```

所以 Gate 28 改为设计 force residual loss。

## 3. 核心公式

模型输出：

```text
LE_pred[n,p,a]
B_pred[n,p,a,j]
```

teacher 标签：

```text
LE_true[n,p,a]
B_true[n,p,a,j]
```

材料矩阵：

```text
D[n,a,b]
```

积分权重：

```text
dV[n,p]
```

应力：

```text
sigma_pred[n,p,a] = D[n,a,b] * LE_pred[n,p,b]
sigma_true[n,p,a] = D[n,a,b] * LE_true[n,p,b]
```

装配力：

```text
F_pred[n,j] =
sum over p,a of B_pred[n,p,a,j] * sigma_pred[n,p,a] * dV[n,p]
```

teacher 装配力：

```text
F_teacher[n,j] =
sum over p,a of B_true[n,p,a,j] * sigma_true[n,p,a] * dV[n,p]
```

force residual loss：

```text
loss_force =
mean over n,j of ((F_pred[n,j] - F_teacher[n,j]) / F_scale[j])^2
```

建议第一版：

```text
F_scale[j] = rms(F_teacher[:,j]) with floor
```

## 4. 为什么不用 Abaqus RF 做第一版

selected-frame Abaqus RF 更接近最终 audit。

但第一版不建议直接用 RF，原因：

1. 需要 source compact 路径
2. 需要 `rigid_projection_P`
3. Windows 和 Linux 路径映射复杂
4. 会把训练 loader 和外部 source 文件强耦合
5. 当前 teacher assembled force 已经能稳定闭合到 RF 的 `0.0127911083`

所以第一版目标是：

```text
先让模型 force 对齐 teacher assembled force
```

然后仍用 selected-frame RF 做训练后 audit。

## 5. dV 选择

第一版使用：

```text
integration_weight_hat
```

原因：

1. loader 已经读取
2. 不依赖 source compact
3. 与当前训练数据同尺度一致

注意：

```text
这不是 selected-frame physical volume
```

所以它只用于训练引导，不作为最终 mechanics gate。

最终 audit 仍使用：

```text
selected-frame volume
```

## 6. elastic_D 如何纳入训练数据

### 6.1 当前问题

`Macro16Arrays` 现在没有 `elastic_D`。

当前 dataset 返回：

```text
branch
point
LE
J
weights
```

缺少：

```text
elastic_D
```

### 6.2 最小扩展

给 `Macro16Arrays` 增加字段：

```text
elastic_d: np.ndarray
```

shape：

```text
[N,6,6]
```

给 `Macro16Dataset` 增加返回：

```text
elastic_D
```

新返回：

```text
branch
point
LE
J
weights
elastic_D
```

### 6.3 elastic_D 来源

读取顺序：

1. compact 自带 `elastic_D`
2. compact 自带 `material_D`
3. source compact 的 `elastic_D`
4. fallback 报错

建议第一版：

```text
只支持 compact 自带 elastic_D 或 source compact elastic_D
没有就报错
```

不建议：

```text
默认 identity D
```

原因：

identity D 只适合调试，不适合正式 Gate 27 之后的训练判断。

### 6.4 source compact 路径问题

读取 source compact 会遇到 Windows 路径。

训练器需要新增可选参数：

```text
--source-path-map FROM=TO
```

和 force audit 保持一致。

但为了减少训练复杂度，建议先在 compact 构建阶段写入：

```text
elastic_D
```

也就是更推荐：

```text
先补 compact 字段
再训练
```

## 7. loader 改动路线

### 路线 A 推荐

新增一个只读脚本：

```text
scripts/enrich_macro16_compact_elastic_d.py
```

作用：

```text
读取 source_compact 的 elastic_D
写入 Macro16 compact
```

优点：

1. 训练不依赖 source path
2. Linux 和 Windows 更稳定
3. dataset 只读 compact 自身字段

### 路线 B 备选

训练时动态读取 source compact。

缺点：

1. 训练 loader 复杂
2. 多 worker 下可能重复打开 source 文件
3. path map 容易出错

结论：

```text
Gate 29 应优先实现路线 A
```

## 8. loss 参数设计

新增参数：

```text
--force-residual-loss-weight
--force-residual-volume-mode hat
--force-residual-scale-mode component-rms
```

默认：

```text
force-residual-loss-weight = 0.0
```

默认不影响旧训练。

第一版只支持：

```text
volume-mode hat
scale-mode component-rms
```

## 9. 训练组合建议

不要去掉原来的 B loss。

建议 loss：

```text
loss =
LE_weight * loss_LE
direct_B_weight * loss_direct_B
force_weight * loss_force
```

先不启用：

```text
force-aware-b-loss-weight
```

原因：

Gate 27 已经证明 strain-volume proxy 不够。

## 10. 三个 Linux 小实验

都用：

```text
rank12
direct-state-b-loss-weight 5
residual-scale 0
jacobian-loss-weight 0
case031 first6
```

### 实验 1

```text
force-residual-loss-weight 0.1
le-loss-weight 1.0
```

### 实验 2

```text
force-residual-loss-weight 0.5
le-loss-weight 1.0
```

### 实验 3

```text
force-residual-loss-weight 1.0
le-loss-weight 0.5
```

目标：

```text
force rel < 0.10
并优于 Gate 23 best 0.1073731865
```

## 11. Linux 命令模板

等 Gate 29 实现后运行：

```bash
cd /home/ydh/IS-FEM/gate23_fixed128_state_b/repo

export PYTHONPATH=src:scripts
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p runs/gate29_force_residual_loss

for item in "0 f01_le1 0.1 1.0" "1 f05_le1 0.5 1.0" "2 f10_le0.5 1.0 0.5"; do
  set -- $item
  gpu=$1
  name=$2
  fw=$3
  lew=$4

  CUDA_VISIBLE_DEVICES=$gpu nohup python3 -m macro_deeponet.train_macro16_boundary_sobolev \
    --compact-list runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt \
    --out-dir runs/gate29_force_residual_loss/$name \
    --plane-gauss-order 4 \
    --thickness-gauss-order 4 \
    --model-style le0-fixed128-state-b \
    --state-b-rank 12 \
    --residual-scale 0.0 \
    --detach-state-b \
    --le-loss-weight $lew \
    --jacobian-loss-weight 0.0 \
    --direct-state-b-loss-weight 5.0 \
    --force-residual-loss-weight $fw \
    --force-residual-volume-mode hat \
    --force-residual-scale-mode component-rms \
    --jacobian-columns all \
    --jacobian-columns-per-batch 48 \
    --rigid-loss-weight 0.0 \
    --epochs 3000 \
    --batch-size 6 \
    --eval-batch-size 2 \
    --hidden-dim 256 \
    --branch-depth 4 \
    --trunk-depth 4 \
    --lr 8e-5 \
    --lr-decay 0.9995 \
    --weight-decay 1e-5 \
    --grad-clip 10 \
    --val-cases 3102 \
    --eval-every 100 \
    --max-eval-frames 6 \
    --num-workers 4 \
    --pin-memory \
    --cuda > runs/gate29_force_residual_loss/$name/train.log 2>&1 &
done
```

## 12. Gate 29 实现顺序

建议下一步按这个顺序：

1. 检查当前 case031 first6 compact 是否已有 `elastic_D`
2. 如果没有，写 enrich 脚本
3. 扩展 loader 读取 `elastic_D`
4. 扩展 Dataset 返回 `elastic_D`
5. 实现 `force_residual_loss_from_state_b`
6. 新增测试
7. 不训练，只提交代码
8. 再启动 Linux 三卡小训练

## 13. 放行标准

Gate 29 小训练通过标准：

```text
teacher force rel < 0.02
model force rel < 0.10
model force rel < Gate 23 best 0.1073731865
B rel 不明显劣化
```

如果不过：

```text
回到 Gate 17 原型逐项对齐 initialization 和 checkpoint selection
```

## 14. 回答关键问题

### 14.1 是否需要改 loader

需要。

因为 force residual loss 需要 `elastic_D`。

### 14.2 是否影响默认训练

不影响。

默认：

```text
force-residual-loss-weight = 0.0
```

### 14.3 是否直接用 Abaqus RF

第一版不用。

先用：

```text
teacher assembled force
```

### 14.4 为什么还要保留 B loss

因为单独 force loss 可能让 force 对了但局部 B 错。

### 14.5 下一步是什么

Gate 29：

```text
实现 elastic_D compact enrichment 和 force residual loss
```
