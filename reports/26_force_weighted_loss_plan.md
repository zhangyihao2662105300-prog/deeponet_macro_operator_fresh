# Gate 26 Force Weighted Loss Plan

日期：2026-06-26

角色：Macro16 Force Weighted Loss Plan Agent

## 1. 目标

设计 force aware B loss。

本轮只写方案：

1. 不训练
2. 不改模型结构
3. 不改 `q48` 顺序
4. 不改 `LE` 顺序
5. 不改 128 点规则

## 2. 为什么 global B rel 不够

Gate 25 已证明：

```text
Gate 24 的 LE rel 更低
Gate 24 的 B rel 更低
Gate 24 的 force rel 更高
```

数值：

| checkpoint | LE rel | B rel | force rel |
|---|---:|---:|---:|
| Gate 23 best | 0.1373813802 | 0.0731878231 | 0.1073731865 |
| Gate 24 best | 0.1045596145 | 0.0710433862 | 0.1781247428 |

原因：

```text
global B rel 平均了积分点，strain component，q dof
force functional 会按 stress 和 dV 把误差重新加权
```

所以小的 B rel 不一定产生小的 force rel。

Gate 25 的具体证据：

1. 最大误差集中在最大 q frame
2. Gate 24 force 幅值在最差 frame 偏低
3. Gate 24 的 force error 由 `E11` 主导
4. 主导 q dof 发生变化

## 3. 现有 B loss

当前 direct physical B loss 近似为：

```text
B_pred = J_pred_norm * LE_std / q_std
loss_B = mean(((B_pred - B_true) / B_scale)^2)
```

它的问题：

```text
每个 B 项主要按统计尺度归一
没有看这个 B 项对 force 的贡献
```

## 4. 方案一 stress weighted B loss

### 4.1 公式

先由 teacher LE 得到 teacher stress：

```text
sigma_true[n,p,a] = D[n,a,b] * LE_true[n,p,b]
```

定义权重：

```text
w_force[n,p,a] = abs(sigma_true[n,p,a]) * dV[n,p]
```

归一化：

```text
w_norm = w_force / mean(w_force)
```

loss：

```text
loss_stress_B =
mean over n,p,a,j of
w_norm[n,p,a] * ((B_pred[n,p,a,j] - B_true[n,p,a,j]) / B_scale[p,a,j])^2
```

### 4.2 含义

这个 loss 强调：

1. 大 stress 的 strain component
2. 大体积权重积分点
3. 对 force 更敏感的 B 项

### 4.3 数据需求

需要：

1. `LE_macro`
2. `B_macro_qdef`
3. `integration_weight_hat`
4. `L_ref`
5. `elastic_D`

如果只用无量纲训练权重，可先用：

```text
dV_hat = integration_weight_hat
```

如果要和 selected-frame force 更一致，需要：

```text
selected-frame dV
```

### 4.4 是否需要改 loader

最小实现不需要改 compact loader 主接口。

原因：

当前 `Macro16Dataset` 已经返回：

```text
weights
```

也就是 `integration_weight_hat`。

但如果要用 selected-frame dV，需要扩展训练数据读取 source compact 或新增 compact 字段。

建议第一版：

```text
先用 integration_weight_hat 做 stress weighted B loss
不碰 source selected-frame path
```

## 5. 方案二 force residual loss

### 5.1 公式

用模型 LE 和模型 B 直接装配训练内 force：

```text
F_pred[n,j] =
sum over p,a of B_pred[n,p,a,j] * sigma_pred[n,p,a] * dV[n,p]
```

teacher force：

```text
F_teacher[n,j] =
sum over p,a of B_true[n,p,a,j] * sigma_true[n,p,a] * dV[n,p]
```

loss：

```text
loss_force =
mean over n,j of ((F_pred[n,j] - F_teacher[n,j]) / F_scale[j])^2
```

其中：

```text
sigma_pred = D * LE_pred
sigma_true = D * LE_true
```

### 5.2 含义

这个 loss 直接对齐 force functional。

优点：

```text
和最终 force audit 更一致
```

风险：

```text
可能隐藏局部 B 错误
可能出现 force 对了但 B 分布错
```

所以不能替代 B loss，只能辅助。

### 5.3 数据需求

需要：

1. `LE_macro`
2. `B_macro_qdef`
3. `integration_weight_hat`
4. `elastic_D`

如果使用 selected-frame force reference，还需要：

1. `source_compact`
2. `source_row`
3. `rigid_projection_P`
4. `RF_projected`
5. `ip_IVOL_abaqus_selected_frames`

第一版不建议直接接 Abaqus RF。

建议先用：

```text
teacher assembled force
```

原因：

这样不需要跨平台 source path，不引入 Abaqus RF 路径依赖。

## 6. 两个方案区别

| 方案 | 约束对象 | 优点 | 风险 |
|---|---|---|---|
| stress weighted B loss | B 局部误差 | 保留 B 标签监督，较稳定 | 仍是 proxy |
| force residual loss | 装配 force | 最贴近 force audit | 可能掩盖局部 B 错误 |

建议顺序：

```text
先 stress weighted B loss
再加 small force residual loss
```

## 7. 最小代码修改方案

### 7.1 新参数

新增：

```text
--stress-weighted-b-loss-weight
--force-residual-loss-weight
--force-loss-volume-mode hat
```

第一版只支持：

```text
hat
```

也就是用 `integration_weight_hat`。

### 7.2 Dataset

当前 dataset 已返回：

```text
_wb
```

训练循环里现在没有用它。

改为：

```text
for xb, pb, leb, jb, wb in train_loader:
```

用 `wb` 作为 `dV_hat`。

### 7.3 elastic D

第一版可用各向同性常数矩阵或从 compact/source 读取。

更稳妥：

```text
从 source compact 读取 elastic_D
并写进 Macro16Arrays
```

但这会扩展 loader。

最小风险方案：

```text
先只实现 stress weighted B loss，不需要 D
用 abs(LE_true) * dV_hat 作为 stress proxy
```

即：

```text
w_force_proxy[n,p,a] = abs(LE_true[n,p,a]) * dV_hat[n,p]
```

这个不是严格 stress，但能先验证 force-aware weighting 是否有效。

第二版再接 `elastic_D`。

### 7.4 推荐第一版实现

先实现：

```text
--force-aware-b-loss-weight
--force-aware-b-weight-mode strain-volume
```

公式：

```text
w[n,p,a] = abs(LE_true[n,p,a]) * weight_hat[n,p]
w = clamp(w / mean(w), min, max)
loss = mean(w * ((B_pred - B_true) / B_scale)^2)
```

建议 clamp：

```text
min 0.1
max 10.0
```

原因：

防止最大 q frame 或 E11 权重过度支配。

## 8. 是否影响默认训练

不影响。

原因：

新增参数默认：

```text
--force-aware-b-loss-weight 0.0
```

旧命令行为不变。

默认 `physical-balanced` B loss 不改。

## 9. 三个 Linux 小实验

只跑 `rank12`。

仍用：

```text
case031 first6
```

### 实验 1

目的：

```text
轻量 force-aware B weighting
```

参数：

```text
direct-state-b-loss-weight 5
force-aware-b-loss-weight 1
le-loss-weight 1
```

### 实验 2

目的：

```text
中等 force-aware B weighting
```

参数：

```text
direct-state-b-loss-weight 5
force-aware-b-loss-weight 3
le-loss-weight 1
```

### 实验 3

目的：

```text
保留 Gate 23 口径，加 force-aware checkpoint pressure
```

参数：

```text
direct-state-b-loss-weight 5
force-aware-b-loss-weight 1
le-loss-weight 0.5
```

## 10. Linux 命令模板

```bash
cd /home/ydh/IS-FEM/gate23_fixed128_state_b/repo

export PYTHONPATH=src:scripts
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p runs/gate27_force_aware_b_loss

for item in "0 fa1_le1 1 1" "1 fa3_le1 3 1" "2 fa1_le0.5 1 0.5"; do
  set -- $item
  gpu=$1
  name=$2
  faw=$3
  lew=$4

  CUDA_VISIBLE_DEVICES=$gpu nohup python3 -m macro_deeponet.train_macro16_boundary_sobolev \
    --compact-list runs/gate08_training_overfit_sanity/overfit_case031_first6_compact_list.txt \
    --out-dir runs/gate27_force_aware_b_loss/$name \
    --plane-gauss-order 4 \
    --thickness-gauss-order 4 \
    --model-style le0-fixed128-state-b \
    --state-b-rank 12 \
    --residual-scale 0.0 \
    --detach-state-b \
    --le-loss-weight $lew \
    --jacobian-loss-weight 0.0 \
    --direct-state-b-loss-weight 5.0 \
    --force-aware-b-loss-weight $faw \
    --force-aware-b-weight-mode strain-volume \
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
    --cuda > runs/gate27_force_aware_b_loss/$name/train.log 2>&1 &
done
```

注意：

```text
上面命令要等代码实现后才能运行
```

## 11. 放行标准

小实验通过标准：

```text
teacher force rel < 0.02
model force rel < 0.10
model force rel 优于 Gate 23 best 0.1073731865
不能明显恶化 B rel
```

如果过：

```text
进入更大 case split 小训练
```

如果不过：

```text
回到 Gate 17 原型逐项对齐初始化和 selection
```

## 12. 回答五个必须问题

### 12.1 loss 公式

第一版：

```text
w = clamp(abs(LE_true) * dV_hat / mean(abs(LE_true) * dV_hat), 0.1, 10)
loss_force_aware_B =
mean(w * ((B_pred - B_true) / B_scale)^2)
```

第二版：

```text
loss_force =
mean(((sum(B_pred * sigma_pred * dV) -
       sum(B_true * sigma_true * dV)) / F_scale)^2)
```

### 12.2 需要哪些数据

第一版需要：

```text
LE_macro
B_macro_qdef
integration_weight_hat
```

第二版需要：

```text
elastic_D
```

如果直接对 Abaqus RF，需要：

```text
source_compact
source_row
RF_projected
ip_IVOL_abaqus_selected_frames
rigid_projection_P
```

### 12.3 是否需要改 loader

第一版不需要大改。

只需要训练循环使用 dataset 已返回的 `weights`。

如果要严格 stress 或 Abaqus RF，需要改 loader。

### 12.4 是否影响默认训练

不影响。

新 loss 默认权重为 `0.0`。

### 12.5 先跑哪三个实验

```text
fa1_le1
fa3_le1
fa1_le0.5
```

都用 Linux 三张 4090 并行跑。
