# 38b Coordinate DeepONet Linux Overfit

## 1 结论

Gate 38b 未通过。

原因不是 Linux GPU 性能不足，也不是训练没有启动。

本轮已经在 Linux 3 张 RTX 4090 上完成 case031 first6 小样本 overfit。结果显示：

1. `LE_rel` 能下降，但没有到 `< 0.05`
2. `AD_B_rel` 基本停在 `0.999`
3. `FD_B_rel` 基本停在 `0.999`
4. `plus_fd_rel` 基本停在 `0.999`
5. coordinate-only trunk 路线当前不能学习 48 列 B

当前判断：

只用 `q48_def_hat + geometry_g + x_gp_hat` 的输入仍不足。

## 2 数据

数据路径：

```text
runs/macro16_coordinate_deeponet/case031_first6_coordinate_dataset.npz
```

frame 数量：

```text
6
```

case：

```text
3101
3102
```

数据字段：

```text
q48_def_hat       [6,48]
geometry_g        [6,14]
x_gp_hat          [6,128,3]
LE_macro          [6,128,6]
B_macro_qdef      [6,128,6,48]
q48_def_hat_plus  [6,48,48]
LE_macro_plus     [6,48,128,6]
```

## 3 已修正的问题

原先 `LE_macro_plus` 的 48 列来自 source TRUE176 keep 顺序。

模型输入的 `q48_def_hat` 是 Macro16 顺序。

这导致 source plus FD 约束列错位。

已在独立实验代码中修正：

1. `plus_macro_directions = 0..47`
2. `plus_source_directions` 记录原 source 列
3. `LE_macro_plus` 按 Macro16 列顺序重排
4. `q48_def_hat_plus` 按 Macro16 列顺序生成

修正后标签审计：

```text
B_macro_qdef directional vs source plus rel = 0.0013563382450265105
```

## 4 训练命令

主要 Linux 命令：

```bash
CUDA_VISIBLE_DEVICES=0 python3 experiments/macro16_coordinate_deeponet/train_coordinate_deeponet.py \
  --prepared runs/macro16_coordinate_deeponet/case031_first6_coordinate_dataset.npz \
  --out-dir runs/macro16_coordinate_deeponet/linux_overfit_38b_state_adplus_b100_e500 \
  --model-style state-linear-residual \
  --epochs 500 \
  --batch-size 6 \
  --basis-dim 64 \
  --hidden-dim 256 \
  --branch-depth 4 \
  --trunk-depth 4 \
  --lr 2e-4 \
  --le-weight 1.0 \
  --b-weight 1.0 \
  --b-loss-source ad-plus-source-fd \
  --b-columns-per-step 48 \
  --fd-column-chunk 12 \
  --fd-step 1e-4 \
  --eval-columns all \
  --eval-every 100 \
  --log-every 100 \
  --cuda
```

并行对照：

```text
linear-residual plus-fd 1000 epoch
state-linear-residual ad-plus-source-fd 100 epoch
state-linear-residual ad-plus-source-fd 500 epoch
state-linear-residual ad 500 epoch
state-linear-residual LE only 500 epoch
```

## 5 结果

| run | epoch | LE_rel | AD_B_rel | FD_B_rel | plus_fd_rel | plus_fd_loss |
|---|---:|---:|---:|---:|---:|---:|
| linear residual plus FD | 1000 | 0.449543 | 0.999104 | 0.999106 | 0.999043 | 1.418173 |
| state residual AD plus source FD | 100 | 0.457490 | 0.999468 | 0.999305 | 1.000067 | 34.226456 |
| state residual AD plus source FD | 500 | 0.457695 | 0.999186 | 0.999058 | 0.999184 | 3.733608 |
| state residual AD only | 500 | 0.417312 | 0.999440 | 0.999348 | 1.004364 | 179.845886 |
| state residual LE only | 500 | 0.151620 | 1.626393 | 1.569099 | 1.664230 | 11665.617188 |

通过标准：

```text
LE_rel < 0.05
AD_B_rel < 0.3
FD_B_rel < 0.3
```

实际结果：

```text
不通过
```

## 6 判断

1. 真实 48 列 source plus FD 已接入
2. plus 列顺序已修正
3. AD 求 B 没断
4. 但 B 相对误差没有下降
5. LE only 可以继续降低 LE，但会让 B 更坏
6. 加强 B 权重不能解决
7. 让 point B prior 看 q 也不能解决

说明当前输入表达仍不够。

## 7 下一步

建议停止沿当前 coordinate-only 输入继续加 epoch。

下一步做 2 件事：

1. 加入标准单元允许的更强点态几何特征，但仍不输入 `J invJ detJ metric`
2. 或先做显式 `B(q,g,x)` direct prior 训练，确认 48 列 B 是否能由这些输入表达

若 direct prior 也失败，说明 `g + x_gp_hat` 信息不足。

若 direct prior 成功，但 LE autograd 失败，说明问题在 LE 参数化和 AD 路径。
