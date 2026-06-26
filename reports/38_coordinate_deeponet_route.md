# 38 Coordinate DeepONet Route

## 1 任务目标

新开独立实验代码，验证新的 Macro16 网络输入形式：

1. Branch 输入为 `q48_def_hat[48] + geometry_g[14]`
2. Trunk 输入只为无量纲物理高斯点坐标 `x_gp_hat[3]`
3. 输出为单个高斯点 `LE[6]`
4. `B` 不直接输出，由 `dLE / d(q48_def_hat)` 自动微分得到
5. 保持 128 点规则，不改 `q48` 顺序，不改 `LE` 顺序

本任务没有改旧 2000 轮源码，没有改当前 `src` 主源码，没有启动大训练。

## 2 读取文件

已读取：

1. `D:\IS-FEM\deeponet_2000epoch_remote_snapshot_20260624\code\src\macro_deeponet\train_true176_generic_sobolev.py`
2. `D:\IS-FEM\deeponet_2000epoch_remote_snapshot_20260624\code\src\macro_deeponet\models.py`
3. `D:\IS-FEM\deeponet_2000epoch_remote_snapshot_20260624\run\config.json`
4. `D:\IS-FEM\deeponet_macro_operator_fresh\src\macro_deeponet\train_macro16_boundary_sobolev.py`
5. `D:\IS-FEM\deeponet_macro_operator_fresh\src\macro_deeponet\macro16_geometry.py`

旧 2000 轮核心思路是：

```text
LE_norm = B_base_norm(point_features) @ q48_norm + DeepONet_residual
```

旧 trunk 使用完整 `point_features`，包含坐标，雅可比，逆雅可比，行列式，度量等信息。

本实验刻意不输入这些，只输入 `x_gp_hat`。

## 3 新增文件

1. `experiments/macro16_coordinate_deeponet/__init__.py`
2. `experiments/macro16_coordinate_deeponet/coordinate_deeponet_model.py`
3. `experiments/macro16_coordinate_deeponet/prepare_coordinate_dataset.py`
4. `experiments/macro16_coordinate_deeponet/train_coordinate_deeponet.py`

输出数据和训练 smoke：

1. `runs/macro16_coordinate_deeponet/case031_first6_coordinate_dataset.npz`
2. `runs/macro16_coordinate_deeponet/case031_first6_coordinate_dataset_summary.json`
3. `runs/macro16_coordinate_deeponet/tiny_overfit_case031_first6`
4. `runs/macro16_coordinate_deeponet/tiny_overfit_case031_first6_leonly`
5. `runs/macro16_coordinate_deeponet/tiny_overfit_case031_first6_plusfd`

## 4 数据输入

使用 compact：

```text
runs/gate08_training_overfit_sanity/case031_first6_overfit_case_split_macro16_source128_teacher.npz
```

数据形状：

1. `q48_def_hat`: `[6, 48]`
2. `geometry_g`: `[6, 14]`
3. `x_gp_hat`: `[6, 128, 3]`
4. `LE_macro`: `[6, 128, 6]`
5. `B_macro_qdef`: `[6, 128, 6, 48]`
6. `q48_def_hat_plus`: `[6, 48, 48]`
7. `LE_macro_plus`: `[6, 48, 128, 6]`

## 5 几何参数

第一版 `geometry_g` 为 14 维：

1. `length_scale_hat`
2. `width_scale_hat`
3. `thickness_scale_hat`
4. `length_width_ratio`
5. `thickness_length_ratio`
6. `in_plane_angle_over_pi`
7. `trapezoid_distortion`
8. `curvature_r`
9. `curvature_s`
10. `twist_curvature`
11. `thickness_mean_hat`
12. `thickness_gradient_r`
13. `thickness_gradient_s`
14. `warping_rms_hat`

这些参数只从 `X16_raw` 和 `L_ref` 提取。

## 6 坐标无量纲

Trunk 输入只使用：

```text
x_gp_hat = (x_gp_raw - X_center) / L_ref
```

当前 compact 没有显式 `ip_xyz`，所以 `x_gp_raw` 由：

```text
X16_raw + macro16_point_xi + Macro16 等参映射
```

内部重建。

注意：

1. `macro16_point_xi` 只用于数据准备阶段重建物理坐标
2. 网络 trunk 不输入父坐标
3. 网络 trunk 不输入 `J`
4. 网络 trunk 不输入 `invJ`
5. 网络 trunk 不输入 `detJ`
6. 网络 trunk 不输入 `metric`
7. 网络 trunk 不输入旧 `point_features`

## 7 扰动和无量纲

Abaqus 48 列前向扰动原始定义在 `q48_raw` 坐标：

```text
q48_raw_plus_j = q48_raw + delta_raw * e_j
```

网络输入是：

```text
q48_def_hat = q48_def_raw / L_ref
```

所以 raw 扰动不能直接作为 qdef_hat 第 j 列。

本实验使用：

```text
q48_def_hat_plus_j
=
q48_def_hat + rigid_projection_P[:, j] * delta_raw / L_ref
```

然后用真实 source plus 数据约束：

```text
模型 LE(q48_def_hat_plus_j) - 模型 LE(q48_def_hat)
------------------------------------------------------
                    delta_raw

对齐

LE128_plus_j - LE128_base
-------------------------
       delta_raw
```

已确认：

1. `delta_raw = 1e-6`
2. source plus slope RMS 约 `5.5086396`
3. `B_macro_qdef` RMS 约 `5.5086379`

这说明 plus 差分数据和当前 qdef_hat B 标签量级一致。

## 8 模型

实现了两个独立实验模型：

1. `plain`

```text
Branch(q48_def_hat, g) dot Trunk(x_gp_hat) -> LE
```

2. `linear-residual`

```text
LE = LE0(g, x_gp_hat)
   + B_prior(g, x_gp_hat) @ q48_def_hat
   + DeepONet residual
```

两者都只输出 `LE`。

`B` 仍然通过：

```text
dLE / d(q48_def_hat)
```

得到。

## 9 验证命令

编译：

```powershell
py -3 -m py_compile experiments\macro16_coordinate_deeponet\coordinate_deeponet_model.py experiments\macro16_coordinate_deeponet\prepare_coordinate_dataset.py experiments\macro16_coordinate_deeponet\train_coordinate_deeponet.py
```

数据准备：

```powershell
py -3 experiments\macro16_coordinate_deeponet\prepare_coordinate_dataset.py --compact-list runs\gate08_training_overfit_sanity\overfit_case031_first6_compact_list.txt --out runs\macro16_coordinate_deeponet\case031_first6_coordinate_dataset.npz --summary runs\macro16_coordinate_deeponet\case031_first6_coordinate_dataset_summary.json
```

plus 前向差分 smoke：

```powershell
py -3 experiments\macro16_coordinate_deeponet\train_coordinate_deeponet.py --prepared runs\macro16_coordinate_deeponet\case031_first6_coordinate_dataset.npz --out-dir runs\macro16_coordinate_deeponet\tiny_overfit_case031_first6_plusfd --model-style linear-residual --epochs 40 --batch-size 6 --hidden-dim 192 --basis-dim 64 --branch-depth 4 --trunk-depth 4 --lr 2e-4 --le-weight 1.0 --b-weight 0.05 --b-loss-source plus-fd --b-columns-per-step 48 --fd-column-chunk 12 --fd-step 1e-4 --eval-columns all --eval-every 10 --log-every 10 --cuda
```

## 10 Smoke 结果

1. plain AD_B smoke

```text
train_LE_rel: 0.4749 -> 0.4101
train_AD_B_rel: 1.0001 -> 1.0003
```

2. LE only 对照

```text
train_LE_rel: 0.4731 -> 0.1554
train_AD_B_rel: 1.0000 -> 1.2497
```

结论：

```text
坐标 trunk 能学 LE 值场，但只学 LE 会让 B 漂移。
```

3. plus-fd 48 列真实前向差分约束

```text
source plus fd loss: 705.83 -> 31.31
train_LE_rel: 0.5353 -> 0.4563
train_AD_B_rel: 1.0167 -> 1.0000
train_FD_B_rel: 1.0167 -> 1.0000
```

结论：

```text
48 列前向差分数据约束已接入并能优化自身损失。
但当前短 smoke 还没有把 AD_B_rel 压下来。
```

## 11 和旧 2000 轮区别

1. 旧 2000 轮 branch 主要是 `q48_raw + X_keep`
2. 本实验 branch 是 `q48_def_hat + geometry_g`
3. 旧 2000 轮 trunk 是完整 `point_features`
4. 本实验 trunk 只有 `x_gp_hat`
5. 旧 2000 轮使用 `B_base(point_features) @ q`
6. 本实验的 linear-residual 只允许 `B_prior(g, x_gp_hat) @ q`
7. 旧 2000 轮可看到 `J/invJ/detJ/metric`
8. 本实验禁止这些几何微分特征进入网络
9. 旧 2000 轮直接针对完整旧点特征训练
10. 本实验先验证坐标输入路线是否值得后续 Linux 训练

## 12 当前判断

1. 代码可编译
2. 数据读取通过
3. 几何参数提取通过
4. 坐标无量纲通过
5. 48 列 source plus 前向差分数据约束已接入
6. LE 值场能明显下降
7. AD_B 通路没有断
8. 但 AD_B_rel 还没有下降

## 13 是否继续 Linux 训练

不建议直接做 Linux 大训练。

建议先做 Linux 小实验：

1. 使用 `--b-loss-source plus-fd`
2. 使用 `--model-style linear-residual`
3. 固定 `case031 first6`
4. 先跑 500 到 2000 epoch 小样本
5. 同时记录 `LE_rel`、`AD_B_rel`、`FD_B_rel`

进入更大训练的最低条件：

1. 极小样本 `LE_rel < 0.05`
2. 极小样本 `AD_B_rel < 0.3`
3. 极小样本 `FD_B_rel < 0.3`

如果 Linux 小样本仍然不能把 `AD_B_rel` 压下来，说明仅用 `g + x_gp_hat` 的 coordinate trunk 信息不足，需要加入更强的标准单元几何表达，但仍不能回到旧 `point_features/J/invJ/detJ/metric` 路线。
