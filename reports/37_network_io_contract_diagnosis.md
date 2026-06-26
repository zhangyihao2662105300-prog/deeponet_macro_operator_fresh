# Gate 37 Network IO Contract Diagnosis

日期：2026-06-26

## 1. 目标

暂停继续训练和继续测试。

检查当前网络输入输出是否写错。

重点检查：

1. 网络实际输入
2. 网络实际输出
3. 训练 loader 是否读错 `q`
4. 训练 loader 是否读错 `LE`
5. 训练 loader 是否读错 `B`
6. 128 点规则是否错
7. 标准化是否导致输入爆炸

本轮不训练。

本轮不改模型。

本轮不改 `q48` 顺序。

本轮不改 `LE` 顺序。

本轮不改 128 点规则。

## 2. 新增脚本

```text
scripts/diagnose_macro16_network_io_contract.py
```

脚本只读。

它检查：

1. compact 字段和 loader 字段是否一致
2. branch 输入实际由哪些列组成
3. point features 实际维度
4. checkpoint 输出 `LE` 形状
5. AD 得到的 `B` 形状
6. AD `B` 是否等于 explicit state B
7. branch 输入标准化最大值
8. point 输入标准化最大值
9. 最大异常列来自哪里

## 3. Linux 诊断

机器：

```text
lab-gpu-ts
```

运行目录：

```text
/home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo
```

命令：

```text
PYTHONPATH=src:scripts CUDA_VISIBLE_DEVICES=0 python3 scripts/diagnose_macro16_network_io_contract.py
```

输入：

```text
runs/gate31_force_residual_16case_qdef/enriched_16case_compact_list.txt
runs/gate36_wind_shell_in_split/train/f01_floor005_train70_72_val73/latest.pt
case 70,71,72,73
```

输出：

```text
runs/gate36_wind_shell_in_split/audits/f01_floor005_train70_72_val73_latest_io_contract.json
```

## 4. 已确认没有错的部分

loader 和 compact 完全一致：

```text
q48_def_hat rel = 0.0
LE_macro rel = 0.0
B_macro_qdef rel = 0.0
X16_hat rel = 0.0
```

128 点规则一致：

```text
macro16_point_xi matches source128 = true
point_count = 128
```

模型输出形状：

```text
LE = [40,128,6]
B = [40,128,6,48]
```

AD 路径和 explicit state B 一致：

```text
AD_J_norm_vs_explicit_state_B_norm rel = 0.0
AD_B_qdef_hat_vs_explicit_state_B_qdef_hat rel = 0.0
```

所以当前没有证据说明：

1. `q48_def_hat` 读成了 `q48_raw`
2. `LE_macro` 读错
3. `B_macro_qdef` 读错
4. 128 点顺序错
5. AD B 路径断掉

## 5. 发现的输入合同问题

当前实际 branch 输入是：

```text
q48_def_hat[48] + X16_hat.flatten[48] + L_ref[1]
```

实际维度：

```text
97
```

这和很多项目文档里的简写：

```text
q48_def_hat + X16_hat
```

不完全一致。

不过本次最严重问题不是这个 `L_ref`，因为 wind shell local `L_ref` 没有爆炸。

真正严重问题是 trunk 输入：

```text
point_features
```

维度：

```text
50
```

最大标准化值：

```text
point_norm_abs_max = 134374.21875
```

对比 branch：

```text
branch_norm_abs_max = 10.595389366149902
```

这说明 Gate 35 只修了 branch `X16_hat` 标准化爆炸，但没有修 trunk `point_features` 标准化爆炸。

## 6. 最大异常列

最大异常来自：

```text
feature = 30
name = ip_invJ_hat_10
case = 73
raw = -0.010862329043447971
mean = 5.563076665993094e-10
std = 8.083641489520232e-08
z = -134374.21875
```

也就是说：

```text
训练 split 里这个 point feature 几乎恒定
case073 里它有正常几何变化
标准化除以极小 std
导致 trunk 输入爆炸
```

这能解释为什么：

1. branch std floor 后仍然失败
2. case073 特别差
3. LE 和 B 在验证几何上一起崩

## 7. 结论

你的判断是对的。

现在不应该继续往下做测试。

当前更像是网络输入标准化合同问题。

结论：

```text
q/LE/B 标签读取没有发现错误
AD B 路径没有发现错误
128 点规则没有发现错误
网络 point_features 输入标准化存在严重爆炸
```

Gate 37 状态：

```text
DIAGNOSIS PASS
TRAINING RELEASE STILL BLOCKED
```

## 8. 下一步

进入 Gate 38。

建议任务：

1. 给 point features 增加 std floor。
2. floor 只作用于 trunk point features。
3. 不改 q48。
4. 不改 LE。
5. 不改 128 点规则。
6. 先做 replay，不训练。
7. 检查 point_norm_abs_max 是否从 `134374` 降到合理范围。

候选 floor：

```text
0.01
0.02
0.05
0.10
```

通过标准：

```text
point_norm_abs_max < 100
最好 < 50
branch_norm_abs_max 仍保持正常
q_std 不变
X16 branch std floor 不变
```

只有 Gate 38 replay 通过后，才允许重新做小训练。
