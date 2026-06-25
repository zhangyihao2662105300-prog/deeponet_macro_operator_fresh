# Gate 34 Branch Normalization Replay

日期：2026-06-26

## 1. 目标

验证 Gate 33 发现的 `X16_hat` 几何标准化离群能否通过 std floor 缓解。

本任务只做 normalization replay。

不训练网络。

不改模型。

不改 `q48` 顺序。

不改 `LE` 顺序。

不改 128 点规则。

## 2. 使用脚本

```text
scripts/replay_macro16_branch_normalization.py
```

Linux 运行目录：

```text
/home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo
```

命令：

```bash
python3 scripts/replay_macro16_branch_normalization.py \
  --compact-list runs/gate31_force_residual_16case_qdef/enriched_16case_compact_list.txt \
  --checkpoint runs/gate31_force_residual_16case_qdef/train/f01_lr8e5/best.pt \
  --out runs/gate31_force_residual_16case_qdef/audits/gate34_branch_normalization_replay.json
```

## 3. Replay 方案

branch 定义不变：

```text
q48_def_hat 48列
X16_hat 48列
L_ref 1列
```

只在 replay 中对 `X16_hat` 的 std 设置下限。

候选：

```text
0
0.01
0.02
0.05
0.1
0.2
```

`q48_def_hat` std 不改。

`L_ref` std 不改。

## 4. 判据

Replay 通过标准：

```text
wind branch max <= 100
wind max / train p99 <= 10
```

这只是输入尺度判据，不是 force closure 判据。

## 5. 结果

| X16 std floor | wind branch max | wind max / train p99 | pass max <= 100 | pass ratio <= 10 |
|---:|---:|---:|---|---|
| 0 | 2254634.899 | 78705.657 | false | false |
| 0.01 | 165.691 | 5.789 | false | true |
| 0.02 | 109.640 | 3.837 | false | true |
| 0.05 | 48.563 | 1.703 | true | true |
| 0.1 | 24.386 | 0.856 | true | true |
| 0.2 | 12.398 | 0.435 | true | true |

最小通过候选：

```text
X16_hat std floor = 0.05
```

对应 wind shell：

```text
branch max = 48.56311643987205
max abs column = 10.518003478646278
wind max / train p99 = 1.7031650932134026
```

原始无 floor：

```text
branch max = 2254634.8991423855
max abs column = 1525076.1154592873
wind max / train p99 = 78705.65689448109
```

## 6. 解释

Gate 33 已证明最大异常来自 `X16_hat_z` 几何列。

这些列在训练 split 中标准差接近 `1e-08`。

`X16_hat std floor = 0.05` 后，wind shell 输入不再百万级离群。

这说明 Gate 31 的 wind shell 失败有明确的输入标准化机制解释。

## 7. 结论

Gate 34 结论：

```text
normalization replay PASS
推荐最小候选 X16_hat std floor = 0.05
```

但这还不是训练通过。

还没有证明 force closure 修复。

## 8. 下一步

进入 Gate 35。

任务：

1. 给训练脚本增加 guarded branch std floor 参数。
2. 默认保持旧行为。
3. 新参数只在显式传入时启用。
4. 先跑 Linux 小训练，不直接扩大数据规模。
5. 训练后必须跑 selected-frame force audit。

建议参数：

```text
--branch-x16-std-floor 0.05
```

禁止事项：

```text
不要改 q48 顺序
不要改 LE 顺序
不要改 128 点规则
不要把 replay 当训练通过
```
