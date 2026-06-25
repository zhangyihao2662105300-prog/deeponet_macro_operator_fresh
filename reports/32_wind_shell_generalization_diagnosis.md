# Gate 32 Wind Shell Generalization Diagnosis

日期：2026-06-26

## 1. 当前状态

Gate 32 已建立只读诊断脚本。

脚本：

```text
scripts/diagnose_macro16_wind_shell_generalization.py
```

本地 Windows 工作区没有 Gate 31 Linux `runs` 数据，因此本文件是任务入口报告，不伪造数值结论。

最终数值报告必须在 Linux 训练目录运行脚本后覆盖本文件。

## 2. 目标

诊断 Gate 31 中风机壳验证集 `case070` 到 `case073` force closure 失败原因。

本任务只做诊断。

不训练网络。

不改模型。

不改 `q48` 顺序。

不改 `LE` 顺序。

不改 128 点规则。

## 3. Gate 31 已知事实

Gate 31 结论：

```text
FAIL
```

有效数据必须来自重新生成的 qdef source128 compact：

```text
runs/gate31_force_residual_16case_qdef/enriched_16case_compact_list.txt
```

旧 compact 训练结果作废，因为缺少：

```text
q48_def_hat
B_macro_qdef
rigid_projection_P
```

Gate 31 已知问题：

1. 16 case 混合训练失败。
2. wind shell `case070` 到 `case073` teacher force closure 很好，但模型 force closure 极差。
3. `case060` 是 q_zero 或近零力特殊样本，不应作为普通 relative force gate 样本。

## 4. Gate 32 必须输出

脚本会统计：

1. train cases 和 wind shell cases 的 `q48_def_hat` 范数。
2. `LE_macro` 范数。
3. `B_macro_qdef` 范数。
4. teacher assembled force 范数。
5. branch 标准化后风机壳是否离群。
6. `case070` 到 `case073` 是否因 q 或 force 太小导致 relative error 爆炸。
7. `case060` 是否应排除普通 relative force gate。

## 5. Linux 命令

在 Linux 训练目录运行：

```bash
cd /home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo
git pull

python3 scripts/run_gate32_wind_shell_generalization_diagnosis.py \
  --source-path-map "D:\\IS-FEM=/home/ydh/IS-FEM" \
  --run-root runs/gate31_force_residual_16case_qdef \
  --run-name f01_lr8e5
```

脚本会自动查找 checkpoint。

如果自动查找失败，先查找：

```bash
find runs/gate31_force_residual_16case_qdef/train/f01_lr8e5 -maxdepth 2 -type f \( -name "*.pt" -o -name "*.pth" \)
```

然后补充：

```bash
--checkpoint 实际文件路径
```

## 6. 当前禁止事项

不要继续扩大训练。

不要把 Gate 30 two-case 成功当作 16-case 放行证据。

不要把 `case060` 的 relative force 数值和正常非零力样本混用。

不要在 Gate 32 诊断前改模型或 loss。

## 7. Gate 32 判定逻辑

如果 wind shell teacher force 通过 `0.02`，但模型 force 失败，则不是老师力闭合坏。

如果 wind shell branch 标准化范数明显大于 train 分布，则优先诊断 normalization 或 split。

如果 wind shell RF 范数远小于 train 分布，则需要给 relative force gate 加 absolute floor 或单独小力样本规则。

如果 `case060` teacher force relative error 本身不通过，则 `case060` 应标记为 near-zero-force special sample。
