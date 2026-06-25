# Gate 32 Wind Shell Generalization Diagnosis

日期：2026-06-26

## 1. 目标

诊断 Gate 31 中 `case070` 到 `case073` 风机壳验证集 force closure 失败原因。

不训练网络。

不改模型。

不改 `q48` 顺序。

不改 `LE` 顺序。

不改 128 点规则。

## 2. 数据

compact list：

```text
/home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo/runs/gate31_force_residual_16case_qdef/enriched_16case_compact_list.txt
```

train cases：

```text
19,25,31,41,43,44,45,46,49,50,60,61
```

wind shell cases：

```text
70,71,72,73
```

branch normalization：

```text
/home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo/runs/gate31_force_residual_16case_qdef/train/f01_lr8e5/best.pt
```

## 3. 分组统计

| group | cases | frames | q p50 | LE p50 | B p50 | branch max | RF p50 | teacher force rel |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | 19,25,31,41,43,44,45,46,49,50,60,61 | 120 | 0.00169088 | 0.0040895 | 1760.54 | 33.0432 | 6.44181 | 0.00856901 |
| wind_shell | 70,71,72,73 | 40 | 0.000109107 | 0.00115624 | 535.149 | 2.25463e+06 | 2.07127 | 2.56391e-05 |

## 4. 逐 Case 统计

| case | frames | q p50 | LE p50 | B p50 | branch max | RF p50 | teacher force rel |
|---|---:|---:|---:|---:|---:|---:|---:|
| 019 | 10 | 0.0151985 | 0.0191803 | 1055.82 | 6.22575 | 21.4034 | 0.00160548 |
| 025 | 10 | 0.00218128 | 0.00315252 | 1055.54 | 7.14224 | 3.44723 | 0.000327393 |
| 031 | 10 | 0.208091 | 0.282465 | 1059.38 | 33.0432 | 543.802 | 0.00858535 |
| 041 | 10 | 0.00108749 | 0.0022921 | 1760.6 | 5.16064 | 3.40987 | 0.000225811 |
| 043 | 10 | 0.00276282 | 0.0050611 | 1761.21 | 10.3844 | 10.1685 | 0.000296671 |
| 044 | 10 | 0.00105632 | 0.00230044 | 1760.62 | 5.16662 | 3.4906 | 0.000211632 |
| 045 | 10 | 0.00227319 | 0.00462515 | 1760.17 | 8.87261 | 8.61142 | 0.000286828 |
| 046 | 10 | 0.00107376 | 0.00302429 | 1760.61 | 4.20022 | 5.36431 | 0.000162025 |
| 049 | 10 | 0.00299191 | 0.00542066 | 1761.21 | 6.10081 | 9.28603 | 0.000440265 |
| 050 | 10 | 0.00388438 | 0.0068074 | 1761.59 | 9.8921 | 11.627 | 0.000575362 |
| 060 | 10 | 0 | 1.33749e-14 | 566.467 | 10.6888 | 1.22397e-11 | 0.826993 |
| 061 | 10 | 0.000409282 | 0.00848767 | 608.741 | 19.5785 | 35.9933 | 1.84304e-05 |
| 070 | 10 | 8.97888e-05 | 0.00096745 | 536.882 | 3.09887 | 2.2753 | 2.56091e-05 |
| 071 | 10 | 9.55489e-05 | 0.00119287 | 581.504 | 18.1749 | 2.74041 | 2.6548e-05 |
| 072 | 10 | 8.95528e-05 | 0.000964905 | 495.526 | 3.38797 | 2.36188 | 2.49487e-05 |
| 073 | 10 | 0.000225664 | 0.00170988 | 533.414 | 2.25463e+06 | 1.4259 | 2.41231e-05 |

## 5. Gate 31 Force Audit 对照

| case | model force rel | teacher force rel | json |
|---|---:|---:|---|
| 019 | 2.85914 | 0.00161182 | /home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo/runs/gate31_force_residual_16case_qdef/audits/f01_lr8e5_case19.json |
| 025 | 20.0507 | 0.000328068 | /home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo/runs/gate31_force_residual_16case_qdef/audits/f01_lr8e5_case25.json |
| 031 | 1.0278 | 0.0103085 | /home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo/runs/gate31_force_residual_16case_qdef/audits/f01_lr8e5_case31.json |
| 041 | 11.1843 | 0.000225604 | /home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo/runs/gate31_force_residual_16case_qdef/audits/f01_lr8e5_case41.json |
| 043 | 3.32268 | 0.000298642 | /home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo/runs/gate31_force_residual_16case_qdef/audits/f01_lr8e5_case43.json |
| 044 | 10.8819 | 0.00021144 | /home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo/runs/gate31_force_residual_16case_qdef/audits/f01_lr8e5_case44.json |
| 045 | 5.00738 | 0.000286018 | /home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo/runs/gate31_force_residual_16case_qdef/audits/f01_lr8e5_case45.json |
| 046 | 7.65603 | 0.000162083 | /home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo/runs/gate31_force_residual_16case_qdef/audits/f01_lr8e5_case46.json |
| 049 | 4.01248 | 0.000440409 | /home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo/runs/gate31_force_residual_16case_qdef/audits/f01_lr8e5_case49.json |
| 050 | 2.92299 | 0.000575513 | /home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo/runs/gate31_force_residual_16case_qdef/audits/f01_lr8e5_case50.json |
| 060 | 8.44726e+12 | 0.812668 | /home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo/runs/gate31_force_residual_16case_qdef/audits/f01_lr8e5_case60.json |
| 061 | 2.75058 | 1.83974e-05 | /home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo/runs/gate31_force_residual_16case_qdef/audits/f01_lr8e5_case61.json |
| 070 | 56.7939 | 2.56137e-05 | /home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo/runs/gate31_force_residual_16case_qdef/audits/f01_lr8e5_case70.json |
| 071 | 48.2835 | 2.65392e-05 | /home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo/runs/gate31_force_residual_16case_qdef/audits/f01_lr8e5_case71.json |
| 072 | 62.0157 | 2.49434e-05 | /home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo/runs/gate31_force_residual_16case_qdef/audits/f01_lr8e5_case72.json |
| 073 | 93.8086 | 2.41213e-05 | /home/ydh/IS-FEM/gate31_force_residual_16case_qdef/repo/runs/gate31_force_residual_16case_qdef/audits/f01_lr8e5_case73.json |

## 6. 诊断标记

```text
data_bad_evidence = False
wind_teacher_force_passes_0p02 = True
wind_branch_max_over_train_p99 = 78705.7
wind_rf_p50_over_train_rf_p50 = 0.321536
wind_branch_outlier_flag = True
wind_force_small_flag = False
wind_model_force_failed_in_gate31 = True
case060_invalid_relative_gate_flag = True
```

## 7. 当前结论

Gate 32 结论：

```text
diagnosis PASS
Gate 31 失败不是数据坏
主要问题是 wind shell branch 标准化离群
```

依据：

1. wind shell teacher force rel 是 `2.56391e-05`，通过 `0.02`。
2. wind shell model force 在 Gate 31 中失败，case070 到 case073 全部很差。
3. wind branch max 是 `2.25463e+06`，train branch max 只有 `33.0432`。
4. `wind_branch_max_over_train_p99 = 78705.7`。
5. wind RF p50 不是近零，`wind_rf_p50_over_train_rf_p50 = 0.321536`，所以不是小力范数导致的主失败。
6. `case060` 的 teacher force rel 是 `0.826993`，应标记为 near-zero-force special sample，不进普通 relative force gate。

当前判断：

```text
不是 q48 顺序错误
不是 LE 顺序错误
不是 128 点规则错误
不是 teacher force 数据坏
是 wind shell 输入标准化或数据分布离群问题
```

## 8. 下一步

不要扩大训练。

进入 Gate 33。

目标是查清楚 branch 离群来自：

1. `q48_def_hat`
2. `X16_hat`
3. `L_ref`
4. train split 中某些列标准差过小
5. `case073` 几何尺度或厚度字段异常
