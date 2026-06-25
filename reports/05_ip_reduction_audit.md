# Gate 05 IP Reduction Audit

Date: 2026-06-25

Role: Gate 05 IP Reduction Agent

## 1. Task Goal

开始 Macro16 source128 积分点降阶审计。

本轮不训练网络，不改模型，不改 `q48` 顺序，不改 `LE` 顺序，不改当前标准 128 点积分规则。Material-only stiffness 只记录为 diagnostic，不作为 Gate 05 通过门槛。

## 2. Data Used

- regular: `D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate04_generality\regular_compact_list.txt`
- lightly_distorted: `D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate04_generality\lightly_distorted_compact_list.txt`
- moderately_distorted: `D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate04_generality\moderately_distorted_compact_list.txt`
- cylindrical_shell: `D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate04_wind_shell_generality\audits\cylindrical_shell\cylindrical_shell_macro16_source128_compact_list.txt`
- conical_shell: `D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate04_wind_shell_generality\audits\conical_shell\conical_shell_macro16_source128_compact_list.txt`
- thickness_varying_shell: `D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate04_wind_shell_generality\audits\thickness_varying_shell\thickness_varying_shell_macro16_source128_compact_list.txt`
- mild_double_curvature_shell: `D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate04_wind_shell_generality\audits\mild_double_curvature_shell\mild_double_curvature_shell_macro16_source128_compact_list.txt`

## 3. Commands Used

```powershell
scripts\run_gate05_ip_reduction_audit.py --out-root runs\gate05_ip_reduction --report reports\05_ip_reduction_audit.md --workers 8 --strict
```

## 4. Candidate Rules

| Rule | Point Count | Rule Kind | Label Mode | Weight Mode | Xi Hash |
|---|---:|---|---|---|---|
| 128 | 128 | source128_subset_with_selected_volume_aggregation | take_source128_at_kept_points | aggregate_source128_selected_frame_volume_to_kept_points | `17e453b87634ff86` |
| 96 | 96 | source128_subset_with_selected_volume_aggregation | take_source128_at_kept_points | aggregate_source128_selected_frame_volume_to_kept_points | `d921799f61a4b682` |
| 64 | 64 | source128_subset_with_selected_volume_aggregation | take_source128_at_kept_points | aggregate_source128_selected_frame_volume_to_kept_points | `68737b30cea463c7` |
| 32 | 32 | source128_subset_with_selected_volume_aggregation | take_source128_at_kept_points | aggregate_source128_selected_frame_volume_to_kept_points | `4f83425c6e3ba894` |
| 18 | 18 | standard18_interpolated_failure_control | linear_parent_interpolation_from_source128 | nearest_aggregate_source128_selected_frame_volume | `7fd329f4e92b21ef` |

说明：96、64、32 是 Gate 05 候选审计点表；128 仍是当前标准规则；18 是失败对照。

## 5. Pass Standard

Gate 05 当前通过标准只看 selected-frame volume 下的恢复力闭合：

```text
force max relative error < 0.02
```

Material-only K 只记录，不作为 full tangent 门槛。

近零反力帧使用 Gate 04 相同口径排除在 relative force 门槛外，阈值为 `force_ref_rms <= 1e-10`；全量结果仍保存在 JSON 明细中。

## 6. Overall Result

| Rule | Data Contract | detJ Positive | Compact Count | Frame Count | Effective Force Count | Near-Zero Excluded | Force Mean | Force Max | Force Pass | Material-only K Mean | Material-only K Max |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 128 | PASS | PASS | 35 | 350 | 31 | 4 | `0.000479951477` | `0.00858572515` | PASS | `0.00208149667` | `0.0319621104` |
| 96 | PASS | PASS | 35 | 350 | 31 | 4 | `0.248532656` | `0.326256518` | FAIL | `0.258278402` | `0.260756542` |
| 64 | PASS | PASS | 35 | 350 | 31 | 4 | `0.379246539` | `0.633171346` | FAIL | `0.411246577` | `0.423485654` |
| 32 | PASS | PASS | 35 | 350 | 31 | 4 | `0.741043294` | `0.994885929` | FAIL | `0.668406936` | `0.675654636` |
| 18 | PASS | PASS | 35 | 350 | 31 | 4 | `0.395497981` | `0.660930586` | FAIL | `0.485509349` | `0.487160029` |

## 7. Per-Class Result

| Class | Rule | Geometry Count | Frame Count | Effective Force Count | Near-Zero Excluded | detJ Positive | Data Contract | Force Mean | Force Max | Force Pass | Material-only K Mean | Material-only K Max |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---:|---:|
| regular | 128 | 3 | 30 | 3 | 0 | PASS | PASS | `0.0035065059` | `0.00858572515` | PASS | `0.0114437671` | `0.0319621104` |
| regular | 96 | 3 | 30 | 3 | 0 | PASS | PASS | `0.231893926` | `0.323922891` | FAIL | `0.255055305` | `0.255256971` |
| regular | 64 | 3 | 30 | 3 | 0 | PASS | PASS | `0.344534072` | `0.360138946` | FAIL | `0.399330949` | `0.404003196` |
| regular | 32 | 3 | 30 | 3 | 0 | PASS | PASS | `0.755324848` | `0.924572293` | FAIL | `0.649509682` | `0.659573046` |
| regular | 18 | 3 | 30 | 3 | 0 | PASS | PASS | `0.239170168` | `0.286420503` | FAIL | `0.481549773` | `0.483418004` |
| lightly_distorted | 128 | 7 | 70 | 7 | 0 | PASS | PASS | `0.000315473517` | `0.000575427184` | PASS | `0.000330240238` | `0.000472801616` |
| lightly_distorted | 96 | 7 | 70 | 7 | 0 | PASS | PASS | `0.261632972` | `0.294038512` | FAIL | `0.259171292` | `0.259252073` |
| lightly_distorted | 64 | 7 | 70 | 7 | 0 | PASS | PASS | `0.375294516` | `0.404745101` | FAIL | `0.411706164` | `0.411778349` |
| lightly_distorted | 32 | 7 | 70 | 7 | 0 | PASS | PASS | `0.746445334` | `0.801840091` | FAIL | `0.664047157` | `0.664344168` |
| lightly_distorted | 18 | 7 | 70 | 7 | 0 | PASS | PASS | `0.316282658` | `0.426280899` | FAIL | `0.484573547` | `0.48479399` |
| moderately_distorted | 128 | 21 | 210 | 17 | 4 | PASS | PASS | `0.000120323364` | `0.000248710625` | PASS | `0.0017063919` | `0.00705856047` |
| moderately_distorted | 96 | 21 | 210 | 17 | 4 | PASS | PASS | `0.251814969` | `0.326256518` | FAIL | `0.259247714` | `0.260756542` |
| moderately_distorted | 64 | 21 | 210 | 17 | 4 | PASS | PASS | `0.399004193` | `0.633171346` | FAIL | `0.4144569` | `0.417159472` |
| moderately_distorted | 32 | 21 | 210 | 17 | 4 | PASS | PASS | `0.720302516` | `0.994885929` | FAIL | `0.673396344` | `0.675654636` |
| moderately_distorted | 18 | 21 | 210 | 17 | 4 | PASS | PASS | `0.470692029` | `0.660930586` | FAIL | `0.486743373` | `0.487160029` |
| cylindrical_shell | 128 | 1 | 10 | 1 | 0 | PASS | PASS | `2.6251543e-05` | `2.6251543e-05` | PASS | `8.7749572e-05` | `8.7749572e-05` |
| cylindrical_shell | 96 | 1 | 10 | 1 | 0 | PASS | PASS | `0.222992718` | `0.222992718` | FAIL | `0.255516953` | `0.255516953` |
| cylindrical_shell | 64 | 1 | 10 | 1 | 0 | PASS | PASS | `0.285125847` | `0.285125847` | FAIL | `0.404921455` | `0.404921455` |
| cylindrical_shell | 32 | 1 | 10 | 1 | 0 | PASS | PASS | `0.765721966` | `0.765721966` | FAIL | `0.661031786` | `0.661031786` |
| cylindrical_shell | 18 | 1 | 10 | 1 | 0 | PASS | PASS | `0.298052939` | `0.298052939` | FAIL | `0.483280176` | `0.483280176` |
| conical_shell | 128 | 1 | 10 | 1 | 0 | PASS | PASS | `2.79727003e-05` | `2.79727003e-05` | PASS | `0.000109797586` | `0.000109797586` |
| conical_shell | 96 | 1 | 10 | 1 | 0 | PASS | PASS | `0.238235978` | `0.238235978` | FAIL | `0.254423481` | `0.254423481` |
| conical_shell | 64 | 1 | 10 | 1 | 0 | PASS | PASS | `0.322499005` | `0.322499005` | FAIL | `0.404030593` | `0.404030593` |
| conical_shell | 32 | 1 | 10 | 1 | 0 | PASS | PASS | `0.761357564` | `0.761357564` | FAIL | `0.659987345` | `0.659987345` |
| conical_shell | 18 | 1 | 10 | 1 | 0 | PASS | PASS | `0.302567291` | `0.302567291` | FAIL | `0.483369547` | `0.483369547` |
| thickness_varying_shell | 128 | 1 | 10 | 1 | 0 | PASS | PASS | `2.55194417e-05` | `2.55194417e-05` | PASS | `9.04030326e-05` | `9.04030326e-05` |
| thickness_varying_shell | 96 | 1 | 10 | 1 | 0 | PASS | PASS | `0.219920269` | `0.219920269` | FAIL | `0.259223069` | `0.259223069` |
| thickness_varying_shell | 64 | 1 | 10 | 1 | 0 | PASS | PASS | `0.290371368` | `0.290371368` | FAIL | `0.423485654` | `0.423485654` |
| thickness_varying_shell | 32 | 1 | 10 | 1 | 0 | PASS | PASS | `0.771938799` | `0.771938799` | FAIL | `0.669100149` | `0.669100149` |
| thickness_varying_shell | 18 | 1 | 10 | 1 | 0 | PASS | PASS | `0.293909848` | `0.293909848` | FAIL | `0.485641647` | `0.485641647` |
| mild_double_curvature_shell | 128 | 1 | 10 | 1 | 0 | PASS | PASS | `2.54226007e-05` | `2.54226007e-05` | PASS | `8.72205944e-05` | `8.72205944e-05` |
| mild_double_curvature_shell | 96 | 1 | 10 | 1 | 0 | PASS | PASS | `0.215396322` | `0.215396322` | FAIL | `0.247013619` | `0.247013619` |
| mild_double_curvature_shell | 64 | 1 | 10 | 1 | 0 | PASS | PASS | `0.414911369` | `0.414911369` | FAIL | `0.377661603` | `0.377661603` |
| mild_double_curvature_shell | 32 | 1 | 10 | 1 | 0 | PASS | PASS | `0.937089136` | `0.937089136` | FAIL | `0.665941128` | `0.665941128` |
| mild_double_curvature_shell | 18 | 1 | 10 | 1 | 0 | PASS | PASS | `0.432653742` | `0.432653742` | FAIL | `0.482260865` | `0.482260865` |

## 8. Gate Result

Gate 05 force-only reduction audit: no reduced candidate passed the current force max threshold. Keep 128 points as the active rule.

## 9. Unresolved Issues

1. Material-only K 仍只是 diagnostic，不代表完整 Abaqus tangent。
2. 96、64、32 当前是候选降阶点表，不能替代 128 标准规则，除非后续正式决策采用。
3. 18 点继续作为失败对照，不阻塞。

## 10. Recommended Next Action

Keep 128 points and avoid network training; if reduction is still desired, design a new candidate rule instead of changing the model.

