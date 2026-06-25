# Macro16 Force And Stiffness Audit

Role: Mechanics Agent

## Scope

This audit uses the regenerated Macro16 source128 compacts that passed the
Dataset Agent data-contract audit. It does not train a network, change model
structure, change q48/LE ordering, or change the standard 128 integration-point
rule.

## Data

Compact list:

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\macro16_source128_teacher_compact_list.txt`

Compact count: 10

Frame count: 100

Dataset gate:

`D:\IS-FEM\deeponet_macro_operator_fresh\reports\macro16_source128_data_audit.md`

## Command

```powershell
$env:PYTHONPATH='src;scripts'; py scripts\audit_macro16_force_stiffness.py --compact-list runs\macro16_source128_rigid_preprocess_audit_real_10case\macro16_source128_teacher_compact_list.txt --out runs\macro16_source128_rigid_preprocess_audit_real_10case\force_stiffness_audit_mechanics.json --weight-mode auto --max-force-rel 2.0e-2 --max-stiffness-rel 2.0e-2 --max-macro-stiffness-symmetry-rel 1.0e-10
```

Output JSON:

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\force_stiffness_audit_mechanics.json`

The audit command was run without `--strict` so the JSON report could be
written even when the closure gate failed. The JSON field `strict_pass` is
`false`.

## Result

Status: FAIL current closure gate

Current closure gate:

- recovery-force max relative error <= `0.02`
- stiffness max relative error <= `0.02`
- macro stiffness symmetry relative error <= `1.0e-10`

Observed:

- recovery-force mean relative error: `0.00973330739371948`
- recovery-force max relative error: `0.03578126940126121`
- stiffness mean relative error: `0.010125161902503614`
- stiffness max relative error: `0.04256563396475579`
- macro stiffness symmetry mean relative error: `1.1844190068589181e-16`
- macro stiffness symmetry max relative error: `1.4366806903130055e-16`

The average force and stiffness errors are near the current reference values,
but the worst force and stiffness errors exceed the `0.02` closure threshold.
Therefore the route does not pass the current force/stiffness closure gate on
this 10-case set.

## Worst Case

Worst recovery-force case:

- case: `case031`
- compact: `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\0002_case031_macro16_source128_teacher.npz`
- source geometry/data: `D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\case031\complete_case031_v2b_local_strain.npz`
- recovery-force relative error: `0.03578126940126121`

Worst stiffness case:

- case: `case031`
- compact: `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\0002_case031_macro16_source128_teacher.npz`
- source geometry/data: `D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\case031\complete_case031_v2b_local_strain.npz`
- stiffness relative error: `0.04256563396475579`

The worst geometry in this checked set is therefore `case031`.

## Physical Weights

Physical assembly used `weight_mode=as-stored` after `--weight-mode auto`.

For all checked compacts:

- `integration_weight_hat_source`: `integration_weight_hat`
- `integration_weight_phys_source`: `integration_weight_phys`
- physical assembly used `integration_weight_phys`

The audit did not use dimensionless `integration_weight_hat` as the physical
assembly weight.

## Per-Case Summary

| Case | Force rel | Stiffness rel | K symmetry rel | Weight mode | Physical weight source |
|---|---:|---:|---:|---|---|
| case019 | `0.00325348313827` | `0.00411561819264` | `1.16449653392e-16` | `as-stored` | `integration_weight_phys` |
| case025 | `0.00271410131937` | `0.00295640883334` | `1.01568884971e-16` | `as-stored` | `integration_weight_phys` |
| case031 | `0.0357812694013` | `0.0425656339648` | `1.11821801628e-16` | `as-stored` | `integration_weight_phys` |
| case041 | `0.00908940478573` | `0.0073214083143` | `1.26508258236e-16` | `as-stored` | `integration_weight_phys` |
| case043 | `0.00810841543023` | `0.00735182432999` | `1.17076629596e-16` | `as-stored` | `integration_weight_phys` |
| case044 | `0.00853281347508` | `0.00731544546301` | `1.08567583648e-16` | `as-stored` | `integration_weight_phys` |
| case045 | `0.00800151301371` | `0.00741212317719` | `1.43668069031e-16` | `as-stored` | `integration_weight_phys` |
| case046 | `0.00692044929796` | `0.00742458593553` | `1.24033033558e-16` | `as-stored` | `integration_weight_phys` |
| case049 | `0.00747788857794` | `0.00739472575385` | `1.18100712017e-16` | `as-stored` | `integration_weight_phys` |
| case050 | `0.00745373549764` | `0.00739384506043` | `1.16624380781e-16` | `as-stored` | `integration_weight_phys` |

## Threshold Checks

| Check | Value | Limit | Pass |
|---|---:|---:|---|
| force max relative error | `0.03578126940126121` | `0.02` | no |
| stiffness max relative error | `0.04256563396475579` | `0.02` | no |
| macro stiffness symmetry relative error | `1.4366806903130055e-16` | `1.0e-10` | yes |

## Unresolved Risks

- The current 10-case / 100-frame set covers the trusted regenerated regular
  and lightly distorted compacts, but does not cover medium-distortion or strong
  non-inverted geometries.
- `case031` dominates both worst force and worst stiffness error. That case
  needs focused mechanical diagnosis before declaring the source128 route closed
  under the current `0.02` max-error gate.
- `B_macro_qdef` remains based on a small-rotation linear projection
  approximation for training labels, but this force/stiffness audit assembles
  physical quantities from `B_macro_qraw` and `integration_weight_phys`.

## Recommended Next Action

Run a focused mechanics diagnosis on `case031`, comparing source 128-IP force
closure, Macro16 source128 assembly, physical weight differences, and
frame/direction-level stiffness residuals. After that, repeat this gate on
medium-distortion and strong non-inverted regenerated source128 compacts.
