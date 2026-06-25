# Macro16 Case031 Mechanics Diagnosis

Role: Mechanics Debug Agent

## Scope

This report diagnoses why `case031` drives the Macro16 source128 force/stiffness
audit above the current `0.02` max-error closure gate. It does not train a
network, change model structure, change q48/LE ordering, or change the standard
128 integration-point rule.

## Inputs

Main audit JSON:

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\force_stiffness_audit_mechanics.json`

Case031 compact:

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\0002_case031_macro16_source128_teacher.npz`

Case031 source compact:

`D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\case031\complete_case031_v2b_local_strain.npz`

Generated local diagnostic JSONs:

- `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\case031_mechanics_diagnosis.json`
- `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\case031_cross_case_geometry_mechanics.json`
- `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\case031_error_grouping.json`
- `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\case031_weight_mode_comparison.json`
- `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\case031_label_B_weight_factorization.json`

Commands used:

```powershell
$env:PYTHONPATH='src;scripts'; py scripts\audit_macro16_force_stiffness.py --compact-list runs\macro16_source128_rigid_preprocess_audit_real_10case\macro16_source128_teacher_compact_list.txt --out runs\macro16_source128_rigid_preprocess_audit_real_10case\force_stiffness_audit_mechanics.json --weight-mode auto --max-force-rel 2.0e-2 --max-stiffness-rel 2.0e-2 --max-macro-stiffness-symmetry-rel 1.0e-10
```

Additional diagnostic snippets were run with:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:PYTHONPATH='src;scripts'; py -
```

Those snippets only read compact/source arrays and wrote the diagnostic JSONs
listed above.

## Summary

`case031` is not a single bad frame. It is a high-amplitude path whose force and
stiffness errors grow with frame index and q magnitude.

Main findings:

- `case031` force rel: `0.03578126940126121`
- `case031` stiffness rel: `0.04256563396475579`
- worst force frame: frame `9`, force rel `0.04339221468264384`
- worst stiffness frame: frame `9`, stiffness rel `0.07612055936290335`
- `LE_macro` equals source `LE128_base`: rel `0.0`, max abs `0.0`
- `B_macro_qraw` equals source `B_LE128_forward`: rel `0.0`, max abs `0.0`
- standard Macro16 source128 assembly and source 128-IP reference-volume
  assembly fail at nearly the same level, so this is not a compact reordering or
  Macro16 field-copy error.

Likely explanation:

`case031` is a large-response case. Reference-volume material assembly remains
reasonable on average, but the last frames move far enough that selected-frame
volume changes reach about `3.85%`. Updating to selected-frame or inferred
volume reduces force error from about `0.036` to about `0.0086`. Stiffness error
still remains about `0.032`, which points to missing nonlinear tangent terms
or a mismatch between material-only `B^T D B dV` stiffness and Abaqus finite
difference response under this large deformation.

## Frame-Level Diagnosis

| Frame | q norm | Force rel | Force max abs diff | Worst F DOF | K rel | K max abs diff | Worst K row,col |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | `0.0385344609` | `0.0199196433` | `0.3463184382` | `21` | `0.0122738997` | `6826.5035125` | `34,9` |
| 1 | `0.0770689219` | `0.0177337970` | `0.7024760262` | `21` | `0.0150360139` | `9299.1186984` | `33,33` |
| 2 | `0.1156033842` | `0.0156114153` | `1.0867158542` | `9` | `0.0160697283` | `8239.5536430` | `34,9` |
| 3 | `0.1541378438` | `0.0144893074` | `1.4366947935` | `9` | `0.0195112496` | `9981.4463562` | `0,24` |
| 4 | `0.1926723061` | `0.0155453364` | `2.5441259553` | `0` | `0.0258026224` | `14617.3922886` | `21,21` |
| 5 | `0.2312067683` | `0.0187684151` | `4.9916701064` | `0` | `0.0336039205` | `19903.4623732` | `21,21` |
| 6 | `0.2697412255` | `0.0235937264` | `8.6247792944` | `0` | `0.0425573092` | `24985.3172992` | `21,21` |
| 7 | `0.3082756876` | `0.0294859665` | `13.6784397290` | `0` | `0.0526605235` | `30291.7080708` | `21,21` |
| 8 | `0.3468101445` | `0.0361183953` | `20.3382945710` | `0` | `0.0638754697` | `36525.9117212` | `16,16` |
| 9 | `0.3853446121` | `0.0433922147` | `28.8537703001` | `0` | `0.0761205594` | `44597.6767845` | `16,16` |

Interpretation:

- The failure is path-amplitude dependent.
- Frames 0-3 are near or under the `0.02` gate.
- Frames 6-9 are the main source of the max-error failure.
- Stiffness error grows more aggressively than force error.

## Force Error Localization

The largest force errors concentrate in boundary x DOFs, especially symmetric
bottom/top x components.

Top DOFs by max absolute force error:

| DOF | Node | Axis | RMS diff | Max abs diff | Frame of max | Signed diff at max |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 0 | x | `12.4105006021` | `28.8537703001` | 9 | `28.8537703001` |
| 24 | 8 | x | `12.0978935376` | `28.0911117291` | 9 | `-28.0911117291` |
| 12 | 4 | x | `11.7646669729` | `27.4976095786` | 9 | `27.4976095786` |
| 36 | 12 | x | `11.4666284315` | `26.7756867497` | 9 | `-26.7756867497` |
| 39 | 13 | x | `7.6825796032` | `18.9245362371` | 9 | `-18.9245362371` |
| 15 | 5 | x | `7.5378496859` | `18.5672378258` | 9 | `18.5672378258` |

Force error grouped by axis:

| Axis | RMS diff | Max abs diff | Axis rel |
|---|---:|---:|---:|
| x | `7.2799990492` | `28.8537703001` | `0.0348030057` |
| y | `0.7683186122` | `3.8806610323` | `0.0243702091` |
| z | `2.3015859993` | `10.6810186177` | `0.0652335210` |

The absolute force residual is dominated by x DOFs. The z-axis relative error is
also high because its reference force norm is smaller.

## Stiffness Error Localization

Worst individual K entries occur in late frames and concentrate in paired
same-axis blocks.

Top K entries:

| Frame | Row node.axis | Col node.axis | Pred | FD ref | Diff |
|---:|---|---|---:|---:|---:|
| 9 | 5.y | 5.y | `7958.8086188` | `52556.4854033` | `-44597.6767845` |
| 9 | 5.y | 13.y | `-7153.7225516` | `-51662.4958254` | `44508.7732738` |
| 9 | 13.y | 13.y | `8276.6171551` | `52779.5553207` | `-44502.9381656` |
| 9 | 13.y | 5.y | `-7153.7225516` | `-51633.0003738` | `44479.2778222` |
| 9 | 1.y | 1.y | `7983.8971612` | `51172.6886034` | `-43188.7914422` |
| 9 | 1.y | 9.y | `-7152.1732862` | `-50252.4524927` | `43100.2792066` |

K error grouped by axis block:

| Block | RMS diff | Max abs diff | Block rel |
|---|---:|---:|---:|
| yy | `5320.9433339` | `44597.6767845` | `0.0918889823` |
| xx | `6488.0661622` | `42321.7126627` | `0.0288163593` |
| zz | `5190.4928155` | `37872.3594687` | `0.0876360101` |
| yx | `1743.5680903` | `9370.9015469` | `0.1081872101` |
| yz | `440.3102847` | `3243.7183047` | `0.1084620375` |

The worst absolute K entries are same-axis yy/xx/zz blocks in late frames. This
is consistent with a tangent mismatch that grows with deformation amplitude,
not with random column-order corruption.

## Geometry, Weights, B, Stress, Material

### Geometry

`case031` has the same reference geometry metrics as `case019` and `case025`:

- span: `[0.0426354, 1.0, 0.998452]`
- aspect min/max: `0.0426354`
- standard source128 det proxy min/max: `0.00124561 / 0.00125591`
- det coefficient of variation: `0.00264749`

Cases `041-050` have more variable det proxy and still lower force/K errors.
So reference geometry distortion alone does not explain the failure.

### Labels and B

Direct field comparison:

- `LE_macro` vs source `LE128_base`: rel `0.0`, max abs `0.0`
- `B_macro_qraw` vs source `B_LE128_forward`: rel `0.0`, max abs `0.0`

This rules out a Macro16 source128 compact copy/reorder bug for `LE_macro` and
`B_macro_qraw`.

### Weights

Weight variants for `case031`:

| Assembly weight | Force rel | Stiffness rel | Weight sum range |
|---|---:|---:|---|
| Macro standard `integration_weight_phys` | `0.0357812694` | `0.04256563396` | `0.0099998801` fixed |
| Source reference `ip_IVOL_abaqus` | `0.0370121642` | `0.04355489906` | `0.0099928621` fixed |
| Source selected-frame `ip_IVOL_abaqus_selected_frames` | `0.0085857252` | `0.03196211043` | `0.0099840736 - 0.0101595623` |
| `IVOL128_inferred_from_DLE` | `0.0086752903` | `0.03197941039` | `0.0099855202 - 0.0101641807` |

Selected-frame Abaqus IVOL changes relative to reference IVOL with frame:

- frame 0: `0.0021382439`
- frame 5: `0.0166564968`
- frame 9: `0.0384994285`

This explains most of the force error: reference-volume assembly is being
compared to an RF response whose effective deformed-frame volume changes by up
to about `3.85%`.

It does not fully explain stiffness: even with selected-frame or inferred
volume, K remains around `0.032`, still above the `0.02` gate.

### Stress and response amplitude

Cross-case comparison shows `case031` is the outlier in response amplitude, not
in B or material matrix:

| Case | q_norm max | LE RMS | Stress RMS | B RMS | Force rel | K rel |
|---|---:|---:|---:|---:|---:|---:|
| case019 | `0.0281675` | `0.000782155` | `137.395` | `5.4991` | `0.00325348` | `0.00411562` |
| case025 | `0.00414115` | `0.000128343` | `21.0661` | `5.49763` | `0.00271410` | `0.00295641` |
| case031 | `0.385345` | `0.0126897` | `2676.22` | `5.52091` | `0.0357813` | `0.0425656` |
| case041-050 max | `0.00749779` | `0.000277439` | `34.927` | `9.17494` | `0.00908940` | `0.00742459` |

Material matrix norm is identical across cases: `589394`.

## Source 128-IP Baseline Comparison

The source 128-IP reference-volume assembly already shows the same failure
scale:

- Macro compact with standard weights: force `0.0357813`, K `0.0425656`
- Source 128-IP with reference `ip_IVOL_abaqus`: force `0.0370122`, K `0.0435549`
- Source 128-IP with inferred volume: force `0.0086753`, K `0.0319794`

So the bad max error is not introduced by the regenerated Macro16 compact. It
is already visible in the teacher/source physical closure when reference-volume
material assembly is compared against the nonlinear Abaqus RF and FD stiffness
for this large-response path.

## Excluding Case031

If `case031` is removed, the remaining 9 cases pass the `0.02` max-error gate:

- force mean rel without case031: `0.006839089392881512`
- force max rel without case031: `0.009089404785732353`
- stiffness mean rel without case031: `0.006520665006697817`
- stiffness max rel without case031: `0.007424585935534821`
- stiffness symmetry max rel without case031: `1.4366806903130055e-16`

This confirms `case031` is the sole gate blocker in the current 10-case set.

## Diagnosis

`case031` is primarily a large-deformation / updated-geometry mechanics closure
case, not a network problem and not a Macro16 compact field contract problem.

Most likely contributors:

1. Force error is largely due to using reference/standard physical weights while
   the Abaqus response reflects selected-frame volume changes. Updating volume
   reduces force rel from about `0.036` to about `0.0086`.
2. K error is only partially explained by volume. Even selected-frame volume
   leaves K rel about `0.032`, suggesting missing nonlinear tangent terms:
   derivative of volume, derivative of B, geometric stiffness/stress stiffness,
   or a mismatch between the material-only `B^T D B dV` audit stiffness and the
   Abaqus finite-difference tangent response.
3. The field data itself is internally consistent: `LE_macro` and
   `B_macro_qraw` exactly match the source labels used for case031.

## Recommended Next Action

Do not train a network on this as an unexplained target yet.

Recommended mechanics debug steps:

1. Add a focused case031 audit mode that can assemble force/K with
   selected-frame `ip_IVOL_abaqus_selected_frames` and
   `IVOL128_inferred_from_DLE`, while keeping the current standard source128
   path unchanged.
2. Split K residual into material stiffness versus finite-difference nonlinear
   tangent residual, explicitly checking whether `dV/dq`, `dB/dq`, or geometric
   stiffness terms are expected under the Abaqus perturbation setup.
3. Re-run case031 with smaller amplitude frames or include only frames 0-3 to
   verify that the source128 material assembly closes below `0.02` in the
   small-response regime.
4. Keep the current 128-point rule unchanged until the case031 mechanics
   mismatch is explained.
