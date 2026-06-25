# Gate 03a Volume Convention Audit

Role: Volume Convention Audit Agent

## Scope

This audit isolates the effect of the physical integration volume used in
`F_int = integral B^T sigma dV` for the Macro16 source128 `case031` failure. It
does not train a network, change model structure, change q48/LE ordering, or
change the standard 128 integration-point rule.

The audit compares three volume definitions:

1. reference/standard Macro16 volume: compact field `integration_weight_phys`;
2. selected-frame Abaqus volume: source field `ip_IVOL_abaqus_selected_frames`;
3. inferred Abaqus volume: source field `IVOL128_inferred_from_DLE`.

## Data Used

Primary diagnosis:

`D:\IS-FEM\deeponet_macro_operator_fresh\reports\macro16_case031_mechanics_diagnosis.md`

Main compact list:

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\macro16_source128_teacher_compact_list.txt`

Compact count: 10

Frame count: 100

Case031 compact:

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\0002_case031_macro16_source128_teacher.npz`

Case031 source compact:

`D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\case031\complete_case031_v2b_local_strain.npz`

Diagnostic JSON inputs:

- `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\case031_weight_mode_comparison.json`
- `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\case031_label_B_weight_factorization.json`
- `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\force_stiffness_audit_mechanics.json`

## Commands Used

Existing Gate 03 audit command:

```powershell
$env:PYTHONPATH='src;scripts'; py scripts\audit_macro16_force_stiffness.py --compact-list runs\macro16_source128_rigid_preprocess_audit_real_10case\macro16_source128_teacher_compact_list.txt --out runs\macro16_source128_rigid_preprocess_audit_real_10case\force_stiffness_audit_mechanics.json --weight-mode auto --max-force-rel 2.0e-2 --max-stiffness-rel 2.0e-2 --max-macro-stiffness-symmetry-rel 1.0e-10
```

This report additionally read the existing case031 diagnostic JSONs and source
compact fields with:

```powershell
rg -n "integration_weight_phys|integration_weight_hat|ip_IVOL_abaqus_selected_frames|IVOL128_inferred_from_DLE|weight_mode|weights_phys" src scripts reports -g "*.py" -g "*.md"
py -
```

No training command was run.

## Current Volume Contract

The current `integration_weight_phys` is the Macro16 X16 standard source128
reference volume:

- compact metadata: `integration_weight_source = macro16_x16_source128_standard_rule`;
- compact metadata: `integration_weight_rule = macro16_x16_standard_source128`;
- compact metadata: `integration_weight_coordinate = hat-dimensionless`;
- compact metadata: `integration_weight_phys_coordinate = physical-volume`;
- builder logic: `integration_weight_phys = integration_weight_hat * L_ref^3`;
- for case031, `L_ref = 1.0`, so `integration_weight_hat` and
  `integration_weight_phys` have the same numeric values.

This is a fixed reference/standard geometry volume. It is not the selected
deformed-frame Abaqus volume.

Implementation evidence:

- `scripts/build_macro16_source128_teacher.py` always writes
  `integration_weight_phys=weights_hat * L_ref^3`.
- If a source volume is requested, the builder stores it separately as
  `requested_integration_weight_phys`; it does not replace
  `integration_weight_phys`.
- `scripts/audit_macro16_force_stiffness.py --weight-mode auto` resolves to
  `as-stored` when `integration_weight_phys` exists, so Gate 03 assembled with
  the compact's standard reference volume.

## Force Error By Volume Definition

For case031, `LE_macro` and `B_macro_qraw` exactly match the source 128-IP
labels:

- `LE_macro` vs `LE128_base`: rel `0.0`, max abs `0.0`;
- `B_macro_qraw` vs `B_LE128_forward`: rel `0.0`, max abs `0.0`.

Therefore this volume audit is not confounded by a Macro16 copy/reorder error.

| Volume definition used in F_int | Force rel | Force diff RMS | Force max abs diff | Weight sum range |
|---|---:|---:|---:|---|
| Macro standard `integration_weight_phys` | `0.03578126940126121` | `4.430425026987133` | `28.853770300121425` | `0.009999880101531744` fixed |
| Source reference `ip_IVOL_abaqus` | `0.03701216424256369` | `4.5828340220213795` | `29.759679403250857` | `0.009992862120270729` fixed |
| Selected-frame `ip_IVOL_abaqus_selected_frames` | `0.00858572515082126` | `1.0630816686925764` | `3.512845611745206` | `0.009984073556552175 - 0.010159562254557386` |
| Inferred `IVOL128_inferred_from_DLE` | `0.00867529029215798` | `1.074171595080427` | `3.660880192841944` | `0.009985520230771774 - 0.010164180679514068` |

Selected-frame and inferred volumes reduce case031 force relative error from
about `3.6%` to about `0.86%`, below the current `0.02` force threshold.

## Frame Volume Drift

The selected-frame Abaqus volume changes with deformation amplitude:

| Frame | Selected-frame volume sum | Rel vs reference |
|---:|---:|---:|
| 0 | `0.009990937869588379` | `0.0021382438882910715` |
| 1 | `0.009985701683035586` | `0.004449505474211322` |
| 2 | `0.009984073556552175` | `0.0069950151030396075` |
| 3 | `0.009990901875426061` | `0.009749201002558943` |
| 4 | `0.010005282143538352` | `0.012912499861835399` |
| 5 | `0.010025907380622812` | `0.01665649675420061` |
| 6 | `0.01005204130342463` | `0.021064605008340016` |
| 7 | `0.010083200621011201` | `0.026170135546489264` |
| 8 | `0.010119095997652039` | `0.031982024363002105` |
| 9 | `0.010159562254557386` | `0.038499428476653524` |

The inferred volume follows the same trend: its frame-9 relative difference
from reference is `0.038471377608945256`. This close agreement supports that
the selected-frame and inferred Abaqus volumes describe the same effective
deformed-frame volume convention.

## Required Answers

### 1. Current `integration_weight_phys` corresponds to which volume?

It corresponds to the Macro16 X16 standard source128 reference volume:
`integration_weight_hat * L_ref^3` from the fixed Macro16 reference geometry and
the fixed 128-point rule. It is frame-invariant for case031 and does not use
selected-frame Abaqus `IVOL`.

### 2. Which volume definition is Abaqus reaction force closer to?

Abaqus reaction force is much closer to selected-frame Abaqus volume and to the
inferred Abaqus volume:

- reference/standard volume force rel: `0.03578126940126121`;
- source reference `ip_IVOL_abaqus` force rel: `0.03701216424256369`;
- selected-frame `ip_IVOL_abaqus_selected_frames` force rel:
  `0.00858572515082126`;
- inferred `IVOL128_inferred_from_DLE` force rel: `0.00867529029215798`.

The selected-frame volume is the closest by a small margin. The inferred volume
is effectively equivalent for this force audit.

### 3. Is case031 force error mainly explained by volume convention?

Yes, for recovery force. Changing only the volume definition, while keeping
`LE` and `B` fixed, reduces force relative error from about `0.036` to about
`0.0086`. That moves case031 below the `0.02` force threshold.

This conclusion is specific to `F_int`. The earlier mechanics diagnosis showed
that stiffness remains above threshold after volume substitution, so volume
convention is the dominant explanation for force error but not a full
explanation for stiffness error.

### 4. Which volume should later physical assembly use?

For comparison against Abaqus reaction forces generated on selected frames,
physical assembly should use selected-frame physical volume as the canonical
audit volume, with `IVOL128_inferred_from_DLE` as a fallback or cross-check when
selected-frame `IVOL` is unavailable.

Recommended naming convention:

- keep `integration_weight_phys_ref` for the current reference/standard volume;
- add `integration_weight_phys_selected` for selected-frame Abaqus volume;
- optionally add `integration_weight_phys_inferred` for the DLE-inferred
  Abaqus volume;
- make the force audit choose the volume explicitly, not implicitly.

Do not silently redefine existing `integration_weight_phys` without updating
the compact contract and reports, because current documents already use it as
the standard reference physical weight.

### 5. If modifying, should compact, loader, or audit script change?

The first required change is the audit script. Gate 03 should support an
explicit volume mode, for example:

- `--physical-volume-mode reference-standard`;
- `--physical-volume-mode selected-frame`;
- `--physical-volume-mode inferred-abaqus`.

The compact should then be regenerated or extended so selected-frame and
inferred physical volumes are carried in clearly named fields. This is needed
for reproducible audits without reaching back into source compacts.

The training loader does not need to change for this force audit unless
training losses or future model outputs explicitly use physical volume. The
current loader uses `integration_weight_hat` for training weights and stores
`integration_weight_phys` only as audit metadata. If the compact contract is
extended, the loader should preserve the existing training behavior and only
record the selected physical-volume source in metadata.

## Result

Volume convention audit status: PASS for diagnosis.

Gate 03 status remains FAIL overall, because the canonical Gate 03 audit still
uses the current `integration_weight_phys` reference/standard volume and because
the stiffness residual is not fully explained by volume convention.

## Unresolved Risks

- The selected-frame volume explanation is proven here for case031 force
  closure, not yet for the full 10-case set.
- Stiffness still requires a separate tangent audit, because selected-frame or
  inferred volume leaves case031 stiffness relative error near `0.032`.
- The project needs a contract decision before renaming or redefining
  `integration_weight_phys`.

## Recommended Next Action

Add explicit physical-volume modes to the force/stiffness audit and rerun Gate
03 force closure on all 10 compacts with selected-frame and inferred volumes.
Then decide the compact contract: preserve `integration_weight_phys` as
reference-standard and add selected/inferred fields, or explicitly migrate the
canonical physical assembly field with a version bump.
