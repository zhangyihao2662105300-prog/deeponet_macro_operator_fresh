# Gate 03b Consistent Tangent Audit

Role: Consistent Tangent Audit Agent

## Scope

This audit analyzes why the material-only stiffness

```text
K_material = integral B^T D B dV
```

differs from the Abaqus finite-difference tangent for the Macro16 source128
`case031` blocker. It does not train a network, change model structure, change
q48 ordering, change LE ordering, or change the standard 128 integration-point
rule.

## Data Used

Primary reports:

- `D:\IS-FEM\deeponet_macro_operator_fresh\reports\macro16_force_stiffness_audit.md`
- `D:\IS-FEM\deeponet_macro_operator_fresh\reports\macro16_case031_mechanics_diagnosis.md`
- `D:\IS-FEM\deeponet_macro_operator_fresh\reports\03a_volume_convention_audit.md`

Main compact list:

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\macro16_source128_teacher_compact_list.txt`

Compact count: 10

Frame count: 100

Focused case compact:

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\0002_case031_macro16_source128_teacher.npz`

Focused source compact:

`D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\case031\complete_case031_v2b_local_strain.npz`

Abaqus source ODB:

`D:\IS-FEM\NNSE_css8_push_tmp\run_logs\true176_cylinder_template_base_audit\sample_894030_cylinder_case031_pair_Axial_Force_Torque_-1_+1\base\t176_cyl_base_894030_base.odb`

## Commands Used

Existing Gate 03 audit command:

```powershell
$env:PYTHONPATH='src;scripts'; py scripts\audit_macro16_force_stiffness.py --compact-list runs\macro16_source128_rigid_preprocess_audit_real_10case\macro16_source128_teacher_compact_list.txt --out runs\macro16_source128_rigid_preprocess_audit_real_10case\force_stiffness_audit_mechanics.json --volume-mode reference --tangent-mode material-only --max-force-rel 2.0e-2 --max-stiffness-rel 2.0e-2 --max-macro-stiffness-symmetry-rel 1.0e-10
```

With the refactored audit script, the explicit tangent-reference diagnostic is:

```powershell
$env:PYTHONPATH='src;scripts'; py scripts\audit_macro16_force_stiffness.py --compact-list runs\macro16_source128_rigid_preprocess_audit_real_10case\macro16_source128_teacher_compact_list.txt --out runs\macro16_source128_rigid_preprocess_audit_real_10case\force_stiffness_audit_full_fd_reference.json --volume-mode selected-frame --tangent-mode full-fd-reference --max-force-rel 2.0e-2 --max-stiffness-rel 2.0e-2
```

Readonly tangent decomposition was computed with:

```powershell
$env:PYTHONPATH='src;scripts'; py -
```

The Python snippet loaded the case031 compact and source compact, then compared:

- `K_material` with reference, selected-frame, and inferred volume;
- `DLE128_forward / delta` against `B_LE128_forward`;
- fixed-B finite difference of `F(B_base, LE_plus, V_base)`;
- fixed-B finite difference of `F(B_base, LE_plus, V_plus)`;
- Abaqus `K_fd = (RF_projected_plus - RF_projected) / delta`.

The Abaqus input was checked with:

```powershell
Select-String -Path D:\IS-FEM\NNSE_css8_push_tmp\run_logs\true176_cylinder_template_base_audit\sample_894030_cylinder_case031_pair_Axial_Force_Torque_-1_+1\base\t176_cyl_base_894030_base.inp -Pattern "\*Step|nlgeom|\*Static|RF|IVOL" -CaseSensitive:$false
```

Relevant line:

```text
*Step, name=STEP_BASE, nlgeom=YES, inc=10
```

No training command was run.

## Current Stiffness Definition

`scripts/audit_macro16_force_stiffness.py` assembles:

```text
stress = D @ LE
F_int  = integral B^T stress dV
K      = integral B^T D B dV
```

The stiffness comparison reference is not another material-only matrix. It is:

```text
K_fd[:, :, direction] =
    (RF_projected_plus[:, direction, :] - RF_projected[:, :]) / delta
```

For case031:

- `delta = 1.0e-6`;
- `perturb_directions = 0..47`;
- `RF_projected_plus` has all 48 columns;
- the Abaqus step uses `nlgeom=YES`.

Therefore the current audit compares a material-only symmetric assembly against
a finite-difference tangent of the nonlinear Abaqus reaction-force map.

## Key Metrics

### Existing Gate 03 Result

`case031` is the current blocker:

| Quantity | Value |
|---|---:|
| force rel, current standard volume | `0.03578126940126121` |
| stiffness rel, current standard volume | `0.04256563396475579` |
| stiffness rel, source reference `ip_IVOL_abaqus` | `0.04355489905712233` |
| stiffness rel, selected-frame `ip_IVOL_abaqus_selected_frames` | `0.031962110431098374` |
| stiffness rel, inferred `IVOL128_inferred_from_DLE` | `0.031979410389008645` |

Volume convention fixes most of the force error, but does not close the
stiffness error.

### B Finite-Difference Check

The source field `DLE128_forward` stores the strain increment, not the already
divided derivative. After division by `delta`, it matches `B_LE128_forward`:

| Check | Value |
|---|---:|
| rel, `(DLE128_forward / delta)` vs `B_LE128_forward` | `2.4832233440785776e-08` |
| RMS diff | `1.3709640351559444e-07` |
| max abs diff | `3.814697265625e-06` |

This rules out a B column-order or DLE scaling explanation for the case031
stiffness failure.

### Tangent Decomposition

All values below are relative to the Abaqus finite-difference tangent
`K_fd = (RF_projected_plus - RF_projected) / delta`.

| Candidate K | Rel error | Diff RMS | Max abs diff |
|---|---:|---:|---:|
| material K, reference IVOL | `0.04355489905712233` | `3536.5084718880466` | `45421.2917854751` |
| material K, selected-frame IVOL | `0.031962110431098374` | `2595.213782283276` | `44382.9797846234` |
| material K, inferred IVOL | `0.031979410389008645` | `2596.6184795512627` | `44380.56277776594` |
| fixed B, `LE_plus`, inferred base volume | `0.03197941048837209` | `2596.6184876192337` | `44380.562789786636` |
| fixed B, `LE_plus`, inferred plus volume | `0.031308753869730865` | `2542.1634695868906` | `44324.557447316336` |

Term-scale checks:

| Difference | Relative norm vs `K_fd` |
|---|---:|
| selected-frame material K minus reference material K | `0.02252250684497405` |
| `dV/dq` term estimated by inferred plus volume | `0.01127990678186004` |
| `(DLE/delta)` material term minus stored B material term | `8.64139404157011e-09` |
| finite-LE forward FD nonlinearity beyond `(DLE/delta)` | `1.8770773342528007e-12` |
| remaining residual after fixed B, `LE_plus`, plus inferred volume | `0.031308753869730865` |

Frame-level trend:

| Frame | material K, reference IVOL | material K, inferred IVOL | fixed B, `LE_plus`, plus inferred volume |
|---:|---:|---:|---:|
| 0 | `0.011465621009113124` | `0.011201274632172659` | `0.012407768138223125` |
| 1 | `0.014245737440642912` | `0.013394485188027947` | `0.013925423070767561` |
| 2 | `0.016380563546147715` | `0.014432316825014226` | `0.014304138098800769` |
| 3 | `0.02050599884617984` | `0.016991098737202892` | `0.016083099923249636` |
| 4 | `0.02706752841207714` | `0.021384977831808146` | `0.01993157135805484` |
| 5 | `0.03496029312224363` | `0.026567964111190643` | `0.024882502764578027` |
| 6 | `0.043934293446103286` | `0.03244459366457307` | `0.030835828735886906` |
| 7 | `0.05403150142361136` | `0.039163443628113116` | `0.03789070968995174` |
| 8 | `0.06523047516483625` | `0.04676442025289234` | `0.04602917406489435` |
| 9 | `0.0774558157191778` | `0.05519345456412767` | `0.055153294765673905` |

The error grows with deformation amplitude. This is the expected signature of
missing nonlinear tangent terms, not random DOF ordering noise.

## Required Checks

### 1. Is `dV/dq` missing?

Yes.

The current material-only audit either uses the fixed reference/standard
`integration_weight_phys` or substitutes a selected/inferred frame volume as a
base value. In both cases, the assembled stiffness is still:

```text
K_material = integral B^T D B V
```

It does not include:

```text
integral B^T sigma (dV/dq) dq
```

The source compact contains a useful finite-difference proxy:
`IVOL128_plus_inferred_from_DLE`. Using it in a fixed-B finite difference
changes the tangent by relative norm `0.01127990678186004` versus `K_fd` and
reduces the total stiffness error from about `0.03198` to `0.03131`.

So `dV/dq` is real and currently absent, but it is not the whole remaining
stiffness mismatch.

### 2. Is `dB/dq` missing?

Yes.

The current stiffness differentiates only the stress path:

```text
d sigma / dq = D B
```

It holds the force projection operator `B^T` fixed. For an internal force of the
form

```text
F_j(q) = sum_p B_paj(q) sigma_pa(q) V_p(q)
```

the full tangent contains:

```text
sum_p B_paj D_ab B_pbk V_p
+ sum_p (dB_paj/dq_k) sigma_pa V_p
+ sum_p B_paj sigma_pa (dV_p/dq_k)
```

The second term is absent from the current audit. The available compact fields
do not contain `B_plus` or an explicit `dB/dq`, so this report cannot isolate
that term numerically by itself. However, after `DLE/delta`, plus strain, and
plus inferred volume are accounted for, the residual is still
`0.031308753869730865`. That remaining residual is consistent with missing
`dB/dq` and other updated-geometry/stress-stiffness terms.

### 3. Is geometric stiffness / stress stiffness missing?

Yes.

For `case031`, Abaqus was run with:

```text
*Step, name=STEP_BASE, nlgeom=YES, inc=10
```

The finite-difference reference is therefore the tangent of a nonlinear
updated-geometry reaction-force response, not the tangent of a small-strain
linear material integral with fixed geometry.

The current `B^T D B dV` includes only the material tangent. It misses terms
that are stress-proportional or geometry-proportional, including:

- change of the force projection `B^T` with displacement;
- stress stiffness / initial-stress terms;
- volume change terms;
- local-frame and updated-coordinate effects carried by Abaqus RF and IVOL;
- any projection/constraint tangent effects in the exported boundary reaction.

The residual grows strongly from early to late frames, matching the increase in
q norm and stress level reported in the case031 mechanics diagnosis.

### 4. What does the Abaqus FD tangent correspond to?

It is not a secant tangent.

A secant quantity would compare force to displacement over the full load path,
for example `RF(q) / q` or `(RF(q) - RF(0)) / (q - 0)`. The audit instead uses
a local forward finite difference:

```text
(RF_projected_plus - RF_projected) / delta
```

It is not a pure material tangent.

The material tangent would be `B^T D B dV`, and the best available material-only
variant still has stiffness relative error about `0.032` against Abaqus FD.

The best classification is:

```text
Abaqus FD tangent = finite-difference approximation to the full nonlinear
projected reaction-force tangent at the selected frame.
```

Because the Abaqus step uses `nlgeom=YES`, this finite-difference tangent should
be treated as a numerical proxy for the full consistent tangent of the Abaqus
response map, including updated-geometry and stress-stiffness effects. It is
not a direct extraction of Abaqus' internal element tangent matrix, so the most
precise name is "Abaqus RF finite-difference full-response tangent".

### 5. Should the current K gate threshold target material-only K or full tangent K?

If the reference remains Abaqus `RF_projected_plus`, the gate must target the
full tangent K, not material-only K.

The current `0.02` stiffness threshold compares against Abaqus full-response
FD. Under that reference, a material-only `B^T D B dV` gate is not physically
consistent for large-response `nlgeom=YES` cases like case031.

Recommended split:

| Gate quantity | Reference | Status |
|---|---|---|
| force closure | Abaqus RF, using selected-frame or inferred volume | should be rerun after volume decision |
| material-only K audit | `B^T D B dV` with explicit volume convention | diagnostic / assembly check, not Abaqus full tangent closure |
| full tangent K gate | Abaqus `RF_projected_plus` FD | requires `dV/dq`, `dB/dq`, and geometric/stress stiffness terms |

Therefore the current Gate 03 stiffness max-error threshold should not be used
to declare `K_material` failed as a material-only assembly. It should be used
only after the audit candidate is upgraded to the same full tangent definition
as the Abaqus finite-difference reference.

## Diagnosis

The material-only stiffness mismatch in case031 is not caused by a network,
q48 ordering, LE ordering, the 128-point rule, or a copied B-label error.

Confirmed:

- `LE_macro` equals source `LE128_base`.
- `B_macro_qraw` equals source `B_LE128_forward`.
- `(DLE128_forward / delta)` matches `B_LE128_forward` with rel
  `2.4832233440785776e-08`.
- selected-frame/inferred volume improves force closure and reduces K error,
  but does not close K.
- Abaqus FD is produced from nonlinear RF differences under `nlgeom=YES`.

Primary cause:

The current K audit candidate is material-only, while the Abaqus reference is a
full nonlinear reaction-force finite-difference tangent. The missing terms are
the updated-volume tangent `dV/dq`, the force-projection tangent `dB/dq`, and
geometric/stress stiffness terms.

## Result

Consistent tangent audit status: PASS for diagnosis.

Gate 03 status remains FAIL under the current single stiffness threshold,
because the audited K candidate and Abaqus FD reference are not the same tangent
definition.

## Unresolved Risks

- `dB/dq` is inferred as missing from the formula and residual, but is not
  directly isolated because the compact does not contain `B_plus`.
- The current finite-difference reference is forward difference with
  `delta=1.0e-6`; a central-difference check would better separate numerical
  FD noise from true tangent asymmetry.
- The conclusion is focused on case031. It should be rerun after the volume
  convention decision across all 10 compacts.
- A solver-facing Macro16 element still needs a route decision: material-only
  stiffness as a quasi-Newton approximation, or full consistent tangent for
  Newton closure.

## Recommended Next Action

1. Keep the 128-point rule unchanged.
2. Do not train the network to absorb this tangent mismatch.
3. Add explicit Gate 03 stiffness modes:
   - `material-only`;
   - `full-fd-reference`;
   - later `full-consistent-candidate`.
4. Extend the tangent audit candidate, not the network, to include:
   - selected-frame or inferred physical volume;
   - `dV/dq` from plus-volume finite differences or an analytic volume
     derivative;
   - `dB/dq` / geometric stiffness terms;
   - stress stiffness terms consistent with the Abaqus `nlgeom=YES` response.
5. Use the `0.02` Abaqus FD stiffness threshold only for the full tangent
   candidate. Keep material-only `B^T D B dV` as a separate assembly diagnostic
   until the project explicitly chooses it as an approximation.
