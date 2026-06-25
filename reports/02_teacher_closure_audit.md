# Gate 02 Teacher Closure Audit

Role: Teacher Closure Audit Agent

## Scope

This audit checks whether the TRUE176/CSS8 128-IP teacher system closes before
it is used as the Macro16 source128 teacher. It does not train a network, change
model structure, change q48/LE ordering, or change the standard 128 integration
point rule.

## Data Used

Compact list:

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\macro16_source128_teacher_compact_list.txt`

Source compact count: 10

Frame count: 100

Cases:

`case019`, `case025`, `case031`, `case041`, `case043`, `case044`, `case045`,
`case046`, `case049`, `case050`

Generated audit JSON:

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\teacher_closure_audit.json`

Existing supporting audit JSON:

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\force_stiffness_audit_mechanics.json`

## Commands Used

The audit read the route and gate docs, inspected source compact fields, then
ran a read-only Python audit snippet:

```powershell
Get-Content -Raw AGENTS.md
Get-Content -Raw docs\PROJECT_BRIEF.md
Get-Content -Raw docs\THEORY_BASELINE.md
Get-Content -Raw docs\THEORY_AUDIT_MATRIX.md
Get-Content -Raw docs\GATE_WORKFLOW.md
rg -n "LE128_base|B_LE128_forward|ip_IVOL|RF|stiffness|force" scripts src tests reports tasks docs -g "*.py" -g "*.md"
py -
```

No training command was run.

## Checks

### 1. LE128_base Credibility

`LE128_base` is credible as an exported Abaqus strain label for this checked
set.

- `LE128_base` is finite for all 10 source compacts.
- `merge_LE128_base_max_abs_diff` max: `0.0`.
- `merge_q48_max_abs_diff` max: `0.0`.
- Source metadata uses `strain_field = LE` and `strain_label_key = LE128_base`.
- Reference integration-point coordinate audit max values are small:
  `audit_ref_ip_xyz_vs_abaqus_coord_max_abs <= 2.384185791015625e-07`.
- `audit_detJ_vs_IVOL_max_abs` is below `4.2928149923682213e-10` on the checked
  compacts.

### 2. B_LE128_forward Finite-Difference Closure

`B_LE128_forward` closes against the stored perturbation strain response.

Definition checked:

```text
B_LE128_forward ~= (LE128_plus - LE128_base) / delta
```

Aggregate result:

- relative error mean: `2.388088973006767e-08`
- relative error max: `2.5944000095411714e-08`

The stored `DLE128_forward` is a finite increment, not a derivative. The correct
comparison is `B_LE128_forward * delta` against `DLE128_forward`; that also
closes with max relative error `2.594400009557519e-08`.

Conclusion: the teacher B label is internally consistent with finite-difference
LE perturbations.

### 3. Teacher Recovery Force Closure

Teacher recovery force was assembled as:

```text
F_int = integral B_LE128_forward^T * (D * LE128_base) * dV
```

and compared against `RF_projected`.

Using reference-frame Abaqus `ip_IVOL_abaqus`:

- force relative error mean: `0.0042454499703150515`
- force relative error max: `0.03701216424256369`
- max case: `case031`

Using selected-frame Abaqus `ip_IVOL_abaqus_selected_frames`:

- force relative error mean: `0.0012727832304676163`
- force relative error max: `0.00858572515082126`
- all 10 cases are below the current `0.02` max-error threshold.

Using `IVOL128_inferred_from_DLE` where available:

- available on 3 cases: `case019`, `case025`, `case031`
- force relative error mean: `0.003563903790481338`
- force relative error max: `0.00867529029215798`

Conclusion: teacher recovery force closes against Abaqus reaction force when the
selected-frame/deformed volume convention is used. Reference-frame IVOL fails
the max gate because of `case031`.

### 4. Teacher Stiffness Closure

Teacher material-only stiffness was assembled as:

```text
K_material = integral B_LE128_forward^T * D * B_LE128_forward * dV
```

and compared against finite-difference reaction stiffness from
`RF_projected_plus`.

Using reference-frame Abaqus `ip_IVOL_abaqus`:

- stiffness relative error mean: `0.004872351568400779`
- stiffness relative error max: `0.043554899057122336`
- max case: `case031`

Using selected-frame Abaqus `ip_IVOL_abaqus_selected_frames`:

- stiffness relative error mean: `0.003664298282935901`
- stiffness relative error max: `0.03196211043109838`
- max case: `case031`

Using `IVOL128_inferred_from_DLE` where available:

- available on 3 cases: `case019`, `case025`, `case031`
- stiffness relative error mean: `0.011448045921376262`
- stiffness relative error max: `0.031979410389008645`

The material-only stiffness matrix is symmetric to numerical precision:

- `K_material` symmetry relative error max: `1.2282559543282374e-16`

The Abaqus finite-difference reaction stiffness is less symmetric, especially
for `case031`:

- `RF_projected_plus` FD stiffness symmetry relative error max:
  `0.024048688993506134`

Conclusion: teacher material-only stiffness does not fully close against the
Abaqus perturbed-reaction tangent on the full 10-case set. The blocker is again
`case031`, and selected-frame volume does not fully remove it.

## Volume And Weight Definitions

Observed teacher volume fields:

- `ip_IVOL_abaqus`: reference-frame Abaqus integration volume.
- `ip_IVOL_abaqus_selected_frames`: selected/deformed-frame Abaqus integration
  volume. Present for all 10 checked source compacts.
- `IVOL128_inferred_from_DLE`: DLE-inferred effective volume. Present for
  `case019`, `case025`, and `case031`.

Important convention:

- `ip_IVOL_abaqus` is fixed per reference geometry.
- `ip_IVOL_abaqus_selected_frames` changes with frame/deformation.
- `IVOL128_inferred_from_DLE` agrees with the selected-frame volume trend where
  available.

For comparison with Abaqus `RF_projected`, the selected-frame volume is the
best-supported force-closure convention in this audit.

## Case Summary

| Case | B FD rel | Force rel, ref IVOL | Force rel, selected IVOL | K rel, ref IVOL | K rel, selected IVOL |
|---|---:|---:|---:|---:|---:|
| case019 | `2.5544693031269094e-08` | `0.0014342091142663523` | `0.0016056888930695088` | `0.0022869398644121105` | `0.001963495107939358` |
| case025 | `2.5944000095411714e-08` | `0.000296702225483718` | `0.00032810364138174833` | `0.00043920492807343164` | `0.00040569562677957087` |
| case031 | `2.4832233440214747e-08` | `0.03701216424256369` | `0.00858572515082126` | `0.043554899057122336` | `0.03196211043109838` |
| case041 | `2.3200892148917887e-08` | `0.0002514367311112018` | `0.00022760185356611055` | `0.0002487246128262585` | `0.00023552780058557876` |
| case043 | `2.3228001387389134e-08` | `0.0008650153795050365` | `0.00030010835349493975` | `0.00048504794721411515` | `0.0004728016155049719` |
| case044 | `2.3071633744761258e-08` | `0.0002365109103951065` | `0.00021369925864745076` | `0.0002514943511586323` | `0.00023777419545614858` |
| case045 | `2.3417878965453503e-08` | `0.0006511720525219387` | `0.00028828583915446623` | `0.0003928005241579775` | `0.00037147123062004244` |
| case046 | `2.323248376131776e-08` | `0.0004211374306090988` | `0.0001628702251890568` | `0.00031141479404259864` | `0.00027852024840126406` |
| case049 | `2.3217224431907414e-08` | `0.0005570058293522622` | `0.00044032190572034055` | `0.0003257617476123257` | `0.00031227823128488803` |
| case050 | `2.3119856294034215e-08` | `0.0007291457873421155` | `0.0005754271836312788` | `0.0004272278573880024` | `0.0004033083416888045` |

## Gate Result

Gate 02 result: FAIL on full teacher stiffness closure.

Passed subchecks:

- `LE128_base` credibility: PASS.
- `B_LE128_forward` finite-difference LE closure: PASS.
- teacher recovery force with selected-frame IVOL: PASS under the current
  `0.02` max-error criterion.

Failed subcheck:

- teacher material-only stiffness versus perturbed Abaqus reaction stiffness:
  FAIL, max selected-frame-IVOL stiffness rel `0.03196211043109838`.

This is not a network issue. It is a mechanics/tangent convention issue already
visible inside the TRUE176/CSS8 teacher system.

## Unresolved Issues

- `K_material = B^T D B dV` is not the complete Abaqus perturbed-reaction
  tangent for the large-response `case031` path.
- Missing tangent terms remain plausible: `dV/dq`, `dB/dq`, and
  geometric/stress stiffness.
- The canonical volume convention still needs a project decision. Selected-frame
  IVOL is best for teacher force closure, but the compact/Gate contract has not
  been migrated.
- `IVOL128_inferred_from_DLE` is only present for 3 of the 10 checked cases.

## Recommended Next Action

Do not train the network from this gate state.

First, decide the Gate 02/03 mechanics convention:

1. Record selected-frame IVOL as the force-closure audit volume, or explicitly
   choose a different convention with rationale.
2. Add a consistent tangent audit that separates material-only stiffness from
   the full Abaqus perturbed-reaction tangent.
3. Re-run Gate 02 and Gate 03 after the volume/tangent contract is explicit.
