# Gate 03 Macro16 Force/Stiffness Re-Audit

Role: Gate 03 Re-Audit Agent

Date: 2026-06-25

## Scope

This re-audit reruns the Macro16 source128 force/stiffness audit with explicit
volume and tangent modes. It does not train a network, change model structure,
change q48 or LE ordering, or change the standard 128 integration-point rule.

Gate 03 is split into three quantities:

1. force closure against Abaqus projected RF;
2. material-only stiffness diagnostic, `B^T D B dV`;
3. full-fd-reference tangent reference path.

`full-fd-reference` is a reference-path diagnostic: it compares Abaqus RF
finite-difference tangent against itself. It verifies the audit branch and
reference extraction, but it is not a newly implemented Macro16 full consistent
tangent.

## Data Used

Compact list:

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\macro16_source128_teacher_compact_list.txt`

Compact count: 10

Frame count: 100

Output JSONs:

- `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\force_stiffness_audit_selected_material_only_reaudit.json`
- `D:\IS-FEM\deeponet_macro_operator_fresh\runs\macro16_source128_rigid_preprocess_audit_real_10case\force_stiffness_audit_selected_full_fd_reference_reaudit.json`

## Commands Used

Force/material diagnostic:

```powershell
$env:PYTHONPATH='src;scripts'; py scripts\audit_macro16_force_stiffness.py --compact-list runs\macro16_source128_rigid_preprocess_audit_real_10case\macro16_source128_teacher_compact_list.txt --out runs\macro16_source128_rigid_preprocess_audit_real_10case\force_stiffness_audit_selected_material_only_reaudit.json --volume-mode selected-frame --tangent-mode material-only --max-force-rel 2.0e-2 --max-stiffness-rel 2.0e-2 --max-macro-stiffness-symmetry-rel 1.0e-10
```

Full FD reference:

```powershell
$env:PYTHONPATH='src;scripts'; py scripts\audit_macro16_force_stiffness.py --compact-list runs\macro16_source128_rigid_preprocess_audit_real_10case\macro16_source128_teacher_compact_list.txt --out runs\macro16_source128_rigid_preprocess_audit_real_10case\force_stiffness_audit_selected_full_fd_reference_reaudit.json --volume-mode selected-frame --tangent-mode full-fd-reference --max-force-rel 2.0e-2 --max-stiffness-rel 2.0e-2
```

No training command was run.

## Pass Standard

Force closure:

- mean relative error reported;
- max relative error must be <= `0.02`.

Material-only stiffness:

- reported as a diagnostic quantity;
- it is not the complete Abaqus nonlinear tangent under large deformation;
- the old `0.02` full-FD stiffness threshold must not be interpreted as a
  full-tangent pass/fail for this material-only candidate.

Full FD reference:

- active stiffness mean/max should be `0.0` because the Abaqus FD tangent is
  used as both candidate and reference;
- this validates the explicit reference branch, not a Macro16 full tangent
  implementation.

## Key Metrics

| Audit mode | Force mean rel | Force max rel | Active stiffness mean rel | Active stiffness max rel | Material-only K mean rel | Material-only K max rel | Strict pass |
|---|---:|---:|---:|---:|---:|---:|---|
| selected-frame + material-only | `0.0012727832304676163` | `0.00858572515082126` | `0.0036642982829359003` | `0.031962110431098374` | `0.0036642982829359003` | `0.031962110431098374` | false |
| selected-frame + full-fd-reference | `0.0012727832304676163` | `0.00858572515082126` | `0.0` | `0.0` | `0.0036642982829359003` | `0.031962110431098374` | true |

Additional reference-path metrics:

| Quantity | Mean rel | Max rel |
|---|---:|---:|
| selected-frame weight vs selected-frame source IVOL | `0.0` | `0.0` |
| material-only macro stiffness symmetry | `1.17607457506591e-16` | `1.3549217800954415e-16` |
| Abaqus FD active tangent asymmetry | `0.0027811227316897136` | `0.024048688993506127` |

## Worst Case

`case031` is still the worst case for selected-frame force closure and the
material-only stiffness diagnostic.

| Case | Force rel | Force max abs diff | Material-only K rel | Material-only K max abs diff |
|---|---:|---:|---:|---:|
| `case031` | `0.00858572515082126` | `3.512845611745206` | `0.031962110431098374` | `44382.9797846234` |
| `case019` | `0.0016056888930695088` | `0.019574757967073975` | `0.001963495107939358` | `2630.710505720228` |
| `case043` | `0.00030010835349493975` | `0.002563377729049243` | `0.0004728016155049718` | `748.6470411662012` |

For `full-fd-reference`, all stiffness relative errors are `0.0` by
construction, so there is no nonzero full-FD stiffness worst case. `case031`
remains the largest force case, the largest material-only K diagnostic case,
and the largest Abaqus FD tangent asymmetry case.

## Gate 03 Split Status

| Gate 03 part | Status | Evidence |
|---|---|---|
| Force closure | PASS on current 10-case set | With `--volume-mode selected-frame`, force mean rel is `0.0012727832304676163` and max rel is `0.00858572515082126`, below `0.02`. |
| Material-only stiffness diagnostic | Diagnostic residual remains; not a full tangent gate | `B^T D B dV` with selected-frame volume has mean rel `0.0036642982829359003` and max rel `0.031962110431098374`, worst `case031`. |
| Full tangent closure | Not yet implemented as a Macro16 candidate | `--tangent-mode full-fd-reference` gives active stiffness rel `0.0` because the Abaqus FD tangent is compared with itself. This verifies the reference path but does not add `dV/dq`, `dB/dq`, geometric stiffness, or stress stiffness to Macro16. |

## Result

Gate 03 should no longer be described as one undifferentiated force/stiffness
failure.

Updated Gate 03 interpretation:

- Macro16 source128 force closure: PASS for the current 10-case set under the
  selected-frame Abaqus IVOL convention.
- Material-only stiffness: keep as a named diagnostic; it remains above `0.02`
  max rel on `case031`.
- Full tangent closure: still incomplete until a real full consistent tangent
  candidate is implemented or explicitly waived with evidence.

Training remains blocked by the gate workflow because Gate 02 full teacher
stiffness and Gate 03 full tangent closure are still unresolved, and Gate 04
and Gate 05 remain incomplete.

## Unresolved Issues

- The selected-frame force result is verified only on the current 10-case,
  100-frame set.
- `case031` remains the material-only stiffness diagnostic blocker.
- `full-fd-reference` has nonzero tangent asymmetry because the Abaqus forward
  RF finite-difference tangent is not symmetric; max asymmetry rel is
  `0.024048688993506127`.
- A solver-facing Macro16 element still needs a decision: implement full
  consistent tangent terms, or explicitly accept material-only stiffness as a
  named approximation.

## Recommended Next Action

Keep the 128-point rule unchanged and do not train the network yet. The next
mechanics task should either implement/audit a full consistent tangent candidate
with `dV/dq`, `dB/dq`, geometric stiffness, and stress stiffness terms, or make
an explicit gate decision that material-only stiffness is accepted as a
quasi-Newton diagnostic rather than a full Newton tangent.
