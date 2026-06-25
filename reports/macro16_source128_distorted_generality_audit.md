# Macro16 Source128 Distorted Generality Audit

Status: FAIL for standard X16 reference weights.

This audit does not train a network.  It checks whether the standard Macro16
source128 point rule can close recovery force and stiffness for medium and
strong non-flipped distorted geometries.

## Hardware And Parallel Run

- Host: ZYHANDZZY
- CPU: Intel Core i9-13900H, 14 cores, 20 logical threads
- Memory: 32 GB physical, about 20 GB free at audit start
- Compact build workers: 8
- Force/stiffness audit workers: 8

The 42 completed Abaqus compacts were already available.  Parallel postprocess
built 42 Macro16 source128 compacts in about 8 seconds and ran the
force/stiffness audit in about 11 seconds.

## Data

- Output root: `D:\IS-FEM\outputs\macro16_source128_generality_distorted_tasks_local`
- Complete compacts: 42
- Macro16 source128 compacts: 42
- Total frames: 420
- Geometries:
  - `case060`, moderately distorted
  - `case061`, strongly distorted but not flipped
- Per geometry q programs: 21

The standard Macro16 contract is preserved:

- geometry input is `X16`
- displacement input is `q48`
- point rule is fixed 128 parent-domain source128 rule
- reference weights are generated from `X16`
- `X_macro` is not model input
- internal fine-grid nodes are not model input

## Main Result

Audit file:

`D:\IS-FEM\outputs\macro16_source128_generality_distorted_tasks_local\postprocess\macro16_source128_x16weights_force_stiffness_audit.json`

Filtered summary:

`D:\IS-FEM\outputs\macro16_source128_generality_distorted_tasks_local\postprocess\macro16_source128_x16weights_filtered_summary.json`

All 42 compacts with standard X16 reference weights:

| Metric | Mean | Max | Threshold |
|---|---:|---:|---:|
| Force relative error | `0.211151` | `0.915421` | `0.02` |
| Stiffness relative error | `0.064072` | `0.066237` | `0.02` |

The all-case force relative error is distorted by `q_zero` and rigid cases,
where reference force is near machine zero.  The nonrigid subset is the more
useful failure signal:

| Metric | Mean | Max | Threshold |
|---|---:|---:|---:|
| Nonrigid force relative error | `0.064924` | `0.091880` | `0.02` |
| Nonrigid stiffness relative error | `0.064132` | `0.066237` | `0.02` |

By geometry, nonrigid standard X16 reference weights:

| Geometry | Force mean | Force max | Stiffness mean | Stiffness max |
|---|---:|---:|---:|---:|
| `case060` moderately distorted | `0.065754` | `0.091880` | `0.064107` | `0.066131` |
| `case061` strongly distorted not flipped | `0.064095` | `0.085904` | `0.064157` | `0.066237` |

## Control Candidate

The same point set with source Abaqus point volumes remains force/stiffness
closed for nonrigid cases:

| Candidate | Force mean | Force max | Stiffness mean | Stiffness max |
|---|---:|---:|---:|---:|
| Source128 Abaqus point volume, nonrigid | `0.003162` | `0.007282` | `0.003328` | `0.008726` |

This means the new distorted q programs and labels are usable.  The failure is
concentrated in the standard X16 reference weight distribution or physical
volume convention, not in network training.

## Interpretation

1. Standard 128-point Macro16 with X16 reference weights does not pass the
   `0.02` force and stiffness thresholds on medium and strong distorted
   geometries.
2. The failure is not caused by reducing 128 points to 18 points, because this
   audit keeps all 128 points.
3. The failure is not primarily a label or q-program failure, because source128
   point volumes close force and stiffness below threshold on the same cases.
4. Rigid and `q_zero` force relative errors should not be used alone, because
   their reference force norm is near zero.  Their absolute residuals are near
   machine precision for pure translations and zero displacement.
5. The next blocker is the standard Macro16 physical weight rule, especially
   how reference X16 weights should compare against Abaqus selected-frame or
   inferred physical volumes.

## Next Step

Do not train the network yet.

First add an explicit physical-volume audit mode:

1. reference standard X16 volume
2. source reference Abaqus volume
3. selected-frame Abaqus volume
4. inferred DLE volume

Then rerun the same 42-case audit and decide whether the standard Macro16
contract needs a selected-frame or inferred physical volume field for force and
stiffness assembly.
