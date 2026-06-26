# Gate 04b Wind-Shell TRUE176 Template Generation Contract

Date: 2026-06-27

Role: Data Generation Contract Repair Agent

## 1. Task Goal

Correct the wind-shell Abaqus data-generation route so it follows the method
used by the original 2000-epoch TRUE176 shape4 training data.

This task did not train a network, did not modify model structure, and did not
change the fixed 128-point integration rule.

## 2. Correction

The previous wind-shell script path used one synthetic mixed `q48` vector and
linear amplitude frames. That path is now retained only as an explicit smoke
mode:

```text
--q-source-mode synthetic-mixed
```

The default formal path is now:

```text
--q-source-mode true176-template
```

The corrected route is:

```text
TRUE176 full48_vector.npy
-> legacy local-frame / H2 normalization
-> target wind-shell keep-node local frames
-> target L_ref scaling
-> 100-frame Abaqus displacement history
-> base + 48 forward perturbation jobs
-> fresh Abaqus LE/B compact export
-> Macro16 source128 compact rebuild
```

Old TRUE176 `LE/B` labels are not reused as new labels. They provide only the
complete `q48` displacement template. The target wind-shell `LE` and forward
`B` labels must be exported from fresh Abaqus runs.

## 3. Code Updated

Updated:

```text
scripts/run_gate04_wind_shell_generality_audit.py
tests/smoke_test.py
```

The script now records q-source metadata in each sample directory and boundary
contract, including:

```text
q_generation_method
true176_case_id
true176_vector_path
legacy_h_ref
target_L_ref
template_amplitude_scale
old_index_for_shape4_index
local_template_preservation_rel
```

The script also refuses to reuse an existing Abaqus export under
`--skip-existing` if the stored `q48_final.npy` differs from the newly selected
template-transferred `q48`.

## 4. Command Examples

Small corrected Gate 04 rerun, one TRUE176 template per wind-shell family:

```powershell
$env:PYTHONPATH='src;scripts'
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
py -3 scripts\run_gate04_wind_shell_generality_audit.py `
  --out-root runs\gate04_wind_shell_true176_template `
  --q-source-mode true176-template `
  --template-cases 1,2,3,4 `
  --template-assignment one-per-family `
  --increments 100 `
  --inner-workers 4 `
  --post-workers 8 `
  --skip-existing
```

Windows machine A example:

```powershell
py -3 scripts\run_gate04_wind_shell_generality_audit.py `
  --out-root D:\IS-FEM\outputs\wind_shell_true176_template_A `
  --families cylindrical_shell,conical_shell `
  --q-source-mode true176-template `
  --template-cases 1:101 `
  --template-assignment cross-product `
  --increments 100 `
  --inner-workers 4 `
  --post-workers 8 `
  --skip-existing
```

Windows machine B example:

```powershell
py -3 scripts\run_gate04_wind_shell_generality_audit.py `
  --out-root D:\IS-FEM\outputs\wind_shell_true176_template_B `
  --families thickness_varying_shell,mild_double_curvature_shell `
  --q-source-mode true176-template `
  --template-cases 1:101 `
  --template-assignment cross-product `
  --increments 100 `
  --inner-workers 4 `
  --post-workers 8 `
  --skip-existing
```

`1:101` means TRUE176 template cases 1 through 100.

## 5. Tests Run

```powershell
$env:PYTHONPATH='src;scripts'
py -m pytest tests\smoke_test.py -q -k "gate04_wind_shell"
py -m pytest -q
```

Result:

```text
2 passed, 82 deselected
90 passed
```

The tests verify:

1. default q-source mode is `true176-template`;
2. default increment count is `100`;
3. the transferred target local normalized displacement preserves the old
   TRUE176 local `q/H2` template;
4. the output remains 48-dimensional.

## 6. Local Preflight

No Abaqus jobs were run in this task. A local Python preflight checked that the
default TRUE176 template source is readable on this machine and that the four
wind-shell geometries remain valid before Abaqus generation.

Template-transfer preflight:

| Family | Script case id | TRUE176 template case | q norm | q abs max | local preservation rel | target L_ref |
|---|---|---:|---:|---:|---:|---:|
| cylindrical_shell | `case070_cylindrical_shell_t176001` | 1 | `0.00013190357041916316` | `4.4591435173508394e-05` | `1.007769482082308e-16` | `1.0` |
| conical_shell | `case071_conical_shell_t176002` | 2 | `0.00014227560620263234` | `5.151312074417775e-05` | `1.4640578531054938e-16` | `1.0786334725474749` |
| thickness_varying_shell | `case072_thickness_varying_shell_t176003` | 3 | `0.01710208236983759` | `0.007703284048038612` | `1.5322404654893525e-16` | `1.0` |
| mild_double_curvature_shell | `case073_mild_double_curvature_shell_t176004` | 4 | `0.01713435242286527` | `0.0075338121279790215` | `1.0514055728583509e-16` | `1.0018869078238446` |

Geometry preflight:

| Family | detJ positive | detJ min | detJ max | L_ref |
|---|---|---:|---:|---:|
| cylindrical_shell | yes | `0.0024618019399308246` | `0.0025508741023290106` | `1.0` |
| conical_shell | yes | `0.0020229055403685166` | `0.002211873210068551` | `1.0786334725474749` |
| thickness_varying_shell | yes | `0.0022548529976938706` | `0.003314747340171471` | `1.0` |
| mild_double_curvature_shell | yes | `0.00248206440218114` | `0.0025035733041578055` | `1.0018869078238446` |

## 7. Current Status

Script contract: PASS.

Fresh Abaqus data generation: not run in this task.

Gate 04 wind-shell force closure under this corrected TRUE176-template route:
pending fresh Abaqus regeneration and audit.

## 8. Recommended Next Action

Regenerate wind-shell Abaqus data with `--q-source-mode true176-template` and
`--increments 100`, then rerun the Macro16 source128 data-contract audit and
selected-frame force closure before using the data for training.
