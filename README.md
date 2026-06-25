# DeepONet Macro-Element Strain Operator

Fresh implementation for a geometry-conditioned macro-element operator:

```text
boundary displacement q_b + macro geometry X + query natural coordinate xi
    -> physical strain epsilon(xi)
```

The intended operator is

```text
epsilon(xi) = G_theta(q_b, X)(xi)
B_macro(xi) = d epsilon(xi) / d q_b
```

`B_macro` is computed by automatic differentiation, not trained as an
independent network output.

## Literature anchor

The attached paper is:

```text
Ouyang, W., Shin, Y., Liu, S.-W. & Lu, L.
NOEM: efficient and scalable finite element method enabled by reusable neural operators.
Nature Computational Science 6, 417-429 (2026).
DOI: 10.1038/s43588-026-00974-2
```

Official paper/code links:

- Paper: https://www.nature.com/articles/s43588-026-00974-2
- Source code: https://github.com/lu-group/noem
- Zenodo archive: https://doi.org/10.5281/zenodo.18678157

NOEM motivates the "neural-operator element" direction: replace a fine
subdomain response with a reusable neural operator inside a FEM variational
framework. This project does not copy NOEM examples; it re-targets the idea to
macro-element strain-field learning for shell/solid-style elements.

## Current scope

This repository now contains two levels.

### 1. Clean Hex8 prototype

This first version is deliberately small and independent:

- 8-node isoparametric hexahedral macro geometry.
- Branch net encodes flattened `q_b` and `X`.
- Trunk net encodes natural coordinate `xi` and optional local geometry
  features `J(xi)` and `detJ(xi)`.
- Geometry is centered/scaled per macro element; boundary displacement is
  centered to remove rigid translation before scaling.
- Output is 6 engineering strain components:
  `[e11, e22, e33, gamma12, gamma13, gamma23]`.
- Synthetic training data is generated from affine, rigid translation, and
  rigid rotation displacement fields over randomized macro geometries.

### 2. TRUE176 / CSS8 4x4 128-IP DeepONet route

This is the production-facing route aligned with the current NNSE training
launcher commit `fc4117d`.

Current data contract:

```text
input  = q48_raw[48] + X_keep[16,3]
output = LE[target_ips, 6]
B      = d LE / d q48_raw
```

The DeepONet form is:

```text
Branch input = standardized [q48_raw, X_keep]        # [B,96]
Trunk input  = point/IP features                     # [B,P,F]
Output       = standardized LE                      # [B,P,6]
AD B         = d(LE_norm)/d(q48_norm)               # [B,P,6,48]
```

The default architecture is now the finite-element flavored residual form:

```text
fe-linear-residual default:
  Branch:   one MLP over standardized [q48_raw, X_keep]
  Trunk:    one MLP over integration-point features
  Baseline: B_base_norm(point_features) @ q48_norm
  Residual: standard segmented branch/trunk dot product for 6 LE channels
```

This makes the dominant finite-element-like relation explicit:

```text
LE_norm = B_base_norm(point) q48_norm + DeepONet_residual(q48, X_keep, point)
```

`B_base_norm` is initialized from the mean training
`d(LE_norm)/d(q48_norm)` and then corrected by a point-feature network.  It is
not given sample labels in the forward pass; the labels are used only through
the Sobolev/raw-B loss.  The raw-B auxiliary loss is enabled by default so the
model is constrained after converting
`J_norm = d(LE_norm)/d(q_norm)` back to `B = J_norm * LE_std / q_std`.
The residual branch is zero-initialized by default, so the first prediction is
the FE-like linear baseline rather than a random residual perturbation.

The standard concat-skip DeepONet is kept as a baseline:

```text
--model-style concat-skip
```

The NOEM/MIONet-style split-branch model is kept as an ablation:

```text
--model-style noem-mionet
```

For a pure NOEM-style comparison, `--model-style noem-mionet` does not use
`q-skip`.  It can be re-enabled explicitly with `--use-q-skip`.

`X_keep` means the 16 reference coordinates attached to the 16 q48 control
nodes.  These are the macro-level geometric coordinates visible together with
`q48_raw`; the 34 non-control CSS8 grid nodes are not Branch inputs.

Length scaling is handled before the usual machine-learning mean/std
standardization.  Current TRUE176 compacts are dimensionless and use:

```text
scale_mode = normalized
H = 1
q_coordinate = q48_raw in normalized length
B_label = dLE/dq48_raw
```

For future physical-size compacts, explicitly provide `H`/`length_scale` and use
`--scale-mode physical`.  The trainer then uses:

```text
q_hat    = q48_raw / H
X_hat    = X_keep / H
x_hat    = x / H
J_hat    = J / H
invJ_hat = H * invJ
detJ_hat = detJ / H^3
B_hat    = H * B_phys
```

In `--scale-mode physical`, Branch geometry must come from explicit physical
`X_keep` or `X_macro` fields.  The `shape4` reconstruction used by the current
TRUE176 data already produces dimensionless hat geometry, so the trainer now
refuses to treat shape4-only geometry as physical coordinates and divide it by
`H`.

The Sobolev loss is trained against `B_hat = dLE/dq_hat`, then the normal ML
standardization uses `mean/std` on the already dimensionless branch/trunk
features.  `H` may be added as a separate physical parameter later, but it is
not needed to repair pure geometric similarity scaling.

The scaling law can be checked directly:

```powershell
cd D:\IS-FEM\deeponet_macro_operator_fresh
$env:PYTHONPATH = "D:\IS-FEM\deeponet_macro_operator_fresh\src"
py scripts\validate_isoparametric_mapping.py
py scripts\validate_isoparametric_scaling.py
py scripts\validate_true176_css8_macro_mapping.py
```

The first script validates the isoparametric map itself: shape-function
interpolation, `J=dx/dxi`, `dN/dx=dN/dxi*J^-1`, Newton inverse mapping,
affine-field reproduction, and `B=dLE/dq` against finite differences.  The
second script validates similarity scaling under `X_phys=H*X_hat` and
`q_phys=H*q_hat`.  The third script validates the actual TRUE176 4x4 CSS8
macro-element route: `shape4 -> X_macro[50,3] -> 16 CSS8 elements -> 128 Gauss
rows`, including local B checks and H scaling of `X_keep`, `x`, `J`, `invJ`,
`detJ`, and `B`.

For generic data, trunk features should be stored in the compact data as
`point_features`, `ip_xyz`, `ip_J`, `ip_detJ`, etc.  In physical mode, prebuilt
`point_features` must have scale-aware names: raw physical fields such as
`ip_xyz_*`, `ip_J_*`, `ip_invJ_*`, `ip_detJ`, or explicit dimensionless names
such as `*_hat`.  Anonymous columns like `point_features_0` are rejected because
the trainer cannot know whether to divide them by `H`.  Physical arbitrary
geometry also cannot use `--point-feature-source shape4-audited` by default;
that route is only allowed with `--allow-physical-shape4-trunk` when the explicit
physical geometry is exactly an H-scaled TRUE176 shape4 geometry.

For the current TRUE176 20260620 compacts, `X_keep`, `X_macro`, and those trunk
fields were omitted, but the needed geometry can be reconstructed from the
audited shape4/CSS8 generation contract:

```text
shape4 -> X_keep[16,3] for Branch
shape4 -> 50 CSS8 nodes -> 16 CSS8 elements -> 128 standard Gauss rows for Trunk
```

The explicit training sources for this dataset are therefore:

```text
--branch-feature-mode xkeep-qraw
--point-feature-source shape4-audited
```

That is different from a blind fallback: original sample NPZ files contain
`ip_keys`, and the audit script checks that their row order is
`element 1 IP1..IP8, ..., element 16 IP1..IP8`.

This is the first step toward the general isoparametric form.  Geometry enters
the branch as q48 control-node coordinates instead of the compressed
`shape4[4]`.  The displacement variable is still `q48_raw`, because the current
label `B_LE128_forward` is `dLE/dq48_raw`.  A future `q_boundary[32,3]` branch
requires matching derivative labels with respect to that variable.  The full
`xnodes-qraw` mode remains available only as an explicit ablation/full-internal-
geometry experiment, not as the default macro-visible input.

The key files are:

```text
src/macro_deeponet/models.py
    MacroDeepONet                  # clean Hex8 prototype
    FELinearResidualDeepONet       # default TRUE176/CSS8 FE-like baseline + residual model
    True176Shape4QrawDeepONet      # standard TRUE176/CSS8 128-IP concat-skip baseline
    NOEMStyleMIONet                # NOEM/MIONet-style split-branch ablation

src/macro_deeponet/true176_data.py
    compact npz loader
    shape4 -> 128-IP point-feature builder
    Sobolev dataset wrapper

src/macro_deeponet/train_true176_deeponet_sobolev.py
    LE + AD-B Sobolev trainer

src/macro_deeponet/train_true176_generic_sobolev.py
    LE + AD-B trainer with explicit data/shape4-audited point features

scripts/audit_true176_shape4_ip_contract.py
    Checks original sample ip_keys before using shape4-audited reconstruction

scripts/launch_true176_deeponet_128ip_sobolev_ddp_linux.sh
    Linux DDP launcher matching the parameter style of fc4117d
```

## Install

Use a Python environment with PyTorch. On this machine, `py` currently resolves
to an Anaconda Python with PyTorch available.

```powershell
cd D:\IS-FEM\deeponet_macro_operator_fresh
py -m pip install -r requirements.txt
```

If PyTorch is already installed, the requirements file may not need to install
anything new.

## Smoke test

```powershell
cd D:\IS-FEM\deeponet_macro_operator_fresh
$env:PYTHONPATH = "D:\IS-FEM\deeponet_macro_operator_fresh\src"
py tests\smoke_test.py
```

## Tiny Hex8 prototype training run

```powershell
cd D:\IS-FEM\deeponet_macro_operator_fresh
$env:PYTHONPATH = "D:\IS-FEM\deeponet_macro_operator_fresh\src"
py -m macro_deeponet.train --epochs 20 --samples 2048 --batch-size 128
```

The output checkpoint is written to `runs/fresh_deeponet/best.pt` by default.

## TRUE176 128-IP Sobolev/DDP training

The main full-frame dataset used for third-machine training is:

```text
/home/ydh/桌面/zhangyihao/true176_shape4_qraw_128ip_training_data_20260620
```

On this Windows machine the corresponding copy is:

```text
E:\true176_shape4_qraw_128ip_training_data_20260620
```

It contains five compact files:

```text
cases001_020          7100 frames
batchA_cases022_060  12000 frames
batchB_cases062_100  12000 frames
batchC_cases102_139  12000 frames
batchD_cases141_176  12600 frames
total                55700 frames
```

Each compact must provide full q48/B48 arrays:

```text
shape4:           [N, 4]
q48_raw:          [N, 48]
LE128_base:       [N, 128, 6]
B_LE128_forward:  [N, 128, 6, 48]
```

On Linux:

```bash
cd /home/ydh/桌面/zhangyihao/deeponet_macro_operator_fresh
bash scripts/launch_true176_deeponet_128ip_sobolev_ddp_linux.sh
```

Important environment overrides:

```bash
BASE=/home/ydh/桌面/zhangyihao
CODE=$BASE/deeponet_macro_operator_fresh
DATA=$BASE/true176_shape4_qraw_128ip_training_data_20260620
OUT_DIR=$BASE/run_logs_128ip_fullframe_20260620/deeponet_true176_128ip_sobolev_ddp_v1
COMPACT_LIST=                  # optional override
POINT_FEATURE_SOURCE=shape4-audited
BRANCH_FEATURE_MODE=xkeep-qraw
SCALE_MODE=normalized
NPROC=3
BATCH_SIZE=4
JAC_COLS_PER_GPU=8
JACOBIAN_METHOD=forward
```

The launcher first tries `$DATA/compact_paths_linux.txt`. If it is missing, it
checks `$DATA/compact_paths.txt`; if that file contains Windows paths, the
launcher automatically writes a Linux-resolved five-compact list under
`$OUT_DIR/compact_paths_resolved.txt`.

Single-process/debug run:

```bash
PYTHONPATH=src python3 -m macro_deeponet.train_true176_generic_sobolev \
  --compact-list /path/to/compact_paths_linux.txt \
  --out-dir runs/deeponet_true176_debug \
  --epochs 2 \
  --batch-size 2 \
  --max-frames-per-compact 16 \
  --target-ips 0,1,2,3 \
  --branch-feature-mode xkeep-qraw \
  --point-feature-source shape4-audited \
  --jacobian-columns 0,1,2,3 \
  --jacobian-columns-per-batch 2 \
  --eval-columns 0,1,2,3 \
  --include-id-features
```

Windows debug run against the E-drive full dataset:

```powershell
cd D:\IS-FEM\deeponet_macro_operator_fresh
$env:PYTHONPATH = "D:\IS-FEM\deeponet_macro_operator_fresh\src"
py -m macro_deeponet.train_true176_generic_sobolev `
  --compact-list E:\true176_shape4_qraw_128ip_training_data_20260620\compact_paths.txt `
  --out-dir runs\true176_E_main5_debug `
  --epochs 1 `
  --batch-size 2 `
  --max-frames-per-compact 4 `
  --target-ips 0,1 `
  --branch-feature-mode xkeep-qraw `
  --point-feature-source shape4-audited `
  --jacobian-columns 0,1 `
  --jacobian-columns-per-batch 1 `
  --eval-columns 0,1 `
  --include-id-features
```

Audit the available original sample files before treating shape4 reconstruction
as the dimensionless TRUE176 point source:

```powershell
cd D:\IS-FEM\deeponet_macro_operator_fresh
$env:PYTHONPATH = "D:\IS-FEM\deeponet_macro_operator_fresh\src"
py scripts\audit_true176_shape4_ip_contract.py `
  --compact-list E:\true176_shape4_qraw_128ip_training_data_20260620\compact_paths.txt `
  --sample-glob "D:\IS-FEM\t176_cyl100_pilot6_cases001_020_full48\**\css8_shape4_nonzero_sample_*.npz" `
  --out runs\true176_shape4_ip_contract_audit.json
```

## Data contract for future Abaqus/fine-mesh data

Each training row should represent one macro-element state and one query point:

```text
q_b:        [n_boundary_nodes, 3]
X:          [n_geometry_nodes, 3]
xi:         [3]                 # natural coordinate in [-1, 1]^3
epsilon:    [6] or [3]           # physical strain components
```

For fine-mesh Gauss-point data, first invert the macro isoparametric map
`x = x_macro(xi)` to obtain the query coordinate `xi`, then train the operator
on `(q_b, X, xi) -> epsilon_ref`.

For the current TRUE176/CSS8 dataset, this is specialized to fixed 128 CSS8
integration points:

```text
q48_raw:       [48]
X_keep:        [16, 3]  # q48 control-node coordinates, reconstructed today
LE128_base:    [128, 6]
B_LE128_forward: [128, 6, 48]
```

The derivative supervision is always with respect to the same raw coordinate
used as model input: `q48_raw`.

For ablations or future richer data, `X_macro[50,3]` can still be stored and
used by `--branch-feature-mode xnodes-qraw`, but that exposes internal CSS8 grid
nodes and should not be treated as the default macro-element interface.

### 3. Macro16 boundary-control solid-shell route

The new v4 route is the standard macro-element interface and does not expose
fine-grid geometry to the model.

```text
contract = v4-macro16-boundary-operator-001
input    = q48_hat[48] + X16_hat[16,3] + standard Macro16 IP features
output   = LE_macro[P,6]
B        = d LE_macro / d q48_hat
```

Rules:

- Geometry input is only `X16` or `X_keep`, ordered with the 16 q48 control
  nodes.
- Displacement input is all 48 components.  Rigid modes are not removed.
- Standard integration-point features are generated from the single 16-node
  Macro16 isoparametric map.
- `X_macro`, CSS8 connectivity, `cell_id`, and `ip_local_rst` are not
  model-visible inputs in this route.
- Training labels may come from a fine FE teacher, but saved model inputs should
  contain only `X16`, `q48`, Macro16 point data, `LE_macro`, `B_macro`, and
  integration weights.

Windows smoke run:

```powershell
cd D:\IS-FEM\deeponet_macro_operator_fresh
$env:PYTHONPATH = "D:\IS-FEM\deeponet_macro_operator_fresh\src"
py -m macro_deeponet.train_macro16_boundary_sobolev `
  --compact-list D:\path\to\macro16_compact_list.txt `
  --out-dir runs\macro16_boundary_debug `
  --epochs 1 `
  --batch-size 2 `
  --jacobian-columns 0,1 `
  --eval-columns 0,1
```
