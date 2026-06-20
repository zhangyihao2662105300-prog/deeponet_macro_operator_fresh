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
input  = shape4[4] + q48_raw[48]
output = LE[target_ips, 6]
B      = d LE / d q48_raw
```

The DeepONet form is:

```text
Branch input = standardized [shape4, q48_raw]       # [B,52]
Trunk input  = deterministic 128-IP CSS8 features   # [B,P,F]
Output       = standardized LE                      # [B,P,6]
AD B         = d(LE_norm)/d(q48_norm)               # [B,P,6,48]
```

The trunk features are built from the fixed 4x4 CSS8 topology and `shape4`.
They include macro natural coordinates, local element Gauss coordinates, exact
point coordinates, local frames, element Jacobian, inverse Jacobian, determinant
features, and optional element/IP id features.

The key files are:

```text
src/macro_deeponet/models.py
    MacroDeepONet                  # clean Hex8 prototype
    True176Shape4QrawDeepONet      # TRUE176/CSS8 128-IP model

src/macro_deeponet/true176_data.py
    compact npz loader
    shape4 -> 128-IP point-feature builder
    Sobolev dataset wrapper

src/macro_deeponet/train_true176_deeponet_sobolev.py
    LE + AD-B Sobolev trainer

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
PYTHONPATH=src python3 -m macro_deeponet.train_true176_deeponet_sobolev \
  --compact-list /path/to/compact_paths_linux.txt \
  --out-dir runs/deeponet_true176_debug \
  --epochs 2 \
  --batch-size 2 \
  --max-frames-per-compact 16 \
  --target-ips 0,1,2,3 \
  --jacobian-columns 0,1,2,3 \
  --jacobian-columns-per-batch 2 \
  --eval-columns 0,1,2,3 \
  --include-id-features
```

Windows debug run against the E-drive full dataset:

```powershell
cd D:\IS-FEM\deeponet_macro_operator_fresh
$env:PYTHONPATH = "D:\IS-FEM\deeponet_macro_operator_fresh\src"
py -m macro_deeponet.train_true176_deeponet_sobolev `
  --compact-list E:\true176_shape4_qraw_128ip_training_data_20260620\compact_paths.txt `
  --out-dir runs\true176_E_main5_debug `
  --epochs 1 `
  --batch-size 2 `
  --max-frames-per-compact 4 `
  --target-ips 0,1 `
  --jacobian-columns 0,1 `
  --jacobian-columns-per-batch 1 `
  --eval-columns 0,1 `
  --include-id-features
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
shape4:        [4]
q48_raw:       [48]
LE128_base:    [128, 6]
B_LE128_forward: [128, 6, 48]
```

The derivative supervision is always with respect to the same raw coordinate
used as model input: `q48_raw`.
