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

No previous repository data or scripts are required.

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

## Tiny training run

```powershell
cd D:\IS-FEM\deeponet_macro_operator_fresh
$env:PYTHONPATH = "D:\IS-FEM\deeponet_macro_operator_fresh\src"
py -m macro_deeponet.train --epochs 20 --samples 2048 --batch-size 128
```

The output checkpoint is written to `runs/fresh_deeponet/best.pt` by default.

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
