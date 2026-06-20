# Literature Trace

## Identified paper

The PDF metadata identifies:

```text
Title: NOEM: efficient and scalable finite element method enabled by reusable neural operators
Authors: Weihang Ouyang, Yeonjong Shin, Si-Wei Liu, Lu Lu
Journal: Nature Computational Science
Volume/pages: 6, 417-429
Year: 2026
DOI: 10.1038/s43588-026-00974-2
```

The article page and code availability section point to:

- https://github.com/lu-group/noem
- https://doi.org/10.5281/zenodo.18678157

## What is reused conceptually

NOEM uses reusable neural operators as finite-element-like subdomain response
models. The relevant conceptual transfer is:

```text
subdomain boundary/state input + local query location -> interior field
```

For the macro-element strain operator here, this becomes:

```text
q_b, X, xi -> epsilon_phys(xi)
```

and

```text
B_macro(xi) = d epsilon_phys(xi) / d q_b
```

## What is intentionally not reused

This fresh project does not depend on:

- previous local IS-FEM code or data,
- NOEM experiment scripts for heat transfer, Darcy flow, or 1D PDEs,
- any previously trained Route B artifacts.

The code is a new PyTorch implementation aimed at the macro-element strain
operator interface.

