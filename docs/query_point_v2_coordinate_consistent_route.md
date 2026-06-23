# v2 Coordinate-Consistent Query-Point Route

Date: 2026-06-23

## Purpose

This document defines the next route after the v1.1-v1.3 audits.

v1.1-v1.3 established that the current Abaqus query-point path can produce
strict fresh compacts, preserve `LE/B` row alignment, use a formal case split,
and train a useful anchored LE head.  The remaining issue is more fundamental
than another learning-rate, epoch, or split tweak:

```text
Which coordinate system is the learned macro-element operator actually using?
```

The v2 route should make that coordinate system explicit.  It should learn a
standard/local macro-element operator and map it back to Abaqus raw coordinates
through recorded transformations.

## Current v1.3 Boundary

The current v1.3 route learns approximately:

```text
q48_raw + point_features -> LE_Abaqus
AD -> d(LE_Abaqus) / d(q48_raw)
```

This is useful as a diagnostic and evidence path, but it is still a raw
Abaqus-coordinate black-box operator.  It can include:

- rigid translation and rotation content in the branch input,
- raw/global coordinate orientation effects in point features,
- B labels tied to `q48_raw`,
- a point-only B prior that is not explicitly state dependent.

Those choices are not necessarily wrong for a narrow diagnostic route, but they
are not the clean finite-element operator contract for the final method.

## v2 Operator Contract

The v2 route should learn:

```text
q_raw_global
  -> q_local
  -> q_useful = T_q q_raw_global

geometry_real
  -> standard/local macro-element geometry

integration point
  -> standard macro coordinates xi, eta, zeta

NN(q_useful, xi_standard, geometry_features)
  -> strain_standard_or_local
  -> LE_Abaqus
```

The Sobolev/AD comparison must also be transformed through the same contract:

```text
B_useful_std = d(strain_standard_or_local) / d(q_useful)
B_raw_hat    = T_eps_to_abq * B_useful_std * T_q
```

Only `B_raw_hat` may be compared with the current Abaqus/Sobolev raw label:

```text
B_LE128_forward = d(LE_Abaqus) / d(q48_raw)
```

It is not valid to directly compare `d(strain_standard)/d(q_useful)` with an
Abaqus raw `B_LE128_forward` label.

## Required v2 Data Fields

A v2 complete compact should retain the existing strict Abaqus fields and add
the coordinate-consistency fields below.

Existing raw Abaqus side:

```text
q48_raw:            [N,48]
LE128_base:         [N,128,6]        # raw Abaqus strain label unless renamed
B_LE128_forward:    [N,128,6,48]     # raw dLE_Abaqus/dq48_raw label
ip_keys:            [128,*] or [N,128,*]
ip_xi:              [128,3] or [N,128,3]
ip_xyz:             [128,3] or [N,128,3]
ip_J:               [128,3,3] or [N,128,3,3]
ip_invJ:            [128,3,3] or [N,128,3,3]
ip_detJ:            [128] or [N,128]
strain_field:       LE
B_label_strain_field: LE
```

New v2 branch-coordinate side:

```text
q_useful:              [N,K]
T_q_raw_to_useful:     [K,48] or [N,K,48]
q_useful_coordinate:   local_rigid_projected
q_useful_rank:         K
q_useful_definition:   text metadata describing local frame and rigid cleanup
```

The convention is:

```text
q_useful[n] = T_q_raw_to_useful @ q48_raw[n]
```

If `T_q_raw_to_useful` is frame dependent, store it as `[N,K,48]` and document
why.

New v2 strain-coordinate side:

```text
strain_output_coordinate:  standard_macro, local_jacobian_frame, local_material, or abaqus_global
T_eps_to_abq:              [6,6], [128,6,6], or [N,128,6,6]
T_eps_from_abq:            optional inverse/audit transform
strain_transform_note:     small-strain tensor rule or LE/log-strain caveat
```

For early v2 pilots, the safest contract is small-strain/local-coordinate
validation.  If the label remains Abaqus logarithmic strain, the transform must
explicitly state whether the 6-vector transform is only an approximation or a
small-deformation pilot assumption.

New v2 B-coordinate metadata:

```text
B_label_q_coordinate:       q48_raw
B_label_output_coordinate:  abaqus_global
B_chain_rule:               B_raw_hat = T_eps_to_abq @ B_useful_std @ T_q
```

Optionally, a compact may also store transformed labels:

```text
LE128_standard:       [N,128,6]
B_useful_abq:         [N,128,6,K]   # v2a pilot, output still Abaqus global
B_local_useful:       [N,128,6,K]   # v2b pilot, local Jacobian frame
B_standard_useful:    [N,128,6,K]
```

If those are stored, they must be derived from the raw labels and transforms
with an audit trail.

## v2a/v2b Pilot Status

The first two one-case pilots use `case050`:

```text
v2a: q48_raw -> q_useful, output strain remains Abaqus global
v2b: Abaqus global LE/B -> local_jacobian_frame LE/B
```

The v2b local frame is built from `ip_J` using Gram-Schmidt.  The current
exporter stores `ip_J` rows as physical derivatives with respect to local
natural coordinates, so the v2b pilot records:

```text
local_frame_source = ip_J_gram_schmidt
local_frame_ip_J_axis_convention = rows
strain_voigt_order = LE11_LE22_LE33_LE12_LE13_LE23
strain_shear_convention = tensor_shear_not_engineering_gamma
```

This is still a local Jacobian-frame tensor-component pilot.  It is not yet a
full covariant standard-coordinate strain formulation.

## Rigid-Body Cleanup

The phrase "remove rigid-body displacement" is not sufficient by itself.  v2
must store the actual linear map:

```text
q_useful = T_q q48_raw
```

This makes the derivative relation auditable:

```text
dq_useful / dq48_raw = T_q
```

The rigid-body projection should be checked with synthetic modes:

```text
T_q * q_translation_x ~= 0
T_q * q_translation_y ~= 0
T_q * q_translation_z ~= 0
T_q * q_rotation_x    ~= 0
T_q * q_rotation_y    ~= 0
T_q * q_rotation_z    ~= 0
```

For curved or shell-like macro-elements, the local frame used to define
rotations and useful displacement must be recorded.

## Standard Trunk Coordinates

The trunk should prefer standard/local integration-point coordinates:

```text
ip_xi / xi_macro / local_r,s,t
```

Geometry and mapping features should remain explicit:

```text
ip_J, ip_invJ, ip_detJ, local frame, X_keep or X_macro
```

This separates the universal query coordinate from the real geometry used for
the isoparametric map.  The network should not have to infer the standard
coordinate system from raw Abaqus `COORD` alone.

## Minimal v2 Pilot

The first v2 pilot should not be a large training run.  Use one strict fresh
case and verify only the coordinate chain:

```text
1. read q48_raw, LE_Abaqus, B_raw, ip_xi, ip_J, ip_invJ, ip_detJ
2. build q_useful = T_q q48_raw
3. build or store strain_standard/local labels
4. verify q_useful reconstruction from T_q
5. verify T_eps_to_abq shape and metadata
6. run an oracle or tiny differentiable map:
      B_raw_hat = T_eps_to_abq @ B_useful_std @ T_q
7. compare B_raw_hat with B_LE128_forward
```

Only after this chain-rule audit passes should v2 training start.

## v2c One-Case Oracle / Tiny Smoke Status

The one-case `case050` v2c smoke extends the v2b local-strain pilot with two
small checks:

```text
1. oracle: LE_local_Bq = B_local_useful @ q_useful
2. tiny training: q_useful + ip_xi -> LE_local_jacobian_frame
```

The oracle chain-rule return remains numerical-zero in projected raw B space:

```text
B_raw_hat_projected_rel = 4.573742799247327e-16
B_raw_hat_raw_rel = 0.00014709789545713997
```

The finite-path linear oracle has:

```text
LE_local_Bq_rel = 0.04432110494813806
LE_local_Bq_cos = 0.9990565149944766
```

The tiny script-local anchored model, using `q_useful` as branch input and
`ip_xi` as trunk input, reached:

```text
best_step = 1800
train_LE_local_rel = 0.0070029981434345245
train_AD_B_local_rel = 0.004019410815089941
train_AD_B_local_cos = 0.9999919533729553
zero_q_LE_local_rms = 0.0
B_model_raw_projected_rel = 0.004044899716973305
B_model_raw_rel = 0.004047574009746313
```

This means the v2 coordinate contract is executable by a network and autograd
for one case.  It is not formal training, not a multi-case generalization
result, and still uses `local_jacobian_frame` tensor components rather than a
complete covariant standard-coordinate strain formulation.

## Reusable Existing Pieces

The current repository already has useful pieces:

- `ip_xi`, `ip_J`, `ip_invJ`, and `ip_detJ` support in the compact schema.
- strict strain and IP guards from v1.1/v1.2.
- q48-to-boundary96 bridge for fresh Abaqus boundary control.
- isoparametric mapping and scaling validators.
- anchored LE evidence showing that `LE(0)=0` is a necessary structural rule.

These should be reused, but the v2 model should not keep treating raw
`q48_raw` as the final branch coordinate without an explicit `T_q`.

## Non-Goals For This Step

Do not do the following as part of v2 contract definition:

- train a new model,
- tune loss weights, learning rate, epochs, or split strategy,
- use old TRUE176 `LE/B` as v2 labels,
- directly compare standard-coordinate AD-B with raw Abaqus B labels,
- move the `query-point-abaqus-v1.1` tag.

## Current Decision

The route should transition from v1.3 diagnostics to:

```text
v2 coordinate-consistent route
```

The next concrete step is a one-case contract audit, not a full LOO run and not
a long training run.
