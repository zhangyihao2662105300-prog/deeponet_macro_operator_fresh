# Query-Point Abaqus Route Workflow

This route should advance by tagged, reviewed steps.  Do not use long training
as a substitute for data-contract checks.

## Version Plan

- `query-point-abaqus-v1`: route baseline.
- `query-point-abaqus-v1.1`: hard guards for data trustworthiness.
- `query-point-abaqus-v1.2`: data-coverage audit for independent load/case directions.
- `query-point-abaqus-v1.3`: anchored LE head diagnostics for the current raw-q route.
- `query-point-abaqus-v2`: coordinate-consistent route with useful boundary
  displacement, standard/local query coordinates, and explicit B chain rule.

## Review Order

For every route change, review in this order:

1. Route concept.
2. Data trustworthiness.
3. Hard guards in code.
4. Tests for realistic failure modes.
5. Whether long training is justified.

## Hard Questions

- Are `LE`, `B`, `ip_keys`, and `point_features` aligned row by row?
- Does merged `B_LE128_forward` come from the same q and LE ordering?
- Is `B` expressed in the same q coordinate system used by the branch input?
- Is validation separated by case or geometry, not just by frame?
- Do all compacts use the same point feature names and order?
- Can exporter audits stop an export when IP order or geometry is wrong?
- Does the launcher default to the intended route, not an older compatibility path?

## v1.1 Gate

Before any long training, the route must have:

- Non-overlapping default validation.
- Case-level or geometry-level formal split.
- Merge checks for `q48_raw`, `LE128_base`, and `ip_keys`.
- Blocking IP audit for `COORD`, and TRUE176/CSS8 `detJ` vs `IVOL` when required.
- Exact point feature name/order matching across compacts.
- Safe behavior for nonstandard Abaqus element labels.
- Query-point Abaqus launcher defaults.
- Local smoke tests and GitHub Actions smoke workflow passing.

## v1.2 Gate

Do not prioritize model architecture changes in v1.2.  The goal is to test
whether poor held-out `LE` prediction is primarily caused by insufficient
independent case/load-direction coverage after v1.1 established the data
contract, `B` link, query-B training strategy, and formal case split.

The minimum data-coverage audit should produce:

- `compact_manifest.csv`
- `compact_manifest.json`
- `q_direction_cosine_matrix.csv`
- `q_direction_abs_cosine_matrix.csv`
- `le_rms_distribution.csv`
- `audit_summary.json`
- `train_80_20_summary.json`
- `loo_summary.json`

Each case should record:

- `case_id`, compact path, frame count.
- `strain_field` and `B_label_strain_field`.
- `q_norm` min/max/mean and representative `q_direction`.
- Internal case q-direction cosine statistics.
- `q_direction_cosine_to_existing_max_abs` and cluster id.
- `LE_rms` min/max/mean and `B_rms_mean`.
- Merge guards for `q48_raw`, `LE128_base`, `ip_keys`, and strain field.
- Reference-frame IP audits for COORD and TRUE176/CSS8 `detJ` vs `IVOL`.

Keep the first v1.2 training strategy fixed:

```text
model_style = query-fe-linear-residual
point_feature_source = data
branch_feature_mode = xkeep-qraw
le_normalization = global-component
b_baseline_warmstart_source = train-only
j_loss_mode = physical
global_b_lr_scale = 0.05
point_b_lr_scale = 0.10
train_point_sample_count = 128
split_mode = case
allow_overlap_val = false
```

Use both 80/20 case split and leave-one-case-out audits before interpreting
LE generalization.  `train_80_20_summary.json` and `loo_summary.json` may start
as coverage/split plans; fill their metric fields only after the corresponding
training runs complete.

## v2 Gate

Do not start long v2 training until a one-case coordinate-consistency audit
passes.  v2 changes the learned operator coordinate contract, not just model
capacity.

The v2 branch coordinate must be explicit:

```text
q_useful = T_q_raw_to_useful @ q48_raw
```

The compact must store `q_useful`, `T_q_raw_to_useful`, and metadata describing
the local frame and rigid-body projection.  The audit must verify that stored
`q_useful` is reproduced from `q48_raw` and `T_q_raw_to_useful`.

The v2 trunk coordinate must prefer standard/local point coordinates:

```text
ip_xi, ip_J, ip_invJ, ip_detJ, local frame / geometry features
```

The v2 strain and B coordinate systems must be explicit:

```text
strain_output_coordinate
B_label_q_coordinate
B_label_output_coordinate
B_chain_rule
```

If the network predicts standard/local strain and differentiates with respect
to `q_useful`, the Sobolev comparison to Abaqus raw labels must use:

```text
B_raw_hat = T_eps_to_abq @ d(strain_standard)/d(q_useful) @ T_q_raw_to_useful
```

Directly comparing `d(strain_standard)/d(q_useful)` with the current
`B_LE128_forward = d(LE_Abaqus)/d(q48_raw)` is not valid.

Use:

```powershell
py -3 scripts/audit_v2_coordinate_consistent_contract.py `
  --compact-list <strict-fresh-compact-list.txt> `
  --out-root <v2-contract-audit-output>
```

Existing v1.x compacts are expected to fail strict v2 until the useful-q and
strain-transform metadata are generated.

## Current Data Contract Wording

Use this wording until a broader query-point label generator exists:

```text
The model architecture supports dynamic query point count P.
The current training data contract is still the Abaqus 128 integration-point
table, or subsets of that table selected by target_ips.  It is not yet a full
arbitrary-query-point labeled data contract.
```

## Change Log Rule

For every meaningful route change, update
`docs/query_point_abaqus_route_changelog.md` with:

- Date.
- Commit hash.
- User-facing behavior change.
- Validation commands.
- Remaining risks.
