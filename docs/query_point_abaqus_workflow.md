# Query-Point Abaqus Route Workflow

This route should advance by tagged, reviewed steps.  Do not use long training
as a substitute for data-contract checks.

## Version Plan

- `query-point-abaqus-v1`: route baseline.
- `query-point-abaqus-v1.1`: hard guards for data trustworthiness.
- `query-point-abaqus-v1.2`: ID feature rules and polished inference entry point.
- `query-point-abaqus-v2`: first long-training-ready route.

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

## Change Log Rule

For every meaningful route change, update
`docs/query_point_abaqus_route_changelog.md` with:

- Date.
- Commit hash.
- User-facing behavior change.
- Validation commands.
- Remaining risks.
