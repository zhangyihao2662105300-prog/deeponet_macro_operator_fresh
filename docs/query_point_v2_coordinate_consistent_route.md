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

## v2l Standard-Operator Preprocessing Contract Status

The v2l step pauses model tuning and extracts preprocessing/postprocessing into
an explicit standard-operator contract.  This is not a new model-performance
claim.

New lightweight utilities:

```text
scripts/build_v2_standard_operator_compacts.py
scripts/audit_v2_standard_operator_contract.py
docs/query_point_v2_standard_operator_preprocessing_contract.md
```

The standard route is now:

```text
Raw Abaqus:
  q48_raw, LE_abq, B_abq

Preprocess:
  q_useful = T_q_raw_to_useful @ q48_raw
  LE_local = T_eps_from_abq @ LE_abq
  B_local_useful = T_eps_from_abq @ B_abq @ T_q_raw_to_useful.T

Network-visible operator:
  LE_local = NN(q_useful, ip_xi)
  B_local_useful_hat = dLE_local_hat / dq_useful

Postprocess:
  LE_abq_hat = T_eps_to_abq @ LE_local_hat
  B_raw_hat = T_eps_to_abq @ B_local_useful_hat @ T_q_raw_to_useful
```

The network should only read:

```text
q_useful
ip_xi
LE_local
B_local_useful
```

Transform and raw fields are preprocessing/postprocessing/audit fields, not
quantities for the network to learn:

```text
q48_raw
LE_abq
B_LE128_forward
T_q_raw_to_useful
T_eps_to_abq
T_eps_from_abq
local_frame_Q
```

10-case standard-operator generation and strict audit:

```text
standard_operator_compact_list =
  D:\IS-FEM\outputs\query_point_v2_standard_operator\multi_case_min10\standard_operator_compact_list.txt

standard_operator_summary =
  D:\IS-FEM\outputs\query_point_v2_standard_operator\multi_case_min10\standard_operator_summary.json

compact_count = 10
strict_pass_count = 10
strict_fail_count = 0
q_useful_transform_rel = 0.0 to 0.0
LE_local_to_abq_roundtrip_rel = 1.6410479539884723e-16 to 3.0740682511657065e-16
B_raw_projected_rel = 4.462975556507417e-16 to 4.652329014360367e-16
B_rigid_residual_rel = 0.00010968263851608675 to 0.0012638931647260517
```

The rigid residual is reported only.  It is not a strict failure because the
standard operator intentionally removes rigid q modes.

Random-geometry isoparametric invariance audit:

```text
script =
  scripts/test_random_geometry_isoparametric_scaling.py

report =
  D:\IS-FEM\outputs\query_point_v2_standard_operator\random_geometry_isoparametric_invariance.json

passed = true
trials = 64
total_points = 512
detJ_min = 0.29031834735368334
ip_xi_invariance_abs = 5.384581669432009e-14
rotation_scale_J_abs = 2.6645352591003757e-15
rotation_scale_invJ_abs = 1.687538997430238e-14
rotation_scale_detJ_abs = 3.410605131648481e-13
B_scaling_abs = 1.1102230246251565e-15
B_scaling_rel = 8.403417513265492e-16
LE_scaled_similarity_abs = 1.7763568394002505e-15
T_eps_roundtrip_abq_rel = 7.168741415637943e-15
T_eps_roundtrip_local_rel = 7.060577084968476e-15
LE_local_to_abq_roundtrip_rel = 3.2594517417223477e-16
B_raw_projected_rel = 5.627036520025794e-16
```

This extends the standard-operator audit from fixed fresh compacts to random
positive-orientation Hex8 geometries under translation, rotation, scaling, and
small geometric distortion.  It is still a geometry/contract audit, not a model
learning result.

Boundary:

```text
no training
no model/loss/lr/epoch/split changes
no old TRUE176 LE/B labels used as v2 labels
no large generated artifacts committed
```

## v3 CSS8 Curved-Shell Contract Clarification

The actual element route uses CSS8 continuum-shell / solid-shell subelements.
This requires one more precision step beyond the v2 standard-operator wording:

```text
The standard domain is a piecewise 4x4 CSS8 parent domain,
not one single flattened HEX8 block.
```

New contract document:

```text
docs/query_point_v3_css8_curved_shell_standard_operator_contract.md
```

Audit script:

```text
scripts/audit_css8_curved_shell_standard_operator_contract.py
```

Audit result:

```text
compact_count = 10
strict_css8_geometry_pass_count = 10
strict_css8_geometry_fail_count = 0
```

Key decisions:

```text
current ip_xi = local CSS8 parent r,s,t
current ip_xi is not enough to identify the 4x4 macro-patch location
final trunk must add ip_macro_xi or equivalent subcell_id + local r,s,t
CSS8 geometry is exactly x(r,s,t)=x_mid(r,s)+0.5*t*director(r,s)
final local strain frame should be Q_stack with local-3 parallel to g_t=dX/dt
current v2 local frame is close but is not the final CSS8 stack-director frame
```

Current measured gap:

```text
current_Q_rel_to_stack_Q_min = 0.015430432383667826
current_Q_rel_to_stack_Q_max = 0.024697166064443367
```

Interpretation:

```text
The CSS8 geometry contract is now clear and audited.
The current v2 standard compacts are useful diagnostics, but they are not the
final CSS8 standard-operator compact because they lack ip_macro_xi and use the
older surface-normal local frame.
```

For CSS8, the standard-domain answer is now fixed:

```text
Use a 4x4 piecewise CSS8 parent patch:
  macro coordinate: (xi_macro, eta_macro, zeta_macro)
  local coordinate: (r, s, t) per CSS8 subelement
```

Do not collapse the curved shell macro element into one flattened HEX8.  A
curved CSS8 patch has an integration-point-dependent stack direction.  The
final local strain coordinate must therefore be `Q_stack(point)`, with local-3
parallel to `dX/dt`, and the final trunk coordinate must include
`ip_macro_xi` or an equivalent `subcell_id + local r,s,t` representation.

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

## v2d Multi-Case Compact Generation Status

The v2d step extends the v2b local-strain compact contract to the current
10-case strict fresh pool:

```text
[19, 25, 31, 41, 43, 44, 45, 46, 49, 50]
```

Source manifest:

```text
D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt
```

Generated v2b compact list:

```text
D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt
```

Generation status:

```text
compact_count = 10
pass_count = 10
fail_count = 0
```

Key audit ranges:

```text
q_useful_removed_rigid_rel:        0.18971368414698137 .. 0.7451366595918001
B_rigid_residual_rel:              0.0001096826385160804 .. 0.0012638931647260521
B_local_chain_rule_projected_rel:  max 4.619812329621079e-16
LE_local_roundtrip_rel:            max 3.0740682511657065e-16
```

Every generated compact passed strict-v2 coordinate audit.  The large
`q_useful_removed_rigid_rel` in some cases means the raw q48 samples can contain
substantial rigid-mode content; this is an audit observation for later training,
not a compact failure.

This step prepares the next possible v2 formal tiny multi-case training smoke.
It does not train a model and does not prove v2 generalization.

## v2e Multi-Case Tiny Training Smoke Status

The v2e smoke uses the 10-case v2b compact list:

```text
D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt
```

Split:

```text
train_cases = [19,25,31,41,43,45,49,50]
val_cases = [44,46]
validation_is_overlapping = false
```

The script-local tiny model uses:

```text
branch input = q_useful
trunk input = ip_xi
output = LE_local_jacobian_frame
AD target = d(LE_local_jacobian_frame)/d(q_useful)
raw backprojection = T_eps_to_abq @ AD_B_local_hat @ T_q_raw_to_useful
```

No `case_id` is used as input.

The execution chain completed and wrote:

```text
D:\IS-FEM\outputs\query_point_v2_tiny_smoke\multi_case_min10_smoke\training_summary.json
D:\IS-FEM\outputs\query_point_v2_tiny_smoke\multi_case_min10_smoke\per_case_attribution.csv
```

Best by smoke selection score was initialization:

```text
best_step = 0
train_LE_local_rel = 29.468563079833984
val_LE_local_rel = 10.836355209350586
train_AD_B_local_rel = 0.29527074098587036
val_AD_B_local_rel = 0.1957888901233673
train_AD_B_local_cos = 0.9554136395454407
val_AD_B_local_cos = 0.9939731955528259
```

Latest step:

```text
latest_step = 3000
train_LE_local_rel = 1.1830968856811523
val_LE_local_rel = 18.03921890258789
train_AD_B_local_rel = 0.3041848838329315
val_AD_B_local_rel = 0.2089325487613678
train_AD_B_local_cos = 0.9526196718215942
val_AD_B_local_cos = 0.9906823635101318
zero_q_LE_local_rms = 0.0
```

This proves the multi-case v2b pool can be loaded, split, trained through a
tiny script-local network, differentiated with respect to `q_useful`, and mapped
back to raw B.  It does not prove v2 generalization.  In fact, this no-case-id
tiny model reduces train LE but worsens validation LE, so the next step should
be formal v2 model/normalization design rather than ad hoc tuning of the smoke.

## v2f Formal Normalization / Model Design Status

v2f analyzes the 10-case v2b compact pool before implementing a formal v2
model.  No formal model training is performed.

Stats output:

```text
D:\IS-FEM\outputs\query_point_v2_design_audit\multi_case_min10_stats\stats_summary.json
```

Key findings:

```text
train q_norm_mean / val q_norm_mean = 27.98834726646046
train LE_local_rms / val LE_local_rms = 41.8936927649456
train B_local_rms / val B_local_rms = 0.8704529505657129
q component std ratio = 214.9691629443162
q covariance rank estimate = 10
B q-column scale ratio = 9.299907348162048
```

The v2e validation cases are low-amplitude cases.  Their B@q oracle is good:

```text
case044 B@q rel = 0.049291182462898236
case046 B@q rel = 0.02763180576935294
```

The global B@q error is dominated by high-amplitude `case031`:

```text
case031 B@q rel = 0.6423729454305975
```

Recommended normalization is a minimal A+C hybrid:

```text
q_useful component std normalization
LE_local six-component normalization
B_local induced normalization B_norm[a,k] = B_local[a,k] * q_std[k] / LE_std[a]
explicit q_amp descriptor
optional q_dir descriptor with zero-q guard
```

Recommended formal v2 model:

```text
branch = q_useful_normed + q_amp (+ optional q_dir)
trunk = ip_xi (+ optional geometry/Jacobian descriptors)
output = LE_local_jacobian_frame
anchored zero-q structure
AD_B_local and raw-B backprojection retained for evaluation
```

The next step should be v2g formal prototype implementation with train-only
normalization artifacts and explicit amplitude descriptors, not more tuning of
the v2e smoke script.

## v2g Formal Prototype Audit Status

v2g implements the v2f recommendation as a script-local formal prototype:

```text
script = scripts/train_v2_formal_prototype.py
compact_list = D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt
train_cases = [19,25,31,41,43,45,49,50]
val_cases = [44,46]
validation_is_overlapping = false
use_q_amp = true
use_q_dir = false
use_geometry_features = false
```

Outputs:

```text
D:\IS-FEM\outputs\query_point_v2_formal_prototype\split_B_min10\training_summary.json
D:\IS-FEM\outputs\query_point_v2_formal_prototype\split_B_min10\normalization_summary.json
D:\IS-FEM\outputs\query_point_v2_formal_prototype\split_B_min10\per_case_attribution.csv
docs/query_point_v2_formal_prototype_audit.md
```

Train-only normalization was used:

```text
q_std_ratio_before_floor = 214.35575102954596
q_std_floored_count = 0
LE_std = [0.006498310714960098, 0.0038321511819958687,
          0.0040618302300572395, 0.004930346272885799,
          0.003132438752800226, 0.003945972304791212]
LE_std_floored_count = 0
q_amp_rms = 0.08326140530721668
```

Best combined checkpoint-equivalent metric:

```text
step = 250
val_LE_local_rel = 10.357998847961426
val_LE_local_rmse = 0.0011231731623411179
val_LE_target_rms = 0.000108435342554003
val_AD_B_local_rel = 0.24915096163749695
val_AD_B_local_cos = 0.9772958159446716
val_B_model_raw_projected_rel = 0.25252825021743774
zero_q_LE_local_rms = 0.0
```

Latest metric:

```text
step = 5000
train_LE_local_rel = 28.909543991088867
val_LE_local_rel = 13.804399490356445
train_AD_B_local_rel = 0.41042250394821167
val_AD_B_local_rel = 0.29874974489212036
val_B_model_raw_projected_rel = 0.3056311309337616
zero_q_LE_local_rms = 0.0
```

Conservative interpretation:

```text
v2g formal prototype audit executes.
The train-only normalization, q_amp descriptor, anchored zero-q structure,
AD-B local metrics, and raw-B backprojection metrics are now auditable.
Model effect is not established.
```

Compared with v2e latest, v2g best reduces held-out LE relative error, but train
LE relative error remains high and AD-B/raw-projected metrics degrade relative
to the v2e latest baseline.  This means the v2 coordinate contract and formal
prototype machinery are in place, while low-amplitude value-field learning
remains unsolved.

## v2h Formal Prototype Sanity / Ablation Audit Status

v2h audits why v2g does not reduce multi-case train LE.  It adds:

```text
scripts/audit_v2_formal_prototype_sanity.py
docs/query_point_v2_formal_prototype_sanity_audit.md
```

No-training sanity output:

```text
D:\IS-FEM\outputs\query_point_v2_design_audit\v2g_sanity\sanity_summary.json
```

Normalization roundtrips are closed:

```text
train q_roundtrip_rel = 4.792623713739907e-17
train LE_roundtrip_rel = 3.441183768746357e-17
train B_roundtrip_rel = 3.787348081354616e-17
val q_roundtrip_rel = 4.0597878073265377e-17
val LE_roundtrip_rel = 3.354931535711588e-17
val B_roundtrip_rel = 3.759929863285529e-17
```

Baselines:

```text
exact B@q train_LE_local_rel = 0.6407233225614739
exact B@q val_LE_local_rel = 0.037104433513767326

train-mean B prior train_LE_local_rel = 29.466125499590156
train-mean B prior val_LE_local_rel = 10.826594009457496
train-mean B prior train_AD_B_local_rel = 0.2952528280656775
train-mean B prior val_AD_B_local_rel = 0.195877397799659
```

The normalized train-mean B prior matches the physical train-mean B prior,
which means the normalized B-prior implementation is not the reason for the
large train LE error.

Single-case `case050` overfit passes:

```text
train_LE_local_rel = 0.03293348103761673
train_AD_B_local_rel = 0.0039431145414710045
train_AD_B_local_cos = 0.9999921321868896
```

Freeze-B-prior and gate-open ablations do not solve multi-case train LE.  The
current narrowed diagnosis is:

```text
normalization bug: unlikely
B-prior normalized/physical mismatch: unlikely
B-prior table corruption: unlikely
gate suppression: unlikely as the main cause
single-case implementation capacity: passes
multi-case formal split learning: still fails
```

The likely remaining issue is that the v2g branch/trunk residual is not
expressive or well-conditioned enough to learn a case/state-dependent
value-field correction on the mixed 8-case train pool, starting from a
train-mean B prior whose `B@q` value prediction is poor for high-amplitude/mixed
train cases.  Broader v2 training should wait until this value-field correction
mechanism is diagnosed.

## v2i Value-Field Correction Diagnostic Status

v2i diagnoses why the formal v2 residual/value branch does not learn the
multi-case LE value-field correction.  It adds:

```text
scripts/diagnose_v2_value_field_correction.py
docs/query_point_v2_value_field_correction_diagnostic.md
```

It also extends `scripts/train_v2_formal_prototype.py` with diagnostic-only
support for:

```text
--write-per-case-history
--use-amp-regime-descriptor
per-case Bq oracle / B-prior / residual attribution
```

Outputs are kept outside git:

```text
D:\IS-FEM\outputs\query_point_v2_design_audit\v2i_value_correction\
```

Key diagnostic results:

```text
LE-only best:
  train_LE_local_rel = 28.903217315673828
  val_LE_local_rel = 10.226631164550781

LE+B best:
  train_LE_local_rel = 28.91905975341797
  val_LE_local_rel = 10.357998847961426

remove case031 best:
  train_LE_local_rel = 36.591793060302734
  val_LE_local_rel = 7.851358890533447

q_dir best:
  train_LE_local_rel = 28.908042907714844
  val_LE_local_rel = 10.261488914489746

amplitude-regime descriptor best:
  train_LE_local_rel = 28.912452697753906
  val_LE_local_rel = 10.579803466796875
```

These runs are diagnostic only.  They do not establish model performance.

Value-anchor baselines narrow the issue further:

```text
global train-mean B prior:
  train_LE_local_rel = 29.466125499590156
  val_LE_local_rel = 10.826594009457496

clustered B prior by q_amp median:
  train_LE_local_rel = 21.19461440553013
  val_LE_local_rel = 5.700911624442508

clustered B prior by q_amp p75:
  train_LE_local_rel = 4.77639109241935
  val_LE_local_rel = 5.700373146340969

per-case train B-prior upper bound:
  train_LE_local_rel = 0.29293951890558007
  train_AD_B_local_rel = 0.027328778282940253
```

Current diagnosis:

```text
AD-B loss alone is not the main cause.
case031 alone is not the main cause.
q_dir alone is insufficient.
simple nonlinear q_amp descriptors are insufficient.
The point-only global train-mean B prior is a poor value anchor.
State/regime-dependent B prior is strongly indicated.
```

The recommended next step is a v2j prototype that changes the value anchor /
B-prior mechanism before broadening training:

```text
B_prior_norm(point, q_amp, q_dir or q_cluster)
```

Start with an auditable low-risk variant such as a two-regime or
amplitude-interpolated B prior, while preserving the anchored zero-q structure,
AD-B-local metrics, and raw-B backprojection metrics.

## v2j State-Dependent B Prior Prototype Status

v2j implements a low-risk diagnostic version of the v2i recommendation.  It
does not change the residual network size and does not make a formal
performance claim.

Added:

```text
scripts/audit_v2_state_dependent_b_prior.py
docs/query_point_v2_state_dependent_b_prior_audit.md
```

`scripts/train_v2_formal_prototype.py` now supports:

```text
--b-prior-mode global_mean
--b-prior-mode amp_median_cluster
--b-prior-mode amp_p75_cluster
--b-prior-mode amp_linear_interp
--detach-b-prior-regime-weight
--no-detach-b-prior-regime-weight
```

The default remains `global_mean`, preserving prior behavior.

No-training anchor audit:

```text
global_mean:
  train_LE_local_rel = 29.46612548828125
  val_LE_local_rel = 10.826595306396484
  train_AD_B_local_rel = 0.29525285959243774
  val_AD_B_local_rel = 0.19587740302085876

amp_p75_cluster:
  threshold = 0.005826574499032849
  cluster sizes = 60 low / 20 high
  train_LE_local_rel = 4.77639102935791
  val_LE_local_rel = 5.700374126434326
  train_AD_B_local_rel = 0.22865082323551178
  val_AD_B_local_rel = 0.10279608517885208
```

This reproduces the v2i p75 anchor baseline and confirms that the catastrophic
global-mean value anchor is not inherent to the v2 coordinate contract.

Training diagnostic:

```text
amp_p75_cluster best_combined:
  step = 1250
  train_LE_local_rel = 4.701291084289551
  val_LE_local_rel = 5.470116138458252
  train_AD_B_local_rel = 0.31725096702575684
  val_AD_B_local_rel = 0.11935402452945709
  val_AD_B_local_cos = 0.9963414072990417
  val_B_model_raw_projected_rel = 0.12048347294330597
  zero_q_LE_local_rms = 0.0
```

Conservative interpretation:

```text
state/regime-dependent B prior clearly improves the LE value anchor;
the p75-cluster prototype starts from a much less catastrophic train_LE state;
short residual training gives only small LE improvement and can degrade AD-B;
model performance is not established.
```

The next v2 step should refine state-dependent B-prior/tangent handling rather
than simply widening the residual branch:

```text
B_prior(point, state/regime)
```

should stay on the route, but its regime gate and AD-B tangent definition need
another targeted audit before formal training claims.

## v2k Tangent / Training-Schedule Diagnostic Status

v2k tests how to supervise AD-B after the state-dependent p75-cluster B prior
is introduced.  It adds:

```text
scripts/diagnose_v2_tangent_schedule.py
docs/query_point_v2_tangent_schedule_audit.md
```

`scripts/train_v2_formal_prototype.py` now supports:

```text
--b-loss-target-mode full_output
--b-loss-target-mode anchor_only
--b-loss-target-mode residual_only
--b-loss-target-mode detached_anchor_plus_residual
--training-schedule joint
--training-schedule anchor_then_residual
--training-schedule le_warmup_then_b
--warmup-steps
```

Defaults preserve prior behavior:

```text
b_loss_target_mode = full_output
training_schedule = joint
```

All v2k runs use:

```text
b_prior_mode = amp_p75_cluster
detach_b_prior_regime_weight = true
```

Shared anchor:

```text
train_LE_local_rel = 4.77639102935791
val_LE_local_rel = 5.700374126434326
train_AD_B_local_rel = 0.22865082323551178
val_AD_B_local_rel = 0.10279608517885208
val_B_model_raw_projected_rel = 0.10271253436803818
```

Best nonzero-step tradeoffs:

```text
A full_output + joint:
  step = 1250
  train_LE_local_rel = 4.701291084289551
  val_LE_local_rel = 5.470116138458252
  train_AD_B_local_rel = 0.31725096702575684
  val_AD_B_local_rel = 0.11935402452945709
  val_B_model_raw_projected_rel = 0.12048347294330597

B anchor_only + joint:
  step = 500
  train_LE_local_rel = 4.682076454162598
  val_LE_local_rel = 5.659628391265869
  train_AD_B_local_rel = 0.2719678580760956
  val_AD_B_local_rel = 0.11224554479122162
  val_B_model_raw_projected_rel = 0.11289863288402557
```

Current v2k interpretation:

```text
No tested tangent/schedule closes the LE/AD-B tradeoff.
The p75 state-dependent anchor remains the strongest route component.
Anchor-only B supervision is the safest current tangent/loss reference.
Full-output supervision gives the best val LE in this audit, but with larger
AD-B/raw-B degradation.
LE warmup does not improve the selected metric.
```

This reinforces the next route direction:

```text
v2l should refine the state-dependent B prior itself, rather than simply
widening the residual network or relying on training schedule tweaks.
```

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
