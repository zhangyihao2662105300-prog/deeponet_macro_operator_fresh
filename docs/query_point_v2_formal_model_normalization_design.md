# v2f Formal v2 Normalization and Model Design

Date: 2026-06-23

## Purpose

v2e proved the engineering execution chain:

```text
multi-case v2b compact list
-> non-overlapping case split
-> q_useful + ip_xi -> LE_local_jacobian_frame
-> AD_B_local
-> raw-B backprojection
-> per-case attribution
```

It did not prove model performance.  The tiny no-case-id smoke reduced train
`LE_local_rel` from `29.47` to `1.18`, but validation `LE_local_rel` worsened
from `10.84` to `18.04`.  Therefore v2f does not continue ad hoc tuning of
steps, learning rate, or hidden size.  It audits data scales and defines the
formal v2 normalization/model design.

No formal training is performed in v2f.

## Inputs

v2b compact list:

```text
D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\multi_case_v2b_min10\v2b_compact_list.txt
```

Stats output:

```text
D:\IS-FEM\outputs\query_point_v2_design_audit\multi_case_min10_stats\stats_summary.json
D:\IS-FEM\outputs\query_point_v2_design_audit\multi_case_min10_stats\per_case_stats.csv
D:\IS-FEM\outputs\query_point_v2_design_audit\multi_case_min10_stats\component_stats.csv
D:\IS-FEM\outputs\query_point_v2_design_audit\multi_case_min10_stats\q_column_stats.csv
D:\IS-FEM\outputs\query_point_v2_design_audit\multi_case_min10_stats\bq_oracle_stats.csv
```

Split analyzed:

```text
train_cases = [19,25,31,41,43,45,49,50]
val_cases   = [44,46]
validation_is_overlapping = false
```

## Data Statistics

### q_useful

Global `q_useful` scale:

```text
q_useful_rms = 0.011491464788205397
q_norm_mean = 0.024060026482276273
q_norm_max = 0.3783465445747462
q component std min/max = 0.00014299915352592528 / 0.03074040833521392
q component std ratio = 214.9691629443162
near-zero q component count = 0
covariance rank estimate = 10
covariance condition estimate = 3705605606.4196844
```

The component standard deviations span more than two orders of magnitude, and
the covariance rank estimate is only 10 for 42 useful coordinates.  The current
10-case pool is therefore direction-limited even after rigid-mode cleanup.

Split scale:

```text
train q_norm_mean = 0.029808772538628757
val q_norm_mean   = 0.001065042256866299
train/val ratio   = 27.98834726646046
```

The validation cases in this split are very low-amplitude compared with the
training cases.

### LE_local

Global `LE_local` scale:

```text
LE_local_rms = 0.004063454592726386
LE component rms min/max = 0.0028023930215038953 / 0.005822362112956812
LE component scale ratio = 2.0776393847256514
```

The six local strain components are not wildly imbalanced globally.

Split scale:

```text
train LE_local_rms = 0.0045427568147768434
val LE_local_rms   = 0.00010843533990343243
train/val ratio    = 41.8936927649456
```

This is the main v2e warning: validation LE amplitude is about 42 times smaller
than train LE amplitude.  A raw relative LE metric on this split is highly
sensitive to small absolute errors in low-amplitude held-out cases.

### B_local

Global `B_local` scale:

```text
B_local_rms = 8.95431355354351
B component rms min/max = 2.467413247526602 / 14.594662559201979
B component scale ratio = 5.914964821491512
B q-column rms min/max = 1.4314227208302548 / 13.3120986797754
B q-column scale ratio = 9.299907348162048
B q-column near-zero count = 0
```

B has stronger component and q-column scale imbalance than LE.  A formal B loss
should not treat all components/columns as equally scaled raw numbers.

Split scale:

```text
train B_local_rms = 8.680994674140301
val B_local_rms   = 9.972962546107135
train/val ratio   = 0.8704529505657129
```

Unlike LE and q amplitude, B scale is similar between train and val.  This helps
explain why v2e AD-B metrics are computable and relatively stable while LE value
generalization is poor.

## B@q Oracle

Oracle:

```text
LE_local_Bq = B_local_useful @ q_useful
```

Global:

```text
global LE_local_Bq_rel = 0.6406777204317293
global LE_local_Bq_cos = 0.9222083945148773
per-case rel min/max = 0.026449914297305376 / 0.6423729454305975
per-case cos min/max = 0.9221427482686505 / 0.9996504202973727
```

Per-case summary:

| case | split | q_norm_mean | LE_rms | B@q rel | B@q cos | B train-mean rel |
|---:|---|---:|---:|---:|---:|---:|
| 19 | train | `0.015198506883004847` | `0.0007938938021386212` | `0.06301661726489277` | `0.9980405277220696` | `0.5395765089981283` |
| 25 | train | `0.002181283665790775` | `0.00012961141553593055` | `0.026449914297305376` | `0.9996504202973727` | `0.5401723805751826` |
| 31 | train | `0.2080905986995055` | `0.012815580464879342` | `0.6423729454305975` | `0.9221427482686505` | `0.607753139850961` |
| 41 | train | `0.0010874863209921893` | `9.279569898301716e-05` | `0.04859275487183188` | `0.9988191465108142` | `0.19587228463309675` |
| 43 | train | `0.0027628200519251253` | `0.0002055836260716503` | `0.05933670307880739` | `0.9984568165587849` | `0.19633047856874797` |
| 44 | val | `0.0010563158044482241` | `9.303416901619705e-05` | `0.049291182462898236` | `0.9987848808676664` | `0.19587904301426834` |
| 45 | train | `0.002273190723595366` | `0.00018566410101577253` | `0.039121364532986515` | `0.9992346130196608` | `0.19530533187711832` |
| 46 | val | `0.001073768709284374` | `0.00012190606742657043` | `0.02763180576935294` | `0.9996214836905544` | `0.195875751978168` |
| 49 | train | `0.0029919110033185875` | `0.0002170997948334918` | `0.03346617689712526` | `0.9994653350599667` | `0.19654158275548464` |
| 50 | train | `0.003884382960897629` | `0.00027288740777478717` | `0.04432110494813806` | `0.9990565149944766` | `0.19645774957849785` |

The B@q oracle is good for low-amplitude cases, including both validation cases.
The global B@q error is dominated by high-amplitude `case031`, where the finite
load path is more nonlinear.

Correlations:

```text
Bq_rel_vs_q_norm_mean = 0.9981798243696219
Bq_rel_vs_LE_local_rms = 0.9984726534079255
Bq_rel_vs_q_removed_rel = -0.4297761590222323
```

This indicates that B@q linearization error is primarily amplitude/LE-scale
driven, not driven by removed rigid-mode content.

## Why v2e Val LE Was Bad

The most likely causes are:

1. The split has severe amplitude mismatch.  Train q/LE scales are roughly
   28x/42x larger than val q/LE scales.
2. The B prior is not the main issue for val cases.  Train-mean B prior relative
   error is about `0.196` on both val cases, matching the v2e val AD-B level.
3. The tiny model has no case/load/amplitude descriptor and can only learn an
   average value-field correction over a mixed-amplitude train set.
4. LE and B losses are on very different raw scales.  B is similar between train
   and val, while LE is not; relative LE on low-amplitude val cases is easily
   hurt by small absolute offsets.
5. The current 10-case q covariance rank estimate is 10, so the 42-dimensional
   useful branch space is not well covered.

Conservative conclusion:

```text
v2e exposed normalization and descriptor design issues.
It did not show that the v2 coordinate contract is wrong.
```

## Normalization Options

### Option A: Global-Component Normalization

Definition:

```text
q_normed[k] = (q_useful[k] - mean_q[k]) / std_q[k]
LE_normed[a] = LE_local[a] / std_LE[a]
B_normed[a,k] = B_local[a,k] * std_q[k] / std_LE[a]
```

Advantages:

- Simple and auditable.
- B normalization follows the chain rule induced by q and LE scales.
- Keeps physical amplitude inside normalized q.

Risks:

- Does not solve train/val amplitude imbalance by itself.
- q covariance is rank-limited and ill-conditioned; component std alone may
  amplify weak directions.

### Option B: Case-Amplitude Normalization

Definition:

```text
alpha_case = representative q norm or load amplitude
q_amp_scaled = q_useful / alpha_case
LE_amp_scaled = LE_local / alpha_case
B_amp_scaled ~= B_local
```

Advantages:

- Reduces amplitude mismatch between cases.
- Separates direction learning from load magnitude in nearly linear regimes.

Risks:

- Requires a robust amplitude definition at inference.
- Nonlinear/high-amplitude cases such as `case031` are not perfectly linear, so
  pure amplitude normalization can hide important physics.

### Option C: Hybrid Direction-Amplitude Design

Definition:

```text
q_amp = ||q_useful||
q_dir = q_useful / max(q_amp, eps)
branch = [q_dir, log_or_scaled_q_amp, optional q_raw/useful stats]
LE target = global-component normalized LE_local
B target = component/q-column normalized B_local
```

Advantages:

- Lets the network explicitly distinguish load direction from load amplitude.
- Keeps amplitude available rather than silently dividing it away.
- Better matches the observed issue: val cases are low-amplitude, while one
  train case is high-amplitude and nonlinear.

Risks:

- Slightly more complex.
- Requires careful zero-q handling because `q_dir` is undefined at zero.

## Recommended Normalization

Use a minimal A+C hybrid:

```text
1. Use global-component normalization for LE_local.
2. Use induced B normalization:
   B_norm[a,k] = B_local[a,k] * q_std[k] / LE_std[a].
3. Use q_useful component std normalization for the raw branch vector.
4. Add explicit q_amp = ||q_useful|| and log_q_amp or scaled_q_amp.
5. Optionally add q_dir for nonzero q, with zero-q handled by an all-zero q_dir
   and q_amp=0.
```

Do not start with pure case-amplitude normalization as the only route.  It is
useful as an ablation, but it may blur nonlinear amplitude effects.

## Formal v2 Architecture Proposal

Inputs:

```text
branch:
  q_useful_normed [42]
  q_amp scalar
  optional q_dir [42]

trunk:
  ip_xi [3]
  optional ip_detJ
  optional compact geometry features from ip_J/local_frame_Q
```

Do not add `case_id` as a first formal feature.  If the model needs extra
information, prefer physical descriptors such as amplitude, load direction, and
geometry/mapping features.

Output:

```text
LE_local_hat [6]
```

Anchored structure:

```text
LE_local_hat =
  B_prior_local(point, optional geometry) @ q_useful
  + gate(q_amp) * (R(q_useful_normed, q_amp, point, geom) - R(0, 0, point, geom))
```

Required structural checks:

```text
q_useful = 0 => LE_local_hat = 0
AD_B_local = dLE_local_hat / d(q_useful)
B_raw_hat = T_eps_to_abq @ AD_B_local @ T_q_raw_to_useful
```

Initial architecture should be modest:

```text
point encoder: ip_xi + optional geometry -> point latent
branch encoder: q_useful_normed + q_amp -> branch latent
B prior head: point latent -> [6,42]
residual head: branch/point fusion -> [6]
```

## Loss and Checkpoint Design

Formal metrics:

```text
LE_local_rel
AD_B_local_rel
AD_B_local_cos
B_raw_projected_rel
B_raw_rel
zero_q_LE_local_rms
per-case attribution
```

Suggested checkpoint score:

```text
score =
  1.0 * val_LE_local_rel
+ 1.0 * val_AD_B_local_rel
+ 0.5 * val_B_model_raw_projected_rel
+ 10.0 * zero_q_rms_norm
```

Also record separate checkpoints:

```text
best_LE
best_B
best_raw_projected
best_combined
latest
```

For low-amplitude validation cases, also report absolute RMS errors alongside
relative errors.  Relative LE alone can look catastrophic when `LE_rms` is
near `1e-4`.

## Next Step: v2g

v2g should implement a formal prototype, not another smoke tuning pass.

Recommended v2g scope:

```text
1. Add a formal v2 normalization artifact computed from train cases only.
2. Implement normalized q/LE/B training in a prototype script.
3. Add q_amp as an explicit branch descriptor.
4. Preserve anchored zero-q structure.
5. Keep raw-B backprojection evaluation.
6. Compare against v2e tiny smoke with the same split.
```

## Boundaries

- No formal model training was performed in v2f.
- No v1.3/v2 production model, loss, learning-rate, epoch, or split policy was
  changed.
- No tag was moved.
- No old TRUE176 `LE/B` labels were used as v2 labels.
- No `.npz`, `.odb`, `.pt`, `.pth`, checkpoint, or generated output artifact is
  committed.
