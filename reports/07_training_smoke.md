# Gate 07 Training Smoke

Date: 2026-06-25

Role: Macro16 Training Smoke Agent

## 1. Task Goal

跑一个小规模 Macro16 training smoke，确认训练链路能工作。

本轮不做大训练，不改模型结构，不改 `q48` 顺序，不改 `LE` 顺序，不改 128 点积分规则。

## 2. Data Used

Smoke compact list:

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate07_training_smoke\macro16_training_smoke_compact_list.txt`

数据来自 Gate 06 readiness 通过的 8 类 Macro16 source128 compact：

1. regular
2. lightly distorted
3. moderately distorted
4. strong non-flipped repaired
5. cylindrical shell
6. conical shell
7. thickness-varying shell
8. mild double-curvature shell

Total compact count: `56`

Training smoke used:

1. `max_frames_per_compact = 2`
2. Train/val split by case
3. Validation cases: `70,71,72,73`
4. Validation overlap: `false`

## 3. Commands Used

Compile post-training force audit:

```powershell
py -3 -m py_compile scripts\audit_macro16_trained_force_closure.py
```

Training smoke:

```powershell
$env:PYTHONPATH='src'
py -3 -m macro_deeponet.train_macro16_boundary_sobolev --compact-list runs\gate07_training_smoke\macro16_training_smoke_compact_list.txt --out-dir runs\gate07_training_smoke\train_smoke --model-style le0 --epochs 2 --batch-size 4 --eval-batch-size 2 --max-frames-per-compact 2 --max-eval-frames 64 --basis-dim 16 --hidden-dim 48 --branch-depth 2 --trunk-depth 2 --jacobian-columns 0,1,2,3 --jacobian-columns-per-batch 2 --eval-columns 0,1,2,3 --le-loss-weight 1.0 --jacobian-loss-weight 0.1 --rigid-loss-weight 0.1 --lr 1.0e-4 --lr-decay 1.0 --weight-decay 0.0 --val-cases 70,71,72,73 --eval-every 1 --cuda
```

Post-training selected-frame force closure smoke:

```powershell
$env:PYTHONPATH='src;scripts'
py -3 scripts\audit_macro16_trained_force_closure.py --checkpoint runs\gate07_training_smoke\train_smoke\best.pt --compact-list runs\gate07_training_smoke\macro16_training_smoke_compact_list.txt --case-list 70,71,72,73 --max-frames 40 --batch-size 1 --out runs\gate07_training_smoke\trained_force_closure_val_cases_70_73_all40.json --cuda
```

## 4. Model Contract

Model:

```text
Macro16BoundaryDeepONetWithLE0
```

Inputs:

```text
q48_def_hat
X16_hat
standard source128 point features
```

Output:

```text
LE
```

B source:

```text
autograd dLE / d(q48_def_hat)
```

Point rule:

```text
standard 128 source128 points
```

## 5. Training Result

Smoke status: PASS

The training command completed, wrote `best.pt`, `latest.pt`, `config.json`, and `training_summary.json`.

Best epoch:

```text
epoch = 2
train_LE_rel = 1.4772585535731129
train_AD_B_rel = 0.9943445914612397
val_LE_rel = 1.3382370356534028
val_AD_B_rel = 1.0026297683883236
```

Split:

```text
validation_split_mode = case
validation_is_overlapping = false
val_cases = [70, 71, 72, 73]
```

Interpretation:

1. Training chain works.
2. `LE` prediction runs.
3. `B` autograd path runs.
4. Checkpoints are written.
5. This 2-epoch smoke is not a convergence run.

## 6. Post-Training Force Closure Smoke

Audit output:

`D:\IS-FEM\deeponet_macro_operator_fresh\runs\gate07_training_smoke\trained_force_closure_val_cases_70_73_all40.json`

Frame count:

```text
40
```

Cases:

```text
[70, 71, 72, 73]
```

Coordinate:

```text
qdef coordinate
reference = rigid_projection_P^T RF_projected_macro
volume = selected-frame ip_IVOL_abaqus_selected_frames
```

Result:

```text
model LE_rel = 1.0567432482182355
model B_qdef_hat_rel = 1.027165402510371
model selected-frame force rel = 0.9805619215409668
teacher selected-frame force rel = 2.563913932469707e-05
```

Interpretation:

1. The post-training force audit chain works.
2. Model-predicted `LE` and autograd `B` can be assembled into force.
3. The teacher still closes force in the same selected-frame volume audit.
4. The 2-epoch smoke model does not close force yet.

## 7. Gate Result

Training smoke result: PASS for chain execution.

Not passed:

1. Convergence is not claimed.
2. Solver readiness is not claimed.
3. Full tangent closure is not claimed.
4. Material-only K is not used as full tangent evidence.

## 8. Unresolved Issues

1. Current smoke is intentionally tiny, so LE/B errors remain high.
2. Post-training selected-frame force error remains high.
3. `B_macro_qdef` still uses the current small-rotation rigid projection approximation.
4. Full tangent remains pending.

## 9. Recommended Next Action

1. If continuing training, keep `val_cases=70,71,72,73` or another explicit case/category holdout.
2. Increase epochs and AD column coverage gradually.
3. After any longer run, rerun the same selected-frame force closure audit.
4. Do not judge the route by training loss alone.
