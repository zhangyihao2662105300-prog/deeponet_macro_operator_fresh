# Gate Task Index

Use this file to select the next Codex task. Read `AGENTS.md`,
`docs/PROJECT_BRIEF.md`, `docs/THEORY_BASELINE.md`,
`docs/THEORY_AUDIT_MATRIX.md`, and `docs/GATE_WORKFLOW.md` before running a
gate.

## Canonical Order

| Gate | Task | Status | Report |
|---|---|---|---|
| 01 | Data contract audit | PASS | `reports/macro16_source128_data_audit.md` |
| 02 | Teacher TRUE176/CSS8 closure audit | FAIL on full teacher stiffness closure | `reports/02_teacher_closure_audit.md` |
| 03 | Macro16 source128 force/stiffness audit | FAIL current max-error gate; failure localized | `reports/macro16_force_stiffness_audit.md`, `reports/macro16_case031_mechanics_diagnosis.md` |
| 04 | Distortion/general geometry audit | PASS for main-route required families; strong non-inverted remains non-blocking robustness data | `reports/04_distortion_generality_audit.md` |
| 05 | Integration-point reduction audit | not complete | `reports/05_ip_reduction_audit.md` |
| 06 | Training gate | blocked | `reports/06_training_gate.md` |

## Existing Task Files

- `tasks/01_macro16_source128_data_audit.md`: historical Gate 01 task packet.
- `tasks/02_macro16_force_stiffness_audit.md`: historical Macro16 mechanics
  task packet; in the canonical workflow this is Gate 03.
- `tasks/03_macro16_distortion_generality.md`: historical Gate 04 task packet.
- `tasks/04_macro16_ip_reduction_gate.md`: historical Gate 05 task packet.
- `tasks/05_network_training_gate.md`: historical Gate 06 task packet.

## Next Recommended Task

Decide the Gate 03 mechanics route after the focused case031 diagnosis:

```text
Read AGENTS.md, docs/PROJECT_BRIEF.md, docs/THEORY_BASELINE.md,
docs/THEORY_AUDIT_MATRIX.md, docs/GATE_WORKFLOW.md,
reports/macro16_force_stiffness_audit.md, and
reports/macro16_case031_mechanics_diagnosis.md.

You are Mechanics Decision Agent.
Do not train a network.
Do not modify model structure.
Do not change q48 ordering, LE ordering, weights, or the 128-point rule.

Goal:
Decide how Gate 03 should handle the localized case031 mechanics failure:
volume convention and complete consistent tangent terms.

Output:
reports/macro16_gate03_mechanics_decision.md
```
