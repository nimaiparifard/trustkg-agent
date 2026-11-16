# Results (n_seeds=10, tasks/run=360, mean ± std over seeds)

Abstention threshold tuned on dev seeds (disjoint from test seeds) with wrong_cost=2.0: **0.15**

| threshold | dev utility (success − wrong_cost × hallucination) |
|---|---|
| 0.05 | 33.8 |
| 0.1 | 33.8 |
| 0.15 | 33.9 |
| 0.2 | 33.7 |
| 0.3 | 32.1 |
| 0.4 | 25.8 |
| 0.5 | 20.7 |
| 0.6 | 15.5 |
| 0.7 | 9.9 |

**Condition: mixed**

| System | Success | Reason.Acc | Abstain | Halluc. | Unsupp. | Ev.Prec | Recovery | Fail-contain | KG/text calls | LLM calls |
|---|---|---|---|---|---|---|---|---|---|---|
| llm_only | 13.1 ± 2.2 | 13.1 ± 2.2 | 0.0 ± 0.0 | 86.9 ± 2.2 | 100.0 ± 0.0 | n/a | n/a | n/a | 0.00 ± 0.00 | 1.00 ± 0.00 |
| react | 15.8 ± 2.7 | 15.8 ± 2.7 | 0.0 ± 0.0 | 84.2 ± 2.7 | 100.0 ± 0.0 | n/a | n/a | n/a | 0.00 ± 0.00 | 2.73 ± 0.05 |
| react_vector_rag | 46.0 ± 5.1 | 46.0 ± 5.1 | 0.0 ± 0.0 | 54.0 ± 5.1 | 43.3 ± 3.9 | 81.1 ± 1.8 | 26.3 ± 4.2 | 26.3 ± 4.2 | 2.73 ± 0.05 | 0.52 ± 0.06 |
| react_kg | 48.9 ± 2.9 | 48.9 ± 2.9 | 0.0 ± 0.0 | 51.1 ± 2.9 | 28.9 ± 3.5 | 81.2 ± 2.2 | 23.7 ± 3.6 | 23.7 ± 3.6 | 2.73 ± 0.05 | 0.32 ± 0.04 |
| trustkg_verify | 52.9 ± 2.9 | 67.7 ± 3.7 | 21.8 ± 1.8 | 25.3 ± 3.1 | 0.0 ± 0.0 | 84.9 ± 1.6 | 34.8 ± 4.8 | 82.2 ± 2.4 | 2.88 ± 0.08 | 0.00 ± 0.00 |
| trustkg_full | 58.0 ± 2.5 | 70.1 ± 2.8 | 17.2 ± 1.3 | 24.8 ± 2.4 | 0.0 ± 0.0 | 86.6 ± 1.5 | 34.8 ± 4.6 | 84.4 ± 2.7 | 2.80 ± 0.06 | 0.00 ± 0.00 |

**Condition: incomplete**

| System | Success | Reason.Acc | Abstain | Halluc. | Unsupp. | Ev.Prec | Recovery | Fail-contain | KG/text calls | LLM calls |
|---|---|---|---|---|---|---|---|---|---|---|
| llm_only | 18.2 ± 2.2 | 18.2 ± 2.2 | 0.0 ± 0.0 | 81.8 ± 2.2 | 100.0 ± 0.0 | n/a | n/a | n/a | 0.00 ± 0.00 | 1.00 ± 0.00 |
| react | 21.6 ± 3.0 | 21.6 ± 3.0 | 0.0 ± 0.0 | 78.4 ± 3.0 | 100.0 ± 0.0 | n/a | n/a | n/a | 0.00 ± 0.00 | 2.75 ± 0.04 |
| react_vector_rag | 68.9 ± 4.8 | 68.9 ± 4.8 | 0.0 ± 0.0 | 31.1 ± 4.8 | 42.7 ± 4.7 | 96.4 ± 1.1 | 40.9 ± 7.9 | 40.9 ± 7.9 | 2.75 ± 0.04 | 0.51 ± 0.07 |
| react_kg | 68.1 ± 3.7 | 68.1 ± 3.7 | 0.0 ± 0.0 | 31.9 ± 3.7 | 51.8 ± 4.2 | 100.0 ± 0.0 | 38.4 ± 5.1 | 38.4 ± 5.1 | 2.75 ± 0.04 | 0.65 ± 0.07 |
| trustkg_verify | 74.6 ± 3.1 | 99.1 ± 1.0 | 24.7 ± 2.8 | 0.6 ± 0.7 | 0.0 ± 0.0 | 99.7 ± 0.4 | 54.6 ± 4.7 | 98.8 ± 1.3 | 3.04 ± 0.06 | 0.00 ± 0.00 |
| trustkg_full | 82.0 ± 2.9 | 98.9 ± 0.6 | 17.1 ± 2.9 | 0.9 ± 0.5 | 0.0 ± 0.0 | 99.6 ± 0.2 | 56.0 ± 5.4 | 97.8 ± 1.3 | 2.94 ± 0.07 | 0.00 ± 0.00 |

**Condition: contradictory**

| System | Success | Reason.Acc | Abstain | Halluc. | Unsupp. | Ev.Prec | Recovery | Fail-contain | KG/text calls | LLM calls |
|---|---|---|---|---|---|---|---|---|---|---|
| llm_only | 18.2 ± 2.2 | 18.2 ± 2.2 | 0.0 ± 0.0 | 81.8 ± 2.2 | 100.0 ± 0.0 | n/a | n/a | n/a | 0.00 ± 0.00 | 1.00 ± 0.00 |
| react | 21.6 ± 3.0 | 21.6 ± 3.0 | 0.0 ± 0.0 | 78.4 ± 3.0 | 100.0 ± 0.0 | n/a | n/a | n/a | 0.00 ± 0.00 | 2.75 ± 0.04 |
| react_vector_rag | 68.9 ± 4.8 | 68.9 ± 4.8 | 0.0 ± 0.0 | 31.1 ± 4.8 | 42.7 ± 4.7 | 96.4 ± 1.1 | 40.9 ± 7.9 | 40.9 ± 7.9 | 2.75 ± 0.04 | 0.51 ± 0.07 |
| react_kg | 67.3 ± 4.0 | 67.3 ± 4.0 | 0.0 ± 0.0 | 32.7 ± 4.0 | 13.2 ± 1.3 | 90.4 ± 1.8 | 35.7 ± 6.9 | 35.7 ± 6.9 | 2.75 ± 0.04 | 0.13 ± 0.01 |
| trustkg_verify | 77.1 ± 1.6 | 100.0 ± 0.0 | 22.9 ± 1.6 | 0.0 ± 0.0 | 0.0 ± 0.0 | 100.0 ± 0.0 | 57.8 ± 3.8 | 100.0 ± 0.0 | 3.08 ± 0.11 | 0.00 ± 0.00 |
| trustkg_full | 85.9 ± 1.3 | 100.0 ± 0.0 | 14.1 ± 1.3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 100.0 ± 0.0 | 64.3 ± 4.3 | 100.0 ± 0.0 | 2.94 ± 0.07 | 0.00 ± 0.00 |

**Condition: stale**

| System | Success | Reason.Acc | Abstain | Halluc. | Unsupp. | Ev.Prec | Recovery | Fail-contain | KG/text calls | LLM calls |
|---|---|---|---|---|---|---|---|---|---|---|
| llm_only | 10.9 ± 1.8 | 10.9 ± 1.8 | 0.0 ± 0.0 | 89.1 ± 1.8 | 100.0 ± 0.0 | n/a | n/a | n/a | 0.00 ± 0.00 | 1.00 ± 0.00 |
| react | 13.3 ± 2.5 | 13.3 ± 2.5 | 0.0 ± 0.0 | 86.7 ± 2.5 | 100.0 ± 0.0 | n/a | n/a | n/a | 0.00 ± 0.00 | 2.71 ± 0.05 |
| react_vector_rag | 35.3 ± 3.8 | 35.3 ± 3.8 | 0.0 ± 0.0 | 64.7 ± 3.8 | 43.5 ± 4.3 | 70.3 ± 2.2 | 20.9 ± 4.9 | 20.9 ± 4.9 | 2.71 ± 0.05 | 0.52 ± 0.06 |
| react_kg | 44.9 ± 3.8 | 44.9 ± 3.8 | 0.0 ± 0.0 | 55.1 ± 3.8 | 12.7 ± 1.7 | 73.7 ± 1.4 | 3.2 ± 2.1 | 3.2 ± 2.1 | 2.71 ± 0.05 | 0.13 ± 0.02 |
| trustkg_verify | 44.0 ± 3.3 | 53.0 ± 3.8 | 17.1 ± 2.1 | 38.9 ± 3.3 | 0.0 ± 0.0 | 76.0 ± 2.0 | 0.0 ± 0.0 | 100.0 ± 0.0 | 2.59 ± 0.04 | 0.00 ± 0.00 |
| trustkg_full | 47.6 ± 3.7 | 63.5 ± 3.6 | 25.1 ± 3.8 | 27.4 ± 3.1 | 0.0 ± 0.0 | 82.8 ± 1.7 | 0.0 ± 0.0 | 100.0 ± 0.0 | 2.55 ± 0.04 | 0.00 ± 0.00 |

**Adaptation under drift (condition: mixed)**

| System | Post-shift succ | Post-shift halluc | Retention | Plasticity | Novel | Final-stage succ |
|---|---|---|---|---|---|---|
| llm_only | 11.8 ± 2.5 | 88.2 ± 2.5 | 21.3 ± 4.5 | 3.1 ± 1.5 | 3.2 ± 2.3 | 9.3 ± 5.3 |
| react | 13.9 ± 3.1 | 86.1 ± 3.1 | 25.8 ± 6.0 | 3.5 ± 1.7 | 2.5 ± 1.8 | 10.7 ± 4.4 |
| react_vector_rag | 39.5 ± 4.9 | 60.5 ± 4.9 | 79.4 ± 5.9 | 3.6 ± 2.3 | 1.8 ± 2.5 | 28.0 ± 5.5 |
| react_kg | 41.8 ± 3.5 | 58.2 ± 3.5 | 84.4 ± 4.6 | 3.3 ± 2.0 | 2.8 ± 2.6 | 29.2 ± 4.1 |
| trustkg_verify | 44.4 ± 3.4 | 29.9 ± 3.6 | 91.7 ± 3.7 | 2.1 ± 1.7 | 0.0 ± 0.0 | 29.3 ± 3.5 |
| trustkg_full | 50.6 ± 2.9 | 29.3 ± 2.7 | 94.7 ± 2.7 | 12.5 ± 2.3 | 4.9 ± 3.8 | 38.7 ± 4.4 |

**Success per stage (mixed)**

| System | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|
| llm_only | 19.5 | 15.7 | 12.5 | 10.3 | 11.0 | 9.3 |
| react | 25.2 | 18.0 | 15.7 | 12.0 | 13.3 | 10.7 |
| react_vector_rag | 78.5 | 56.5 | 46.3 | 35.8 | 31.0 | 28.0 |
| react_kg | 84.3 | 61.3 | 50.2 | 35.8 | 32.7 | 29.2 |
| trustkg_verify | 95.2 | 69.0 | 53.0 | 39.0 | 31.8 | 29.3 |
| trustkg_full | 95.3 | 70.3 | 55.7 | 46.0 | 42.2 | 38.7 |

**Ablation (condition: mixed)**

| System | Success | Halluc. | Abstain | Post-shift succ | Post-shift halluc | KG/text calls |
|---|---|---|---|---|---|---|
| full | 58.0 ± 2.5 | 24.8 ± 2.4 | 17.2 ± 1.3 | 50.6 ± 2.9 | 29.3 ± 2.7 | 2.80 ± 0.06 |
| -memory | 52.9 ± 2.9 | 25.3 ± 3.1 | 21.8 ± 1.8 | 44.4 ± 3.4 | 29.9 ± 3.6 | 2.88 ± 0.08 |
| -conflict_resolution | 55.7 ± 2.4 | 24.2 ± 2.3 | 20.1 ± 1.3 | 48.9 ± 2.7 | 28.6 ± 2.7 | 2.80 ± 0.06 |
| -freshness | 58.5 ± 2.4 | 26.7 ± 2.4 | 14.8 ± 0.8 | 51.1 ± 2.7 | 31.6 ± 2.7 | 2.80 ± 0.06 |
| -recovery | 52.4 ± 3.2 | 21.6 ± 2.5 | 26.0 ± 2.8 | 46.4 ± 3.7 | 25.8 ± 2.8 | 2.36 ± 0.06 |
| -backtracking | 58.0 ± 2.5 | 24.8 ± 2.4 | 17.2 ± 1.3 | 50.6 ± 2.9 | 29.3 ± 2.7 | 2.79 ± 0.06 |
| -abstention | 59.2 ± 2.5 | 40.8 ± 2.5 | 0.0 ± 0.0 | 51.9 ± 2.9 | 48.1 ± 2.9 | 2.80 ± 0.06 |

**Ablation: full minus ablated (paired, 95% bootstrap CI over seeds)**

| Removed | Metric | Δ (full − ablated) | 95% CI |
|---|---|---|---|
| -memory | post_shift_success | +6.1 pp | [+5.5, +6.8] |
| -memory | hallucination_rate | -0.5 pp | [-1.7, +0.8] |
| -conflict_resolution | post_shift_success | +1.7 pp | [+1.2, +2.1] |
| -conflict_resolution | hallucination_rate | +0.6 pp | [+0.0, +1.1] |
| -freshness | post_shift_success | -0.6 pp | [-0.9, -0.3] |
| -freshness | hallucination_rate | -1.9 pp | [-2.3, -1.4] |
| -recovery | post_shift_success | +4.1 pp | [+3.3, +5.2] |
| -recovery | hallucination_rate | +3.1 pp | [+2.6, +3.7] |
| -backtracking | post_shift_success | +0.0 pp | [+0.0, +0.0] |
| -backtracking | hallucination_rate | +0.0 pp | [+0.0, +0.1] |
| -abstention | post_shift_success | -1.4 pp | [-1.7, -1.0] |
| -abstention | hallucination_rate | -16.0 pp | [-16.7, -15.3] |

**Hypothesis tests (paired over seeds; difference a − b, 95% bootstrap CI)**

_Condition: mixed_

| H | Comparison | Metric | Δ | 95% CI | Supported |
|---|---|---|---|---|---|
| H1 | KG vs. no KG: lower hallucination (react_kg vs react) | hallucination_rate | -33.1 pp | [-35.9, -30.6] | yes |
| H1 | KG vs. vector RAG: lower hallucination (react_kg vs react_vector_rag) | hallucination_rate | -2.9 pp | [-5.8, -0.6] | yes |
| H2 | Verification: lower hallucination (trustkg_verify vs react_kg) | hallucination_rate | -25.8 pp | [-27.5, -24.0] | yes |
| H2 | Verification: fewer unsupported answers (trustkg_verify vs react_kg) | unsupported_rate | -28.9 pp | [-31.0, -26.9] | yes |
| H2 | Verification: higher reasoning accuracy on answered tasks | reasoning_accuracy | +18.8 pp | [+16.8, +20.5] | yes |
| H3 | Memory: higher post-shift success (trustkg_full vs trustkg_verify) | post_shift_success | +6.1 pp | [+5.5, +6.8] | yes |
| H3 | Memory: lower post-shift hallucination | post_shift_hallucination | -0.6 pp | [-2.1, +0.9] | no |
| H3 | Memory: higher final-stage success | final_stage_success | +9.3 pp | [+7.0, +11.5] | yes |

_Condition: incomplete_

| H | Comparison | Metric | Δ | 95% CI | Supported |
|---|---|---|---|---|---|
| H1 | KG vs. no KG: lower hallucination (react_kg vs react) | hallucination_rate | -46.5 pp | [-48.3, -45.0] | yes |
| H1 | KG vs. vector RAG: lower hallucination (react_kg vs react_vector_rag) | hallucination_rate | +0.8 pp | [-2.1, +3.9] | no |
| H2 | Verification: lower hallucination (trustkg_verify vs react_kg) | hallucination_rate | -31.3 pp | [-33.1, -29.1] | yes |
| H2 | Verification: fewer unsupported answers (trustkg_verify vs react_kg) | unsupported_rate | -51.8 pp | [-54.3, -49.4] | yes |
| H2 | Verification: higher reasoning accuracy on answered tasks | reasoning_accuracy | +31.1 pp | [+28.9, +32.9] | yes |
| H3 | Memory: higher post-shift success (trustkg_full vs trustkg_verify) | post_shift_success | +8.7 pp | [+7.6, +9.7] | yes |
| H3 | Memory: lower post-shift hallucination | post_shift_hallucination | +0.4 pp | [+0.1, +0.7] | no |
| H3 | Memory: higher final-stage success | final_stage_success | +18.3 pp | [+16.2, +20.3] | yes |

_Condition: contradictory_

| H | Comparison | Metric | Δ | 95% CI | Supported |
|---|---|---|---|---|---|
| H1 | KG vs. no KG: lower hallucination (react_kg vs react) | hallucination_rate | -45.7 pp | [-49.0, -42.3] | yes |
| H1 | KG vs. vector RAG: lower hallucination (react_kg vs react_vector_rag) | hallucination_rate | +1.6 pp | [-2.0, +4.8] | no |
| H2 | Verification: lower hallucination (trustkg_verify vs react_kg) | hallucination_rate | -32.7 pp | [-34.9, -30.3] | yes |
| H2 | Verification: fewer unsupported answers (trustkg_verify vs react_kg) | unsupported_rate | -13.2 pp | [-14.0, -12.5] | yes |
| H2 | Verification: higher reasoning accuracy on answered tasks | reasoning_accuracy | +32.7 pp | [+30.3, +34.9] | yes |
| H3 | Memory: higher post-shift success (trustkg_full vs trustkg_verify) | post_shift_success | +10.2 pp | [+9.3, +11.2] | yes |
| H3 | Memory: lower post-shift hallucination | post_shift_hallucination | +0.0 pp | [+0.0, +0.0] | no |
| H3 | Memory: higher final-stage success | final_stage_success | +23.5 pp | [+21.5, +25.5] | yes |

_Condition: stale_

| H | Comparison | Metric | Δ | 95% CI | Supported |
|---|---|---|---|---|---|
| H1 | KG vs. no KG: lower hallucination (react_kg vs react) | hallucination_rate | -31.6 pp | [-33.3, -29.6] | yes |
| H1 | KG vs. vector RAG: lower hallucination (react_kg vs react_vector_rag) | hallucination_rate | -9.7 pp | [-11.0, -8.6] | yes |
| H2 | Verification: lower hallucination (trustkg_verify vs react_kg) | hallucination_rate | -16.1 pp | [-17.2, -14.9] | yes |
| H2 | Verification: fewer unsupported answers (trustkg_verify vs react_kg) | unsupported_rate | -12.7 pp | [-13.6, -11.6] | yes |
| H2 | Verification: higher reasoning accuracy on answered tasks | reasoning_accuracy | +8.1 pp | [+7.6, +8.7] | yes |
| H3 | Memory: higher post-shift success (trustkg_full vs trustkg_verify) | post_shift_success | +4.3 pp | [+3.1, +5.4] | yes |
| H3 | Memory: lower post-shift hallucination | post_shift_hallucination | -13.9 pp | [-16.1, -11.9] | yes |
| H3 | Memory: higher final-stage success | final_stage_success | +7.8 pp | [+4.5, +11.2] | yes |
