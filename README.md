# TrustKG-Agent

**A Knowledge-Grounded and Self-Reflective Agent for Reliable Multi-Step Reasoning**

A from-scratch (no LangChain / CrewAI) research prototype plus an evaluation harness that
asks: *when the knowledge an agent relies on is incomplete, contradictory or outdated, do
structured knowledge, evidence verification and experience memory make it more reliable?*

Design: [architecture.md](architecture.md) · Full numbers: [results/summary.md](results/summary.md)

```bash
pip install -r requirements.txt
python -m pytest -q                                    # 21 tests (3 need Neo4j and skip without it)
python -m evaluation.experiments --config configs/default.json --out results   # ~20 s, regenerates results/
python demo.py                                         # readable reasoning traces

docker compose up -d                                   # optional: Neo4j backend (tests/test_neo4j.py)
python -m evaluation.run_ollama                        # optional: real local LLM via Ollama (~25 min on CPU)
```

> **Read this first — what the results do and do not show.** The main results (Sections 6-7,
> 10 seeds) come from a *controlled synthetic benchmark* with a *simulated LLM* (explicit
> parametric-knowledge and hallucination parameters, templated questions, rule-based
> planner/reader). They measure how the **architecture** behaves under controlled KG defects.
> A smaller run with a **real local LLM** (Qwen2.5-7B via Ollama, Section 6b; 2 seeds) shows
> which simulation assumptions do *not* hold for a real model. No real-world dataset was used.

## 1. Research Motivation

LLM agents answer confidently even when their knowledge is missing or stale. A knowledge graph
gives explicit, attributable facts, but a KG is itself imperfect (gaps, conflicting sources,
facts that stop being true). Prior work (e.g. KnowAgent, TKG-Thinker) shows that combining
planning, structured knowledge and agentic reasoning is an active area — so simply attaching a
KG to an agent is not a contribution. This project instead studies **trustworthiness and
adaptation**: abstaining/recovering under bad evidence, and learning from the environment.
(No literature review was performed for this repo; no novelty claim is made.)

## 2. Research Questions and Hypotheses

**RQ:** Can structured knowledge and evidence-based verification improve the reliability and
adaptability of LLM-based agents in multi-step reasoning tasks?

* **H1** A KG reduces multi-step reasoning errors.
* **H2** Evidence verification reduces unsupported / wrong answers.
* **H3** Experience memory + knowledge updates improve performance as the environment changes.

## 3. Architecture

Six modules (details in [architecture.md](architecture.md)):

| Module | Code |
|---|---|
| Planning (question → Task DAG, per-step staleness uncertainty, replanning suffix) | [agents/planner.py](agents/planner.py) |
| Knowledge (schema, graph stores, builder, retrieval, updates) | [knowledge/](knowledge/) |
| Execution (ReAct loop over the DAG, tools: `kg_lookup`, `text_search`, `llm`; backtracking search) | [agents/executor.py](agents/executor.py) |
| Trust & verification (noisy-OR evidence, conflict penalty, staleness discount, claim checks, abstain) | [agents/verifier.py](agents/verifier.py) |
| Adaptive memory (episodic log, per-relation staleness model, KG updates from feedback) | [agents/memory.py](agents/memory.py) |
| Evaluation (benchmarks, metrics, experiments, threshold tuning) | [evaluation/](evaluation/) |

How verification works: each hop's evidence is scored `conf × (1 − hazard)^age` (age = stages
since the fact was last verified); evidence for the same answer is combined by noisy-OR; a
rival answer multiplies in `(1 − rival_score)` (conflict penalty). Chains are searched with
backtracking (beam 2); missing/contradictory KG evidence triggers a text cross-check
(*recovery*); the final chain is decomposed into atomic claims, each checked against evidence;
the agent answers only if the chain confidence ≥ threshold and every claim is supported, else
it **abstains**. The memory estimates each relation's real change rate from feedback
(Gamma-prior, exposure-based) and refreshes/updates/deprecates KG facts.

## 4. Experimental Setup

* **World (≈1000 entities):** 600 persons, 200 companies, 120 cities, 60 countries, 20 blocs;
  relations `works_for → headquartered_in → located_in → member_of` (functional).
* **Drift over 5 stages** (stage 0 has none): per-stage change probabilities
  works_for 0.20, headquartered_in 0.10, member_of 0.05, located_in 0; 20 new persons appear per
  stage. By stage 5 about 50% of stage-0 facts are outdated.
* **What the agent sees:** only a stage-0 snapshot — a KG (8% facts missing, 6% with a
  conflicting low-confidence fact from a noisy source) and a text corpus (15% missing, 6%
  conflicting). Curated KG facts have confidence 0.9.
* **Tasks:** 60 per stage × 6 stages = 360 per run; 2–4 hop questions
  ("What is the bloc of the country of the headquarters city of the employer of Person_0011?").
  Gold answers come from the *current* world, so stale knowledge yields wrong answers.
  Task categories: `stable` / `changed` (chain differs from stage 0) / `novel` (new entity).
* **Feedback:** after 30% of tasks the environment reveals the true fact chain (an *oracle*
  correction). Only the memory-equipped system uses it.
* **Conditions** (isolating each KG defect): `mixed` (above), `incomplete` (20% missing, no
  drift), `contradictory` (20% conflicts, no drift), `stale` (2× drift).
* **Simulated LLM:** frozen stage-0 knowledge, knows 55% of facts, otherwise returns a
  type-correct *hallucination* (never abstains); one-shot multi-hop skips a hop w.p. 0.15.
* **Seeds:** 10 test seeds (0–9); each seed fixes the world, drift, and task stream, shared by
  all systems (paired). Abstention threshold (0.15) was tuned on 5 *separate* dev seeds
  (100–104) maximizing `success − 2 × hallucination` — the factor 2 (wrong answers cost more
  than abstaining) is a design choice; the sweep is in [results/summary.md](results/summary.md).
* **Metrics:** *Success* = answered and correct; *Reasoning accuracy* = correct among answered;
  *Hallucination* = answered and wrong (fraction of all tasks); *Unsupported* = answered
  without full retrieved evidence for every hop; *Evidence precision* = cited retrieved
  facts true in the current world; *Recovery* = correct among tasks that hit missing/conflicting
  evidence; *Fail-contain* = same tasks, not answered wrongly (abstaining counts);
  *Tool calls* = KG + text lookups per task; *Post-shift* = stages ≥ 1.

## 5. Baselines

| System | What it is |
|---|---|
| `llm_only` | One-shot parametric answer |
| `react` | Step-by-step, each hop answered by the LLM |
| `react_vector_rag` | Each hop from TF-IDF text retrieval (LLM guess if nothing found) |
| `react_kg` | Each hop from the KG, naive conflict handling (most recent), LLM guess on gaps |
| `trustkg_verify` | KG + verification + recovery + backtracking + abstention |
| `trustkg_full` | + adaptive memory (feedback → KG updates, learned staleness) |

## 6. Quantitative Results (mean ± std over 10 seeds, %)

**Condition `mixed`**

| System | Success | Reason. acc | Abstain | Halluc. | Unsupp. | Ev. prec | Recovery | Fail-contain | KG/text calls | LLM calls |
|---|---|---|---|---|---|---|---|---|---|---|
| llm_only | 13.1 ± 2.2 | 13.1 | 0 | 86.9 ± 2.2 | 100 | n/a | n/a | n/a | 0 | 1.00 |
| react | 15.8 ± 2.7 | 15.8 | 0 | 84.2 ± 2.7 | 100 | n/a | n/a | n/a | 0 | 2.73 |
| react_vector_rag | 46.0 ± 5.1 | 46.0 | 0 | 54.0 ± 5.1 | 43.3 | 81.1 | 26.3 | 26.3 | 2.73 | 0.52 |
| react_kg | 48.9 ± 2.9 | 48.9 | 0 | 51.1 ± 2.9 | 28.9 | 81.2 | 23.7 | 23.7 | 2.73 | 0.32 |
| trustkg_verify | 52.9 ± 2.9 | 67.7 | 21.8 | 25.3 ± 3.1 | 0 | 84.9 | 34.8 | 82.2 | 2.88 | 0 |
| **trustkg_full** | **58.0 ± 2.5** | **70.1** | 17.2 | **24.8 ± 2.4** | 0 | **86.6** | 34.8 | 84.4 | 2.80 | 0 |

**Single-defect conditions — success / hallucination**

| System | incomplete | contradictory | stale |
|---|---|---|---|
| llm_only | 18.2 / 81.8 | 18.2 / 81.8 | 10.9 / 89.1 |
| react_vector_rag | 68.9 / 31.1 | 68.9 / 31.1 | 35.3 / 64.7 |
| react_kg | 68.1 / 31.9 | 67.3 / 32.7 | 44.9 / 55.1 |
| trustkg_verify | 74.6 / **0.6** | 77.1 / **0.0** | 44.0 / 38.9 |
| trustkg_full | **82.0** / 0.9 | **85.9** / 0.0 | **47.6** / **27.4** |

(`react_vector_rag`/`llm_only` rows are identical in `incomplete` and `contradictory` because
they do not depend on the KG-noise parameters, only the fixed text corpus.)

**Adaptation under drift (`mixed`)** — success per stage (world drifts from stage 1):

| System | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|
| llm_only | 19.5 | 15.7 | 12.5 | 10.3 | 11.0 | 9.3 |
| react_kg | 84.3 | 61.3 | 50.2 | 35.8 | 32.7 | 29.2 |
| trustkg_verify | 95.2 | 69.0 | 53.0 | 39.0 | 31.8 | 29.3 |
| trustkg_full | 95.3 | 70.3 | 55.7 | 46.0 | 42.2 | 38.7 |

Post-shift retention on unchanged-fact tasks: `trustkg_full` 94.7 vs `react_kg` 84.4;
plasticity on changed-fact tasks: `trustkg_full` 12.5 vs ≤ 3.6 for all others;
novel-entity tasks: 4.9 vs ≤ 2.8 (all low; see Limitations). Plots:
[stages_mixed.png](results/stages_mixed.png), [conditions.png](results/conditions.png).

### Hypotheses (paired bootstrap over seeds, 95% CI; full tables in results/summary.md)

* **H1 — supported against no-KG** (hallucination −33.1 pp, CI [−35.9, −30.6] in `mixed`).
  Against **vector RAG** the KG advantage is small in `mixed` (−2.9 pp, [−5.8, −0.6]), absent
  in `incomplete`/`contradictory` (CIs include 0), and clear only under heavy drift (−9.7 pp).
  Much of the gain over `llm_only`/`react` is retrieval itself, not the *graph* structure.
* **H2 — supported:** verification lowers hallucination by 25.8 pp (`mixed`), 31–33 pp (single
  defects), 16 pp (`stale`), and eliminates unsupported answers by construction (no LLM guessing).
  Under incomplete/contradictory KGs it nearly eliminates wrong answers (≤ 0.6%) at the cost of
  abstaining on ~23%. **Under drift it does not:** 25–39% of tasks are still answered wrongly,
  because age-based staleness discounting cannot tell *which* old facts changed.
* **H3 — partly supported:** memory raises post-shift success (+6.1 pp, [+5.5, +6.8]) and
  final-stage success (+9.3 pp) in `mixed`; it does **not** significantly reduce hallucination in
  `mixed` (−0.6 pp, CI includes 0), but does under `stale` (−13.9 pp). Its gain depends on the
  assumed oracle feedback on 30% of tasks.

## 6b. Real LLM run (Qwen2.5-7B-Instruct via Ollama) - small, indicative only

Same architecture, same environments (condition `mixed`, 3 stages x 15 tasks = 60 tasks/run,
**seeds 1-2 only**, mean over 2 seeds - far too few for significance tests). The real LLM is
used as (i) **planner**: it receives a *natural-language* paraphrase of the question and must
produce the relation chain, (ii) **reader** of retrieved text chunks (the answer must be grounded in a
passage), and (iii) parametric answerer for the LLM-only/ReAct baselines (it may answer `UNKNOWN` =
abstain). Relation names are snapped to the schema and type-inconsistent chains are rejected.
Run by [evaluation/run_ollama.py](evaluation/run_ollama.py); raw replies are cached in
[results/ollama/](results/ollama/) so it reruns deterministically.

| System | Plan acc | Success | Abstain | Halluc. | Ev. prec | LLM calls | Sim. LLM success / halluc. (same envs) |
|---|---|---|---|---|---|---|---|
| llm_only | 66.7 | 0.0 | 100.0 | 0.0 | n/a | 0.67 | 15.8 / 84.2 |
| react | 66.7 | 0.0 | 100.0 | 0.0 | n/a | 0.67 | 16.7 / 83.3 |
| react_vector_rag | 66.7 | 15.8 ± 5.9 | 77.5 | 6.7 | 81.1 | 0.44 | 50.8 / 49.2 |
| react_kg | 66.7 | 29.2 ± 1.2 | 55.0 | 15.8 | 82.5 | 0.22 | 55.0 / 45.0 |
| trustkg_verify | 66.7 | 38.3 ± 2.4 | 44.2 | 17.5 | 86.0 | 0.00 | 60.8 / 24.2 |
| trustkg_full | 66.7 | 38.3 ± 2.4 | 45.0 | 16.7 | 86.3 | 0.00 | 60.8 / 23.3 |

What this shows (and does not):

* **A real LLM does not hallucinate the way the simulation assumed.** On invented entities Qwen
  answered `UNKNOWN` to every parametric question, so `llm_only`/`react` got 0% success but also
  0% hallucination. The simulation's "always guesses a type-correct entity" is a pessimistic
  assumption (and the real behaviour depends on the prompt, which here explicitly allows `UNKNOWN`).
  The large hallucination gaps between baselines and TrustKG in Section 6 are therefore **partly an
  artefact of that assumption**; with this prompt the real-LLM baselines are safe but useless on
  facts they cannot know.
* **Planning is the bottleneck:** the LLM recovered the exact relation chain for 67% of
  questions. All the other 33% were *invalid* chains rejected by the schema/type check (0 valid-but-wrong
  plans in 120 tasks), so they became safe abstentions. Success of the KG systems is roughly
  plan accuracy x simulated-LLM success (0.667 x 60.8 = 40.5 vs 38.3 observed).
* Among agents that *do* answer, KG-grounded verification has the highest success and
  evidence precision, but with 2 seeds x 60 tasks the differences between `react_kg`, `trustkg_verify`
  and `trustkg_full` are not statistically meaningful, and memory shows no effect over this short
  horizon. Hallucination under drift (16-18%) is the same staleness problem as before.
* Caveats: the planner few-shot prompt was designed while looking at seed 0 (excluded from this
  run); Ollama ran **CPU-only** (the container has no GPU), which is why the run is small.

## 7. Ablation Study (`mixed`; full minus ablated, paired 95% CI)

| Removed component | Δ post-shift success | Δ hallucination | Reading |
|---|---|---|---|
| memory | +6.1 pp [5.5, 6.8] | −0.5 [−1.7, 0.8] | memory helps coverage/success, not safety |
| conflict resolution | +1.7 [1.2, 2.1] | +0.6 [0.0, 1.1] | small gain |
| freshness discount | −0.6 [−0.9, −0.3] | −1.9 [−2.3, −1.4] | small safety gain, tiny success cost |
| recovery (text cross-check) | +4.1 [3.3, 5.2] | **+3.1** [2.6, 3.7] | more answers, but a few more wrong ones (text corpus is stale too) |
| backtracking | 0.0 | 0.0 | no measurable effect in this benchmark |
| abstention | −1.4 [−1.7, −1.0] | **−16.0** [−16.7, −15.3] | the dominant safety mechanism |

## 8. Limitations

* **Synthetic world; main results use a simulated LLM.** Sections 6-7 show architectural
  effects under assumptions I chose (p_know = 0.55, always-hallucinate); Section 6b shows a real
  LLM violates the latter. The real-LLM run is small (2 seeds, 60 tasks), CPU-only, one model,
  one prompt design; questions are still generated from 4 relation templates (paraphrased in
  2 variants each), so it does not test open-ended language understanding.
* **Neo4j** was verified only as a drop-in backend: tests/test_neo4j.py checks identical agent
  metrics to the in-memory graph on a small world; the large experiments never used it and no
  performance/scale testing was done.
* **Oracle feedback** (true fact chain on 30% of tasks) is a strong assumption; real feedback
  is noisy and sparser. Memory gains scale with it.
* **Outdated-fact detection is weak:** only age × relation-volatility. Hallucination under drift
  stays 25–39%. Genuinely detecting change (e.g. temporal KGs, source freshness) is future work.
* **Thresholds/costs are design choices:** the abstention threshold is dev-tuned under
  `wrong_cost = 2`; other operating points trade success for safety (see the sweep).
* **Forgetting is not stress-tested:** memory is unbounded and the LLM is never fine-tuned, so
  `retention` mostly reflects the KG staying intact. No continual-learning algorithm is implemented.
* **Small functional schema** (4 relations, one object per subject); the 10 seeds are the
  statistical unit, so CIs are over worlds, not over tasks. No related-work comparison was run.

## 9. Future Research Directions

1. Run the full 10-seed suite with a real LLM (needs a GPU-backed Ollama; planning accuracy is the
   bottleneck), LLM-based claim decomposition, genuinely open-ended questions, and real KGs
   (e.g. temporal KGs) where the LLM actually has parametric knowledge to compare against.
2. Better staleness detection (temporal validity, change-point detection, source reliability
   learning) to attack the remaining drift-induced hallucinations.
3. Learned/calibrated confidence and risk–coverage analysis instead of one tuned threshold.
4. Noisy, sparse, delayed feedback; bounded memory with consolidation; LLM-side continual learning
   with forgetting measurements.
5. Benchmarks beyond a synthetic chain schema; comparison with published KG-agent methods.

## Repository layout

```
agents/       planner (rule-based + LLM), executor (ReAct + verified search), verifier, memory, llm (simulated + Ollama), systems
knowledge/    schema, graph_store (in-memory), neo4j_store (Neo4j), graph_builder, graph_retriever, graph_updater
environments/ dynamic_kg.py   - drifting world + noisy KG/corpus + task stream (templated + natural-language) + feedback
evaluation/   benchmarks, metrics, experiments, tune, run_ollama
configs/      default.json, ollama.json    tests/  21 tests    docker-compose.yml (Neo4j)
results/      summary.md, summary.json, raw.json, plots;  results/ollama/  real-LLM run + reply cache
```
