===============================================================================
TrustKG-Agent
A Knowledge-Grounded and Self-Reflective Agent for Reliable Multi-Step Reasoning
===============================================================================

1. OVERVIEW
-------------------------------------------------------------------------------
TrustKG-Agent is an LLM-based agent that uses a Knowledge Graph (KG) for
reasoning, planning, and verifying its own answers, instead of relying only on
the parametric knowledge of the language model.

The project targets three research areas:
  1. Planning & Reasoning
  2. Structured Knowledge
  3. Trustworthy and Adaptive Agents

Research question:
  Can structured knowledge and evidence-based verification improve the
  reliability and adaptability of LLM-based agents in multi-step reasoning
  tasks?

Hypotheses:
  H1  Using a Knowledge Graph reduces multi-step reasoning errors.
  H2  Evidence-based verification reduces the number of unsupported answers.
  H3  Experience memory and knowledge updates improve agent performance in
      changing environments.


2. HIGH-LEVEL ARCHITECTURE
-------------------------------------------------------------------------------
The system is split into six modules:

  +-------------------------------------------------------------+
  |               1. Task Understanding & Planning              |
  |   goal decomposition | plan generation | uncertainty est.   |
  +------------------------------+------------------------------+
                                 | task DAG
                                 v
  +-------------------------------------------------------------+
  |                   3. Agent Execution                        |
  |     tool use | multi-step actions | environment interaction |
  +--------------+-------------------------------+--------------+
                 | queries                       ^ evidence
                 v                               |
  +-------------------------------------------------------------+
  |             2. Knowledge-Grounded Reasoning                 |
  |      Knowledge Graph | GraphRAG | multi-hop retrieval       |
  +------------------------------+------------------------------+
                                 | draft answer + evidence
                                 v
  +-------------------------------------------------------------+
  |                 4. Trust & Verification                     |
  |  evidence checking | hallucination detection | confidence   |
  +------+-----------------------+------------------------------+
         | accept / revise /     | verified outcome
         | abstain / replan      v
         |        +----------------------------------------------+
         |        |            5. Adaptive Memory                |
         |        | episodic memory | knowledge updates |        |
         |        | learning over time                           |
         |        +----------------------+-----------------------+
         |                               |
         v                               v
  +-------------------------------------------------------------+
  |                6. Evaluation & Feedback                     |
  |        success | reliability | adaptation | recovery        |
  +-------------------------------------------------------------+

Control loop (per task):
  1. Planner decomposes the goal into a task DAG and estimates uncertainty
     for each sub-goal.
  2. Executor runs sub-goals in a ReAct-style loop (thought -> action ->
     observation), calling tools and the KG retriever.
  3. Retriever returns multi-hop KG paths and supporting text chunks as
     evidence; each intermediate claim is linked to its evidence.
  4. Verifier splits the draft answer into atomic claims, checks each claim
     against the evidence, detects contradictions or stale facts, and
     assigns a confidence score.
  5. Based on the verdict the agent: accepts the answer, revises it,
     replans the failed sub-goals, or abstains when confidence is too low.
  6. The outcome (plan, actions, errors, corrections) is written to memory;
     confirmed new facts are proposed as KG updates.
  7. The evaluation module logs all metrics for the run.


3. MODULE DETAILS
-------------------------------------------------------------------------------

3.1 Task Understanding & Planning  (agents/planner.py)
  - Goal decomposition: break a complex task into ordered sub-goals.
  - Plan generation: represent the plan as a Task DAG (nodes = sub-goals,
    edges = dependencies).
  - Uncertainty estimation: score each sub-goal so that risky steps get
    extra retrieval and stricter verification.
  - Replanning: rebuild only the failed part of the DAG when verification
    or execution fails.
  - Uses past episodes from memory to reuse plans that worked before.

3.2 Knowledge-Grounded Reasoning  (knowledge/)
  - graph_builder.py   : build the KG (entities, relations, provenance,
                         timestamps) from source data and store in Neo4j.
  - graph_retriever.py : hybrid retrieval = GraphRAG (subgraph / community
                         retrieval) + vector search over text chunks;
                         multi-hop path retrieval for compositional queries.
  - graph_updater.py   : add, modify, and deprecate facts; keep version and
                         validity time for each edge so stale facts can be
                         detected.
  - Every retrieved fact carries: source, timestamp, confidence.

3.3 Agent Execution  (agents/executor.py)
  - Custom Python orchestration (no dependency on LangChain / CrewAI for the
    core loop).
  - ReAct loop over the plan DAG.
  - Tool registry: KG query (Cypher), vector search, calculator, and
    environment actions.
  - Tracks every tool call, observation, and error for later analysis.

3.4 Trust & Verification  (agents/verifier.py)
  - Claim decomposition: split the answer into small, checkable claims.
  - Evidence retrieval per claim and claim-evidence relevance scoring
    (supported / contradicted / not enough evidence).
  - Hallucination detection: flag claims with no supporting evidence.
  - Conflict detection: detect contradictory or outdated facts in the KG.
  - Confidence estimation: aggregate claim-level scores into an answer
    confidence.
  - Decision policy: accept, revise, replan, or abstain.

3.5 Adaptive Memory  (agents/memory.py)
  - Episodic memory: stores past tasks, plans, actions, failures, and fixes.
  - Semantic memory: distilled, verified facts and lessons learned.
  - Knowledge updates: verified new information is pushed to the KG through
    graph_updater.py.
  - Learning over time: retrieve relevant past episodes to guide planning
    and avoid repeating errors. (No LLM fine-tuning in the first version;
    a continual-learning method can be added later.)

3.6 Evaluation & Feedback  (evaluation/)
  - benchmarks.py   : task sets for multi-step reasoning and dynamic KG tasks.
  - metrics.py      : metric implementations (see Section 5).
  - experiments.py  : runs baselines, ablations, and the full system with
                      fixed configs and seeds for reproducibility.


4. DYNAMIC ENVIRONMENT FOR CONTINUAL LEARNING  (environments/dynamic_kg.py)
-------------------------------------------------------------------------------
A small simulation environment:
  - Initial KG with ~1000 concepts and their relations.
  - Over time (in stages):
      * new facts are added,
      * some relations change,
      * some facts become outdated,
      * new task types are given to the agent.
  - Controlled noise injection: incomplete, contradictory, and outdated
    information.

Questions measured in this environment:
  1. Does the agent learn the new information?
  2. Does it retain previous knowledge (no forgetting)?
  3. Can it detect outdated information?
  4. Does it correct its planning after the environment changes?


5. EXPERIMENTAL DESIGN
-------------------------------------------------------------------------------
Configurations (incremental ablation):
  Baseline       LLM only (no agent)
  Experiment 1   ReAct agent
  Experiment 2   Agent + vector RAG
  Experiment 3   Agent + Knowledge Graph
  Experiment 4   Agent + KG + Verification
  Experiment 5   Full architecture with Adaptive Memory

Metrics:
  - Task Success Rate
  - Reasoning Accuracy
  - Evidence Precision
  - Hallucination Rate
  - Recovery Rate (successful recovery after an error)
  - Number of Tool Calls (efficiency)
  - Adaptation Performance (on the dynamic KG)

Key stress test:
  Can the system avoid wrong decisions when the KG contains incomplete,
  contradictory, or outdated information?


6. TECHNOLOGY STACK
-------------------------------------------------------------------------------
  LLM                  Qwen or Llama (locally runnable model)
  Agent orchestration  Python, custom implementation
  Knowledge Graph      Neo4j
  Retrieval            GraphRAG + vector search
  Planning             Task DAG + ReAct
  Verification         Evidence-based verification
  Memory               Episodic + semantic memory
  Evaluation           Python, PyTorch, defined benchmarks


7. REPOSITORY STRUCTURE
-------------------------------------------------------------------------------
TrustKG-Agent/
|
|-- agents/
|   |-- planner.py
|   |-- executor.py
|   |-- verifier.py
|   `-- memory.py
|
|-- knowledge/
|   |-- graph_builder.py
|   |-- graph_retriever.py
|   `-- graph_updater.py
|
|-- evaluation/
|   |-- benchmarks.py
|   |-- metrics.py
|   `-- experiments.py
|
|-- environments/
|   `-- dynamic_kg.py
|
|-- configs/
|-- experiments/
|-- results/
|-- architecture.md
|-- README.md
`-- requirements.txt


8. DEVELOPMENT PLAN (4 WEEKS)
-------------------------------------------------------------------------------
  Week 1 - Core Agent
    planner, task decomposition, tool calling, Neo4j connection
  Week 2 - Knowledge Grounding
    graph retrieval, multi-hop reasoning, linking evidence to answers,
    evidence verification
  Week 3 - Trustworthiness
    contradiction detection, confidence estimation, abstention under
    uncertainty, tests with incomplete information
  Week 4 - Adaptation & Evaluation
    KG updates, episodic memory, distribution-shift experiments,
    baseline comparison


9. README CONTENTS
-------------------------------------------------------------------------------
  - Research Motivation
  - Research Questions
  - Architecture
  - Experimental Setup
  - Baselines
  - Quantitative Results
  - Ablation Study
  - Limitations
  - Future Research Directions


10. IMPLEMENTATION NOTES (what was actually built)
-------------------------------------------------------------------------------
  - The main experiments run without external services on knowledge/graph_store.py
    (InMemoryGraph). knowledge/neo4j_store.py (+ docker-compose.yml) implements the same
    interface; tests/test_neo4j.py verifies identical agent metrics on a real Neo4j server.
  - The main experiments use a controllable simulated LLM (agents/llm.py: SimulatedLLM:
    frozen parametric knowledge, type-correct hallucination) with a rule-based planner and
    text reader. agents/llm.py:OllamaLLM (Qwen2.5-7B via Ollama, cached, deterministic) can act
    as planner (natural-language question -> relation chain, schema/type validated), text
    reader (grounded in passages) and baseline answerer; see evaluation/run_ollama.py.
  - Retrieval: GraphRetriever (KG lookup / multi-hop), VectorRetriever
    (TF-IDF cosine), HybridRetriever (KG first, text as recovery evidence).
  - Verification: noisy-OR evidence combination, conflict penalty, staleness
    discount (1 - hazard)^age, claim-level checks, accept/abstain threshold
    tuned on dev seeds. Search over the plan DAG backtracks (beam=2).
  - Memory: episodic log; semantic per-relation staleness (hazard) estimate;
    feedback -> GraphUpdater (confirm / change / new, old facts deprecated).
  - Evaluation: evaluation/{benchmarks,metrics,experiments,tune}.py;
    `python -m evaluation.experiments` regenerates results/.
