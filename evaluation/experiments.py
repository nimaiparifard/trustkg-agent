"""Run the baselines, the full system and ablations with fixed configs and seeds.

Usage:  python -m evaluation.experiments --config configs/default.json --out results
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List

import numpy as np

from agents.executor import AgentConfig
from agents.systems import ABLATIONS, SYSTEMS, build_agent
from environments.dynamic_kg import DynamicKGEnvironment, EnvConfig

from .benchmarks import describe, load_config, make_env
from .metrics import TaskRecord, compute_metrics, paired_bootstrap


def run_agent(cfg: AgentConfig, env: DynamicKGEnvironment, seed: int, llm_cfg: dict, llm=None, graph=None) -> Dict:
    """Run one agent through all stages of one environment trajectory.

    With `llm` (a real backend) the agent receives the natural-language question and its
    LLM planner must recover the relation chain itself."""
    agent = build_agent(cfg, env, seed, llm=llm, graph=graph, **llm_cfg)
    recs: List[TaskRecord] = []
    for stage in range(env.cfg.n_stages + 1):
        truth = env.truth(stage)
        for task in env.tasks[stage]:
            res = agent.solve(task.nl_text if llm is not None else task.text, stage)
            plan_ok = res.plan is not None and res.plan.start == task.start and res.plan.relations == task.rels
            answered = not res.abstained and res.answer is not None
            correct = answered and res.answer == task.gold
            ev = [e for e in res.evidence if e.kind != "llm"]
            recs.append(TaskRecord(
                stage, task.category, task.hops, answered, correct,
                answered and res.fully_supported(task.hops),
                len(ev), sum(truth.get((e.subject, e.relation)) == e.object for e in ev),
                res.tool_calls, res.llm_calls, res.failure_events, res.recovery_attempted, plan_ok))
            fb = env.feedback(task)
            if fb:
                agent.feedback(fb, stage, correct)
    return compute_metrics(recs, env.cfg.n_stages)


def run_suite(systems: Dict[str, AgentConfig], base: EnvConfig, condition: str, seeds: List[int],
              llm_cfg: dict, llm=None) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = defaultdict(list)
    for seed in seeds:
        env = make_env(base, condition, seed)
        for name, cfg in systems.items():
            out[name].append(run_agent(cfg, env, seed, llm_cfg, llm))
    return out


# ------------------------------------------------------------------ reporting
SCALAR = ["plan_accuracy", "task_success", "reasoning_accuracy", "coverage", "abstain_rate", "hallucination_rate",
          "unsupported_rate", "evidence_precision", "recovery_rate", "failure_containment", "tool_calls", "llm_calls",
          "post_shift_success", "post_shift_hallucination", "final_stage_success", "retention",
          "plasticity", "novel_success"]


def agg(runs: List[Dict], key: str):
    vals = [r[key] for r in runs if r[key] is not None]
    return (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0) if vals else None


def summarize(results: Dict[str, List[Dict]]) -> Dict:
    s = {n: {k: agg(runs, k) for k in SCALAR} for n, runs in results.items()}
    for n, runs in results.items():
        s[n]["stage_success"] = np.mean([r["stage_success"] for r in runs], 0).tolist()
        s[n]["stage_hallucination"] = np.mean([r["stage_hallucination"] for r in runs], 0).tolist()
    return s


def fmt(v, pct=True):
    if v is None:
        return "n/a"
    m, sd = v
    return f"{100*m:.1f} ± {100*sd:.1f}" if pct else f"{m:.2f} ± {sd:.2f}"


def table(summary: Dict, cols: List[str], title: str = "") -> str:
    head = {"task_success": "Success", "reasoning_accuracy": "Reason.Acc", "abstain_rate": "Abstain",
            "hallucination_rate": "Halluc.", "unsupported_rate": "Unsupp.", "evidence_precision": "Ev.Prec",
            "recovery_rate": "Recovery", "failure_containment": "Fail-contain", "tool_calls": "KG/text calls", "llm_calls": "LLM calls",
            "post_shift_success": "Post-shift succ", "post_shift_hallucination": "Post-shift halluc",
            "retention": "Retention", "plasticity": "Plasticity", "novel_success": "Novel", "coverage": "Coverage",
            "final_stage_success": "Final-stage succ", "plan_accuracy": "Plan acc"}
    lines = [f"**{title}**" if title else "", "",
             "| System | " + " | ".join(head.get(c, c) for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for name, m in summary.items():
        row = [fmt(m[c], pct=c not in ("tool_calls", "llm_calls")) for c in cols]
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def hypothesis_tests(results: Dict[str, List[Dict]]) -> List[Dict]:
    """Paired (same seed/world) bootstrap tests for H1-H3."""
    def col(sys, key):
        return [r[key] for r in results[sys]]
    tests = [
        ("H1", "KG vs. no KG: lower hallucination (react_kg vs react)", "react_kg", "react", "hallucination_rate", -1),
        ("H1", "KG vs. vector RAG: lower hallucination (react_kg vs react_vector_rag)", "react_kg",
         "react_vector_rag", "hallucination_rate", -1),
        ("H2", "Verification: lower hallucination (trustkg_verify vs react_kg)", "trustkg_verify", "react_kg",
         "hallucination_rate", -1),
        ("H2", "Verification: fewer unsupported answers (trustkg_verify vs react_kg)", "trustkg_verify",
         "react_kg", "unsupported_rate", -1),
        ("H2", "Verification: higher reasoning accuracy on answered tasks", "trustkg_verify", "react_kg",
         "reasoning_accuracy", +1),
        ("H3", "Memory: higher post-shift success (trustkg_full vs trustkg_verify)", "trustkg_full",
         "trustkg_verify", "post_shift_success", +1),
        ("H3", "Memory: lower post-shift hallucination", "trustkg_full", "trustkg_verify",
         "post_shift_hallucination", -1),
        ("H3", "Memory: higher final-stage success", "trustkg_full", "trustkg_verify",
         "final_stage_success", +1),
    ]
    out = []
    for h, desc, a, b, key, direction in tests:
        if a in results and b in results:
            d, lo, hi, p = paired_bootstrap(col(a, key), col(b, key))
            supported = (lo > 0) if direction > 0 else (hi < 0)
            out.append(dict(h=h, desc=desc, metric=key, diff=d, lo=lo, hi=hi, supported=bool(supported)))
    return out


def plot(summary_by_cond: Dict[str, Dict], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    s = summary_by_cond["mixed"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, m in s.items():
        xs = range(len(m["stage_success"]))
        ax[0].plot(xs, [100 * v for v in m["stage_success"]], marker="o", label=name)
        ax[1].plot(xs, [100 * v for v in m["stage_hallucination"]], marker="o", label=name)
    ax[0].set(title="Task success per stage (mixed KG)", xlabel="stage (world drifts)", ylabel="% tasks")
    ax[1].set(title="Hallucination (confident wrong) per stage", xlabel="stage", ylabel="% tasks")
    for name, m in s.items():
        ax[2].scatter(100 * m["task_success"][0], 100 * m["hallucination_rate"][0], s=60)
        ax[2].annotate(name, (100 * m["task_success"][0], 100 * m["hallucination_rate"][0]), fontsize=7)
    ax[2].set(title="Success vs. hallucination", xlabel="task success %", ylabel="hallucination %")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stages_mixed.png"), dpi=140)
    plt.close(fig)

    conds = list(summary_by_cond)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    w = 0.8 / len(s)
    for i, name in enumerate(s):
        ax[0].bar([c + i * w for c in range(len(conds))],
                  [100 * summary_by_cond[c][name]["task_success"][0] for c in conds], w, label=name)
        ax[1].bar([c + i * w for c in range(len(conds))],
                  [100 * summary_by_cond[c][name]["hallucination_rate"][0] for c in conds], w, label=name)
    for a, t in zip(ax, ["Task success %", "Hallucination rate %"]):
        a.set_xticks([c + 0.4 for c in range(len(conds))])
        a.set_xticklabels(conds)
        a.set_title(t + " by KG defect")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "conditions.png"), dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=None, help="override number of seeds")
    args = ap.parse_args()
    conf = load_config(args.config)
    base = EnvConfig.from_dict(conf.get("env", {}))
    seeds = list(range(args.seeds or conf.get("n_seeds", 10)))
    llm_cfg = conf.get("llm", {})
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    conds = conf.get("conditions", ["mixed", "incomplete", "contradictory", "stale"])
    systems, ablations = dict(SYSTEMS), dict(ABLATIONS)
    tuning = None
    if conf.get("tune_threshold", True):
        from .tune import tune_threshold
        tuning = tune_threshold(base, conds, conf.get("dev_seeds", list(range(100, 105))), llm_cfg,
                                 conf.get("wrong_cost", 2.0))
        th = tuning["best"]
        systems = {n: (replace(c, threshold=th) if c.verify else c) for n, c in systems.items()}
        ablations = {n: replace(c, threshold=th) for n, c in ablations.items()}
        print(f"[{time.time()-t0:5.0f}s] tuned threshold on dev seeds: {th}")

    raw: Dict = {"config": conf, "seeds": seeds, "tuning": tuning}
    summaries: Dict[str, Dict] = {}
    tests_by_cond: Dict[str, List[Dict]] = {}
    for cond in conds:
        res = run_suite(systems, base, cond, seeds, llm_cfg)
        raw[f"main/{cond}"] = res
        summaries[cond] = summarize(res)
        tests_by_cond[cond] = hypothesis_tests(res)
        print(f"[{time.time()-t0:5.0f}s] main suite: {cond}")
    abl = run_suite(ablations, base, "mixed", seeds, llm_cfg)
    raw["ablation/mixed"] = abl
    abl_summary = summarize(abl)
    abl_tests = []
    for name in ablations:
        if name != "full":
            for key, direction in [("post_shift_success", +1), ("hallucination_rate", -1)]:
                d, lo, hi, _ = paired_bootstrap([r[key] for r in abl["full"]], [r[key] for r in abl[name]])
                abl_tests.append(dict(component=name, metric=key, diff_full_minus_ablated=d, lo=lo, hi=hi))
    print(f"[{time.time()-t0:5.0f}s] ablations")

    # env description for the report
    desc = describe(make_env(base, "mixed", 0))
    with open(os.path.join(args.out, "raw.json"), "w") as f:
        json.dump(raw, f)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"summaries": summaries, "ablation": abl_summary, "hypothesis_tests": tests_by_cond,
                   "ablation_tests": abl_tests, "env": desc, "tuning": tuning}, f, indent=1)
    plot(summaries, args.out)

    md = [f"# Results (n_seeds={len(seeds)}, tasks/run={desc['n_tasks']}, mean ± std over seeds)", ""]
    if tuning:
        md += [f"Abstention threshold tuned on dev seeds (disjoint from test seeds) with wrong_cost={conf.get('wrong_cost', 2.0)}: **{tuning['best']}**", "",
               "| threshold | dev utility (success − wrong_cost × hallucination) |", "|---|---|"]
        md += [f"| {r['threshold']} | {100*r['utility']:.1f} |" for r in tuning["rows"]] + [""]
    main_cols = ["task_success", "reasoning_accuracy", "abstain_rate", "hallucination_rate", "unsupported_rate",
                 "evidence_precision", "recovery_rate", "failure_containment", "tool_calls", "llm_calls"]
    for cond, s in summaries.items():
        md += [table(s, main_cols, f"Condition: {cond}"), ""]
    md += [table(summaries["mixed"], ["post_shift_success", "post_shift_hallucination", "retention",
                                      "plasticity", "novel_success", "final_stage_success"],
                 "Adaptation under drift (condition: mixed)"), ""]
    md += ["**Success per stage (mixed)**", "", "| System | " + " | ".join(f"s{i}" for i in range(len(
        summaries["mixed"]["llm_only"]["stage_success"]))) + " |",
           "|---|" + "---|" * len(summaries["mixed"]["llm_only"]["stage_success"])]
    for n, m in summaries["mixed"].items():
        md.append(f"| {n} | " + " | ".join(f"{100*v:.1f}" for v in m["stage_success"]) + " |")
    md += ["", table(abl_summary, ["task_success", "hallucination_rate", "abstain_rate", "post_shift_success",
                                   "post_shift_hallucination", "tool_calls"], "Ablation (condition: mixed)"), ""]
    md += ["**Ablation: full minus ablated (paired, 95% bootstrap CI over seeds)**", "",
           "| Removed | Metric | Δ (full − ablated) | 95% CI |", "|---|---|---|---|"]
    for t in abl_tests:
        md.append(f"| {t['component']} | {t['metric']} | {100*t['diff_full_minus_ablated']:+.1f} pp | "
                  f"[{100*t['lo']:+.1f}, {100*t['hi']:+.1f}] |")
    md += ["", "**Hypothesis tests (paired over seeds; difference a − b, 95% bootstrap CI)**", ""]
    for cond, tests in tests_by_cond.items():
        md += [f"_Condition: {cond}_", "", "| H | Comparison | Metric | Δ | 95% CI | Supported |", "|---|---|---|---|---|---|"]
        for t in tests:
            md.append(f"| {t['h']} | {t['desc']} | {t['metric']} | {100*t['diff']:+.1f} pp | "
                      f"[{100*t['lo']:+.1f}, {100*t['hi']:+.1f}] | {'yes' if t['supported'] else 'no'} |")
        md.append("")
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"done in {time.time()-t0:.0f}s -> {args.out}/summary.md")


if __name__ == "__main__":
    main()
