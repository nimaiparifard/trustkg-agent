"""Run the six systems with a REAL local LLM (Ollama) and, for comparison, with the simulated LLM
on the identical environments.  Resumable (per-seed JSON) and cached (results/ollama_cache.jsonl).

    python -m evaluation.run_ollama --config configs/ollama.json --out results/ollama

With the real LLM the agent gets the *natural-language* question; the LLM must plan (question ->
relation chain) and read retrieved text. Entities are invented, so its parametric knowledge is ~nil.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from agents.llm import OllamaLLM
from agents.systems import SYSTEMS
from environments.dynamic_kg import EnvConfig

from .benchmarks import load_config, make_env
from .experiments import SCALAR, agg, fmt, summarize, table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ollama.json")
    ap.add_argument("--out", default="results/ollama")
    args = ap.parse_args()
    conf = load_config(args.config)
    base = EnvConfig.from_dict(conf["env"])
    os.makedirs(args.out, exist_ok=True)
    llm = OllamaLLM(conf["model"], conf["host"], cache_path=os.path.join(args.out, "ollama_cache.jsonl"))
    from .experiments import run_agent
    t0 = time.time()
    real, sim = {n: [] for n in SYSTEMS}, {n: [] for n in SYSTEMS}
    for seed in conf["seeds"]:
        path = os.path.join(args.out, f"seed{seed}.json")
        if os.path.exists(path):
            d = json.load(open(path))
        else:
            env = make_env(base, conf["condition"], seed)
            d = {"real": {}, "sim": {}}
            for name, cfg in SYSTEMS.items():
                d["sim"][name] = run_agent(cfg, env, seed, conf["llm"])
                d["real"][name] = run_agent(cfg, env, seed, {}, llm=llm)
                print(f"[{time.time()-t0:6.0f}s] seed {seed} {name:18s} real success="
                      f"{100*d['real'][name]['task_success']:.1f}% plan_acc={100*d['real'][name]['plan_accuracy']:.0f}% "
                      f"(llm calls so far {llm.calls}, cache hits {llm.cache_hits})", flush=True)
            json.dump(d, open(path, "w"))
        for n in SYSTEMS:
            real[n].append(d["real"][n])
            sim[n].append(d["sim"][n])
    cols = ["plan_accuracy", "task_success", "reasoning_accuracy", "abstain_rate", "hallucination_rate",
            "unsupported_rate", "evidence_precision", "tool_calls", "llm_calls"]
    md = [f"# Real LLM ({conf['model']}) vs simulated LLM — condition `{conf['condition']}`, seeds {conf['seeds']}, "
          f"{(conf['env']['n_stages']+1)*conf['env']['tasks_per_stage']} tasks/run (mean ± std over seeds, %)", "",
          table(summarize(real), cols, f"Real LLM: {conf['model']} (natural-language questions)"), "",
          table(summarize(sim), cols, "Simulated LLM (templated questions), same environments"), ""]
    open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8").write("\n".join(md))
    json.dump({"real": summarize(real), "sim": summarize(sim)}, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(f"done in {time.time()-t0:.0f}s -> {args.out}/summary.md")


if __name__ == "__main__":
    main()
