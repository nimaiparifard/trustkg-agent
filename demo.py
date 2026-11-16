"""Show reasoning traces of the full agent on a few tasks of the dynamic environment.

    python demo.py [seed]
"""
import sys

from agents.systems import SYSTEMS, build_agent
from environments.dynamic_kg import DynamicKGEnvironment, EnvConfig, condition_config


def main(seed: int = 0) -> None:
    env = DynamicKGEnvironment(condition_config(EnvConfig(), "mixed"), seed)
    agent = build_agent(SYSTEMS["trustkg_full"], env, seed)
    for stage in (0, 3):
        print(f"\n=== stage {stage} (world drift vs. stage 0: {100*env.drift_stats()[stage]:.0f}% of facts) ===")
        for task in env.tasks[stage][:4]:
            res = agent.solve(task.text, stage)
            verdict = "ABSTAIN" if res.abstained else ("CORRECT" if res.answer == task.gold else "WRONG")
            print(f"\nQ: {task.text}\n   gold={task.gold}  answer={res.answer}  conf={res.confidence:.2f}  -> {verdict}")
            for line in res.trace:
                print("   ", line)
            fb = env.feedback(task)
            if fb:
                agent.feedback(fb, stage, res.answer == task.gold)
                print("    FEEDBACK received -> KG updated, staleness model refined")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
