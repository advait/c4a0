"""Summarize convergence runs: per run, the dev-suite score curve across generations.

Usage: uv run python scripts/converge_report.py <dir> [--budget 256] [--every 5]

<dir> is one training base dir (e.g. training/) or a directory holding one base dir per run
(e.g. ctrl-r1/, q05-r1/). The headline is the
convergence predicate: MCTS keeps-the-result top-1 at --budget. Strict fastest-win top-1 and the
opening buckets (plies 0-11) are shown beside it. Runs whose arm name differs only in a trailing
-r<N> are grouped, so replicate spread is visible.
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def top1(policy_eval: dict) -> float | None:
    tallies = policy_eval["by_bucket"].values()
    n = sum(t["n"] for t in tallies)
    return sum(t["hits"] for t in tallies) / n if n else None


def bucket(policy_eval: dict, name: str) -> float | None:
    t = policy_eval["by_bucket"].get(name)
    return t["hits"] / t["n"] if t and t["n"] else None


def curve(run_dir: Path, budget: str) -> list[dict]:
    rows = []
    for meta in run_dir.glob("*/metadata.json"):
        gen = json.loads(meta.read_text())
        ev = gen.get("evals", {}).get("v1/dev")
        if not ev or budget not in ev.get("mcts", {}):
            continue
        keeps = ev.get("mcts_keeps", {}).get(budget)
        rows.append(
            {
                "gen": gen["gen_n"],
                "keeps": top1(keeps) if keeps else None,
                "strict": top1(ev["mcts"][budget]),
                "keeps_00_05": bucket(keeps, "00-05") if keeps else None,
                "keeps_06_11": bucket(keeps, "06-11") if keeps else None,
            }
        )
    return sorted(rows, key=lambda r: r["gen"])


def pct(x: float | None) -> str:
    return "  -  " if x is None else f"{x * 100:5.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--budget", default="256")
    ap.add_argument("--every", type=int, default=5)
    args = ap.parse_args()

    # One run (generation dirs inside) or a directory of runs.
    if any(args.experiment.glob("*/metadata.json")):
        runs = {args.experiment.name: curve(args.experiment, args.budget)}
    else:
        runs = {
            d.name: curve(d, args.budget)
            for d in sorted(args.experiment.iterdir())
            if d.is_dir()
        }
    arms: dict[str, list[str]] = defaultdict(list)
    for name in runs:
        arms[re.sub(r"-r\d+$", "", name)].append(name)

    print(
        f"MCTS-{args.budget} on v1/dev. keeps = predicate (keeps the game result), strict = fastest win"
    )
    for arm, names in arms.items():
        print(f"\n{arm}")
        for name in names:
            rows = runs[name]
            if not rows:
                print(f"  {name}: no evaluated generations")
                continue
            shown = [r for r in rows if r["gen"] % args.every == 0 or r is rows[-1]]
            line = " ".join(f"g{r['gen']}:{pct(r['keeps'])}" for r in shown)
            last = rows[-1]
            late = [r["keeps"] for r in rows if r["keeps"] is not None][-10:]
            print(
                f"  {name}: {line}\n"
                f"    last gen {last['gen']}: keeps {pct(last['keeps'])} strict {pct(last['strict'])}"
                f" | plies 0-5 {pct(last['keeps_00_05'])} 6-11 {pct(last['keeps_06_11'])}"
                f" | mean of last {len(late)}: {pct(statistics.mean(late)) if late else '  -  '}"
            )


if __name__ == "__main__":
    main()
