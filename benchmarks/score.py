"""Aggregate results/*.jsonl into a comparison table."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def main(results_dir: str = "results") -> None:
    rows = defaultdict(lambda: {"n": 0, "correct": 0, "tokens": 0, "latency": 0.0,
                                "unfinished": 0, "errors": 0})
    for path in sorted(Path(results_dir).glob("*.jsonl")):
        task, mode, model = path.stem.split(".", 2)
        key = (task, mode, model)
        for line in open(path):
            r = json.loads(line)
            row = rows[key]
            row["n"] += 1
            row["correct"] += bool(r.get("correct"))
            row["tokens"] += r.get("tokens", 0)
            row["latency"] += r.get("latency_s", 0)
            row["unfinished"] += not r.get("finished", True)
            row["errors"] += "error" in r

    hdr = f"{'task':<16}{'mode':<9}{'model':<40}{'n':>5}{'acc%':>8}{'tok/q':>10}{'s/q':>8}{'unfin':>7}{'err':>5}"
    print(hdr)
    print("-" * len(hdr))
    for (task, mode, model), r in sorted(rows.items()):
        n = r["n"] or 1
        print(f"{task:<16}{mode:<9}{model:<40}{r['n']:>5}"
              f"{100*r['correct']/n:>8.1f}{r['tokens']//n:>10}{r['latency']/n:>8.1f}"
              f"{r['unfinished']:>7}{r['errors']:>5}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
