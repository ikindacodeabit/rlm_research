"""Aggregate results/*.jsonl into a comparison table.

Prints a human-readable table. With --csv PATH it also writes a machine-readable
CSV that carries the memory-BUDGET and peak-context columns needed for the
budget-sweep analysis (plot_budget.py).

Usage:
    python benchmarks/score.py [results_dir]
    python benchmarks/score.py results --csv results/budget_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def aggregate(results_dir: str):
    rows = defaultdict(lambda: {"n": 0, "correct": 0, "tokens": 0, "latency": 0.0,
                                "unfinished": 0, "errors": 0,
                                "peak_sum": 0, "peak_n": 0, "budget": None})
    for path in sorted(Path(results_dir).rglob("*.jsonl")):
        variant = path.parent.name if path.parent != Path(results_dir) else "root"
        task, mode, model = path.stem.split(".", 2)
        key = (task, variant, mode, model)
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            row = rows[key]
            row["n"] += 1
            row["correct"] += bool(r.get("correct"))
            row["tokens"] += r.get("tokens", 0)
            row["latency"] += r.get("latency_s", 0)
            row["unfinished"] += not r.get("finished", True)
            row["errors"] += "error" in r
            # budget / peak-context come from the RLM metrics dict (absent for vanilla)
            metrics = r.get("metrics") or {}
            if metrics.get("budget") is not None:
                row["budget"] = metrics["budget"]
            peak = metrics.get("peak_context_tokens")
            if peak is not None:
                row["peak_sum"] += peak
                row["peak_n"] += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="results")
    ap.add_argument("--csv", default=None, help="also write the table to this CSV path")
    args = ap.parse_args()

    rows = aggregate(args.results_dir)

    hdr = (f"{'task':<14}{'variant':<16}{'mode':<9}{'model':<22}"
           f"{'budget':>8}{'n':>5}{'acc%':>8}{'tok/q':>9}{'s/q':>7}"
           f"{'peakctx':>9}{'unfin':>7}{'err':>5}")
    print(hdr)
    print("-" * len(hdr))
    csv_rows = []
    for (task, variant, mode, model), r in sorted(rows.items()):
        n = r["n"] or 1
        budget = r["budget"] if r["budget"] is not None else ""
        peak = round(r["peak_sum"] / r["peak_n"]) if r["peak_n"] else ""
        acc = 100 * r["correct"] / n
        tok_q = r["tokens"] // n
        s_q = r["latency"] / n
        print(f"{task:<14}{variant:<16}{mode:<9}{model:<22}"
              f"{str(budget):>8}{r['n']:>5}{acc:>8.1f}{tok_q:>9}{s_q:>7.1f}"
              f"{str(peak):>9}{r['unfinished']:>7}{r['errors']:>5}")
        csv_rows.append({
            "task": task, "variant": variant, "mode": mode, "model": model,
            "budget": budget, "n": r["n"], "acc": round(acc, 1),
            "tok_per_q": tok_q, "s_per_q": round(s_q, 1), "peak_ctx": peak,
            "finished": r["n"] - r["unfinished"], "unfin": r["unfinished"],
            "err": r["errors"],
        })

    if args.csv:
        fields = ["task", "variant", "mode", "model", "budget", "n", "acc",
                  "tok_per_q", "s_per_q", "peak_ctx", "finished", "unfin", "err"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nwrote {args.csv} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
