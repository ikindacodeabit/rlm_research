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
    rows = defaultdict(lambda: {"n": 0, "score": 0.0, "tokens": 0, "latency": 0.0,
                                "unfinished": 0, "errors": 0,
                                "peak_sum": 0, "peak_n": 0, "budget": None,
                                "abstain": 0,
                                # Truncation exposure for the vanilla arm.
                                "seen": 0, "seen_n": 0, "ctx_frac": 0.0, "ctx_n": 0})
    for path in sorted(Path(results_dir).rglob("*.jsonl")):
        variant = path.parent.name if path.parent != Path(results_dir) else "root"
        task, mode, model = path.stem.split(".", 2)
        # Dedupe by example id: a retried (non-error) record replaces an earlier
        # errored one from e.g. a transient NIM outage; later records win ties.
        by_id: dict = {}
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            prev = by_id.get(r["id"])
            if prev is None or not r.get("error") or prev.get("error"):
                by_id[r["id"]] = r
        for r in by_id.values():
            # group per RULER/LongBench/LOFT subset when present; one group
            # otherwise. `metric` is part of the key so a single cell can never
            # silently average, say, qa_f1 with rouge or subspan-EM.
            key = (task, variant, mode, model, r.get("subset") or "",
                   r.get("metric") or "?")
            row = rows[key]
            row["n"] += 1
            if r.get("score") is None:
                # Pre-standard records stored only a boolean `correct`. Deriving a
                # score from it silently reports 0.0 for a partially-correct answer,
                # so refuse rather than invent a number — run scripts/rescore.py.
                raise SystemExit(
                    f"{path}: record {r.get('id')!r} has no `score` field. "
                    "Re-score it first: python scripts/rescore.py <results_dir>"
                )
            row["score"] += float(r["score"])
            row["tokens"] += r.get("tokens", 0)
            row["latency"] += r.get("latency_s", 0)
            row["unfinished"] += not r.get("finished", True)
            # Abstention: the RLM can return pred=None (max_steps / ungrounded_final /
            # run_timeout) and score a structural 0. Vanilla always emits something and
            # can pick up accidental partial credit, so a score column alone is not a
            # like-for-like comparison -- ~12% of RLM records abstain. Report it.
            row["abstain"] += r.get("pred") is None
            row["errors"] += "error" in r
            # Truncation exposure. The vanilla arm is capped by --vanilla-char-limit,
            # so on a context longer than the cap its score is bounded by how often
            # the answer survived the cut -- measured at 47.1% for niah, which is
            # EXACTLY what vanilla scored. A score column alone reports that ceiling
            # as model accuracy. `seen%` is over records whose gold is quoted in the
            # context at all; `ctx%` covers the rest (derived answers like multikey's
            # sum, which is never present at any length).
            if r.get("gold_visible") is not None:
                row["seen_n"] += 1
                row["seen"] += bool(r["gold_visible"])
            if r.get("context_chars"):
                row["ctx_n"] += 1
                row["ctx_frac"] += r.get("context_chars_used", 0) / r["context_chars"]
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

    # "score%" not "acc%": for qa_f1/rouge/code_sim this column is a mean metric
    # value, not an accuracy, and calling it accuracy propagated into every plot.
    hdr = (f"{'task':<14}{'subset':<22}{'variant':<16}{'mode':<9}{'model':<34}"
           f"{'metric':<16}{'budget':>8}{'n':>5}{'score%':>8}{'abst%':>7}"
           f"{'seen%':>7}{'ctx%':>6}"
           f"{'tok/q':>9}{'s/q':>7}{'peakctx':>9}{'unfin':>7}{'err':>5}")
    print(hdr)
    print("-" * len(hdr))
    csv_rows = []
    for (task, variant, mode, model, subset, metric), r in sorted(rows.items()):
        n = r["n"] or 1
        budget = r["budget"] if r["budget"] is not None else ""
        peak = round(r["peak_sum"] / r["peak_n"]) if r["peak_n"] else ""
        acc = 100 * r["score"] / n
        abstain = 100 * r["abstain"] / n
        tok_q = r["tokens"] // n
        s_q = r["latency"] / n
        # seen% is the vanilla arm's CEILING: it cannot beat the rate at which the
        # gold survived truncation. Blank for the RLM (it reads the whole context)
        # and for derived answers, where ctx% is the signal instead.
        seen = 100 * r["seen"] / r["seen_n"] if r["seen_n"] else None
        ctxp = 100 * r["ctx_frac"] / r["ctx_n"] if r["ctx_n"] else None
        seen_s = f"{seen:.1f}" if seen is not None else ""
        ctx_s = f"{ctxp:.0f}" if ctxp is not None else ""
        print(f"{task:<14}{subset:<22}{variant:<16}{mode:<9}{model:<34}"
              f"{metric:<16}{str(budget):>8}{r['n']:>5}{acc:>8.1f}{abstain:>7.1f}"
              f"{seen_s:>7}{ctx_s:>6}"
              f"{tok_q:>9}{s_q:>7.1f}{str(peak):>9}{r['unfinished']:>7}{r['errors']:>5}")
        csv_rows.append({
            "task": task, "subset": subset, "variant": variant, "mode": mode,
            "model": model, "metric": metric, "budget": budget, "n": r["n"],
            "score": round(acc, 1), "acc": round(acc, 1),
            "abstain_pct": round(abstain, 1),
            "gold_seen_pct": round(seen, 1) if seen is not None else "",
            "ctx_kept_pct": round(ctxp, 1) if ctxp is not None else "",
            "tok_per_q": tok_q, "s_per_q": round(s_q, 1), "peak_ctx": peak,
            "finished": r["n"] - r["unfinished"], "unfin": r["unfinished"],
            "err": r["errors"],
        })

    if args.csv:
        # `acc` is kept as an alias of `score` so results/plot_budget.py and any
        # existing CSV consumer keep working after the rename.
        fields = ["task", "subset", "variant", "mode", "model", "metric", "budget",
                  "n", "score", "acc", "abstain_pct", "gold_seen_pct", "ctx_kept_pct",
                  "tok_per_q", "s_per_q", "peak_ctx", "finished", "unfin", "err"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nwrote {args.csv} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
