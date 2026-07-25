#!/usr/bin/env python3
"""Classify RLM failures from saved transcripts, to separate harness bugs from
genuine retrieval failures.

run_benchmark writes one transcript per RLM example to
  {out_dir}/transcripts/{task}.rlm.{slug}/{id}.json
carrying {question, answers, pred, end_reason, metrics, transcript:[...]}.

Each example lands in one of five buckets:
  correct                  the SCORER gave it 1.0
  partial credit           0 < score < 1 (F1/ROUGE/recall/coverage-style metrics)
  no answer (<reason>)     pred is None — ungrounded FINAL, max_steps, exception
  SAW gold, reported wrong scored 0, but the gold text WAS in a REPL observation
                           -> a reporting/harness problem, OR a strict-metric
                              artifact (e.g. a LOFT gold span with a stray token)
  never found gold         scored 0 and the gold never appeared in any observation
                           -> search/capability problem, not a bug

Correctness comes from benchmarks.run_benchmark.score_record — the same function
score.py's numbers come from — so these buckets reconcile with the results table.
Only the "SAW gold" test is deliberately laxer than the metric; that is its job.

Usage:
    python scripts/triage_transcripts.py loft32k
    python scripts/triage_transcripts.py loft32k --show "never found gold"
    python scripts/triage_transcripts.py ruler16k --root results/ruler16k_nothink
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.longbench_metrics import normalize_answer
from benchmarks.run_benchmark import score_record

LOFT_MULTIVALUE = {"qampari", "quest"}


def _example_from(d: dict, task: str, subset: str) -> dict:
    """Rebuild the loader fields score_record needs from a transcript."""
    ex = {"id": subset, "answers": d.get("answers") or [], "subset": subset,
          "metric": d.get("metric"), "all_classes": d.get("all_classes")}
    if not ex["metric"]:
        ex["metric"] = "loft_subspan_em" if task.startswith("loft") else "recall"
    if ex["metric"].startswith("loft"):
        ex["multi_value"] = subset.rsplit("_", 1)[0] in LOFT_MULTIVALUE
        ex["answer_prefix"] = "Final Answer:"
    return ex


def classify(d: dict, task: str = "", subset: str = "") -> str:
    """Bucket one transcript.

    "correct" is decided by THE SCORER (score_record), not by a private containment
    test — this script used to carry its own definition of correct that agreed with
    neither `recall` nor the LongBench metrics, so its bucket counts could not be
    reconciled with score.py.

    "SAW gold, reported wrong" deliberately uses a LAXER test than the metric: it
    asks whether the gold text ever reached a REPL observation. That is the point of
    the bucket — it separates "the agent never found it" (a retrieval failure) from
    "the agent found it and the answer still did not score" (a reporting failure, OR
    a strict-metric artifact such as a gold span carrying a stray token).
    """
    try:
        _, score, _ = score_record(_example_from(d, task, subset), d.get("pred"))
    except Exception:
        score = 0.0
    if score == 1.0:
        return "correct"
    if d.get("pred") is None:
        return f"no answer ({d.get('end_reason') or '?'})"
    if 0.0 < score < 1.0:
        return f"partial credit (0<score<1)"
    golds = [g for g in (normalize_answer(str(a)) for a in (d.get("answers") or [])) if g]
    obs = normalize_answer(
        " ".join(t.get("observation") or "" for t in (d.get("transcript") or []))
    )
    if golds and any(g in obs for g in golds):
        return "SAW gold, reported wrong"
    return "never found gold"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", help="task stem, e.g. loft32k (matches {task}.rlm.*)")
    ap.add_argument("--root", default="results",
                    help="directory to search under (default: results)")
    ap.add_argument("--show", default=None,
                    help="print one full transcript from this bucket")
    args = ap.parse_args()

    pattern = os.path.join(args.root, "**", "transcripts",
                           f"{args.task}.rlm.*", "*.json")
    paths = glob.glob(pattern, recursive=True)
    if not paths:
        # transcripts/ may sit directly under --root (no nesting)
        paths = glob.glob(os.path.join(args.root, "transcripts",
                                       f"{args.task}.rlm.*", "*.json"))
    if not paths:
        sys.exit(f"no transcripts matched {pattern}\n"
                 f"try: find {args.root} -type d -name '{args.task}.rlm.*'")

    buckets: Counter = Counter()
    per_subset: dict[str, Counter] = {}
    examples: dict[str, str] = {}
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"[skip] {p}: {e}", file=sys.stderr)
            continue
        b = classify(d, args.task, Path(p).stem.rsplit("-", 1)[0])
        buckets[b] += 1
        # subset is not stored in the transcript; recover it from the id/filename
        subset = Path(p).stem.rsplit("-", 1)[0]
        per_subset.setdefault(subset, Counter())[b] += 1
        examples.setdefault(b, p)

    total = sum(buckets.values())
    print(f"{total} transcripts under {args.root}\n")
    for b, n in buckets.most_common():
        print(f"{n:5d}  {100*n/total:5.1f}%  {b}")
        print(f"                e.g. {examples[b]}")

    if len(per_subset) > 1:
        print("\nper subset:")
        names = [b for b, _ in buckets.most_common()]
        w = max(len(s) for s in per_subset)
        print(" " * (w + 2) + "  ".join(f"{n[:18]:>18}" for n in names))
        for s in sorted(per_subset):
            row = "  ".join(f"{per_subset[s][n]:>18}" for n in names)
            print(f"{s:<{w}}  {row}")

    if args.show:
        p = examples.get(args.show)
        if not p:
            sys.exit(f"\nno example in bucket {args.show!r}")
        d = json.load(open(p))
        print(f"\n{'='*70}\n{p}\nend_reason={d.get('end_reason')}")
        print(f"gold  : {d.get('answers')}")
        print(f"pred  : {d.get('pred')!r}")
        for t in d.get("transcript") or []:
            print(f"\n--- step {t.get('step')} ---")
            print("CODE:\n" + (t.get("code") or "<none>"))
            print("OBS :\n" + (t.get("observation") or "")[:1500])


if __name__ == "__main__":
    main()
