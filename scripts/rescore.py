#!/usr/bin/env python3
"""Re-score existing results under the current scoring standard, from stored preds.

Every record already stores `pred`, so changing a metric never requires re-running
a benchmark — this recomputes `score`, `metric`, `secondary_scores` and `correct`
in place. No GPU, no model calls.

What moves, and why:
  * LOFT multi-value (qampari/quest) — the primary metric changed from `coverage`
    to LOFT's official all-or-nothing subspan-EM.
  * LOFT single-value — answers now go through the shared answer-extraction layer
    before scoring, as upstream does.
  * `recall` tasks (niah/multikey/ruler) — `recall` now shares the repo-wide
    normalize_answer, which additionally strips punctuation and articles. Expect
    small upward movement only where punctuation blocked a match.
  * LongBench v1/v2 — expected UNCHANGED except for NFD-normalisation edge cases
    on accented text. Any other movement here is a bug in the standardisation.

Usage:
  python scripts/rescore.py results --dry-run     # report deltas, write nothing
  python scripts/rescore.py results               # rewrite in place (.bak kept)
  python scripts/rescore.py results/loft32k_nothink
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.longbench_metrics import EXACT_MATCH_METRICS
from benchmarks.run_benchmark import score_record

# LOFT records written before this change lack the fields the new scorer needs.
# Both are reconstructible: the prefix is a constant in the upstream data, and the
# single/multi-value split is a property of the dataset, not of the row.
LOFT_MULTIVALUE = {"qampari", "quest"}
LOFT_ANSWER_PREFIX = "Final Answer:"


def backfill(rec: dict) -> dict:
    """Rebuild the loader-side fields score_record needs from a stored record."""
    ex = {
        "id": rec.get("id"),
        "answers": rec.get("answers") or [],
        "subset": rec.get("subset"),
        "metric": rec.get("metric"),
        "all_classes": rec.get("all_classes"),
    }
    # Pre-standard LOFT rows carry metric "loft_coverage" or "loft_subspan_em";
    # both become the official primary. multi_value/answer_prefix are properties of
    # the dataset, so they are reconstructible even for rows that predate storing
    # them (run_benchmark now persists both).
    if (ex["metric"] or "").startswith("loft"):
        ex["metric"] = "loft_subspan_em"
        subset = ex.get("subset") or ""
        ex["multi_value"] = rec.get(
            "multi_value", subset.rsplit("_", 1)[0] in LOFT_MULTIVALUE)
        ex["answer_prefix"] = rec.get("answer_prefix", LOFT_ANSWER_PREFIX)
    return ex


def unscoreable(ex: dict) -> str | None:
    """Why this record cannot be faithfully re-scored, or None if it can.

    `classification` (trec) needs the loader's `all_classes` list, which older
    records do not carry — scoring without it silently yields 0.0 for every row.
    Refuse rather than corrupt; re-run the task to pick up the field.
    """
    if ex.get("metric") == "classification" and not ex.get("all_classes"):
        return "classification record has no stored `all_classes`"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="results")
    ap.add_argument("--dry-run", action="store_true",
                    help="report deltas without writing anything")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the .bak copy (not recommended)")
    args = ap.parse_args()

    paths = sorted(Path(args.results_dir).rglob("*.jsonl"))
    if not paths:
        sys.exit(f"no *.jsonl under {args.results_dir}")

    # (task, variant, mode, subset) -> [old_sum, new_sum, n, old_metric, new_metric]
    cells: dict[tuple, list] = defaultdict(
        lambda: [0.0, 0.0, 0, set(), set()])
    n_changed = 0
    skipped: dict[str, int] = defaultdict(int)
    root = Path(args.results_dir)

    for path in paths:
        variant = path.parent.name if path.parent != root else "root"
        try:
            task, mode, _model = path.stem.split(".", 2)
        except ValueError:
            print(f"[skip] {path}: unexpected filename", file=sys.stderr)
            continue

        out_lines, changed_here = [], 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            old_score = rec.get("score")
            old_metric = rec.get("metric")
            ex = backfill(rec)
            if not ex["metric"]:
                out_lines.append(line)
                continue
            why = unscoreable(ex)
            if why:
                skipped[why] += 1
                out_lines.append(line)
                continue
            try:
                metric, score, secondary = score_record(ex, rec.get("pred"))
            except Exception as e:
                print(f"[skip] {path}:{rec.get('id')}: {e}", file=sys.stderr)
                out_lines.append(line)
                continue

            rec["metric"] = metric
            rec["score"] = round(score, 4)
            if secondary:
                rec["secondary_scores"] = {k: round(v, 4)
                                           for k, v in secondary.items()}
            rec["correct"] = (rec["score"] == 1.0
                              if metric in EXACT_MATCH_METRICS else None)

            c = cells[(task, variant, mode, rec.get("subset") or "")]
            c[0] += float(old_score) if old_score is not None else 0.0
            c[1] += rec["score"]
            c[2] += 1
            c[3].add(old_metric or "?")
            c[4].add(metric)
            if old_score is None or abs(float(old_score) - rec["score"]) > 1e-9:
                changed_here += 1
            out_lines.append(json.dumps(rec))

        n_changed += changed_here
        if not args.dry_run and changed_here:
            if not args.no_backup:
                shutil.copy2(path, path.with_suffix(".jsonl.bak"))
            path.write_text("\n".join(out_lines) + "\n")

    hdr = (f"{'task':<14}{'variant':<20}{'mode':<9}{'subset':<22}"
           f"{'metric old->new':<30}{'n':>5}{'old%':>8}{'new%':>8}{'delta':>8}")
    print(hdr)
    print("-" * len(hdr))
    for (task, variant, mode, subset), c in sorted(cells.items()):
        old_sum, new_sum, n, old_m, new_m = c
        if not n:
            continue
        old_pct, new_pct = 100 * old_sum / n, 100 * new_sum / n
        om, nm = "/".join(sorted(old_m)), "/".join(sorted(new_m))
        change = f"{om} -> {nm}" if om != nm else om
        flag = "" if abs(new_pct - old_pct) < 0.05 else "  <-- moved"
        print(f"{task:<14}{variant:<20}{mode:<9}{subset:<22}{change:<30}"
              f"{n:>5}{old_pct:>8.1f}{new_pct:>8.1f}{new_pct - old_pct:>+8.1f}{flag}")

    for why, n in sorted(skipped.items()):
        print(f"\nSKIPPED {n} record(s): {why}")
    print(f"\n{n_changed} record(s) changed score across {len(paths)} file(s)")
    if args.dry_run:
        print("dry run — nothing written")
    elif n_changed:
        print("originals kept alongside as *.jsonl.bak")


if __name__ == "__main__":
    main()
