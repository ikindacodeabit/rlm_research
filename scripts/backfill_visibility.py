#!/usr/bin/env python3
"""Backfill the vanilla arm's truncation exposure onto results written before it.

WHY this exists. `niah` vanilla scored 47.1% and `multikey` vanilla scored 0.0%.
Neither number is a model result:

    niah    : gold inside the 100k window for 153/325 = 47.1%   (score: 47.1%)
    multikey: all 8 planted values inside the window  0/325 =  0.0%   (score: 0.0%)

Both generators build 200,000-char contexts while `--vanilla-char-limit` is 100,000,
so the vanilla arm was answering correctly on essentially every example it could
see. Reporting "RLM 88.0 vs vanilla 0.0" without that ceiling attributes a truncation
artefact to the model.

run_benchmark.py now records `gold_visible` / `context_chars*` at run time. This
recovers the same fields for records already on disk, which is possible without a
GPU because the synthetic generators are seeded and therefore reproducible, and the
file-backed tasks reload from $RLM_DATA_DIR.

Usage:
  python scripts/backfill_visibility.py results --dry-run
  python scripts/backfill_visibility.py results
  python scripts/backfill_visibility.py archive/bad_stacked_format --char-limit 100000
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.datasets import TASKS
from benchmarks.run_benchmark import _gold_visible

_CACHE: dict[str, dict] = {}


def examples_for(task: str, want: int) -> dict:
    """id -> example for `task`, covering at least `want` examples.

    `want` matters: the synthetic generators take a limit and DEFAULT TO 50, while
    the campaign ran them at 325. Loading with limit=None would silently cover only
    ids 0..49 and leave 275 records per cell un-backfilled.
    """
    cached = _CACHE.get(task)
    if cached is not None and len(cached) >= want:
        return cached
    loader = TASKS.get(task)
    if loader is None:
        _CACHE[task] = {}
        return {}
    try:
        by_id = {ex["id"]: ex for ex in loader(want)}
    except Exception as e:  # missing $RLM_DATA_DIR file, offline HF, etc.
        print(f"  ! cannot load task {task!r}: {type(e).__name__}: {e}")
        by_id = {}
    _CACHE[task] = by_id
    return by_id


def _needed(records: list[dict]) -> int:
    """How many examples the loader must emit to cover every id in the file.

    Synthetic ids end in their generation index (`niah-200000-317`), and the loader
    emits 0..n-1, so the highest index seen sets the count. Falls back to the record
    count for file-backed tasks, where the limit is per-subset anyway.
    """
    hi = 0
    for r in records:
        tail = str(r.get("id", "")).rsplit("-", 1)[-1]
        if tail.isdigit():
            hi = max(hi, int(tail) + 1)
    return max(hi, len(records))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="results")
    ap.add_argument("--char-limit", type=int, default=100_000,
                    help="the --vanilla-char-limit the run used (default 100000)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    totals = defaultdict(int)
    for path in sorted(Path(args.results_dir).rglob("*.jsonl")):
        task, mode, _model = path.stem.split(".", 2)
        if mode != "vanilla":
            continue  # the RLM reads the whole context; the ceiling is vanilla-only
        raw, parsed = [], []
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A job killed mid-write leaves a partial last line. Keep it
                # verbatim rather than dropping a record.
                raw.append(line)
                continue
            raw.append(rec)
            parsed.append(rec)

        by_id = examples_for(task, _needed(parsed))
        if not by_id:
            continue

        out, changed, seen, seen_n = [], 0, 0, 0
        for rec in raw:
            if isinstance(rec, str):
                out.append(rec)
                continue
            ex = by_id.get(rec.get("id"))
            if ex is not None and "gold_visible" not in rec:
                full = len(ex.get("context") or "")
                used = min(full, args.char_limit)
                # NB: assumes no shrink retry fired. True for the synthetics (prose
                # filler tokenizes near 4 chars/token, so 100k chars fits 40960) but
                # NOT for dense subsets -- those errored outright and have no score
                # to explain anyway.
                rec.update(context_chars=full, context_chars_used=used,
                           truncated=used < full, shrink_retries=0)
                rec["gold_visible"] = _gold_visible(ex, rec)
                changed += 1
            if rec.get("gold_visible") is not None:
                seen_n += 1
                seen += bool(rec["gold_visible"])
            out.append(json.dumps(rec))

        if not changed:
            continue
        ceiling = f"{100*seen/seen_n:.1f}%" if seen_n else "n/a (derived answer)"
        print(f"{path}: +{changed} records, vanilla ceiling {ceiling}")
        totals["files"] += 1
        totals["records"] += changed
        if not args.dry_run:
            bak = path.with_suffix(path.suffix + ".bak")
            if not bak.exists():  # write-once: never clobber the pristine original
                shutil.copy2(path, bak)
            path.write_text("\n".join(out) + "\n")

    verb = "would update" if args.dry_run else "updated"
    print(f"\n{verb} {totals['records']} records across {totals['files']} files")
    if args.dry_run:
        print("re-run without --dry-run to write")
    else:
        print("now: python benchmarks/score.py " + args.results_dir)


if __name__ == "__main__":
    main()
