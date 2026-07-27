"""Run vanilla vs RLM on a long-context task. Resumable JSONL checkpointing.

Usage:
  python benchmarks/run_benchmark.py --task niah --mode rlm --limit 1 --debug
  python benchmarks/run_benchmark.py --task longbench_v2 --mode both --limit 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlm.client import NIMClient, RateLimiter
from rlm.rlm import RLM, MemoryBudget, Scratchpad, vanilla_answer
from benchmarks.answer_extraction import extract_answers, strip_reasoning
from benchmarks.datasets import TASKS
from benchmarks.longbench_metrics import (
    EXACT_MATCH_METRICS,
    LOFT_METRICS,
    normalize_answer,
    score_example_all,
)


def recall(pred: str | None, answers: list[str]) -> float:
    """Partial-credit score in [0,1]: fraction of gold answers present in pred.

    For single-answer tasks this is just 0.0/1.0 (a plain substring hit). For
    RULER's multi-needle subsets (multivalue/multiquery/cwe/fwe) it gives partial
    credit, matching RULER's recall metric.

    In-house metric: RULER and the synthetic niah/multikey tasks have no upstream
    per-example scorer to mirror, unlike LongBench and LOFT. It now shares
    longbench_metrics.normalize_answer with every other metric — it used to carry
    its own weaker normaliser (lower+collapse only, no punctuation/article
    stripping), which made "correct" mean three different things across the repo.
    """
    if pred is None or not answers:
        return 0.0
    p = normalize_answer(str(pred))
    hits = sum(1 for a in answers if normalize_answer(str(a)) in p)
    return hits / len(answers)


# Metrics whose gold is a LABEL, not a span quoted from the document: a letter
# (choice), a bare number (count), a class name (classification). Containment finds
# those in any text, so "the gold survived truncation" is trivially true and tells
# you nothing -- every LongBench-v2 `choice` cell reported seen% 100.0 while the
# vanilla arm was retaining only 15-22% of the context.
_LABEL_METRICS = {"choice", "count", "classification"}

# Above this many occurrences in the full context, a gold string is ambient rather
# than located, so its presence in the retained prefix is not evidence of anything.
_MAX_GOLD_OCCURRENCES = 10


def _gold_visible(ex: dict, stats: dict) -> bool | None:
    """Did the gold survive the vanilla arm's truncation?

    None means "the question is not meaningful here", which happens two ways:

    * The answer is DERIVED, not quoted -- a summary, or `multikey`'s sum of eight
      planted values -- so it appears at no length. False would read as a truncation
      failure when the string was never present to begin with.
    * The answer is a LABEL or otherwise ambient in the text, so containment is
      trivially satisfied and True would overstate what vanilla could see.

    In both cases the retained-context fraction is the honest signal instead.
    """
    used = stats.get("context_chars_used")
    if used is None:
        return None
    if (ex.get("metric") or "") in _LABEL_METRICS:
        return None
    full = ex.get("context") or ""
    # Distinctive = present, and present in only a few places. A 7-digit passkey
    # occurs once; "B" occurs on every page.
    present = [g for g in (str(a) for a in (ex.get("answers") or [])) if g
               and 0 < full.count(g) <= _MAX_GOLD_OCCURRENCES]
    if not present:
        return None
    return any(g in full[:used] for g in present)


def score_record(ex: dict, pred: str | None) -> tuple[str, float, dict]:
    """Score one prediction. IDENTICAL for the vanilla and RLM arms.

    This is the single scoring site in the harness: it takes no `mode` argument, so
    the two arms cannot diverge here.

    What is and is NOT normalised, and why:
      * Reasoning blocks are stripped for every metric — a <think> trace is never
        part of any benchmark's intended answer.
      * ANSWER EXTRACTION runs only for LOFT, because parsing the model's output
        into an answer list is part of LOFT's own official evaluation
        (google-deepmind/loft). LongBench v1/v2 and RULER define their metrics over
        the RAW generation, so extracting there would diverge from the published
        numbers.

    That leaves the arms free to differ in answer STYLE on the raw-text metrics, so
    parity for those is enforced on the GENERATION side instead: every loader
    supplies an `answer_format` that both arms receive (see benchmarks/datasets.py,
    rlm.vanilla_answer and ROOT_SYSTEM_PROMPT). Fidelity to the official metric and
    fairness between arms are thus handled in the two places each belongs.

    Returns (metric_name, primary_score, secondary_scores).
    """
    metric = ex.get("metric")
    if not metric:
        raise KeyError(
            f"example {ex.get('id')!r} has no `metric`; every loader in "
            "benchmarks/datasets.py must set one explicitly"
        )
    if pred is None:
        return metric, 0.0, {}

    clean = strip_reasoning(pred)
    if metric == "recall":
        return metric, recall(clean, ex["answers"]), {}

    # LOFT only — see the docstring. Parsing the output into an answer list is part
    # of LOFT's own official evaluation; LongBench/RULER score the raw generation.
    pred_answers = None
    if metric in LOFT_METRICS:
        pred_answers = extract_answers(
            clean,
            answer_prefix=ex.get("answer_prefix"),
            expect_list=bool(ex.get("multi_value")),
        )
    primary, secondary = score_example_all(
        metric, clean, ex["answers"],
        dataset=ex.get("subset"),
        all_classes=ex.get("all_classes"),
        pred_answers=pred_answers,
        multi_value=bool(ex.get("multi_value")),
    )
    return metric, primary, secondary


def load_done(path: Path) -> set:
    """IDs already completed. Records that died with a harness/API exception
    (e.g. a transient NIM outage) don't count — re-running retries them."""
    if not path.exists():
        return set()
    done = set()
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        if not r.get("error"):
            done.add(r["id"])
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--mode", default="both", choices=["vanilla", "rlm", "both"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--root-model", default="meta/llama-3.3-70b-instruct",
                    help="NIM model id for vanilla baseline AND the RLM root")
    ap.add_argument("--sub-model", default="meta/llama-3.1-8b-instruct",
                    help="NIM model id for recursive sub-calls (cheap is fine)")
    ap.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    ap.add_argument("--rpm", type=int, default=35)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--exec-timeout", type=float, default=60.0,
                    help="wall-clock seconds of PURE-PYTHON execution a single RLM code block "
                         "may run before being aborted (guards against model-generated infinite "
                         "loops); time spent inside llm_query sub-calls is excluded")
    ap.add_argument("--run-timeout", type=float, default=900.0,
                    help="wall-clock seconds for ONE example before the RLM gives up "
                         "(end_reason=run_timeout). exec-timeout does not bound "
                         "llm_query time, so without this a single pathological "
                         "example can stall a job for hours. 0 disables")
    ap.add_argument("--max-sub-calls", type=int, default=40,
                    help="cap on llm_query calls per example; further calls return a "
                         "notice instead of hitting the API. 0 disables")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="max generation tokens per model call (root and sub). Unset uses the "
                         "client default (4096); raise it for Qwen3 'think' runs whose "
                         "<think> reasoning shares this budget with the visible answer")
    ap.add_argument("--vanilla-char-limit", type=int, default=400_000)
    ap.add_argument("--vanilla-max-prompt-tokens", type=int, default=None,
                    help="hard TOKEN ceiling for the vanilla prompt; the char limit "
                         "alone overflows the served window on densely-tokenising "
                         "subsets (RULER cwe / niah_multikey_3), which the server "
                         "rejects with a 400. Set to (max-model-len - max-tokens - "
                         "margin), e.g. 34000 for a 40960 window")
    # --- RLM memory-budget knobs (no budget unless --max-context-tokens is set) ---
    # Eviction-only: out-of-budget turns are simply dropped (no notes/summarization).
    ap.add_argument("--max-context-tokens", type=int, default=None,
                    help="cap the RLM root's context window (tokens); unset = unbounded (legacy)")
    ap.add_argument("--keep-recent-turns", type=int, default=3,
                    help="recent (assistant,observation) pairs to keep verbatim under budget")
    # --- RLM scratchpad knobs (off unless --scratchpad; orthogonal to the budget) ---
    ap.add_argument("--scratchpad", action="store_true",
                    help="give the RLM root a note() tool; notes are re-shown every turn "
                         "and survive budget eviction")
    ap.add_argument("--max-notes-tokens", type=int, default=1024,
                    help="cap on the scratchpad block injected into context")
    ap.add_argument("--out", default="results")
    ap.add_argument("--debug", action="store_true",
                    help="print every RLM step (model reply, code, REPL output) live")
    ap.add_argument("--no-think", action="store_true",
                    help="disable Qwen3 thinking mode (chat_template_kwargs.enable_thinking=False)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["vanilla", "rlm"] if args.mode == "both" else [args.mode]

    # One limiter shared by root + sub so the rpm cap is per-account (two separate
    # limiters would let the process issue up to ~2x rpm and trip 429s).
    limiter = RateLimiter(args.rpm)
    client_kw = dict(base_url=args.base_url, rpm=args.rpm, limiter=limiter)
    if args.max_tokens is not None:
        client_kw["max_tokens"] = args.max_tokens
    root = NIMClient(model=args.root_model, **client_kw)
    sub = NIMClient(model=args.sub_model, **client_kw)
    if args.no_think:
        eb = {"chat_template_kwargs": {"enable_thinking": False}}
        root.extra_body = sub.extra_body = eb
    budget = None
    if args.max_context_tokens is not None:
        budget = MemoryBudget(
            max_context_tokens=args.max_context_tokens,
            keep_recent_turns=args.keep_recent_turns,
        )
    scratchpad = None
    if args.scratchpad:
        scratchpad = Scratchpad(max_notes_tokens=args.max_notes_tokens)
    rlm = RLM(root_client=root, sub_client=sub, max_steps=args.max_steps, budget=budget,
              scratchpad=scratchpad, exec_timeout=args.exec_timeout,
              run_timeout=args.run_timeout or None,
              max_sub_calls=args.max_sub_calls or None)

    for mode in modes:
        slug = args.root_model.replace("/", "_")
        res_path = out_dir / f"{args.task}.{mode}.{slug}.jsonl"
        tdir = out_dir / "transcripts" / f"{args.task}.{mode}.{slug}"
        tdir.mkdir(parents=True, exist_ok=True)
        done = load_done(res_path)
        print(f"== {args.task} / {mode} -> {res_path} ({len(done)} already done)")

        n, correct, score_sum = 0, 0, 0.0
        with open(res_path, "a") as fout:
            for ex in TASKS[args.task](args.limit):
                if ex["id"] in done:
                    continue
                t0 = time.time()
                tok0 = root.usage.total_tokens + sub.usage.total_tokens
                sub0 = sub.usage.calls
                record = {"id": ex["id"], "mode": mode, "answers": ex["answers"]}
                if ex.get("subset") is not None:
                    record["subset"] = ex["subset"]
                # Persist everything the scorer consumed, so scripts/rescore.py can
                # reproduce the score from the record alone. `all_classes` (trec) and
                # `multi_value` (LOFT) were previously loader-only, which made a
                # re-score of those subsets silently wrong.
                for k in ("all_classes", "multi_value", "answer_prefix"):
                    if ex.get(k) is not None:
                        record[k] = ex[k]
                try:
                    # Same output contract to both arms (None for tasks that carry
                    # their format instruction inside the question text already).
                    answer_format = ex.get("answer_format")
                    if mode == "vanilla":
                        vstats: dict = {}
                        pred = vanilla_answer(
                            root, ex["context"], ex["question"],
                            char_limit=args.vanilla_char_limit,
                            answer_format=answer_format,
                            max_prompt_tokens=args.vanilla_max_prompt_tokens,
                            stats=vstats)
                        record.update(pred=pred, steps=1, finished=True, end_reason="")
                        # Was the gold even inside the prompt we sent? Without this a
                        # truncated vanilla arm's score is indistinguishable from a
                        # model failure -- niah vanilla scored 47.1%, which is EXACTLY
                        # the 153/325 rate at which the needle survived the 100k cut.
                        record.update(vstats)
                        record["gold_visible"] = _gold_visible(ex, vstats)
                    else:
                        r = rlm.run(ex["context"], ex["question"],
                                    answer_format=answer_format)
                        record.update(pred=r.answer, steps=r.steps,
                                      finished=r.finished, end_reason=r.end_reason,
                                      metrics=r.metrics)
                        # Always save the full transcript for post-mortems:
                        with open(tdir / f"{ex['id']}.json", "w") as tf:
                            json.dump({"question": ex["question"],
                                       "answers": ex["answers"],
                                       "pred": r.answer,
                                       "end_reason": r.end_reason,
                                       "metrics": r.metrics,
                                       "transcript": r.transcript}, tf, indent=2)
                        if args.debug:
                            for t in r.transcript:
                                print(f"\n--- step {t['step']} ---")
                                print("MODEL REPLY:\n" + (t.get("reply") or "")[:2000])
                                print("EXECUTED CODE:\n" + (t.get("code") or "<none>"))
                                print("REPL OUTPUT:\n" + t["observation"][:2000])
                            print(f"\nEND: reason={r.end_reason} pred={r.answer!r}\n")
                except Exception as e:
                    record.update(pred=None, error=f"{type(e).__name__}: {e}",
                                  steps=0, finished=False, end_reason="exception")
                # One scoring site for BOTH arms — see score_record(). Every loader
                # sets `metric` explicitly; there is no silent fallback any more.
                metric, score_val, secondary = score_record(ex, record.get("pred"))
                record["metric"] = metric
                record["score"] = round(score_val, 4)
                if secondary:
                    record["secondary_scores"] = {k: round(v, 4)
                                                  for k, v in secondary.items()}
                # `correct` only means something for exact-match-family metrics; an
                # F1/ROUGE of 1.0 is a continuous score that saturated, not an exact
                # hit, and reporting it as "exact-correct" overstates the result.
                record["correct"] = (record["score"] == 1.0
                                     if metric in EXACT_MATCH_METRICS else None)
                record["latency_s"] = round(time.time() - t0, 2)
                record["tokens"] = root.usage.total_tokens + sub.usage.total_tokens - tok0
                record["sub_calls"] = sub.usage.calls - sub0
                record["ctx_chars"] = len(ex["context"])
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                n += 1
                correct += bool(record["correct"])
                score_sum += record["score"]
                print(f"  [{ex['id']}] correct={record['correct']} "
                      f"score={record['score']} "
                      f"steps={record['steps']} sub_calls={record['sub_calls']} "
                      f"end={record.get('end_reason','')} tokens={record['tokens']} "
                      f"t={record['latency_s']}s")
        if n:
            # exact-correct is only meaningful for exact-match tasks (niah/ruler);
            # mean score is the honest number for F1/ROUGE/choice (LongBench).
            print(f"== {mode}: {correct}/{n} exact-correct, mean score {100*score_sum/n:.1f}%")
    print(f"Total API calls: root={root.usage.calls}, sub={sub.usage.calls}; "
          f"tokens: {root.usage.total_tokens + sub.usage.total_tokens}")


if __name__ == "__main__":
    main()
