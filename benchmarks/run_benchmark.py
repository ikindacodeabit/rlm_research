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

from rlm.client import NIMClient
from rlm.rlm import RLM, MemoryBudget, vanilla_answer
from benchmarks.datasets import TASKS


def normalize(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def is_correct(pred: str | None, answers: list[str]) -> bool:
    if pred is None:
        return False
    p = normalize(pred)
    return any(normalize(a) in p for a in answers)


def recall(pred: str | None, answers: list[str]) -> float:
    """Partial-credit score in [0,1]: fraction of gold answers present in pred.

    For single-answer tasks this is just 0.0/1.0 (== is_correct). For RULER's
    multi-needle subsets (multivalue/multiquery/cwe/fwe) it gives partial credit,
    matching RULER's recall metric.
    """
    if pred is None or not answers:
        return 0.0
    p = normalize(pred)
    hits = sum(1 for a in answers if normalize(a) in p)
    return hits / len(answers)


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(l)["id"] for l in open(path) if l.strip()}


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
    ap.add_argument("--vanilla-char-limit", type=int, default=400_000)
    # --- RLM memory-budget knobs (no budget unless --max-context-tokens is set) ---
    # Eviction-only: out-of-budget turns are simply dropped (no notes/summarization).
    ap.add_argument("--max-context-tokens", type=int, default=None,
                    help="cap the RLM root's context window (tokens); unset = unbounded (legacy)")
    ap.add_argument("--keep-recent-turns", type=int, default=3,
                    help="recent (assistant,observation) pairs to keep verbatim under budget")
    ap.add_argument("--out", default="results")
    ap.add_argument("--debug", action="store_true",
                    help="print every RLM step (model reply, code, REPL output) live")
    ap.add_argument("--no-think", action="store_true",
                    help="disable Qwen3 thinking mode (chat_template_kwargs.enable_thinking=False)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["vanilla", "rlm"] if args.mode == "both" else [args.mode]

    root = NIMClient(model=args.root_model, base_url=args.base_url, rpm=args.rpm)
    sub = NIMClient(model=args.sub_model, base_url=args.base_url, rpm=args.rpm)
    if args.no_think:
        eb = {"chat_template_kwargs": {"enable_thinking": False}}
        root.extra_body = sub.extra_body = eb
    budget = None
    if args.max_context_tokens is not None:
        budget = MemoryBudget(
            max_context_tokens=args.max_context_tokens,
            keep_recent_turns=args.keep_recent_turns,
        )
    rlm = RLM(root_client=root, sub_client=sub, max_steps=args.max_steps, budget=budget)

    for mode in modes:
        slug = args.root_model.replace("/", "_")
        res_path = out_dir / f"{args.task}.{mode}.{slug}.jsonl"
        tdir = out_dir / "transcripts" / f"{args.task}.{mode}.{slug}"
        tdir.mkdir(parents=True, exist_ok=True)
        done = load_done(res_path)
        print(f"== {args.task} / {mode} -> {res_path} ({len(done)} already done)")

        n, correct = 0, 0
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
                try:
                    if mode == "vanilla":
                        pred = vanilla_answer(root, ex["context"], ex["question"],
                                              char_limit=args.vanilla_char_limit)
                        record.update(pred=pred, steps=1, finished=True, end_reason="")
                    else:
                        r = rlm.run(ex["context"], ex["question"])
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
                record["score"] = round(recall(record.get("pred"), ex["answers"]), 4)
                record["correct"] = record["score"] == 1.0
                record["latency_s"] = round(time.time() - t0, 2)
                record["tokens"] = root.usage.total_tokens + sub.usage.total_tokens - tok0
                record["sub_calls"] = sub.usage.calls - sub0
                record["ctx_chars"] = len(ex["context"])
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                n += 1
                correct += record["correct"]
                print(f"  [{ex['id']}] correct={record['correct']} "
                      f"steps={record['steps']} sub_calls={record['sub_calls']} "
                      f"end={record.get('end_reason','')} tokens={record['tokens']} "
                      f"t={record['latency_s']}s")
        if n:
            print(f"== {mode}: {correct}/{n} correct ({100*correct/n:.1f}%)")
    print(f"Total API calls: root={root.usage.calls}, sub={sub.usage.calls}; "
          f"tokens: {root.usage.total_tokens + sub.usage.total_tokens}")


if __name__ == "__main__":
    main()
