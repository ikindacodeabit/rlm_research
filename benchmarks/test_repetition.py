"""Offline unit test for the repetition breaker (no GPU / no server).

Regression for the failure seen on LongBench short-context subsets (trec,
triviaqa) with Qwen3-8B under --no-think: the root re-emits the SAME code cell
every turn — identical code -> identical observation — and burns all max_steps
without ever calling FINAL (end_reason=max_steps, score 0).

  A. three consecutive IDENTICAL code cells trigger a forced GROUNDED fallback:
     the run ends early with end_reason="repetition_broken" and a real answer
     (from the sub-LLM over accumulated output), NOT max_steps.
  B. a model that VARIES its code is never touched by the breaker and finishes
     normally via FINAL.

Run:  python benchmarks/test_repetition.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlm.rlm import RLM


def block(code: str) -> str:
    return f"```python\n{code}\n```"


class ScriptedClient:
    """Returns scripted replies in order; repeats the last one forever."""

    def __init__(self, replies):
        self.replies = replies
        self.i = 0
        self.calls = 0
        self.extra_body = None

    def chat(self, messages, **kw):
        self.calls += 1
        reply = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return reply


def test_identical_code_triggers_fallback():
    # The root keeps running the exact same (useless) search. Steps: 1 records it,
    # 2 nudges, 3 fires the grounded fallback -> answer from the sub client.
    root = ScriptedClient([block('print(context.find("zzz"))')])  # -1 forever
    sub = ScriptedClient(["FALLBACK ANSWER"])
    rlm = RLM(root_client=root, sub_client=sub, max_steps=20)
    res = rlm.run(context="some document text", task="what is the type?")

    assert res.end_reason == "repetition_broken", f"breaker did not fire: {res.end_reason}"
    assert res.finished, "run should finish with a fallback answer"
    assert res.answer == "FALLBACK ANSWER", f"wrong fallback answer: {res.answer!r}"
    assert res.steps == 3, f"should break on the 3rd identical cell, got step {res.steps}"
    assert sub.calls == 1, f"fallback should make exactly one sub-call: {sub.calls}"
    print("A. 3 identical code cells -> grounded fallback (no max_steps waste)")


def test_varying_code_finishes_normally():
    # Distinct code each turn -> breaker never fires; normal FINAL path.
    root = ScriptedClient([
        block('a = context.find("hel")\nprint(a)'),
        block('b = context[a:a+5]\nprint(b)'),
        block('FINAL(b)'),
    ])
    rlm = RLM(root_client=root, sub_client=ScriptedClient(["unused"]), max_steps=20)
    res = rlm.run(context="xxhello world", task="find the greeting")

    assert res.end_reason == "final_called", f"expected normal finish: {res.end_reason}"
    assert res.answer == "hello", f"wrong answer: {res.answer!r}"
    assert res.steps == 3, f"should finish in 3 varied steps, got {res.steps}"
    print("B. varied code finishes normally; breaker leaves it alone")


def main():
    test_identical_code_triggers_fallback()
    test_varying_code_finishes_normally()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
