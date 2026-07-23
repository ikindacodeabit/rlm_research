"""Offline unit test for the code-exec watchdog (no GPU / no server).

Regression for the bug where the SIGALRM code-timeout wrapped the WHOLE code
block — including the blocking llm_query() sub-LLM call — so a legit-but-slow
sub-call (or several chunked ones) was aborted and mislabeled "infinite loop".

  A. a sub-call that takes LONGER than exec_timeout must NOT trip the timeout
     (only pure-Python time is bounded); the run completes normally.
  B. a genuine pure-Python infinite loop STILL trips the timeout (protection
     intact), and the harness recovers on the next turn.

Run:  python benchmarks/test_exec_timeout.py
"""
from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlm.rlm import RLM


def block(code: str) -> str:
    return f"```python\n{code}\n```"


class ScriptedClient:
    """Root: returns scripted replies in order (ignores the messages sent)."""

    def __init__(self, replies):
        self.replies = replies
        self.i = 0
        self.extra_body = None

    def chat(self, messages, **kw):
        reply = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return reply


class SleepyClient:
    """Sub: every chat() blocks for `delay` seconds, then returns a fixed answer."""

    def __init__(self, delay: float, answer: str):
        self.delay = delay
        self.answer = answer
        self.extra_body = None

    def chat(self, messages, **kw):
        time.sleep(self.delay)
        return self.answer


TIMEOUT = 0.3  # pure-Python budget per code block


def test_slow_subcall_not_timed_out():
    # The sub-call sleeps 2x the exec_timeout; with the fix the watchdog is paused
    # around it, so the block finishes and the answer is grounded in real output.
    root = ScriptedClient([
        block('ans = llm_query("please answer")\nprint("SUBANS", ans)'),
        block('FINAL(ans)'),
    ])
    sub = SleepyClient(delay=TIMEOUT * 2, answer="the answer is 42")
    rlm = RLM(root_client=root, sub_client=sub, max_steps=6, exec_timeout=TIMEOUT)
    res = rlm.run(context="irrelevant context", task="what is the answer?")

    joined = "\n".join(t["observation"] for t in res.transcript)
    assert "[TIMEOUT]" not in joined, f"slow sub-call was wrongly timed out:\n{joined}"
    assert res.finished, f"run did not finish: {res.end_reason}"
    assert res.answer == "the answer is 42", f"wrong answer: {res.answer!r}"
    assert res.metrics["sub_calls"] == 1, f"sub-call not made: {res.metrics}"
    print("A. slow sub-call NOT timed out; run finished with grounded answer")


def test_infinite_loop_times_out():
    # A real pure-Python infinite loop must still be aborted, and the harness must
    # recover on the next turn (recovered variable -> grounded FINAL).
    root = ScriptedClient([
        block('while True:\n    pass'),
        block('x = "recovered"\nprint(x)'),
        block('FINAL(x)'),
    ])
    rlm = RLM(root_client=root, sub_client=ScriptedClient(["unused"]),
              max_steps=6, exec_timeout=TIMEOUT)
    res = rlm.run(context="irrelevant", task="loop then recover")

    assert "[TIMEOUT]" in res.transcript[0]["observation"], \
        f"infinite loop was NOT aborted: {res.transcript[0]['observation']!r}"
    assert res.finished and res.answer == "recovered", \
        f"harness did not recover after timeout: {res.end_reason} {res.answer!r}"
    print("B. infinite loop timed out; harness recovered on the next turn")


def main():
    if not hasattr(signal, "SIGALRM"):
        print("SKIP: SIGALRM unavailable on this platform (watchdog disabled)")
        return
    test_slow_subcall_not_timed_out()
    test_infinite_loop_times_out()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
