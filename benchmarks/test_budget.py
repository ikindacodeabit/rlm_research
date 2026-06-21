"""Offline unit test for the eviction-only MemoryBudget (no GPU / no server).

Drives the RLM with a scripted stub client through enough turns to force
eviction, then checks:
  * eviction actually fires under a budget, and bounds context growth
    (budgeted peak context << unbounded peak context for the same script);
  * the scratchpad is gone: metrics has `evictions` but not `notes_saved`,
    and calling the removed `note(...)` tool raises inside the REPL;
  * grounding still works (FINAL on a variable seen in real output succeeds).

Run:  python benchmarks/test_budget.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlm.rlm import RLM, MemoryBudget


class StubClient:
    """Returns scripted replies in order; ignores the messages it's sent."""

    def __init__(self, replies):
        self.replies = replies
        self.i = 0
        self.extra_body = None

    def chat(self, messages, **kw):
        reply = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return reply


def block(code: str) -> str:
    return f"```python\n{code}\n```"


# A scripted session: read three big chunks (grows history), try the removed
# note() tool, surface the answer in real output, then FINAL on the variable.
REPLIES = [
    block('print("CHUNK1 " + context[0:1200])'),
    block('print("CHUNK2 " + context[1200:2400])'),
    block('print("CHUNK3 " + context[2400:3600])'),
    block('note("this tool should no longer exist")'),
    block('answer = context[:13]; print("FOUND", answer)'),
    block('FINAL(answer)'),
]

CONTEXT = "PASSKEY-12345 " + ("filler text blah " * 2000)
TASK = "Return the passkey at the start of the document."


def run(budget):
    rlm = RLM(root_client=StubClient(list(REPLIES)),
              sub_client=StubClient(["unused"]),
              max_steps=10, budget=budget)
    return rlm.run(CONTEXT, TASK)


def main():
    budgeted = run(MemoryBudget(max_context_tokens=1024, keep_recent_turns=1))
    unbounded = run(None)

    m = budgeted.metrics

    # 1. eviction fired and is recorded under the new metric name
    assert m["evictions"] > 0, f"expected evictions > 0, got {m}"
    assert m["budget"] == 1024, f"budget metric wrong: {m}"

    # 2. scratchpad machinery is gone
    assert "notes_saved" not in m, f"notes_saved should be gone: {m}"
    assert "strategy" not in m, f"strategy should be gone: {m}"
    note_obs = next((t["observation"] for t in budgeted.transcript
                     if t["code"] and "note(" in t["code"]), "")
    assert "EXCEPTION" in note_obs or "NameError" in note_obs, \
        f"removed note() tool should error in REPL, got: {note_obs!r}"

    # 3. budget bounds context growth vs the same unbounded script
    assert m["peak_context_tokens"] < unbounded.metrics["peak_context_tokens"], (
        f"budgeted peak {m['peak_context_tokens']} should be < unbounded "
        f"{unbounded.metrics['peak_context_tokens']}")

    # 4. grounding still works: FINAL on a real variable returns the answer
    assert budgeted.finished, f"run did not finish: {budgeted.end_reason}"
    assert budgeted.answer == "PASSKEY-12345", f"wrong answer: {budgeted.answer!r}"

    print("budgeted metrics:", {k: m[k] for k in
          ("evictions", "peak_context_tokens", "budget", "steps")})
    print("unbounded peak_context_tokens:",
          unbounded.metrics["peak_context_tokens"])
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
