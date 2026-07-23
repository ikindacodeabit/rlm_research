"""Offline unit test for the opt-in Scratchpad (no GPU / no server).

Drives the RLM with a scripted stub client and checks, in three configurations:

  A. scratchpad ONLY (no budget):
       * note() works and is counted in metrics["notes_saved"];
       * saved notes are re-injected into the root's view on later turns
         (the bug that killed the old scratchpad: notes were only injected
         under a budget, so unbounded runs never saw them);
       * no eviction happens; grounded FINAL still works.

  B. scratchpad + MemoryBudget:
       * eviction fires (dropped turns' raw output leaves the sent view);
       * the note SURVIVES eviction — it is still in the last sent view even
         though the turn that produced it was evicted;
       * grounded FINAL on a variable still works.

  C. default (neither): note() does not exist in the REPL and metrics carry
     no scratchpad keys — the legacy eviction-only path is untouched.

Run:  python benchmarks/test_scratchpad.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlm.rlm import RLM, MemoryBudget, Scratchpad


class RecordingStubClient:
    """Returns scripted replies in order; records every messages list it is sent."""

    def __init__(self, replies):
        self.replies = replies
        self.i = 0
        self.extra_body = None
        self.sent = []  # one messages-list per chat() call

    def chat(self, messages, **kw):
        self.sent.append([dict(m) for m in messages])
        reply = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return reply


def block(code: str) -> str:
    return f"```python\n{code}\n```"


NOTE_TEXT = "passkey candidate = PASSKEY-12345"

# Save a note early, then churn through big chunks so (under a budget) the
# note-taking turn is evicted, then FINAL on a variable seen in real output.
REPLIES = [
    block(f'answer = context[:13]\nprint(note("{NOTE_TEXT}"), "FOUND", answer)'),
    block('print("CHUNK1 " + context[0:1200])'),
    block('print("CHUNK2 " + context[1200:2400])'),
    block('print("CHUNK3 " + context[2400:3600])'),
    block('print("RECALL", answer)'),
    block('FINAL(answer)'),
]

CONTEXT = "PASSKEY-12345 " + ("filler text blah " * 2000)
TASK = "Return the passkey at the start of the document."


def run(budget, scratchpad, replies=REPLIES):
    root = RecordingStubClient(list(replies))
    rlm = RLM(root_client=root, sub_client=RecordingStubClient(["unused"]),
              max_steps=10, budget=budget, scratchpad=scratchpad)
    return rlm.run(CONTEXT, TASK), root


def texts(messages) -> str:
    return "\n".join(m["content"] for m in messages)


def test_scratchpad_only():
    res, root = run(budget=None, scratchpad=Scratchpad())
    m = res.metrics

    assert m.get("scratchpad") is True, f"scratchpad flag missing: {m}"
    assert m["notes_saved"] == 1, f"expected 1 note saved, got {m}"
    assert m["evictions"] == 0, f"no budget -> no evictions, got {m}"

    # note() confirmed inside the REPL observation
    obs1 = res.transcript[0]["observation"]
    assert "[saved note #1]" in obs1, f"note() confirmation missing: {obs1!r}"

    # the note is visible in every request AFTER it was saved (call 0 saves it)
    for i, msgs in enumerate(root.sent[1:], start=1):
        assert NOTE_TEXT in texts(msgs), f"note missing from request {i}"
    assert NOTE_TEXT not in texts(root.sent[0]), "note leaked into first request"
    # and the system prompt advertises the tool
    assert "note(text" in root.sent[0][0]["content"]

    assert res.finished and res.answer == "PASSKEY-12345", \
        f"bad result: {res.end_reason} {res.answer!r}"
    print("A. scratchpad-only:", {k: m[k] for k in
          ("notes_saved", "evictions", "peak_context_tokens", "steps")})


def test_scratchpad_with_budget():
    res, root = run(budget=MemoryBudget(max_context_tokens=1024, keep_recent_turns=1),
                    scratchpad=Scratchpad())
    m = res.metrics

    assert m["evictions"] > 0, f"expected evictions under 1024-token budget: {m}"
    assert m["notes_saved"] == 1, f"expected 1 note saved, got {m}"

    last = texts(root.sent[-1])
    # the turn that saved the note (and printed CHUNK output) was evicted...
    assert "CHUNK1" not in last, "old raw output should have been evicted"
    # ...but the note survived into the final view
    assert NOTE_TEXT in last, "note should survive eviction"
    assert "[MEMORY NOTICE]" in last and "SCRATCHPAD" in last.upper()

    assert res.finished and res.answer == "PASSKEY-12345", \
        f"bad result: {res.end_reason} {res.answer!r}"
    print("B. scratchpad+budget:", {k: m[k] for k in
          ("notes_saved", "evictions", "peak_context_tokens", "budget", "steps")})


def test_default_unchanged():
    res, root = run(budget=None, scratchpad=None,
                    replies=[block('note("should not exist")'),
                             block('answer = context[:13]\nprint("FOUND", answer)'),
                             block('FINAL(answer)')])
    m = res.metrics

    assert "scratchpad" not in m and "notes_saved" not in m, \
        f"scratchpad keys must not appear when disabled: {m}"
    obs = res.transcript[0]["observation"]
    assert "EXCEPTION" in obs or "NameError" in obs, \
        f"note() must not exist when scratchpad is off, got: {obs!r}"
    assert "note(text" not in root.sent[0][0]["content"], \
        "system prompt must not advertise note() when disabled"
    assert res.finished and res.answer == "PASSKEY-12345"
    print("C. default path unchanged (note() raises, no scratchpad keys)")


def main():
    test_scratchpad_only()
    test_scratchpad_with_budget()
    test_default_unchanged()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
