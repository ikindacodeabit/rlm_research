"""The vanilla arm must fit the server's window, whatever the tokenizer says.

Regression test for the campaign's largest single source of lost data: RULER `cwe`
and `niah_multikey_3`, and LongBench-v2 "Long Structured Data", each returned a 400
"maximum context length is 40960 tokens" on 100%/100%/80% of vanilla records. Those
cells were then scored 0.0 -- a harness failure that reads as a model result.

The predictive shrink in vanilla_answer could not prevent it: TokenCounter falls back
to `len // 4` with no tiktoken installed, which is the very assumption the shrink was
added to escape, so the loop no-ops. The server is now the authority.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rlm.rlm import vanilla_answer, _is_context_overflow


class ContextOverflow(Exception):
    status_code = 400
    def __str__(self):
        return ("Error code: 400 - This model's maximum context length is 40960 "
                "tokens. However, you requested 61348 tokens")


class OtherBadRequest(Exception):
    status_code = 400
    def __str__(self):
        return "Error code: 400 - unsupported parameter: 'reasoning_effort'"


class WindowedClient:
    """A server with a hard prompt ceiling, like vLLM at --max-model-len."""
    def __init__(self, ceiling_chars=40_000):
        self.ceiling, self.sizes = ceiling_chars, []

    def chat(self, messages):
        p = messages[0]["content"]
        self.sizes.append(len(p))
        if len(p) > self.ceiling:
            raise ContextOverflow()
        return "42"


def test_overflow_is_distinguished_from_other_400s():
    assert _is_context_overflow(ContextOverflow())
    assert not _is_context_overflow(OtherBadRequest())
    print("  ran test_overflow_is_distinguished_from_other_400s")


def test_shrinks_until_the_server_accepts():
    c = WindowedClient()
    # A context that passes the len//4 estimate (100k chars -> 25k < 34k) yet is
    # still rejected: exactly the dense-subset case that no-oped before.
    out = vanilla_answer(c, "x " * 60_000, "What is the answer?",
                         char_limit=100_000, max_prompt_tokens=34_000)
    assert out == "42", out
    assert len(c.sizes) > 1, "never retried -- the predictive loop no-oped again"
    assert c.sizes[-1] <= c.ceiling
    assert c.sizes == sorted(c.sizes, reverse=True), "must shrink monotonically"
    print(f"  ran test_shrinks_until_the_server_accepts (tried {len(c.sizes)} sizes)")


def test_a_fitting_prompt_is_sent_once():
    c = WindowedClient()
    vanilla_answer(c, "short context", "q?", char_limit=100_000,
                   max_prompt_tokens=34_000)
    assert len(c.sizes) == 1, "added a retry to a prompt that already fit"
    print("  ran test_a_fitting_prompt_is_sent_once")


def test_non_length_400_still_propagates():
    class Fail:
        def chat(self, messages): raise OtherBadRequest()
    try:
        vanilla_answer(Fail(), "ctx", "q?", char_limit=100_000,
                       max_prompt_tokens=34_000)
    except OtherBadRequest:
        print("  ran test_non_length_400_still_propagates")
    else:
        raise AssertionError("a real bad request was swallowed by the shrink loop")


def test_truncation_note_survives_shrinking():
    """The model must keep being told the document is partial after each shrink."""
    c = WindowedClient()
    vanilla_answer(c, "x " * 60_000, "q?", char_limit=100_000,
                   max_prompt_tokens=34_000)
    seen = []

    class Recorder(WindowedClient):
        def chat(self, messages):
            seen.append(messages[0]["content"])
            return super().chat(messages)

    vanilla_answer(Recorder(), "x " * 60_000, "q?", char_limit=100_000,
                   max_prompt_tokens=34_000)
    assert all("[NOTE: document truncated]" in p for p in seen), seen[-1][:200]
    print("  ran test_truncation_note_survives_shrinking")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("\nOK - vanilla always fits the server window")
