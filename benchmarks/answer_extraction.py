"""LOFT's answer-parsing step, applied identically to both arms.

SCOPE — read this before reusing it elsewhere. This is NOT a universal
"normalise every prediction" layer, and deliberately so:

  * LOFT's official evaluation (google-deepmind/loft) parses the model's output
    into an answer list BEFORE scoring, so parsing IS part of that benchmark's
    metric definition. This module implements that step.
  * LongBench v1/v2 and RULER define their metrics over the RAW generation.
    Running predictions through an extractor there would diverge from the
    published numbers, so `score_record` does not do it.

Arm parity for those benchmarks is handled on the GENERATION side instead: every
loader supplies an `answer_format` that both `vanilla_answer` and the RLM's
ROOT_SYSTEM_PROMPT receive, so the two arms are asked for the same answer shape
and the official raw-text metric is then fair to both. See benchmarks/datasets.py.
"""
from __future__ import annotations

import ast
import re

# Qwen3 and friends emit a visible reasoning block. In this repo it is normally
# stripped SERVER-side (every slurm script serves with --reasoning-parser qwen3,
# and client.chat returns message.content, not reasoning_content), so this is a
# defensive guard for the Mac/NIM path and for any run that omits the flag.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
_THINK_LEADING_CLOSE_RE = re.compile(r"\A.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str | None) -> str:
    """Remove <think>...</think> reasoning from a raw completion.

    Handles three shapes, all of which occur in practice:
      * a well-formed block;
      * a stray leading `</think>` (server consumed the opening tag);
      * an UNCLOSED `<think>` (generation hit max_tokens mid-thought) — everything
        from the tag onward is reasoning, so it is dropped rather than scored.
    """
    if not text:
        return ""
    out = _THINK_BLOCK_RE.sub(" ", text)
    while "</think>" in out:
        out = _THINK_LEADING_CLOSE_RE.sub(" ", out, count=1)
    out = _THINK_OPEN_RE.sub(" ", out)
    return out.strip()


def _iter_balanced_lists(text: str):
    """Yield substrings of `text` that look like balanced [...] spans.

    Quote-aware: brackets inside string literals are ignored, so
    `["a [ b", "c"]` parses instead of confusing the depth counter. Candidates are
    yielded OUTERMOST-first (left to right by start), so a nested list is returned
    whole rather than having its last inner list win.
    """
    i, n = 0, len(text)
    while i < n:
        if text[i] != "[":
            i += 1
            continue
        depth, quote, esc, j = 0, "", False, i
        while j < n:
            ch = text[j]
            if quote:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == quote:
                    quote = ""
            elif ch in "\"'":
                quote = ch
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    yield text[i : j + 1]
                    break
            j += 1
        i += 1


def _parse_list(text: str) -> list[str] | None:
    """Return the first parseable balanced Python list in `text`, else None."""
    for candidate in _iter_balanced_lists(text):
        try:
            val = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue
        if isinstance(val, (list, tuple)):
            # Flatten one level: a model that emits [["a","b"],["c","d"]] means four
            # answers, not two stringified sublists. LOFT gold answers are always
            # flat strings, so a nested list is malformed output to be recovered.
            items: list[str] = []
            for v in val:
                for x in (v if isinstance(v, (list, tuple)) else [v]):
                    s = str(x).strip()
                    if s:
                        items.append(s)
            if items:
                return items
    return None


def _after_prefix(text: str, answer_prefix: str) -> str | None:
    """Text following the LAST occurrence of `answer_prefix` that has content."""
    key = answer_prefix.strip().rstrip(":").strip()
    if not key:
        return None
    low_text, low_key = text.lower(), key.lower()
    idx = low_text.rfind(low_key)
    while idx >= 0:
        after = text[idx + len(low_key) :].lstrip(" :\t\n")
        if after.strip():
            return after
        # The model merely MENTIONED the cue with nothing after it ("I cannot find
        # the Final Answer"). Fall back to an earlier occurrence rather than
        # abandoning the prefix logic and scoring the whole reply.
        idx = low_text.rfind(low_key, 0, idx)
    return None


def extract_answers(
    prediction: str | None,
    *,
    answer_prefix: str | None = None,
    expect_list: bool = False,
) -> list[str]:
    """Parse a raw prediction from EITHER arm into LOFT's answer list.

    Resolution order:
      1. text after `answer_prefix` (the cue LOFT asks both arms to emit);
      2. within that, a bracketed Python list when `expect_list` (multi-value);
      3. otherwise the first line of the cued text.

    The list parse is gated on `expect_list` and runs only on cued text. Running it
    unconditionally meant any stray bracket hijacked the answer — a citation marker
    turned "Final Answer: Paris [1]" into ["1"], scoring a correct answer 0.0.

    KNOWN LENIENCY, deliberately kept: when no cue is present the WHOLE reply is
    treated as the answer, so a model that merely quotes retrieved text containing
    the gold can score under subspan-EM containment. Rejecting uncued replies was
    tried and is worse: the RLM's native answer is `str(FINAL(x))`, which carries no
    cue, so strictness zeroed correct RLM answers while cued vanilla prose passed —
    a larger asymmetry than the one it removed. The real fix is on the generation
    side, where both arms now receive the same `answer_format` contract.

    Returns [] for an empty/None prediction; the caller scores that as 0.0.
    """
    text = strip_reasoning(prediction)
    if not text:
        return []

    cued = _after_prefix(text, answer_prefix) if answer_prefix else None
    body = cued if cued is not None else text

    if expect_list:
        items = _parse_list(body)
        if items is not None:
            return items

    if cued is not None:
        body = body.split("\n", 1)[0] if not expect_list else body
    body = body.strip()
    if not body:
        return []

    if expect_list:
        # An enumerated or comma-joined answer is the natural PROSE form of a list;
        # without this, only the RLM's str(FINAL([...])) would ever parse and the
        # extraction step would itself become an arm asymmetry.
        lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip()
                 for ln in body.split("\n")]
        lines = [ln for ln in lines if ln]
        if len(lines) > 1:
            return lines
        if len(lines) == 1 and "," in lines[0]:
            parts = [p.strip() for p in lines[0].split(",") if p.strip()]
            if len(parts) > 1:
                return parts
        body = lines[0] if lines else body

    return [body] if body else []
