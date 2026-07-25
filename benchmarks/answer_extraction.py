"""One shared answer-extraction layer, applied identically to BOTH arms.

Why this exists: the vanilla arm emits free prose from a single completion, while
the RLM arm emits `str(FINAL(x))` — plus four other channels (`final_in_prose`,
`final_var_in_prose`, `repetition_broken` which answers from the SUB model, and
`gave_up_no_code` which returns the raw reply). Feeding those raw strings straight
into a string-matching metric measures answer STYLE as much as correctness, in
opposite directions depending on the metric: `count_score`/`retrieval_score` divide
by the number of digit-runs in the prediction and so punish prose, while ROUGE-L's
precision denominator punishes length, and `str(FINAL(["a","b"]))` injects brackets
and quotes that punish the RLM under token-F1.

The fix is to normalise the SHAPE of the prediction once, before any metric runs, so
the metric sees a comparable answer regardless of which arm produced it.

`extract_answers` mirrors LOFT's parsing (google-deepmind/loft, and
sparse-attention-hub `benchmark/loft/calculate_metrics.py`): prefer a bracketed
Python list, else the text following the answer prefix, else the whole string.
"""
from __future__ import annotations

import ast
import re

# Qwen3 and friends emit a visible reasoning block. In this repo it is normally
# stripped SERVER-side (every slurm script serves with --reasoning-parser qwen3,
# and client.chat returns message.content, not reasoning_content), so this is a
# defensive guard for the Mac/NIM path and for any run that omits the flag. It is
# a no-op on predictions that never contained a think block.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_DANGLING_THINK_RE = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str | None) -> str:
    """Remove <think>...</think> blocks from a raw completion."""
    if not text:
        return ""
    out = _THINK_RE.sub(" ", text)
    # An unclosed block means generation was cut off mid-thought; a stray closing
    # tag means the opening tag was consumed by the server parser. Both leave
    # reasoning text that would otherwise be scored as the answer.
    if "</think>" in out:
        out = _DANGLING_THINK_RE.sub(" ", out)
    return out.strip()


def _parse_list(text: str) -> list[str] | None:
    """Return the LAST bracketed Python list in `text`, or None.

    Last, not first: models often restate the requested format ("in the format
    ['answer1', 'answer2']") before giving the real answer.
    """
    starts = [m.start() for m in re.finditer(r"\[", text)]
    for start in reversed(starts):
        depth, end = 0, None
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            continue
        try:
            val = ast.literal_eval(text[start:end])
        except (ValueError, SyntaxError):
            continue
        if isinstance(val, (list, tuple)):
            items = [str(v).strip() for v in val if str(v).strip()]
            if items:
                return items
    return None


def extract_answers(
    prediction: str | None,
    *,
    answer_prefix: str | None = None,
    expect_list: bool = False,
) -> list[str]:
    """Normalise a raw prediction from EITHER arm into a list of answer strings.

    Resolution order (LOFT's):
      1. the last bracketed list that parses as a Python literal;
      2. the text after `answer_prefix` (e.g. "Final Answer:"), first line only;
      3. the whole cleaned string.

    Returns [] for an empty/None prediction — the caller scores that as 0.0.
    `expect_list` only affects step 2: multi-value tasks keep the full remainder
    (it may contain a comma list), single-value tasks stop at the first newline.
    """
    text = strip_reasoning(prediction)
    if not text:
        return []

    items = _parse_list(text)
    if items is not None:
        return items

    if answer_prefix:
        key = answer_prefix.strip().rstrip(":").strip().lower()
        idx = text.lower().rfind(key.lower())
        if key and idx >= 0:
            after = text[idx + len(key) :].lstrip(" :\t\n")
            if after.strip():
                text = after if expect_list else after.split("\n", 1)[0]

    text = text.strip()
    if not text:
        return []
    if expect_list and "," in text and "\n" not in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) > 1:
            return parts
    return [text]


def extract_answer(
    prediction: str | None, *, answer_prefix: str | None = None
) -> str:
    """Single-string form of `extract_answers`, for scalar-answer metrics."""
    items = extract_answers(prediction, answer_prefix=answer_prefix)
    return items[0] if items else ""
