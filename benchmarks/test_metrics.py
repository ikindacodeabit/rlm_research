"""Offline unit test for the scoring standard (no GPU / no server / no network).

Guards the property the whole harness depends on: **a prediction is scored by its
benchmark's official metric, and identically no matter which arm produced it.**

Before this existed, nothing in the repo tested any metric, and three different
definitions of "correct" had drifted apart.

Run:  python benchmarks/test_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.answer_extraction import extract_answers, strip_reasoning
from benchmarks.longbench_metrics import (
    EXACT_MATCH_METRICS,
    KNOWN_METRICS,
    _METRIC_FN,
    loft_scores,
    normalize_answer,
    score_example,
)
from benchmarks.run_benchmark import recall, score_record

failures: list[str] = []


def check(label: str, got, want, tol: float = 1e-6) -> None:
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    if not ok:
        failures.append(f"{label}: got {got!r}, want {want!r}")


# --------------------------------------------------------------------------- #
# 1. LOFT — faithful to google-deepmind/loft evaluation/{utils,rag}.py
# --------------------------------------------------------------------------- #
def test_loft_single_value() -> None:
    """utils.compute_subspan_em: max over golds of `gold in pred`. UNIDIRECTIONAL."""
    s = lambda g, p: loft_scores(g, p, multi_value=False)

    check("loft/single exact", s(["the following day"], ["the following day"])["subspan_em"], 1.0)
    check("loft/single gold-in-prose", s(["Paris"], ["The answer is Paris."])["subspan_em"], 1.0)
    check("loft/single normalisation", s(["The Beach"], ["the beach!"])["subspan_em"], 1.0)

    # The case that triggered this whole change. LOFT gold spans are extracted from
    # source sentences and sometimes carry a stray leading token ("island"). Under
    # the official unidirectional rule this is 0.0 even though the answer is right.
    # Asserting 0.0 documents that the standard is strict BY DESIGN — Gemini-1.5-Pro
    # scores 0.28-0.35 on QUEST-128k. Do not "fix" this by making it bidirectional:
    # that would silently diverge from every published LOFT number.
    got = s(["island Koh Phi Phi"],
            ["The Beach was filmed in Koh Phi Phi, which is located in Thailand."])
    check("loft/single stray-gold-token is 0", got["subspan_em"], 0.0)
    check("loft/single f1 gives partial credit", round(got["f1"], 2), 0.38)

    # pred inside gold does NOT count for single-value (it does for multi-value)
    check("loft/single is one-way", s(["island Koh Phi Phi"], ["Koh Phi Phi"])["subspan_em"], 0.0)
    check("loft/single em", s(["paris"], ["Paris"])["em"], 1.0)
    check("loft/single em rejects prose", s(["paris"], ["The answer is Paris"])["em"], 0.0)


def test_loft_multi_value() -> None:
    """rag.compute_multi_value_subspan_em: bidirectional + matching, ALL-or-nothing."""
    s = lambda g, p: loft_scores(g, p, multi_value=True)

    check("loft/multi all matched", s(["alpha", "beta"], ["alpha", "beta"])["subspan_em"], 1.0)
    check("loft/multi extra pred ok", s(["alpha", "beta"], ["alpha", "beta", "x"])["subspan_em"], 1.0)
    # all-or-nothing: one missing gold zeroes it, even though coverage is 0.5
    partial = s(["alpha", "beta"], ["alpha"])
    check("loft/multi missing gold -> 0", partial["subspan_em"], 0.0)
    check("loft/multi coverage still 0.5", partial["coverage"], 0.5)
    # bidirectional: a pred contained IN a gold counts here (unlike single-value)
    check("loft/multi is two-way", s(["island koh phi phi"], ["koh phi phi"])["subspan_em"], 1.0)
    # the matching must be to DISTINCT preds — two golds cannot share one pred
    check("loft/multi needs distinct preds",
          s(["alpha", "alpha beta"], ["alpha beta"])["subspan_em"], 0.0)
    check("loft/multi em is set equality", s(["a", "b"], ["b", "a"])["em"], 1.0)
    check("loft/multi em rejects extra", s(["a", "b"], ["a", "b", "c"])["em"], 0.0)


def test_loft_edge_cases() -> None:
    check("loft/empty pred", loft_scores(["a"], [], multi_value=False)["subspan_em"], 0.0)
    # upstream's float(all([])) would vacuously credit a question with no golds
    check("loft/empty gold not credited",
          loft_scores([], ["a"], multi_value=True)["subspan_em"], 0.0)
    check("loft/None pred", score_example("loft_subspan_em", None, ["a"]), 0.0)


# --------------------------------------------------------------------------- #
# 2. Answer extraction — the shared layer, LOFT's parsing rules
# --------------------------------------------------------------------------- #
def test_extraction() -> None:
    check("extract/list", extract_answers("Final Answer: ['Koh Phi Phi']"), ["Koh Phi Phi"])
    check("extract/multi list", extract_answers("Final Answer: ['a', 'b']"), ["a", "b"])
    # a restated format template must not win over the real answer
    check("extract/last list wins",
          extract_answers("use ['answer1', 'answer2']\nFinal Answer: ['Real']"), ["Real"])
    check("extract/after prefix",
          extract_answers("blah\nFinal Answer: Paris\ntrailing", answer_prefix="Final Answer:"),
          ["Paris"])
    check("extract/no cue -> whole string",
          extract_answers("Paris is the answer"), ["Paris is the answer"])
    check("extract/empty", extract_answers(""), [])
    check("extract/none", extract_answers(None), [])
    check("extract/think block", extract_answers("<think>maybe X</think>Final Answer: ['Y']"), ["Y"])
    check("extract/dangling think", strip_reasoning("leaked</think> answer"), "answer")
    check("extract/no think is noop", strip_reasoning("plain answer"), "plain answer")


# --------------------------------------------------------------------------- #
# 3. The other official metrics (regression guard — these must not drift)
# --------------------------------------------------------------------------- #
def test_longbench_metrics() -> None:
    check("qa_f1", round(score_example("qa_f1", "the capital is Paris", ["Paris"]), 3), 0.5)
    check("rouge", round(score_example("rouge", "a b c d", ["a b c e"]), 2), 0.75)
    check("choice", score_example("choice", "Answer: B", ["B"]), 1.0)
    check("choice wrong", score_example("choice", "Answer: C", ["B"]), 0.0)
    check("count", score_example("count", "there are 3", ["3"]), 1.0)
    check("count penalises extra numbers", score_example("count", "3 of 20", ["3"]), 0.5)
    check("retrieval", score_example("retrieval", "Paragraph 4 is it", ["Paragraph 4"]), 1.0)
    check("classification",
          score_example("classification", "the answer is ABBR", ["ABBR"],
                        all_classes=["ABBR", "DESC"]), 1.0)
    check("code_sim", round(score_example("code_sim", "x = 1", ["x = 2"]), 1), 0.8)
    # trec/triviaqa/samsum truncate at the first newline (upstream rule)
    check("truncate-first-line",
          score_example("qa_f1", "Paris\nand more text", ["Paris"], dataset="triviaqa"), 1.0)


def test_recall() -> None:
    check("recall single", recall("the passkey is 8742547", ["8742547"]), 1.0)
    check("recall partial", recall("a and b", ["a", "b", "c"]), 2 / 3)
    check("recall none", recall(None, ["a"]), 0.0)
    # now shares normalize_answer: punctuation must not block a match
    check("recall ignores punctuation", recall("answer: Paris.", ["Paris"]), 1.0)


def test_normalisation() -> None:
    check("normalize lowercases", normalize_answer("The Beach"), "beach")
    check("normalize drops articles", normalize_answer("a cat the dog"), "cat dog")
    check("normalize strips punctuation", normalize_answer("hi, there!"), "hi there")
    # NFD: composed (U+00E9) vs decomposed (e + U+0301) accents must compare equal.
    # The two are visually identical in source, so assert they really differ first --
    # otherwise a well-meaning editor "fixing" the encoding makes this vacuous.
    composed, decomposed = "café", "café"
    assert composed != decomposed, "test setup: these must differ as raw strings"
    check("normalize NFD", normalize_answer(composed), normalize_answer(decomposed))
    check("normalize NFD scores equal",
          loft_scores([composed], [decomposed], multi_value=False)["subspan_em"], 1.0)


# --------------------------------------------------------------------------- #
# 4. THE regression guard for this whole change: arm parity
# --------------------------------------------------------------------------- #
def test_arm_parity() -> None:
    """The same prediction string must score identically for vanilla and RLM.

    score_record is the single scoring site and takes no `mode` argument, so parity
    holds by construction. This test exists to make any future re-introduction of a
    per-arm code path fail loudly.
    """
    examples = [
        {"id": "x", "metric": "loft_subspan_em", "multi_value": False,
         "answer_prefix": "Final Answer:", "answers": ["Paris"]},
        {"id": "y", "metric": "loft_subspan_em", "multi_value": True,
         "answer_prefix": "Final Answer:", "answers": ["alpha", "beta"]},
        {"id": "z", "metric": "qa_f1", "answers": ["Paris"]},
        {"id": "w", "metric": "recall", "answers": ["8742547"]},
        {"id": "v", "metric": "choice", "answers": ["B"]},
    ]
    # one prose-shaped string (vanilla's style) and one bare/list string (FINAL()'s)
    preds = [
        "Based on the documents, the answer is Paris.",
        "Final Answer: ['Paris']",
        "['alpha', 'beta']",
        "8742547",
        "B",
        "<think>hmm</think>Final Answer: ['alpha', 'beta']",
    ]
    for ex in examples:
        for pred in preds:
            a = score_record(ex, pred)
            b = score_record(ex, pred)
            check(f"parity {ex['metric']}/{pred[:18]!r}", a, b)

    # and the substantive half: an answer that is correct in either style scores the
    # same, so the metric cannot reward answer shape
    ex = {"id": "p", "metric": "loft_subspan_em", "multi_value": True,
          "answer_prefix": "Final Answer:", "answers": ["alpha", "beta"]}
    prose = score_record(ex, "Final Answer: ['alpha', 'beta']")[1]
    bare = score_record(ex, "['alpha', 'beta']")[1]
    check("parity prose vs bare list", prose, bare)
    check("parity both correct", prose, 1.0)


def test_metric_registry() -> None:
    """Every metric a loader can emit must be scoreable, and vice versa."""
    from benchmarks import datasets  # noqa: F401  (import check)

    for name in _METRIC_FN:
        check(f"registry/{name} known", name in KNOWN_METRICS, True)
    check("registry/recall known", "recall" in KNOWN_METRICS, True)
    check("registry/loft known", "loft_subspan_em" in KNOWN_METRICS, True)
    # `correct` must only be claimed for exact-match-family metrics
    for name in ("qa_f1", "rouge", "code_sim"):
        check(f"registry/{name} not exact-match", name in EXACT_MATCH_METRICS, False)
    for name in ("choice", "recall", "loft_subspan_em"):
        check(f"registry/{name} is exact-match", name in EXACT_MATCH_METRICS, True)

    # an unknown metric must raise, not silently score 0
    try:
        score_example("no_such_metric", "x", ["y"])
        failures.append("registry: unknown metric did not raise")
    except ValueError:
        pass

    # a loader example with no metric must raise, not fall back
    try:
        score_record({"id": "q", "answers": ["a"]}, "a")
        failures.append("registry: missing metric did not raise")
    except KeyError:
        pass


def main() -> None:
    for fn in (test_loft_single_value, test_loft_multi_value, test_loft_edge_cases,
               test_extraction, test_longbench_metrics, test_recall,
               test_normalisation, test_arm_parity, test_metric_registry):
        fn()
        print(f"  ran {fn.__name__}")
    if failures:
        print(f"\nFAIL ({len(failures)}):")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("\nOK - scoring standard holds")


if __name__ == "__main__":
    main()
