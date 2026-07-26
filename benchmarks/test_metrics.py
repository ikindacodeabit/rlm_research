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
    """LOFT answer parsing. Note extraction is LOFT-only -- see score_record."""
    P = "Final Answer:"
    # list parsing is gated on expect_list (multi-value tasks)
    check("extract/list", extract_answers("Final Answer: ['a','b']", answer_prefix=P,
                                          expect_list=True), ["a", "b"])
    check("extract/after prefix",
          extract_answers("blah\nFinal Answer: Paris\ntrailing", answer_prefix=P), ["Paris"])
    check("extract/no cue -> whole string",
          extract_answers("Paris is the answer"), ["Paris is the answer"])
    check("extract/empty", extract_answers(""), [])
    check("extract/none", extract_answers(None), [])
    check("extract/think block",
          extract_answers("<think>maybe X</think>Final Answer: Y", answer_prefix=P), ["Y"])

    # REGRESSION: a stray bracket used to hijack the answer. "Final Answer: Paris [1]"
    # parsed to ['1'] and scored a CORRECT answer 0.0.
    check("regress/citation marker not a list",
          extract_answers("Final Answer: Paris [1]", answer_prefix=P), ["Paris [1]"])
    check("regress/subscript not a list",
          extract_answers("Final Answer: data[3]", answer_prefix=P), ["data[3]"])

    # REGRESSION: the scan took the LAST/innermost list, dropping half a nested one.
    check("regress/nested list flattened",
          extract_answers('Final Answer: [["a","b"],["c","d"]]', answer_prefix=P,
                          expect_list=True), ["a", "b", "c", "d"])
    # REGRESSION: a bracket inside a quoted string broke the depth counter.
    check("regress/bracket inside string",
          extract_answers('Final Answer: ["a [ b", "c"]', answer_prefix=P,
                          expect_list=True), ["a [ b", "c"])

    # REGRESSION: prose list forms must parse, or only the RLM's str(FINAL([...]))
    # would ever match and extraction would itself become an arm asymmetry.
    check("regress/numbered prose list",
          extract_answers("Final Answer:\n1. Washington, D.C.\n2. Paris",
                          answer_prefix=P, expect_list=True), ["Washington, D.C.", "Paris"])
    check("regress/bulleted prose list",
          extract_answers("Final Answer:\n- alpha\n- beta", answer_prefix=P,
                          expect_list=True), ["alpha", "beta"])

    # REGRESSION: a trailing MENTION of the cue with nothing after it used to abandon
    # the prefix logic entirely and score the whole reply.
    check("regress/cue mentioned last falls back",
          extract_answers("Final Answer: Paris\nI cannot find the Final Answer",
                          answer_prefix=P), ["Paris"])


def test_strip_reasoning() -> None:
    check("think/well formed", strip_reasoning("<think>x</think>real"), "real")
    check("think/no think is noop", strip_reasoning("plain answer"), "plain answer")
    check("think/dangling close", strip_reasoning("leaked</think> answer"), "answer")
    # REGRESSION: an UNCLOSED block (generation cut off at max_tokens) was left in
    # place and scored as the answer, despite the comment claiming otherwise.
    check("regress/unclosed think", strip_reasoning("<think>abc"), "")
    check("regress/answer then unclosed", strip_reasoning("answer\n<think>oops"), "answer")
    check("regress/two closers", strip_reasoning("A</think> mid </think> B"), "B")


def test_loft_secondaries() -> None:
    """REGRESSION: the stored secondaries must not contradict the primary."""
    # coverage was exact set intersection while subspan used containment, so the two
    # disagreed inside a single call.
    r = loft_scores(["alpha"], ["alpha smith"], multi_value=True)
    check("regress/coverage is containment", r["coverage"], 1.0)
    check("regress/coverage agrees with subspan", r["subspan_em"], 1.0)
    # f1 was a max over the gold x pred cross product, so one lucky pred saturated it
    # while every other gold was missed.
    r2 = loft_scores(["alpha", "beta"], ["zzz", "alpha"], multi_value=True)
    check("regress/f1 does not saturate", r2["f1"], 0.5)
    check("regress/f1 consistent with coverage", r2["coverage"], 0.5)


def test_extraction_is_loft_only() -> None:
    """Pins the fidelity-vs-parity split (see score_record's docstring).

    LongBench/RULER metrics are defined over the RAW generation, so score_record
    must NOT run answer extraction for them; parity for those is handled by the
    answer_format contract both arms receive. Only LOFT parses, because parsing is
    part of LOFT's own official evaluation.
    """
    from benchmarks.datasets import METRIC_ANSWER_FORMAT

    # LOFT: the list wrapper is parsed away, so both answer shapes score alike.
    loft = {"id": "l", "metric": "loft_subspan_em", "multi_value": True,
            "answer_prefix": "Final Answer:", "answers": ["alpha", "beta"]}
    check("loft parses list", score_record(loft, "Final Answer: ['alpha','beta']")[1], 1.0)
    check("loft parses bare list", score_record(loft, "['alpha', 'beta']")[1], 1.0)

    # Non-LOFT: scored on the raw text, matching upstream.
    lb = {"id": "b", "metric": "qa_f1", "subset": "qasper", "answers": ["Paris"]}
    check("longbench scores raw text", score_record(lb, "Paris")[1], 1.0)

    # Every metric has a format contract available...
    for name in ("qa_f1", "rouge", "classification", "count", "retrieval",
                 "code_sim", "choice", "recall"):
        check(f"answer_format/{name} exists", name in METRIC_ANSWER_FORMAT, True)

    # ...but a loader must NOT attach one when its question already states the
    # format. Stacking two instructions dropped niah nothink RLM 80.0% -> 38.0%:
    # greedy decoding switched from a regex to relevant_text.split()[-1].
    from benchmarks.datasets import TASKS
    for task in ("niah", "multikey"):
        ex = next(iter(TASKS[task](1)))
        check(f"no stacked format/{task}", ex.get("answer_format"), None)
        check(f"question carries format/{task}",
              "only" in ex["question"].lower(), True)


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
    # The RLM's native answer carries no cue; requiring one scored it 0.0 while cued
    # vanilla prose passed -- a larger asymmetry than the looseness it removed.
    check("parity uncued RLM list still scores", bare, 1.0)


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
               test_loft_secondaries, test_extraction, test_strip_reasoning,
               test_extraction_is_loft_only, test_longbench_metrics, test_recall,
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
