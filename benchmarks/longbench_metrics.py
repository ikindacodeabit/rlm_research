"""Official-style LongBench (v1) per-dataset metrics, dependency-free.

LongBench scores every dataset with a TASK-SPECIFIC metric (F1 for QA, ROUGE-L
for summarization, classification/retrieval/count for synthetic, edit-similarity
for code) rather than one global accuracy. Reusing the harness's substring
`recall` for summarization would report near-zero everywhere and make the
RLM-vs-vanilla comparison meaningless, so we mirror the upstream metrics from
THUDM/LongBench (github.com/THUDM/LongBench, `metrics.py`).

To stay install-free on a laptop we reimplement the few that need third-party
libs: ROUGE-L via an LCS (the `rouge` pkg's ROUGE-L is F-measure over the LCS),
and code edit-similarity via difflib.SequenceMatcher (a stand-in for fuzz.ratio).
Chinese datasets are intentionally NOT handled here (they need jieba); the loader
defaults to the 16 English subsets.

Every metric returns a float in [0, 1]; the runner takes the max over gold
answers, exactly like upstream.
"""
from __future__ import annotations

import difflib
import re
import string
import unicodedata
from collections import Counter


# --------------------------------------------------------------------------- #
# Normalisation (SQuAD-style) for token-F1
# --------------------------------------------------------------------------- #
def normalize_answer(s: str) -> str:
    """Lower, strip punctuation, drop articles, collapse whitespace.

    NFD-normalise first, matching google-deepmind/loft `evaluation/utils.py` and
    THUDM/LongBench. Without it, accented gold spans decompose differently from
    the model's output and compare unequal despite being the same text.
    """
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _f1(prediction_tokens: list[str], ground_truth_tokens: list[str]) -> float:
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **_) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    return _f1(pred_tokens, gold_tokens)


# --------------------------------------------------------------------------- #
# ROUGE-L (F-measure over the longest common subsequence of tokens)
# --------------------------------------------------------------------------- #
def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_score(prediction: str, ground_truth: str, **_) -> float:
    pred_tokens = prediction.split()
    gold_tokens = ground_truth.split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# Synthetic / classification / code metrics (upstream verbatim in spirit)
# --------------------------------------------------------------------------- #
def classification_score(prediction: str, ground_truth: str, all_classes=None, **_) -> float:
    all_classes = all_classes or []
    em_match = [c for c in all_classes if c in prediction]
    # a longer class label that strictly contains the gold shouldn't count
    for term in list(em_match):
        if term in ground_truth and term != ground_truth:
            em_match.remove(term)
    if ground_truth in em_match:
        return 1.0 / len(em_match)
    return 0.0


def retrieval_score(prediction: str, ground_truth: str, **_) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    gold_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right = sum(1 for n in numbers if n == gold_id)
    return right / len(numbers)


def count_score(prediction: str, ground_truth: str, **_) -> float:
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right = sum(1 for n in numbers if n == str(ground_truth))
    return right / len(numbers)


def code_sim_score(prediction: str, ground_truth: str, **_) -> float:
    # take the first "real" code line (upstream skips comment/markdown lines)
    line = ""
    for ln in prediction.lstrip("\n").split("\n"):
        if "`" not in ln and "#" not in ln and "//" not in ln:
            line = ln
            break
    return difflib.SequenceMatcher(None, line, ground_truth).ratio()


# Multiple-choice accuracy for LongBench v2 (A/B/C/D). Not part of v1, but kept
# here so the runner has one task-aware metric dispatch.
def choice_score(prediction: str, ground_truth: str, **_) -> float:
    m = re.search(r"\b([ABCD])\b", (prediction or "").strip().upper())
    pred = m.group(1) if m else (prediction or "").strip().upper()[:1]
    return 1.0 if pred == str(ground_truth).strip().upper() else 0.0


# --------------------------------------------------------------------------- #
# LOFT (Long-Context Frontiers) RAG metrics
# --------------------------------------------------------------------------- #
# Faithful port of google-deepmind/loft `evaluation/utils.py` + `evaluation/rag.py`.
# The LOFT paper (ACL Findings 2025) uses SUBSPAN EM as the primary metric for all
# RAG tasks; EM / F1 / coverage are secondaries and are reported alongside so the
# headline can be changed later without re-running anything.
#
# Both arms' predictions arrive here already parsed by benchmarks.answer_extraction
# into a list of answer strings, exactly as upstream's process_prediction does.
# All comparisons happen on normalize_answer'd text.
#
# Single-value (nq, hotpotqa, musique) and multi-value (qampari, quest) use
# DIFFERENT rules upstream, and the difference is not cosmetic:
#   single: max over golds of `gold in pred`            -- unidirectional
#   multi : bidirectional containment, then a maximum matching, scored
#           ALL-OR-NOTHING (`float(all(aligned_scores))`)
def _kuhn_matching(adj: list[set[int]], n_right: int) -> int:
    """Maximum-cardinality bipartite matching (Kuhn's augmenting paths).

    Upstream calls scipy.optimize.linear_sum_assignment(-scores) on a 0/1 matrix
    and then asks whether every gold row got a 1. On a 0/1 matrix that is exactly
    a maximum-cardinality matching, so this is an equivalent result — not an
    approximation — and it keeps scipy out of the dependency list (it is neither
    declared in requirements.txt nor installed in the venv).
    """
    match_r = [-1] * n_right

    def augment(u: int, seen: set[int]) -> bool:
        for v in adj[u]:
            if v in seen:
                continue
            seen.add(v)
            if match_r[v] == -1 or augment(match_r[v], seen):
                match_r[v] = u
                return True
        return False

    return sum(1 for u in range(len(adj)) if augment(u, set()))


def loft_scores(
    gold_answers: list[str], pred_answers: list[str], *, multi_value: bool
) -> dict[str, float]:
    """Every LOFT RAG metric at once. `subspan_em` is the primary."""
    golds = [g for g in (normalize_answer(str(a)) for a in gold_answers) if g]
    preds = [p for p in (normalize_answer(str(a)) for a in pred_answers) if p]
    if not golds or not preds:
        # Upstream scores an empty prediction as 0 across the board. An empty gold
        # list would make `all([])` vacuously True upstream; we return 0 instead of
        # crediting a question that has no answer to find.
        return {"subspan_em": 0.0, "em": 0.0, "f1": 0.0, "coverage": 0.0}

    if multi_value:
        # rag.compute_multi_value_subspan_em: bidirectional containment + matching,
        # then float(all(...)) -- every gold must be matched to a DISTINCT pred.
        adj = [
            {j for j, p in enumerate(preds) if g in p or p in g}
            for g in golds
        ]
        subspan = 1.0 if _kuhn_matching(adj, len(preds)) == len(golds) else 0.0
        em = float(set(golds) == set(preds))                 # compute_em_multi_value
        # compute_coverage: containment-based, to agree with subspan above. Exact set
        # intersection would report 0.0 for gold "alpha" vs pred "alpha smith" while
        # subspan reported 1.0 -- two secondaries disagreeing inside one call.
        coverage = sum(
            1 for g in golds if any(g in p or p in g for p in preds)
        ) / len(golds)
        # Per-gold best match, AVERAGED. A max over the gold x pred cross product
        # let one lucky prediction saturate F1 while every other gold was missed,
        # which would silently mislead anyone promoting f1 to the headline.
        f1 = sum(max(_f1(p.split(), g.split()) for p in preds)
                 for g in golds) / len(golds)
    else:
        # utils.compute_subspan_em / compute_em / compute_f1, each max over golds,
        # against the single extracted prediction.
        pred = preds[0]
        subspan = max(1.0 if g in pred else 0.0 for g in golds)
        em = max(float(g == pred) for g in golds)
        coverage = float(sum(g in pred for g in golds)) / len(golds)
        f1 = max(_f1(pred.split(), g.split()) for g in golds)

    return {"subspan_em": subspan, "em": em, "f1": f1, "coverage": coverage}


# --------------------------------------------------------------------------- #
# dataset (subset) -> metric, and the few-shot prediction truncation rule
# --------------------------------------------------------------------------- #
DATASET2METRIC = {
    "narrativeqa": "qa_f1",
    "qasper": "qa_f1",
    "multifieldqa_en": "qa_f1",
    "hotpotqa": "qa_f1",
    "2wikimqa": "qa_f1",
    "musique": "qa_f1",
    "gov_report": "rouge",
    "qmsum": "rouge",
    "multi_news": "rouge",
    "trec": "classification",
    "triviaqa": "qa_f1",
    "samsum": "rouge",
    "passage_count": "count",
    "passage_retrieval_en": "retrieval",
    "lcc": "code_sim",
    "repobench-p": "code_sim",
}

# upstream truncates the prediction at the first newline for these few-shot tasks
_TRUNCATE_FIRST_LINE = {"trec", "triviaqa", "samsum", "lsht"}

_METRIC_FN = {
    "qa_f1": qa_f1_score,
    "rouge": rouge_score,
    "classification": classification_score,
    "retrieval": retrieval_score,
    "count": count_score,
    "code_sim": code_sim_score,
    "choice": choice_score,
}

# LOFT metrics are computed as a family by loft_scores(); this maps the stored
# metric name to the key of the primary number. `recall` is handled by the caller
# (benchmarks/run_benchmark.py) because it predates this module.
_LOFT_PRIMARY = {"loft_subspan_em": "subspan_em"}

# Metrics whose OFFICIAL evaluation includes parsing the model output into an
# answer list before scoring. Only these get benchmarks.answer_extraction applied;
# LongBench/RULER score the raw generation (see run_benchmark.score_record).
LOFT_METRICS = frozenset(_LOFT_PRIMARY)

# Metrics whose score==1.0 genuinely means "exactly right", so a boolean `correct`
# is meaningful. For F1/ROUGE/code_sim a 1.0 is a continuous score that happened to
# saturate, and reporting it as "exact-correct" is misleading.
EXACT_MATCH_METRICS = {
    "choice", "count", "retrieval", "classification", "recall", "loft_subspan_em",
}

KNOWN_METRICS = set(_METRIC_FN) | set(_LOFT_PRIMARY) | {"recall"}


def score_example(metric: str, prediction: str | None, answers: list[str],
                  *, dataset: str | None = None, all_classes=None,
                  pred_answers: list[str] | None = None,
                  multi_value: bool = False) -> float:
    """Primary score for one example under `metric` (upstream max-over-golds rule).

    `pred_answers` is the output of benchmarks.answer_extraction.extract_answers and
    is REQUIRED for the LOFT metrics, whose upstream definition operates on a parsed
    answer list rather than raw text. Other metrics keep taking the raw `prediction`,
    matching their upstream definitions.
    """
    return score_example_all(
        metric, prediction, answers, dataset=dataset, all_classes=all_classes,
        pred_answers=pred_answers, multi_value=multi_value,
    )[0]


def score_example_all(metric: str, prediction: str | None, answers: list[str],
                      *, dataset: str | None = None, all_classes=None,
                      pred_answers: list[str] | None = None,
                      multi_value: bool = False) -> tuple[float, dict[str, float]]:
    """(primary_score, secondary_scores) for one example.

    Secondaries are stored on the record so the headline metric can be changed
    later without re-running the benchmark.
    """
    if metric in _LOFT_PRIMARY:
        if prediction is None:
            return 0.0, {"subspan_em": 0.0, "em": 0.0, "f1": 0.0, "coverage": 0.0}
        preds = pred_answers if pred_answers is not None else [prediction]
        scores = loft_scores([str(a) for a in answers], preds,
                             multi_value=multi_value)
        return scores[_LOFT_PRIMARY[metric]], scores

    if prediction is None:
        return 0.0, {}
    pred = prediction
    if dataset in _TRUNCATE_FIRST_LINE:
        pred = pred.lstrip("\n").split("\n")[0]
    fn = _METRIC_FN.get(metric)
    if fn is None:
        raise ValueError(f"unknown metric {metric!r}; known: {sorted(KNOWN_METRICS)}")
    best = 0.0
    for gold in answers:
        best = max(best, fn(pred, str(gold), all_classes=all_classes))
    return best, {}
