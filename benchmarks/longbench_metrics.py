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
from collections import Counter


# --------------------------------------------------------------------------- #
# Normalisation (SQuAD-style) for token-F1
# --------------------------------------------------------------------------- #
def normalize_answer(s: str) -> str:
    """Lower, strip punctuation, drop articles, collapse whitespace."""

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


def score_example(metric: str, prediction: str | None, answers: list[str],
                  *, dataset: str | None = None, all_classes=None) -> float:
    """Apply a LongBench metric, taking the max over gold answers (upstream rule)."""
    if prediction is None:
        return 0.0
    pred = prediction
    if dataset in _TRUNCATE_FIRST_LINE:
        pred = pred.lstrip("\n").split("\n")[0]
    fn = _METRIC_FN.get(metric)
    if fn is None:
        raise ValueError(f"unknown LongBench metric {metric!r}")
    best = 0.0
    for gold in answers:
        best = max(best, fn(pred, str(gold), all_classes=all_classes))
    return best
