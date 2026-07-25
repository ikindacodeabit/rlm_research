"""Benchmark task loaders.

Each loader yields dicts: {"id", "context", "question", "answers": [str, ...]}.
Synthetic tasks are generated locally (no internet needed on compute nodes);
HF-backed tasks read from a local cache populated by slurm/download_data.sh.
"""
from __future__ import annotations

import json
import os
import random
import string
import sys
from pathlib import Path

from benchmarks.longbench_metrics import DATASET2METRIC

DATA_DIR = Path(os.environ.get("RLM_DATA_DIR", os.path.expanduser("~/rlm_data")))

WORDS = (
    "ocean mountain forest river cloud stone meadow valley harbor lantern "
    "compass voyage thunder ember willow falcon marble quartz cedar prairie"
).split()


def _filler(rng: random.Random, n_chars: int) -> str:
    out, total = [], 0
    while total < n_chars:
        sent = " ".join(rng.choices(WORDS, k=rng.randint(8, 14))).capitalize() + "."
        out.append(sent)
        total += len(sent) + 1
    return " ".join(out)


def gen_niah(n_examples: int = 50, ctx_chars: int = 200_000, seed: int = 0):
    """Single needle-in-a-haystack: retrieve a planted passkey."""
    rng = random.Random(seed)
    for i in range(n_examples):
        key = "".join(rng.choices(string.digits, k=7))
        needle = f" The secret passkey is {key}. Remember it. "
        body = _filler(rng, ctx_chars)
        pos = rng.randint(0, len(body) - 1)
        context = body[:pos] + needle + body[pos:]
        yield {
            "id": f"niah-{ctx_chars}-{i}",
            "context": context,
            "question": "What is the secret passkey mentioned in the document? Reply with the number only.",
            "answers": [key],
        }


def gen_multikey(n_examples: int = 50, ctx_chars: int = 200_000, n_keys: int = 8, seed: int = 1):
    """Multi-needle aggregation: sum planted values (RULER-style, harder)."""
    rng = random.Random(seed)
    for i in range(n_examples):
        vals = [rng.randint(10, 99) for _ in range(n_keys)]
        body = _filler(rng, ctx_chars)
        for j, v in enumerate(vals):
            pos = rng.randint(0, len(body) - 1)
            body = body[:pos] + f" Asset {j} has value {v} credits. " + body[pos:]
        yield {
            "id": f"multikey-{ctx_chars}-{i}",
            "context": body,
            "question": f"There are {n_keys} assets (Asset 0..{n_keys-1}), each with a value in credits. "
                        "What is the SUM of all asset values? Reply with the number only.",
            "answers": [str(sum(vals))],
        }


def load_longbench_v2(limit: int | None = None):
    """LongBench v2 (THUDM/LongBench-v2): multiple-choice over very long contexts.

    Cached as one JSONL by scripts/download_data.py. Examples are tagged with
    their `domain` as the `subset` (6 domains: single/multi-doc QA, long ICL,
    long-dialogue, code-repo, structured data) so score.py reports per-domain
    rows. `limit` is PER SUBSET (domain), matching the RULER loader. Scoring uses
    the `choice` metric (exact A/B/C/D match), not substring recall.
    """
    path = DATA_DIR / "longbench_v2.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts/download_data.py first.")
    per_subset = limit or 200
    seen: dict[str, int] = {}
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                subset = ex.get("domain") or "unknown"
                if seen.get(subset, 0) >= per_subset:
                    continue
                choices = "\n".join(f"({c}) {ex[f'choice_{c}']}" for c in ("A", "B", "C", "D") if ex.get(f"choice_{c}"))
                rec = {
                    "id": ex.get("_id", f"lb2-{i}"),
                    "subset": subset,
                    "metric": "choice",
                    "context": ex["context"],
                    "question": f"{ex['question']}\n{choices}\nAnswer with the letter (A/B/C/D) only.",
                    "answers": [ex["answer"]],
                }
            except Exception as e:  # skip one bad row, don't kill the whole task
                print(f"[datasets] skipping malformed longbench_v2 row {i}: {e}", file=sys.stderr)
                continue
            seen[subset] = seen.get(subset, 0) + 1
            yield rec


def load_longbench(limit: int | None = None):
    """LongBench v1 (THUDM/LongBench): 16 English subsets, each scored with its
    own task-specific metric (F1 / ROUGE-L / classification / retrieval / count /
    code edit-similarity — see longbench_metrics.DATASET2METRIC).

    Cached as one JSONL by scripts/download_data.py; rows carry the upstream
    fields {dataset, input, context, answers, all_classes}. `limit` is PER SUBSET
    (dataset), matching the RULER loader, so one run sweeps every subset. Each
    yielded example carries `metric` (so run_benchmark scores it correctly) and
    `all_classes` (needed by the classification metric for trec).
    """
    path = DATA_DIR / "longbench.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts/download_data.py first.")
    per_subset = limit or 200
    seen: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                subset = ex.get("dataset") or "unknown"
                k = seen.get(subset, 0)
                if k >= per_subset:
                    continue
                answers = ex.get("answers") or []
                rec = {
                    "id": ex.get("_id", f"{subset}-{k}"),
                    "subset": subset,
                    "metric": DATASET2METRIC.get(subset, "qa_f1"),
                    "all_classes": ex.get("all_classes"),
                    "context": ex["context"],
                    "question": ex.get("input", "Answer the question based on the document above."),
                    "answers": [str(a) for a in answers] if isinstance(answers, list) else [str(answers)],
                }
            except Exception as e:  # skip one bad row, don't kill the whole task
                print(f"[datasets] skipping malformed longbench row: {e}", file=sys.stderr)
                continue
            seen[subset] = k + 1
            yield rec


def load_oolong(limit: int | None = None):
    path = DATA_DIR / "oolong.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run slurm/download_data.sh on the login node first.")
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                ex = json.loads(line)
                rec = {
                    "id": ex.get("id", f"oolong-{i}"),
                    "context": ex["context"],
                    "question": ex["question"],
                    "answers": ex["answers"] if isinstance(ex.get("answers"), list) else [str(ex.get("answer", ""))],
                }
            except Exception as e:  # skip one bad row, don't kill the whole task
                print(f"[datasets] skipping malformed oolong row {i}: {e}", file=sys.stderr)
                continue
            yield rec


# LOFT RAG: 5 datasets x 3 context lengths. nq/hotpotqa/musique are single-value
# (one gold answer), qampari/quest are multi-value (a set of gold answers), which
# is what picks the metric — see longbench_metrics.
LOFT_MULTIVALUE = {"qampari", "quest"}


def _load_loft(name: str, limit: int | None = None):
    """LOFT (Long-Context Frontiers) RAG, read from a JSONL cached by
    scripts/download_data.py. `name` is the task stem ("loft32k"/"loft128k"/"loft1m").

    Five subsets per length (nq, hotpotqa, musique, qampari, quest), each tagged
    with `subset` so score.py breaks the table down per subset, exactly like
    RULER and LongBench. `limit` is PER SUBSET (upstream ships 100 test rows each).

    The cached `context` already carries LOFT's own instruction preamble (output
    format + the document corpus), so it is self-contained for the REPL. The
    `answer_prefix` ("Final Answer: ") is appended to the QUESTION rather than
    dropped, because run_benchmark never reads answer_prefix and without it the
    model has no cue to emit the list format the metrics expect.
    """
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts/download_data.py first.")
    per_subset = limit or 100
    seen: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                subset = ex.get("subset") or "unknown"
                k = seen.get(subset, 0)
                if k >= per_subset:
                    continue
                base = subset.rsplit("_", 1)[0]          # "nq_32k" -> "nq"
                question = ex.get("question", "")
                prefix = (ex.get("answer_prefix") or "").strip()
                if prefix:
                    question = f"{question}\n\n{prefix}"
                answers = ex.get("answers") or []
                rec = {
                    "id": f"{subset}-{k}",
                    "subset": subset,
                    "metric": "loft_coverage" if base in LOFT_MULTIVALUE else "loft_subspan_em",
                    "context": ex["context"],
                    "question": question,
                    "answers": [str(a) for a in answers] if isinstance(answers, list) else [str(answers)],
                }
            except Exception as e:  # skip one bad row, don't kill the whole task
                print(f"[datasets] skipping malformed {name} row: {e}", file=sys.stderr)
                continue
            seen[subset] = k + 1
            yield rec


def _load_ruler(name: str, limit: int | None = None):
    """RULER (xAlg-AI/att-hub-ruler-{16,32}k), read from a JSONL cached by
    slurm/download_data.sh. `name` is the task stem ("ruler16k" / "ruler32k").

    RULER ships 13 task SUBSETS (niah_single_*, niah_multikey_*, niah_multivalue,
    niah_multiquery, vt, cwe, fwe, qa_1, qa_2). Each example is tagged with its
    `subset` so the runner/scorer can break metrics down per subset.

    `limit` here means PER SUBSET (not a global first-N): we yield up to `limit`
    examples from every subset, so a single run exercises all of them. The cached
    rows use a stable schema with proper context/question separation:
        {"subset", "context", "question", "answer_prefix", "answers": [...]}
    """
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run slurm/download_data.sh on the login node first.")
    per_subset = limit or 50
    seen: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                subset = ex.get("subset") or ex.get("task") or "unknown"
                k = seen.get(subset, 0)
                if k >= per_subset:
                    continue
                outs = ex.get("answers", ex.get("outputs", ex.get("answer", "")))
                answers = outs if isinstance(outs, list) else [str(outs)]
                # question + answer_prefix (e.g. "...is") cues the expected answer format
                q = " ".join(x for x in (ex.get("question", ""), ex.get("answer_prefix", "")) if x).strip()
                rec = {
                    "id": f"{name}-{subset}-{k}",
                    "subset": subset,
                    "context": ex["context"],
                    "question": q or "Answer the query stated in the document above. Reply with the answer only.",
                    "answers": [str(a) for a in answers],
                }
            except Exception as e:  # skip one bad row, don't kill the whole task
                print(f"[datasets] skipping malformed {name} row: {e}", file=sys.stderr)
                continue
            seen[subset] = k + 1
            yield rec


TASKS = {
    "niah": lambda limit: gen_niah(n_examples=limit or 50),
    "niah-1m": lambda limit: gen_niah(n_examples=limit or 20, ctx_chars=1_000_000, seed=7),
    "multikey": lambda limit: gen_multikey(n_examples=limit or 50),
    "longbench": load_longbench,
    "longbench_v2": load_longbench_v2,
    "oolong": load_oolong,
    "ruler16k": lambda limit: _load_ruler("ruler16k", limit),
    "ruler32k": lambda limit: _load_ruler("ruler32k", limit),
    "loft32k": lambda limit: _load_loft("loft32k", limit),
    "loft128k": lambda limit: _load_loft("loft128k", limit),
    "loft1m": lambda limit: _load_loft("loft1m", limit),
}
