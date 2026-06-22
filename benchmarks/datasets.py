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
from pathlib import Path

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
    """Reads the JSONL cached by slurm/download_data.sh."""
    path = DATA_DIR / "longbench_v2.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run slurm/download_data.sh on the login node first.")
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            ex = json.loads(line)
            choices = "\n".join(f"({k}) {ex[k]}" for k in ("choice_A", "choice_B", "choice_C", "choice_D") if ex.get(k))
            yield {
                "id": ex.get("_id", f"lb2-{i}"),
                "context": ex["context"],
                "question": f"{ex['question']}\n{choices}\nAnswer with the letter (A/B/C/D) only.",
                "answers": [ex["answer"]],
            }


def load_oolong(limit: int | None = None):
    path = DATA_DIR / "oolong.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run slurm/download_data.sh on the login node first.")
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            ex = json.loads(line)
            yield {
                "id": ex.get("id", f"oolong-{i}"),
                "context": ex["context"],
                "question": ex["question"],
                "answers": ex["answers"] if isinstance(ex.get("answers"), list) else [str(ex.get("answer", ""))],
            }


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
            ex = json.loads(line)
            subset = ex.get("subset") or ex.get("task") or "unknown"
            k = seen.get(subset, 0)
            if k >= per_subset:
                continue
            seen[subset] = k + 1
            outs = ex.get("answers", ex.get("outputs", ex.get("answer", "")))
            answers = outs if isinstance(outs, list) else [str(outs)]
            # question + answer_prefix (e.g. "...is") cues the expected answer format
            q = " ".join(x for x in (ex.get("question", ""), ex.get("answer_prefix", "")) if x).strip()
            yield {
                "id": f"{name}-{subset}-{k}",
                "subset": subset,
                "context": ex["context"],
                "question": q or "Answer the query stated in the document above. Reply with the answer only.",
                "answers": [str(a) for a in answers],
            }


TASKS = {
    "niah": lambda limit: gen_niah(n_examples=limit or 50),
    "niah-1m": lambda limit: gen_niah(n_examples=limit or 20, ctx_chars=1_000_000, seed=7),
    "multikey": lambda limit: gen_multikey(n_examples=limit or 50),
    "longbench_v2": load_longbench_v2,
    "oolong": load_oolong,
    "ruler16k": lambda limit: _load_ruler("ruler16k", limit),
    "ruler32k": lambda limit: _load_ruler("ruler32k", limit),
}
