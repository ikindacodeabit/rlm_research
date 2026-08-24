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
import subprocess
import sys
import urllib.request
import zipfile
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


def _loft_download(dataset: str = "nq", category: str = "retrieval") -> Path:
    """Fetch+unzip a LOFT dataset from the public GCS bucket (google-deepmind/loft).

    https://storage.googleapis.com/loft-bench/{category}/{dataset}.zip unzips
    (verified against the real zip, which flattens directly to the dataset name
    despite the README's ASCII tree implying a data/{category}/ prefix) to:
    {dataset}/{32k,128k,1m}/{corpus,dev_queries,test_queries,few_shot_queries}.jsonl
    """
    root = DATA_DIR / "loft"
    task_dir = root / dataset
    if task_dir.exists():
        return task_dir
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / f"{dataset}.zip"
    url = f"https://storage.googleapis.com/loft-bench/{category}/{dataset}.zip"
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    zip_path.unlink()
    if not task_dir.exists():
        raise FileNotFoundError(
            f"expected {task_dir} after extracting {url} — LOFT's zip layout may have changed"
        )
    return task_dir


def load_loft32k(limit: int | None = None, dataset: str = "nq"):
    """LOFT retrieval, 32k-token bucket (google-deepmind/loft).

    Context = every passage in corpus.jsonl joined with LOFT's own
    "ID: {pid} | TITLE: {title} | CONTENT: {passage}" format (see
    prompts/constants/common.py: CORPUS_FORMAT), so the RLM/vanilla harness sees
    exactly the same corpus-in-context text LOFT's own eval uses.
    """
    task_dir = _loft_download(dataset, "retrieval") / "32k"
    corpus_path = task_dir / "corpus.jsonl"
    queries_path = task_dir / "test_queries.jsonl"
    if not queries_path.exists():
        queries_path = task_dir / "dev_queries.jsonl"

    passages = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            passages.append(
                f"ID: {c['pid']} | TITLE: {c.get('title_text', '').strip()} | "
                f"CONTENT: {c['passage_text'].strip()}"
            )
    context = "\n\n".join(passages)

    n = 0
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            if limit and n >= limit:
                break
            q = json.loads(line)
            # answers/qrels are [[doc_id, relevance], ...] pairs (retrieval gold is
            # a document id, not free text) — keep only the relevant doc ids, since
            # that's the literal string the RLM/vanilla answer needs to contain.
            qrels = q.get("answers") or q.get("metadata", {}).get("qrels") or []
            doc_ids = [str(pair[0]) for pair in qrels if pair]
            yield {
                "id": f"loft32k-{dataset}-{q.get('qid', n)}",
                "context": context,
                "question": (
                    q.get("query_text", q.get("query", q.get("question", "")))
                    + "\nAnswer with the document ID (e.g. doc12345) of the single most "
                    "relevant passage above."
                ),
                "answers": doc_ids,
            }
            n += 1


def _ruler_generate(task: str, n_examples: int, ctx_tokens: int, tokenizer: str) -> list[dict]:
    """Vendor-call NVIDIA RULER's own orchestrator (scripts/data/prepare.py) so we
    reuse their exact needle-placement / haystack-sizing / template-wrapping logic
    instead of reimplementing it. Requires a checkout of NVIDIA/RULER at
    $RULER_REPO_DIR (git clone https://github.com/NVIDIA/RULER).

    IMPORTANT: prepare.py, not niah.py/qa.py directly — the per-task scripts take
    a --template arg that defaults to '' (empty). prepare.py is what looks up the
    real template string from scripts/synthetic.yaml + synthetic/constants.py and
    wraps it with the model chat template before invoking niah.py/qa.py. Calling
    niah.py/qa.py directly would silently generate blank-template garbage.

    prepare.py also needs `wonderwords` and nltk's `punkt`/`punkt_tab` data —
    ensure those are installed/downloaded before calling this.

    Linux/macOS only: prepare.py builds a shell command string containing the
    multi-line --template value and runs it via subprocess.run(..., shell=True).
    On native Windows that always shells out to cmd.exe (regardless of which
    shell launched Python), and cmd.exe truncates a double-quoted argument at
    its first embedded newline — silently producing a corrupt, truncated
    template with no error. Verified by reproducing on this checkout: output
    `input` came back as a ~130-char fragment cut off right before `{context}`.
    molab/Kaggle containers are Linux, so this is a non-issue there — this
    guard exists only to fail loudly instead of ever again silently emitting
    garbage examples on a Windows dev machine.
    """
    if sys.platform == "win32":
        raise RuntimeError(
            "RULER's scripts/data/prepare.py cannot run correctly on native Windows "
            "(embedded newline in --template gets truncated by cmd.exe under "
            "subprocess.run(shell=True), corrupting every generated example with no "
            "error). Run this on Linux/macOS/WSL — e.g. inside the molab/Kaggle "
            "notebook, not on the local Windows dev machine."
        )
    ruler_dir = Path(os.environ.get("RULER_REPO_DIR", DATA_DIR / "RULER"))
    if not ruler_dir.exists():
        raise FileNotFoundError(
            f"{ruler_dir} not found. Clone NVIDIA's generator first:\n"
            f"  git clone https://github.com/NVIDIA/RULER {ruler_dir}\n"
            "  pip install wonderwords nltk\n"
            '  python -c "import nltk; nltk.download(\'punkt\'); nltk.download(\'punkt_tab\')"'
        )
    if task not in {"niah_single_1", "niah_single_2", "niah_single_3",
                     "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
                     "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe",
                     "qa_1", "qa_2"}:
        raise ValueError(f"unsupported ruler task {task!r} — must be a name from scripts/synthetic.yaml")

    out_dir = DATA_DIR / "ruler_gen" / str(ctx_tokens)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / task / "validation.jsonl"
    if not out_path.exists():
        cmd = [
            sys.executable, str(ruler_dir / "scripts" / "data" / "prepare.py"),
            "--save_dir", str(out_dir),
            "--benchmark", "synthetic",
            "--task", task,
            "--tokenizer_path", tokenizer,
            "--tokenizer_type", "hf",
            "--max_seq_length", str(ctx_tokens),
            "--num_samples", str(n_examples),
            "--model_template_type", "base",
        ]
        subprocess.run(cmd, check=True, cwd=ruler_dir / "scripts" / "data")

    records = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records[:n_examples]


# Marker prepare.py's "base" template puts right before the rendered task_template
# (see scripts/template.py Templates['base']) — used only to strip that wrapper
# back off so `context` doesn't include RULER's own instruction boilerplate twice.
_RULER_QUESTION_MARKERS = {
    "qa": "\n\nQuestion: ",
    "niah": "\nWhat are all the special magic",
}


def load_ruler32k(limit: int | None = None, task: str = "niah_single_1",
                   tokenizer: str = "Qwen/Qwen3-4B-Instruct-2507"):
    """RULER (NVIDIA), generated at a 32768-token target length via RULER's own
    scripts/data/prepare.py ("ruler32k" = the RULER suite at --max_seq_length 32768).

    RULER's generator merges haystack+question into one "input" field (see
    scripts/data/synthetic/constants.py TASKS[...]['template']) — there is no
    native context/question split. We split on the last literal occurrence of
    the task's own question-lead-in text so the RLM harness gets a clean
    `context` (haystack only) and `question` (the actual ask), matching what
    RULER's own template renders.
    """
    n = limit or 50
    records = _ruler_generate(task, n, 32768, tokenizer)
    kind = "qa" if task.startswith("qa") else "niah" if task.startswith("niah") else None
    marker = _RULER_QUESTION_MARKERS.get(kind)
    for i, r in enumerate(records):
        text = r["input"]
        idx = text.rfind(marker) if marker else -1
        if idx == -1:
            raise ValueError(
                f"ruler32k[{task}] record {i}: expected marker {marker!r} not found in "
                f"generated 'input' ({len(text)} chars: {text!r}). This means RULER's "
                "generator produced something other than the expected template — "
                "possibly truncated (see _ruler_generate's Windows/shell=True note) — "
                "rather than a real context/question boundary we can silently guess at."
            )
        context, question = text[:idx], text[idx:].lstrip("\n")
        answers = r["outputs"] if isinstance(r["outputs"], list) else [r["outputs"]]
        yield {
            "id": f"ruler32k-{task}-{r.get('index', i)}",
            "context": context,
            "question": question,
            "answers": [str(a) for a in answers],
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


TASKS = {
    "niah": lambda limit: gen_niah(n_examples=limit or 50),
    "niah-1m": lambda limit: gen_niah(n_examples=limit or 20, ctx_chars=1_000_000, seed=7),
    "multikey": lambda limit: gen_multikey(n_examples=limit or 50),
    "longbench_v2": load_longbench_v2,
    "oolong": load_oolong,
    "loft32k": load_loft32k,
    "ruler32k": load_ruler32k,
}
