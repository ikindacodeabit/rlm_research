#!/usr/bin/env python3
"""Download the three benchmarks to $RLM_DATA_DIR as normalised JSONL (Mac/NIM).

Unlike slurm/download_data.sh (login-node + Prajna paths), this runs anywhere
with internet — your Mac. It caches:
  * ruler16k.jsonl      RULER-16k, 13 subsets   (xAlg-AI/att-hub-ruler-16k)
  * ruler32k.jsonl      RULER-32k, 13 subsets   (xAlg-AI/att-hub-ruler-32k)
  * longbench.jsonl     LongBench v1, 16 EN subsets (THUDM/LongBench)
  * longbench_v2.jsonl  LongBench v2, 6 domains (THUDM/LongBench-v2)
  * loft32k.jsonl       LOFT RAG @32k, 5 subsets (f20180301/loft-rag-*-32k)
  * loft128k.jsonl      LOFT RAG @128k, 5 subsets (f20180301/loft-rag-*-128k)

The loaders in benchmarks/datasets.py read exactly these files. Re-running skips
files that already exist unless --force is given. Choose a subset of benchmarks
with --only ruler16k,ruler32k,longbench,longbench_v2. This is the single,
cross-platform downloader — slurm/download_data.sh calls it on the login node.

Usage:
  python scripts/download_data.py
  python scripts/download_data.py --only longbench --force
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("RLM_DATA_DIR", os.path.expanduser("~/rlm_data")))

# 16 English LongBench v1 subsets (the canonical English benchmark). Chinese
# subsets are omitted: they need a Chinese-capable model + jieba-based metrics.
LONGBENCH_EN = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]

RULER_SUBSETS = [
    "cwe", "fwe", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue", "niah_single_1", "niah_single_2",
    "niah_single_3", "qa_1", "qa_2", "vt",
]


def _col(ex, *names, default=""):
    for nm in names:
        if nm in ex and ex[nm] is not None:
            return ex[nm]
    return default


RULER_REPOS = {
    "ruler16k": "xAlg-AI/att-hub-ruler-16k",
    "ruler32k": "xAlg-AI/att-hub-ruler-32k",
}


def _download_ruler(out: Path, stem: str):
    from datasets import load_dataset
    repo = RULER_REPOS[stem]
    print(f"Downloading {stem} from {repo} ...")
    counts = {}
    with open(out, "w") as f:
        for sub in RULER_SUBSETS:
            try:
                rows = load_dataset(repo, sub, split=sub)
            except Exception:                   # split name may differ; take the first
                d = load_dataset(repo, sub)
                rows = d[next(iter(d.keys()))]
            for ex in rows:
                ans = _col(ex, "answer", "outputs", "answers", default="")
                f.write(json.dumps({
                    "subset": sub,
                    "context": _col(ex, "context", "input"),
                    "question": _col(ex, "question"),
                    "answer_prefix": _col(ex, "answer_prefix"),
                    "answers": ans if isinstance(ans, list) else [str(ans)],
                }) + "\n")
            counts[sub] = len(rows)
    print(f"  {stem}: {sum(counts.values())} rows / {len(counts)} subsets")


# LOFT RAG (Long-Context Frontiers). sparse-attention-hub's benchmark/loft
# resolves each task to "f20180301/loft-rag-{dataset}-{length}"; we mirror that
# so the two harnesses read identical data. 100 test rows per subset.
LOFT_DATASETS = ["nq", "hotpotqa", "musique", "qampari", "quest"]

LOFT_LENGTHS = {"loft32k": "32k", "loft128k": "128k", "loft1m": "1m"}


def _download_loft(out: Path, stem: str):
    from datasets import load_dataset
    length = LOFT_LENGTHS[stem]
    print(f"Downloading {stem} (LOFT RAG @ {length}) ...")
    counts = {}
    with open(out, "w") as f:
        for ds in LOFT_DATASETS:
            repo = f"f20180301/loft-rag-{ds}-{length}"
            try:
                rows = load_dataset(repo, split="test")
            except Exception as e:
                print(f"  WARN: {repo} unavailable ({e}); skipping")
                continue
            for ex in rows:
                ans = _col(ex, "answers", "answer", default=[])
                f.write(json.dumps({
                    "subset": _col(ex, "task", default=f"{ds}_{length}"),
                    "context": _col(ex, "context"),
                    "question": _col(ex, "question"),
                    "answer_prefix": _col(ex, "answer_prefix"),
                    "answers": ans if isinstance(ans, list) else [str(ans)],
                }) + "\n")
            counts[ds] = len(rows)
    print(f"  {stem}: {sum(counts.values())} rows / {len(counts)} subsets")


def download_longbench(out: Path):
    # datasets>=3 dropped script loaders, and THUDM/LongBench ships a loader
    # script; its data lives in data.zip as data/{subset}.jsonl. Read directly.
    import zipfile
    from huggingface_hub import hf_hub_download
    print("Downloading LongBench v1 (16 English subsets) ...")
    zip_path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    counts = {}
    with zipfile.ZipFile(zip_path) as z, open(out, "w") as fout:
        for sub in LONGBENCH_EN:
            member = f"data/{sub}.jsonl"
            n = 0
            with z.open(member) as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    ex = json.loads(raw)
                    fout.write(json.dumps({
                        "_id": ex.get("_id"),
                        "dataset": sub,
                        "input": ex.get("input", ""),
                        "context": ex.get("context", ""),
                        "answers": ex.get("answers", []),
                        "all_classes": ex.get("all_classes", []),
                    }) + "\n")
                    n += 1
            counts[sub] = n
            print(f"  {sub}: {n}")
    print(f"  longbench: {sum(counts.values())} rows / {len(counts)} subsets")


def download_longbench_v2(out: Path):
    # THUDM/LongBench-v2 ships a single data.json array (no parquet/script).
    from huggingface_hub import hf_hub_download
    print("Downloading LongBench v2 ...")
    src = hf_hub_download("THUDM/LongBench-v2", "data.json", repo_type="dataset")
    with open(src) as f:
        rows = json.load(f)
    domains = {}
    with open(out, "w") as fout:
        for ex in rows:
            domains[ex.get("domain", "?")] = domains.get(ex.get("domain", "?"), 0) + 1
            fout.write(json.dumps(ex) + "\n")
    print(f"  longbench_v2: {len(rows)} rows across {len(domains)} domains: "
          + ", ".join(f"{k}={v}" for k, v in sorted(domains.items())))


JOBS = {
    "ruler16k": ("ruler16k.jsonl", lambda out: _download_ruler(out, "ruler16k")),
    "ruler32k": ("ruler32k.jsonl", lambda out: _download_ruler(out, "ruler32k")),
    "longbench": ("longbench.jsonl", download_longbench),
    "longbench_v2": ("longbench_v2.jsonl", download_longbench_v2),
    "loft32k": ("loft32k.jsonl", lambda out: _download_loft(out, "loft32k")),
    "loft128k": ("loft128k.jsonl", lambda out: _download_loft(out, "loft128k")),
    # ~5.6 MB of context per row x 500 rows: ~2.8 GB, and far past any window we
    # serve. Not in the default set — request it explicitly with --only loft1m.
    "loft1m": ("loft1m.jsonl", lambda out: _download_loft(out, "loft1m")),
}

_DEFAULT_JOBS = [j for j in JOBS if j != "loft1m"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma list of: ruler16k,ruler32k,longbench,longbench_v2,"
                         "loft32k,loft128k,loft1m (default: all but loft1m)")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    wanted = args.only.split(",") if args.only else list(_DEFAULT_JOBS)
    print(f"RLM_DATA_DIR = {DATA_DIR}")
    for name in wanted:
        fname, fn = JOBS[name]
        path = DATA_DIR / fname
        if path.exists() and not args.force:
            print(f"skip {name} (exists: {path}; use --force to refresh)")
            continue
        fn(path)
    print("Done.")


if __name__ == "__main__":
    main()
