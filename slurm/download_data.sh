#!/bin/bash
# Run on the LOGIN node (needs internet). Caches benchmark data to scratch.
# Usage: bash slurm/download_data.sh
set -euo pipefail

export RLM_DATA_DIR="${RLM_DATA_DIR:-$HOME/rlm_data}"
mkdir -p "$RLM_DATA_DIR"
export HF_HOME="$HOME/hf_cache"

source ~/venvs/rlm/bin/activate

python - <<'EOF'
import json, os
from datasets import load_dataset

out = os.environ["RLM_DATA_DIR"]

# --- LongBench v2 ---
print("Downloading LongBench v2 ...")
ds = load_dataset("THUDM/LongBench-v2", split="train")
with open(f"{out}/longbench_v2.jsonl", "w") as f:
    for ex in ds:
        f.write(json.dumps(dict(ex)) + "\n")
print(f"LongBench v2: {len(ds)} examples")

# --- RULER 16k & 32k ---
# Source: xAlg-AI/att-hub-ruler-{16,32}k (the sparse-attention-hub RULER builds).
# The 13 RULER subsets are separate CONFIGS, each with a split of the same name,
# and rows carry proper context/question/answer_prefix/answer fields. We iterate
# every subset and normalise to one stable schema so the loader
# (benchmarks/datasets.py:_load_ruler) is decoupled from upstream column names.
# (16k fits Qwen3-8B's 40960 window for vanilla; 32k overflows on dense subsets
# like cwe, so vanilla there is truncated — see slurm/run_eval_ruler.slurm.)
RULER_SUBSETS = ["cwe", "fwe", "niah_multikey_1", "niah_multikey_2",
                 "niah_multikey_3", "niah_multiquery", "niah_multivalue",
                 "niah_single_1", "niah_single_2", "niah_single_3",
                 "qa_1", "qa_2", "vt"]
def _col(ex, *names, default=""):
    for nm in names:
        if nm in ex and ex[nm] is not None:
            return ex[nm]
    return default
for stem, repo in [("ruler16k", "xAlg-AI/att-hub-ruler-16k"),
                   ("ruler32k", "xAlg-AI/att-hub-ruler-32k")]:
    print(f"Downloading {stem} from {repo} ...")
    with open(f"{out}/{stem}.jsonl", "w") as f:
        counts = {}
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
    print(f"{stem}: {sum(counts.values())} examples across {len(counts)} subsets: "
          + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

# --- OOLONG ---
# The OOLONG benchmark splits live on the Hugging Face Hub; the repo/config
# names have changed since release — check https://huggingface.co/datasets?search=oolong
# and the official RLM repo (github.com/alexzhang13/rlm) for the exact loader,
# then mirror the pattern above to write oolong.jsonl with fields:
#   {"id", "context", "question", "answers": [...]}.
EOF

echo "Done. Data in $RLM_DATA_DIR"
