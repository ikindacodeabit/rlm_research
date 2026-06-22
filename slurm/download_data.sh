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

# --- RULER 32k ---
# Prebuilt mirror on the Hub, one config per sequence length; we take 32768.
# Each row carries the synthetic prompt (`input`), the gold answer(s) (`outputs`)
# and the task SUBSET (`task`). We normalise to a stable schema so the loader
# (benchmarks/datasets.py:load_ruler32k) doesn't depend on upstream column names.
print("Downloading RULER 32k ...")
rds = load_dataset("simonjegou/ruler", "32768")
split = next(iter(rds.keys()))          # single split; name varies by repo
rows = rds[split]
def _col(ex, *names, default=""):
    for nm in names:
        if nm in ex and ex[nm] is not None:
            return ex[nm]
    return default
with open(f"{out}/ruler32k.jsonl", "w") as f:
    subsets = {}
    for i, ex in enumerate(rows):
        sub = _col(ex, "task", "subset", default="unknown")
        outs = _col(ex, "outputs", "answers", "answer", default="")
        f.write(json.dumps({
            "index": _col(ex, "index", default=i),
            "subset": sub,
            "input": _col(ex, "input", "context", "prompt"),
            "outputs": outs if isinstance(outs, list) else [str(outs)],
        }) + "\n")
        subsets[sub] = subsets.get(sub, 0) + 1
print(f"RULER 32k: {len(rows)} examples across {len(subsets)} subsets: "
      + ", ".join(f"{k}={v}" for k, v in sorted(subsets.items())))

# --- OOLONG ---
# The OOLONG benchmark splits live on the Hugging Face Hub; the repo/config
# names have changed since release — check https://huggingface.co/datasets?search=oolong
# and the official RLM repo (github.com/alexzhang13/rlm) for the exact loader,
# then mirror the pattern above to write oolong.jsonl with fields:
#   {"id", "context", "question", "answers": [...]}.
EOF

echo "Done. Data in $RLM_DATA_DIR"
