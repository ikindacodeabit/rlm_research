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

# --- OOLONG ---
# The OOLONG benchmark splits live on the Hugging Face Hub; the repo/config
# names have changed since release — check https://huggingface.co/datasets?search=oolong
# and the official RLM repo (github.com/alexzhang13/rlm) for the exact loader,
# then mirror the pattern above to write oolong.jsonl with fields:
#   {"id", "context", "question", "answers": [...]}.
EOF

echo "Done. Data in $RLM_DATA_DIR"
