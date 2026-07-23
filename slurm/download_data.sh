#!/bin/bash
# Run on the LOGIN node (needs internet). Caches benchmark data to $RLM_DATA_DIR.
#
# Thin wrapper over the single, cross-platform downloader
# (scripts/download_data.py) so the Prajna and Mac data paths never diverge — in
# particular LongBench-v2 is fetched via hf_hub_download(data.json), which is the
# method verified to work (plain load_dataset has no parquet/script to load).
#
# Usage: bash slurm/download_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."                          # repo root, so scripts/ resolves

export RLM_DATA_DIR="${RLM_DATA_DIR:-$HOME/rlm_data}"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
mkdir -p "$RLM_DATA_DIR"

source ~/venvs/rlm/bin/activate

# What the SLURM eval scripts consume: longbench_v2 (run_eval_local / _budget /
# _scratchpad) and ruler16k/ruler32k (run_eval_ruler). niah/multikey are generated
# in-process (no download). longbench (v1) is fetched too — harmless, and it lets
# the same cache serve the Mac all-subset harness. Override with DATASETS, e.g.
#   DATASETS=ruler16k,longbench_v2 bash slurm/download_data.sh
DATASETS="${DATASETS:-ruler16k,ruler32k,longbench,longbench_v2}"
python scripts/download_data.py --only "$DATASETS"

# --- OOLONG (still TODO) ---
# The OOLONG splits' HF repo/config names have changed since release; add an
# `oolong` job to scripts/download_data.py (writing oolong.jsonl with fields
# {"id","context","question","answers":[...]}) once the exact loader is pinned.
# See github.com/alexzhang13/rlm and https://huggingface.co/datasets?search=oolong.

echo "Done. Data in $RLM_DATA_DIR"
