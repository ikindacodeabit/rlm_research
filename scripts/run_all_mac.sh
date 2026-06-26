#!/bin/bash
# Run RULER-32k + LongBench v1 + LongBench v2 (every subset) on NVIDIA NIM from a
# Mac — vanilla vs RLM — then aggregate a per-subset comparison table + CSV.
#
# No GPU/SLURM/vLLM: everything is OpenAI-compatible calls to the hosted NIM
# catalog (https://integrate.api.nvidia.com/v1). Resumable — re-run after an
# interruption and it skips examples already in results/mac_nim/*.jsonl.
#
# Prereqs:
#   export NVIDIA_API_KEY=nvapi-...            # from build.nvidia.com
#   python scripts/download_data.py           # caches data to $RLM_DATA_DIR
#
# Usage:
#   bash scripts/run_all_mac.sh                # all 3 benchmarks, defaults
#   LIMIT=5 bash scripts/run_all_mac.sh        # quick smoke
#   TASKS="longbench" RPM=120 bash scripts/run_all_mac.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Pick up the key from a local, gitignored .env if it isn't already exported
# (handy because each shell here starts fresh, so a one-off `export` won't stick).
if [ -z "${NVIDIA_API_KEY:-}" ] && [ -f .env ]; then
  set -a; . ./.env; set +a
fi
if [ -z "${NVIDIA_API_KEY:-}" ]; then
  echo "ERROR: NVIDIA_API_KEY not set. Either 'export NVIDIA_API_KEY=nvapi-...'" >&2
  echo "       or create a .env file containing: NVIDIA_API_KEY=nvapi-..." >&2
  echo "       (get a free key at https://build.nvidia.com)" >&2
  exit 1
fi

PY="./.venv/bin/python"
[ -x "$PY" ] || PY="python3"

export RLM_DATA_DIR="${RLM_DATA_DIR:-$HOME/rlm_data}"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export PYTHONUNBUFFERED=1

# --- knobs (override via env) ---
ROOT_MODEL="${ROOT_MODEL:-meta/llama-3.3-70b-instruct}"   # vanilla baseline + RLM root
SUB_MODEL="${SUB_MODEL:-meta/llama-3.1-8b-instruct}"      # cheap RLM recursive calls
TASKS="${TASKS:-ruler32k longbench longbench_v2}"
LIMIT="${LIMIT:-20}"                 # PER SUBSET
RPM="${RPM:-35}"                     # NIM free tier ~40 rpm; raise if your tier allows
MAX_STEPS="${MAX_STEPS:-30}"         # RLM code turns per example
VANILLA_CHAR_LIMIT="${VANILLA_CHAR_LIMIT:-300000}"  # ~90k tok; keeps vanilla under Llama-70B's 128k window
OUTDIR="${OUTDIR:-results/mac_nim}"

mkdir -p "$OUTDIR" logs
LOG="logs/run_all_mac.$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

{
echo "=== config ==="
echo "root=$ROOT_MODEL sub=$SUB_MODEL"
echo "tasks='$TASKS' limit=$LIMIT/subset rpm=$RPM max_steps=$MAX_STEPS vanilla_char_limit=$VANILLA_CHAR_LIMIT"
echo "data=$RLM_DATA_DIR out=$OUTDIR"

for TASK in $TASKS; do
  echo ""
  echo "=== $(date '+%F %T') :: $TASK (vanilla + rlm, limit=$LIMIT/subset) ==="
  "$PY" benchmarks/run_benchmark.py \
    --task "$TASK" --mode both --limit "$LIMIT" \
    --base-url https://integrate.api.nvidia.com/v1 \
    --root-model "$ROOT_MODEL" --sub-model "$SUB_MODEL" \
    --rpm "$RPM" --max-steps "$MAX_STEPS" \
    --vanilla-char-limit "$VANILLA_CHAR_LIMIT" \
    --out "$OUTDIR" \
    || echo "WARN: $TASK failed, continuing"
done

echo ""
echo "=== $(date '+%F %T') :: scoring -> $OUTDIR/comparison.csv ==="
"$PY" benchmarks/score.py "$OUTDIR" --csv "$OUTDIR/comparison.csv"
echo "=== done ==="
} 2>&1 | tee "$LOG"
