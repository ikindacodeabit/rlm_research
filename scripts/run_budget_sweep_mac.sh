#!/bin/bash
# Budgeted-RLM sweep: RULER-32k + LongBench v1 + LongBench v2 (every subset) on
# NVIDIA NIM from a Mac, with the RLM root's MEMORY BUDGET (--max-context-tokens)
# swept across several levels. Mirrors slurm/run_eval_budget.slurm's sweep, but
# via the hosted NIM catalog instead of a local vLLM server.
#
# Vanilla is budget-invariant (it doesn't use MemoryBudget), so this script only
# runs --mode rlm. For an unbudgeted vanilla baseline, use scripts/run_all_mac.sh.
#
# No GPU/SLURM/vLLM: everything is OpenAI-compatible calls to the hosted NIM
# catalog (https://integrate.api.nvidia.com/v1). Resumable — re-run after an
# interruption and it skips examples already in $OUTDIR/b<BUDGET>/*.jsonl.
#
# Prereqs:
#   export NVIDIA_API_KEY=nvapi-...            # from build.nvidia.com
#   python scripts/download_data.py           # caches data to $RLM_DATA_DIR
#
# Usage:
#   bash scripts/run_budget_sweep_mac.sh                     # full sweep, defaults
#   LIMIT=2 BUDGETS="4096" bash scripts/run_budget_sweep_mac.sh   # smoke test
#   TASKS="longbench" BUDGETS="8192 16384" bash scripts/run_budget_sweep_mac.sh
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
ROOT_MODEL="${ROOT_MODEL:-qwen/qwen3-next-80b-a3b-instruct}"
SUB_MODEL="${SUB_MODEL:-qwen/qwen3-next-80b-a3b-instruct}"
TASKS="${TASKS:-ruler32k longbench longbench_v2}"
BUDGETS="${BUDGETS:-2048 4096 8192 16384 32768}"   # RLM root max-context-tokens
KEEP_RECENT_TURNS="${KEEP_RECENT_TURNS:-3}"
LIMIT="${LIMIT:-20}"                 # PER SUBSET
RPM="${RPM:-35}"                     # NIM free tier ~40 rpm; raise if your tier allows
MAX_STEPS="${MAX_STEPS:-30}"         # RLM code turns per example
OUTDIR="${OUTDIR:-results/qwen3_budget_sweep}"

mkdir -p "$OUTDIR" logs
LOG="logs/run_budget_sweep_mac.$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

{
echo "=== config ==="
echo "root=$ROOT_MODEL sub=$SUB_MODEL"
echo "tasks='$TASKS' budgets='$BUDGETS' keep_recent_turns=$KEEP_RECENT_TURNS"
echo "limit=$LIMIT/subset rpm=$RPM max_steps=$MAX_STEPS"
echo "data=$RLM_DATA_DIR out=$OUTDIR"

for BUDGET in $BUDGETS; do
  for TASK in $TASKS; do
    echo ""
    echo "=== $(date '+%F %T') :: $TASK budget=$BUDGET (rlm, limit=$LIMIT/subset) ==="
    "$PY" benchmarks/run_benchmark.py \
      --task "$TASK" --mode rlm --limit "$LIMIT" \
      --base-url https://integrate.api.nvidia.com/v1 \
      --root-model "$ROOT_MODEL" --sub-model "$SUB_MODEL" \
      --rpm "$RPM" --max-steps "$MAX_STEPS" \
      --max-context-tokens "$BUDGET" --keep-recent-turns "$KEEP_RECENT_TURNS" \
      --out "$OUTDIR/b${BUDGET}" \
      || echo "WARN: $TASK budget=$BUDGET failed, continuing"
  done
done

echo ""
echo "=== $(date '+%F %T') :: scoring -> $OUTDIR/comparison.csv ==="
"$PY" benchmarks/score.py "$OUTDIR" --csv "$OUTDIR/comparison.csv"
echo "=== done ==="
} 2>&1 | tee "$LOG"
