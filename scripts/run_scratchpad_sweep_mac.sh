#!/bin/bash
# Scratchpad-RLM sweep on NVIDIA NIM from a Mac: the root gets a persistent
# note() scratchpad, run (a) unbounded — scratchpad only — and (b) under the
# same memory budgets as scripts/run_budget_sweep_mac.sh, giving a three-way
# comparison: eviction-only vs scratchpad-only vs scratchpad+budget.
# Mirrors slurm/run_eval_scratchpad.slurm, but via the hosted NIM catalog
# instead of a local vLLM server.
#
# Resumable — re-run after an interruption and it skips examples already in
# $OUTDIR/sp_<none|b####>/*.jsonl.
#
# Prereqs:
#   export NVIDIA_API_KEY=nvapi-...            # from build.nvidia.com (or .env)
#   python scripts/download_data.py            # caches data to $RLM_DATA_DIR
#
# Usage:
#   bash scripts/run_scratchpad_sweep_mac.sh                        # full sweep
#   LIMIT=2 BUDGETS="none" bash scripts/run_scratchpad_sweep_mac.sh # smoke: scratchpad only
#   TASKS="longbench" BUDGETS="none 8192" bash scripts/run_scratchpad_sweep_mac.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Pick up the key from a local, gitignored .env if it isn't already exported.
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
BUDGETS="${BUDGETS:-none 2048 4096 8192 16384 32768}"  # "none" = scratchpad only
KEEP_RECENT_TURNS="${KEEP_RECENT_TURNS:-3}"
MAX_NOTES_TOKENS="${MAX_NOTES_TOKENS:-1024}"
LIMIT="${LIMIT:-20}"                 # PER SUBSET
RPM="${RPM:-35}"                     # NIM free tier ~40 rpm; raise if your tier allows
MAX_STEPS="${MAX_STEPS:-30}"         # RLM code turns per example
OUTDIR="${OUTDIR:-results/qwen3_scratchpad_sweep}"

mkdir -p "$OUTDIR" logs
LOG="logs/run_scratchpad_sweep_mac.$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

{
echo "=== config ==="
echo "root=$ROOT_MODEL sub=$SUB_MODEL"
echo "tasks='$TASKS' budgets='$BUDGETS' keep_recent_turns=$KEEP_RECENT_TURNS max_notes_tokens=$MAX_NOTES_TOKENS"
echo "limit=$LIMIT/subset rpm=$RPM max_steps=$MAX_STEPS"
echo "data=$RLM_DATA_DIR out=$OUTDIR"

for BUDGET in $BUDGETS; do
  if [ "$BUDGET" = none ]; then
    BUDGET_FLAGS=""
    CELL="sp_none"
  else
    BUDGET_FLAGS="--max-context-tokens $BUDGET --keep-recent-turns $KEEP_RECENT_TURNS"
    CELL="sp_b${BUDGET}"
  fi
  for TASK in $TASKS; do
    echo ""
    echo "=== $(date '+%F %T') :: $TASK scratchpad budget=$BUDGET (rlm, limit=$LIMIT/subset) ==="
    "$PY" benchmarks/run_benchmark.py \
      --task "$TASK" --mode rlm --limit "$LIMIT" \
      --base-url https://integrate.api.nvidia.com/v1 \
      --root-model "$ROOT_MODEL" --sub-model "$SUB_MODEL" \
      --rpm "$RPM" --max-steps "$MAX_STEPS" \
      --scratchpad --max-notes-tokens "$MAX_NOTES_TOKENS" \
      $BUDGET_FLAGS \
      --out "$OUTDIR/$CELL" \
      || echo "WARN: $TASK budget=$BUDGET failed, continuing"
  done
done

echo ""
echo "=== $(date '+%F %T') :: scoring -> $OUTDIR/comparison.csv ==="
"$PY" benchmarks/score.py "$OUTDIR" --csv "$OUTDIR/comparison.csv"
echo "=== done ==="
} 2>&1 | tee "$LOG"
