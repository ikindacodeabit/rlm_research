#!/bin/bash
#SBATCH --job-name=rlm-loft
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --output=logs/rlm-loft.%J.out
#SBATCH --error=logs/rlm-loft.%J.err

# NOTE: superseded by slurm/run_eval_task.slurm, which runs every task with
# IDENTICAL flags (this script's --vanilla-char-limit / results-dir naming differ
# from its siblings, which confounds cross-dataset comparison). Kept for
# reproducing earlier runs; use the unified driver for new campaigns.

# LOFT (Long-Context Frontiers) RAG: base RLM (unbounded) + vanilla, thinking
# on/off, over the 5 subsets of ONE context length. The loft loader samples PER
# SUBSET (--limit is per-subset) and tags each record with its `subset`, so
# score.py reports one row per (subset, mode, thinking).
#
# Single-value subsets (nq/hotpotqa/musique) score subspan-EM; multi-value ones
# (qampari/quest) score coverage — see benchmarks/longbench_metrics.py.
#
# Download first on the login node:
#   DATASETS=loft32k,loft128k bash slurm/download_data.sh
#
# Pick the length with TASK (default loft32k):
#   TASK=loft128k sbatch slurm/run_eval_loft.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

# Refuse to start if another rlm-loft job is already running — two jobs writing
# the same results/*.jsonl duplicate all remaining work (load_done is read ONCE
# per cell, so the second job cannot see the first job's in-flight rows).
if squeue -u "$USER" -h -n rlm-loft -o %i | grep -qv "^${SLURM_JOB_ID}$"; then
  echo "another rlm-loft job is already running; exiting"; exit 1
fi

source ~/venvs/rlm/bin/activate
export PYTHONUNBUFFERED=1                # per-line flush so the .out tracks progress live
export RLM_DATA_DIR="${RLM_DATA_DIR:-$HOME/rlm_data}"
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1                 # never try the internet
export NVIDIA_API_KEY=local-dummy       # client requires it to be set; vLLM ignores it
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL=Qwen/Qwen3-8B
TASK="${TASK:-loft32k}"

# --- start ONE vLLM server for every cell (Qwen3-8B native 40960 window) ---
vllm serve "$MODEL" --served-model-name "$MODEL" --port 8000 --max-model-len 40960 \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.90 --enforce-eager -O0 > logs/vllm.$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

# wait for the server to come up (model load takes a few minutes)
for i in $(seq 1 60); do
  curl -s -o /dev/null http://localhost:8000/v1/models && break
  sleep 10
done
curl -s http://localhost:8000/v1/models | head -c 200 || { echo "vLLM failed to start, see logs/vllm.$SLURM_JOB_ID.log"; exit 1; }
echo "vLLM up."

# --- 2x2 grid: thinking on/off x vanilla/rlm, base RLM (no --max-context-tokens).
# --limit is PER SUBSET (upstream ships 100 test rows each; 25 keeps one length
# inside the walltime). Separate dirs per thinking condition so the append+skip
# checkpointing never collides: results/{task}_{think,nothink}.
# --vanilla-char-limit 100000 (~33k tok worst case) keeps the vanilla prompt
# inside the 40960 window; LOFT contexts run 180 KB at 32k and ~720 KB at 128k,
# so vanilla is honestly truncated — RLM reads the full corpus from the REPL.
LIMIT="${LIMIT:-25}"
for THINK in nothink think; do
  if [ "$THINK" = nothink ]; then THINK_FLAG="--no-think"; else THINK_FLAG=""; fi
  OUTDIR="results/${TASK}_$THINK"
  for MODE in vanilla rlm; do
    echo "=== $(date '+%F %T') :: $TASK / $MODE / $THINK (limit=$LIMIT per subset) ==="
    # more generation room for the RLM root on think runs (see run_eval_local.slurm);
    # vanilla stays at 4096 — its large prompt + max_tokens must fit the window.
    if [ "$THINK" = think ] && [ "$MODE" = rlm ]; then MAXTOK_FLAG="--max-tokens 8192"; else MAXTOK_FLAG=""; fi
    python benchmarks/run_benchmark.py \
      --task "$TASK" --mode "$MODE" --limit "$LIMIT" \
      --base-url http://localhost:8000/v1 \
      --root-model "$MODEL" --sub-model "$MODEL" \
      $THINK_FLAG $MAXTOK_FLAG \
      --rpm 600 \
      --max-steps 50 \
      --vanilla-char-limit 100000 \
      --out "$OUTDIR" \
      || echo "WARN: $TASK/$MODE/$THINK failed, continuing"
  done
done
echo "=== done ==="
