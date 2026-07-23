#!/bin/bash
#SBATCH --job-name=rlm-scratch
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --output=logs/rlm-scratch.%J.out
#SBATCH --error=logs/rlm-scratch.%J.err

# Scratchpad-RLM sweep on local Qwen3-8B (vLLM): the root gets a persistent
# note() scratchpad, run (a) unbounded — scratchpad only — and (b) under the
# same memory budgets as the eviction-only sweep, so the three-way comparison
# is: eviction-only (run_eval_budget.slurm) vs scratchpad-only vs
# scratchpad+budget. Writes to its own result dirs; nothing here disturbs the
# baselines from run_eval_local.slurm / run_eval_budget.slurm.
#
#   sbatch slurm/run_eval_scratchpad.slurm                          # everything
#   sbatch --export=ALL,THINK_MODES=nothink,BUDGETS=none  slurm/run_eval_scratchpad.slurm   # scratchpad-only cells
#   sbatch --export=ALL,THINK_MODES=think,BUDGETS="2048 8192" slurm/run_eval_scratchpad.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

source ~/venvs/rlm/bin/activate
export PYTHONUNBUFFERED=1                # flush python stdout per line so the .out tracks
                                        # progress live (see run_eval_budget.slurm)
export RLM_DATA_DIR="${RLM_DATA_DIR:-$HOME/rlm_data}"
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1                 # never try the internet
export NVIDIA_API_KEY=local-dummy       # client requires it to be set; vLLM ignores it
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL=Qwen/Qwen3-8B

# --- start ONE vLLM server for every cell ---
# Serve at Qwen3-8B's native 40960 (same reasoning as run_eval_budget.slurm:
# the top budget is 32768 and the model must generate on top of the prompt).
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

# --- the sweep: (none + budgets) x thinking x task (RLM+scratchpad only) ---
# BUDGETS may contain the literal "none" for the unbounded scratchpad-only cell.
# Results: results/<think>_sp_<none|b####>/{task}.rlm.{slug}.jsonl — the "sp"
# infix keeps these dirs disjoint from the eviction-only <think>_b#### dirs, so
# score.py reports each as its own variant row. APPEND-mode checkpointing makes
# the whole sweep resumable on requeue.
BUDGETS="${BUDGETS:-none 2048 4096 8192 16384 32768}"
THINK_MODES="${THINK_MODES:-nothink think}"
MAX_NOTES_TOKENS="${MAX_NOTES_TOKENS:-1024}"
for THINK in $THINK_MODES; do
  # think runs give the RLM root more generation room (Qwen3 <think> shares the
  # budget); safe on the 40960 window for the budgeted cells (<=32768 + 8192).
  if [ "$THINK" = nothink ]; then THINK_FLAG="--no-think"; MAXTOK_FLAG=""; else THINK_FLAG=""; MAXTOK_FLAG="--max-tokens 8192"; fi
  for BUDGET in $BUDGETS; do
    if [ "$BUDGET" = none ]; then
      BUDGET_FLAGS=""
      OUTDIR="results/${THINK}_sp_none"
    else
      BUDGET_FLAGS="--max-context-tokens $BUDGET --keep-recent-turns 3"
      OUTDIR="results/${THINK}_sp_b${BUDGET}"
    fi
    for TASK in niah multikey longbench_v2; do
      echo "=== $(date '+%F %T') :: $TASK / rlm+scratchpad / $THINK / b=$BUDGET ==="
      python benchmarks/run_benchmark.py \
        --task "$TASK" --mode rlm --limit 50 \
        --base-url http://localhost:8000/v1 \
        --root-model "$MODEL" --sub-model "$MODEL" \
        $THINK_FLAG $MAXTOK_FLAG \
        --rpm 600 \
        --max-steps 50 \
        --scratchpad --max-notes-tokens "$MAX_NOTES_TOKENS" \
        $BUDGET_FLAGS \
        --out "$OUTDIR" \
        || echo "WARN: $TASK/rlm+scratchpad/$THINK/b=$BUDGET failed, continuing"
    done
  done
done
echo "=== done ==="
