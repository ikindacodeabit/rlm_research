#!/bin/bash
#SBATCH --job-name=rlm-budget
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --output=logs/rlm-budget.%J.out
#SBATCH --error=logs/rlm-budget.%J.err

# Budgeted-RLM sweep: RLM under a hard ROOT context-token cap (eviction-only),
# swept across several budget levels x thinking on/off x every task. Base RLM
# (unbounded) and the vanilla baselines were measured separately by
# run_eval_local.slurm; this script ONLY runs the budgeted RLM cells, into
# their own result dirs, so nothing here disturbs those baselines.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

source ~/venvs/rlm/bin/activate
export PYTHONUNBUFFERED=1                # flush python stdout per line so the .out tracks
                                        # progress live (non-TTY stdout is block-buffered,
                                        # so per-item lines/summaries otherwise only appear
                                        # when each cell's process exits — and are lost if
                                        # it's killed/requeued mid-cell)
export RLM_DATA_DIR="${RLM_DATA_DIR:-$HOME/rlm_data}"
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1                 # never try the internet
export NVIDIA_API_KEY=local-dummy       # client requires it to be set; vLLM ignores it
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL=Qwen/Qwen3-8B

# --- start ONE vLLM server for every cell ---
# NOTE on --max-model-len: the MemoryBudget caps the ROOT *prompt* (system +
# begin + recent turns). The model still has to GENERATE on top of that prompt,
# so the server window must exceed the largest budget. Our top budget is 32768,
# so we serve at Qwen3-8B's native 40960 to leave ~8k tokens of generation
# headroom (vs 32768 in run_eval_local.slurm, which had no large-prompt cap).
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

# --- the sweep: budget x thinking x task (RLM only) ---
# Results separated per (thinking, budget) so score.py reports each as its own
# variant row: results/<think>_b<budget>/{task}.rlm.{slug}.jsonl. APPEND-mode
# checkpointing makes the whole sweep resumable on requeue.
# Both selectable via env so the two halves can run as separate jobs in
# parallel (they write to disjoint dirs: nothink_b*/ vs think_b*/), e.g.
#   sbatch --export=ALL,THINK_MODES=think   slurm/run_eval_budget.slurm
#   sbatch --export=ALL,THINK_MODES=nothink slurm/run_eval_budget.slurm
BUDGETS="${BUDGETS:-2048 4096 8192 16384 32768}"
THINK_MODES="${THINK_MODES:-nothink think}"
for THINK in $THINK_MODES; do
  if [ "$THINK" = nothink ]; then THINK_FLAG="--no-think"; else THINK_FLAG=""; fi
  for BUDGET in $BUDGETS; do
    OUTDIR="results/${THINK}_b${BUDGET}"
    for TASK in niah multikey longbench_v2; do
      echo "=== $(date '+%F %T') :: $TASK / rlm / $THINK / b=$BUDGET ==="
      python benchmarks/run_benchmark.py \
        --task "$TASK" --mode rlm --limit 50 \
        --base-url http://localhost:8000/v1 \
        --root-model "$MODEL" --sub-model "$MODEL" \
        $THINK_FLAG \
        --rpm 600 \
        --max-steps 50 \
        --max-context-tokens "$BUDGET" \
        --keep-recent-turns 3 \
        --out "$OUTDIR" \
        || echo "WARN: $TASK/rlm/$THINK/b=$BUDGET failed, continuing"
    done
  done
done
echo "=== done ==="
