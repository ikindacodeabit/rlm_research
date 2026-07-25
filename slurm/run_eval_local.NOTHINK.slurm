#!/bin/bash
#SBATCH --job-name=rlm-nothink
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --output=logs/rlm-nothink.%J.out
#SBATCH --error=logs/rlm-nothink.%J.err

# NOTE: superseded by slurm/run_eval_task.slurm, which runs every task with
# IDENTICAL flags (this script's --vanilla-char-limit / results-dir naming differ
# from its siblings, which confounds cross-dataset comparison). Kept for
# reproducing earlier runs; use the unified driver for new campaigns.

# Quick NO-THINK-ONLY pass: the fast half of run_eval_local.slurm's 2x2 grid.
# Runs {niah, multikey, longbench_v2} x {vanilla, rlm} with Qwen3 thinking OFF,
# into results/nothink/ — the SAME dir the full script uses for its nothink cells,
# so the two are interchangeable and resumable together (append + skip-by-id).
# Use this when you only want the nothink numbers; use run_eval_local.slurm for
# the full thinking-on/off comparison.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

source ~/venvs/rlm/bin/activate
export RLM_DATA_DIR="${RLM_DATA_DIR:-$HOME/rlm_data}"
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1                 # never try the internet
export NVIDIA_API_KEY=local-dummy       # client requires it to be set; vLLM ignores it
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL=Qwen/Qwen3-8B

# --- start vLLM server on this node's GPU ---
# Serve at Qwen3-8B's native 40960 (same as run_eval_local.slurm): vLLM rejects any
# request where prompt + max_tokens > max-model-len, so the larger window lets the
# ~30k-token vanilla long-context prompt fit. --reasoning-parser qwen3 is a no-op
# under --no-think (no <think> block) but keeps the serve line consistent.
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

# --- run the grid against the local server: no rate limit ---
# For the RLM memory-budget / scratchpad sweeps, use run_eval_budget.slurm /
# run_eval_scratchpad.slurm instead (they add --max-context-tokens / --scratchpad).
for TASK in niah multikey longbench_v2; do
  for MODE in vanilla rlm; do
    echo "=== $(date '+%F %T') :: $TASK / $MODE / nothink ==="
    python benchmarks/run_benchmark.py \
      --task "$TASK" --mode "$MODE" --limit 50 \
      --base-url http://localhost:8000/v1 \
      --root-model "$MODEL" --sub-model "$MODEL" \
      --no-think \
      --rpm 600 \
      --max-steps 50 \
      --vanilla-char-limit 120000 \
      --out results/nothink \
      || echo "WARN: $TASK/$MODE failed, continuing"
  done
done
echo "=== done ==="
