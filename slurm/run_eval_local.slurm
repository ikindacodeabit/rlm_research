#!/bin/bash
#SBATCH --job-name=rlm-local
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --output=logs/rlm-local.%J.out
#SBATCH --error=logs/rlm-local.%J.err

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
vllm serve "$MODEL" --served-model-name "$MODEL" --port 8000 --max-model-len 32768 \
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
# To sweep the RLM memory budget instead of an unbounded root context, add e.g.:
#   --max-context-tokens 4096 --keep-recent-turns 2 --fold-strategy summarize --max-notes-tokens 512
# (omit --max-context-tokens for the legacy unbounded behaviour).
for TASK in niah multikey longbench_v2; do
  for MODE in vanilla rlm; do
    echo "=== $(date '+%F %T') :: $TASK / $MODE ==="
    python benchmarks/run_benchmark.py \
      --task "$TASK" --mode "$MODE" --limit 50 \
      --base-url http://localhost:8000/v1 \
      --root-model "$MODEL" --sub-model "$MODEL" \
      --no-think \
      --rpm 600 \
      --max-steps 20 \
      --vanilla-char-limit 120000 \
      --out results \
      || echo "WARN: $TASK/$MODE failed, continuing"
  done
done
echo "=== done ==="
