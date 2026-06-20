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

# Single model, thinking toggled per-run on the CLIENT (--no-think sets
# chat_template_kwargs.enable_thinking=False). Qwen3-8B is the thinking-capable
# model; "thinking off" is NOT a different model, just this flag.
MODEL=Qwen/Qwen3-8B

# --- start ONE vLLM server that serves both thinking and non-thinking runs ---
# --reasoning-parser qwen3 routes Qwen3's <think>...</think> into reasoning_content,
# leaving message.content (the ```python block the harness reads) clean. It is
# REQUIRED for the thinking runs and HARMLESS for the no-think runs (no think block
# to extract). If your vLLM build rejects 'qwen3', try 'deepseek_r1'
# (vllm serve --help | grep -i reason).
vllm serve "$MODEL" --served-model-name "$MODEL" --port 8000 --max-model-len 32768 \
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

# --- the 2x2 grid (thinking on/off x vanilla/rlm), run over every task ---
# Results MUST be separated by thinking condition: run_benchmark writes to
# {task}.{mode}.{slug}.jsonl in APPEND mode and SKIPS ids already in that file.
# Same task+mode+model for both conditions would collide and the 2nd run would
# skip everything -> send thinking -> results/think, no-think -> results/nothink.
for THINK in nothink think; do
  if [ "$THINK" = nothink ]; then THINK_FLAG="--no-think"; else THINK_FLAG=""; fi
  OUTDIR="results/$THINK"
  for TASK in niah multikey longbench_v2; do
    for MODE in vanilla rlm; do
      echo "=== $(date '+%F %T') :: $TASK / $MODE / $THINK ==="
      python benchmarks/run_benchmark.py \
        --task "$TASK" --mode "$MODE" --limit 50 \
        --base-url http://localhost:8000/v1 \
        --root-model "$MODEL" --sub-model "$MODEL" \
        $THINK_FLAG \
        --rpm 600 \
        --max-steps 50 \
        --vanilla-char-limit 120000 \
        --out "$OUTDIR" \
        || echo "WARN: $TASK/$MODE/$THINK failed, continuing"
    done
  done
done
echo "=== done ==="
