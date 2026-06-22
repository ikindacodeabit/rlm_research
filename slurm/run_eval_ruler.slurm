#!/bin/bash
#SBATCH --job-name=rlm-ruler
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --output=logs/rlm-ruler.%J.out
#SBATCH --error=logs/rlm-ruler.%J.err

# RULER-32k: base RLM (unbounded) + vanilla, thinking on/off, over ALL ~13
# RULER subsets in one task. The ruler32k loader samples PER SUBSET (--limit is
# per-subset), and run_benchmark tags each record with its `subset`, so score.py
# reports one row per (subset, variant, mode). Download first on the login node:
#   bash slurm/download_data.sh   (writes $RLM_DATA_DIR/ruler32k.jsonl)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

source ~/venvs/rlm/bin/activate
export PYTHONUNBUFFERED=1                # per-line flush so the .out tracks progress live
export RLM_DATA_DIR="${RLM_DATA_DIR:-$HOME/rlm_data}"
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1                 # never try the internet
export NVIDIA_API_KEY=local-dummy       # client requires it to be set; vLLM ignores it
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL=Qwen/Qwen3-8B

# --- start ONE vLLM server for every cell ---
# RULER-32k inputs run to ~32k tokens, and vanilla stuffs the whole thing into the
# prompt, so the window must exceed 32k to leave generation headroom: serve at
# Qwen3-8B's native 40960 (vs 32768 in run_eval_local.slurm).
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

# --- 2x2 grid: thinking on/off x vanilla/rlm, base RLM (no --max-context-tokens) ---
# Separate dirs per thinking condition (append+skip checkpointing would otherwise
# collide): think -> results/ruler_think, no-think -> results/ruler_nothink.
# --limit 50 is PER SUBSET (~13 x 50 examples per cell).
LIMIT="${LIMIT:-50}"
for THINK in nothink think; do
  if [ "$THINK" = nothink ]; then THINK_FLAG="--no-think"; else THINK_FLAG=""; fi
  OUTDIR="results/ruler_$THINK"
  for MODE in vanilla rlm; do
    echo "=== $(date '+%F %T') :: ruler32k / $MODE / $THINK (limit=$LIMIT per subset) ==="
    python benchmarks/run_benchmark.py \
      --task ruler32k --mode "$MODE" --limit "$LIMIT" \
      --base-url http://localhost:8000/v1 \
      --root-model "$MODEL" --sub-model "$MODEL" \
      $THINK_FLAG \
      --rpm 600 \
      --max-steps 50 \
      --vanilla-char-limit 200000 \
      --out "$OUTDIR" \
      || echo "WARN: ruler32k/$MODE/$THINK failed, continuing"
  done
done
echo "=== done ==="
