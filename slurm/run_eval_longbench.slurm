#!/bin/bash
#SBATCH --job-name=rlm-longbench
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --output=logs/rlm-longbench.%J.out
#SBATCH --error=logs/rlm-longbench.%J.err

# LongBench v1: base RLM (unbounded) + vanilla, thinking on/off, over ALL 16
# English subsets in one task. The longbench loader samples PER SUBSET (--limit is
# per-subset), and EACH subset is scored with its OWN metric (F1 / ROUGE-L /
# classification / retrieval / count / code edit-similarity — see
# benchmarks/longbench_metrics.py), so score.py reports one row per
# (subset, mode, thinking). Download first on the login node:
#   bash slurm/download_data.sh   (writes $RLM_DATA_DIR/longbench.jsonl)

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

# --- 2x2 grid: thinking on/off x vanilla/rlm, base RLM (no --max-context-tokens),
# over all 16 subsets. --limit is PER SUBSET (default 50). Separate dirs per
# thinking condition so append+skip checkpointing never collides:
#   results/longbench_{think,nothink}.
# --vanilla-char-limit 100000 (~33k tok worst case) keeps the vanilla prompt inside
# the 40960 window; the longest subsets (narrativeqa/gov_report) are honestly
# truncated for vanilla only — RLM reads the full doc from the REPL, never the prompt.
LIMIT="${LIMIT:-50}"
for THINK in nothink think; do
  if [ "$THINK" = nothink ]; then THINK_FLAG="--no-think"; else THINK_FLAG=""; fi
  OUTDIR="results/longbench_$THINK"
  for MODE in vanilla rlm; do
    echo "=== $(date '+%F %T') :: longbench / $MODE / $THINK (limit=$LIMIT per subset) ==="
    # more generation room for the RLM root on think runs (see run_eval_local.slurm);
    # vanilla stays at 4096 — its large prompt + max_tokens must fit the window.
    if [ "$THINK" = think ] && [ "$MODE" = rlm ]; then MAXTOK_FLAG="--max-tokens 8192"; else MAXTOK_FLAG=""; fi
    python benchmarks/run_benchmark.py \
      --task longbench --mode "$MODE" --limit "$LIMIT" \
      --base-url http://localhost:8000/v1 \
      --root-model "$MODEL" --sub-model "$MODEL" \
      $THINK_FLAG $MAXTOK_FLAG \
      --rpm 600 \
      --max-steps 50 \
      --vanilla-char-limit 100000 \
      --out "$OUTDIR" \
      || echo "WARN: longbench/$MODE/$THINK failed, continuing"
  done
done
echo "=== done ==="
