#!/bin/bash
#SBATCH --job-name=rlm-lbv2
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --output=logs/rlm-lbv2.%J.out
#SBATCH --error=logs/rlm-lbv2.%J.err

# NOTE: superseded by slurm/run_eval_task.slurm, which runs every task with
# IDENTICAL flags (this script's --vanilla-char-limit / results-dir naming differ
# from its siblings, which confounds cross-dataset comparison). Kept for
# reproducing earlier runs; use the unified driver for new campaigns.

# LongBench v2, STANDALONE: base RLM (unbounded) + vanilla, thinking on/off, over
# ALL 6 domains in one task. The longbench_v2 loader samples PER DOMAIN (--limit is
# per-domain) and tags each row with its `domain` as `subset`, scored by exact
# A/B/C/D choice accuracy, so score.py reports one row per (domain, mode, thinking).
#
# NOTE: run_eval_local.slurm ALSO runs longbench_v2 (bundled with niah/multikey,
# into results/{think,nothink}/). This script is for running v2 ALONE — e.g. a
# larger/smaller per-domain LIMIT — and writes to its OWN dirs
# (results/longbench_v2_{think,nothink}/), so it never races that job's files.
# Download first on the login node:  bash slurm/download_data.sh
#   (writes $RLM_DATA_DIR/longbench_v2.jsonl)

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
# over all 6 domains. --limit is PER DOMAIN (default 50). Separate dirs per thinking
# condition so append+skip checkpointing never collides:
#   results/longbench_v2_{think,nothink}.
# --vanilla-char-limit 120000 (~32k tok) keeps the vanilla prompt inside the 40960
# window; v2 contexts are very long, so vanilla is honestly truncated there — RLM
# reads the full doc from the REPL, never the prompt.
LIMIT="${LIMIT:-50}"
for THINK in nothink think; do
  if [ "$THINK" = nothink ]; then THINK_FLAG="--no-think"; else THINK_FLAG=""; fi
  OUTDIR="results/longbench_v2_$THINK"
  for MODE in vanilla rlm; do
    echo "=== $(date '+%F %T') :: longbench_v2 / $MODE / $THINK (limit=$LIMIT per domain) ==="
    # more generation room for the RLM root on think runs (see run_eval_local.slurm);
    # vanilla stays at 4096 — its large prompt + max_tokens must fit the window.
    if [ "$THINK" = think ] && [ "$MODE" = rlm ]; then MAXTOK_FLAG="--max-tokens 8192"; else MAXTOK_FLAG=""; fi
    python benchmarks/run_benchmark.py \
      --task longbench_v2 --mode "$MODE" --limit "$LIMIT" \
      --base-url http://localhost:8000/v1 \
      --root-model "$MODEL" --sub-model "$MODEL" \
      $THINK_FLAG $MAXTOK_FLAG \
      --rpm 600 \
      --max-steps 50 \
      --vanilla-char-limit 120000 \
      --out "$OUTDIR" \
      || echo "WARN: longbench_v2/$MODE/$THINK failed, continuing"
  done
done
echo "=== done ==="
