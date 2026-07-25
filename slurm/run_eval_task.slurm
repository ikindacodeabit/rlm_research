#!/bin/bash
#SBATCH --job-name=rlm-eval
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --output=logs/%x.%J.out
#SBATCH --error=logs/%x.%J.err

# ONE driver for every task, so a cross-dataset comparison is not confounded by
# per-script flag drift. The previous six scripts disagreed on --vanilla-char-limit
# (120000 vs 100000, i.e. how much document the vanilla arm ever sees), on results
# dir naming, and on whether a concurrency guard existed at all.
#
# Everything below is IDENTICAL for every task; only TASK and LIMIT vary.
#
#   sbatch --job-name=rlm-longbench --export=ALL,TASK=longbench,LIMIT=25 \
#          slurm/run_eval_task.slurm
#
# or just use scripts/submit_campaign.sh, which submits one job per task.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

TASK="${TASK:?set TASK=<niah|multikey|ruler16k|ruler32k|longbench|longbench_v2|loft32k|loft128k>}"
LIMIT="${LIMIT:-25}"
THINK_CONDITIONS="${THINK_CONDITIONS:-nothink think}"

# Refuse to start if another job is already running THIS task. Two jobs writing the
# same results/*.jsonl duplicate all remaining work: load_done is read ONCE per cell,
# so the second job cannot see the first job's in-flight rows.
GUARD_NAME="${SLURM_JOB_NAME:-rlm-eval}"
if squeue -u "$USER" -h -n "$GUARD_NAME" -o %i | grep -qv "^${SLURM_JOB_ID}$"; then
  echo "another '$GUARD_NAME' job is already running; exiting to avoid duplicate work"
  exit 1
fi

source ~/venvs/rlm/bin/activate
export PYTHONUNBUFFERED=1                # per-line flush so the .out tracks progress live
export RLM_DATA_DIR="${RLM_DATA_DIR:-$HOME/rlm_data}"
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1                 # never try the internet
export NVIDIA_API_KEY=local-dummy       # client requires it to be set; vLLM ignores it
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL="${MODEL:-Qwen/Qwen3-8B}"
# SLURM can co-schedule two of these jobs on ONE node (each asks for only 1 GPU),
# and a hardcoded port then makes the second vLLM fail to bind. Derive a per-job
# port from the job id so co-located jobs never collide, then verify it is actually
# free before serving.
PORT="${PORT:-$((10000 + SLURM_JOB_ID % 20000))}"
for _try in $(seq 1 50); do
  if ! (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then break; fi
  exec 3<&- 2>/dev/null || true
  PORT=$((PORT + 1))
done
echo "serving on port $PORT (node $(hostname), job $SLURM_JOB_ID)"

# --- one vLLM server for every cell (Qwen3-8B native 40960 window) ---
vllm serve "$MODEL" --served-model-name "$MODEL" --port "$PORT" --max-model-len 40960 \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.90 --enforce-eager -O0 > logs/vllm.$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

for i in $(seq 1 60); do
  curl -s -o /dev/null "http://localhost:$PORT/v1/models" && break
  sleep 10
done
curl -s "http://localhost:$PORT/v1/models" | head -c 200 || {
  echo "vLLM failed to start, see logs/vllm.$SLURM_JOB_ID.log"; exit 1; }
echo "vLLM up. task=$TASK limit=$LIMIT"

# --- 2x2 grid: thinking on/off x vanilla/rlm ---
# Results are separated by thinking condition because run_benchmark appends to
# {task}.{mode}.{slug}.jsonl and SKIPS ids already present: the same task+mode+model
# under two conditions in one dir would silently mix configurations.
# Dir scheme is uniform: results/{task}_{think,nothink} for EVERY task.
#
# --vanilla-char-limit 100000 (~33k tok worst case) keeps the vanilla prompt inside
# the 40960 window with headroom for --max-tokens. 120000 had none and risks vLLM
# rejecting prompt+max_tokens > max-model-len.
for THINK in $THINK_CONDITIONS; do
  if [ "$THINK" = nothink ]; then THINK_FLAG="--no-think"; else THINK_FLAG=""; fi
  OUTDIR="results/${TASK}_${THINK}"
  for MODE in vanilla rlm; do
    echo "=== $(date '+%F %T') :: $TASK / $MODE / $THINK (limit=$LIMIT) ==="
    # Qwen3 <think> tokens share the generation budget with the visible answer, so
    # give the RLM root more room on think runs. Vanilla keeps the 4096 default
    # because its prompt is large and prompt + max_tokens must fit the window.
    if [ "$THINK" = think ] && [ "$MODE" = rlm ]; then MAXTOK_FLAG="--max-tokens 8192"; else MAXTOK_FLAG=""; fi
    python benchmarks/run_benchmark.py \
      --task "$TASK" --mode "$MODE" --limit "$LIMIT" \
      --base-url "http://localhost:$PORT/v1" \
      --root-model "$MODEL" --sub-model "$MODEL" \
      $THINK_FLAG $MAXTOK_FLAG \
      --rpm 600 \
      --max-steps 50 \
      --run-timeout 900 \
      --max-sub-calls 40 \
      --vanilla-char-limit 100000 \
      --out "$OUTDIR" \
      || echo "WARN: $TASK/$MODE/$THINK failed, continuing"
  done
done
echo "=== done: $TASK ==="
