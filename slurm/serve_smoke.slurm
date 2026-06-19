#!/bin/bash
# Spin up vLLM on a Prajna GPU node and verify it serves YOUR model end-to-end.
# Mirrors the proven run_eval_local.slurm serve block, trimmed to a smoke test:
#   serve -> wait -> /v1/models healthcheck -> one raw chat completion ->
#   one 1-example RLM+vanilla run on synthetic NIAH (no dataset download needed) -> exit.
#
# Usage:
#   1) set MODEL below to your exact model (HF id already in the cache, OR an absolute path)
#   2) sbatch slurm/serve_smoke.slurm
#   3) watch logs/serve-smoke.<JOBID>.out  and  logs/vllm.<JOBID>.log
#SBATCH --job-name=vllm-smoke
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --output=logs/serve-smoke.%J.out
#SBATCH --error=logs/serve-smoke.%J.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

source ~/venvs/rlm/bin/activate
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1                 # never touch the internet (compute nodes may be air-gapped)
export TRANSFORMERS_OFFLINE=1
export NVIDIA_API_KEY=local-dummy       # the client requires it to be set; vLLM ignores it
export VLLM_USE_FLASHINFER_SAMPLER=0

# >>> THE ONE LINE YOU CHANGE <<<
# Your "llama 4b instruct". Use the HF id if it's in $HF_HOME/hub (e.g. meta-llama/Llama-3.2-3B-Instruct),
# or an absolute path to the snapshot dir. Find it with:  ls $HF_HOME/hub
MODEL="Qwen/Qwen3-8B"   # <-- set to YOUR exact model id / path

PORT=8000
echo "=== $(date '+%F %T') :: node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
echo "=== model: $MODEL ==="

# --- start vLLM (same flags that worked for Qwen2.5-7B) ---
vllm serve "$MODEL" --port "$PORT" \
  --served-model-name "$MODEL" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager -O0 \
  > "logs/vllm.$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

# --- wait for the server (model load can take a few minutes) ---
echo "waiting for vLLM on :$PORT ..."
UP=0
for i in $(seq 1 90); do
  if curl -s -o /dev/null "http://localhost:$PORT/v1/models"; then UP=1; break; fi
  # bail early if the server process already died
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "ERROR: vLLM exited during startup. See logs/vllm.$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
[ "$UP" = 1 ] || { echo "ERROR: vLLM never became ready. See logs/vllm.$SLURM_JOB_ID.log"; exit 1; }

echo "--- /v1/models ---"
curl -s "http://localhost:$PORT/v1/models"; echo

echo "--- raw chat completion sanity check ---"
curl -s "http://localhost:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly the word: READY\"}],\"max_tokens\":8,\"temperature\":0}"
echo

# --- 1-example RLM + vanilla smoke on synthetic NIAH (generated locally, no HF dataset needed) ---
echo "=== $(date '+%F %T') :: NIAH limit=1, mode=both ==="
python benchmarks/run_benchmark.py \
  --task niah --mode both --limit 1 \
  --base-url "http://localhost:$PORT/v1" \
  --root-model "$MODEL" --sub-model "$MODEL" \
  --no-think \
  --rpm 600 --max-steps 20 \
  --vanilla-char-limit 120000 \
  --out results --debug

echo "=== $(date '+%F %T') :: smoke done — if you saw READY + a niah result, vLLM is good ==="
