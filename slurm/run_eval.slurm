#!/bin/bash
#SBATCH --job-name=rlm-eval
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --output=logs/rlm.%J.out
#SBATCH --error=logs/rlm.%J.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

source ~/venvs/rlm/bin/activate
export RLM_DATA_DIR="${RLM_DATA_DIR:-$HOME/rlm_data}"
: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY before submitting}"

# --- connectivity check: fail fast if compute node has no internet ---
curl -s -o /dev/null -w "NIM API reachability: HTTP %{http_code}\n" --max-time 20 \
  https://integrate.api.nvidia.com/v1/models \
  -H "Authorization: Bearer ${NVIDIA_API_KEY}" \
  || { echo "ERROR: no outbound internet from $(hostname) — ask hpc@iitb.ac.in about a proxy"; exit 1; }

# --- sequential grid: one process, full 30 rpm to itself ---
for TASK in niah multikey longbench_v2; do
  for MODE in vanilla rlm; do
    echo "=== $(date '+%F %T') :: $TASK / $MODE ==="
    python benchmarks/run_benchmark.py \
      --task "$TASK" --mode "$MODE" \
      --limit 50 --rpm 30 --out results \
      || echo "WARN: $TASK/$MODE failed, continuing with next cell"
  done
done

echo "=== $(date '+%F %T') :: all cells done ==="
