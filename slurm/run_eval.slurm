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


# ============================== RETIRED ==============================
# This is the old NIM-hosted script. It passes NO --root-model, so it silently
# falls back to run_benchmark's argparse default (meta/llama-3.3-70b-instruct)
# against https://integrate.api.nvidia.com/v1, with --max-steps 12 and
# --vanilla-char-limit 400000, and writes into the BARE results/ root -- which
# then pollutes every score.py table with a differently-configured model.
#
# Use the unified driver instead:
#   sbatch --job-name=rlm-<task> --export=ALL,TASK=<task>,LIMIT=25 \
#          slurm/run_eval_task.slurm
# or scripts/submit_campaign.sh for the whole sweep.
echo "slurm/run_eval.slurm is RETIRED -- use slurm/run_eval_task.slurm (see header)."
exit 1
# =====================================================================

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
