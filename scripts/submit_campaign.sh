#!/bin/bash
# Submit the full RLM-vs-vanilla campaign: one SLURM job per task, all sharing the
# single driver slurm/run_eval_task.slurm so no flag drifts between datasets.
#
#   bash scripts/submit_campaign.sh              # submit everything
#   bash scripts/submit_campaign.sh --dry-run    # print the sbatch lines only
#   TASKS="longbench loft32k" bash scripts/submit_campaign.sh
#
# Each job runs the full 2x2 grid (vanilla/rlm x nothink/think) for its task and
# fits inside one 48h walltime at the default limits. Jobs are independent: one
# failing or hitting the walltime does not affect the others, and re-submitting
# resumes from the JSONL checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY=1

TASKS="${TASKS:-niah multikey ruler32k longbench longbench_v2 loft32k}"

# --limit is PER SUBSET for every file-backed task, but a GLOBAL example count for
# the synthetic generators. At a uniform limit niah would contribute 25 examples
# against longbench's 400 (16 subsets), so any cross-dataset mean would be ~16x
# weighted toward longbench. Scale the synthetics to a comparable n instead.
limit_for() {
  case "$1" in
    niah|multikey|niah-1m) echo "${SYNTH_LIMIT:-325}" ;;
    *)                     echo "${LIMIT:-25}" ;;
  esac
}

# Data each task needs in $RLM_DATA_DIR (synthetics generate in-process).
data_for() {
  case "$1" in
    niah|multikey|niah-1m) echo "" ;;
    ruler16k)              echo "ruler16k.jsonl" ;;
    ruler32k)              echo "ruler32k.jsonl" ;;
    longbench)             echo "longbench.jsonl" ;;
    longbench_v2)          echo "longbench_v2.jsonl" ;;
    loft32k)               echo "loft32k.jsonl" ;;
    loft128k)              echo "loft128k.jsonl" ;;
    *)                     echo "" ;;
  esac
}

DATA_DIR="${RLM_DATA_DIR:-$HOME/rlm_data}"
missing=""
for t in $TASKS; do
  f="$(data_for "$t")"
  [ -n "$f" ] && [ ! -s "$DATA_DIR/$f" ] && missing="$missing $t($f)"
done
if [ -n "$missing" ]; then
  echo "ERROR: missing dataset files in $DATA_DIR:$missing"
  echo "Fetch them on the LOGIN node first, e.g.:"
  echo "  DATASETS=ruler32k,longbench,longbench_v2,loft32k bash slurm/download_data.sh"
  exit 1
fi

echo "Campaign: results/ <- $TASKS"
for t in $TASKS; do
  lim="$(limit_for "$t")"
  cmd=(sbatch --job-name="rlm-$t" --export="ALL,TASK=$t,LIMIT=$lim" slurm/run_eval_task.slurm)
  if [ -n "$DRY" ]; then
    echo "  ${cmd[*]}"
  else
    echo "  submitting $t (limit=$lim)"
    "${cmd[@]}"
  fi
done

[ -n "$DRY" ] && exit 0
echo
echo "Watch:   squeue -u \$USER -o '%.10i %.20j %.10M %.6D %R'"
echo "Score:   python benchmarks/score.py results"
