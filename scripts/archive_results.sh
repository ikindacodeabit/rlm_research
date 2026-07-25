#!/bin/bash
# Move the current results/ tree out of the way so a campaign starts clean.
#
#   bash scripts/archive_results.sh            # -> archive/results_pre_<date>/
#   bash scripts/archive_results.sh my_label   # -> archive/my_label/
#
# WHY outside results/: benchmarks/score.py does rglob("*.jsonl") over whatever
# directory you point it at, so an archive left ANYWHERE under results/ is still
# swept into every table. archive/ is a sibling, and is git-ignored.
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:-results_pre_$(date +%Y-%m-%d)}"
DEST="archive/$LABEL"

if [ ! -d results ] || [ -z "$(ls -A results 2>/dev/null)" ]; then
  echo "results/ is empty or absent — nothing to archive."
  mkdir -p results
  exit 0
fi

if [ -e "$DEST" ]; then
  echo "ERROR: $DEST already exists; pass a different label so nothing is overwritten."
  exit 1
fi

mkdir -p archive
echo "Archiving:"
du -sh results 2>/dev/null || true
mv results "$DEST"
mkdir -p results

echo
echo "Moved results/ -> $DEST"
echo "  records:  $(find "$DEST" -name '*.jsonl' | wc -l | tr -d ' ') jsonl files"
echo "  backups:  $(find "$DEST" -name '*.jsonl.bak' | wc -l | tr -d ' ') .bak files preserved"
echo
echo "results/ is now empty and ready for the campaign."
echo "Score the archive any time with:  python benchmarks/score.py $DEST"
