#!/usr/bin/env bash
# trace_blindspot.sh
#
# The two evaluations that close the certification blind spot, run over every
# finished trace-removal cell (eval/results/trace_audit/lists/models_*.txt):
#   reliance GROUP   source-grid reliance on the q90 benchmark photos (eval/source_grid_reliance.py)
#                    for the cells whose tag matches GROUP (e.g. efficientvit_b3)
#   controls         the same test with a lossless final save on the paper checkpoints (positive controls)
#   probe            the realistic content-free probe (before/after-rotation conditions, 1,000 textures)
#                    on every cell
# Results under eval/results/trace_audit/source_grid_reliance/ and certify_realistic/.
#
#   screen -dmS irae-bs-effvit bash -c "bash eval/trace_blindspot.sh reliance efficientvit_b3 > logs/trace_removal/blindspot_effvit.log 2>&1"

set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate irae
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${1:?usage: trace_blindspot.sh reliance GROUP | controls | probe}"
LISTS=eval/results/trace_audit/lists
OUT=eval/results/trace_audit
mkdir -p "$OUT/source_grid_reliance" "$OUT/certify_realistic"

case "$MODE" in
  reliance)
    GROUP="${2:?group}"
    cat "$LISTS"/models_*"$GROUP"*.txt > "$LISTS/reliance_$GROUP.txt"
    echo "RELIANCE $GROUP: $(wc -l < "$LISTS/reliance_$GROUP.txt") models $(date '+%F %T')"
    python eval/source_grid_reliance.py --models "$LISTS/reliance_$GROUP.txt" --out-dir "$OUT/source_grid_reliance"
    ;;
  controls)
    echo "CONTROLS (lossless final save) $(date '+%F %T')"
    python eval/source_grid_reliance.py --models "$LISTS/reliance_paper.txt" --quality 0 \
      --out-dir "$OUT/source_grid_reliance/lossless_controls"
    ;;
  probe)
    for f in "$LISTS"/models_*.txt; do
      IFS='|' read -r APPROACH TAG CK < "$f"
      [[ -f "$OUT/certify_realistic/${TAG}_n1000.json" ]] && continue
      echo "======== PROBE $TAG $(date '+%F %T') ========"
      python eval/grid_probe.py --approach "$APPROACH" --ckpt "$CK" --n 1000 \
        --conditions "lossless/jpeg90,jpeg95/jpeg90,jpeg75/jpeg90" \
        --out "$OUT/certify_realistic/${TAG}_n1000.json"
      echo "grid_probe exit code $? (1 = a gate failed)"
    done
    ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
echo "BLINDSPOT $MODE ${2:-} DONE $(date '+%F %T')"
