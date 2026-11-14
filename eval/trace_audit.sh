#!/usr/bin/env bash
# trace_audit.sh
#
# Runs the JPEG gap (eval/jpeg_gap.py, clean vs. one q90 save) and the
# content-free probe (eval/grid_probe.py) for every checkpoint in a list file
# and appends one summary line per cell to eval/results/trace_audit/summary.tsv.
#
# List format (one cell per line, built by eval/trace_audit_lists.py):
#   label|approach|checkpoint|test_dir|seeds|probe_size
#
# Cells whose result files already exist are skipped, so the script can be
# re-run to resume. Run from the repository root:
#   bash eval/trace_audit.sh eval/results/trace_audit/lists/paper_drcd.txt

set -uo pipefail

LIST="${1:?usage: trace_audit.sh LIST_FILE}"
# AUDIT_CONDITIONS: comma list for jpeg_gap.py; the summary always reads the clean and
# jpeg90 entries, extra conditions (e.g. blur3,blur3+jpeg90 for trace-removed models)
# are kept in the gap JSON
AUDIT_CONDITIONS="${AUDIT_CONDITIONS:-clean,jpeg90}"
# AUDIT_ROTATION_MODE: crop (paper), crop_fixed or pad; must match the model's training crop
AUDIT_ROTATION_MODE="${AUDIT_ROTATION_MODE:-crop}"
# AUDIT_ORDER: image order of jpeg_gap.py: sorted (all results recorded before November 2026)
# or pinned (the paper's image-angle pairing, as train.py --test-only)
AUDIT_ORDER="${AUDIT_ORDER:-sorted}"
# AUDIT_OUT_DIR: folder for the result files and summary.tsv
OUT_DIR="${AUDIT_OUT_DIR:-eval/results/trace_audit}"
SUMMARY="$OUT_DIR/summary.tsv"
mkdir -p "$OUT_DIR"
[[ -f "$SUMMARY" ]] || printf 'label\tapproach\tclean_acc10\tjpeg90_acc10\tgap_points\tclean_mae\tjpeg90_mae\tclean_median\tjpeg90_median\tjpeg90_axis_share\tprobe_lossless\tprobe_jpeg90\tprobe_jpeg75\tprobe_jpeg50\tcheckpoint\n' > "$SUMMARY"

while IFS='|' read -r label approach ckpt test_dir seeds probe_size; do
  [[ -z "$label" || "$label" == \#* ]] && continue
  gap_json="$OUT_DIR/${label}_gap.json"
  probe_json="$OUT_DIR/${label}_probe.json"
  if [[ ! -f "$ckpt" ]]; then
    echo "[$label] MISSING checkpoint: $ckpt"
    continue
  fi
  echo ""
  echo "=== [$label] $(date '+%F %T') approach=$approach test=$test_dir seeds=$seeds"
  if [[ ! -f "$gap_json" ]]; then
    python eval/jpeg_gap.py --approach "$approach" --ckpt "$ckpt" --test-dir "$test_dir" \
      --seeds "$seeds" --conditions "$AUDIT_CONDITIONS" --rotation-mode "$AUDIT_ROTATION_MODE" \
      --order "$AUDIT_ORDER" --out "$gap_json" 2>&1 | grep -v "INFO\|WARNING\|Seed set"
  else
    echo "gap exists, skipping"
  fi
  if [[ ! -f "$probe_json" ]]; then
    python eval/grid_probe.py --approach "$approach" --ckpt "$ckpt" --size "$probe_size" \
      --rotation-mode "$AUDIT_ROTATION_MODE" --out "$probe_json" 2>&1 | grep -v "INFO\|WARNING\|Seed set"
  else
    echo "probe exists, skipping"
  fi
  if [[ -f "$gap_json" && -f "$probe_json" ]] && ! grep -q "^${label}$(printf '\t')" "$SUMMARY"; then
    python - "$label" "$approach" "$gap_json" "$probe_json" "$ckpt" >> "$SUMMARY" <<'EOF'
import json, sys
label, approach, gap_json, probe_json, ckpt = sys.argv[1:]
g = json.load(open(gap_json))["mean"]; p = json.load(open(probe_json))["conditions"]
c, j = g["clean"], g["jpeg90"]
pr = lambda k: f"{p[k]['mod90_within5']:.3f}" if k in p else "-"
print("\t".join([label, approach, f"{c['acc10']:.4f}", f"{j['acc10']:.4f}", f"{(c['acc10']-j['acc10'])*100:.1f}",
                 f"{c['mae']:.2f}", f"{j['mae']:.2f}", f"{c['median']:.2f}", f"{j['median']:.2f}",
                 f"{j['axis_share_pred']:.3f}", pr("lossless"), pr("jpeg90"), pr("jpeg75"), pr("jpeg50"), ckpt]))
EOF
  fi
done < "$LIST"

echo ""
echo "=== audit of $LIST finished $(date '+%F %T'); summary in $SUMMARY"
