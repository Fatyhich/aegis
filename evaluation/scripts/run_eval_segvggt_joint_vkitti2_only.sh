#!/usr/bin/env bash
#
# Rerun ONLY VKITTI2 for N ∈ {4, 6, 8} after pad_masks fix.
# Replica runs from previous sweep are kept intact.
#
# Run from repo root:
#   bash evaluation/scripts/run_eval_segvggt_joint_vkitti2_only.sh
#
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="$(cd "$EVAL_DIR/.." && pwd)"
PYTHON="${PYTHON:-python3}"

OUTPUT_BASE="$PROJECT_ROOT/evaluation/results/segvggt_dpt_joint_sweep"

N_FRAMES_LIST=(4 6 8)
NUM_VIS=30

VKITTI2_CFG="$EVAL_DIR/configs/config_eval_vkitti2_segvggt_joint.yaml"

SUMMARY_FILE="$OUTPUT_BASE/summary_vkitti2_rerun.txt"
TMPDIR_CFG="$OUTPUT_BASE/tmp_configs"
mkdir -p "$OUTPUT_BASE" "$TMPDIR_CFG"

cd "$EVAL_DIR"

echo "============================================================"
echo " SegVGGT-DPT Joint — VKITTI2 rerun  (N ∈ {4,6,8})"
echo " $(date)"
echo "============================================================"
echo ""

{
    printf "%-6s %-10s %8s %8s %8s %8s | %8s %8s %8s %8s\n" \
        "N" "Dataset" "Bin1" "Bin2" "Bin3" "Bin4" "R@1_avg" "R@5_avg" "AUPRC" "Pairs"
    printf "%s\n" "$(printf '%.0s-' {1..104})"
} > "$SUMMARY_FILE"

make_config() {
    local src_cfg="$1"
    local n_frames="$2"
    local out_cfg="$3"
    sed "s|^  N_FRAMES:.*|  N_FRAMES: $n_frames|" "$src_cfg" > "$out_cfg"
}

extract_summary() {
    local n_frames="$1"
    local dataset="$2"
    local json_file="$3"

    if [ ! -f "$json_file" ]; then
        printf "%-6s %-10s %s\n" "N=$n_frames" "$dataset" "FAILED" >> "$SUMMARY_FILE"
        return
    fi

    $PYTHON -c "
import json
with open('$json_file') as f:
    data = json.load(f)

bins = sorted(data.keys())
auprc_vals, r1_vals, r5_vals = [], [], []
total_pairs = 0
bin_auprc = []

for b in bins:
    m = data[b]
    auprc = m.get('AUPRC', m.get('auprc', 0.0))
    r1 = m.get('R@1', m.get('recall_at_1', 0.0))
    r5 = m.get('R@5', m.get('recall_at_5', 0.0))
    n = m.get('num_pairs', m.get('count', 0))
    bin_auprc.append(auprc)
    auprc_vals.append(auprc * max(n, 1))
    r1_vals.append(r1 * max(n, 1))
    r5_vals.append(r5 * max(n, 1))
    total_pairs += n

tp = max(total_pairs, 1)
r1_avg = sum(r1_vals) / tp
r5_avg = sum(r5_vals) / tp
auprc_avg = sum(auprc_vals) / tp

bin_str = ' '.join(f'{v:8.4f}' for v in bin_auprc)
print('N=$n_frames'.ljust(6) + ' ' + f'$dataset'.ljust(10) + ' ' +
      bin_str + f' | {r1_avg:8.4f} {r5_avg:8.4f} {auprc_avg:8.4f} {total_pairs:8d}')
" >> "$SUMMARY_FILE"
}

for n in "${N_FRAMES_LIST[@]}"; do
    echo ""
    echo "============================================================"
    echo " N_FRAMES = $n  (VKITTI2)"
    echo "============================================================"

    tmp_cfg="$TMPDIR_CFG/N${n}_vkitti2.yaml"
    out_dir="$OUTPUT_BASE/N${n}_vkitti2"
    make_config "$VKITTI2_CFG" "$n" "$tmp_cfg"

    echo "[$(date +%H:%M:%S)] N=$n / VKITTI2 ..."
    $PYTHON scripts/eval_segvggt_vkitti2.py \
        --config "$tmp_cfg" \
        --output_dir "$out_dir" \
        --device cuda \
        --visualize --num_vis "$NUM_VIS" \
        2>&1 | tee "$out_dir.log" \
        || echo "[ERROR] N=$n / VKITTI2 failed"

    extract_summary "$n" "VKITTI2" "$out_dir/metrics_by_bin.json"
done

echo ""
echo "============================================================"
echo " SUMMARY (VKITTI2 rerun) — $(date)"
echo "============================================================"
cat "$SUMMARY_FILE"
echo ""
echo "Saved to: $SUMMARY_FILE"
echo "Done!"
