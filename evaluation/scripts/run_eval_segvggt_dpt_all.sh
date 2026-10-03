#!/usr/bin/env bash
#
# Evaluate SegVGGT-DPT on Replica + VKITTI2 across 4 checkpoints.
# Run from repo root:
#   bash evaluation/scripts/run_eval_segvggt_dpt_all.sh
#
# Designed to be left running in tmux overnight.
#
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="$(cd "$EVAL_DIR/.." && pwd)"
PYTHON="/home/dennis/Projects/claude-venv/.venv/bin/python3"

CKPT_DIR="$PROJECT_ROOT/results/segvggt_dpt/v3-001"
OUTPUT_BASE="$PROJECT_ROOT/evaluation/results/segvggt_dpt_sweep"

CHECKPOINTS=("best.pth" "epoch_001.pth" "epoch_002.pth" "epoch_003.pth")
CKPT_NAMES=("best" "epoch_001" "epoch_002" "epoch_003")

REPLICA_CFG="$EVAL_DIR/configs/config_eval_replica_segvggt.yaml"
VKITTI2_CFG="$EVAL_DIR/configs/config_eval_vkitti2_segvggt.yaml"

SUMMARY_FILE="$OUTPUT_BASE/summary.txt"
TMPDIR_CFG="$OUTPUT_BASE/tmp_configs"
mkdir -p "$OUTPUT_BASE" "$TMPDIR_CFG"

cd "$EVAL_DIR"

echo "============================================================"
echo " SegVGGT-DPT evaluation sweep"
echo " $(date)"
echo " Checkpoints: ${CKPT_NAMES[*]}"
echo " Datasets: Replica, VKITTI2"
echo "============================================================"
echo ""

# ── Summary header ──
{
    printf "%-12s %-10s %8s %8s %8s %8s | %8s %8s %8s %8s\n" \
        "Checkpoint" "Dataset" "Bin1" "Bin2" "Bin3" "Bin4" "R@1_avg" "R@5_avg" "AUPRC" "Pairs"
    printf "%s\n" "$(printf '%.0s-' {1..110})"
} > "$SUMMARY_FILE"

# ── Helper: make temp config with swapped checkpoint path ──
make_config() {
    local src_cfg="$1"
    local ckpt_path="$2"
    local out_cfg="$3"
    sed "s|CHECKPOINT:.*|CHECKPOINT: $ckpt_path|" "$src_cfg" > "$out_cfg"
}

# ── Helper: extract metrics_by_bin.json into summary row ──
extract_summary() {
    local ckpt_name="$1"
    local dataset="$2"
    local json_file="$3"

    if [ ! -f "$json_file" ]; then
        printf "%-12s %-10s %s\n" "$ckpt_name" "$dataset" "FAILED" >> "$SUMMARY_FILE"
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
print(f'$ckpt_name'.ljust(12) + ' ' + f'$dataset'.ljust(10) + ' ' +
      bin_str + f' | {r1_avg:8.4f} {r5_avg:8.4f} {auprc_avg:8.4f} {total_pairs:8d}')
" >> "$SUMMARY_FILE"
}

# ── Main loop ──
for i in "${!CHECKPOINTS[@]}"; do
    ckpt="${CHECKPOINTS[$i]}"
    name="${CKPT_NAMES[$i]}"
    ckpt_path="$CKPT_DIR/$ckpt"

    if [ ! -f "$ckpt_path" ]; then
        echo "[SKIP] $ckpt_path not found"
        continue
    fi

    # Relative path from evaluation/ dir (configs use ../results/...)
    ckpt_rel="../results/segvggt_dpt/v3-001/$ckpt"

    echo ""
    echo "============================================================"
    echo " Checkpoint: $name  ($ckpt)"
    echo "============================================================"

    # ── Replica ──
    tmp_cfg="$TMPDIR_CFG/${name}_replica.yaml"
    out_dir="$OUTPUT_BASE/${name}_replica"
    make_config "$REPLICA_CFG" "$ckpt_rel" "$tmp_cfg"

    echo "[$(date +%H:%M:%S)] $name / Replica ..."
    $PYTHON scripts/eval_replica_table2.py \
        --config "$tmp_cfg" \
        --output_dir "$out_dir" \
        --device cuda \
        2>&1 | tee "$out_dir.log" \
        || echo "[ERROR] $name / Replica failed"

    extract_summary "$name" "Replica" "$out_dir/metrics_by_bin.json"

    # ── VKITTI2 ──
    tmp_cfg="$TMPDIR_CFG/${name}_vkitti2.yaml"
    out_dir="$OUTPUT_BASE/${name}_vkitti2"
    make_config "$VKITTI2_CFG" "$ckpt_rel" "$tmp_cfg"

    echo "[$(date +%H:%M:%S)] $name / VKITTI2 ..."
    $PYTHON scripts/eval_segvggt_vkitti2.py \
        --config "$tmp_cfg" \
        --output_dir "$out_dir" \
        --device cuda \
        2>&1 | tee "$out_dir.log" \
        || echo "[ERROR] $name / VKITTI2 failed"

    extract_summary "$name" "VKITTI2" "$out_dir/metrics_by_bin.json"

done

# ── Print final summary ──
echo ""
echo ""
echo "============================================================"
echo " SUMMARY — $(date)"
echo "============================================================"
echo ""
cat "$SUMMARY_FILE"
echo ""
echo "Saved to: $SUMMARY_FILE"
echo "Individual results in: $OUTPUT_BASE/"
echo "Done!"
