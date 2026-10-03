"""
Visualize SegVGGT-DPT Joint multi-frame sweep results.

Reads metrics_by_bin.json from
  evaluation/results/segvggt_dpt_joint_sweep/N{2,4,6,8}_{replica,vkitti2}/

Produces a 2-row, 2-col figure:
  Row 1: AUPRC vs N, one line per pose bin (Replica | VKITTI2)
  Row 2: Overall weighted R@1 / R@5 / AUPRC bar chart vs N (Replica | VKITTI2)

Output: evaluation/results/segvggt_dpt_joint_sweep/figure_sweep.{png,pdf}
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SWEEP_DIR = Path(__file__).parent.parent / "results" / "segvggt_dpt_joint_sweep"
N_FRAMES = [2, 4, 6, 8]
DATASETS = ["replica", "vkitti2"]
DATASET_TITLES = {"replica": "Replica (indoor)", "vkitti2": "Virtual KITTI 2 (driving)"}

# Both datasets actually use the same pose bins despite the VKITTI2 config
# claiming [0,20]/[20,40]/[40,60]/[60,90] — the real sampled angles are
# 0-45 / 45-90 / 90-135 / 135-180 (verified against pairs JSONs).
# Use a single canonical labeling for both panels.
CANONICAL_BINS = ["0–45°", "45–90°", "90–135°", "135–180°"]


def canonical_bin(raw_key: str, ds: str) -> str:
    """Map raw bin key from metrics_by_bin.json to canonical label."""
    # Bins are sorted by lower bound, so map by index.
    parts = raw_key.split("-")
    lower = int(parts[0])
    if ds == "replica":
        order = {0: 0, 45: 1, 90: 2, 135: 3}
    else:
        order = {0: 0, 20: 1, 40: 2, 60: 3}
    return CANONICAL_BINS[order[lower]]


def load_sweep() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per_bin_df, overall_df)."""
    per_bin_rows, overall_rows = [], []

    for n in N_FRAMES:
        for ds in DATASETS:
            f = SWEEP_DIR / f"N{n}_{ds}" / "metrics_by_bin.json"
            if not f.exists():
                continue
            data = json.loads(f.read_text())
            bin_keys = sorted(data.keys(), key=lambda b: int(b.split("-")[0]))

            auprc_w = r1_w = r5_w = total = 0
            for b in bin_keys:
                m = data[b]
                nq = m.get("num_queries", 0)
                per_bin_rows.append({
                    "N": n,
                    "Dataset": DATASET_TITLES[ds],
                    "Pose bin": canonical_bin(b, ds),
                    "AUPRC": m.get("AUPRC", 0.0),
                    "R@1": m.get("R@1", 0.0),
                    "R@5": m.get("R@5", 0.0),
                    "num_queries": nq,
                })
                auprc_w += m.get("AUPRC", 0.0) * nq
                r1_w += m.get("R@1", 0.0) * nq
                r5_w += m.get("R@5", 0.0) * nq
                total += nq

            if total > 0:
                overall_rows.append({"N": n, "Dataset": DATASET_TITLES[ds],
                                     "Metric": "AUPRC", "Value": auprc_w / total})
                overall_rows.append({"N": n, "Dataset": DATASET_TITLES[ds],
                                     "Metric": "R@1", "Value": r1_w / total})
                overall_rows.append({"N": n, "Dataset": DATASET_TITLES[ds],
                                     "Metric": "R@5", "Value": r5_w / total})

    return pd.DataFrame(per_bin_rows), pd.DataFrame(overall_rows)


def plot(per_bin: pd.DataFrame, overall: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    palette_bins = sns.color_palette("viridis", n_colors=4)
    palette_metrics = sns.color_palette("rocket_r", n_colors=3)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8),
                             gridspec_kw={"height_ratios": [1.4, 1.0]})

    # Row 1: AUPRC vs N per bin (shared y so panels are directly comparable)
    line_handles = None
    for ax, ds in zip(axes[0], DATASETS):
        title = DATASET_TITLES[ds]
        sub = per_bin[per_bin["Dataset"] == title]
        sns.lineplot(data=sub, x="N", y="AUPRC", hue="Pose bin",
                     hue_order=CANONICAL_BINS,
                     marker="o", markersize=8, linewidth=2.0,
                     palette=palette_bins, ax=ax, legend=(ds == "replica"))
        # Highlight N=6 as the best choice (thicker vertical band)
        ax.axvline(x=6, color="goldenrod", linewidth=3.5, alpha=0.35, zorder=0)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("N (frames per tuple)")
        ax.set_ylabel("AUPRC" if ds == "replica" else "")
        ax.set_xticks(N_FRAMES)
        if ds == "replica":
            line_handles = ax.get_legend_handles_labels()
            ax.get_legend().remove()

    # Single shared legend above both line panels
    if line_handles is not None:
        fig.legend(line_handles[0], line_handles[1],
                   title="Pose bin", loc="upper center",
                   bbox_to_anchor=(0.5, 0.965), ncol=4, frameon=True, fontsize=10)

    # Row 2: overall weighted metrics bar chart
    for ax, ds in zip(axes[1], DATASETS):
        title = DATASET_TITLES[ds]
        sub = overall[overall["Dataset"] == title]
        sns.barplot(data=sub, x="N", y="Value", hue="Metric",
                    palette=palette_metrics, ax=ax,
                    hue_order=["AUPRC", "R@1", "R@5"])
        ax.set_xlabel("N (frames per tuple)")
        ax.set_ylabel("Weighted score" if ds == "replica" else "")
        ax.set_ylim(0, 1.0)
        ax.legend(title="", loc="lower right", frameon=True, fontsize=9, ncol=3)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)

    fig.suptitle("SegVGGT-DPT Joint: effect of N (multi-frame attention) by dataset",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {out_path.with_suffix('.png')}")
    print(f"Saved: {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    per_bin, overall = load_sweep()
    print(f"Loaded {len(per_bin)} per-bin rows, {len(overall)} overall rows")
    plot(per_bin, overall, SWEEP_DIR / "figure_sweep")
