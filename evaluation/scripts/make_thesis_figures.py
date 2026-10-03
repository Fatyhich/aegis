"""
Generate 5 thesis figures (PDF, serif font, 300 dpi):

  1. fig_overall_auprc.pdf      — horizontal bars: overall AUPRC (Replica | VKITTI2)
  2. fig_bins_replica.pdf       — grouped bars: AUPRC by 4 angle bins (Replica)
  3. fig_bins_vkitti2.pdf       — grouped bars: AUPRC by 4 angle bins (VKITTI2)
  4. fig_error_decomposition.pdf — stacked bars: error types × 3 models
  5. fig_training_curves.pdf    — 2×2 train/val loss from TB events
  6. fig_architecture.pdf       — SegVGGT-DPT Joint pipeline schematic

Run:
  source ~/nabo-projects-mounted/claude-venv/bin/activate
  python3 -m evaluation.scripts.make_thesis_figures
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams["font.family"]  = "serif"
plt.rcParams["font.serif"]   = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["pdf.fonttype"] = 42       # editable text in PDF
plt.rcParams["ps.fonttype"]  = 42
plt.rcParams["axes.grid"]    = False
plt.rcParams["savefig.dpi"]  = 300

REPO       = Path(__file__).resolve().parents[2]
OUT_DIR    = REPO / "evaluation" / "thesis_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Palette
C_OURS     = "#D97757"   # orange — our models
C_BASELINE = "#B0B0B0"   # grey   — external baselines
C_METHODS  = {
    "Sinkhorn":  "#8C8C8C",
    "LGv2":      "#D97757",
    "DPT v3":    "#4A90D9",
    "Joint":     "#5BAD8F",
}
C_ERR = {
    "false_dustbin": "#4A90D9",
    "wrong_match":   "#D97757",
    "false_match":   "#C0392B",
}


# ---------------------------------------------------------------------------
# Figure 1 — overall AUPRC
# ---------------------------------------------------------------------------
def fig_overall_auprc():
    replica = [
        ("SegMASt3R+LGv2 (ours)",    84.48, True),
        ("SegVGGT-DPT Joint (ours)", 81.37, True),
        ("SegVGGT-DPT v3 (ours)",    79.70, True),
        ("SegMASt3R (Sinkhorn)",     79.63, False),
        ("DA3 Sequential",           54.29, False),
        ("vizEnc+DINOv2",            49.48, False),
        ("vizEnc+Radio",             40.69, False),
        ("LiftFeat",                 35.69, False),
    ]
    vkitti = [
        ("SegMASt3R+LGv2 (ours)",    82.78, True),
        ("SegMASt3R (Sinkhorn)",     56.91, False),
        ("SegVGGT-DPT v3 (ours)",    52.28, True),
        ("DA3 Sequential",           40.67, False),
        ("LiftFeat",                 32.35, False),
        ("vizEnc+DINOv2",            30.22, False),
        ("vizEnc+Radio",             27.54, False),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))

    for ax, data, label in zip(axes, [replica, vkitti], ["Replica", "Virtual KITTI 2"]):
        names  = [d[0] for d in data]
        vals   = [d[1] for d in data]
        ours   = [d[2] for d in data]
        colors = [C_OURS if o else C_BASELINE for o in ours]

        y = np.arange(len(names))
        ax.barh(y, vals, color=colors, edgecolor="none", height=0.68)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel(f"Overall AUPRC (%) — {label}", fontsize=11)
        ax.set_xlim(0, max(vals) * 1.13)

        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.008, i, f"{v:.2f}",
                    va="center", fontsize=9.5)

        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)

    # Shared legend (ours / baseline)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_OURS,     label="Ours"),
        plt.Rectangle((0, 0), 1, 1, color=C_BASELINE, label="Baselines"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = OUT_DIR / "fig_overall_auprc.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Figure 2 — AUPRC by angle bins (one file per dataset)
# ---------------------------------------------------------------------------
def _grouped_bars(ax, data, y_min, x_labels):
    """data: dict[method_name] = list of AUPRC values over x_labels."""
    methods = list(data.keys())
    n_groups   = len(x_labels)
    n_methods  = len(methods)
    bar_w      = 0.78 / n_methods
    x_base     = np.arange(n_groups)

    for i, method in enumerate(methods):
        vals  = data[method]
        offs  = (i - (n_methods - 1) / 2) * bar_w
        color = C_METHODS[method]
        bars = ax.bar(x_base + offs, vals, width=bar_w,
                      color=color, edgecolor="none", label=method)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.6,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x_base)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylim(y_min, 100)
    ax.set_ylabel("AUPRC (%)", fontsize=11)
    ax.set_xlabel("Camera rotation angle bin", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_bins_replica():
    data = {
        "Sinkhorn": [80.48, 77.04, 79.83, 81.72],
        "LGv2":     [91.17, 83.59, 77.09, 70.28],
        "DPT v3":   [81.20, 80.93, 78.07, 73.36],
        "Joint":    [81.66, 83.03, 81.85, 77.69],
    }
    x_labels = ["0–45°", "45–90°", "90–135°", "135–180°"]

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    _grouped_bars(ax, data, y_min=30, x_labels=x_labels)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32),
              ncol=4, frameon=False, fontsize=10,
              handlelength=1.5, columnspacing=1.5)
    fig.tight_layout()
    out = OUT_DIR / "fig_bins_replica.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_bins_vkitti2():
    data = {
        "Sinkhorn": [57.98, 55.96, 57.26, 49.06],
        "LGv2":     [79.94, 88.52, 92.16, 78.34],
        "DPT v3":   [46.60, 58.69, 68.72, 73.29],
        "Joint":    [47.35, 57.33, 66.24, 71.05],   # N=2
    }
    x_labels = ["0–45°", "45–90°", "90–135°", "135–180°"]

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    _grouped_bars(ax, data, y_min=20, x_labels=x_labels)

    # Note about Joint N=2
    ax.text(0.99, 0.02, "Joint: N = 2 (single full sweep available)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, style="italic", color="#555555")

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32),
              ncol=4, frameon=False, fontsize=10,
              handlelength=1.5, columnspacing=1.5)
    fig.tight_layout()
    out = OUT_DIR / "fig_bins_vkitti2.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Figure 3 — error decomposition
# ---------------------------------------------------------------------------
def fig_error_decomposition():
    models        = ["DPT v2", "DPT v3"]
    subtitles     = ["v2: unclamped τ", r"v3: τ $\leq$ 1.0, $\lambda$=0.3"]
    false_dustbin = np.array([0.059, 0.038])
    wrong_match   = np.array([0.035, 0.033])
    false_match   = np.array([0.024, 0.031])

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    x = np.arange(len(models))
    w = 0.48

    b1 = ax.bar(x, false_dustbin, w, color=C_ERR["false_dustbin"],
                label="false_dustbin (GT match → model said dustbin)")
    b2 = ax.bar(x, wrong_match, w, bottom=false_dustbin,
                color=C_ERR["wrong_match"],
                label="wrong_match (matched, but picked wrong j)")
    b3 = ax.bar(x, false_match, w, bottom=false_dustbin + wrong_match,
                color=C_ERR["false_match"],
                label="false_match (no GT, yet model matched)")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Error rate (fraction of segments)", fontsize=11)
    ax.set_ylim(0, 0.14)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Subtitles placed in the lower margin, clear of tick labels
    for xi, sub in zip(x, subtitles):
        ax.text(xi, -0.085, sub, ha="center", va="top",
                fontsize=9.5, color="#555555", style="italic",
                transform=ax.get_xaxis_transform())

    # Per-segment value labels (white, bold)
    for container, base in [(b1, np.zeros(2)),
                            (b2, false_dustbin),
                            (b3, false_dustbin + wrong_match)]:
        for rect, offset in zip(container, base):
            h = rect.get_height()
            if h < 0.012:
                continue
            ax.text(rect.get_x() + rect.get_width() / 2,
                    offset + h / 2,
                    f"{h:.3f}", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")

    # Annotate -36% drop in false_dustbin (v2 → v3)
    y_v2 = false_dustbin[0]       # 0.059
    y_v3 = false_dustbin[1]       # 0.038
    ax.annotate(
        "",
        xy=(x[1] - w / 2 - 0.02, y_v3),
        xytext=(x[0] + w / 2 + 0.02, y_v2),
        arrowprops=dict(arrowstyle="->", color="#1F3A66",
                        lw=1.6, connectionstyle="arc3,rad=-0.25"),
    )
    ax.text((x[0] + x[1]) / 2, (y_v2 + y_v3) / 2 + 0.018,
            r"$-36\%$", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#1F3A66")

    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "fig_error_decomposition.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Figure 4 — training curves (2×2)
# ---------------------------------------------------------------------------
def _load_tb_scalar(tb_dir: str, tag: str):
    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return np.array([]), np.array([])
    evs = ea.Scalars(tag)
    return (np.array([e.step for e in evs]),
            np.array([e.value for e in evs]))


def fig_training_curves():
    runs = [
        ("SegMASt3R + LGv2",   REPO / "results" / "segmast3r_lg_v2" / "tb"),
        ("SegVGGT (precomp.)", REPO / "results" / "segvggt"          / "tb"),
        ("SegVGGT-DPT v3",     REPO / "results" / "segvggt_dpt" / "v3-001" / "tb"),
        ("SegVGGT-DPT Joint",  REPO / "results" / "segvggt_dpt" / "joint-001" / "tb"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()

    for ax, (name, tb_path) in zip(axes, runs):
        s_tr, v_tr = _load_tb_scalar(str(tb_path), "train/loss")
        s_vl, v_vl = _load_tb_scalar(str(tb_path), "val/loss")

        ax.plot(s_tr, v_tr, color="#2E4A7B", lw=1.0, alpha=0.85, label="train/loss")
        if s_vl.size:
            ax.plot(s_vl, v_vl, color=C_OURS, lw=1.8, linestyle="--",
                    marker="o", markersize=3.5, label="val/loss")

        ax.set_xlabel("global step", fontsize=10)
        ax.set_ylabel("loss", fontsize=10)
        ax.text(0.02, 0.96, name, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left")

        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        # Highlight train/val gap on Joint
        if name == "SegVGGT-DPT Joint" and s_vl.size and s_tr.size:
            # Annotate the final gap
            final_tr = v_tr[-1]
            final_vl = v_vl[-1]
            x_last   = max(s_tr[-1], s_vl[-1])
            ax.annotate(
                "",
                xy=(x_last * 0.93, final_tr),
                xytext=(x_last * 0.93, final_vl),
                arrowprops=dict(arrowstyle="<->", color="#C0392B", lw=1.6),
            )
            ax.text(x_last * 0.91, (final_tr + final_vl) / 2,
                    f"gap\n+{final_vl - final_tr:.2f}",
                    fontsize=9, color="#C0392B", ha="right", va="center",
                    fontweight="bold")

        ax.legend(loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / "fig_training_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Figure 5 — architecture diagram (SegVGGT-DPT Joint)
# ---------------------------------------------------------------------------
def _box(ax, xy, wh, text, frozen=False, fontsize=9.5, zorder=2):
    x, y = xy
    w, h = wh
    if frozen:
        face  = "#EDEDED"
        edge  = "#606060"
        style = (0, (4, 3))
        weight = "normal"
    else:
        face  = "#FCE4D6"
        edge  = C_OURS
        style = "solid"
        weight = "bold"
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.6 if not frozen else 1.1,
        edgecolor=edge, facecolor=face, linestyle=style, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight=weight, zorder=zorder + 1)


def _arrow(ax, start, end, lw=1.2, color="#404040", zorder=1):
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=12,
        color=color, lw=lw, zorder=zorder,
    ))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(11.5, 9.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12.5)
    ax.axis("off")

    # Column x-centers for 3 frames (frame N rendered as "frame N")
    cx   = [3.5, 6.7, 9.9]
    box_w = 2.1
    box_h = 0.72

    # Row y-centers (top → bottom)
    Y = {
        "image":    11.2,
        "encoder":  10.0,
        "layers":    8.9,
        "dpt":       7.7,
        "pool":      6.4,
        "proj":      5.2,
        "embed":     4.1,
        "joint":     2.8,
        "matcher":   1.3,
    }

    # Dots between frame 2 and frame N
    ax.text((cx[1] + cx[2]) / 2, Y["image"] + box_h / 2, r"$\cdots$",
            ha="center", va="center", fontsize=18, color="#606060")

    # 1. Input images
    for i, x in enumerate(cx):
        label = [f"Image 1", f"Image 2", f"Image N"][i]
        _box(ax, (x - box_w / 2, Y["image"]), (box_w, box_h), label,
             frozen=False, fontsize=10)
        # "ours" outline off — images aren't trainable; use neutral style
        # Override with neutral look
    # repaint image boxes neutral
    for i, x in enumerate(cx):
        label = ["Image 1", "Image 2", "Image N"][i]
        ax.add_patch(FancyBboxPatch(
            (x - box_w / 2, Y["image"]), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.0, edgecolor="#888888", facecolor="white",
            zorder=3,
        ))
        ax.text(x, Y["image"] + box_h / 2, label,
                ha="center", va="center", fontsize=10, zorder=4)

    # 2. VGGT Encoder — single wide block spanning all 3 columns (shared)
    enc_x0 = cx[0] - box_w / 2
    enc_x1 = cx[2] + box_w / 2
    _box(ax, (enc_x0, Y["encoder"]), (enc_x1 - enc_x0, box_h),
         "VGGT Encoder (frozen, shared weights)",
         frozen=True, fontsize=10.5)

    # 3. VGGT Aggregator layers
    _box(ax, (enc_x0, Y["layers"]), (enc_x1 - enc_x0, box_h),
         r"VGGT Aggregator — extract layers $\{5, 11, 17, 23\}$  (frozen)",
         frozen=True, fontsize=10)

    # 4. DPT Fusion — one per frame
    for x in cx:
        _box(ax, (x - box_w / 2, Y["dpt"]), (box_w, box_h),
             "DPT Fusion\nConv1×1 + 2×RCU", frozen=False, fontsize=9)

    # 5. Masked pooling + SAM branch
    #    SAM branch on the left: box at x ≈ 0.4 pointing into pooling row
    sam_w, sam_h = 1.9, 0.78
    sam_x, sam_y = 0.35, Y["pool"] - 0.03
    ax.add_patch(FancyBboxPatch(
        (sam_x, sam_y), sam_w, sam_h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.0, edgecolor="#666666", facecolor="#F2F2F2",
        linestyle=(0, (4, 3)), zorder=3,
    ))
    ax.text(sam_x + sam_w / 2, sam_y + sam_h / 2,
            "SAM masks\n(precomputed)",
            ha="center", va="center", fontsize=9, zorder=4)

    # Pooling row: one per frame
    for x in cx:
        _box(ax, (x - box_w / 2, Y["pool"]), (box_w, box_h),
             "Masked Avg Pool", frozen=False, fontsize=9.5)
    # Arrow SAM → first pooling box
    _arrow(ax,
           (sam_x + sam_w, sam_y + sam_h / 2),
           (cx[0] - box_w / 2 - 0.05, Y["pool"] + box_h / 2),
           lw=1.3, color="#666666")
    # Horizontal feed across pooling row (SAM provides masks for all frames)
    ax.plot([cx[0] + box_w / 2, cx[2] - box_w / 2],
            [Y["pool"] + box_h / 2, Y["pool"] + box_h / 2],
            color="#BBBBBB", lw=0.7, linestyle=":", zorder=0)

    # 6. Projection Linear 256→128 + LN
    for x in cx:
        _box(ax, (x - box_w / 2, Y["proj"]), (box_w, box_h),
             "Linear 256 → 128\n+ LayerNorm", frozen=False, fontsize=9)

    # 7. Frame embeddings (added)
    for i, x in enumerate(cx):
        _box(ax, (x - box_w / 2, Y["embed"]), (box_w, box_h),
             f"+ Frame embed[{['1','2','N'][i]}]", frozen=False, fontsize=9)

    # 8. Joint self-attention — wide block
    _box(ax, (enc_x0, Y["joint"]), (enc_x1 - enc_x0, box_h + 0.18),
         r"Joint Self-Attention $\times 3$  over all segments of all $N$ frames",
         frozen=False, fontsize=11)

    # 9. Matcher
    mw = 5.0
    _box(ax, ((enc_x0 + enc_x1) / 2 - mw / 2, Y["matcher"]),
         (mw, box_h + 0.12),
         r"DoubleSoftmax Matcher → pairwise score $S(i, j)$",
         frozen=False, fontsize=10.5)

    # ── Vertical arrows per column (image → encoder → layers → dpt → pool → proj → embed → joint)
    rows_seq = ["image", "encoder", "layers", "dpt", "pool", "proj", "embed", "joint"]
    for x in cx:
        for a, b in zip(rows_seq[:-1], rows_seq[1:]):
            y_top = Y[a]                          # bottom of upper box
            y_bot = Y[b] + box_h + (0.18 if b == "joint" else 0)
            _arrow(ax, (x, y_top), (x, y_bot), lw=1.1)

    # Joint → matcher: one central arrow
    _arrow(ax,
           (cx[1], Y["joint"]),
           (cx[1], Y["matcher"] + box_h + 0.12),
           lw=1.3, color=C_OURS)

    # ── Legend box: frozen vs trainable + param counts
    leg_x0, leg_y0 = 11.25, 9.85          # bottom-left corner (data units)
    leg_w,  leg_h  = 2.55, 2.25
    leg_x1 = leg_x0 + leg_w
    leg_y1 = leg_y0 + leg_h

    ax.add_patch(FancyBboxPatch(
        (leg_x0, leg_y0), leg_w, leg_h,
        boxstyle="round,pad=0.05", linewidth=0.8,
        edgecolor="#555555", facecolor="white", zorder=5,
    ))

    sw_w, sw_h = 0.55, 0.30
    sw_x       = leg_x0 + 0.22
    label_x    = sw_x + sw_w + 0.15

    # frozen sample (top row)
    frz_y = leg_y1 - 0.45
    ax.add_patch(FancyBboxPatch(
        (sw_x, frz_y), sw_w, sw_h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.0, edgecolor="#606060", facecolor="#EDEDED",
        linestyle=(0, (3, 2)), zorder=6,
    ))
    ax.text(label_x, frz_y + sw_h / 2, "frozen",
            fontsize=9.5, va="center", ha="left", zorder=7)

    # trainable sample
    trn_y = frz_y - 0.55
    ax.add_patch(FancyBboxPatch(
        (sw_x, trn_y), sw_w, sw_h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.6, edgecolor=C_OURS, facecolor="#FCE4D6", zorder=6,
    ))
    ax.text(label_x, trn_y + sw_h / 2, "trainable",
            fontsize=9.5, va="center", ha="left", zorder=7)

    # divider
    div_y = trn_y - 0.28
    ax.plot([leg_x0 + 0.22, leg_x1 - 0.22], [div_y, div_y],
            color="#BBBBBB", lw=0.7, zorder=6)

    # param counts (two separate lines → exact vertical control)
    ax.text(leg_x0 + leg_w / 2, div_y - 0.30,
            r"$\sim$10.7M trainable",
            ha="center", va="center", fontsize=9.5, zorder=7)
    ax.text(leg_x0 + leg_w / 2, div_y - 0.62,
            r"$\sim$1B frozen",
            ha="center", va="center", fontsize=9.5, zorder=7)

    fig.tight_layout()
    out = OUT_DIR / "fig_architecture.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fig_overall_auprc()
    fig_bins_replica()
    fig_bins_vkitti2()
    fig_error_decomposition()
    fig_training_curves()
    fig_architecture()
    print(f"\nAll figures saved to: {OUT_DIR}")
