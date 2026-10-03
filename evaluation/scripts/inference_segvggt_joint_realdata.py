"""
SegVGGT-DPT Joint inference + visualization on real photos (no GT).

Per scene:
  1. FastSAM segmentation → per-frame masks (cached in RAM).
  2. Forward through joint model:
     - scene_02-trunc (8 frames): single-shot, N=8, all C(8,2)=28 pairs.
     - other scenes:  sliding window W=6, stride=2.  Per-window all 15 pairs.
  3. Cross-window match aggregation: best-score-per-source-segment, then
     union-find over (frame, local_seg_id) nodes → global instance IDs.
  4. Per-frame overlay PNG with consistent color per global ID.
  5. Timing breakdown: VGGT Aggregator (backbone) vs DPT-fusion+matching head.

Run:
  source ~/nabo-projects-mounted/claude-venv/bin/activate
  python3 evaluation/scripts/inference_segvggt_joint_realdata.py \
    --config evaluation/configs/config_eval_vkitti2_segvggt_joint.yaml \
    --data_root data \
    --output_dir results/segvggt_joint_realdata
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from PIL import Image
from PIL.ImageOps import exif_transpose

# Make `evaluation.*` and `training.*` importable
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_EVAL_ROOT = _PROJECT_ROOT / "evaluation"
for p in (_PROJECT_ROOT, _EVAL_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.model_infer import SegVGGTDPTJointInfer, _mutual_match_filter  # noqa: E402


IMG_NORM = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
TARGET_H = 336
TARGET_W = 512
M_PRIME = 50
WINDOW_SIZE = 6
WINDOW_STRIDE = 2
DISPLAY_H = 540
DISPLAY_W = 960  # 16:9-ish for 4K source readability


# ---------------------------------------------------------------------------
# Image / mask helpers
# ---------------------------------------------------------------------------

def load_image_tensor(path: Path) -> torch.Tensor:
    """Resize to TARGET_H × TARGET_W, normalize to [-1, 1]."""
    img = Image.open(path).convert("RGB")
    img = exif_transpose(img)
    img = img.resize((TARGET_W, TARGET_H), Image.BILINEAR)
    return IMG_NORM(img)


def load_image_rgb_for_display(path: Path) -> np.ndarray:
    """Load original frame, downscale for overlay rendering."""
    img = Image.open(path).convert("RGB")
    img = exif_transpose(img)
    img = img.resize((DISPLAY_W, DISPLAY_H), Image.BILINEAR)
    return np.array(img)


def list_frames(scene_dir: Path) -> list[Path]:
    return sorted((scene_dir / "frames").glob("*.jpg"))


# ---------------------------------------------------------------------------
# FastSAM wrapper
# ---------------------------------------------------------------------------

def init_segmentor(weights: str = "FastSAM-x.pt", device: str = "cuda:0"):
    from ultralytics import FastSAM
    print(f"[seg] loading FastSAM weights = {weights}")
    return FastSAM(weights)


def segment_frame(seg_model, image_path: Path, device: str = "cuda:0",
                  imgsz: int = 1024, conf: float = 0.4, iou: float = 0.9,
                  ) -> torch.Tensor:
    """
    Run FastSAM on a single frame, return masks resized to TARGET_H×TARGET_W.

    Returns:
        (M, TARGET_H, TARGET_W) uint8 tensor on CPU. M ≤ M_PRIME, sorted by area.
    """
    results = seg_model(
        str(image_path),
        device=device,
        retina_masks=True,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
    )
    r = results[0]
    if r.masks is None or r.masks.data.shape[0] == 0:
        return torch.zeros((0, TARGET_H, TARGET_W), dtype=torch.uint8)

    masks = r.masks.data.detach().to("cpu", dtype=torch.float32)  # (M, H, W)
    # Resize to target via nearest
    masks = F.interpolate(
        masks.unsqueeze(1), size=(TARGET_H, TARGET_W), mode="nearest"
    ).squeeze(1)
    masks = (masks > 0.5).to(torch.uint8)

    # Sort by area descending
    areas = masks.float().sum(dim=(1, 2))
    order = torch.argsort(areas, descending=True)
    masks = masks[order]
    if masks.shape[0] > M_PRIME:
        masks = masks[:M_PRIME]
    # Drop empty masks (area 0) just in case
    nonempty = masks.float().sum(dim=(1, 2)) > 0
    masks = masks[nonempty]
    return masks.contiguous()


# ---------------------------------------------------------------------------
# Forward pass with backbone/matching split timing
# ---------------------------------------------------------------------------

def forward_window(
    inner,                 # SegVGGTDPTJoint module (model.inner)
    images: torch.Tensor,  # (1, N, 3, H, W) on device, [-1, 1]
    masks_list: list[torch.Tensor],  # list[N] of (1, M_v, H, W) on device
    pair_indices: list[tuple[int, int]],
    device: torch.device,
) -> tuple[dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
           float, float]:
    """
    Replicates SegVGGTDPTJoint.forward_multiframe but splits time:
      - backbone_s = VGGT Aggregator
      - matching_s = DPT fusion + masked pool + projection + joint attn + matcher
    """
    is_cuda = device.type == "cuda"

    with torch.no_grad():
        # --- Backbone -----------------------------------------------------
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        all_feats, H_p, W_p = inner.extract_multi_layer_nframes(images)
        if is_cuda:
            torch.cuda.synchronize()
        backbone_s = time.perf_counter() - t0

        # --- Matching head ------------------------------------------------
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        N = images.shape[1]
        fused = [inner.fusion(all_feats[v]) for v in range(N)]

        proj_dtype = next(inner.proj.parameters()).dtype
        projs = []
        for v in range(N):
            m = masks_list[v]
            if m.shape[-2:] != (H_p, W_p):
                m = F.interpolate(m.float(), (H_p, W_p), mode="nearest")
            # masked average pooling — bmm form
            B, D, H, W = fused[v].shape
            M = m.shape[1]
            feat_flat = fused[v].view(B, D, H * W)
            masks_flat = m.view(B, M, H * W).float()
            area = masks_flat.sum(dim=-1, keepdim=True).clamp(min=1)
            dsc = torch.bmm(feat_flat, masks_flat.transpose(1, 2)) / area.transpose(1, 2)
            proj = inner.proj(dsc.transpose(1, 2).to(dtype=proj_dtype))
            projs.append(proj)

        # Joint self-attention (with frame embeddings on first layer)
        for i, layer in enumerate(inner.joint_attn_layers):
            projs = layer(projs, frame_embeds_added=(i > 0))

        results: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for (a, b) in pair_indices:
            log_mutual, m0, m1 = inner.matcher(
                projs[a].transpose(1, 2), projs[b].transpose(1, 2)
            )
            results[(a, b)] = (log_mutual, m0, m1)

        if is_cuda:
            torch.cuda.synchronize()
        matching_s = time.perf_counter() - t0

    return results, backbone_s, matching_s


# ---------------------------------------------------------------------------
# Union-Find for global instance IDs
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def add(self, node):
        if node not in self.parent:
            self.parent[node] = node

    def find(self, node):
        # Iterative path-compression
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:
            self.parent[node], node = root, self.parent[node]
        return root

    def union(self, a, b):
        self.add(a); self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ---------------------------------------------------------------------------
# Match extraction from window output
# ---------------------------------------------------------------------------

def extract_pair_matches(
    log_mutual: torch.Tensor,  # (1, M, N)
    match0: torch.Tensor,      # (1, M)
    M: int, N: int,
    device: torch.device,
) -> list[tuple[int, int, float]]:
    """
    Apply mutual + matchability filter, return list of (i, j, score).
    score = exp(log_mutual)[i, j] in [0, 1] (proxy of confidence).
    """
    scores = torch.exp(log_mutual)            # (1, M, N)
    scores = scores[:, :M, :N]
    match_result = _mutual_match_filter(scores, match0[:, :M], device)  # (1, M)
    out: list[tuple[int, int, float]] = []
    mr = match_result[0]
    sc = scores[0]
    for i in range(M):
        j = int(mr[i].item())
        if j < 0:
            continue
        out.append((i, j, float(sc[i, j].item())))
    return out


def update_best_matches(
    best: dict[tuple[int, int], dict[int, tuple[int, float]]],
    f_a: int, f_b: int,
    pair_matches: list[tuple[int, int, float]],
):
    key = (f_a, f_b)
    if key not in best:
        best[key] = {}
    inner = best[key]
    for (i, j, sc) in pair_matches:
        prev = inner.get(i)
        if prev is None or sc > prev[1]:
            inner[i] = (j, sc)


def build_global_instance_ids(
    n_frames: int,
    masks_per_frame: list[torch.Tensor],
    best_matches: dict[tuple[int, int], dict[int, tuple[int, float]]],
) -> tuple[dict[tuple[int, int], int], int]:
    """
    Build global instance IDs via constrained union-find.

    Two safeguards against clustering distinct same-frame segments together:
      (1) Per-pair (f_a, f_b) bipartite filtering — each j in f_b kept by
          only one i in f_a (highest score).
      (2) Greedy union-find ordered by score: reject any merge that would
          place two segments from the same frame into one component.

    Returns:
        gid_of: {(frame_idx, local_seg_idx): global_instance_id}
        n_global: number of unique global IDs
    """
    # 1. Per-pair bipartite filter: enforce j-side injectivity too.
    edges: list[tuple[float, int, int, int, int]] = []
    for (f_a, f_b), inner in best_matches.items():
        # inner: {i: (j, score)} — already injective on i.
        # Keep, per j, only the best i pointing to it.
        j_best: dict[int, tuple[int, float]] = {}
        for i, (j, sc) in inner.items():
            prev = j_best.get(j)
            if prev is None or sc > prev[1]:
                j_best[j] = (i, sc)
        for j, (i, sc) in j_best.items():
            edges.append((sc, f_a, i, f_b, j))
    edges.sort(key=lambda e: -e[0])  # high score first

    # 2. Constrained union-find — reject merges that conflict on frames.
    uf = UnionFind()
    # Frames represented in each component, keyed by component root.
    comp_frames: dict = {}
    for f, masks in enumerate(masks_per_frame):
        for i in range(masks.shape[0]):
            node = (f, i)
            uf.add(node)
            comp_frames[node] = {f}

    for (_sc, f_a, i, f_b, j) in edges:
        a = (f_a, i); b = (f_b, j)
        ra = uf.find(a); rb = uf.find(b)
        if ra == rb:
            continue
        fa_set = comp_frames[ra]
        fb_set = comp_frames[rb]
        if fa_set & fb_set:
            # Same frame appears in both components → merging would put two
            # segments of one frame into one global instance. Reject.
            continue
        uf.union(a, b)
        new_root = uf.find(a)
        merged = fa_set | fb_set
        comp_frames[new_root] = merged
        # Stale entries for old roots are harmless (never queried again)
        # since UF.find always returns the canonical root.

    # 3. Canonicalize root IDs in order of first appearance.
    root_to_gid: dict = {}
    gid_of: dict = {}
    for f, masks in enumerate(masks_per_frame):
        for i in range(masks.shape[0]):
            root = uf.find((f, i))
            if root not in root_to_gid:
                root_to_gid[root] = len(root_to_gid)
            gid_of[(f, i)] = root_to_gid[root]
    return gid_of, len(root_to_gid)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_palette(n: int) -> np.ndarray:
    """Golden-ratio HSV palette → (n, 3) uint8."""
    import colorsys
    colors = []
    for k in range(n):
        h = (k * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.95)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.array(colors, dtype=np.uint8)


def render_overlay(
    rgb: np.ndarray,            # (H_disp, W_disp, 3) uint8
    masks_336x512: torch.Tensor,  # (M, 336, 512) uint8
    gids: list[int],
    palette: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Draw masks (resized to display res) on top of rgb with per-gid color + label."""
    H_disp, W_disp = rgb.shape[:2]
    out = rgb.astype(np.float32).copy()
    M = masks_336x512.shape[0]
    if M == 0:
        return rgb.copy()

    masks_disp = F.interpolate(
        masks_336x512.float().unsqueeze(0),
        size=(H_disp, W_disp), mode="nearest"
    ).squeeze(0).numpy().astype(bool)  # (M, H, W)

    # Composite — paint biggest first so small overlays show on top
    for i in range(M):
        col = palette[gids[i] % len(palette)].astype(np.float32)
        m = masks_disp[i]
        out[m] = (1.0 - alpha) * out[m] + alpha * col

    out = np.clip(out, 0, 255).astype(np.uint8)

    # Annotate centroid with gid
    try:
        from PIL import ImageDraw, ImageFont
        pil = Image.fromarray(out)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
            )
        except Exception:
            font = ImageFont.load_default()
        for i in range(M):
            m = masks_disp[i]
            ys, xs = np.where(m)
            if len(xs) < 30:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            text = str(gids[i])
            # halo for legibility
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                draw.text((cx + dx, cy + dy), text, fill=(0, 0, 0), font=font)
            draw.text((cx, cy), text, fill=(255, 255, 255), font=font)
        out = np.array(pil)
    except Exception as exc:
        print(f"[viz] annotation skipped: {exc}")

    return out


# ---------------------------------------------------------------------------
# Sliding-window GIF
# ---------------------------------------------------------------------------

def render_sliding_window_gif(
    frame_paths: list[Path],
    masks_per_frame: list[torch.Tensor],
    gid_of: dict,
    palette: np.ndarray,
    track_lengths: dict[int, int],
    min_track_len: int,
    window_size: int,
    out_path: Path,
    tile_w: int = 480,
    tile_h: int = 270,
    gap: int = 8,
    duration_ms: int = 600,
    line_min_track_len: int = 5,
):
    """
    Render a right-to-left sliding-window GIF.

    Each animation tick shows `window_size` consecutive frames placed
    horizontally (oldest left, newest right). Per tick the window advances
    by 1 — so visually frames slide leftward across the screen.

    Only segments whose global instance ID has track length >= min_track_len
    are drawn (filters false matches).

    For tracks with len >= line_min_track_len, the centroid of the same
    global instance is connected with a colored line between adjacent
    frames in the window — visualizing the trajectory of the match.
    """
    from PIL import ImageDraw, ImageFont

    n = len(frame_paths)
    if n < window_size:
        print(f"  [gif] skipped — only {n} frames, need >= {window_size}")
        return

    # Centroid of every kept segment in tile-coordinates: {(frame, gid): (cx, cy)}
    centroids: dict[tuple[int, int], tuple[int, int]] = {}

    # Pre-render per-frame tile (RGB + filtered colored masks + index label)
    tiles: list[np.ndarray] = []
    n_kept_total = 0
    for f, fp in enumerate(frame_paths):
        rgb = np.array(
            Image.open(fp).convert("RGB").resize((tile_w, tile_h), Image.BILINEAR)
        )
        masks = masks_per_frame[f]
        keep_idx: list[int] = []
        keep_gids: list[int] = []
        for i in range(masks.shape[0]):
            gid = gid_of[(f, i)]
            if track_lengths.get(gid, 0) >= min_track_len:
                keep_idx.append(i)
                keep_gids.append(gid)
        if keep_idx:
            kept = masks[keep_idx]
        else:
            kept = torch.zeros((0, *masks.shape[1:]), dtype=masks.dtype)
        n_kept_total += kept.shape[0]

        # Compute centroids at tile resolution for kept masks (for the lines)
        if kept.shape[0] > 0:
            kept_at_tile = F.interpolate(
                kept.float().unsqueeze(0), size=(tile_h, tile_w), mode="nearest"
            ).squeeze(0).numpy().astype(bool)
            for k_local, gid in enumerate(keep_gids):
                if track_lengths.get(gid, 0) < line_min_track_len:
                    continue
                m = kept_at_tile[k_local]
                ys, xs = np.where(m)
                if len(xs) >= 30:
                    centroids[(f, gid)] = (int(xs.mean()), int(ys.mean()))

        tile = render_overlay(rgb, kept, keep_gids, palette, alpha=0.55)

        # Index/header label
        try:
            pil = Image.fromarray(tile)
            draw = ImageDraw.Draw(pil)
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16,
                )
            except Exception:
                font = ImageFont.load_default()
            label = f"{fp.stem}  ({len(keep_idx)})"
            draw.rectangle([(0, 0), (170, 26)], fill=(0, 0, 0))
            draw.text((6, 4), label, fill=(255, 255, 255), font=font)
            tile = np.array(pil)
        except Exception:
            pass
        tiles.append(tile)

    print(f"  [gif] kept segments after track-len>={min_track_len} filter: "
          f"{n_kept_total} (total = {sum(m.shape[0] for m in masks_per_frame)})")
    n_line_tracks = sum(1 for v in track_lengths.values() if v >= line_min_track_len)
    print(f"  [gif] connector lines drawn for {n_line_tracks} tracks "
          f"(track len >= {line_min_track_len})")

    # Compose horizontally per tick
    strip_w = window_size * tile_w + (window_size - 1) * gap
    strip_h = tile_h + 24  # bottom strip for tick caption

    gif_frames: list[Image.Image] = []
    n_ticks = n - window_size + 1
    for t in range(n_ticks):
        canvas = np.full((strip_h, strip_w, 3), 22, dtype=np.uint8)
        for w in range(window_size):
            x = w * (tile_w + gap)
            canvas[:tile_h, x:x + tile_w] = tiles[t + w]

        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)

        # Connector lines for long tracks across adjacent frames within window
        for w_idx in range(window_size - 1):
            f_a = t + w_idx
            f_b = t + w_idx + 1
            x_off_a = w_idx * (tile_w + gap)
            x_off_b = (w_idx + 1) * (tile_w + gap)
            for gid, tlen in track_lengths.items():
                if tlen < line_min_track_len:
                    continue
                pa = centroids.get((f_a, gid))
                pb = centroids.get((f_b, gid))
                if pa is None or pb is None:
                    continue
                color = tuple(int(c) for c in palette[gid % len(palette)])
                p1 = (x_off_a + pa[0], pa[1])
                p2 = (x_off_b + pb[0], pb[1])
                # Black halo for legibility
                draw.line([p1, p2], fill=(0, 0, 0), width=4)
                draw.line([p1, p2], fill=color, width=2)
                # Endpoint dots
                for (cx, cy) in (p1, p2):
                    draw.ellipse(
                        [(cx - 4, cy - 4), (cx + 4, cy + 4)],
                        fill=color, outline=(0, 0, 0),
                    )

        # Bottom caption
        try:
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14,
                )
            except Exception:
                font = ImageFont.load_default()
            cap = (f"window {t + 1}/{n_ticks}  frames "
                   f"{frame_paths[t].stem}…{frame_paths[t + window_size - 1].stem}  "
                   f"  masks ≥ {min_track_len}  lines ≥ {line_min_track_len}")
            draw.text((6, tile_h + 4), cap, fill=(220, 220, 220), font=font)
        except Exception:
            pass
        gif_frames.append(pil)

    gif_frames[0].save(
        out_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"  [gif] saved {out_path}  ({len(gif_frames)} ticks, "
          f"{strip_w}x{strip_h}px)")


# ---------------------------------------------------------------------------
# Per-scene runners
# ---------------------------------------------------------------------------

def run_scene(
    scene_dir: Path,
    out_dir: Path,
    model: SegVGGTDPTJointInfer | None,
    seg_model,
    device: torch.device,
    mode: str,  # "single" or "sliding"
    make_gif: bool = True,
    gif_window: int = 4,
    min_track_len: int = 3,
    line_min_track_len: int = 5,
    from_cache: bool = False,
):
    print(f"\n========== {scene_dir.name} ({mode}) ==========")
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = out_dir / "frame_overlays"
    overlays_dir.mkdir(exist_ok=True)

    frame_paths = list_frames(scene_dir)
    n_frames = len(frame_paths)
    print(f"  frames: {n_frames}")

    cache_path = out_dir / "cache.pt"
    if from_cache and cache_path.exists():
        print(f"  [cache] loading {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        masks_per_frame = cache["masks_per_frame"]
        gid_of = {tuple(k): v for k, v in cache["gid_of"]}
        track_lengths = cache["track_lengths"]
        n_global = cache["n_global"]
        timings = cache["timings"]
        # Re-render overlays + (optionally) GIF only.
        palette = make_palette(max(n_global, 32))
        for f, fp in enumerate(frame_paths):
            rgb = load_image_rgb_for_display(fp)
            gids = [gid_of[(f, i)] for i in range(masks_per_frame[f].shape[0])]
            out_img = render_overlay(rgb, masks_per_frame[f], gids, palette)
            Image.fromarray(out_img).save(overlays_dir / f"{fp.stem}.png")
        if make_gif:
            render_sliding_window_gif(
                frame_paths, masks_per_frame, gid_of, palette,
                track_lengths, min_track_len=min_track_len,
                window_size=gif_window,
                line_min_track_len=line_min_track_len,
                out_path=out_dir
                / f"sliding_W{gif_window}_T{min_track_len}_L{line_min_track_len}.gif",
            )
        print(f"  [cache] done → {out_dir}")
        return timings

    # ------------------------------------------------------------------
    # 1. Segment all frames once
    # ------------------------------------------------------------------
    seg_t0 = time.perf_counter()
    masks_per_frame: list[torch.Tensor] = []
    img_tensors: list[torch.Tensor] = []
    for fp in frame_paths:
        masks_per_frame.append(segment_frame(seg_model, fp, device=str(device)))
        img_tensors.append(load_image_tensor(fp))
    seg_total_s = time.perf_counter() - seg_t0
    mask_counts = [m.shape[0] for m in masks_per_frame]
    print(f"  segmentation total: {seg_total_s:.2f}s  "
          f"(mean {seg_total_s / n_frames:.2f}s/frame, masks per frame: "
          f"min={min(mask_counts)} mean={np.mean(mask_counts):.1f} max={max(mask_counts)})")

    # ------------------------------------------------------------------
    # 2. Inference: build pair scores
    # ------------------------------------------------------------------
    inner = model.inner
    inner.eval()

    best_matches: dict[tuple[int, int], dict[int, tuple[int, float]]] = {}
    backbone_s_total = 0.0
    matching_s_total = 0.0
    n_windows = 0
    backbone_s_warmup = 0.0
    matching_s_warmup = 0.0

    if mode == "single":
        # One forward with N=n_frames
        frame_indices = list(range(n_frames))
        n_windows = 1
        windows = [(0, frame_indices)]
    else:
        windows = []
        n_windows = 0
        if n_frames <= WINDOW_SIZE:
            windows.append((0, list(range(n_frames))))
        else:
            i = 0
            while i + WINDOW_SIZE <= n_frames:
                windows.append((i, list(range(i, i + WINDOW_SIZE))))
                i += WINDOW_STRIDE
            # Tail window: ensure last frame included
            tail_start = n_frames - WINDOW_SIZE
            if tail_start != windows[-1][0]:
                windows.append((tail_start, list(range(tail_start, n_frames))))
        n_windows = len(windows)

    print(f"  windows: {n_windows}")

    for w_idx, (w_start, frame_idxs) in enumerate(windows):
        N_w = len(frame_idxs)
        # Stack images
        images = torch.stack([img_tensors[f] for f in frame_idxs], dim=0).unsqueeze(0).to(device)  # (1, N, 3, H, W)
        # Per-view masks: list[N] of (1, M_v, H, W)
        masks_list = []
        empty_view_indices = []
        for v_local, f in enumerate(frame_idxs):
            m = masks_per_frame[f]
            if m.shape[0] == 0:
                # Dummy mask so model doesn't break (frame has no segments).
                m = torch.zeros((1, TARGET_H, TARGET_W), dtype=torch.uint8)
                empty_view_indices.append(v_local)
            masks_list.append(m.unsqueeze(0).float().to(device))

        # All ordered pairs (a, b), a < b
        pair_indices = [(a, b) for a in range(N_w) for b in range(a + 1, N_w)]

        results, backbone_s, matching_s = forward_window(
            inner, images, masks_list, pair_indices, device,
        )

        # Warmup: skip first window from average ONLY if we have >= 2 windows
        if n_windows >= 2 and w_idx == 0:
            backbone_s_warmup += backbone_s
            matching_s_warmup += matching_s
        else:
            backbone_s_total += backbone_s
            matching_s_total += matching_s

        # Extract matches per pair → update best_matches
        for (a, b), (lm, m0, _m1) in results.items():
            # If either view was a dummy, skip
            if a in empty_view_indices or b in empty_view_indices:
                continue
            f_a = frame_idxs[a]
            f_b = frame_idxs[b]
            M = masks_per_frame[f_a].shape[0]
            N = masks_per_frame[f_b].shape[0]
            if M == 0 or N == 0:
                continue
            pair_matches = extract_pair_matches(lm, m0, M, N, device)
            update_best_matches(best_matches, f_a, f_b, pair_matches)

        print(f"  window {w_idx + 1}/{n_windows} frames {frame_idxs[0]}..{frame_idxs[-1]}: "
              f"backbone={backbone_s * 1000:.0f}ms  matching={matching_s * 1000:.0f}ms")

    # ------------------------------------------------------------------
    # 3. Union-find → global instance IDs
    # ------------------------------------------------------------------
    gid_of, n_global = build_global_instance_ids(
        n_frames, masks_per_frame, best_matches,
    )
    track_lengths: dict[int, int] = {}
    summary: dict[int, list] = {}
    for (f, i), gid in gid_of.items():
        track_lengths[gid] = track_lengths.get(gid, 0) + 1
        summary.setdefault(gid, []).append([frame_paths[f].stem, i])

    track_len_arr = np.array(list(track_lengths.values()))
    print(f"  global instances: {n_global}  "
          f"track-length: mean={track_len_arr.mean():.2f} "
          f"median={int(np.median(track_len_arr))} max={int(track_len_arr.max())}")

    # ------------------------------------------------------------------
    # 4. Visualize per frame
    # ------------------------------------------------------------------
    palette = make_palette(max(n_global, 32))
    for f, fp in enumerate(frame_paths):
        rgb = load_image_rgb_for_display(fp)
        gids = [gid_of[(f, i)] for i in range(masks_per_frame[f].shape[0])]
        out_img = render_overlay(rgb, masks_per_frame[f], gids, palette)
        Image.fromarray(out_img).save(overlays_dir / f"{fp.stem}.png")

    # ------------------------------------------------------------------
    # 5. Save timings + summary
    # ------------------------------------------------------------------
    eligible = (n_windows - 1) if n_windows >= 2 else n_windows
    timings = {
        "scene": scene_dir.name,
        "mode": mode,
        "n_frames": n_frames,
        "n_windows": n_windows,
        "warmup_windows_excluded": 1 if n_windows >= 2 else 0,
        "segmentation_total_s": round(seg_total_s, 4),
        "backbone_total_s": round(backbone_s_total, 4),
        "matching_total_s": round(matching_s_total, 4),
        "backbone_total_s_with_warmup": round(backbone_s_total + backbone_s_warmup, 4),
        "matching_total_s_with_warmup": round(matching_s_total + matching_s_warmup, 4),
        "mean_backbone_ms_per_window": round(backbone_s_total * 1000 / eligible, 2),
        "mean_matching_ms_per_window": round(matching_s_total * 1000 / eligible, 2),
        "n_global_instances": n_global,
        "mean_track_length": round(float(track_len_arr.mean()), 3),
        "median_track_length": int(np.median(track_len_arr)),
        "max_track_length": int(track_len_arr.max()),
    }
    (out_dir / "timings.json").write_text(json.dumps(timings, indent=2))
    (out_dir / "instance_summary.json").write_text(json.dumps(summary, indent=2))

    # Cache for fast re-rendering (e.g. GIF parameter sweeps)
    torch.save(
        {
            "masks_per_frame": masks_per_frame,
            "gid_of": [(list(k), v) for k, v in gid_of.items()],
            "track_lengths": track_lengths,
            "n_global": n_global,
            "timings": timings,
        },
        cache_path,
    )

    if make_gif:
        render_sliding_window_gif(
            frame_paths, masks_per_frame, gid_of, palette,
            track_lengths, min_track_len=min_track_len,
            window_size=gif_window,
            out_path=out_dir / f"sliding_W{gif_window}_T{min_track_len}.gif",
        )

    print(f"  saved → {out_dir}")
    return timings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_model_from_config(cfg_path: Path, device: torch.device) -> SegVGGTDPTJointInfer:
    cfg = yaml.safe_load(cfg_path.read_text())
    # Normalize relative paths in MODEL.* w.r.t. project root
    proj_root = _PROJECT_ROOT.resolve()
    for k in ("CHECKPOINT", "VGGT_CKPT"):
        v = cfg["MODEL"][k]
        if not os.path.isabs(v):
            cfg["MODEL"][k] = str((proj_root / v.lstrip("./")).resolve())
        # If still missing, try same path without leading "../"
        if not Path(cfg["MODEL"][k]).exists():
            cand = proj_root / v.replace("../", "").lstrip("./")
            if cand.exists():
                cfg["MODEL"][k] = str(cand.resolve())
    print(f"  CHECKPOINT = {cfg['MODEL']['CHECKPOINT']}")
    print(f"  VGGT_CKPT  = {cfg['MODEL']['VGGT_CKPT']}")
    model = SegVGGTDPTJointInfer(cfg)
    model.prepare(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(_EVAL_ROOT / "configs/config_eval_vkitti2_segvggt_joint.yaml"),
    )
    parser.add_argument("--data_root", default=str(_PROJECT_ROOT / "data"))
    parser.add_argument(
        "--output_dir",
        default=str(_PROJECT_ROOT / "results/segvggt_joint_realdata"),
    )
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="Subset of scene dir names; default = all 4")
    parser.add_argument("--sam_weights", default="FastSAM-x.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--make_gif", action=argparse.BooleanOptionalAction, default=True,
                        help="Render sliding-window GIF (default: yes)")
    parser.add_argument("--gif_window", type=int, default=4,
                        help="Sliding-window size in GIF (default 4)")
    parser.add_argument("--min_track_len", type=int, default=3,
                        help="Drop tracks shorter than this in the GIF (default 3)")
    parser.add_argument("--line_min_track_len", type=int, default=5,
                        help="Connect centroids with a line for tracks ≥ this length (default 5)")
    parser.add_argument("--from_cache", action="store_true",
                        help="Load cached masks/gid_of from out_dir, skip seg+inference")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    cfg_path = Path(args.config)
    data_root = Path(args.data_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.from_cache:
        model = None
        seg_model = None
        print("\n[setup] cache mode — skipping model + segmentor load")
    else:
        print("\n[setup] loading joint model …")
        model = setup_model_from_config(cfg_path, device)

        print("\n[setup] loading FastSAM …")
        seg_model = init_segmentor(args.sam_weights, device=str(device))

    # Decide which scenes
    if args.scenes:
        scenes = args.scenes
    else:
        scenes = sorted(p.name for p in data_root.iterdir()
                        if p.is_dir() and (p / "frames").is_dir())

    all_timings = {}
    for scene_name in scenes:
        scene_dir = data_root / scene_name
        if not (scene_dir / "frames").is_dir():
            print(f"[skip] {scene_name}: no frames/ subdir")
            continue
        mode = "single" if scene_name == "scene_02-trunc" else "sliding"
        out_dir = out_root / scene_name
        try:
            timings = run_scene(
                scene_dir, out_dir, model, seg_model, device, mode,
                make_gif=args.make_gif,
                gif_window=args.gif_window,
                min_track_len=args.min_track_len,
                line_min_track_len=args.line_min_track_len,
                from_cache=args.from_cache,
            )
            all_timings[scene_name] = timings
        except Exception as exc:
            print(f"[fail] {scene_name}: {exc}")
            import traceback; traceback.print_exc()

    (out_root / "timings_summary.json").write_text(json.dumps(all_timings, indent=2))
    print("\n=== ALL DONE ===")
    print(f"results → {out_root}")


if __name__ == "__main__":
    main()
