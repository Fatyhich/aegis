"""
Precompute frozen MASt3R backbone + masked_average_pooling for all images.

Stores per-image pooled segment descriptors (M, 24) float16 — not dense features.
This is 1000× smaller than dense (24, H, W) and directly usable by the matcher.

Output layout:
    <feat_root>/<scene>/<image_name>.pt  →  float16 tensor (M, 24)

Disk estimate: ~63K images × avg 60 masks × 24 × 2 bytes ≈ 180 MB total.

Usage:
    python -m training.scripts.precompute_features \\
        --config training/config/segmast3r_train.yaml \\
        --feat_root /mnt/vol2_raid/shared/datasets/ScanNet++/mast3r_feat \\
        --backbone_batch 16 \\
        --device cuda

Resume-safe: already-computed .pt files are skipped automatically.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from training.paths import setup_segmast3r_path
setup_segmast3r_path()


def _load_masks(mask_pkl: Path, max_masks: int) -> torch.Tensor:
    """Load and decode masks from pkl → (M, H, W) float. Returns empty tensor if none."""
    from training.data.dataset import _decode_rles_batched
    with open(mask_pkl, "rb") as f:
        rles = pickle.load(f)["mask_coco_rles_resized"]
    if not rles:
        return torch.zeros(0, 1, 1)  # M=0; spatial size irrelevant (will be skipped)
    return _decode_rles_batched(rles[:max_masks]).float()   # (M, H, W)


def precompute(cfg, feat_root: Path, backbone_batch: int, device: str):
    from training.data.dataset import ScanNetPPSegDataset, _load_image
    from src.models.mast3r_segfeat.diff_masked_pooling import masked_average_pooling

    MAX_MASKS   = 128
    target_size = max(cfg.DATASET.HEIGHT, cfg.DATASET.WIDTH)
    resize_mode = cfg.DATASET.RESIZE_MODE

    # ── Dataset (for metadata: image paths, mask paths) ───────────────
    ds = ScanNetPPSegDataset(
        metadata_path=cfg.DATASET.METADATA_PATH,
        processed_root=cfg.DATASET.DATA_ROOT,
        masks_root=cfg.DATASET.SEGDATA_ROOT,
        pairs_root=cfg.DATASET.PAIRS_ROOT,
        target_size=target_size,
        resize_mode=resize_mode,
    )

    # ── Unique images, filter already done ────────────────────────────
    unique_ids = sorted({idx for pair in ds.pairs for idx in pair})
    todo = [
        idx for idx in unique_ids
        if not (feat_root / ds.scenes[ds.sceneids[idx]] / f"{ds.images[idx]}.pt").exists()
    ]
    print(f"Unique images : {len(unique_ids):,}")
    print(f"Already done  : {len(unique_ids) - len(todo):,}")
    print(f"To compute    : {len(todo):,}")

    if not todo:
        print("All features already cached.")
        return

    # ── Load backbone ─────────────────────────────────────────────────
    from mast3r_src.mast3r.model import load_model
    print(f"\nLoading backbone from {cfg.MODEL.MAST3R_CKPT} …")
    backbone = load_model(cfg.MODEL.MAST3R_CKPT, device=device, verbose=True)
    backbone.eval()

    feat_root.mkdir(parents=True, exist_ok=True)

    # ── Main loop ──────────────────────────────────────────────────────
    # Backbone runs in batches; pooling runs per-image (variable M masks).
    with tqdm(total=len(todo), desc="Precompute", unit="img") as pbar:
        for start in range(0, len(todo), backbone_batch):
            batch_ids = todo[start : start + backbone_batch]

            # 1. Load images
            imgs = torch.stack([
                _load_image(ds._get_img_path(idx), target_size, resize_mode)
                for idx in batch_ids
            ]).to(device)   # (B, 3, H, W)

            # 2. Backbone forward (self-pair, same as extract_desc in training)
            B = imgs.shape[0]
            view = {"img": imgs, "instance": [str(i) for i in range(B)]}
            with torch.no_grad():
                pred1, _ = backbone(view, view)
            feats = pred1["desc"].permute(0, 3, 1, 2)  # (B, 24, H, W)

            # 3. Per-image: load masks → pool → save
            for k, idx in enumerate(batch_ids):
                scene    = ds.scenes[ds.sceneids[idx]]
                name     = ds.images[idx]
                out_path = feat_root / scene / f"{name}.pt"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                feat = feats[k].unsqueeze(0)   # (1, 24, H, W)

                masks = _load_masks(ds._mask_pkl(idx), MAX_MASKS)  # (M, H_m, W_m)
                M = masks.shape[0]

                if M == 0:
                    # No masks: save empty (0, 24) tensor
                    torch.save(torch.zeros(0, 24, dtype=torch.float16), out_path)
                else:
                    masks = masks.to(device)
                    # Align masks to descriptor grid if needed
                    _, _, dH, dW = feat.shape
                    if masks.shape[-2:] != (dH, dW):
                        masks = F.interpolate(
                            masks.unsqueeze(0), (dH, dW), mode="nearest"
                        ).squeeze(0)

                    with torch.no_grad():
                        dsc = masked_average_pooling(feat, masks.unsqueeze(0))  # (1, 24, M)
                    # Save (M, 24) float16
                    torch.save(dsc[0].T.half().cpu(), out_path)

            pbar.update(len(batch_ids))

    print(f"\nDone. Pooled descriptors saved to {feat_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         required=True)
    parser.add_argument("--feat_root",      required=True)
    parser.add_argument("--backbone_batch", type=int, default=16,
                        help="Images per backbone forward pass")
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--dry_run",        action="store_true",
                        help="Print estimate and exit without writing")
    args = parser.parse_args()

    from training.engine.utils import load_cfg
    cfg = load_cfg(args.config)

    if args.dry_run:
        from training.data.dataset import ScanNetPPSegDataset
        target_size = max(cfg.DATASET.HEIGHT, cfg.DATASET.WIDTH)
        ds = ScanNetPPSegDataset(
            metadata_path=cfg.DATASET.METADATA_PATH,
            processed_root=cfg.DATASET.DATA_ROOT,
            masks_root=cfg.DATASET.SEGDATA_ROOT,
            pairs_root=cfg.DATASET.PAIRS_ROOT,
            target_size=target_size,
            resize_mode=cfg.DATASET.RESIZE_MODE,
        )
        feat_root = Path(args.feat_root)
        unique_ids = sorted({idx for pair in ds.pairs for idx in pair})
        todo = [
            idx for idx in unique_ids
            if not (feat_root / ds.scenes[ds.sceneids[idx]] / f"{ds.images[idx]}.pt").exists()
        ]
        print(f"Unique images : {len(unique_ids):,}")
        print(f"Already done  : {len(unique_ids) - len(todo):,}")
        print(f"To compute    : {len(todo):,}")
        # Pooled (M,24) float16 — rough estimate assuming avg 60 masks per image
        avg_masks = 60
        total_mb = len(unique_ids) * avg_masks * 24 * 2 / 1e6
        todo_mb  = len(todo)       * avg_masks * 24 * 2 / 1e6
        print(f"Estimated disk: ~{total_mb:.0f} MB total  |  ~{todo_mb:.0f} MB remaining")
        print(f"  (pooled descriptors, avg {avg_masks} masks/image, float16)")
        print("--dry_run: exiting without writing.")
    else:
        precompute(cfg, Path(args.feat_root), args.backbone_batch, args.device)
