"""
ScanNetPP training dataset for SegMASt3R training.

Two precompute modes:
  1. Online: load images → backbone(view0, view1) → descriptors at training time
  2. Per-pair precomputed: load <pair_dsc_root>/<scene>/<name_i>__<name_j>.pt
     containing {"dsc0": (M,24) fp16, "dsc1": (N,24) fp16}

Per-pair precompute is required because MASt3R decoder uses cross-attention:
descriptors for img0 depend on which img1 it's paired with.

Data sources:
  1. masks_resize_mast3r/<scene>/<img>.pkl        → mask_coco_rles_resized
  2. masks_resize_mast3r_res/<scene>/<i>__<j>.pkl → seg_corr_list

Resize modes:
  "square"       — resize to exact target×target (stackable batches)
  "longest_side" — resize longest side to target, preserve AR (variable sizes)
"""

import os
import pickle
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from pycocotools import mask as mask_utils

MAX_MASKS = 100
ResizeMode = Literal["square", "longest_side"]

try:
    from torchvision.io import read_file as _tv_read_file, decode_jpeg as _tv_decode_jpeg
    _TV_JPEG = True
except ImportError:
    _TV_JPEG = False


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load_image(path: Path, target: int, mode: ResizeMode) -> torch.Tensor:
    """
    Load image → (3, H, W) float32 normalized to [-1, 1].
    JPEG: torchvision decode_jpeg (2–4× faster than PIL on CPU).
    Other formats / fallback: PIL.
    """
    ext = path.suffix.lower()
    if _TV_JPEG and ext in (".jpg", ".jpeg"):
        raw = _tv_read_file(str(path))
        img = _tv_decode_jpeg(raw)              # (3, H, W) uint8
        if mode == "square":
            img = TF.resize(img, [target, target],
                            interpolation=TF.InterpolationMode.BILINEAR,
                            antialias=True)
        else:
            h, w = img.shape[-2], img.shape[-1]
            scale = target / max(h, w)
            img = TF.resize(img, [int(h * scale), int(w * scale)],
                            interpolation=TF.InterpolationMode.BILINEAR,
                            antialias=True)
        return TF.normalize(img.float().div(255.0), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    else:
        from PIL import Image
        from PIL.ImageOps import exif_transpose
        pil = exif_transpose(Image.open(path).convert("RGB"))
        if mode == "square":
            pil = pil.resize((target, target), Image.BILINEAR)
        else:
            w, h = pil.size
            scale = target / max(w, h)
            pil = pil.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        img = TF.to_tensor(pil)
        return TF.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def _decode_rles_batched(rles: list) -> torch.Tensor:
    """Decode list of COCO RLEs to (M, H, W) uint8 tensor."""
    for rle in rles:
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
    arr = mask_utils.decode(rles)                                           # (H, W, M) uint8
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))  # (M, H, W)


def _resize_masks(masks: torch.Tensor, size_hw: tuple) -> torch.Tensor:
    """(M, H_src, W_src) → (M, H_dst, W_dst) nearest-neighbor."""
    H, W = size_hw
    if masks.shape[0] == 0:
        return torch.zeros(0, H, W, dtype=masks.dtype)
    return TF.resize(masks, [H, W], interpolation=TF.InterpolationMode.NEAREST)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ScanNetPPSegDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        processed_root: str,
        masks_root: str,
        pairs_root: str,
        target_size: int = 512,
        max_masks: int = MAX_MASKS,
        resize_mode: ResizeMode = "square",
        feat_root: str = "",            # DEPRECATED: per-image precompute is invalid
        pair_dsc_root: str = "",        # per-pair precomputed descriptors
    ):
        assert resize_mode in ("square", "longest_side"), \
            f"resize_mode must be 'square' or 'longest_side', got '{resize_mode}'"

        if feat_root:
            raise ValueError(
                "feat_root (per-image precompute) is no longer supported. "
                "MASt3R uses cross-attention: descriptors depend on the paired image. "
                "Use pair_dsc_root for per-pair precomputed descriptors instead."
            )

        self.processed_root = Path(processed_root)
        self.masks_root     = Path(masks_root)
        self.pairs_root     = Path(pairs_root)
        self.pair_dsc_root  = Path(pair_dsc_root) if pair_dsc_root else None
        self.target_size    = target_size
        self.max_masks      = max_masks
        self.resize_mode    = resize_mode
        self._profile       = bool(os.environ.get("PROFILE_DATALOADER"))

        meta = np.load(metadata_path, allow_pickle=True)
        self.scenes       = meta["scenes"]
        self.sceneids     = meta["sceneids"]
        self.images       = meta["images"]
        self.intrinsics   = meta["intrinsics"].astype(np.float32)
        self.trajectories = meta["trajectories"].astype(np.float32)
        raw_pairs         = meta["pairs"][:, :2].astype(int)

        self.pairs = self._filter_pairs(raw_pairs)

        # Build image path cache (only needed if not using precomputed descriptors)
        self._img_path_cache = {} if self.pair_dsc_root else self._build_img_cache()

        dsc_info = f" | pair_dsc_root='{pair_dsc_root}'" if pair_dsc_root else ""
        print(f"ScanNetPPSegDataset: {len(self.pairs):,} pairs "
              f"(from {len(raw_pairs):,} total, "
              f"{len(self.pairs)/len(raw_pairs)*100:.1f}% available) "
              f"| resize_mode='{resize_mode}' target={target_size}"
              f"{dsc_info}"
              f"{' | PROFILING ON' if self._profile else ''}")

        if self._profile:
            self._prof_t_img   = []
            self._prof_t_mask  = []
            self._prof_t_rle   = []
            self._prof_t_total = []
            self._prof_n       = 0

    # ------------------------------------------------------------------

    def _pair_pkl(self, idx_i: int, idx_j: int) -> Path:
        scene = self.scenes[self.sceneids[idx_i]]
        return self.pairs_root / scene / \
            f"{self.images[idx_i]}__{self.images[idx_j]}.pkl"

    def _mask_pkl(self, idx: int) -> Path:
        scene = self.scenes[self.sceneids[idx]]
        return self.masks_root / scene / f"{self.images[idx]}.pkl"

    def _pair_dsc_path(self, idx_i: int, idx_j: int) -> Path:
        """Per-pair precomputed descriptor file path."""
        scene = self.scenes[self.sceneids[idx_i]]
        return self.pair_dsc_root / scene / \
            f"{self.images[idx_i]}__{self.images[idx_j]}.pt"

    def _filter_pairs(self, raw_pairs):
        valid = []
        for i, j in raw_pairs:
            i, j = int(i), int(j)
            if not self._pair_pkl(i, j).exists():
                continue
            if self.pair_dsc_root and not self._pair_dsc_path(i, j).exists():
                continue
            valid.append((i, j))
        return valid

    def _build_img_cache(self) -> dict:
        unique_ids = set()
        for i, j in self.pairs:
            unique_ids.add(i)
            unique_ids.add(j)

        cache = {}
        exts = (".jpg", ".JPG", ".jpeg", ".png")
        for idx in unique_ids:
            scene = self.scenes[self.sceneids[idx]]
            name  = self.images[idx]
            base  = self.processed_root / scene / "images" / name
            for ext in exts:
                p = base.with_suffix(ext)
                if p.exists():
                    cache[idx] = p
                    break
        return cache

    def _get_img_path(self, idx: int) -> Path:
        try:
            return self._img_path_cache[idx]
        except KeyError:
            scene = self.scenes[self.sceneids[idx]]
            name  = self.images[idx]
            raise FileNotFoundError(
                f"Image not found: {self.processed_root}/{scene}/images/{name}.*")

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        _t0 = time.perf_counter() if self._profile else 0.0

        idx_i, idx_j = self.pairs[idx]
        scene  = self.scenes[self.sceneids[idx_i]]
        name_i = self.images[idx_i]
        name_j = self.images[idx_j]

        # ── Per-pair precomputed descriptors or images ─────────────────
        _t1 = time.perf_counter() if self._profile else 0.0
        if self.pair_dsc_root is not None:
            pair_data = torch.load(
                self._pair_dsc_path(idx_i, idx_j),
                map_location="cpu", weights_only=True,
            )
            dsc0 = pair_data["dsc0"].float()    # (M0, 24)
            dsc1 = pair_data["dsc1"].float()    # (M1, 24)
            img0 = img1 = None
            H0 = W0 = H1 = W1 = self.target_size
        else:
            img0 = _load_image(self._get_img_path(idx_i), self.target_size, self.resize_mode)
            img1 = _load_image(self._get_img_path(idx_j), self.target_size, self.resize_mode)
            dsc0 = dsc1 = None
            _, H0, W0 = img0.shape
            _, H1, W1 = img1.shape

        # ── Load RLE pickles ───────────────────────────────────────────
        _t2 = time.perf_counter() if self._profile else 0.0
        with open(self._mask_pkl(idx_i), "rb") as f:
            rles0 = pickle.load(f)["mask_coco_rles_resized"]
        with open(self._mask_pkl(idx_j), "rb") as f:
            rles1 = pickle.load(f)["mask_coco_rles_resized"]

        # ── Decode RLEs + resize ───────────────────────────────────────
        _t3 = time.perf_counter() if self._profile else 0.0
        if rles0:
            masks0 = _decode_rles_batched(rles0[:self.max_masks])
            if masks0.shape[-2:] != (H0, W0):
                masks0 = _resize_masks(masks0, (H0, W0))
        else:
            masks0 = torch.zeros(0, H0, W0, dtype=torch.uint8)

        if rles1:
            masks1 = _decode_rles_batched(rles1[:self.max_masks])
            if masks1.shape[-2:] != (H1, W1):
                masks1 = _resize_masks(masks1, (H1, W1))
        else:
            masks1 = torch.zeros(0, H1, W1, dtype=torch.uint8)

        masks0 = masks0.to(torch.uint8)
        masks1 = masks1.to(torch.uint8)
        M0, M1 = masks0.shape[0], masks1.shape[0]

        # ── Correspondences ────────────────────────────────────────────
        with open(self._pair_pkl(idx_i, idx_j), "rb") as f:
            seg_corr_raw = pickle.load(f).get("seg_corr_list", [])

        corr_filtered = [
            (i, j) for i, j in seg_corr_raw if i < M0 and j < M1
        ]
        seg_corr = (torch.tensor(corr_filtered, dtype=torch.long)
                    if corr_filtered else torch.zeros((0, 2), dtype=torch.long))

        # ── Optional profiling ─────────────────────────────────────────
        if self._profile:
            _t4 = time.perf_counter()
            self._prof_t_img.append(_t2 - _t1)
            self._prof_t_mask.append(_t3 - _t2)
            self._prof_t_rle.append(_t4 - _t3)
            self._prof_t_total.append(_t4 - _t0)
            self._prof_n += 1
            if self._prof_n % 100 == 0:
                n = 100
                print(
                    f"[DataLoader @{self._prof_n}] "
                    f"img={np.mean(self._prof_t_img[-n:])*1e3:.1f}ms  "
                    f"mask_pkl={np.mean(self._prof_t_mask[-n:])*1e3:.1f}ms  "
                    f"rle_decode={np.mean(self._prof_t_rle[-n:])*1e3:.1f}ms  "
                    f"total={np.mean(self._prof_t_total[-n:])*1e3:.1f}ms"
                )

        out = {
            "masks0":   masks0,
            "masks1":   masks1,
            "seg_corr": seg_corr,
            "valid":    seg_corr.shape[0] > 0,
            "scene":    scene,
            "name_i":   name_i,
            "name_j":   name_j,
        }
        if dsc0 is not None:
            out["dsc0"] = dsc0
            out["dsc1"] = dsc1
            # Image paths for lazy visualization in validator
            out["img_path_i"] = str(self._get_img_path(idx_i)) if self._img_path_cache else ""
            out["img_path_j"] = str(self._get_img_path(idx_j)) if self._img_path_cache else ""
        else:
            out["img0"] = img0
            out["img1"] = img1
        return out


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def collate_fn_square(batch):
    out = {
        "masks0":   [b["masks0"]   for b in batch],
        "masks1":   [b["masks1"]   for b in batch],
        "seg_corr": [b["seg_corr"] for b in batch],
        "valid":    torch.tensor([b["valid"] for b in batch]),
        "scene":    [b["scene"]    for b in batch],
        "name_i":   [b["name_i"]   for b in batch],
        "name_j":   [b["name_j"]   for b in batch],
    }
    if "dsc0" in batch[0]:
        out["dsc0"]       = [b["dsc0"] for b in batch]
        out["dsc1"]       = [b["dsc1"] for b in batch]
        out["img_path_i"] = [b["img_path_i"] for b in batch]
        out["img_path_j"] = [b["img_path_j"] for b in batch]
    else:
        out["img0"] = torch.stack([b["img0"] for b in batch])
        out["img1"] = torch.stack([b["img1"] for b in batch])
    return out


def collate_fn_longest_side(batch):
    out = {
        "masks0":   [b["masks0"]   for b in batch],
        "masks1":   [b["masks1"]   for b in batch],
        "seg_corr": [b["seg_corr"] for b in batch],
        "valid":    torch.tensor([b["valid"] for b in batch]),
        "scene":    [b["scene"]    for b in batch],
        "name_i":   [b["name_i"]   for b in batch],
        "name_j":   [b["name_j"]   for b in batch],
    }
    if "dsc0" in batch[0]:
        out["dsc0"]       = [b["dsc0"] for b in batch]
        out["dsc1"]       = [b["dsc1"] for b in batch]
        out["img_path_i"] = [b["img_path_i"] for b in batch]
        out["img_path_j"] = [b["img_path_j"] for b in batch]
    else:
        out["img0"] = [b["img0"] for b in batch]
        out["img1"] = [b["img1"] for b in batch]
    return out


def get_collate_fn(resize_mode: ResizeMode):
    if resize_mode == "square":
        return collate_fn_square
    return collate_fn_longest_side
