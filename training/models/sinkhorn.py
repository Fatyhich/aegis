"""
SegMASt3R: frozen MASt3R backbone + masked_average_pooling + Sinkhorn matcher.

Architecture (per SegMASt3R paper):
    img0 (B,3,H,W) ──► [MASt3R frozen] ──► desc0 (B, H, W, 24)
                                                  │ permute
                                           feat0 (B, 24, H, W)
                                                  │
    masks0 (B, M, H, W) ──► masked_avg_pool ──► dsc0 (B, 24, M)
                                                        │
    dsc1 (B, 24, N) ◄── masked_avg_pool ◄── feat1      │
           │                                            │
           └────────────── Sinkhorn ────────────────────┘
                               │
                         log_P (B, M+1, N+1)   ← dustbin included

Key facts confirmed from MASt3R source + README:
  - model(view1, view2) where view = {'img': (B,3,H,W), 'true_shape': optional}
  - pred['desc']: (B, H, W, 24)  — HWC layout, output_mode='pts3d+desc24'
  - true_shape is optional — inferred from img.shape[-2:] if absent
  - Trainable params: only Sinkhorn dustbin_score (~1 scalar)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.paths import setup_segmast3r_path
setup_segmast3r_path()

from src.models.mast3r_segfeat.diff_masked_pooling import masked_average_pooling
from src.models.mast3r_segfeat.diff_feature_matcher import featureMatcher


class SegMASt3R(nn.Module):

    DESC_DIM = 24  # confirmed: output_mode='pts3d+desc24'

    def __init__(self, mast3r_ckpt: str, matcher_cfg: dict, device: str = "cuda"):
        super().__init__()
        self.backbone = self._load_backbone(mast3r_ckpt, device)
        self._freeze(self.backbone)
        self.matcher = featureMatcher(matcher_cfg)

    @staticmethod
    def _load_backbone(ckpt_path: str, device: str):
        from mast3r_src.mast3r.model import load_model
        model = load_model(ckpt_path, device=device, verbose=True)
        model.eval()
        return model

    @staticmethod
    def _freeze(module: nn.Module):
        for p in module.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    def extract_desc(
        self, img0: torch.Tensor, img1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract dense per-pixel descriptors from MASt3R using cross-pair.

        MASt3R decoder uses cross-attention between view0 and view1.
        Both views must be different images for meaningful descriptors.

        Args:
            img0: (B, 3, H, W) normalized [-1, 1]
            img1: (B, 3, H, W) normalized [-1, 1]
        Returns:
            desc0: (B, DESC_DIM, H, W)
            desc1: (B, DESC_DIM, H, W)
        """
        B = img0.shape[0]
        view0 = {
            "img": img0,
            "instance": [f"0_{i}" for i in range(B)],
        }
        view1 = {
            "img": img1,
            "instance": [f"1_{i}" for i in range(B)],
        }
        with torch.no_grad():
            pred0, pred1 = self.backbone(view0, view1)
        desc0 = pred0["desc"].permute(0, 3, 1, 2).contiguous()
        desc1 = pred1["desc"].permute(0, 3, 1, 2).contiguous()
        return desc0, desc1

    # ------------------------------------------------------------------
    def forward(
        self,
        img0: torch.Tensor,    # (B, 3, H, W)
        img1: torch.Tensor,    # (B, 3, H, W)
        masks0: torch.Tensor,  # (B, M, H, W) float
        masks1: torch.Tensor,  # (B, N, H, W) float
    ):
        """
        Returns:
            log_P: (B, M+1, N+1)  log Sinkhorn with dustbin
            dsc0:  (B, 24, M)     segment descriptors img0
            dsc1:  (B, 24, N)     segment descriptors img1
        """
        feat0, feat1 = self.extract_desc(img0, img1)   # (B, 24, H, W) each

        # Align masks spatial size to descriptor grid (should match, but safe)
        _, _, dH, dW = feat0.shape
        if masks0.shape[-2:] != (dH, dW):
            masks0 = F.interpolate(masks0.float(), (dH, dW), mode="nearest")
            masks1 = F.interpolate(masks1.float(), (dH, dW), mode="nearest")

        dsc0 = masked_average_pooling(feat0, masks0.float())  # (B, 24, M)
        dsc1 = masked_average_pooling(feat1, masks1.float())  # (B, 24, N)

        log_P = self.matcher(dsc0, dsc1)   # (B, M+1, N+1)
        return log_P, dsc0, dsc1


# ---------------------------------------------------------------------------
# Loss (SuperGlue-style NLL on Sinkhorn output)
# ---------------------------------------------------------------------------

def build_gt_matrix(
    seg_corr: torch.Tensor,   # (K, 2)
    M: int, N: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Binary GT matrix G of shape (M+1, N+1):
      G[i, j] = 1  for matched pairs (i,j) in seg_corr
      G[i, N] = 1  for unmatched segments in img0  (dustbin column)
      G[M, j] = 1  for unmatched segments in img1  (dustbin row)
    """
    G = torch.zeros(M + 1, N + 1, device=device)

    if seg_corr.shape[0] > 0:
        G[seg_corr[:, 0], seg_corr[:, 1]] = 1.0
        matched_i = set(seg_corr[:, 0].tolist())
        matched_j = set(seg_corr[:, 1].tolist())
    else:
        matched_i, matched_j = set(), set()

    unmatched_i = [i for i in range(M) if i not in matched_i]
    unmatched_j = [j for j in range(N) if j not in matched_j]

    if unmatched_i:
        G[torch.tensor(unmatched_i, device=device), N] = 1.0
    if unmatched_j:
        G[M, torch.tensor(unmatched_j, device=device)] = 1.0

    return G


def superglue_nll_loss(
    log_P: torch.Tensor,       # (B, M_max+1, N_max+1)
    seg_corr_list: list,        # list[B] of (K_b, 2) tensors
    masks0_list: list,          # list[B] of (M_b, H, W) — for actual M per sample
    masks1_list: list,          # list[B] of (N_b, H, W)
) -> torch.Tensor:
    """
    Mean NLL loss over the batch.
    For each sample: loss = -mean(log_P[GT=1])
    """
    total = log_P.new_zeros(1)
    B = log_P.shape[0]

    for b in range(B):
        M = masks0_list[b].shape[0]
        N = masks1_list[b].shape[0]
        corr = seg_corr_list[b].to(log_P.device)
        G = build_gt_matrix(corr, M, N, log_P.device)   # (M+1, N+1)
        lP = log_P[b, :M + 1, :N + 1]                   # slice to actual size
        total = total + -(lP * G).sum() / G.sum().clamp(min=1)

    return total / B
