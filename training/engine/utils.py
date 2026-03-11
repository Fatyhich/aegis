import importlib.util
import math
from pathlib import Path

import torch


def pad_masks_to_batch(masks_list: list, device) -> torch.Tensor:
    """Pad variable-M masks → (B, M_max, H, W) float."""
    M_max = max(m.shape[0] for m in masks_list)
    H, W  = masks_list[0].shape[-2:]
    out = []
    for m in masks_list:
        m = m.to(device)
        pad = M_max - m.shape[0]
        if pad > 0:
            m = torch.cat([m, torch.zeros(pad, H, W, dtype=m.dtype, device=m.device)], 0)
        out.append(m)
    return torch.stack(out).float()


def pad_descriptors_to_batch(dsc_list: list, device) -> torch.Tensor:
    """Pad variable-M precomputed descriptors → (B, M_max, D) float."""
    M_max = max(d.shape[0] for d in dsc_list)
    D     = dsc_list[0].shape[1]
    out   = []
    for d in dsc_list:
        d   = d.to(device).float()
        pad = M_max - d.shape[0]
        if pad > 0:
            d = torch.cat([d, torch.zeros(pad, D, dtype=d.dtype, device=d.device)], 0)
        out.append(d)
    return torch.stack(out)   # (B, M_max, D)


def load_cfg(config_path=None):
    # Load our own default — not the original project's configs/default.py
    _here = Path(__file__).resolve().parent.parent / "config"
    _candidates = [
        _here / "default.py",
    ]
    spec = None
    for p in _candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("segmast3r_default", p)
            break
    if spec is None:
        raise FileNotFoundError(
            f"default.py not found. Looked in: {_candidates}"
        )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.cfg

    cfg.defrost()
    if config_path:
        cfg.merge_from_file(config_path)
    else:
        # --mock: small/fast overrides, no mixed precision needed
        cfg.SAVE_DIR = "results/mock_run"
        cfg.TRAINING.BATCH_SIZE    = 4
        cfg.TRAINING.NUM_WORKERS   = 0
        cfg.TRAINING.EPOCHS        = 2
        cfg.TRAINING.WARMUP_STEPS  = 50
        cfg.TRAINING.LOG_INTERVAL  = 5
        cfg.TRAINING.VAL_INTERVAL  = 20
        cfg.TRAINING.SAVE_INTERVAL = 50
        cfg.ACCELERATE.MIXED_PRECISION = "no"
    cfg.freeze()
    return cfg


def compute_matching_metrics(log_P: torch.Tensor,
                              seg_corr_list: list,
                              masks0_list: list,
                              masks1_list: list) -> dict:
    """
    Matching quality metrics from Sinkhorn log-prob matrix.

    matching_accuracy — fraction of GT pairs where row-wise argmax
                        (excluding dustbin col) picks the correct target segment.
                        Core metric: directly measures if model learned to match.

    mean_gt_logprob   — mean log_P[i,j] for GT pairs.
                        Tracks confidence on correct matches.
                        Should increase during training.

    dustbin_rate      — fraction of img0 segments whose argmax (including dustbin)
                        lands on the dustbin column.
                        ~50% is healthy (many segments genuinely unmatched).
                        ~100% means model collapsed to "assign everything to dustbin".
    """
    total_correct = 0
    total_gt      = 0
    total_logprob = 0.0
    total_segs    = 0
    total_dustbin = 0

    for b in range(log_P.shape[0]):
        M    = masks0_list[b].shape[0]
        N    = masks1_list[b].shape[0]
        corr = seg_corr_list[b]           # (K, 2)
        lP   = log_P[b, :M, :N+1]        # (M, N+1) — last col = dustbin

        if corr.shape[0] > 0:
            pred_j = lP[:, :N].argmax(dim=1)  # argmax excluding dustbin
            for k in range(corr.shape[0]):
                i0, j0 = corr[k, 0].item(), corr[k, 1].item()
                if i0 < M:
                    total_correct += int(pred_j[i0].item() == j0)
                    total_gt      += 1
                    total_logprob += log_P[b, i0, j0].item()

        pred_full      = lP.argmax(dim=1)   # argmax over all N+1 cols
        total_segs    += M
        total_dustbin += (pred_full == N).sum().item()

    return {
        "matching_accuracy": total_correct / max(total_gt,   1),
        "mean_gt_logprob":   total_logprob / max(total_gt,   1),
        "dustbin_rate":      total_dustbin / max(total_segs, 1),
    }
