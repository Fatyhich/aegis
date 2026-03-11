import math
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from training.paths import setup_segmast3r_path
setup_segmast3r_path()

from training.engine.utils import (
    pad_masks_to_batch,
    pad_descriptors_to_batch,
    compute_matching_metrics,
)
from training.engine.validator import run_validation
from src.utils.debug_utils import plot_sinkhorn_debug


def build_lr_scheduler(optimizer, cfg, total_steps: int):
    warmup = cfg.TRAINING.WARMUP_STEPS
    sched  = cfg.TRAINING.LR_SCHEDULER

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        if sched == "cosine":
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step(model, batch, device, loss_fn):
    if "dsc0" in batch:
        # Precomputed mode: skip backbone + pooling
        dsc0   = pad_descriptors_to_batch(batch["dsc0"], device)  # (B, M_max, 24)
        dsc1   = pad_descriptors_to_batch(batch["dsc1"], device)
        output = model(None, None, dsc0_pre=dsc0, dsc1_pre=dsc1)
    else:
        img0   = batch["img0"].to(device)
        img1   = batch["img1"].to(device)
        masks0 = pad_masks_to_batch(batch["masks0"], device)
        masks1 = pad_masks_to_batch(batch["masks1"], device)
        output = model(img0, img1, masks0, masks1)
    loss = loss_fn(output, batch["seg_corr"], batch["masks0"], batch["masks1"])
    # output[0]  = score matrix (log_P for sinkhorn, log_mutual for LG)
    # output[-2] = dsc0,  output[-1] = dsc1
    return loss, output


def _save_ckpt(path, model, optimizer, scheduler, epoch, global_step,
               metrics: dict, best_val_ma=None, accelerator=None):
    m = accelerator.unwrap_model(model) if accelerator is not None else model
    payload = {
        "epoch":           epoch,
        "global_step":     global_step,
        "model_state":     m.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "metrics":         metrics,
    }
    if best_val_ma is not None:
        payload["best_val_ma"] = best_val_ma
    torch.save(payload, path)


def train(cfg, mock=False, resume=None):
    # ── Accelerator ───────────────────────────────────────────────
    mixed_prec = getattr(getattr(cfg, "ACCELERATE", None), "MIXED_PRECISION", "no")
    accelerator = Accelerator(mixed_precision=mixed_prec, log_with=None)
    device = accelerator.device

    # ── Save dir / TensorBoard ────────────────────────────────────
    save_dir = Path(cfg.SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if accelerator.is_main_process:
        tb_dir = save_dir / "tb"
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"Device: {device}  |  num_processes: {accelerator.num_processes}  "
              f"|  mixed_precision: {mixed_prec}")
        print(f"Tensorboard: tensorboard --logdir {tb_dir}")

    # ── Dataset ───────────────────────────────────────────────────
    if mock:
        if accelerator.is_main_process:
            print("MOCK run — synthetic data, lightweight backbone")
        from training.data.mock import MockDataset, mock_collate
        ds_train   = MockDataset(n=128)
        ds_val     = MockDataset(n=32)
        collate_fn = mock_collate
    else:
        from training.data.dataset import ScanNetPPSegDataset, get_collate_fn
        feat_root = getattr(cfg.DATASET, "PRECOMPUTED_FEAT_ROOT", "") or None
        ds_full = ScanNetPPSegDataset(
            metadata_path=cfg.DATASET.METADATA_PATH,
            processed_root=cfg.DATASET.DATA_ROOT,
            masks_root=cfg.DATASET.SEGDATA_ROOT,
            pairs_root=cfg.DATASET.PAIRS_ROOT,
            target_size=max(cfg.DATASET.HEIGHT, cfg.DATASET.WIDTH),
            resize_mode=cfg.DATASET.RESIZE_MODE,
            feat_root=feat_root,
        )
        n_total = len(ds_full)
        n_val   = max(1, int(n_total * cfg.DATASET.VAL_FRACTION))
        indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
        ds_val   = Subset(ds_full, indices[:n_val].tolist())
        ds_train = Subset(ds_full, indices[n_val:].tolist())
        collate_fn = get_collate_fn(cfg.DATASET.RESIZE_MODE)
        if accelerator.is_main_process:
            print(f"Train: {len(ds_train):,}  |  Val: {len(ds_val):,}")

    loader_train = DataLoader(
        ds_train,
        batch_size=cfg.TRAINING.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.TRAINING.NUM_WORKERS if not mock else 0,
        collate_fn=collate_fn,
        pin_memory=(accelerator.device.type == "cuda"),
        prefetch_factor=cfg.TRAINING.PREFETCH_FACTOR if not mock else None,
        persistent_workers=(cfg.TRAINING.NUM_WORKERS > 0 and not mock),
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=cfg.TRAINING.BATCH_SIZE,
        shuffle=False,
        num_workers=min(4, cfg.TRAINING.NUM_WORKERS) if not mock else 0,
        collate_fn=collate_fn,
        pin_memory=(accelerator.device.type == "cuda"),
    )

    # ── Model / loss / metrics ────────────────────────────────────
    arch = getattr(cfg.MODEL, "ARCH", "sinkhorn")

    if not mock and feat_root and arch != "lightglue":
        raise ValueError(
            f"PRECOMPUTED_FEAT_ROOT is set but ARCH='{arch}' does not support "
            f"precomputed descriptors. Only arch='lightglue' supports this mode."
        )

    matcher_cfg = {
        "TYPE": cfg.FEATURE_MATCHER.TYPE,
        "SINKHORN": {
            "NUM_IT":             cfg.FEATURE_MATCHER.SINKHORN.NUM_IT,
            "DUSTBIN_SCORE_INIT": cfg.FEATURE_MATCHER.SINKHORN.DUSTBIN_SCORE_INIT,
        }
    }

    if mock:
        import torch.nn.functional as F
        from src.models.mast3r_segfeat.diff_feature_matcher import featureMatcher
        from src.models.mast3r_segfeat.diff_masked_pooling import masked_average_pooling

        class _MockSegMASt3R(torch.nn.Module):
            def __init__(self, matcher_cfg, desc_dim=24):
                super().__init__()
                self.conv    = torch.nn.Conv2d(3, desc_dim, 3, padding=1)
                self.matcher = featureMatcher(matcher_cfg)
                self.desc_dim = desc_dim

            def extract_desc(self, imgs):
                return self.conv(imgs)  # (B, D, H, W)

            def forward(self, img0, img1, masks0, masks1):
                d0 = self.extract_desc(img0)
                d1 = self.extract_desc(img1)
                dH, dW = d0.shape[-2:]
                if masks0.shape[-2:] != (dH, dW):
                    masks0 = F.interpolate(masks0.float(), (dH, dW), mode="nearest")
                    masks1 = F.interpolate(masks1.float(), (dH, dW), mode="nearest")
                dsc0 = masked_average_pooling(d0, masks0.float())
                dsc1 = masked_average_pooling(d1, masks1.float())
                return self.matcher(dsc0, dsc1), dsc0, dsc1

        model = _MockSegMASt3R(matcher_cfg)
        arch  = "sinkhorn"   # mock shares the sinkhorn interface

    if arch == "lightglue":
        from training.models.lightglue import (
            SegMASt3RLG, lightglue_loss, lightglue_loss_deep,
            compute_matching_metrics_lg,
        )
        lg = cfg.MODEL.LG
        if not mock:
            model = SegMASt3RLG(
                mast3r_ckpt=cfg.MODEL.MAST3R_CKPT,
                proj_dim=lg.PROJ_DIM,
                n_layers=lg.N_LAYERS,
                n_heads=lg.N_HEADS,
                use_grad_checkpoint=lg.GRAD_CHECKPOINT,
                deep_supervision=lg.DEEP_SUPERVISION,
                device="cpu",
            )
        _deep   = lg.DEEP_SUPERVISION
        _lambda = lg.LAMBDA_MATCH
        if _deep:
            def loss_fn(output, seg_corr, masks0, masks1):
                return lightglue_loss_deep(output[0], seg_corr, masks0, masks1,
                                           lambda_match=_lambda)
            def metrics_fn(output, seg_corr, masks0, masks1):
                lm, m0, _ = output[0][-1]   # last layer
                return compute_matching_metrics_lg(lm, m0, seg_corr, masks0, masks1)
            def score_mat(output, b, M, N):
                return output[0][-1][0][b, :M, :N].cpu()
        else:
            def loss_fn(output, seg_corr, masks0, masks1):
                return lightglue_loss(output[0], output[1], output[2],
                                      seg_corr, masks0, masks1, lambda_match=_lambda)
            def metrics_fn(output, seg_corr, masks0, masks1):
                return compute_matching_metrics_lg(output[0], output[1],
                                                   seg_corr, masks0, masks1)
            def score_mat(output, b, M, N):
                return output[0][b, :M, :N].cpu()

    else:  # sinkhorn (default)
        from training.models.sinkhorn import SegMASt3R, superglue_nll_loss as _nll
        if not mock:
            model = SegMASt3R(
                mast3r_ckpt=cfg.MODEL.MAST3R_CKPT,
                matcher_cfg=matcher_cfg,
                device="cpu",
            )
        def loss_fn(output, seg_corr, masks0, masks1):
            return _nll(output[0], seg_corr, masks0, masks1)
        def metrics_fn(output, seg_corr, masks0, masks1):
            return compute_matching_metrics(output[0].cpu(), seg_corr, masks0, masks1)
        def score_mat(output, b, M, N):
            return output[0][b, :M, :N].cpu()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.TRAINING.LR, weight_decay=cfg.TRAINING.WEIGHT_DECAY)

    # ── Prepare: wraps model in DDP, handles mixed precision ──────
    model, optimizer, loader_train, loader_val = accelerator.prepare(
        model, optimizer, loader_train, loader_val
    )

    # ── Scheduler ─────────────────────────────────────────────────
    # Built after prepare so len(loader_train) reflects the per-process count.
    iters_per_epoch = len(loader_train)
    total_steps     = iters_per_epoch * cfg.TRAINING.EPOCHS
    scheduler       = build_lr_scheduler(optimizer, cfg, total_steps)

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in trainable)
        print(f"Trainable params: {n_params:,}")
        print(f"Iters/epoch: {iters_per_epoch:,}  |  Total steps: {total_steps:,}")
        if writer:
            writer.add_text("config/trainable_params", str(n_params), 0)
            writer.add_text("config/yaml", str(dict(cfg)), 0)
            writer.add_text("config/iters_per_epoch", str(iters_per_epoch), 0)

    # ── Resume ────────────────────────────────────────────────────
    start_epoch  = 0
    global_step  = 0
    best_val_ma  = 0.0

    ckpt_path = resume or getattr(cfg, "RESUME", None) or ""
    if not ckpt_path:
        ckpt_path = None
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        accelerator.unwrap_model(model).load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        best_val_ma = ckpt.get("best_val_ma", 0.0)
        if accelerator.is_main_process:
            print(f"Resumed from {ckpt_path}  (epoch {start_epoch}, step {global_step})")

    # ── Training loop ─────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.TRAINING.EPOCHS):
        model.train()
        epoch_loss  = 0.0
        epoch_start = time.time()

        pbar = tqdm(loader_train,
                    desc=f"Epoch {epoch+1}/{cfg.TRAINING.EPOCHS}",
                    dynamic_ncols=True,
                    disable=not accelerator.is_local_main_process)

        for batch in pbar:
            optimizer.zero_grad()
            loss, output = train_step(model, batch, device, loss_fn)
            accelerator.backward(loss)

            if cfg.TRAINING.GRAD_CLIP > 0:
                accelerator.clip_grad_norm_(trainable, cfg.TRAINING.GRAD_CLIP)

            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += accelerator.num_processes
            lr_now       = scheduler.get_last_lr()[0]

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_now:.2e}")

            # ── TB: train scalars (main process only) ─────────────
            if accelerator.is_main_process and global_step % cfg.TRAINING.LOG_INTERVAL == 0:
                if writer:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/lr",   lr_now,      global_step)
                    unwrapped = accelerator.unwrap_model(model)
                    if hasattr(unwrapped, "matcher") and hasattr(unwrapped.matcher, "dustbin_score"):
                        ds_score = unwrapped.matcher.dustbin_score.item()
                        writer.add_scalar("train/dustbin_score", ds_score, global_step)
                tqdm.write(f"  step {global_step:>7d} | "
                           f"loss {loss.item():.4f} | lr {lr_now:.2e}")

            # ── Debug plots ───────────────────────────────────────
            if cfg.DEBUG and global_step % 500 == 0 and accelerator.is_main_process:
                M    = batch["masks0"][0].shape[0]
                N    = batch["masks1"][0].shape[0]
                lP   = score_mat(output, 0, M, N).unsqueeze(0)  # (1, M, N)
                corr = batch["seg_corr"][0]
                G_dbg = torch.zeros_like(lP)
                if corr.shape[0] > 0:
                    G_dbg[0, corr[:, 0], corr[:, 1]] = 1.0
                dsc0_dbg = output[-2]   # (B, D, M)
                dsc1_dbg = output[-1]   # (B, D, N)
                plot_sinkhorn_debug(str(save_dir), global_step,
                                    lP, G_dbg,
                                    torch.einsum("bdn,bdm->bnm", dsc0_dbg, dsc1_dbg))

            # ── Validation ────────────────────────────────────────
            if global_step % cfg.TRAINING.VAL_INTERVAL == 0:
                model.eval()
                # run_validation is a collective call (all processes participate)
                vm = run_validation(model, loader_val, device, accelerator,
                                    loss_fn=loss_fn, metrics_fn=metrics_fn,
                                    score_mat_fn=score_mat,
                                    writer=writer, global_step=global_step,
                                    n_vis_batches=4,
                                    vis_img_size=max(cfg.DATASET.HEIGHT, cfg.DATASET.WIDTH),
                                    vis_resize_mode=cfg.DATASET.RESIZE_MODE)
                model.train()

                if accelerator.is_main_process:
                    if writer:
                        writer.add_scalar("val/loss",              vm["loss"],              global_step)
                        writer.add_scalar("val/matching_accuracy", vm["matching_accuracy"], global_step)
                        writer.add_scalar("val/mean_gt_logprob",   vm["mean_gt_logprob"],   global_step)
                        writer.add_scalar("val/dustbin_rate",      vm["dustbin_rate"],      global_step)
                    tqdm.write(
                        f"  *** VAL {global_step} | "
                        f"loss={vm['loss']:.4f} | "
                        f"MA={vm['matching_accuracy']:.3f} | "
                        f"gt_lp={vm['mean_gt_logprob']:.3f} | "
                        f"dustbin={vm['dustbin_rate']:.3f} ***"
                    )

                # wait_for_everyone is a collective op — must be called by ALL ranks.
                # Moving it outside is_main_process prevents the NCCL deadlock where
                # rank 0 blocks on the barrier while rank 1 has already moved forward.
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    if vm["matching_accuracy"] > best_val_ma:
                        best_val_ma = vm["matching_accuracy"]
                        _save_ckpt(save_dir / "best.pth", model, optimizer, scheduler,
                                   epoch, global_step, vm, best_val_ma=best_val_ma,
                                   accelerator=accelerator)
                        tqdm.write(f"  → New best  MA={best_val_ma:.3f}")

            # ── Periodic checkpoint ───────────────────────────────
            if global_step % cfg.TRAINING.SAVE_INTERVAL == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    _save_ckpt(save_dir / f"step_{global_step:07d}.pth",
                               model, optimizer, scheduler,
                               epoch, global_step, {"loss": loss.item()},
                               best_val_ma=best_val_ma,
                               accelerator=accelerator)

        # ── End of epoch ──────────────────────────────────────────
        avg_loss = epoch_loss / iters_per_epoch
        elapsed  = time.time() - epoch_start

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if writer:
                writer.add_scalar("train/epoch_loss", avg_loss, epoch + 1)
            print(f"Epoch {epoch+1} | avg_loss={avg_loss:.4f} | {elapsed/60:.1f} min")
            _save_ckpt(save_dir / f"epoch_{epoch+1:03d}.pth",
                       model, optimizer, scheduler,
                       epoch + 1, global_step, {"loss": avg_loss},
                       best_val_ma=best_val_ma,
                       accelerator=accelerator)

    if writer:
        writer.close()
    if accelerator.is_main_process:
        print(f"\nDone. Best val MA: {best_val_ma:.3f}")
        print(f"Tensorboard: tensorboard --logdir {save_dir / 'tb'}")
