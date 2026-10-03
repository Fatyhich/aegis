"""
Benchmark torch.compile on frozen VGGT Aggregator.

Measures wall-clock time per forward pass with and without compilation.
Run: uv run --active python training/scripts/bench_compile.py

Optional args:
  --ckpt PATH      path to VGGT weights (default: third_party/vggt_weights.pt)
  --n-frames N     number of frames (default: 4)
  --batch-size B   batch size (default: 2)
  --warmup W       warmup iterations (default: 5)
  --iters I        timed iterations (default: 20)
"""

import argparse
import time
import sys
import os

import torch
import torch.nn.functional as F

# Setup VGGT path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from training.paths import setup_vggt_path
setup_vggt_path()
from vggt.models.aggregator import Aggregator


def load_aggregator(ckpt_path: str, device: str) -> Aggregator:
    agg = Aggregator()
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    prefix = "aggregator."
    agg_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not agg_state:
        agg_state = state
    agg.load_state_dict(agg_state, strict=True)
    agg.eval()
    for p in agg.parameters():
        p.requires_grad_(False)
    return agg.to(device)


def bench(agg, images, tag, warmup, iters):
    """Run warmup + timed iterations, return mean ms/iter."""
    torch.cuda.synchronize()

    for _ in range(warmup):
        with torch.no_grad():
            agg(images)
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            agg(images)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    print(f"  {tag}: {mean:.1f} ± {std:.1f} ms/iter  "
          f"(min={min(times):.1f}, max={max(times):.1f})")
    return mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="third_party/vggt_weights.pt")
    parser.add_argument("--n-frames", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Batch={args.batch_size}, N_frames={args.n_frames}, "
          f"warmup={args.warmup}, iters={args.iters}")
    print()

    # Load
    print("Loading aggregator...")
    agg = load_aggregator(args.ckpt, device)
    print(f"  Params: {sum(p.numel() for p in agg.parameters()):,}")

    # Fake input: (B, N, 3, 336, 504) → resize to 336x504 (multiple of 14)
    H, W = 336, 504
    images = torch.randn(args.batch_size, args.n_frames, 3, H, W,
                         device=device, dtype=torch.float32)
    # Normalize to [0, 1] (VGGT expects this)
    images = images.sigmoid()

    # --- Eager ---
    print("\n[1/4] Eager (baseline):")
    t_eager = bench(agg, images, "eager", args.warmup, args.iters)

    # --- TF32 ---
    print("\n[2/4] Eager + TF32 matmul precision:")
    torch.set_float32_matmul_precision("high")
    t_tf32 = bench(agg, images, "eager+tf32", args.warmup, args.iters)

    # --- torch.compile default (TF32 stays on) ---
    print("\n[3/4] torch.compile + TF32:")
    try:
        agg_compiled = torch.compile(agg)
        # Extra warmup for compilation
        print("  Compiling (first pass may be slow)...")
        with torch.no_grad():
            agg_compiled(images)
        torch.cuda.synchronize()
        t_default = bench(agg_compiled, images, "compile-default",
                         args.warmup, args.iters)
    except Exception as e:
        print(f"  FAILED: {e}")
        t_default = None

    # --- torch.compile reduce-overhead ---
    print("\n[4/4] torch.compile reduce-overhead + TF32:")
    try:
        agg_ro = torch.compile(agg, mode="reduce-overhead")
        print("  Compiling (first pass may be slow)...")
        with torch.no_grad():
            agg_ro(images)
        torch.cuda.synchronize()
        t_ro = bench(agg_ro, images, "compile-reduce-overhead",
                    args.warmup, args.iters)
    except Exception as e:
        print(f"  FAILED: {e}")
        t_ro = None

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  eager:                   {t_eager:.1f} ms/iter")
    print(f"  eager+tf32:              {t_tf32:.1f} ms/iter  ({t_eager/t_tf32:.2f}x)")
    if t_default:
        print(f"  compile+tf32:            {t_default:.1f} ms/iter  ({t_eager/t_default:.2f}x)")
    if t_ro:
        print(f"  compile-ro+tf32:         {t_ro:.1f} ms/iter  ({t_eager/t_ro:.2f}x)")
    print("=" * 60)

    # Memory
    if device == "cuda":
        print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
