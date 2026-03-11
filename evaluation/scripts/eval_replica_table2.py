"""
Main Evaluation Script for SegMASt3R Table 2 Reproduction on Replica Dataset

Orchestrates the full evaluation pipeline:
1. Load configuration and pre-sampled image pairs
2. Initialize model from checkpoint
3. Run inference on all pairs
4. Compute metrics (AUPRC, R@1, R@5) grouped by pose bin
5. Print results in Table 2 format
6. Save detailed results to JSON
"""

import sys
from pathlib import Path

# Add evaluation/ directory to path for local imports
_eval_dir = Path(__file__).parent.parent
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

import argparse
import json
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import numpy as np

from datasets.replica_dataset import ReplicaSegmentMatchDataset, collate_fn, load_pairs_from_json
from core.model_infer import MASt3RSegFeatInfer
from core.ground_truth_generator import generate_instance_correspondences
from core.eval_metrics import compute_metrics, aggregate_metrics_by_bin, print_table2_format


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(cfg: dict, device: torch.device) -> MASt3RSegFeatInfer:
    """
    Initialize model and load checkpoint.

    Args:
        cfg: Configuration dictionary
        device: Device to load model on

    Returns:
        Initialized model ready for inference
    """
    print("Initializing model...")

    # Create model
    model = MASt3RSegFeatInfer(cfg)

    # Load checkpoint
    checkpoint_path = cfg['MODEL']['CHECKPOINT']
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Prepare model for inference
    model.prepare(device)

    print("Model loaded successfully!")
    return model


def evaluate_batch(
    model: MASt3RSegFeatInfer,
    batch: dict,
    device: torch.device
) -> dict:
    """
    Evaluate a single batch and compute metrics.

    Args:
        model: Initialized model
        batch: Batch dictionary from dataloader
        device: Device for computation

    Returns:
        Dictionary with metrics for each sample in batch
    """
    # Extract batch data
    img0 = batch['img0'].to(device)  # (B, 3, H, W)
    img1 = batch['img1'].to(device)  # (B, 3, H, W)
    masks0_list = batch['masks0']  # List of (M_i, H, W) tensors
    masks1_list = batch['masks1']  # List of (N_i, H, W) tensors
    instance_ids0_list = batch['instance_ids0']
    instance_ids1_list = batch['instance_ids1']
    pose_bins = batch['pose_bin']

    batch_size = img0.shape[0]
    results = []

    # Process each sample in batch
    for i in range(batch_size):
        # Get single sample
        img0_single = img0[i:i+1]  # (1, 3, H, W)
        img1_single = img1[i:i+1]  # (1, 3, H, W)
        masks0_single = masks0_list[i].unsqueeze(0).to(device)  # (1, M, H, W)
        masks1_single = masks1_list[i].unsqueeze(0).to(device)  # (1, N, H, W)
        instance_ids0 = instance_ids0_list[i]
        instance_ids1 = instance_ids1_list[i]
        pose_bin = pose_bins[i]

        # Skip if no masks in either frame
        if len(instance_ids0) == 0 or len(instance_ids1) == 0:
            results.append({
                'pose_bin': pose_bin,
                'metrics': {
                    'AUPRC': 0.0,
                    'R@1': 0.0,
                    'R@5': 0.0,
                    'num_queries': 0
                }
            })
            continue

        # Run model inference
        with torch.no_grad():
            match_result, scores = model.infer_pair(
                img0_single, img1_single, masks0_single, masks1_single
            )
            # match_result: (1, M), scores: (1, M, N)

        # Convert to numpy
        scores_np = scores[0].cpu().numpy()  # (M, N)

        # Generate ground truth correspondences
        gt_matrix = generate_instance_correspondences(instance_ids0, instance_ids1)  # (M, N)

        # Compute metrics
        metrics = compute_metrics(scores_np, gt_matrix)

        results.append({
            'pose_bin': pose_bin,
            'metrics': metrics
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SegMASt3R on Replica dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_eval_replica.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference"
    )

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    print(f"Loaded config from: {args.config}")

    # Override output dir if specified
    if args.output_dir is not None:
        cfg['EVAL']['OUTPUT_DIR'] = args.output_dir

    output_dir = Path(cfg['EVAL']['OUTPUT_DIR'])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pairs
    pairs_file = cfg['EVAL']['PAIRS_FILE']
    if not Path(pairs_file).exists():
        raise FileNotFoundError(
            f"Pairs file not found: {pairs_file}\n"
            "Please run sample_pairs.py first to generate the pairs file."
        )

    pairs = load_pairs_from_json(pairs_file)
    print(f"Loaded {len(pairs)} pairs from {pairs_file}")

    # Print distribution by pose bin
    bin_names = [f"{b[0]}-{b[1]}" for b in cfg['EVAL']['POSE_BINS']]
    bin_counts = [0] * len(bin_names)
    for pair in pairs:
        bin_counts[pair['pose_bin']] += 1

    print("\nPair distribution by pose bin:")
    for bin_name, count in zip(bin_names, bin_counts):
        print(f"  {bin_name}°: {count} pairs")

    # Create dataset
    dataset = ReplicaSegmentMatchDataset(
        data_root=cfg['DATASET']['DATA_ROOT'],
        instance_mask_root=cfg['DATASET']['INSTANCE_MASK_ROOT'],
        pairs=pairs,
        target_h=cfg['DATASET']['RESIZE_H'],
        target_w=cfg['DATASET']['RESIZE_W'],
        m_prime=cfg['DATASET'].get('M_PRIME', None)
    )

    print(f"\nDataset size: {len(dataset)}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg['EVAL']['BATCH_SIZE'],
        shuffle=False,
        num_workers=cfg['EVAL']['NUM_WORKERS'],
        collate_fn=collate_fn
    )

    print(f"Batch size: {cfg['EVAL']['BATCH_SIZE']}")
    print(f"Number of batches: {len(dataloader)}")

    # Setup model
    model = setup_model(cfg, device)

    # Run evaluation
    print("\n" + "="*80)
    print("Starting evaluation...")
    print("="*80 + "\n")

    results_by_bin = defaultdict(list)

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch_results = evaluate_batch(model, batch, device)

        # Accumulate results by pose bin
        for result in batch_results:
            pose_bin = result['pose_bin']
            bin_name = bin_names[pose_bin]
            results_by_bin[bin_name].append(result['metrics'])

    # Aggregate metrics by pose bin
    print("\n" + "="*80)
    print("Aggregating results...")
    print("="*80 + "\n")

    aggregated_metrics = aggregate_metrics_by_bin(results_by_bin)

    # Print results in Table 2 format
    table_str = print_table2_format(aggregated_metrics)
    print(table_str)

    # Save results to JSON
    results_json_path = output_dir / "metrics_by_bin.json"
    with open(results_json_path, 'w') as f:
        json.dump(aggregated_metrics, f, indent=2)
    print(f"\nSaved detailed metrics to: {results_json_path}")

    # Save formatted table to text file
    table_txt_path = output_dir / "table2_results.txt"
    with open(table_txt_path, 'w') as f:
        f.write(table_str)
    print(f"Saved formatted table to: {table_txt_path}")

    # Save all raw results
    raw_results_path = output_dir / "raw_results.json"
    raw_results = {
        'config': cfg,
        'results_by_bin': {k: v for k, v in results_by_bin.items()},
        'aggregated_metrics': aggregated_metrics
    }
    with open(raw_results_path, 'w') as f:
        json.dump(raw_results, f, indent=2)
    print(f"Saved raw results to: {raw_results_path}")

    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
