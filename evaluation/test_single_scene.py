"""
Quick test script for SegMASt3R on a single Replica scene with visualization.

Samples a few image pairs from one scene, runs inference, and saves visualizations.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import yaml

from replica_dataset import ReplicaSegmentMatchDataset
from model_infer import MASt3RSegFeatInfer
from ground_truth_generator import generate_instance_correspondences
from eval_metrics import compute_metrics


def visualize_matches(
    img0_np,
    img1_np,
    masks0,
    masks1,
    instance_ids0,
    instance_ids1,
    scores,
    gt_matrix,
    match_indices,
    save_path,
    top_k=5
):
    """
    Visualize segment matching results.

    Args:
        img0_np: (H, W, 3) RGB image 0, values in [0, 1]
        img1_np: (H, W, 3) RGB image 1, values in [0, 1]
        masks0: (M, H, W) binary masks for image 0
        masks1: (N, H, W) binary masks for image 1
        instance_ids0: List of M instance IDs for image 0
        instance_ids1: List of N instance IDs for image 1
        scores: (M, N) matching scores
        gt_matrix: (M, N) ground truth binary matrix
        match_indices: (M,) predicted match indices (-1 for no match)
        save_path: Path to save visualization
        top_k: Number of top matches to visualize
    """
    M = len(instance_ids0)
    N = len(instance_ids1)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))

    # Plot images with all masks
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(img0_np)
    ax1.set_title(f'Image 0 ({M} segments)', fontsize=12)
    ax1.axis('off')

    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(img1_np)
    ax2.set_title(f'Image 1 ({N} segments)', fontsize=12)
    ax2.axis('off')

    # Plot masks overlay
    ax3 = plt.subplot(2, 3, 3)
    # Create colored overlay for masks0
    mask_overlay0 = np.zeros_like(img0_np)
    for i, mask in enumerate(masks0):
        color = plt.cm.tab20(i % 20)[:3]
        mask_overlay0[mask > 0] = color
    ax3.imshow(img0_np * 0.5 + mask_overlay0 * 0.5)
    ax3.set_title('Segments in Image 0', fontsize=12)
    ax3.axis('off')

    ax4 = plt.subplot(2, 3, 4)
    # Create colored overlay for masks1
    mask_overlay1 = np.zeros_like(img1_np)
    for i, mask in enumerate(masks1):
        color = plt.cm.tab20(i % 20)[:3]
        mask_overlay1[mask > 0] = color
    ax4.imshow(img1_np * 0.5 + mask_overlay1 * 0.5)
    ax4.set_title('Segments in Image 1', fontsize=12)
    ax4.axis('off')

    # Plot score matrix as heatmap
    ax5 = plt.subplot(2, 3, 5)
    im = ax5.imshow(scores, aspect='auto', cmap='viridis')
    ax5.set_xlabel('Target segments (Image 1)', fontsize=10)
    ax5.set_ylabel('Query segments (Image 0)', fontsize=10)
    ax5.set_title('Matching Scores', fontsize=12)
    plt.colorbar(im, ax=ax5)

    # Plot ground truth matrix
    ax6 = plt.subplot(2, 3, 6)
    im2 = ax6.imshow(gt_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax6.set_xlabel('Target segments (Image 1)', fontsize=10)
    ax6.set_ylabel('Query segments (Image 0)', fontsize=10)
    ax6.set_title('Ground Truth Matches', fontsize=12)
    plt.colorbar(im2, ax=ax6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_top_matches(
    img0_np,
    img1_np,
    masks0,
    masks1,
    instance_ids0,
    instance_ids1,
    scores,
    gt_matrix,
    save_path,
    num_examples=5
):
    """
    Visualize top-k segment matches side-by-side.

    Shows query segment from img0 and its top predicted match from img1,
    along with ground truth label.
    """
    M = len(instance_ids0)

    # Find queries with matches
    queries_with_gt = []
    for i in range(M):
        if gt_matrix[i].sum() > 0:  # Has at least one GT match
            top_match_idx = np.argmax(scores[i])
            top_score = scores[i, top_match_idx]
            is_correct = gt_matrix[i, top_match_idx] == 1
            queries_with_gt.append((i, top_match_idx, top_score, is_correct))

    # Sort by score (show highest confidence predictions)
    queries_with_gt.sort(key=lambda x: x[2], reverse=True)

    # Take top num_examples
    num_to_show = min(num_examples, len(queries_with_gt))

    if num_to_show == 0:
        print(f"No matches to visualize, skipping {save_path}")
        return

    # Create figure
    fig, axes = plt.subplots(num_to_show, 4, figsize=(16, 4*num_to_show))
    if num_to_show == 1:
        axes = axes.reshape(1, -1)

    for row, (query_idx, match_idx, score, is_correct) in enumerate(queries_with_gt[:num_to_show]):
        # Query segment from img0
        mask0 = masks0[query_idx]
        colored_img0 = img0_np.copy()
        colored_img0[mask0 > 0] = colored_img0[mask0 > 0] * 0.5 + np.array([0, 1, 0]) * 0.5  # Green overlay

        axes[row, 0].imshow(img0_np)
        axes[row, 0].set_title(f'Query {query_idx}\nID: {instance_ids0[query_idx]}', fontsize=10)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(colored_img0)
        axes[row, 1].set_title('Query Segment', fontsize=10)
        axes[row, 1].axis('off')

        # Matched segment from img1
        mask1 = masks1[match_idx]
        colored_img1 = img1_np.copy()
        color = [0, 1, 0] if is_correct else [1, 0, 0]  # Green if correct, red if wrong
        colored_img1[mask1 > 0] = colored_img1[mask1 > 0] * 0.5 + np.array(color) * 0.5

        axes[row, 2].imshow(img1_np)
        axes[row, 2].set_title(f'Target {match_idx}\nID: {instance_ids1[match_idx]}', fontsize=10)
        axes[row, 2].axis('off')

        axes[row, 3].imshow(colored_img1)
        status = "✓ CORRECT" if is_correct else "✗ WRONG"
        axes[row, 3].set_title(f'Predicted Match\nScore: {score:.3f} {status}', fontsize=10)
        axes[row, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def denormalize_image(img_tensor):
    """Convert from [-1, 1] normalized tensor to [0, 1] numpy array."""
    img = img_tensor.cpu().numpy()
    img = (img + 1) / 2  # [-1, 1] -> [0, 1]
    img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    img = np.clip(img, 0, 1)
    return img


def test_scene(
    scene_name: str,
    num_pairs: int,
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    data_root: str = None,
    instance_mask_root: str = None,
    device: str = 'cuda'
):
    """
    Test on a single scene and save visualizations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Testing on scene: {scene_name}")
    print(f"Output directory: {output_dir}")

    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Override data paths if provided
    if data_root is not None:
        cfg['DATASET']['DATA_ROOT'] = data_root
    if instance_mask_root is not None:
        cfg['DATASET']['INSTANCE_MASK_ROOT'] = instance_mask_root

    # Update checkpoint path
    cfg['MODEL']['CHECKPOINT'] = checkpoint_path

    # Print paths being used
    print(f"Data root: {cfg['DATASET']['DATA_ROOT']}")
    print(f"Instance mask root: {cfg['DATASET']['INSTANCE_MASK_ROOT']}")

    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Sample some pairs from the scene
    print(f"\nSampling {num_pairs} pairs from {scene_name}...")
    from sample_pairs import sample_pairs_for_scene

    scene_path = Path(cfg['DATASET']['DATA_ROOT']) / scene_name
    pose_bins = [(0, 45), (45, 90), (90, 135), (135, 180)]

    # Sample fewer pairs per bin for quick test
    pairs_per_bin = max(1, num_pairs // len(pose_bins))

    pairs = sample_pairs_for_scene(
        scene_path=scene_path,
        num_pairs_per_bin=pairs_per_bin,
        temporal_gap_min=5,
        pose_bins=pose_bins,
        random_seed=42
    )

    # Add scene name
    for pair in pairs:
        pair['scene'] = scene_name

    print(f"Sampled {len(pairs)} pairs")

    # Create dataset
    dataset = ReplicaSegmentMatchDataset(
        data_root=cfg['DATASET']['DATA_ROOT'],
        instance_mask_root=cfg['DATASET']['INSTANCE_MASK_ROOT'],
        pairs=pairs,
        target_h=cfg['DATASET']['RESIZE_H'],
        target_w=cfg['DATASET']['RESIZE_W'],
        m_prime=cfg['DATASET'].get('M_PRIME', None)
    )

    # Load model
    print("\nLoading model...")
    model = MASt3RSegFeatInfer(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.prepare(device)
    print("Model loaded!")

    # Run inference and visualize
    print(f"\nRunning inference on {len(dataset)} pairs...")

    results = []

    for idx in tqdm(range(min(num_pairs, len(dataset)))):
        sample = dataset[idx]

        # Prepare batch
        img0 = sample['img0'].unsqueeze(0).to(device)  # (1, 3, H, W)
        img1 = sample['img1'].unsqueeze(0).to(device)
        masks0 = sample['masks0'].unsqueeze(0).to(device)  # (1, M, H, W)
        masks1 = sample['masks1'].unsqueeze(0).to(device)
        instance_ids0 = sample['instance_ids0']
        instance_ids1 = sample['instance_ids1']

        # Skip if no masks
        if len(instance_ids0) == 0 or len(instance_ids1) == 0:
            print(f"  Skipping pair {idx}: no segments")
            continue

        # Run inference
        with torch.no_grad():
            match_result, scores = model.infer_pair(img0, img1, masks0, masks1)

        # Convert to numpy
        scores_np = scores[0].cpu().numpy()  # (M, N)
        match_indices = match_result[0].cpu().numpy()  # (M,)

        # Generate ground truth
        gt_matrix = generate_instance_correspondences(instance_ids0, instance_ids1)

        # Compute metrics
        metrics = compute_metrics(scores_np, gt_matrix)

        # Prepare images for visualization
        img0_np = denormalize_image(sample['img0'])
        img1_np = denormalize_image(sample['img1'])
        masks0_np = sample['masks0'].cpu().numpy()
        masks1_np = sample['masks1'].cpu().numpy()

        # Save overview visualization
        vis_path = output_dir / f"pair_{idx:03d}_overview.png"
        visualize_matches(
            img0_np, img1_np,
            masks0_np, masks1_np,
            instance_ids0, instance_ids1,
            scores_np, gt_matrix, match_indices,
            vis_path
        )

        # Save top matches visualization
        vis_path2 = output_dir / f"pair_{idx:03d}_matches.png"
        visualize_top_matches(
            img0_np, img1_np,
            masks0_np, masks1_np,
            instance_ids0, instance_ids1,
            scores_np, gt_matrix,
            vis_path2,
            num_examples=5
        )

        results.append({
            'idx': idx,
            'scene': scene_name,
            'idx0': sample['idx0'],
            'idx1': sample['idx1'],
            'angle': sample['angle'],
            'pose_bin': sample['pose_bin'],
            'num_segments0': len(instance_ids0),
            'num_segments1': len(instance_ids1),
            'metrics': metrics
        })

        print(f"  Pair {idx}: {sample['idx0']}->{sample['idx1']}, "
              f"angle={sample['angle']:.1f}°, "
              f"segments={len(instance_ids0)}x{len(instance_ids1)}, "
              f"AUPRC={metrics['AUPRC']:.3f}, R@1={metrics['R@1']:.3f}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if len(results) > 0:
        avg_auprc = np.mean([r['metrics']['AUPRC'] for r in results])
        avg_r1 = np.mean([r['metrics']['R@1'] for r in results])
        avg_r5 = np.mean([r['metrics']['R@5'] for r in results])

        print(f"Scene: {scene_name}")
        print(f"Pairs evaluated: {len(results)}")
        print(f"Average AUPRC: {avg_auprc:.3f}")
        print(f"Average R@1: {avg_r1:.3f}")
        print(f"Average R@5: {avg_r5:.3f}")
        print(f"\nVisualizations saved to: {output_dir}")
        print(f"Total images: {len(results) * 2}")
    else:
        print("No valid pairs found!")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Test SegMASt3R on a single Replica scene")
    parser.add_argument(
        "--scene",
        type=str,
        default="office_0",
        help="Scene name (e.g., office_0, room_0)"
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=10,
        help="Number of pairs to test"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_eval_replica.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/segmast3r_spp.ckpt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to Replica dataset root (overrides config)"
    )
    parser.add_argument(
        "--instance_mask_root",
        type=str,
        default=None,
        help="Path to instance masks root (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/test_single_scene",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )

    args = parser.parse_args()

    test_scene(
        scene_name=args.scene,
        num_pairs=args.num_pairs,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        data_root=args.data_root,
        instance_mask_root=args.instance_mask_root,
        device=args.device
    )


if __name__ == "__main__":
    main()
