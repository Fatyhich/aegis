"""
Evaluation with FastSAM segmentation (realistic experiment matching the paper).

Instead of using ground truth instance masks, we:
1. Run FastSAM on all images to generate masks
2. Define ground truth using IoU > 0.5 between masks across frames
3. Evaluate segment matching performance
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import yaml
from collections import defaultdict

from segmentor import SegmentationPipeline
from model_infer import MASt3RSegFeatInfer
from eval_metrics import compute_metrics, aggregate_metrics_by_bin, print_table2_format
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt


# Image normalization (same as in replica_dataset.py)
ImgNorm = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def denormalize_image(img_tensor):
    """Convert from [-1, 1] normalized tensor to [0, 1] numpy array."""
    img = img_tensor.cpu().numpy()
    img = (img + 1) / 2  # [-1, 1] -> [0, 1]
    img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    img = np.clip(img, 0, 1)
    return img


def visualize_fastsam_results(
    img0_tensor,
    img1_tensor,
    masks0,
    masks1,
    scores,
    gt_matrix,
    save_path,
    pair_info=None
):
    """
    Visualize FastSAM segmentation and matching results.

    Args:
        img0_tensor: (3, H, W) normalized image tensor
        img1_tensor: (3, H, W) normalized image tensor
        masks0: (M, H, W) binary masks
        masks1: (N, H, W) binary masks
        scores: (M, N) matching scores
        gt_matrix: (M, N) IoU-based ground truth
        save_path: Path to save visualization
        pair_info: Dict with scene, idx0, idx1, angle info
    """
    # Denormalize images
    img0_np = denormalize_image(img0_tensor)
    img1_np = denormalize_image(img1_tensor)

    M = masks0.shape[0]
    N = masks1.shape[0]

    # Create figure
    fig = plt.figure(figsize=(20, 10))

    # Title with pair info
    if pair_info:
        title = f"Scene: {pair_info.get('scene', 'unknown')}, " \
                f"Frames: {pair_info.get('idx0', '?')}->{pair_info.get('idx1', '?')}, " \
                f"Angle: {pair_info.get('angle', 0):.1f}°"
        fig.suptitle(title, fontsize=14, fontweight='bold')

    # Row 1: Original images
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(img0_np)
    ax1.set_title(f'Image 0', fontsize=11)
    ax1.axis('off')

    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(img1_np)
    ax2.set_title(f'Image 1', fontsize=11)
    ax2.axis('off')

    # Row 1: FastSAM masks overlay
    ax3 = plt.subplot(2, 4, 3)
    mask_overlay0 = np.zeros_like(img0_np)
    for i in range(M):
        color = plt.cm.tab20(i % 20)[:3]
        mask_overlay0[masks0[i] > 0] = color
    ax3.imshow(img0_np * 0.5 + mask_overlay0 * 0.5)
    ax3.set_title(f'FastSAM Masks ({M} segments)', fontsize=11)
    ax3.axis('off')

    ax4 = plt.subplot(2, 4, 4)
    mask_overlay1 = np.zeros_like(img1_np)
    for i in range(N):
        color = plt.cm.tab20(i % 20)[:3]
        mask_overlay1[masks1[i] > 0] = color
    ax4.imshow(img1_np * 0.5 + mask_overlay1 * 0.5)
    ax4.set_title(f'FastSAM Masks ({N} segments)', fontsize=11)
    ax4.axis('off')

    # Row 2: GT and Predicted score matrices
    ax5 = plt.subplot(2, 4, 5)
    im1 = ax5.imshow(gt_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax5.set_xlabel('Target segments (Img 1)', fontsize=9)
    ax5.set_ylabel('Query segments (Img 0)', fontsize=9)
    ax5.set_title(f'Ground Truth (IoU>0.5)\n{int(gt_matrix.sum())} matches', fontsize=11)
    plt.colorbar(im1, ax=ax5, fraction=0.046)

    ax6 = plt.subplot(2, 4, 6)
    im2 = ax6.imshow(scores, aspect='auto', cmap='viridis')
    ax6.set_xlabel('Target segments (Img 1)', fontsize=9)
    ax6.set_ylabel('Query segments (Img 0)', fontsize=9)
    ax6.set_title('Predicted Scores', fontsize=11)
    plt.colorbar(im2, ax=ax6, fraction=0.046)

    # Row 2: Top matches visualization
    ax7 = plt.subplot(2, 4, 7)
    # Show top-3 predicted matches
    top_matches_text = "Top-3 Predictions:\n"
    for i in range(min(3, M)):
        top_idx = np.argmax(scores[i])
        top_score = scores[i, top_idx]
        is_correct = gt_matrix[i, top_idx] == 1 if top_idx < N else False
        status = "✓" if is_correct else "✗"
        top_matches_text += f"{status} Q{i}->T{top_idx}: {top_score:.3f}\n"

    ax7.text(0.1, 0.5, top_matches_text, fontsize=10,
             verticalalignment='center', fontfamily='monospace')
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1)
    ax7.axis('off')
    ax7.set_title('Match Examples', fontsize=11)

    # Row 2: Metrics
    ax8 = plt.subplot(2, 4, 8)
    # Compute quick metrics
    from eval_metrics import compute_metrics
    metrics = compute_metrics(scores, gt_matrix)

    metrics_text = "Metrics:\n\n"
    metrics_text += f"AUPRC: {metrics['AUPRC']:.3f}\n"
    metrics_text += f"R@1:   {metrics['R@1']:.3f}\n"
    metrics_text += f"R@5:   {metrics['R@5']:.3f}\n"
    metrics_text += f"\nQueries: {metrics['num_queries']}\n"
    metrics_text += f"GT matches: {int(gt_matrix.sum())}"

    ax8.text(0.1, 0.5, metrics_text, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    ax8.axis('off')
    ax8.set_title('Performance', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_match_examples(
    img0_tensor,
    img1_tensor,
    masks0,
    masks1,
    scores,
    gt_matrix,
    save_path,
    num_examples=5
):
    """
    Visualize individual segment matches side-by-side.
    Shows both predicted and GT matches for comparison.
    """
    img0_np = denormalize_image(img0_tensor)
    img1_np = denormalize_image(img1_tensor)

    M = masks0.shape[0]
    N = masks1.shape[0]

    # Find queries with GT matches
    queries_with_gt = []
    for i in range(M):
        if gt_matrix[i].sum() > 0:
            pred_idx = np.argmax(scores[i])
            pred_score = scores[i, pred_idx]
            is_correct = gt_matrix[i, pred_idx] == 1

            # Get GT match with highest IoU (first match in GT row)
            gt_indices = np.where(gt_matrix[i] > 0)[0]
            if len(gt_indices) > 0:
                # Pick the one with highest score among GT matches
                gt_idx = gt_indices[np.argmax(scores[i, gt_indices])]
            else:
                gt_idx = None

            queries_with_gt.append({
                'query_idx': i,
                'pred_idx': pred_idx,
                'gt_idx': gt_idx,
                'score': pred_score,
                'is_correct': is_correct
            })

    if len(queries_with_gt) == 0:
        return

    # Sort by score
    queries_with_gt.sort(key=lambda x: x['score'], reverse=True)
    num_to_show = min(num_examples, len(queries_with_gt))

    # Create figure with 6 columns: Query, Query Seg, Pred Img, Pred Match, GT Img, GT Match
    fig, axes = plt.subplots(num_to_show, 6, figsize=(20, 4*num_to_show))
    if num_to_show == 1:
        axes = axes.reshape(1, -1)

    for row, match_info in enumerate(queries_with_gt[:num_to_show]):
        query_idx = match_info['query_idx']
        pred_idx = match_info['pred_idx']
        gt_idx = match_info['gt_idx']
        score = match_info['score']
        is_correct = match_info['is_correct']

        # Column 0-1: Query segment
        mask0 = masks0[query_idx]
        colored_img0 = img0_np.copy()
        colored_img0[mask0 > 0] = colored_img0[mask0 > 0] * 0.5 + np.array([0, 1, 0]) * 0.5

        axes[row, 0].imshow(img0_np)
        axes[row, 0].set_title(f'Query {query_idx}', fontsize=10)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(colored_img0)
        axes[row, 1].set_title('Query Segment', fontsize=10)
        axes[row, 1].axis('off')

        # Column 2-3: Predicted match
        mask1_pred = masks1[pred_idx] if pred_idx < N else np.zeros_like(masks1[0])
        pred_color = [0, 1, 0] if is_correct else [1, 0, 0]
        colored_img1_pred = img1_np.copy()
        colored_img1_pred[mask1_pred > 0] = colored_img1_pred[mask1_pred > 0] * 0.5 + np.array(pred_color) * 0.5

        axes[row, 2].imshow(img1_np)
        axes[row, 2].set_title(f'Target {pred_idx}', fontsize=9)
        axes[row, 2].axis('off')

        pred_status = "✓ CORRECT" if is_correct else "✗ WRONG"
        axes[row, 3].imshow(colored_img1_pred)
        axes[row, 3].set_title(f'PREDICTED\nScore: {score:.3f}\n{pred_status}', fontsize=9)
        axes[row, 3].axis('off')

        # Column 4-5: GT match
        if gt_idx is not None:
            mask1_gt = masks1[gt_idx]
            colored_img1_gt = img1_np.copy()
            colored_img1_gt[mask1_gt > 0] = colored_img1_gt[mask1_gt > 0] * 0.5 + np.array([0, 1, 0]) * 0.5

            axes[row, 4].imshow(img1_np)
            axes[row, 4].set_title(f'Target {gt_idx}', fontsize=9)
            axes[row, 4].axis('off')

            # Show GT score for this match
            gt_score = scores[query_idx, gt_idx]
            axes[row, 5].imshow(colored_img1_gt)
            axes[row, 5].set_title(f'GROUND TRUTH\nScore: {gt_score:.3f}\n✓ (IoU>0.5)', fontsize=9)
            axes[row, 5].axis('off')
        else:
            # No GT match (shouldn't happen if we filtered correctly)
            axes[row, 4].axis('off')
            axes[row, 5].text(0.5, 0.5, 'No GT\nmatch', ha='center', va='center', fontsize=10)
            axes[row, 5].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_mask_iou(mask1, mask2):
    """
    Compute IoU between two binary masks.

    Args:
        mask1: (H, W) binary mask
        mask2: (H, W) binary mask

    Returns:
        IoU value in [0, 1]
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return intersection / union


def generate_iou_ground_truth(masks0, masks1, iou_threshold=0.5):
    """
    Generate ground truth based on IoU between masks.

    Args:
        masks0: (M, H, W) binary masks from image 0
        masks1: (N, H, W) binary masks from image 1
        iou_threshold: Minimum IoU to consider a match (default 0.5)

    Returns:
        Binary ground truth matrix (M, N)
    """
    M = masks0.shape[0]
    N = masks1.shape[0]

    gt = np.zeros((M, N), dtype=np.uint8)

    for i in range(M):
        for j in range(N):
            iou = compute_mask_iou(masks0[i], masks1[j])
            if iou >= iou_threshold:
                gt[i, j] = 1

    return gt


def run_fastsam_on_image(segmentor, image_path, target_h, target_w, max_masks=50):
    """
    Run FastSAM on an image and return masks.

    Args:
        segmentor: SegmentationPipeline instance
        image_path: Path to image
        target_h: Target height for resizing
        target_w: Target width for resizing
        max_masks: Maximum number of masks to keep

    Returns:
        masks: (M, H, W) binary masks tensor
        image: (3, H, W) normalized image tensor
    """
    # Run FastSAM
    results = segmentor.segment(str(image_path), conf=0.4, iou=0.9, imgsz=1024)

    # Load and preprocess image
    img_pil = Image.open(image_path).convert('RGB')
    img_pil = img_pil.resize((target_w, target_h), Image.BILINEAR)
    img_tensor = ImgNorm(img_pil)

    # Extract masks
    if results.masks is None or len(results.masks) == 0:
        # No masks found
        return torch.zeros((0, target_h, target_w), dtype=torch.uint8), img_tensor

    # Get mask data
    masks_data = results.masks.data.cpu().numpy()  # (M, H_orig, W_orig)

    # Resize masks to target size
    masks_list = []
    for mask in masks_data:
        # Convert to 0-255 for cv2.resize
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_resized = Image.fromarray(mask_uint8).resize((target_w, target_h), Image.NEAREST)
        mask_binary = (np.array(mask_resized) > 127).astype(np.uint8)
        masks_list.append(mask_binary)

    # Limit number of masks
    if len(masks_list) > max_masks:
        masks_list = masks_list[:max_masks]

    if len(masks_list) == 0:
        return torch.zeros((0, target_h, target_w), dtype=torch.uint8), img_tensor

    masks_tensor = torch.from_numpy(np.stack(masks_list, axis=0))

    return masks_tensor, img_tensor


def evaluate_with_fastsam(
    pairs_file,
    data_root,
    checkpoint_path,
    output_dir,
    config_path,
    num_pairs=None,
    device='cuda',
    iou_threshold=0.5,
    save_visualizations=False,
    num_vis_samples=10
):
    """
    Run evaluation using FastSAM for segmentation.

    Args:
        save_visualizations: If True, save visualization images
        num_vis_samples: Number of pairs to visualize (if save_visualizations=True)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_visualizations:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        print(f"Visualizations will be saved to: {vis_dir}")
    else:
        vis_dir = None

    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['MODEL']['CHECKPOINT'] = checkpoint_path
    cfg['DATASET']['DATA_ROOT'] = data_root

    # Load pairs
    with open(pairs_file, 'r') as f:
        pairs = json.load(f)

    if num_pairs is not None:
        pairs = pairs[:num_pairs]

    print(f"Evaluating {len(pairs)} pairs with FastSAM segmentation")
    print(f"IoU threshold for ground truth: {iou_threshold}")

    # Initialize FastSAM
    print("\nInitializing FastSAM...")
    fastsam_checkpoint = "checkpoints/ultralytics/FastSAM-x.pt"
    if not Path(fastsam_checkpoint).exists():
        print(f"ERROR: FastSAM checkpoint not found at {fastsam_checkpoint}")
        print("Please download FastSAM-x.pt")
        return

    segmentor = SegmentationPipeline(fastsam_model_path=fastsam_checkpoint)

    # Initialize model
    print("\nInitializing SegMASt3R model...")
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = MASt3RSegFeatInfer(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.prepare(device)

    # Run evaluation
    print("\n" + "="*80)
    print("Running evaluation with FastSAM...")
    print("="*80 + "\n")

    results_by_bin = defaultdict(list)
    bin_names = [f"{b[0]}-{b[1]}" for b in cfg['EVAL']['POSE_BINS']]

    target_h = cfg['DATASET']['RESIZE_H']
    target_w = cfg['DATASET']['RESIZE_W']

    vis_counter = 0  # Counter for visualizations

    for pair_idx, pair in enumerate(tqdm(pairs, desc="Evaluating pairs")):
        scene = pair['scene']
        idx0 = pair['idx0']
        idx1 = pair['idx1']
        pose_bin = pair['pose_bin']

        # Get image paths
        data_root_path = Path(data_root)
        img0_path = data_root_path / scene / "Sequence_1" / "rgb" / f"rgb_{idx0}.png"
        img1_path = data_root_path / scene / "Sequence_1" / "rgb" / f"rgb_{idx1}.png"

        # Run FastSAM on both images
        try:
            masks0, img0 = run_fastsam_on_image(segmentor, img0_path, target_h, target_w)
            masks1, img1 = run_fastsam_on_image(segmentor, img1_path, target_h, target_w)
        except Exception as e:
            print(f"\nError processing pair {idx0}-{idx1}: {e}")
            continue

        # Skip if no masks
        if masks0.shape[0] == 0 or masks1.shape[0] == 0:
            continue

        # Prepare for model inference
        img0_batch = img0.unsqueeze(0).to(device)
        img1_batch = img1.unsqueeze(0).to(device)
        masks0_batch = masks0.unsqueeze(0).to(device)
        masks1_batch = masks1.unsqueeze(0).to(device)

        # Run model
        with torch.no_grad():
            match_result, scores = model.infer_pair(img0_batch, img1_batch, masks0_batch, masks1_batch)

        scores_np = scores[0].cpu().numpy()

        # Generate IoU-based ground truth
        masks0_np = masks0.numpy()
        masks1_np = masks1.numpy()
        gt_matrix = generate_iou_ground_truth(masks0_np, masks1_np, iou_threshold)

        # Compute metrics
        metrics = compute_metrics(scores_np, gt_matrix)

        if metrics['num_queries'] > 0:
            bin_name = bin_names[pose_bin]
            results_by_bin[bin_name].append(metrics)

            # Save visualizations if enabled
            if save_visualizations and vis_counter < num_vis_samples:
                pair_info = {
                    'scene': scene,
                    'idx0': idx0,
                    'idx1': idx1,
                    'angle': pair.get('angle', 0)
                }

                # Save overview
                vis_path = vis_dir / f"pair_{pair_idx:04d}_overview.png"
                visualize_fastsam_results(
                    img0, img1, masks0_np, masks1_np,
                    scores_np, gt_matrix, vis_path, pair_info
                )

                # Save match examples
                vis_path2 = vis_dir / f"pair_{pair_idx:04d}_matches.png"
                visualize_match_examples(
                    img0, img1, masks0_np, masks1_np,
                    scores_np, gt_matrix, vis_path2, num_examples=5
                )

                vis_counter += 1

    # Aggregate results
    print("\n" + "="*80)
    print("Aggregating results...")
    print("="*80 + "\n")

    aggregated_metrics = aggregate_metrics_by_bin(results_by_bin)

    # Print results
    table_str = print_table2_format(aggregated_metrics)
    print(table_str)

    # Save results
    results_json_path = output_dir / "metrics_by_bin_fastsam.json"
    with open(results_json_path, 'w') as f:
        json.dump(aggregated_metrics, f, indent=2)
    print(f"\nSaved metrics to: {results_json_path}")

    table_txt_path = output_dir / "table2_results_fastsam.txt"
    with open(table_txt_path, 'w') as f:
        f.write(table_str)
    print(f"Saved table to: {table_txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate with FastSAM segmentation")
    parser.add_argument("--pairs", type=str, default="pairs_replica_3200.json")
    parser.add_argument("--data_root", type=str, default="/mnt/vol3/datasets/semantic-replica")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/segmast3r_spp.ckpt")
    parser.add_argument("--config", type=str, default="configs/config_eval_replica.yaml")
    parser.add_argument("--output_dir", type=str, default="results/replica_fastsam")
    parser.add_argument("--num_pairs", type=int, default=None, help="Limit number of pairs (for testing)")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for GT")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visualize", action="store_true", help="Save visualizations (default: False)")
    parser.add_argument("--num_vis", type=int, default=10, help="Number of pairs to visualize (default: 10)")

    args = parser.parse_args()

    evaluate_with_fastsam(
        pairs_file=args.pairs,
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        config_path=args.config,
        num_pairs=args.num_pairs,
        device=args.device,
        iou_threshold=args.iou_threshold,
        save_visualizations=args.visualize,
        num_vis_samples=args.num_vis
    )


if __name__ == "__main__":
    main()
