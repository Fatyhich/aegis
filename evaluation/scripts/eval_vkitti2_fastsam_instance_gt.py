"""
Evaluation on Virtual KITTI 2: FastSAM segmentation + Instance ID ground truth

Pipeline:
1. Use FastSAM to generate masks (realistic input)
2. Use Virtual KITTI 2 instance IDs (trackIDs) to define ground truth
3. Assign each FastSAM mask to a trackID based on maximum overlap
4. Two FastSAM masks match if they have the same trackID
"""

import sys
from pathlib import Path

# Add evaluation/ directory to path for local imports
_eval_dir = Path(__file__).parent.parent
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import yaml
from collections import defaultdict

from core.segmentor import SegmentationPipeline
from core.model_infer import MASt3RSegFeatInfer
from core.eval_metrics import compute_metrics, aggregate_metrics_by_bin, print_table2_format
from eval_with_fastsam import run_fastsam_on_image, denormalize_image
from PIL import Image
import matplotlib.pyplot as plt


def load_vkitti2_instance_mask(mask_path: str) -> np.ndarray:
    """
    Load instance mask from Virtual KITTI 2.

    Args:
        mask_path: Path to instancegt PNG file

    Returns:
        Instance mask (H, W) with trackIDs (0 = background)
    """
    img = Image.open(mask_path)
    mask = np.array(img, dtype=np.uint8)

    # In VKITTI2: pixel_value = trackID + 1
    # Convert to trackIDs: trackID = pixel_value - 1
    # Keep 0 as background
    track_ids = mask.copy()
    track_ids[track_ids > 0] = track_ids[track_ids > 0] - 1

    return track_ids


def assign_fastsam_masks_to_trackids(fastsam_masks, track_img):
    """
    Assign each FastSAM mask to a trackID based on maximum overlap.

    Args:
        fastsam_masks: (M, H, W) FastSAM binary masks
        track_img: (H, W) trackID image (0 = background)

    Returns:
        track_ids: List of M trackIDs (one per FastSAM mask)
                   -1 if mask doesn't overlap with any tracked object
    """
    M = fastsam_masks.shape[0]
    track_ids = []

    for i in range(M):
        mask = fastsam_masks[i].numpy() if torch.is_tensor(fastsam_masks[i]) else fastsam_masks[i]

        # Get trackIDs that overlap with this mask
        overlapping_pixels = track_img[mask > 0]

        if len(overlapping_pixels) == 0:
            # No overlap
            track_ids.append(-1)
            continue

        # Find most common trackID (excluding background 0)
        unique_ids, counts = np.unique(overlapping_pixels, return_counts=True)

        # Filter out background
        valid_mask = unique_ids > 0
        if not valid_mask.any():
            track_ids.append(-1)
            continue

        unique_ids = unique_ids[valid_mask]
        counts = counts[valid_mask]

        # Get trackID with maximum overlap
        best_id = unique_ids[np.argmax(counts)]
        track_ids.append(int(best_id))

    return track_ids


def generate_trackid_gt_for_fastsam(track_ids0, track_ids1):
    """
    Generate ground truth based on trackIDs assigned to FastSAM masks.

    Args:
        track_ids0: List of M trackIDs for masks0
        track_ids1: List of N trackIDs for masks1

    Returns:
        Binary ground truth matrix (M, N)
    """
    M = len(track_ids0)
    N = len(track_ids1)

    gt = np.zeros((M, N), dtype=np.uint8)

    for i in range(M):
        for j in range(N):
            id0 = track_ids0[i]
            id1 = track_ids1[j]

            # Match if same trackID and both are valid (> 0)
            if id0 > 0 and id1 > 0 and id0 == id1:
                gt[i, j] = 1

    return gt


def visualize_vkitti2_results(
    img0_tensor,
    img1_tensor,
    fastsam_masks0,
    fastsam_masks1,
    track_ids0,
    track_ids1,
    scores,
    gt_matrix,
    save_path,
    pair_info=None
):
    """Visualize Virtual KITTI 2 matching results."""
    img0_np = denormalize_image(img0_tensor)
    img1_np = denormalize_image(img1_tensor)

    M = fastsam_masks0.shape[0]
    N = fastsam_masks1.shape[0]

    # Create figure
    fig = plt.figure(figsize=(20, 10))

    if pair_info:
        title = f"Scene: {pair_info.get('scene', 'unknown')}, Variant: {pair_info.get('variant', 'unknown')}\n"
        title += f"Frames: {pair_info.get('idx0', '?')}->{pair_info.get('idx1', '?')}, "
        title += f"Angle: {pair_info.get('angle', 0):.1f}°"
        fig.suptitle(title, fontsize=14, fontweight='bold')

    # Row 1: Images and FastSAM masks
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(img0_np)
    ax1.set_title(f'Image 0', fontsize=11)
    ax1.axis('off')

    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(img1_np)
    ax2.set_title(f'Image 1', fontsize=11)
    ax2.axis('off')

    ax3 = plt.subplot(2, 4, 3)
    overlay0 = np.zeros_like(img0_np)
    for i in range(M):
        color = plt.cm.tab20(i % 20)[:3]
        overlay0[fastsam_masks0[i] > 0] = color
    ax3.imshow(img0_np * 0.5 + overlay0 * 0.5)
    ax3.set_title(f'FastSAM Masks ({M} segments)', fontsize=11)
    ax3.axis('off')

    ax4 = plt.subplot(2, 4, 4)
    overlay1 = np.zeros_like(img1_np)
    for i in range(N):
        color = plt.cm.tab20(i % 20)[:3]
        overlay1[fastsam_masks1[i] > 0] = color
    ax4.imshow(img1_np * 0.5 + overlay1 * 0.5)
    ax4.set_title(f'FastSAM Masks ({N} segments)', fontsize=11)
    ax4.axis('off')

    # Row 2: GT, Scores, and Info
    ax5 = plt.subplot(2, 4, 5)
    im1 = ax5.imshow(gt_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax5.set_xlabel('Target segments (Img 1)', fontsize=9)
    ax5.set_ylabel('Query segments (Img 0)', fontsize=9)
    ax5.set_title(f'GT (TrackIDs)\n{int(gt_matrix.sum())} matches', fontsize=11)
    plt.colorbar(im1, ax=ax5, fraction=0.046)

    ax6 = plt.subplot(2, 4, 6)
    im2 = ax6.imshow(scores, aspect='auto', cmap='viridis')
    ax6.set_xlabel('Target segments (Img 1)', fontsize=9)
    ax6.set_ylabel('Query segments (Img 0)', fontsize=9)
    ax6.set_title('Predicted Scores', fontsize=11)
    plt.colorbar(im2, ax=ax6, fraction=0.046)

    # TrackID info
    ax7 = plt.subplot(2, 4, 7)
    info_text = "FastSAM → TrackIDs:\n\n"
    info_text += "Image 0 (first 10):\n"
    for i in range(min(10, M)):
        info_text += f"  Mask {i}: trackID={track_ids0[i]}\n"
    info_text += f"\nImage 1 (first 10):\n"
    for i in range(min(10, N)):
        info_text += f"  Mask {i}: trackID={track_ids1[i]}\n"

    ax7.text(0.05, 0.95, info_text, fontsize=8,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1)
    ax7.axis('off')

    # Metrics
    ax8 = plt.subplot(2, 4, 8)
    from core.eval_metrics import compute_metrics
    metrics = compute_metrics(scores, gt_matrix)

    metrics_text = "Metrics:\n\n"
    metrics_text += f"AUPRC: {metrics['AUPRC']:.3f}\n"
    metrics_text += f"R@1:   {metrics['R@1']:.3f}\n"
    metrics_text += f"R@5:   {metrics['R@5']:.3f}\n"
    metrics_text += f"\nQueries: {metrics['num_queries']}\n"
    metrics_text += f"GT matches: {int(gt_matrix.sum())}\n\n"

    # Count shared trackIDs
    shared_ids = set([id for id in track_ids0 if id > 0]) & \
                 set([id for id in track_ids1 if id > 0])
    metrics_text += f"Shared vehicles: {len(shared_ids)}"

    ax8.text(0.1, 0.5, metrics_text, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    ax8.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_gt_vs_predicted_pairs(
    img0_tensor,
    img1_tensor,
    fastsam_masks0,
    fastsam_masks1,
    track_ids0,
    track_ids1,
    scores,
    gt_matrix,
    save_path,
    num_examples=5
):
    """
    Visualize GT vs Predicted mask pairs side-by-side for VKITTI2.
    """
    img0_np = denormalize_image(img0_tensor)
    img1_np = denormalize_image(img1_tensor)

    M = fastsam_masks0.shape[0]
    N = fastsam_masks1.shape[0]

    # Find all GT matches
    gt_pairs = []
    for i in range(M):
        for j in range(N):
            if gt_matrix[i, j] == 1:
                pred_j = np.argmax(scores[i])
                pred_score = scores[i, pred_j]
                is_correct = (pred_j == j)
                gt_score = scores[i, j]

                gt_pairs.append({
                    'query_idx': i,
                    'gt_target_idx': j,
                    'pred_target_idx': pred_j,
                    'track_id': track_ids0[i],
                    'gt_score': gt_score,
                    'pred_score': pred_score,
                    'is_correct': is_correct
                })

    if len(gt_pairs) == 0:
        return

    gt_pairs.sort(key=lambda x: x['gt_score'], reverse=True)
    num_to_show = min(num_examples, len(gt_pairs))

    fig, axes = plt.subplots(num_to_show, 5, figsize=(20, 4*num_to_show))
    if num_to_show == 1:
        axes = axes.reshape(1, -1)

    for row, pair_info in enumerate(gt_pairs[:num_to_show]):
        query_idx = pair_info['query_idx']
        gt_idx = pair_info['gt_target_idx']
        pred_idx = pair_info['pred_target_idx']
        track_id = pair_info['track_id']
        gt_score = pair_info['gt_score']
        pred_score = pair_info['pred_score']
        is_correct = pair_info['is_correct']

        # Column 0: Query image
        axes[row, 0].imshow(img0_np)
        axes[row, 0].set_title(f'Image 0\nQuery {query_idx}', fontsize=10)
        axes[row, 0].axis('off')

        # Column 1: Query mask
        colored_img0 = img0_np.copy()
        mask0 = fastsam_masks0[query_idx]
        colored_img0[mask0 > 0] = colored_img0[mask0 > 0] * 0.5 + np.array([1, 1, 0]) * 0.5
        axes[row, 1].imshow(colored_img0)
        axes[row, 1].set_title(f'Query Mask\nTrackID={track_id}', fontsize=10)
        axes[row, 1].axis('off')

        # Column 2: GT match
        colored_img1_gt = img1_np.copy()
        mask1_gt = fastsam_masks1[gt_idx]
        colored_img1_gt[mask1_gt > 0] = colored_img1_gt[mask1_gt > 0] * 0.5 + np.array([0, 1, 0]) * 0.5
        axes[row, 2].imshow(colored_img1_gt)
        axes[row, 2].set_title(f'GT MATCH\nTarget {gt_idx}\nScore: {gt_score:.3f}', fontsize=10, color='green')
        axes[row, 2].axis('off')

        # Column 3: Predicted match
        colored_img1_pred = img1_np.copy()
        mask1_pred = fastsam_masks1[pred_idx]
        pred_color = [0, 1, 0] if is_correct else [1, 0, 0]
        colored_img1_pred[mask1_pred > 0] = colored_img1_pred[mask1_pred > 0] * 0.5 + np.array(pred_color) * 0.5

        status = "✓ CORRECT" if is_correct else "✗ WRONG"
        title_color = 'green' if is_correct else 'red'
        axes[row, 3].imshow(colored_img1_pred)
        axes[row, 3].set_title(f'PREDICTED\nTarget {pred_idx}\nScore: {pred_score:.3f}\n{status}',
                              fontsize=10, color=title_color)
        axes[row, 3].axis('off')

        # Column 4: Info
        info_text = f"Query: {query_idx}\n"
        info_text += f"TrackID: {track_id}\n\n"
        info_text += f"GT Match: {gt_idx}\n"
        info_text += f"GT Score: {gt_score:.3f}\n\n"
        info_text += f"Predicted: {pred_idx}\n"
        info_text += f"Pred Score: {pred_score:.3f}\n\n"

        if is_correct:
            info_text += "✓ Model found\n  correct match!"
            bg_color = 'lightgreen'
        else:
            info_text += f"✗ Model predicted\n  wrong target!\n"
            info_text += f"  (should be {gt_idx})"
            bg_color = 'lightcoral'

        axes[row, 4].text(0.1, 0.5, info_text, fontsize=9,
                         verticalalignment='center', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.5))
        axes[row, 4].set_xlim(0, 1)
        axes[row, 4].set_ylim(0, 1)
        axes[row, 4].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_vkitti2_fastsam_instance_gt(
    pairs_file,
    data_root,
    checkpoint_path,
    output_dir,
    config_path,
    num_pairs=None,
    device='cuda',
    save_visualizations=False,
    num_vis_samples=10
):
    """
    Evaluate SegMASt3R on Virtual KITTI 2 using FastSAM masks + TrackID ground truth.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_visualizations:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['MODEL']['CHECKPOINT'] = checkpoint_path

    # Load pairs
    with open(pairs_file, 'r') as f:
        pairs = json.load(f)

    if num_pairs is not None:
        pairs = pairs[:num_pairs]

    print(f"Evaluating {len(pairs)} pairs on Virtual KITTI 2")
    print("Method: FastSAM segmentation + TrackID ground truth")

    # Initialize FastSAM
    print("\nInitializing FastSAM...")
    fastsam_checkpoint = "checkpoints/ultralytics/FastSAM-x.pt"
    segmentor = SegmentationPipeline(fastsam_model_path=fastsam_checkpoint)

    # Initialize SegMASt3R
    print("\nInitializing SegMASt3R...")
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = MASt3RSegFeatInfer(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.prepare(device)

    # Run evaluation
    print("\n" + "="*80)
    print("Running evaluation...")
    print("="*80 + "\n")

    results_by_bin = defaultdict(list)
    bin_names = [f"{b[0]}-{b[1]}" for b in cfg['EVAL']['POSE_BINS']]

    target_h = cfg['DATASET']['RESIZE_H']
    target_w = cfg['DATASET']['RESIZE_W']

    vis_counter = 0
    data_root = Path(data_root)

    for pair_idx, pair in enumerate(tqdm(pairs, desc="Evaluating pairs")):
        scene = pair['scene']
        variant = pair['variant']
        idx0 = pair['idx0']
        idx1 = pair['idx1']
        pose_bin = pair['pose_bin']

        # Get paths
        rgb0_path = data_root / "vkitti_rgb" / scene / variant / "frames" / "rgb" / "Camera_0" / f"rgb_{idx0:05d}.jpg"
        rgb1_path = data_root / "vkitti_rgb" / scene / variant / "frames" / "rgb" / "Camera_0" / f"rgb_{idx1:05d}.jpg"

        inst0_path = data_root / "vkitti_instanceSegmentation" / scene / variant / "frames" / "instanceSegmentation" / "Camera_0" / f"instancegt_{idx0:05d}.png"
        inst1_path = data_root / "vkitti_instanceSegmentation" / scene / variant / "frames" / "instanceSegmentation" / "Camera_0" / f"instancegt_{idx1:05d}.png"

        if not rgb0_path.exists() or not rgb1_path.exists():
            print(f"\nWarning: RGB images not found for {scene}/{variant} frames {idx0}-{idx1}")
            continue

        if not inst0_path.exists() or not inst1_path.exists():
            print(f"\nWarning: Instance masks not found for {scene}/{variant} frames {idx0}-{idx1}")
            continue

        # Run FastSAM
        try:
            fastsam_masks0, img0 = run_fastsam_on_image(segmentor, rgb0_path, target_h, target_w)
            fastsam_masks1, img1 = run_fastsam_on_image(segmentor, rgb1_path, target_h, target_w)
        except Exception as e:
            print(f"\nError with FastSAM on pair {idx0}-{idx1}: {e}")
            continue

        if fastsam_masks0.shape[0] == 0 or fastsam_masks1.shape[0] == 0:
            continue

        # Load trackID masks
        track_img0 = load_vkitti2_instance_mask(str(inst0_path))
        track_img1 = load_vkitti2_instance_mask(str(inst1_path))

        # Resize to match target size
        track_img0_resized = np.array(Image.fromarray(track_img0).resize((target_w, target_h), Image.NEAREST))
        track_img1_resized = np.array(Image.fromarray(track_img1).resize((target_w, target_h), Image.NEAREST))

        # Assign FastSAM masks to trackIDs
        track_ids0 = assign_fastsam_masks_to_trackids(fastsam_masks0, track_img0_resized)
        track_ids1 = assign_fastsam_masks_to_trackids(fastsam_masks1, track_img1_resized)

        # Generate ground truth
        gt_matrix = generate_trackid_gt_for_fastsam(track_ids0, track_ids1)

        # Skip if no valid matches
        if gt_matrix.sum() == 0:
            continue

        # Run SegMASt3R
        img0_batch = img0.unsqueeze(0).to(device)
        img1_batch = img1.unsqueeze(0).to(device)
        masks0_batch = fastsam_masks0.unsqueeze(0).to(device)
        masks1_batch = fastsam_masks1.unsqueeze(0).to(device)

        with torch.no_grad():
            match_result, scores = model.infer_pair(img0_batch, img1_batch, masks0_batch, masks1_batch)

        scores_np = scores[0].cpu().numpy()

        # Compute metrics
        metrics = compute_metrics(scores_np, gt_matrix)

        if metrics['num_queries'] > 0:
            bin_name = bin_names[pose_bin]
            results_by_bin[bin_name].append(metrics)

            # Save visualizations
            if save_visualizations and vis_counter < num_vis_samples:
                pair_info = {
                    'scene': scene,
                    'variant': variant,
                    'idx0': idx0,
                    'idx1': idx1,
                    'angle': pair.get('angle', 0)
                }

                vis_path = vis_dir / f"pair_{pair_idx:04d}_vkitti2.png"
                visualize_vkitti2_results(
                    img0, img1,
                    fastsam_masks0.numpy(), fastsam_masks1.numpy(),
                    track_ids0, track_ids1,
                    scores_np, gt_matrix, vis_path, pair_info
                )

                # GT vs Predicted visualization
                vis_path2 = vis_dir / f"pair_{pair_idx:04d}_gt_vs_pred.png"
                visualize_gt_vs_predicted_pairs(
                    img0, img1,
                    fastsam_masks0.numpy(), fastsam_masks1.numpy(),
                    track_ids0, track_ids1,
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
    results_json_path = output_dir / "metrics_vkitti2_fastsam_trackid_gt.json"
    with open(results_json_path, 'w') as f:
        json.dump(aggregated_metrics, f, indent=2)
    print(f"\nSaved metrics to: {results_json_path}")

    table_txt_path = output_dir / "table2_vkitti2_results.txt"
    with open(table_txt_path, 'w') as f:
        f.write(table_str)
    print(f"Saved table to: {table_txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SegMASt3R on Virtual KITTI 2")
    parser.add_argument("--pairs", type=str, default="pairs_vkitti2.json")
    parser.add_argument("--data_root", type=str, default="/mnt/vol3/datasets/virtual-KITTI-2")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/segmast3r_spp.ckpt")
    parser.add_argument("--config", type=str, default="configs/config_eval_vkitti2.yaml")
    parser.add_argument("--output_dir", type=str, default="results/vkitti2_fastsam_trackid_gt")
    parser.add_argument("--num_pairs", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--num_vis", type=int, default=10)

    args = parser.parse_args()

    evaluate_vkitti2_fastsam_instance_gt(
        pairs_file=args.pairs,
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        config_path=args.config,
        num_pairs=args.num_pairs,
        device=args.device,
        save_visualizations=args.visualize,
        num_vis_samples=args.num_vis
    )


if __name__ == "__main__":
    main()
