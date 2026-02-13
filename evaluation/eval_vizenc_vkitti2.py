"""
Evaluate VizEnc on Virtual KITTI 2 dataset with SAM1 segmentation + trackID GT.

Similar to eval_vkitti2_fastsam_instance_gt.py but using VizEnc (SAM1 + DINOv2/NaRADIO).
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from vizenc_inference import VizEncMatcher, load_and_resize_image
from eval_metrics import compute_metrics, aggregate_metrics_by_bin, print_table2_format


def load_vkitti2_instance_mask(instance_mask_path):
    """
    Load Virtual KITTI 2 instance mask (PNG format).

    Format: pixel_value = trackID + 1
    Background has trackID = 0 (pixel value = 1)
    """
    mask_img = Image.open(instance_mask_path)
    mask_array = np.array(mask_img)

    # Convert pixel values to trackIDs
    track_ids = mask_array.astype(np.int32) - 1

    return track_ids


def assign_sam_masks_to_trackids(sam_masks, track_img):
    """
    Assign each SAM mask to a trackID based on maximum overlap.

    Args:
        sam_masks: numpy array (M, H, W) - binary masks from SAM
        track_img: numpy array (H, W) - trackIDs

    Returns:
        List of trackIDs for each mask (length M)
    """
    track_ids = []

    for mask in sam_masks:
        # Get trackIDs covered by this mask
        covered_tracks = track_img[mask]

        if len(covered_tracks) == 0:
            track_ids.append(-1)  # No coverage
            continue

        # Count pixels per trackID
        unique, counts = np.unique(covered_tracks, return_counts=True)

        # Remove background (trackID = 0)
        valid_idx = unique > 0
        if not valid_idx.any():
            track_ids.append(-1)
            continue

        valid_unique = unique[valid_idx]
        valid_counts = counts[valid_idx]

        # Assign to trackID with maximum overlap
        best_track = valid_unique[np.argmax(valid_counts)]
        track_ids.append(int(best_track))

    return track_ids


def generate_trackid_gt_for_sam(track_ids0, track_ids1):
    """
    Generate ground truth matrix where gt[i,j] = 1 if same trackID.

    Args:
        track_ids0: List of trackIDs for masks in image 0
        track_ids1: List of trackIDs for masks in image 1

    Returns:
        numpy array (M, N) - binary ground truth matrix
    """
    M = len(track_ids0)
    N = len(track_ids1)
    gt_matrix = np.zeros((M, N), dtype=np.float32)

    for i, id0 in enumerate(track_ids0):
        if id0 == -1:  # Invalid mask
            continue

        for j, id1 in enumerate(track_ids1):
            if id1 == -1:
                continue

            if id0 == id1:
                gt_matrix[i, j] = 1.0

    return gt_matrix


def denormalize_image(img_tensor):
    """Convert from [-1, 1] normalized tensor to [0, 1] numpy array for visualization."""
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.cpu().numpy()
        if img.ndim == 3 and img.shape[0] == 3:  # (C, H, W)
            img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        img = (img + 1) / 2  # [-1, 1] -> [0, 1]
        img = np.clip(img, 0, 1)
        return img
    else:
        # Already numpy array
        img = np.array(img_tensor)
        if img.max() > 1.0:
            img = img / 255.0
        return np.clip(img, 0, 1)


def visualize_vkitti2_results(img0, img1, masks0, masks1, track_ids0, track_ids1,
                               scores, gt_matrix, save_path, pair_info):
    """Visualize VizEnc matching results for VKITTI2 (overview)."""
    fig = plt.figure(figsize=(20, 10))

    # Original images
    ax1 = plt.subplot(2, 3, 1)
    if isinstance(img0, torch.Tensor):
        img0_np = img0.permute(1, 2, 0).cpu().numpy()
        img0_np = (img0_np + 1) / 2  # [-1,1] -> [0,1]
    else:
        img0_np = np.array(img0) / 255.0
    ax1.imshow(img0_np)
    ax1.set_title(f"Image 0 (Frame {pair_info['idx0']})")
    ax1.axis('off')

    ax2 = plt.subplot(2, 3, 2)
    if isinstance(img1, torch.Tensor):
        img1_np = img1.permute(1, 2, 0).cpu().numpy()
        img1_np = (img1_np + 1) / 2
    else:
        img1_np = np.array(img1) / 255.0
    ax2.imshow(img1_np)
    ax2.set_title(f"Image 1 (Frame {pair_info['idx1']})")
    ax2.axis('off')

    # SAM masks visualization
    ax3 = plt.subplot(2, 3, 3)
    if len(masks0) > 0:
        mask_overlay = np.zeros((*masks0[0].shape, 3))
        for idx, mask in enumerate(masks0):
            color = np.random.rand(3)
            mask_overlay[mask] = color
        ax3.imshow(img0_np)
        ax3.imshow(mask_overlay, alpha=0.5)
    ax3.set_title(f"SAM Masks 0 ({len(masks0)} masks)")
    ax3.axis('off')

    ax4 = plt.subplot(2, 3, 4)
    if len(masks1) > 0:
        mask_overlay = np.zeros((*masks1[0].shape, 3))
        for idx, mask in enumerate(masks1):
            color = np.random.rand(3)
            mask_overlay[mask] = color
        ax4.imshow(img1_np)
        ax4.imshow(mask_overlay, alpha=0.5)
    ax4.set_title(f"SAM Masks 1 ({len(masks1)} masks)")
    ax4.axis('off')

    # Match scores heatmap
    ax5 = plt.subplot(2, 3, 5)
    if scores.size > 0:
        im = ax5.imshow(scores, cmap='hot', aspect='auto', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax5, label='Cosine Similarity')
    ax5.set_xlabel('Mask Index (Image 1)')
    ax5.set_ylabel('Mask Index (Image 0)')
    ax5.set_title('Match Scores (Cosine Similarity)')

    # Ground truth matrix
    ax6 = plt.subplot(2, 3, 6)
    if gt_matrix.size > 0:
        im = ax6.imshow(gt_matrix, cmap='Greys', aspect='auto', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax6, label='GT Match')
    ax6.set_xlabel('Mask Index (Image 1)')
    ax6.set_ylabel('Mask Index (Image 0)')
    ax6.set_title('Ground Truth (TrackID Match)')

    # Add scene info
    scene_info = f"Scene: {pair_info['scene']}, Variant: {pair_info['variant']}"
    if 'angle' in pair_info:
        scene_info += f", Angle: {pair_info['angle']:.1f}°"
    fig.suptitle(f"VizEnc Matching Results\n{scene_info}", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def visualize_gt_vs_predicted_pairs(
    img0, img1,
    masks0, masks1,
    track_ids0, track_ids1,
    scores, gt_matrix,
    save_path, num_examples=5
):
    """
    Visualize GT mask pairs vs Predicted mask pairs side-by-side for VKITTI2.

    Shows:
    - Query mask from Image 0
    - GT match from Image 1 (based on trackIDs)
    - Predicted match from Image 1 (based on VizEnc scores)

    Args:
        img0, img1: PIL Images or numpy arrays
        masks0, masks1: numpy arrays (M, H, W) and (N, H, W)
        track_ids0, track_ids1: Lists of trackIDs
        scores: numpy array (M, N) - similarity scores
        gt_matrix: numpy array (M, N) - ground truth
        save_path: Path to save visualization
        num_examples: Number of examples to show
    """
    # Convert images to numpy if needed
    img0_np = denormalize_image(img0)
    img1_np = denormalize_image(img1)

    M = masks0.shape[0]
    N = masks1.shape[0]

    # Find all GT matches
    gt_pairs = []
    for i in range(M):
        for j in range(N):
            if gt_matrix[i, j] == 1:
                # Get predicted match for query i
                pred_j = np.argmax(scores[i])
                pred_score = scores[i, pred_j]
                is_correct = (pred_j == j)

                # Get score for GT match
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
        print("No GT pairs to visualize")
        return

    # Sort by GT score (show best GT matches first)
    gt_pairs.sort(key=lambda x: x['gt_score'], reverse=True)
    num_to_show = min(num_examples, len(gt_pairs))

    # Create figure: 5 columns (Image0, Query, GT Match, Predicted, Info)
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

        # Column 1: Query mask highlighted
        colored_img0 = img0_np.copy()
        mask0 = masks0[query_idx]
        colored_img0[mask0 > 0] = colored_img0[mask0 > 0] * 0.5 + np.array([1, 1, 0]) * 0.5  # Yellow
        axes[row, 1].imshow(colored_img0)
        axes[row, 1].set_title(f'Query Mask\nTrackID={track_id}', fontsize=10)
        axes[row, 1].axis('off')

        # Column 2: GT match from Image 1
        colored_img1_gt = img1_np.copy()
        mask1_gt = masks1[gt_idx]
        colored_img1_gt[mask1_gt > 0] = colored_img1_gt[mask1_gt > 0] * 0.5 + np.array([0, 1, 0]) * 0.5  # Green
        axes[row, 2].imshow(colored_img1_gt)
        axes[row, 2].set_title(f'GT MATCH\nTarget {gt_idx}\nScore: {gt_score:.3f}', fontsize=10, color='green')
        axes[row, 2].axis('off')

        # Column 3: Predicted match from Image 1
        colored_img1_pred = img1_np.copy()
        mask1_pred = masks1[pred_idx]
        pred_color = [0, 1, 0] if is_correct else [1, 0, 0]  # Green if correct, red if wrong
        colored_img1_pred[mask1_pred > 0] = colored_img1_pred[mask1_pred > 0] * 0.5 + np.array(pred_color) * 0.5

        status = "✓ CORRECT" if is_correct else "✗ WRONG"
        title_color = 'green' if is_correct else 'red'
        axes[row, 3].imshow(colored_img1_pred)
        axes[row, 3].set_title(f'PREDICTED\nTarget {pred_idx}\nScore: {pred_score:.3f}\n{status}',
                              fontsize=10, color=title_color)
        axes[row, 3].axis('off')

        # Column 4: Comparison info
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


def evaluate_vizenc_vkitti2(
    pairs_file: str,
    data_root: str,
    output_dir: str,
    config_path: str,
    encoder_type: str = 'dinov2',
    encoder_model: str = None,
    sam_checkpoint: str = None,
    num_pairs: int = None,
    device: str = 'cuda',
    save_visualizations: bool = False,
    num_vis_samples: int = 10
):
    """Run VizEnc evaluation on Virtual KITTI 2 dataset."""

    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Load pairs
    with open(pairs_file, 'r') as f:
        pairs = json.load(f)

    if num_pairs is not None:
        pairs = pairs[:num_pairs]

    print(f"Evaluating {len(pairs)} pairs on Virtual KITTI 2")
    print(f"Method: VizEnc (SAM1 + {encoder_type.upper()})")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_visualizations:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

    # Initialize VizEnc
    print("\nInitializing VizEnc...")
    matcher = VizEncMatcher(
        encoder_type=encoder_type,
        sam_checkpoint=sam_checkpoint,
        encoder_model=encoder_model,
        device=device
    )

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

        # Load and resize images
        img0 = load_and_resize_image(rgb0_path, target_h, target_w)
        img1 = load_and_resize_image(rgb1_path, target_h, target_w)

        # Run VizEnc matching (with target size for consistent dimensions)
        try:
            masks0, masks1, scores = matcher.match_pair(
                rgb0_path, rgb1_path, target_size=(target_h, target_w)
            )
        except Exception as e:
            print(f"\nError with VizEnc on pair {idx0}-{idx1}: {e}")
            continue

        if len(masks0) == 0 or len(masks1) == 0:
            continue

        # Load trackID masks
        track_img0 = load_vkitti2_instance_mask(str(inst0_path))
        track_img1 = load_vkitti2_instance_mask(str(inst1_path))

        # Resize to match target size
        track_img0_resized = np.array(Image.fromarray(track_img0).resize((target_w, target_h), Image.NEAREST))
        track_img1_resized = np.array(Image.fromarray(track_img1).resize((target_w, target_h), Image.NEAREST))

        # Assign SAM masks to trackIDs
        track_ids0 = assign_sam_masks_to_trackids(masks0, track_img0_resized)
        track_ids1 = assign_sam_masks_to_trackids(masks1, track_img1_resized)

        # Generate ground truth
        gt_matrix = generate_trackid_gt_for_sam(track_ids0, track_ids1)

        # Skip if no valid matches
        if gt_matrix.sum() == 0:
            continue

        # Compute metrics
        metrics = compute_metrics(scores, gt_matrix)

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

                # Overview visualization
                vis_path = vis_dir / f"pair_{pair_idx:04d}_vizenc.png"
                visualize_vkitti2_results(
                    img0, img1,
                    masks0, masks1,
                    track_ids0, track_ids1,
                    scores, gt_matrix, vis_path, pair_info
                )

                # GT vs Predicted visualization
                vis_path2 = vis_dir / f"pair_{pair_idx:04d}_gt_vs_pred.png"
                visualize_gt_vs_predicted_pairs(
                    img0, img1,
                    masks0, masks1,
                    track_ids0, track_ids1,
                    scores, gt_matrix, vis_path2, num_examples=5
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
    results_json_path = output_dir / f"metrics_vizenc_{encoder_type}.json"
    with open(results_json_path, 'w') as f:
        json.dump(aggregated_metrics, f, indent=2)
    print(f"\nSaved metrics to: {results_json_path}")

    table_txt_path = output_dir / f"table2_vizenc_{encoder_type}.txt"
    with open(table_txt_path, 'w') as f:
        f.write(table_str)
    print(f"Saved table to: {table_txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VizEnc on Virtual KITTI 2")
    parser.add_argument("--pairs", type=str, default="pairs_vkitti2.json")
    parser.add_argument("--data_root", type=str, default="/mnt/vol3/datasets/virtual-KITTI-2")
    parser.add_argument("--config", type=str, default="configs/config_eval_vkitti2.yaml")
    parser.add_argument("--encoder", type=str, default="dinov2", choices=["dinov2", "naradio"])
    parser.add_argument("--encoder_model", type=str, default=None,
                        help="Model name (e.g., facebook/dinov2-base or radio_v2.5-b)")
    parser.add_argument("--sam_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/vizenc_vkitti2")
    parser.add_argument("--num_pairs", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--num_vis", type=int, default=10)

    args = parser.parse_args()

    evaluate_vizenc_vkitti2(
        pairs_file=args.pairs,
        data_root=args.data_root,
        output_dir=args.output_dir,
        config_path=args.config,
        encoder_type=args.encoder,
        encoder_model=args.encoder_model,
        sam_checkpoint=args.sam_checkpoint,
        num_pairs=args.num_pairs,
        device=args.device,
        save_visualizations=args.visualize,
        num_vis_samples=args.num_vis
    )


if __name__ == "__main__":
    main()
