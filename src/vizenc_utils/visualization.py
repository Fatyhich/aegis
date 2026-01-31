"""
Visualization utilities for object segmentation and tracking.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def show_anns(anns, borders=True):
    """
    Display segmentation masks with random colors.

    Args:
        anns: List of annotation dictionaries with 'segmentation' and 'area' keys
        borders: Whether to draw borders around masks
    """
    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                   sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0

    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask

        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8),
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)
            # Smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                       for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


def visualize_masks_with_descriptions(image, masks, figsize=(25, 20)):
    """
    Visualize masks with numbered overlays and a separate description panel.

    Args:
        image: PIL Image
        masks: List of mask dictionaries with 'bbox' and 'description' keys
        figsize: Figure size tuple
    """
    fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=figsize,
                                          gridspec_kw={'width_ratios': [3, 1]})

    # Image with masks and numbers
    ax_img.imshow(image)
    plt.sca(ax_img)
    show_anns(masks)

    # Add numbers at mask centers
    for idx, mask_data in enumerate(masks):
        x, y, w, h = mask_data['bbox']
        center_x = x + w / 2
        center_y = y + h / 2

        ax_img.text(center_x, center_y, str(idx),
                   color='white', fontsize=14, weight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='black', alpha=0.6))

    ax_img.axis('off')

    # Description list
    ax_text.axis('off')
    descriptions = "\n\n".join([f"{i}: {mask['description']}"
                               for i, mask in enumerate(masks)])
    ax_text.text(0.05, 0.50, descriptions,
                fontsize=11, va='center', ha='left',
                family='monospace', wrap=True)

    plt.tight_layout()
    plt.show()


def visualize_object_tracking_with_metrics(image0, masks0, image1, masks1,
                                          matches, metrics, title_suffix=""):
    """
    Visualize object tracking between two frames with bounding boxes and metrics.

    Args:
        image0: PIL Image for frame 0
        masks0: List of masks for frame 0
        image1: PIL Image for frame 1
        masks1: List of masks for frame 1
        matches: List of (idx0, idx1, similarity) tuples
        metrics: Dictionary with tracking metrics
        title_suffix: Additional text for title
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10))

    # Titles with metrics
    title0 = f'Frame 0\nObjects: {len(masks0)}'
    title1 = f'Frame 1 {title_suffix}\nObjects: {len(masks1)}, Matches: {metrics["n_matches"]}\n'
    title1 += f'Confidence: {metrics["confidence_score"]:.3f}, Similarity: {metrics["avg_similarity"]:.3f}'

    ax0.imshow(image0)
    ax0.set_title(title0, fontsize=14)
    ax0.axis('off')

    ax1.imshow(image1)
    ax1.set_title(title1, fontsize=14)
    ax1.axis('off')

    # Draw matches with colored bounding boxes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(matches)))

    for (i, j, sim), color in zip(matches, colors):
        # Bbox centers
        x0, y0, w0, h0 = masks0[i]['bbox']
        x1, y1, w1, h1 = masks1[j]['bbox']

        center0 = (x0 + w0/2, y0 + h0/2)
        center1 = (x1 + w1/2, y1 + h1/2)

        # Draw bboxes on both frames
        rect0 = mpatches.Rectangle((x0, y0), w0, h0,
                                   linewidth=3, edgecolor=color, facecolor='none')
        rect1 = mpatches.Rectangle((x1, y1), w1, h1,
                                   linewidth=3, edgecolor=color, facecolor='none')
        ax0.add_patch(rect0)
        ax1.add_patch(rect1)

        # Labels with numbers and similarity
        ax0.text(center0[0], center0[1], str(i),
                color='white', fontsize=14, weight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))

        ax1.text(center1[0], center1[1], f"{j}\n{sim:.2f}",
                color='white', fontsize=12, weight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))

    plt.tight_layout()
    plt.show()


def visualize_anchor(anchor_db, track_id, mask_db, frames_paths):
    """
    Visualize all observations of a specific anchor.

    Args:
        anchor_db: Anchor database dict
        track_id: ID of anchor to visualize
        mask_db: List of (frame_name, masks) tuples
        frames_paths: List of Path objects for all frames
    """
    if track_id not in anchor_db['anchors']:
        print(f"Track ID {track_id} not found in anchor database!")
        return

    anchor = anchor_db['anchors'][track_id]

    # Print anchor metadata
    print(f"{'='*80}")
    print(f"ANCHOR {track_id} - {anchor['category'].upper()}")
    print(f"{'='*80}")
    print(f"Observations: {anchor['n_observations']}")
    print(f"Frames: {anchor['frames']}")
    print(f"First seen: frame {anchor['first_seen']}")
    print(f"Last seen: frame {anchor['last_seen']}")
    print(f"Average confidence: {anchor['avg_confidence']:.3f}")
    print(f"\nDescriptions:")
    for i, (frame_idx, desc) in enumerate(zip(anchor['frames'], anchor['descriptions'])):
        print(f"  Frame {frame_idx}: \"{desc}\"")
    print(f"{'='*80}\n")

    n_obs = anchor['n_observations']

    # Create figure with grid: 2 rows per observation (mask + crop)
    # Columns = max observations (up to 6 for reasonable display)
    max_cols = min(6, n_obs)
    n_rows = 2 * ((n_obs + max_cols - 1) // max_cols)  # 2 rows per column

    fig = plt.figure(figsize=(4 * max_cols, 6 * (n_rows // 2)))

    # Process each observation
    for obs_idx, (frame_idx, bbox, conf) in enumerate(zip(
        anchor['frames'], anchor['bboxes'], anchor['confidences']
    )):
        # Find the frame in frames_paths
        if frame_idx >= len(frames_paths):
            print(f"Warning: frame_idx {frame_idx} out of range (max {len(frames_paths)-1})")
            continue

        frame_path = frames_paths[frame_idx]
        frame_name = frame_path.name

        # Load image
        from PIL import Image
        image = Image.open(frame_path).convert("RGB")
        img_array = np.array(image)

        # Find the mask with this track_id in mask_db
        mask_data = None
        for db_frame_name, masks in mask_db:
            if db_frame_name == frame_name:
                # Find mask with matching track_id
                for m in masks:
                    if m.get('track_id') == track_id:
                        mask_data = m
                        break
                break

        if mask_data is None:
            print(f"Warning: Could not find mask for track {track_id} in frame {frame_name}")
            continue

        # Extract mask and bbox
        x, y, w, h = bbox
        mask_seg = mask_data['segmentation']

        # Column index for grid
        col_idx = obs_idx % max_cols
        row_base = (obs_idx // max_cols) * 2

        # === Top row: Full frame with mask overlay ===
        ax_full = plt.subplot(n_rows, max_cols, row_base * max_cols + col_idx + 1)
        ax_full.imshow(img_array)

        # Overlay mask with transparency
        mask_overlay = np.zeros((*mask_seg.shape, 4))
        mask_overlay[mask_seg] = [1, 0, 0, 0.5]  # Red with 50% alpha
        ax_full.imshow(mask_overlay)

        # Draw bounding box
        rect = mpatches.Rectangle((x, y), w, h,
                                  linewidth=3, edgecolor='yellow', facecolor='none')
        ax_full.add_patch(rect)

        ax_full.set_title(f"Frame {frame_idx}\n(obs {obs_idx+1}/{n_obs}, conf: {conf:.3f})",
                         fontsize=12, weight='bold')
        ax_full.axis('off')

        # === Bottom row: Cropped region ===
        ax_crop = plt.subplot(n_rows, max_cols, (row_base + 1) * max_cols + col_idx + 1)

        # Extract crop
        x_int, y_int = int(x), int(y)
        w_int, h_int = int(w), int(h)
        crop = img_array[y_int:y_int+h_int, x_int:x_int+w_int]

        if crop.size > 0:
            ax_crop.imshow(crop)
            ax_crop.set_title(f"Crop {w_int}x{h_int}", fontsize=10)
        else:
            ax_crop.text(0.5, 0.5, "Invalid crop", ha='center', va='center')
        ax_crop.axis('off')

    plt.tight_layout()
    plt.show()
