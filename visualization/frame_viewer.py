#!/usr/bin/env python3
"""
Frame viewer with mask overlay for chunk graphs.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def generate_colors(n):
    """Generate N visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(tuple(int(x * 255) for x in rgb))
    return colors


def draw_masks_on_frame(image, nodes, show_track_id=True, show_category=True):
    """
    Draw masks and annotations on frame.

    Args:
        image: PIL Image
        nodes: List of MaskNode instances
        show_track_id: Show track ID labels
        show_category: Show category labels

    Returns:
        PIL Image with overlays
    """
    # Create overlay
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, 'RGBA')

    # Generate colors for track IDs
    track_ids = list(set(n.track_id for n in nodes if n.track_id is not None))
    track_colors = {tid: color for tid, color in zip(track_ids, generate_colors(len(track_ids)))}

    # Default color for non-tracked objects
    default_color = (128, 128, 128)

    # Draw each mask
    for node in nodes:
        x, y, w, h = node.bbox

        # Get color
        if node.track_id is not None and node.track_id in track_colors:
            color = track_colors[node.track_id]
        else:
            color = default_color

        # Draw bbox
        thickness = 3 if node.is_anchor else 1
        for t in range(thickness):
            draw.rectangle(
                [x+t, y+t, x+w-t, y+h-t],
                outline=color + (200,),
                width=1
            )

        # Draw mask overlay
        if node.segmentation is not None:
            mask_array = node.segmentation.astype(np.uint8) * 255
            mask_img = Image.fromarray(mask_array, mode='L')

            # Create colored overlay
            colored = Image.new('RGBA', image.size, color + (80,))
            overlay.paste(colored, (0, 0), mask_img)

        # Draw label
        labels = []
        if show_track_id and node.track_id is not None:
            labels.append(f"T{node.track_id}")
        if show_category:
            labels.append(node.category)

        if labels:
            label_text = " ".join(labels)
            # Simple text (no font loading to avoid dependencies)
            text_bbox = draw.textbbox((x, y-15), label_text)
            draw.rectangle(text_bbox, fill=color + (200,))
            draw.text((x, y-15), label_text, fill=(255, 255, 255, 255))

    return overlay


def visualize_chunk_frames(graph, frames_dir, output_dir=None):
    """
    Visualize all frames in a chunk with mask overlays.

    Args:
        graph: ChunkGraph instance
        frames_dir: Path to frames directory
        output_dir: Optional output directory for saving images

    Returns:
        List of PIL Images
    """
    frames_dir = Path(frames_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    images = []

    # Group nodes by frame
    from collections import defaultdict
    nodes_by_frame = defaultdict(list)
    for node in graph.nodes:
        nodes_by_frame[node.frame_idx].append(node)

    # Process each frame
    for frame_idx in sorted(nodes_by_frame.keys()):
        # Find frame file (try common patterns)
        frame_files = list(frames_dir.glob(f"*{frame_idx:06d}*")) + \
                     list(frames_dir.glob(f"frame_{frame_idx:06d}*")) + \
                     list(frames_dir.glob(f"*{frame_idx:04d}*"))

        if not frame_files:
            print(f"Warning: Frame {frame_idx} not found in {frames_dir}")
            continue

        frame_path = frame_files[0]

        # Load image
        image = Image.open(frame_path).convert('RGB')

        # Draw masks
        nodes = nodes_by_frame[frame_idx]
        result = draw_masks_on_frame(image, nodes)

        images.append(result)

        # Save if output_dir specified
        if output_dir:
            output_path = output_dir / f"frame_{frame_idx:04d}_annotated.jpg"
            result.save(output_path, quality=95)
            print(f"Saved: {output_path}")

    return images
