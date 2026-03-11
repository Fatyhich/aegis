#!/usr/bin/env python3
"""
Gradio visualization app for chunk graphs.

Simple web interface for exploring chunk graphs.

Usage:
    python visualization/app.py --chunks-dir output/chunks
"""

import sys
from pathlib import Path
import argparse


try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    print("Error: Gradio not installed. Install with: pip install gradio")
    GRADIO_AVAILABLE = False
    sys.exit(1)

from aegis.graph.exporters import PickleExporter, JSONExporter
from PIL import Image, ImageDraw
from collections import defaultdict
import numpy as np
import colorsys


def generate_colors(n):
    """Generate N visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(tuple(int(x * 255) for x in rgb))
    return colors


def draw_masks_on_frame(image, nodes, show_track_id=True, show_category=True):
    """Draw masks and annotations on frame."""
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, 'RGBA')

    # Generate colors for track IDs
    track_ids = list(set(n.track_id for n in nodes if n.track_id is not None))
    track_colors = {tid: color for tid, color in zip(track_ids, generate_colors(len(track_ids)))}
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

        # Draw mask overlay if available
        if node.segmentation is not None:
            try:
                mask_array = node.segmentation.astype(np.uint8) * 255
                mask_img = Image.fromarray(mask_array, mode='L')
                colored = Image.new('RGBA', image.size, color + (80,))
                overlay.paste(colored, (0, 0), mask_img)
            except:
                pass  # Skip if segmentation not available

        # Draw label
        labels = []
        if show_track_id and node.track_id is not None:
            labels.append(f"T{node.track_id}")
        if show_category:
            labels.append(node.category)

        if labels:
            label_text = " ".join(labels)
            text_bbox = draw.textbbox((x, y-15), label_text)
            draw.rectangle(text_bbox, fill=color + (200,))
            draw.text((x, y-15), label_text, fill=(255, 255, 255, 255))

    return overlay


class ChunkGraphViewer:
    """Simple Gradio app for viewing chunk graphs."""

    def __init__(self, chunks_dir: Path, frames_dir: Path = None):
        self.chunks_dir = Path(chunks_dir)
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.current_graph = None

    def list_chunks(self):
        """List available chunk files."""
        pkl_files = sorted(self.chunks_dir.glob("*.pkl"))
        return [str(f.name) for f in pkl_files]

    def load_chunk(self, chunk_name):
        """Load chunk graph from file."""
        if not chunk_name:
            return "No chunk selected"

        chunk_path = self.chunks_dir / chunk_name

        try:
            self.current_graph = PickleExporter.load_graph(chunk_path)
            stats = self.current_graph.get_statistics()

            # Format statistics
            info = f"""# Chunk: {self.current_graph.chunk_id}

## Statistics
- **Total Nodes**: {stats['total_nodes']}
- **Anchor Nodes**: {stats['anchor_nodes']}
- **Non-anchor Nodes**: {stats['non_anchor_nodes']}
- **Inner Edges** (spatial): {stats['inner_edges']}
- **Inter Edges** (temporal): {stats['inter_edges']}
- **Frames**: {stats['frames']}

## Categories
"""
            for cat, count in sorted(stats['categories'].items(), key=lambda x: -x[1]):
                info += f"- **{cat}**: {count}\n"

            return info

        except Exception as e:
            return f"Error loading chunk: {e}"

    def show_nodes(self):
        """Show node list."""
        if not self.current_graph:
            return "No graph loaded"

        nodes_info = "# Nodes\n\n"
        anchors = [n for n in self.current_graph.nodes if n.is_anchor]
        non_anchors = [n for n in self.current_graph.nodes if not n.is_anchor]

        nodes_info += f"## Anchor Nodes ({len(anchors)})\n\n"
        for node in anchors[:20]:  # Show first 20
            nodes_info += f"- **{node.node_id}** (Track {node.track_id})\n"
            nodes_info += f"  - Category: {node.category}\n"
            nodes_info += f"  - Frame: {node.frame_idx}, Conf: {node.confidence:.3f}\n"
            nodes_info += f"  - Desc: {node.description[:100]}...\n\n"

        if len(anchors) > 20:
            nodes_info += f"... and {len(anchors) - 20} more\n\n"

        nodes_info += f"## Non-anchor Nodes ({len(non_anchors)})\n\n"
        for node in non_anchors[:10]:  # Show first 10
            nodes_info += f"- **{node.node_id}**\n"
            nodes_info += f"  - Category: {node.category}, Frame: {node.frame_idx}\n\n"

        if len(non_anchors) > 10:
            nodes_info += f"... and {len(non_anchors) - 10} more\n"

        return nodes_info

    def show_edges(self, edge_type="inner"):
        """Show edge list."""
        if not self.current_graph:
            return "No graph loaded"

        edges = self.current_graph.inner_edges if edge_type == "inner" else self.current_graph.inter_edges
        edge_name = "Inner Edges (Spatial)" if edge_type == "inner" else "Inter Edges (Temporal)"

        edges_info = f"# {edge_name}\n\n"
        edges_info += f"Total: {len(edges)}\n\n"

        for i, edge in enumerate(edges[:30]):  # Show first 30
            edges_info += f"{i+1}. {edge}\n"

        if len(edges) > 30:
            edges_info += f"\n... and {len(edges) - 30} more\n"

        return edges_info

    def visualize_frame(self, frame_idx):
        """Visualize specific frame with masks."""
        if not self.current_graph:
            return None

        if not self.frames_dir or not self.frames_dir.exists():
            return None

        # Get nodes for this frame
        nodes = [n for n in self.current_graph.nodes if n.frame_idx == frame_idx]

        if not nodes:
            return None

        # Find frame file
        frame_files = list(self.frames_dir.glob(f"*{frame_idx:06d}*")) + \
                     list(self.frames_dir.glob(f"frame_{frame_idx:06d}*")) + \
                     list(self.frames_dir.glob(f"*{frame_idx:04d}*"))

        if not frame_files:
            return None

        # Load image
        image = Image.open(frame_files[0]).convert('RGB')

        # Draw masks
        result = draw_masks_on_frame(image, nodes)

        return result

    def get_frame_indices(self):
        """Get list of frame indices in current graph."""
        if not self.current_graph:
            return []

        frame_indices = sorted(set(n.frame_idx for n in self.current_graph.nodes))
        return frame_indices

    def create_interface(self):
        """Create Gradio interface."""
        with gr.Blocks(title="VizEnc Chunk Graph Viewer") as demo:
            gr.Markdown("# VizEnc Topological Graph Viewer")
            gr.Markdown(f"Chunks directory: `{self.chunks_dir}`")

            with gr.Row():
                chunk_dropdown = gr.Dropdown(
                    choices=self.list_chunks(),
                    label="Select Chunk",
                    interactive=True
                )
                load_btn = gr.Button("Load")

            with gr.Tabs():
                with gr.Tab("Statistics"):
                    stats_output = gr.Markdown()

                with gr.Tab("Nodes"):
                    nodes_output = gr.Markdown()
                    show_nodes_btn = gr.Button("Show Nodes")

                with gr.Tab("Inner Edges"):
                    inner_edges_output = gr.Markdown()
                    show_inner_btn = gr.Button("Show Inner Edges")

                with gr.Tab("Inter Edges"):
                    inter_edges_output = gr.Markdown()
                    show_inter_btn = gr.Button("Show Inter Edges")

                with gr.Tab("Frame Viewer"):
                    if self.frames_dir and self.frames_dir.exists():
                        gr.Markdown(f"Frames directory: `{self.frames_dir}`")
                        frame_slider = gr.Slider(
                            minimum=0,
                            maximum=100,
                            step=1,
                            label="Frame Index",
                            interactive=True
                        )
                        frame_image = gr.Image(label="Frame with Masks", type="pil")
                        visualize_btn = gr.Button("Visualize Frame")
                    else:
                        gr.Markdown("⚠️ No frames directory specified. Use `--frames-dir` to enable frame visualization.")

            # Event handlers
            load_btn.click(
                fn=self.load_chunk,
                inputs=[chunk_dropdown],
                outputs=[stats_output]
            )

            show_nodes_btn.click(
                fn=self.show_nodes,
                outputs=[nodes_output]
            )

            show_inner_btn.click(
                fn=lambda: self.show_edges("inner"),
                outputs=[inner_edges_output]
            )

            show_inter_btn.click(
                fn=lambda: self.show_edges("inter"),
                outputs=[inter_edges_output]
            )

            # Frame viewer handlers
            if self.frames_dir and self.frames_dir.exists():
                def update_slider():
                    """Update slider range when graph is loaded."""
                    indices = self.get_frame_indices()
                    if indices:
                        return gr.Slider(minimum=min(indices), maximum=max(indices), value=min(indices))
                    return gr.Slider()

                load_btn.click(
                    fn=update_slider,
                    outputs=[frame_slider]
                )

                visualize_btn.click(
                    fn=self.visualize_frame,
                    inputs=[frame_slider],
                    outputs=[frame_image]
                )

        return demo


def main():
    parser = argparse.ArgumentParser(description="Launch Gradio visualization app")
    parser.add_argument(
        '--chunks-dir',
        type=str,
        default='output/chunks',
        help='Directory containing chunk files'
    )
    parser.add_argument(
        '--frames-dir',
        type=str,
        default=None,
        help='Directory containing original frames (for visualization)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port to run server on'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create public share link'
    )

    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    if not chunks_dir.exists():
        print(f"Error: Chunks directory not found: {chunks_dir}")
        sys.exit(1)

    frames_dir = Path(args.frames_dir) if args.frames_dir else None

    viewer = ChunkGraphViewer(chunks_dir, frames_dir)
    demo = viewer.create_interface()

    print(f"\nLaunching Gradio app...")
    print(f"Chunks directory: {chunks_dir}")
    if frames_dir:
        print(f"Frames directory: {frames_dir}")
    demo.launch(
        server_port=args.port,
        share=args.share
    )


if __name__ == '__main__':
    main()
