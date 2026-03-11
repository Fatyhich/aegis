#!/usr/bin/env python3
"""
Simple visualization script for chunk graphs.

Displays basic statistics and can export graph visualization.

For full interactive visualization, use the Gradio app (visualization/app.py).

Usage:
    python scripts/visualize_chunk.py chunk.pkl
    python scripts/visualize_chunk.py output/chunks/*.pkl
"""

import argparse
import sys
from pathlib import Path


from aegis.graph.exporters import PickleExporter, JSONExporter


def print_graph_info(graph):
    """Print detailed graph information."""
    stats = graph.get_statistics()

    print("\n" + "="*60)
    print(f"CHUNK: {graph.chunk_id}")
    print("="*60)

    print("\nStatistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Anchor nodes: {stats['anchor_nodes']}")
    print(f"  Non-anchor nodes: {stats['non_anchor_nodes']}")
    print(f"  Inner edges (spatial): {stats['inner_edges']}")
    print(f"  Inter edges (temporal): {stats['inter_edges']}")
    print(f"  Frames: {stats['frames']}")

    print("\nCategories:")
    for category, count in sorted(stats['categories'].items(), key=lambda x: -x[1]):
        print(f"  {category}: {count}")

    # Sample nodes
    print("\nSample Nodes:")
    for i, node in enumerate(graph.nodes[:5]):
        print(f"  {i+1}. {node}")

    # Sample edges
    if graph.inner_edges:
        print("\nSample Inner Edges (first 3):")
        for edge in graph.inner_edges[:3]:
            print(f"  {edge}")

    if graph.inter_edges:
        print("\nSample Inter Edges (first 3):")
        for edge in graph.inter_edges[:3]:
            print(f"  {edge}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize chunk graph statistics"
    )

    parser.add_argument(
        'chunk_files',
        type=str,
        nargs='+',
        help='Chunk file(s) to visualize'
    )

    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed node and edge information'
    )

    args = parser.parse_args()

    # Process each file
    for chunk_path in args.chunk_files:
        chunk_file = Path(chunk_path)

        if not chunk_file.exists():
            print(f"Warning: File not found: {chunk_file}")
            continue

        # Load graph
        if chunk_file.suffix == '.pkl':
            graph = PickleExporter.load_graph(chunk_file)
        elif chunk_file.suffix == '.json':
            graph = JSONExporter.load_graph(chunk_file)
        else:
            print(f"Warning: Unknown format: {chunk_file.suffix}")
            continue

        # Print info
        print_graph_info(graph)

        if args.detailed:
            print("\n" + "="*60)
            print("DETAILED INFORMATION")
            print("="*60)

            print("\nAll Anchor Nodes:")
            anchors = [n for n in graph.nodes if n.is_anchor]
            for node in anchors:
                print(f"  Track {node.track_id}: {node.category}")
                print(f"    Frame: {node.frame_idx}, Conf: {node.confidence:.3f}")
                print(f"    Desc: {node.description[:80]}...")

    print("\n" + "="*60)
    print(f"Processed {len(args.chunk_files)} files")
    print("="*60)


if __name__ == '__main__':
    main()
