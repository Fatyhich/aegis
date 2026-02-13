#!/usr/bin/env python3
"""
Export chunk graph to different formats.

Usage:
    python scripts/export_graph.py chunk.pkl --format neo4j --output graph.cypher
    python scripts/export_graph.py chunk.pkl --format json --output graph.json
"""

import argparse
import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from vizenc_baseline.graph.exporters import Neo4jExporter, JSONExporter, PickleExporter


def main():
    parser = argparse.ArgumentParser(
        description="Export chunk graph to different formats"
    )

    parser.add_argument(
        'chunk_file',
        type=str,
        help='Input chunk file (pickle or json)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['neo4j', 'json', 'pkl'],
        required=True,
        help='Output format'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file path'
    )

    parser.add_argument(
        '--include-segmentation',
        action='store_true',
        help='Include segmentation masks (JSON only, large files!)'
    )

    parser.add_argument(
        '--no-embedding',
        action='store_true',
        help='Exclude embeddings (JSON only)'
    )

    args = parser.parse_args()

    # Load chunk
    chunk_file = Path(args.chunk_file)
    if not chunk_file.exists():
        print(f"Error: File not found: {chunk_file}")
        sys.exit(1)

    print(f"Loading chunk from: {chunk_file}")

    # Auto-detect input format
    if chunk_file.suffix == '.pkl':
        graph = PickleExporter.load_graph(chunk_file)
    elif chunk_file.suffix == '.json':
        graph = JSONExporter.load_graph(chunk_file)
    else:
        print(f"Error: Unknown input format: {chunk_file.suffix}")
        sys.exit(1)

    print(f"Loaded graph: {graph}")

    # Export
    output_file = Path(args.output)
    print(f"Exporting to: {output_file}")

    if args.format == 'neo4j':
        exporter = Neo4jExporter()
        cypher = exporter.export_graph(graph)
        with open(output_file, 'w') as f:
            f.write(cypher)

    elif args.format == 'json':
        exporter = JSONExporter()
        exporter.save_graph(
            graph,
            output_file,
            include_segmentation=args.include_segmentation,
            include_embedding=not args.no_embedding
        )

    elif args.format == 'pkl':
        exporter = PickleExporter()
        exporter.save_graph(graph, output_file)

    print(f"Export complete!")
    print(f"Output: {output_file}")


if __name__ == '__main__':
    main()
