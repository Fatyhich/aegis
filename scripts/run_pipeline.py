#!/usr/bin/env python3
"""
VizEnc Pipeline Runner

Run the complete pipeline from command line.

Usage:
    python scripts/run_pipeline.py --config-dir configs/
    python scripts/run_pipeline.py --config-dir configs/ --project-dir /path/to/project
"""

import argparse
import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from vizenc_baseline.config import load_all_configs
from vizenc_baseline.pipeline import VizEncPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run VizEnc pipeline with topological graph construction"
    )

    parser.add_argument(
        '--config-dir',
        type=str,
        default='configs',
        help='Directory containing YAML config files (default: configs/)'
    )

    parser.add_argument(
        '--project-dir',
        type=str,
        default=None,
        help='Project directory (default: current directory)'
    )

    parser.add_argument(
        '--max-chunks',
        type=int,
        default=None,
        help='Maximum number of chunks to process (default: all frames in folder)'
    )

    # Override parameters (optional)
    parser.add_argument(
        '--frames-dir',
        type=str,
        default=None,
        help='Override frames directory from config'
    )

    parser.add_argument(
        '--sam-checkpoint',
        type=str,
        default=None,
        help='Override SAM checkpoint path from config'
    )

    parser.add_argument(
        '--dataset-name',
        type=str,
        default=None,
        help='Override dataset name from config'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory from config'
    )

    parser.add_argument(
        '--encoder',
        type=str,
        choices=['dinov2', 'naradio'],
        default=None,
        help='Override encoder type (dinov2 or naradio)'
    )

    parser.add_argument(
        '--start-frame',
        type=int,
        default=None,
        help='Override start frame index'
    )

    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Number of frames per chunk (overrides chunk_size in config)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'mps'],
        default=None,
        help='Device to use for models (cuda, cpu, mps)'
    )

    args = parser.parse_args()

    # Load configs
    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)

    print("Loading configurations...")
    pipeline_config, models_config, graph_config = load_all_configs(config_dir)

    # Apply command-line overrides
    if args.frames_dir:
        print(f"  ✓ Overriding frames_dir: {args.frames_dir}")
        pipeline_config.data.frames_dir = args.frames_dir

    if args.sam_checkpoint:
        print(f"  ✓ Overriding SAM checkpoint: {args.sam_checkpoint}")
        models_config.sam.checkpoint_path = args.sam_checkpoint

    if args.dataset_name:
        print(f"  ✓ Overriding dataset_name: {args.dataset_name}")
        pipeline_config.data.dataset_name = args.dataset_name

    if args.output_dir:
        print(f"  ✓ Overriding output_dir: {args.output_dir}")
        pipeline_config.output.output_dir = args.output_dir

    if args.encoder:
        print(f"  ✓ Overriding encoder: {args.encoder}")
        models_config.encoder.type = args.encoder

    if args.start_frame is not None:
        print(f"  ✓ Overriding start_frame: {args.start_frame}")
        pipeline_config.data.start_frame = args.start_frame

    if args.max_frames is not None:
        print(f"  ✓ Overriding chunk_size (frames per chunk): {args.max_frames}")
        pipeline_config.data.chunk_size = args.max_frames

    if args.max_chunks is not None:
        print(f"  ✓ Limiting to {args.max_chunks} chunks")
        pipeline_config.data.max_chunks = args.max_chunks

    # Store device in config if provided
    device = args.device if args.device else 'cuda'
    if args.device:
        print(f"  ✓ Overriding device: {args.device}")

    print("\nConfiguration Summary:")
    print(f"  Dataset: {pipeline_config.data.dataset_name}")
    print(f"  Frames: {pipeline_config.data.frames_dir}")
    print(f"  Start frame: {pipeline_config.data.start_frame}")
    print(f"  Frames per chunk: {pipeline_config.data.chunk_size}")
    max_chunks_str = str(pipeline_config.data.max_chunks) if pipeline_config.data.max_chunks else "all"
    print(f"  Max chunks: {max_chunks_str}")
    print(f"  SAM: {models_config.sam.version}")
    print(f"  SAM checkpoint: {models_config.sam.checkpoint_path if models_config.sam.checkpoint_path else 'auto-download'}")
    print(f"  Encoder: {models_config.encoder.type}")
    print(f"  Device: {device}")
    print(f"  Florence-2: {'Enabled' if models_config.florence.enabled else 'Disabled'}")
    print(f"  Zero-shot: {'Enabled' if models_config.zero_shot.enabled else 'Disabled'}")
    print(f"  Output: {pipeline_config.output.output_dir}")
    print(f"  Export formats: {', '.join(pipeline_config.output.export_formats)}")

    # Initialize pipeline
    project_dir = Path(args.project_dir) if args.project_dir else Path.cwd()
    pipeline = VizEncPipeline(
        pipeline_config,
        models_config,
        graph_config,
        project_dir
    )

    # Run pipeline
    print("\n" + "="*60)
    print("STARTING PIPELINE")
    print("="*60)

    try:
        results = pipeline.run()

        print("\n" + "="*60)
        print("SUCCESS")
        print("="*60)
        print(f"Processed {len(results)} chunks")
        print(f"\nOutput files:")
        for i, result in enumerate(results):
            print(f"\nChunk {i + 1}:")
            for fmt, filepath in result['files'].items():
                print(f"  {fmt}: {filepath}")

    except Exception as e:
        print("\n" + "="*60)
        print("ERROR")
        print("="*60)
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
