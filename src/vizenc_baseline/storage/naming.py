"""
Naming conventions for chunk graph files.

Format: {dataset_name}_{start_frame:04d}_chunk{size}_{YYYYMMDD-HHMM}.{ext}
Example: egowalk_0004_chunk8_20260204-1430.pkl
"""

from datetime import datetime
from pathlib import Path
from typing import Literal


def generate_chunk_filename(
    dataset_name: str,
    start_frame: int,
    chunk_size: int,
    format: Literal['pkl', 'json', 'neo4j'] = 'pkl',
    timestamp: str = None
) -> str:
    """
    Generate filename for chunk graph.

    Args:
        dataset_name: Name of dataset (e.g., 'egowalk')
        start_frame: Starting frame index
        chunk_size: Number of frames in chunk
        format: Output format
        timestamp: Optional timestamp (default: current time)

    Returns:
        Filename string
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")

    # Extension mapping
    ext_map = {
        'pkl': 'pkl',
        'json': 'json',
        'neo4j': 'cypher'
    }
    ext = ext_map.get(format, 'pkl')

    return f"{dataset_name}_{start_frame:04d}_chunk{chunk_size}_{timestamp}.{ext}"


def parse_chunk_filename(filename: str) -> dict:
    """
    Parse chunk filename to extract metadata.

    Args:
        filename: Chunk filename

    Returns:
        Dictionary with parsed fields
    """
    stem = Path(filename).stem  # Remove extension
    parts = stem.split('_')

    if len(parts) < 4:
        raise ValueError(f"Invalid chunk filename: {filename}")

    # Extract components
    dataset_name = parts[0]
    start_frame = int(parts[1])

    # Extract chunk size (format: "chunk8")
    chunk_part = parts[2]
    if not chunk_part.startswith('chunk'):
        raise ValueError(f"Invalid chunk part: {chunk_part}")
    chunk_size = int(chunk_part[5:])

    # Timestamp
    timestamp = parts[3] if len(parts) > 3 else None

    return {
        'dataset_name': dataset_name,
        'start_frame': start_frame,
        'chunk_size': chunk_size,
        'timestamp': timestamp
    }
