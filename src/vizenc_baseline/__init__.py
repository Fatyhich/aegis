"""
VizEnc Baseline - Modular Pipeline for Visual Encoding with Topological Graphs.

Refactored from all-in-one.ipynb into a production-ready Python baseline.

Key Features:
- Mask-based cropping with transparent background (RGBA)
- Topological graph with all masks as nodes
- Inner edges (spatial) and inter edges (temporal)
- Multi-format export (Pickle, JSON, Neo4j Cypher)
- YAML-based configuration
"""

__version__ = "1.0.0"

from .config import (
    load_pipeline_config,
    load_models_config,
    load_graph_config,
    load_all_configs,
    PipelineConfig,
    ModelsConfig,
    GraphConfig,
)

from .preprocessing import MaskCropper

from .graph import (
    MaskNode,
    InnerEdge,
    InterEdge,
    ChunkGraph,
    Neo4jExporter,
    JSONExporter,
    PickleExporter,
)

from .storage import (
    ChunkGraphStorage,
    generate_chunk_filename,
    parse_chunk_filename,
)

from .pipeline import (
    ChunkProcessor,
    VizEncPipeline,
)

__all__ = [
    # Config
    'load_pipeline_config',
    'load_models_config',
    'load_graph_config',
    'load_all_configs',
    'PipelineConfig',
    'ModelsConfig',
    'GraphConfig',
    # Preprocessing
    'MaskCropper',
    # Graph
    'MaskNode',
    'InnerEdge',
    'InterEdge',
    'ChunkGraph',
    'Neo4jExporter',
    'JSONExporter',
    'PickleExporter',
    # Storage
    'ChunkGraphStorage',
    'generate_chunk_filename',
    'parse_chunk_filename',
    # Pipeline
    'ChunkProcessor',
    'VizEncPipeline',
]
