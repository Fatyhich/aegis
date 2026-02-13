"""
JSON exporter for chunk graphs.
"""

import json
import numpy as np
from pathlib import Path
from typing import Union
from ..graph_builder import ChunkGraph


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class JSONExporter:
    """Export ChunkGraph to JSON format."""

    def export_graph(
        self,
        graph: ChunkGraph,
        include_segmentation: bool = False,
        include_embedding: bool = True,
        indent: int = 2
    ) -> str:
        """
        Export graph to JSON string.

        Args:
            graph: ChunkGraph instance
            include_segmentation: Include binary masks (very large!)
            include_embedding: Include embedding vectors
            indent: JSON indentation

        Returns:
            JSON string
        """
        data = graph.to_dict(include_segmentation, include_embedding)
        return json.dumps(data, indent=indent, cls=NumpyEncoder)

    def save_graph(
        self,
        graph: ChunkGraph,
        filepath: Union[str, Path],
        include_segmentation: bool = False,
        include_embedding: bool = True,
        indent: int = 2
    ):
        """
        Save graph to JSON file.

        Args:
            graph: ChunkGraph instance
            filepath: Output file path
            include_segmentation: Include binary masks
            include_embedding: Include embeddings
            indent: JSON indentation
        """
        json_str = self.export_graph(
            graph, include_segmentation, include_embedding, indent
        )

        with open(filepath, 'w') as f:
            f.write(json_str)

    @staticmethod
    def load_graph(filepath: Union[str, Path]) -> ChunkGraph:
        """
        Load graph from JSON file.

        Args:
            filepath: Input file path

        Returns:
            ChunkGraph instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        return ChunkGraph.from_dict(data)
