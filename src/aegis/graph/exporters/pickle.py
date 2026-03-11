"""
Pickle exporter for chunk graphs.

Fast binary serialization for Python.
"""

import pickle
from pathlib import Path
from typing import Union
from ..graph_builder import ChunkGraph


class PickleExporter:
    """Export ChunkGraph to pickle format."""

    @staticmethod
    def save_graph(graph: ChunkGraph, filepath: Union[str, Path]):
        """
        Save graph to pickle file.

        Args:
            graph: ChunkGraph instance
            filepath: Output file path
        """
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_graph(filepath: Union[str, Path]) -> ChunkGraph:
        """
        Load graph from pickle file.

        Args:
            filepath: Input file path

        Returns:
            ChunkGraph instance
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
