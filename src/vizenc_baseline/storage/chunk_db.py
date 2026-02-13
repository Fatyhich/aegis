"""
Chunk graph storage manager.

Handles saving/loading chunk graphs in multiple formats.
"""

from pathlib import Path
from typing import Union, List, Dict
from datetime import datetime

from ..graph import ChunkGraph
from ..graph.exporters import Neo4jExporter, JSONExporter, PickleExporter
from .naming import generate_chunk_filename


class ChunkGraphStorage:
    """Manages storage of chunk graphs."""

    def __init__(self, output_dir: Union[str, Path], dataset_name: str):
        """
        Initialize storage manager.

        Args:
            output_dir: Directory for saving graphs
            dataset_name: Dataset name for filename generation
        """
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Exporters
        self.neo4j_exporter = Neo4jExporter()
        self.json_exporter = JSONExporter()
        self.pickle_exporter = PickleExporter()

    def save_graph(
        self,
        graph: ChunkGraph,
        start_frame: int,
        chunk_size: int,
        formats: List[str] = ['pkl', 'json', 'neo4j'],
        include_segmentation: bool = False,
        include_embedding: bool = True
    ) -> Dict[str, Path]:
        """
        Save graph in multiple formats.

        Args:
            graph: ChunkGraph to save
            start_frame: Starting frame index
            chunk_size: Chunk size
            formats: List of formats to export
            include_segmentation: Include masks in JSON
            include_embedding: Include embeddings in JSON

        Returns:
            Dictionary mapping format to filepath
        """
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")

        for fmt in formats:
            filename = generate_chunk_filename(
                self.dataset_name, start_frame, chunk_size, fmt, timestamp
            )
            filepath = self.output_dir / filename

            if fmt == 'pkl':
                self.pickle_exporter.save_graph(graph, filepath)
            elif fmt == 'json':
                self.json_exporter.save_graph(
                    graph, filepath,
                    include_segmentation=include_segmentation,
                    include_embedding=include_embedding
                )
            elif fmt == 'neo4j':
                cypher = self.neo4j_exporter.export_graph(graph)
                with open(filepath, 'w') as f:
                    f.write(cypher)
            else:
                print(f"Warning: Unknown format '{fmt}', skipping")
                continue

            saved_files[fmt] = filepath

        return saved_files

    def load_graph(
        self,
        filepath: Union[str, Path],
        format: str = None
    ) -> ChunkGraph:
        """
        Load graph from file.

        Args:
            filepath: Path to graph file
            format: Format override (auto-detected from extension if None)

        Returns:
            ChunkGraph instance
        """
        filepath = Path(filepath)

        if format is None:
            # Auto-detect from extension
            ext = filepath.suffix.lower()
            if ext == '.pkl':
                format = 'pkl'
            elif ext == '.json':
                format = 'json'
            else:
                raise ValueError(f"Cannot auto-detect format from extension: {ext}")

        if format == 'pkl':
            return self.pickle_exporter.load_graph(filepath)
        elif format == 'json':
            return self.json_exporter.load_graph(filepath)
        else:
            raise ValueError(f"Cannot load from format: {format}")

    def list_chunks(self, format: str = 'pkl') -> List[Path]:
        """
        List all chunk files in output directory.

        Args:
            format: File format to list

        Returns:
            List of file paths
        """
        ext_map = {'pkl': '*.pkl', 'json': '*.json', 'neo4j': '*.cypher'}
        pattern = ext_map.get(format, '*.pkl')

        return sorted(self.output_dir.glob(pattern))
