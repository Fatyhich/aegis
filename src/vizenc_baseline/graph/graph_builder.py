"""
Topological graph builder for chunks.

Creates a graph where:
- Every detected mask is a node
- Inner edges connect masks within a frame (spatial)
- Inter edges connect masks across frames (temporal)
"""

from typing import List, Optional, Dict, Any
from .node import MaskNode
from .edge import InnerEdge, InterEdge
from .spatial import SpatialRelationBuilder
from .temporal import TemporalEdgeBuilder


class ChunkGraph:
    """
    Topological graph for a chunk of frames.

    Contains:
    - All masks as nodes (not just anchors)
    - Inner edges for spatial relations
    - Inter edges for temporal tracking
    """

    def __init__(self, chunk_id: str, config: dict):
        """
        Initialize chunk graph.

        Args:
            chunk_id: Unique identifier for this chunk
            config: Graph configuration dict
        """
        self.chunk_id = chunk_id
        self.config = config

        # Graph components
        self.nodes: List[MaskNode] = []
        self.inner_edges: List[InnerEdge] = []
        self.inter_edges: List[InterEdge] = []

        # Builders
        self.spatial_builder = SpatialRelationBuilder(config.get('inner', {}))
        self.temporal_builder = TemporalEdgeBuilder()

        # Statistics
        self._statistics: Optional[Dict[str, Any]] = None

    def add_frame_masks(
        self,
        frame_idx: int,
        masks: List[dict],
        anchor_db: Optional[dict] = None
    ) -> List[MaskNode]:
        """
        Add all masks from a frame as nodes and build inner edges.

        Args:
            frame_idx: Frame index
            masks: List of mask dictionaries from SAM processing
            anchor_db: Optional anchor database to check anchor status

        Returns:
            List of created MaskNode instances
        """
        frame_nodes = []

        for mask_idx, mask_data in enumerate(masks):
            # Generate unique node ID
            node_id = f"{self.chunk_id}_f{frame_idx:04d}_m{mask_idx:03d}"

            # Check if this is an anchor (tracked across multiple frames)
            track_id = mask_data.get('track_id')
            is_anchor = False
            if track_id is not None and anchor_db is not None:
                anchor = anchor_db.get('anchors', {}).get(track_id)
                # Anchor = object tracked in MORE than 1 frame
                if anchor and anchor.get('n_observations', 0) > 1:
                    is_anchor = True

            # Create node
            node = MaskNode(
                node_id=node_id,
                frame_idx=frame_idx,
                mask_idx=mask_idx,
                chunk_id=self.chunk_id,
                is_anchor=is_anchor,
                track_id=track_id,
                bbox=mask_data['bbox'],
                segmentation=mask_data.get('segmentation'),
                embedding=mask_data['embedding'],
                category=mask_data.get('category', 'unknown'),
                description=mask_data.get('description', ''),
                confidence=mask_data.get('predicted_iou', mask_data.get('stability_score', 0.0))
            )

            self.nodes.append(node)
            frame_nodes.append(node)

        # Build inner edges for this frame
        if len(frame_nodes) >= 2:
            inner_edges = self.spatial_builder.build_frame_relations(
                frame_nodes, frame_idx
            )
            self.inner_edges.extend(inner_edges)

        # Invalidate cached statistics
        self._statistics = None

        return frame_nodes

    def build_inter_edges(
        self,
        matches: List[tuple],
        prev_frame_idx: int,
        curr_frame_idx: int
    ):
        """
        Build inter edges between two consecutive frames.

        Args:
            matches: List of (prev_idx, curr_idx, similarity) tuples
            prev_frame_idx: Previous frame index
            curr_frame_idx: Current frame index
        """
        # Get nodes for both frames
        prev_nodes = [n for n in self.nodes if n.frame_idx == prev_frame_idx]
        curr_nodes = [n for n in self.nodes if n.frame_idx == curr_frame_idx]

        # Build inter edges
        inter_edges = self.temporal_builder.build_tracking_edges(
            prev_nodes, curr_nodes, matches, prev_frame_idx, curr_frame_idx
        )
        self.inter_edges.extend(inter_edges)

        # Invalidate cached statistics
        self._statistics = None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute graph statistics.

        Returns:
            Dictionary with statistics
        """
        if self._statistics is None:
            anchor_nodes = [n for n in self.nodes if n.is_anchor]

            self._statistics = {
                'chunk_id': self.chunk_id,
                'total_nodes': len(self.nodes),
                'anchor_nodes': len(anchor_nodes),
                'non_anchor_nodes': len(self.nodes) - len(anchor_nodes),
                'inner_edges': len(self.inner_edges),
                'inter_edges': len(self.inter_edges),
                'frames': len(set(n.frame_idx for n in self.nodes)),
                'categories': self._count_categories()
            }

        return self._statistics

    def _count_categories(self) -> Dict[str, int]:
        """Count nodes by category."""
        counts = {}
        for node in self.nodes:
            counts[node.category] = counts.get(node.category, 0) + 1
        return counts

    def to_dict(
        self,
        include_segmentation: bool = False,
        include_embedding: bool = True
    ) -> dict:
        """
        Convert graph to dictionary for serialization.

        Args:
            include_segmentation: Include binary masks (large!)
            include_embedding: Include embedding vectors

        Returns:
            Dictionary representation
        """
        return {
            'chunk_id': self.chunk_id,
            'config': self.config,
            'nodes': [n.to_dict(include_segmentation, include_embedding)
                     for n in self.nodes],
            'inner_edges': [e.to_dict() for e in self.inner_edges],
            'inter_edges': [e.to_dict() for e in self.inter_edges],
            'statistics': self.get_statistics()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ChunkGraph':
        """
        Create ChunkGraph from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ChunkGraph instance
        """
        graph = cls(data['chunk_id'], data.get('config', {}))

        # Reconstruct nodes
        graph.nodes = [MaskNode.from_dict(n) for n in data['nodes']]

        # Reconstruct edges
        graph.inner_edges = [InnerEdge.from_dict(e) for e in data['inner_edges']]
        graph.inter_edges = [InterEdge.from_dict(e) for e in data['inter_edges']]

        return graph

    def __repr__(self):
        stats = self.get_statistics()
        return (f"ChunkGraph(id={self.chunk_id}, "
                f"nodes={stats['total_nodes']}, "
                f"anchors={stats['anchor_nodes']}, "
                f"inner_edges={stats['inner_edges']}, "
                f"inter_edges={stats['inter_edges']})")
