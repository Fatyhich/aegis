"""
Temporal edge builder for inter-frame tracking.
"""

from typing import List, Tuple
from .node import MaskNode
from .edge import InterEdge


class TemporalEdgeBuilder:
    """
    Builds temporal tracking edges between frames.

    Creates InterEdge instances based on mask matching results.
    """

    def build_tracking_edges(
        self,
        prev_nodes: List[MaskNode],
        curr_nodes: List[MaskNode],
        matches: List[Tuple[int, int, float]],
        prev_frame_idx: int,
        curr_frame_idx: int
    ) -> List[InterEdge]:
        """
        Build inter edges based on matching results.

        Args:
            prev_nodes: Nodes from previous frame
            curr_nodes: Nodes from current frame
            matches: List of (prev_idx, curr_idx, similarity) tuples
            prev_frame_idx: Previous frame index
            curr_frame_idx: Current frame index

        Returns:
            List of InterEdge instances
        """
        edges = []

        for prev_idx, curr_idx, similarity in matches:
            # Validate indices
            if prev_idx >= len(prev_nodes) or curr_idx >= len(curr_nodes):
                continue

            prev_node = prev_nodes[prev_idx]
            curr_node = curr_nodes[curr_idx]

            # Create inter edge
            edge = InterEdge(
                source_id=prev_node.node_id,
                target_id=curr_node.node_id,
                source_frame=prev_frame_idx,
                target_frame=curr_frame_idx,
                similarity=similarity,
                track_id=curr_node.track_id  # Use current node's track_id
            )

            edges.append(edge)

        return edges
