"""
Spatial relation builder for inner edges (within single frame).
"""

import math
from typing import List, Tuple
from .node import MaskNode
from .edge import InnerEdge


class SpatialRelationBuilder:
    """
    Builds spatial relations between masks in a single frame.

    Simple version: Connect nearby objects based on proximity threshold.
    Can be extended with more sophisticated relations (above, below, overlaps, etc.)
    """

    def __init__(self, config: dict):
        """
        Initialize spatial relation builder.

        Args:
            config: Configuration dict with 'proximity_threshold' (pixels)
        """
        self.proximity_threshold = config.get('proximity_threshold', 200.0)

    def build_frame_relations(self, nodes: List[MaskNode], frame_idx: int) -> List[InnerEdge]:
        """
        Build inner edges between all masks in a single frame.

        Connects objects that are spatially close (distance < threshold).

        Args:
            nodes: List of MaskNode instances from the same frame
            frame_idx: Frame index (for edge metadata)

        Returns:
            List of InnerEdge instances
        """
        edges = []
        n = len(nodes)

        if n < 2:
            return edges  # No edges if less than 2 nodes

        # Compute bbox centers
        centers = [self._get_bbox_center(node.bbox) for node in nodes]

        # Build edges between nearby objects
        for i in range(n):
            for j in range(i + 1, n):
                distance = self._euclidean_distance(centers[i], centers[j])

                # Connect if within proximity threshold
                if distance < self.proximity_threshold:
                    edge = InnerEdge(
                        source_id=nodes[i].node_id,
                        target_id=nodes[j].node_id,
                        relation='spatial',  # Simplified - can be extended
                        frame_idx=frame_idx,
                        distance=distance
                    )
                    edges.append(edge)

        return edges

    @staticmethod
    def _get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
        """Get center point of bbox."""
        x, y, w, h = bbox
        return (x + w / 2, y + h / 2)

    @staticmethod
    def _euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Compute Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Future extension: More sophisticated spatial relations
class AdvancedSpatialRelations:
    """
    Advanced spatial relation extraction (for future use).

    Can detect:
    - Directional: 'above', 'below', 'left_of', 'right_of'
    - Topological: 'overlaps', 'contains', 'inside'
    - Metric: 'near', 'far'
    """

    @staticmethod
    def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """Compute Intersection over Union."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Intersection rectangle
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def get_directional_relation(bbox1: List[float], bbox2: List[float]) -> str:
        """Get directional relation (above/below/left/right)."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        center1 = (x1 + w1 / 2, y1 + h1 / 2)
        center2 = (x2 + w2 / 2, y2 + h2 / 2)

        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]

        # Use larger difference to determine primary direction
        if abs(dx) > abs(dy):
            return 'right_of' if dx > 0 else 'left_of'
        else:
            return 'below' if dy > 0 else 'above'
