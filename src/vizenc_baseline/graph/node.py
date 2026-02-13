"""
Graph node representation for masks.

Each detected mask becomes a node in the topological graph.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class MaskNode:
    """
    Represents a single mask as a node in the graph.

    Every detected mask is a node, with is_anchor flag for tracked objects.
    """
    node_id: str
    frame_idx: int
    mask_idx: int
    chunk_id: str
    bbox: List[float]
    embedding: np.ndarray
    category: str
    description: str
    confidence: float
    is_anchor: bool = False
    track_id: Optional[int] = None
    segmentation: Optional[np.ndarray] = None

    def to_dict(self, include_segmentation: bool = False, include_embedding: bool = True) -> dict:
        """
        Convert node to dictionary for serialization.

        Args:
            include_segmentation: Include binary mask (large!)
            include_embedding: Include embedding vector

        Returns:
            Dictionary representation
        """
        data = {
            'node_id': self.node_id,
            'frame_idx': self.frame_idx,
            'mask_idx': self.mask_idx,
            'chunk_id': self.chunk_id,
            'is_anchor': self.is_anchor,
            'bbox': self.bbox,
            'category': self.category,
            'description': self.description,
            'confidence': self.confidence,
        }

        if self.track_id is not None:
            data['track_id'] = self.track_id

        if include_embedding:
            data['embedding'] = self.embedding.tolist()

        if include_segmentation and self.segmentation is not None:
            data['segmentation'] = self.segmentation.tolist()

        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'MaskNode':
        """Create MaskNode from dictionary."""
        # Convert lists back to numpy arrays
        if 'embedding' in data and isinstance(data['embedding'], list):
            data['embedding'] = np.array(data['embedding'])
        if 'segmentation' in data and isinstance(data['segmentation'], list):
            data['segmentation'] = np.array(data['segmentation'])

        return cls(**data)

    def __repr__(self):
        return (f"MaskNode(id={self.node_id}, frame={self.frame_idx}, "
                f"category={self.category}, anchor={self.is_anchor}, "
                f"track={self.track_id})")
