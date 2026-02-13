"""
Graph edge representations for spatial and temporal relations.

Two types:
- InnerEdge: Spatial relations within a single frame
- InterEdge: Temporal tracking between frames
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class InnerEdge:
    """
    Spatial relation between masks in the SAME frame.

    Represents spatial proximity/relationships between objects.
    """
    source_id: str
    target_id: str
    edge_type: str = "inner"
    relation: str = "spatial"  # Can be extended: 'above', 'below', 'overlaps', etc.
    frame_idx: int = 0
    distance: float = 0.0  # Euclidean distance between centers

    def to_dict(self) -> dict:
        """Convert edge to dictionary."""
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'edge_type': self.edge_type,
            'relation': self.relation,
            'frame_idx': self.frame_idx,
            'distance': self.distance
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'InnerEdge':
        """Create InnerEdge from dictionary."""
        return cls(**data)

    def __repr__(self):
        return (f"InnerEdge({self.source_id} -> {self.target_id}, "
                f"frame={self.frame_idx}, dist={self.distance:.1f})")


@dataclass
class InterEdge:
    """
    Temporal tracking relation between masks in DIFFERENT frames.

    Represents object tracking/matching across time.
    """
    source_id: str
    target_id: str
    edge_type: str = "inter"
    source_frame: int = 0
    target_frame: int = 0
    similarity: float = 0.0  # Cosine similarity
    track_id: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert edge to dictionary."""
        data = {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'edge_type': self.edge_type,
            'source_frame': self.source_frame,
            'target_frame': self.target_frame,
            'similarity': self.similarity
        }

        if self.track_id is not None:
            data['track_id'] = self.track_id

        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'InterEdge':
        """Create InterEdge from dictionary."""
        return cls(**data)

    def __repr__(self):
        return (f"InterEdge({self.source_id} -> {self.target_id}, "
                f"f{self.source_frame}->{self.target_frame}, "
                f"sim={self.similarity:.3f}, track={self.track_id})")
