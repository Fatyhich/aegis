"""
Utilities for the vizEnc pipeline.
"""

from .matching import (
    greedy_match_objects,
    optimal_match_objects,
    category_similarity,
    iou_match_objects,
    aggregate_frame_matches
)

from .bbox_utils import (
    convert_vlkgp_bbox,
    convert_vizenc_bbox,
    compute_iou,
    compute_iou_matrix,
    get_bbox_center,
    get_bbox_area,
    normalize_bbox,
    denormalize_bbox
)

from .metrics import compute_unsupervised_metrics
from .tracking import assign_track_ids, get_track_summary
from .anchors import (
    create_anchor_db,
    update_anchors,
    get_anchor_summary,
    get_anchor_by_id,
    export_anchors_for_graph
)

__all__ = [
    # Matching
    'greedy_match_objects',
    'optimal_match_objects',
    'category_similarity',
    'iou_match_objects',
    'aggregate_frame_matches',

    # Bbox utilities
    'convert_vlkgp_bbox',
    'convert_vizenc_bbox',
    'compute_iou',
    'compute_iou_matrix',
    'get_bbox_center',
    'get_bbox_area',
    'normalize_bbox',
    'denormalize_bbox',

    # Metrics
    'compute_unsupervised_metrics',

    # Tracking
    'assign_track_ids',
    'get_track_summary',

    # Anchors
    'create_anchor_db',
    'update_anchors',
    'get_anchor_summary',
    'get_anchor_by_id',
    'export_anchors_for_graph',
]
