"""
Bounding box utilities for combining vl-kgp and vizEnc pipelines.

Handles bbox format conversion and IoU computation.
"""

import numpy as np
from typing import List, Tuple, Union


def convert_vlkgp_bbox(bbox_norm: List[float], img_width: int = None, img_height: int = None) -> List[float]:
    """
    Convert vl-kgp bbox format to standard [x_min, y_min, x_max, y_max].

    vl-kgp format: [y_min, x_min, y_max, x_max] (normalized 0-1)
    Output: [x_min, y_min, x_max, y_max] (normalized or pixel)

    Args:
        bbox_norm: vl-kgp normalized bbox [y_min, x_min, y_max, x_max]
        img_width: Optional image width for pixel conversion
        img_height: Optional image height for pixel conversion

    Returns:
        Bbox in standard format [x_min, y_min, x_max, y_max]
    """
    if len(bbox_norm) != 4:
        return [0, 0, 0, 0]

    y_min, x_min, y_max, x_max = bbox_norm

    if img_width is not None and img_height is not None:
        # Convert to pixel coordinates
        return [
            x_min * img_width,
            y_min * img_height,
            x_max * img_width,
            y_max * img_height
        ]
    else:
        # Keep normalized
        return [x_min, y_min, x_max, y_max]


def convert_vizenc_bbox(bbox: List[float], normalize: bool = False,
                        img_width: int = None, img_height: int = None) -> List[float]:
    """
    Convert vizEnc bbox format to standard [x_min, y_min, x_max, y_max].

    vizEnc format: [x, y, w, h] (pixel coordinates)
    Output: [x_min, y_min, x_max, y_max]

    Args:
        bbox: vizEnc bbox [x, y, w, h]
        normalize: If True, normalize to 0-1 range
        img_width: Image width (required if normalize=True)
        img_height: Image height (required if normalize=True)

    Returns:
        Bbox in standard format [x_min, y_min, x_max, y_max]
    """
    if len(bbox) != 4:
        return [0, 0, 0, 0]

    x, y, w, h = bbox
    x_min, y_min = x, y
    x_max, y_max = x + w, y + h

    if normalize and img_width is not None and img_height is not None:
        return [
            x_min / img_width,
            y_min / img_height,
            x_max / img_width,
            y_max / img_height
        ]
    else:
        return [x_min, y_min, x_max, y_max]


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bboxes.

    Both bboxes should be in format [x_min, y_min, x_max, y_max].
    Works with both normalized and pixel coordinates.

    Args:
        bbox1: First bbox [x_min, y_min, x_max, y_max]
        bbox2: Second bbox [x_min, y_min, x_max, y_max]

    Returns:
        IoU value between 0 and 1
    """
    if len(bbox1) != 4 or len(bbox2) != 4:
        return 0.0

    # Get coordinates
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Check for no intersection
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    # Compute areas
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # Avoid division by zero
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def compute_iou_matrix(bboxes1: List[List[float]], bboxes2: List[List[float]]) -> np.ndarray:
    """
    Compute IoU matrix between two sets of bboxes.

    Args:
        bboxes1: List of N bboxes [x_min, y_min, x_max, y_max]
        bboxes2: List of M bboxes [x_min, y_min, x_max, y_max]

    Returns:
        NxM matrix of IoU values
    """
    n = len(bboxes1)
    m = len(bboxes2)

    if n == 0 or m == 0:
        return np.zeros((n, m))

    iou_matrix = np.zeros((n, m))

    for i, bbox1 in enumerate(bboxes1):
        for j, bbox2 in enumerate(bboxes2):
            iou_matrix[i, j] = compute_iou(bbox1, bbox2)

    return iou_matrix


def get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Get center point of a bbox.

    Args:
        bbox: Bbox [x_min, y_min, x_max, y_max]

    Returns:
        Tuple (center_x, center_y)
    """
    if len(bbox) != 4:
        return (0, 0)

    x_min, y_min, x_max, y_max = bbox
    return ((x_min + x_max) / 2, (y_min + y_max) / 2)


def get_bbox_area(bbox: List[float]) -> float:
    """
    Get area of a bbox.

    Args:
        bbox: Bbox [x_min, y_min, x_max, y_max]

    Returns:
        Area value
    """
    if len(bbox) != 4:
        return 0

    x_min, y_min, x_max, y_max = bbox
    return max(0, (x_max - x_min) * (y_max - y_min))


def normalize_bbox(bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """
    Normalize pixel bbox to 0-1 range.

    Args:
        bbox: Bbox in pixels [x_min, y_min, x_max, y_max]
        img_width: Image width
        img_height: Image height

    Returns:
        Normalized bbox [x_min, y_min, x_max, y_max]
    """
    if len(bbox) != 4 or img_width <= 0 or img_height <= 0:
        return [0, 0, 0, 0]

    x_min, y_min, x_max, y_max = bbox
    return [
        x_min / img_width,
        y_min / img_height,
        x_max / img_width,
        y_max / img_height
    ]


def denormalize_bbox(bbox_norm: List[float], img_width: int, img_height: int) -> List[float]:
    """
    Convert normalized bbox to pixel coordinates.

    Args:
        bbox_norm: Normalized bbox [x_min, y_min, x_max, y_max] (0-1)
        img_width: Image width
        img_height: Image height

    Returns:
        Bbox in pixels [x_min, y_min, x_max, y_max]
    """
    if len(bbox_norm) != 4:
        return [0, 0, 0, 0]

    x_min, y_min, x_max, y_max = bbox_norm
    return [
        x_min * img_width,
        y_min * img_height,
        x_max * img_width,
        y_max * img_height
    ]
