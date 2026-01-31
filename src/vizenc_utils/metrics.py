"""
Unsupervised quality metrics for object tracking.
"""

import numpy as np


def compute_unsupervised_metrics(matches, frame0_embeddings, frame1_embeddings,
                                 frame0_bboxes, frame1_bboxes):
    """
    Compute quality metrics for object tracking without ground truth.

    Metrics:
    - avg_similarity: Mean cosine similarity of matched pairs (higher = better)
    - match_rate: Proportion of objects successfully matched (0-1)
    - spatial_consistency: How reasonable object movements are (0-1, based on displacement)
    - avg_displacement: Mean pixel displacement of matched objects
    - confidence_score: Weighted combination of all metrics (0-1)

    Args:
        matches: List of (idx0, idx1, similarity) tuples from matching algorithm
        frame0_embeddings: numpy array (N0, D) - embeddings from frame 0
        frame1_embeddings: numpy array (N1, D) - embeddings from frame 1
        frame0_bboxes: List of [x, y, w, h] bounding boxes for frame 0
        frame1_bboxes: List of [x, y, w, h] bounding boxes for frame 1

    Returns:
        Dictionary with metrics:
        - avg_similarity: float
        - match_rate: float (0-1)
        - spatial_consistency: float (0-1)
        - avg_displacement: float (pixels)
        - confidence_score: float (0-1)
        - n_matches: int
        - unmatched_f0: int
        - unmatched_f1: int
    """

    if not matches:
        return {
            'avg_similarity': 0,
            'match_rate': 0,
            'spatial_consistency': 0,
            'confidence_score': 0,
            'n_matches': 0,
            'avg_displacement': 0,
            'unmatched_f0': len(frame0_embeddings),
            'unmatched_f1': len(frame1_embeddings)
        }

    # 1. Average visual similarity of matches (higher = better)
    avg_similarity = np.mean([sim for _, _, sim in matches])

    # 2. Match rate: proportion of objects that were matched
    n0, n1 = len(frame0_embeddings), len(frame1_embeddings)
    matched_f0 = len(set(i for i, j, sim in matches))
    matched_f1 = len(set(j for i, j, sim in matches))
    match_rate = (matched_f0 + matched_f1) / (n0 + n1)

    # 3. Spatial consistency: check if object movements are reasonable
    displacements = []
    for i, j, sim in matches:
        x0, y0, w0, h0 = frame0_bboxes[i]
        x1, y1, w1, h1 = frame1_bboxes[j]

        # Calculate center points
        center0 = np.array([x0 + w0/2, y0 + h0/2])
        center1 = np.array([x1 + w1/2, y1 + h1/2])

        # Euclidean distance between centers
        displacement = np.linalg.norm(center1 - center0)
        displacements.append(displacement)

    avg_displacement = np.mean(displacements)

    # Normalize displacement: assume 200px is max reasonable movement between frames
    # Objects shouldn't "teleport" across the image
    spatial_consistency = 1 - min(avg_displacement / 200, 1.0)

    # 4. Combined confidence score (weighted average)
    # Prioritize visual similarity (60%), then match rate (20%), then spatial (20%)
    confidence_score = (avg_similarity * 0.6 +
                       match_rate * 0.2 +
                       spatial_consistency * 0.2)

    return {
        'avg_similarity': avg_similarity,
        'match_rate': match_rate,
        'spatial_consistency': spatial_consistency,
        'avg_displacement': avg_displacement,
        'confidence_score': confidence_score,
        'n_matches': len(matches),
        'unmatched_f0': n0 - matched_f0,
        'unmatched_f1': n1 - matched_f1
    }
