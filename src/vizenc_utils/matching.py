"""
Object matching algorithms for tracking across frames.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment


def greedy_match_objects(frame0_embeddings, frame1_embeddings, threshold=0.7):
    """
    Greedy matching algorithm using cosine similarity.
    Finds local optimum by greedily selecting best matches.

    Args:
        frame0_embeddings: numpy array of shape (N0, D) - embeddings from frame 0
        frame1_embeddings: numpy array of shape (N1, D) - embeddings from frame 1
        threshold: minimum similarity threshold for matching

    Returns:
        List of tuples (idx0, idx1, similarity) for matched objects
    """
    sim_matrix = cosine_similarity(frame0_embeddings, frame1_embeddings)

    matches = []
    used_frame1 = set()

    # For each object in frame 0, find best match in frame 1
    for i in range(len(frame0_embeddings)):
        best_j = None
        best_sim = threshold

        for j in range(len(frame1_embeddings)):
            if j not in used_frame1 and sim_matrix[i, j] > best_sim:
                best_sim = sim_matrix[i, j]
                best_j = j

        if best_j is not None:
            matches.append((i, best_j, best_sim))
            used_frame1.add(best_j)

    return matches


def optimal_match_objects(frame0_embeddings, frame1_embeddings, threshold=0.7):
    """
    Optimal matching using Hungarian algorithm (linear_sum_assignment).
    Finds global optimum for bipartite matching problem.

    Args:
        frame0_embeddings: numpy array of shape (N0, D) - embeddings from frame 0
        frame1_embeddings: numpy array of shape (N1, D) - embeddings from frame 1
        threshold: minimum similarity threshold for matching

    Returns:
        List of tuples (idx0, idx1, similarity) for matched objects
    """
    sim_matrix = cosine_similarity(frame0_embeddings, frame1_embeddings)

    # Hungarian algorithm minimizes cost, so invert similarity
    cost_matrix = 1 - sim_matrix

    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Filter by threshold
    matches = []
    for i, j in zip(row_ind, col_ind):
        sim = sim_matrix[i, j]
        if sim > threshold:
            matches.append((i, j, sim))

    return matches


def category_similarity(name1, name2):
    """
    Compute fuzzy string similarity between two category names.

    Args:
        name1: First category name
        name2: Second category name

    Returns:
        Similarity score between 0 and 1
    """
    from difflib import SequenceMatcher

    if not name1 or not name2:
        return 0.0

    # Normalize strings
    s1 = name1.lower().strip()
    s2 = name2.lower().strip()

    # Exact match
    if s1 == s2:
        return 1.0

    # Check if one contains the other
    if s1 in s2 or s2 in s1:
        return 0.85

    # Fuzzy matching
    return SequenceMatcher(None, s1, s2).ratio()


def iou_match_objects(vlkgp_objects, vizenc_objects,
                      iou_weight=0.6, category_weight=0.4,
                      score_threshold=0.3):
    """
    Match objects between vl-kgp and vizEnc using IoU + category similarity.

    Uses Hungarian algorithm for optimal bipartite matching.

    Args:
        vlkgp_objects: List of dicts with keys: 'name', 'bbox' (standard format)
        vizenc_objects: List of dicts with keys: 'category', 'bbox' (standard format)
        iou_weight: Weight for IoU in combined score (default 0.6)
        category_weight: Weight for category similarity (default 0.4)
        score_threshold: Minimum combined score for valid match (default 0.3)

    Returns:
        List of tuples (vlkgp_idx, vizenc_idx, score) for matched objects
    """
    from .bbox_utils import compute_iou_matrix

    n_vlkgp = len(vlkgp_objects)
    n_vizenc = len(vizenc_objects)

    if n_vlkgp == 0 or n_vizenc == 0:
        return []

    # Extract bboxes
    vlkgp_bboxes = [obj['bbox'] for obj in vlkgp_objects]
    vizenc_bboxes = [obj['bbox'] for obj in vizenc_objects]

    # Compute IoU matrix
    iou_matrix = compute_iou_matrix(vlkgp_bboxes, vizenc_bboxes)

    # Compute category similarity matrix
    cat_matrix = np.zeros((n_vlkgp, n_vizenc))
    for i, vlkgp_obj in enumerate(vlkgp_objects):
        vlkgp_name = vlkgp_obj.get('name', '')
        for j, vizenc_obj in enumerate(vizenc_objects):
            vizenc_cat = vizenc_obj.get('category', '')
            cat_matrix[i, j] = category_similarity(vlkgp_name, vizenc_cat)

    # Combined score matrix
    score_matrix = iou_weight * iou_matrix + category_weight * cat_matrix

    # Hungarian matching (maximize score = minimize negative score)
    cost_matrix = -score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Filter by threshold
    matches = []
    for i, j in zip(row_ind, col_ind):
        score = score_matrix[i, j]
        if score >= score_threshold:
            matches.append((i, j, float(score)))

    return matches


def aggregate_frame_matches(frame_matches_list):
    """
    Aggregate matches across multiple frames to determine final object pairs.

    Args:
        frame_matches_list: List of dicts with structure:
            {
                'frame_index': int,
                'matches': [(vlkgp_obj_id, vizenc_anchor_id, score), ...]
            }

    Returns:
        Dict mapping vlkgp_obj_id -> {
            'anchor_id': best matching vizenc anchor,
            'total_score': accumulated score,
            'match_count': number of frames matched,
            'frame_indices': list of frames where matched
        }
    """
    # Accumulate scores for each (vlkgp_id, anchor_id) pair
    pair_scores = {}  # (vlkgp_id, anchor_id) -> {'score': float, 'count': int, 'frames': []}

    for frame_data in frame_matches_list:
        frame_idx = frame_data['frame_index']
        for vlkgp_id, anchor_id, score in frame_data['matches']:
            key = (vlkgp_id, anchor_id)
            if key not in pair_scores:
                pair_scores[key] = {'score': 0, 'count': 0, 'frames': []}
            pair_scores[key]['score'] += score
            pair_scores[key]['count'] += 1
            pair_scores[key]['frames'].append(frame_idx)

    # For each vlkgp object, find best matching anchor
    vlkgp_to_anchor = {}

    # Group by vlkgp_id
    vlkgp_candidates = {}
    for (vlkgp_id, anchor_id), data in pair_scores.items():
        if vlkgp_id not in vlkgp_candidates:
            vlkgp_candidates[vlkgp_id] = []
        vlkgp_candidates[vlkgp_id].append({
            'anchor_id': anchor_id,
            'total_score': data['score'],
            'match_count': data['count'],
            'frame_indices': data['frames']
        })

    # Select best anchor for each vlkgp object
    for vlkgp_id, candidates in vlkgp_candidates.items():
        # Sort by match_count (primary) and total_score (secondary)
        candidates.sort(key=lambda x: (x['match_count'], x['total_score']), reverse=True)
        best = candidates[0]
        vlkgp_to_anchor[vlkgp_id] = best

    return vlkgp_to_anchor
