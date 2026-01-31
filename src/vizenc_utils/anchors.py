"""
Anchor-based object tracking for building knowledge graphs.

Anchors represent persistent semantic objects in the environment,
accumulated across multiple frames.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def create_anchor_db():
    """
    Create empty anchor database.

    Returns:
        dict: Empty anchor database
    """
    return {
        'anchors': {},      # track_id -> anchor data
        'next_track_id': 0  # Counter for new track IDs
    }


def update_anchors(anchor_db, frame_idx, current_masks, prev_masks, matches,
                   threshold=0.7, averaging_method='mean', ema_alpha=0.3):
    """
    Update anchor database with new frame observations.

    Algorithm:
    1. For each matched pair (prev -> current):
       - If prev has track_id: verify with anchor, update or split
       - If prev has no track_id: create new anchor from pair
    2. For unmatched current objects:
       - Try to match with all anchors (re-identification)
       - If no match: create new anchor

    Args:
        anchor_db: Anchor database dict
        frame_idx: Current frame index
        current_masks: List of masks from current frame
        prev_masks: List of masks from previous frame (or None for first frame)
        matches: List of (prev_idx, curr_idx, similarity) tuples
        threshold: Similarity threshold for anchor matching
        averaging_method: 'mean' or 'ema' for embedding averaging
        ema_alpha: Alpha parameter for EMA (0-1, only used if method='ema')

    Returns:
        None (modifies anchor_db in-place)
    """
    anchors = anchor_db['anchors']

    # Track which current objects have been assigned
    assigned_current = set()

    # === Step 1: Process matched pairs ===
    for prev_idx, curr_idx, match_sim in matches:
        curr_mask = current_masks[curr_idx]
        prev_mask = prev_masks[prev_idx] if prev_masks else None

        # Check if previous object has track_id
        if prev_mask and 'track_id' in prev_mask and prev_mask['track_id'] is not None:
            track_id = prev_mask['track_id']

            # Verify with anchor
            if track_id in anchors:
                anchor = anchors[track_id]
                anchor_emb = anchor['embedding']
                curr_emb = curr_mask['embedding']

                # Check similarity with anchor
                anchor_sim = cosine_similarity([curr_emb], [anchor_emb])[0][0]

                if anchor_sim > threshold:
                    # GOOD MATCH - Update anchor
                    _update_anchor(anchor, curr_mask, frame_idx, anchor_sim,
                                  averaging_method, ema_alpha)
                    curr_mask['track_id'] = track_id
                    assigned_current.add(curr_idx)
                else:
                    # BAD MATCH - ID drift detected, create new anchor
                    track_id = _create_new_anchor(anchor_db, curr_mask, frame_idx, match_sim)
                    curr_mask['track_id'] = track_id
                    assigned_current.add(curr_idx)
            else:
                # Anchor doesn't exist (shouldn't happen, but handle gracefully)
                track_id = _create_new_anchor(anchor_db, curr_mask, frame_idx, match_sim)
                curr_mask['track_id'] = track_id
                assigned_current.add(curr_idx)
        else:
            # Previous object has no track_id - create new anchor from matched pair
            if prev_mask:
                # Average embeddings from both frames
                emb0 = prev_mask['embedding']
                emb1 = curr_mask['embedding']
                avg_emb = (emb0 + emb1) / 2.0

                track_id = _create_anchor_from_pair(
                    anchor_db, prev_mask, curr_mask,
                    avg_emb, frame_idx - 1, frame_idx, match_sim
                )

                # Assign track_id to both masks
                prev_mask['track_id'] = track_id
                curr_mask['track_id'] = track_id
                assigned_current.add(curr_idx)
            else:
                # No previous mask - create new anchor
                track_id = _create_new_anchor(anchor_db, curr_mask, frame_idx, match_sim)
                curr_mask['track_id'] = track_id
                assigned_current.add(curr_idx)

    # === Step 2: Re-identification for unmatched objects ===
    for curr_idx, curr_mask in enumerate(current_masks):
        if curr_idx in assigned_current:
            continue  # Already assigned

        # Try to match with all anchors
        curr_emb = curr_mask['embedding']
        best_track_id = None
        best_sim = threshold

        for track_id, anchor in anchors.items():
            anchor_emb = anchor['embedding']
            sim = cosine_similarity([curr_emb], [anchor_emb])[0][0]

            if sim > best_sim:
                best_sim = sim
                best_track_id = track_id

        if best_track_id is not None:
            # RE-IDENTIFICATION - Revive anchor
            anchor = anchors[best_track_id]
            _update_anchor(anchor, curr_mask, frame_idx, best_sim,
                          averaging_method, ema_alpha)
            curr_mask['track_id'] = best_track_id
        else:
            # NEW OBJECT - Create new anchor
            track_id = _create_new_anchor(anchor_db, curr_mask, frame_idx, 1.0)
            curr_mask['track_id'] = track_id


def _create_new_anchor(anchor_db, mask, frame_idx, confidence):
    """Create new anchor from single observation."""
    track_id = anchor_db['next_track_id']
    anchor_db['next_track_id'] += 1

    anchor_db['anchors'][track_id] = {
        'embedding': mask['embedding'].copy(),
        'embeddings_history': [mask['embedding'].copy()],
        'frames': [frame_idx],
        'bboxes': [mask['bbox']],
        'descriptions': [mask.get('description', '')],
        'category': mask.get('category', 'unknown'),
        'first_seen': frame_idx,
        'last_seen': frame_idx,
        'n_observations': 1,
        'confidences': [confidence],
        'avg_confidence': confidence
    }

    return track_id


def _create_anchor_from_pair(anchor_db, mask0, mask1, avg_emb,
                             frame0_idx, frame1_idx, confidence):
    """Create new anchor from matched pair of observations."""
    track_id = anchor_db['next_track_id']
    anchor_db['next_track_id'] += 1

    anchor_db['anchors'][track_id] = {
        'embedding': avg_emb.copy(),
        'embeddings_history': [mask0['embedding'].copy(), mask1['embedding'].copy()],
        'frames': [frame0_idx, frame1_idx],
        'bboxes': [mask0['bbox'], mask1['bbox']],
        'descriptions': [mask0.get('description', ''), mask1.get('description', '')],
        'category': mask1.get('category', 'unknown'),
        'first_seen': frame0_idx,
        'last_seen': frame1_idx,
        'n_observations': 2,
        'confidences': [confidence, confidence],
        'avg_confidence': confidence
    }

    return track_id


def _update_anchor(anchor, new_mask, frame_idx, confidence,
                  averaging_method='mean', ema_alpha=0.3):
    """Update existing anchor with new observation."""
    new_emb = new_mask['embedding'].copy()

    # Update embedding using specified method
    if averaging_method == 'ema':
        # Exponential Moving Average
        anchor['embedding'] = ema_alpha * new_emb + (1 - ema_alpha) * anchor['embedding']
    else:  # 'mean'
        # Simple average - recalculate from history
        anchor['embeddings_history'].append(new_emb)
        anchor['embedding'] = np.mean(anchor['embeddings_history'], axis=0)

    # Update metadata
    anchor['frames'].append(frame_idx)
    anchor['bboxes'].append(new_mask['bbox'])
    anchor['descriptions'].append(new_mask.get('description', ''))
    anchor['last_seen'] = frame_idx
    anchor['n_observations'] += 1
    anchor['confidences'].append(confidence)
    anchor['avg_confidence'] = np.mean(anchor['confidences'])

    # Update category if available
    if 'category' in new_mask:
        anchor['category'] = new_mask['category']


def get_anchor_summary(anchor_db):
    """
    Get summary statistics about anchors.

    Args:
        anchor_db: Anchor database dict

    Returns:
        dict with summary statistics
    """
    anchors = anchor_db['anchors']

    if not anchors:
        return {
            'n_anchors': 0,
            'total_observations': 0,
            'avg_observations_per_anchor': 0,
            'categories': {}
        }

    total_obs = sum(a['n_observations'] for a in anchors.values())
    categories = {}

    for anchor in anchors.values():
        cat = anchor['category']
        categories[cat] = categories.get(cat, 0) + 1

    return {
        'n_anchors': len(anchors),
        'total_observations': total_obs,
        'avg_observations_per_anchor': total_obs / len(anchors),
        'categories': categories,
        'track_ids': list(anchors.keys())
    }


def get_anchor_by_id(anchor_db, track_id):
    """
    Get anchor data by track ID.

    Args:
        anchor_db: Anchor database dict
        track_id: Track ID to retrieve

    Returns:
        dict with anchor data or None if not found
    """
    return anchor_db['anchors'].get(track_id)


def export_anchors_for_graph(anchor_db, min_observations=2):
    """
    Export anchors in format suitable for knowledge graph construction.

    Args:
        anchor_db: Anchor database dict
        min_observations: Minimum observations to include anchor

    Returns:
        List of anchor dicts with graph-relevant fields
    """
    graph_nodes = []

    for track_id, anchor in anchor_db['anchors'].items():
        if anchor['n_observations'] < min_observations:
            continue

        graph_nodes.append({
            'id': track_id,
            'type': anchor['category'],
            'description': anchor['descriptions'][-1],  # Latest description
            'embedding': anchor['embedding'].tolist(),
            'observations': anchor['n_observations'],
            'confidence': anchor['avg_confidence'],
            'frames': anchor['frames'],
            'first_seen': anchor['first_seen'],
            'last_seen': anchor['last_seen']
        })

    return graph_nodes
