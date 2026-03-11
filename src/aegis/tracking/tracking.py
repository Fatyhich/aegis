"""
Object tracking utilities - assign persistent track IDs across frames.
"""


def assign_track_ids(mask_db, matches):
    """
    Assign persistent track IDs to objects across frames based on matches.

    Track IDs are assigned as follows:
    - Frame 0: All objects get new sequential track IDs (0, 1, 2, ...)
    - Frame N: Matched objects inherit track_id from previous frame,
               unmatched objects get new track IDs

    Args:
        mask_db: List of (frame_name, masks) tuples
                 masks will be modified in-place to add 'track_id' field
        matches: List of (idx0, idx1, similarity) tuples from matching algorithm

    Returns:
        int: Next available track ID (for future frames)
    """
    if len(mask_db) == 0:
        return 0

    # Track ID counter
    next_track_id = 0

    # Frame 0: Initialize all objects with new track IDs
    for mask in mask_db[0][1]:
        mask['track_id'] = next_track_id
        next_track_id += 1

    # Subsequent frames: Assign track IDs based on matches
    if len(mask_db) >= 2:
        prev_masks = mask_db[-2][1]  # Previous frame
        curr_masks = mask_db[-1][1]  # Current frame

        # Initialize all current frame objects as unmatched (None)
        for mask in curr_masks:
            mask['track_id'] = None

        # Assign track IDs to matched objects
        for idx_prev, idx_curr, sim in matches:
            curr_masks[idx_curr]['track_id'] = prev_masks[idx_prev]['track_id']

        # Assign new track IDs to unmatched objects
        for mask in curr_masks:
            if mask['track_id'] is None:
                mask['track_id'] = next_track_id
                next_track_id += 1

    return next_track_id


def get_track_summary(mask_db):
    """
    Get summary statistics about tracks across all frames.

    Args:
        mask_db: List of (frame_name, masks) tuples with track_id field

    Returns:
        dict with:
        - n_tracks: Total number of unique tracks
        - track_lengths: Dict mapping track_id -> number of frames it appears in
        - tracks_per_frame: List of number of tracks per frame
    """
    if len(mask_db) == 0:
        return {'n_tracks': 0, 'track_lengths': {}, 'tracks_per_frame': []}

    # Collect all track IDs
    all_track_ids = set()
    track_lengths = {}
    tracks_per_frame = []

    for frame_name, masks in mask_db:
        frame_tracks = set()
        for mask in masks:
            track_id = mask.get('track_id')
            if track_id is not None:
                all_track_ids.add(track_id)
                frame_tracks.add(track_id)
                track_lengths[track_id] = track_lengths.get(track_id, 0) + 1
        tracks_per_frame.append(len(frame_tracks))

    return {
        'n_tracks': len(all_track_ids),
        'track_lengths': track_lengths,
        'tracks_per_frame': tracks_per_frame
    }


def get_object_trajectories(mask_db):
    """
    Extract object trajectories (center positions over time).

    Args:
        mask_db: List of (frame_name, masks) tuples with track_id and bbox

    Returns:
        Dict mapping track_id -> list of (frame_idx, center_x, center_y) tuples
    """
    trajectories = {}

    for frame_idx, (frame_name, masks) in enumerate(mask_db):
        for mask in masks:
            track_id = mask.get('track_id')
            if track_id is not None:
                x, y, w, h = mask['bbox']
                center_x = x + w / 2
                center_y = y + h / 2

                if track_id not in trajectories:
                    trajectories[track_id] = []
                trajectories[track_id].append((frame_idx, center_x, center_y))

    return trajectories
