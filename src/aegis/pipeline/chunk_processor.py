"""
Chunk processor - processes a chunk of frames and builds graph.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

from aegis.pipeline.processing import process_masks_with_features, filter_masks
from aegis.tracking.matching import optimal_match_objects, greedy_match_objects
from aegis.tracking.anchors import create_anchor_db, update_anchors
from aegis.graph import ChunkGraph


class ChunkProcessor:
    """Processes a chunk of frames and builds topological graph."""

    def __init__(self, config: dict, models: dict):
        self.config = config
        self.models = models

    def process_chunk(
        self,
        frames: List[Path],
        start_frame_idx: int,
        chunk_id: str
    ) -> Tuple[ChunkGraph, dict, list]:
        graph = ChunkGraph(chunk_id, self.config.get('graph', {}))
        anchor_db = create_anchor_db() if self.config.get('anchors', {}).get('enabled', True) else None
        mask_db = []

        print(f"\nProcessing chunk: {chunk_id}")
        print(f"Frames: {start_frame_idx} to {start_frame_idx + len(frames) - 1}")

        pbar = tqdm(enumerate(frames), total=len(frames), desc="Processing frames", ncols=100)

        for i, frame_path in pbar:
            abs_frame_idx = start_frame_idx + i
            pbar.set_description(f"Frame {abs_frame_idx}")

            image = Image.open(frame_path).convert("RGB")

            pbar.set_postfix_str("SAM segmentation...")
            masks = self.models['sam'].generate(np.array(image))
            pbar.set_postfix_str(f"{len(masks)} masks")

            pbar.set_postfix_str("Extracting features...")
            masks = process_masks_with_features(
                image, masks,
                self.config.get('processing', {}),
                self.models
            )

            if self.config.get('filtering', {}).get('enabled', False):
                pbar.set_postfix_str("Filtering masks...")
                masks = filter_masks(
                    masks, image.size,
                    self.config.get('filtering', {})
                )

            if i > 0:
                pbar.set_postfix_str("Matching with prev frame...")
                prev_masks = mask_db[-1][1]
                prev_emb = np.array([m['embedding'] for m in prev_masks])
                curr_emb = np.array([m['embedding'] for m in masks])

                matching_config = self.config.get('matching', {})
                if matching_config.get('algorithm', 'hungarian') == 'hungarian':
                    matches = optimal_match_objects(
                        prev_emb, curr_emb,
                        threshold=matching_config.get('threshold', 0.7)
                    )
                else:
                    matches = greedy_match_objects(
                        prev_emb, curr_emb,
                        threshold=matching_config.get('threshold', 0.7)
                    )

                pbar.set_postfix_str(f"Matched {len(matches)} objects")

                if anchor_db is not None:
                    anchor_config = self.config.get('anchors', {})
                    update_anchors(
                        anchor_db,
                        abs_frame_idx,
                        masks,
                        prev_masks,
                        matches,
                        threshold=anchor_config.get('threshold', 0.7),
                        averaging_method=anchor_config.get('averaging_method', 'mean'),
                        ema_alpha=anchor_config.get('ema_alpha', 0.3)
                    )

                current_matches = matches

            else:
                if anchor_db is not None:
                    anchor_config = self.config.get('anchors', {})
                    update_anchors(
                        anchor_db,
                        abs_frame_idx,
                        masks,
                        None,
                        [],
                        threshold=anchor_config.get('threshold', 0.7),
                        averaging_method=anchor_config.get('averaging_method', 'mean'),
                        ema_alpha=anchor_config.get('ema_alpha', 0.3),
                        skip_reidentification=True
                    )
                current_matches = None

            pbar.set_postfix_str("Building graph...")
            frame_nodes = graph.add_frame_masks(abs_frame_idx, masks, anchor_db)

            if i > 0 and current_matches:
                graph.build_inter_edges(current_matches, abs_frame_idx - 1, abs_frame_idx)

            mask_db.append((frame_path.name, masks))
            pbar.set_postfix_str(f"✓ {len(masks)} masks, {len(frame_nodes)} nodes")

        pbar.close()

        stats = graph.get_statistics()
        print(f"\n✓ Chunk complete:")
        print(f"  • Total nodes: {stats['total_nodes']}")
        print(f"  • Anchor nodes: {stats['anchor_nodes']}")
        print(f"  • Inner edges: {stats['inner_edges']}")
        print(f"  • Inter edges: {stats['inter_edges']}")

        return graph, anchor_db, mask_db
