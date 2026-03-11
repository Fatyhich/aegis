"""
Main AEGIS pipeline orchestrator.
"""

from pathlib import Path
from typing import List, Dict, Any
import torch
from tqdm import tqdm

from aegis.config import PipelineConfig, ModelsConfig, GraphConfig
from aegis.storage import ChunkGraphStorage
from aegis.pipeline.chunk_processor import ChunkProcessor
from aegis.segmentation.sam import init_sam
from aegis.encoders.dinov2 import load_dinov2_model
from aegis.encoders.naradio import load_naradio_encoder
from aegis.encoders.florence import load_florence_model


class VizEncPipeline:
    """Main AEGIS pipeline."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        models_config: ModelsConfig,
        graph_config: GraphConfig,
        project_dir: Path = None
    ):
        self.pipeline_config = pipeline_config
        self.models_config = models_config
        self.graph_config = graph_config
        self.project_dir = project_dir or Path.cwd()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        self.models = self._load_models()

        self.storage = ChunkGraphStorage(
            self.pipeline_config.output.output_dir,
            self.pipeline_config.data.dataset_name
        )

        config_dict = self._build_config_dict()
        self.chunk_processor = ChunkProcessor(config_dict, self.models)

    def _load_models(self) -> Dict[str, Any]:
        models = {}

        print("Loading SAM...")
        models['sam'] = init_sam(
            self.project_dir,
            version=self.models_config.sam.version,
            checkpoint_path=self.models_config.sam.checkpoint_path,
            model_type=self.models_config.sam.model_type,
            points_per_side=self.models_config.sam.points_per_side,
            pred_iou_thresh=self.models_config.sam.pred_iou_thresh,
            device=str(self.device)
        )

        encoder_type = self.models_config.encoder.type
        print(f"Loading {encoder_type.upper()} encoder...")

        if encoder_type == 'dinov2':
            visual_encoder, visual_processor = load_dinov2_model(
                model_name=self.models_config.encoder.dinov2.model_name,
                device=str(self.device)
            )
            models['visual_encoder'] = visual_encoder
            models['visual_processor'] = visual_processor
            models['visual_device'] = self.device

        elif encoder_type == 'naradio':
            visual_encoder = load_naradio_encoder(
                self.project_dir,
                input_resolution=tuple(self.models_config.encoder.naradio.resolution),
                model_version=self.models_config.encoder.naradio.version,
                lang_model=self.models_config.encoder.naradio.lang_model,
                device=str(self.device)
            )
            models['visual_encoder'] = visual_encoder
            models['visual_processor'] = None
            models['visual_device'] = self.device

        if self.models_config.florence.enabled:
            print("Loading Florence-2...")
            florence_model, florence_processor = load_florence_model(
                model_name=self.models_config.florence.model_name,
                device=str(self.device),
                dtype=torch.float16
            )
            models['florence_model'] = florence_model
            models['florence_processor'] = florence_processor
            models['florence_device'] = self.device
        else:
            models['florence_model'] = None
            models['florence_processor'] = None

        print("All models loaded!")
        return models

    def _build_config_dict(self) -> dict:
        return {
            'processing': {
                'encoder': self.models_config.encoder.type,
                'use_florence': self.models_config.florence.enabled,
                'use_zero_shot': self.models_config.zero_shot.enabled,
                'zero_shot_labels': self.models_config.zero_shot.labels,
                'use_batch': self.pipeline_config.processing.use_batch,
                'batch_size': self.pipeline_config.processing.batch_size,
            },
            'filtering': {
                'enabled': self.pipeline_config.filtering.enabled,
                'excluded_categories': set(self.pipeline_config.filtering.excluded_categories),
                'min_mask_ratio': self.pipeline_config.filtering.min_mask_ratio,
            },
            'matching': {
                'algorithm': self.pipeline_config.matching.algorithm,
                'threshold': self.pipeline_config.matching.threshold,
            },
            'anchors': {
                'enabled': self.pipeline_config.anchors.enabled,
                'averaging_method': self.pipeline_config.anchors.averaging_method,
                'ema_alpha': self.pipeline_config.anchors.ema_alpha,
                'threshold': self.pipeline_config.anchors.threshold,
            },
            'graph': {
                'inner': {
                    'proximity_threshold': self.graph_config.inner.proximity_threshold,
                },
                'inter': {
                    'similarity_threshold': self.graph_config.inter.similarity_threshold,
                },
            }
        }

    def _load_frames(self) -> List[Path]:
        frames_dir = Path(self.pipeline_config.data.frames_dir)

        for pattern in ['*.jpg', '*.png', 'frame_*.png']:
            frames = sorted(frames_dir.glob(pattern))
            if frames:
                break

        if not frames:
            raise FileNotFoundError(f"No frames found in {frames_dir}")

        print(f"Found {len(frames)} total frames in folder")

        start_idx = self.pipeline_config.data.start_frame
        frames = frames[start_idx:]

        chunk_size = self.pipeline_config.data.chunk_size
        max_chunks = self.pipeline_config.data.max_chunks

        if max_chunks is not None:
            max_frames = max_chunks * chunk_size
            frames = frames[:max_frames]
            print(f"Limited to {max_chunks} chunks ({max_frames} frames)")

        print(f"Will process {len(frames)} frames starting from index {start_idx}")

        return frames

    def run(self) -> List:
        frames = self._load_frames()
        chunk_size = self.pipeline_config.data.chunk_size
        start_frame = self.pipeline_config.data.start_frame

        results = []

        num_chunks = (len(frames) + chunk_size - 1) // chunk_size
        chunk_indices = range(0, len(frames), chunk_size)

        if num_chunks > 1:
            chunk_iter = tqdm(chunk_indices, desc="Processing chunks", total=num_chunks, ncols=100)
        else:
            chunk_iter = chunk_indices

        for chunk_idx in chunk_iter:
            chunk_frames = frames[chunk_idx:chunk_idx + chunk_size]
            abs_start_idx = start_frame + chunk_idx
            abs_end_idx = abs_start_idx + len(chunk_frames)
            chunk_id = f"chunk_{abs_start_idx:04d}_{abs_end_idx:04d}"

            print(f"\n{'='*60}")
            print(f"CHUNK {chunk_idx // chunk_size + 1}/{num_chunks}: {chunk_id}")
            print(f"{'='*60}")

            graph, anchor_db, mask_db = self.chunk_processor.process_chunk(
                chunk_frames, abs_start_idx, chunk_id
            )

            saved = self.storage.save_graph(
                graph,
                abs_start_idx,
                chunk_size,
                formats=self.pipeline_config.output.export_formats,
                include_segmentation=self.graph_config.export.include_segmentation,
                include_embedding=self.graph_config.export.include_embedding
            )

            print(f"\nSaved graphs:")
            for fmt, filepath in saved.items():
                print(f"  {fmt}: {filepath}")

            results.append({
                'graph': graph,
                'anchor_db': anchor_db,
                'mask_db': mask_db,
                'files': saved
            })

        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Processed {len(results)} chunks")
        print(f"Output directory: {self.storage.output_dir}")

        return results
