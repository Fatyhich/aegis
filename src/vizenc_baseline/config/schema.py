"""
Pydantic schemas for configuration validation.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Data configuration."""
    dataset_name: str
    frames_dir: str
    start_frame: int = 0
    chunk_size: int = 8  # frames per chunk
    max_chunks: Optional[int] = None  # None = process all frames


class FilteringConfig(BaseModel):
    """Filtering configuration."""
    enabled: bool = True
    excluded_categories: List[str] = Field(default_factory=list)
    min_mask_ratio: float = Field(default=0.1, ge=0.0, le=1.0)


class MatchingConfig(BaseModel):
    """Matching configuration."""
    algorithm: Literal["greedy", "hungarian"] = "hungarian"
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class AnchorConfig(BaseModel):
    """Anchor configuration."""
    enabled: bool = True
    averaging_method: Literal["mean", "ema"] = "mean"
    ema_alpha: float = Field(default=0.3, ge=0.0, le=1.0)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class ProcessingConfig(BaseModel):
    """Processing configuration."""
    use_batch: bool = False
    batch_size: int = 8


class OutputConfig(BaseModel):
    """Output configuration."""
    output_dir: str
    export_formats: List[Literal["pkl", "json", "neo4j"]] = ["pkl", "json", "neo4j"]


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""
    data: DataConfig
    filtering: FilteringConfig
    matching: MatchingConfig
    anchors: AnchorConfig
    processing: ProcessingConfig
    output: OutputConfig


class SAMConfig(BaseModel):
    """SAM configuration."""
    version: Literal["sam1", "sam2"] = "sam1"
    checkpoint_path: Optional[str] = None  # None for auto-download
    model_type: str = "vit_h"
    points_per_side: int = 8
    pred_iou_thresh: float = 0.95


class DINOv2Config(BaseModel):
    """DINOv2 configuration."""
    model_name: str = "facebook/dinov2-base"
    target_size: List[int] = [224, 224]


class NaRadIOConfig(BaseModel):
    """NaRadIO configuration."""
    version: str = "radio_v2.5-b"
    lang_model: Literal["clip", "siglip"] = "siglip"
    resolution: List[int] = [512, 512]


class EncoderConfig(BaseModel):
    """Encoder configuration."""
    type: Literal["dinov2", "naradio"] = "naradio"
    dinov2: DINOv2Config = Field(default_factory=DINOv2Config)
    naradio: NaRadIOConfig = Field(default_factory=NaRadIOConfig)


class FlorenceConfig(BaseModel):
    """Florence-2 configuration."""
    enabled: bool = True
    model_name: str = "microsoft/Florence-2-large-ft"


class ZeroShotConfig(BaseModel):
    """Zero-shot classification configuration."""
    enabled: bool = True
    labels: List[str] = Field(default_factory=list)


class ModelsConfig(BaseModel):
    """Models configuration."""
    sam: SAMConfig
    encoder: EncoderConfig
    florence: FlorenceConfig
    zero_shot: ZeroShotConfig


class InnerEdgeConfig(BaseModel):
    """Inner edge (spatial) configuration."""
    proximity_threshold: float = 200.0


class InterEdgeConfig(BaseModel):
    """Inter edge (temporal) configuration."""
    similarity_threshold: float = 0.7


class GraphExportConfig(BaseModel):
    """Graph export configuration."""
    include_segmentation: bool = False
    include_embedding: bool = True
    neo4j: dict = Field(default_factory=lambda: {"batch_size": 1000})


class GraphConfig(BaseModel):
    """Graph configuration."""
    inner: InnerEdgeConfig = Field(default_factory=InnerEdgeConfig)
    inter: InterEdgeConfig = Field(default_factory=InterEdgeConfig)
    export: GraphExportConfig = Field(default_factory=GraphExportConfig)
