"""
SAM (Segment Anything Model) initialization utilities.
Supports both SAM 1 and SAM 2.
"""

import sys
from pathlib import Path


def init_sam1(project_dir, checkpoint_path, model_type="vit_h",
              points_per_side=32, pred_iou_thresh=0.88, device="cuda"):
    """
    Initialize SAM 1 with automatic mask generator.

    Args:
        project_dir: Path to project root (where segment-anything is located)
        checkpoint_path: Path to SAM checkpoint file
        model_type: SAM model type ("vit_h", "vit_l", "vit_b")
        points_per_side: Number of points per side for grid sampling
        pred_iou_thresh: IoU threshold for mask quality
        device: Device to load model on ('cuda', 'cpu', 'mps')

    Returns:
        SamAutomaticMaskGenerator instance
    """
    # Add segment-anything to Python path
    sam_path = Path(project_dir) / "segment-anything"
    if str(sam_path) not in sys.path:
        sys.path.insert(0, str(sam_path))

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    # Load SAM model
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device)

    # Create mask generator
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
    )

    return mask_generator


def init_sam2(project_dir, checkpoint_path, model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
              device="cuda", apply_postprocessing=False):
    """
    Initialize SAM 2 with automatic mask generator.

    Args:
        project_dir: Path to project root (where sam2 is located)
        checkpoint_path: Path to SAM 2 checkpoint file
        model_cfg: Path to model config file (relative to sam2 directory)
        device: Device to load model on ('cuda', 'cpu', 'mps')
        apply_postprocessing: Whether to apply post-processing

    Returns:
        SAM2AutomaticMaskGenerator instance
    """
    # Add sam2 to Python path
    sam2_path = Path(project_dir) / "sam2"
    if str(sam2_path) not in sys.path:
        sys.path.insert(0, str(sam2_path))

    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    # Build SAM 2 model
    sam2 = build_sam2(model_cfg, str(checkpoint_path),
                     device=device, apply_postprocessing=apply_postprocessing)

    # Create mask generator
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    return mask_generator


def init_sam(project_dir, version="sam1", **kwargs):
    """
    Unified SAM initialization function.

    Args:
        project_dir: Path to project root
        version: "sam1" or "sam2"
        **kwargs: Additional arguments passed to init_sam1 or init_sam2

    Returns:
        Mask generator instance
    """
    if version == "sam1":
        return init_sam1(project_dir, **kwargs)
    elif version == "sam2":
        return init_sam2(project_dir, **kwargs)
    else:
        raise ValueError(f"Unknown SAM version: {version}. Use 'sam1' or 'sam2'.")
