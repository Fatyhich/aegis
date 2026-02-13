"""
VizEnc inference module for segment matching evaluation.

Uses SAM for segmentation + DINOv2/NaRADIO encoders for matching.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from segmentation.sam import init_sam
from encoders.dinov2 import load_dinov2_model, get_dinov2_embeddings_batch
from encoders.naradio import load_naradio_encoder, get_naradio_embeddings_batch


class VizEncMatcher:
    """VizEnc segment matching system."""

    def __init__(self, encoder_type='dinov2', sam_checkpoint=None,
                 encoder_model=None, device='cuda'):
        """
        Initialize VizEnc matcher.

        Args:
            encoder_type: 'dinov2' or 'naradio'
            sam_checkpoint: Path to SAM checkpoint
            encoder_model: Model name/version for encoder
            device: Device to use
        """
        self.encoder_type = encoder_type
        self.device = device

        # Initialize SAM1
        print("Initializing SAM1...")
        project_dir = Path(__file__).parent.parent
        sam_checkpoint = sam_checkpoint or "/mnt/vol0/weights/sam/sam_vit_h_4b8939.pth"

        # Auto-detect model_type from checkpoint filename
        checkpoint_name = Path(sam_checkpoint).name
        if 'vit_b' in checkpoint_name:
            model_type = 'vit_b'
        elif 'vit_l' in checkpoint_name:
            model_type = 'vit_l'
        else:
            model_type = 'vit_h'

        print(f"Using SAM1 model: {model_type}")

        self.sam = init_sam(
            project_dir=project_dir,
            version='sam1',
            checkpoint_path=sam_checkpoint,
            model_type=model_type,
            device=device
        )

        # Initialize encoder
        print(f"Initializing {encoder_type.upper()} encoder...")
        if encoder_type == 'dinov2':
            model_name = encoder_model or "facebook/dinov2-base"
            self.encoder, self.processor = load_dinov2_model(
                model_name=model_name,
                device=device
            )
        elif encoder_type == 'naradio':
            model_version = encoder_model or "radio_v2.5-b"
            self.encoder = load_naradio_encoder(
                project_dir=project_dir,
                input_resolution=(512, 512),
                model_version=model_version,
                device=device
            )
            self.processor = None
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def segment_image(self, image_rgb):
        """
        Generate masks for an image using SAM.

        Args:
            image_rgb: PIL Image or numpy array (H, W, 3)

        Returns:
            numpy array (N, H, W) - binary masks
        """
        if isinstance(image_rgb, Image.Image):
            image_rgb = np.array(image_rgb)

        # SAM expects RGB uint8
        masks_data = self.sam.generate(image_rgb)

        if len(masks_data) == 0:
            return np.zeros((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=bool)

        # Extract masks
        masks = np.array([m['segmentation'] for m in masks_data])
        return masks

    def extract_mask_crops(self, image_rgb, masks):
        """
        Extract image crops for each mask.

        Args:
            image_rgb: PIL Image or numpy array (H, W, 3)
            masks: numpy array (N, H, W) - binary masks

        Returns:
            List of PIL Images (crops)
        """
        if isinstance(image_rgb, np.ndarray):
            image_rgb = Image.fromarray(image_rgb)

        crops = []
        for mask in masks:
            # Find bounding box
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)

            if not rows.any() or not cols.any():
                # Empty mask - use 1x1 crop
                crops.append(Image.new('RGB', (1, 1)))
                continue

            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            # Crop image
            crop = image_rgb.crop((cmin, rmin, cmax + 1, rmax + 1))
            crops.append(crop)

        return crops

    def encode_crops(self, crops, batch_size=8):
        """
        Extract embeddings for image crops.

        Args:
            crops: List of PIL Images
            batch_size: Batch size for encoding

        Returns:
            numpy array (N, D) - L2-normalized embeddings
        """
        if len(crops) == 0:
            return np.zeros((0, 768))  # Default embedding dim

        if self.encoder_type == 'dinov2':
            embeddings = get_dinov2_embeddings_batch(
                crops, self.encoder, self.processor,
                self.device, batch_size=batch_size
            )
        elif self.encoder_type == 'naradio':
            embeddings = get_naradio_embeddings_batch(
                crops, self.encoder, batch_size=batch_size
            )
        else:
            raise ValueError(f"Unknown encoder: {self.encoder_type}")

        embeddings = np.array(embeddings)

        # L2 normalize embeddings (unit vectors)
        # After normalization: cosine_similarity = dot_product (faster!)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        embeddings = embeddings / norms

        return embeddings

    def match_pair(self, img0_path, img1_path, target_size=None):
        """
        Match segments between two images.

        Args:
            img0_path: Path to first image
            img1_path: Path to second image
            target_size: Optional (height, width) to resize images before segmentation

        Returns:
            Tuple of:
                - masks0: numpy array (M, H, W)
                - masks1: numpy array (N, H, W)
                - scores: numpy array (M, N) - similarity matrix
        """
        # Load images
        img0 = Image.open(img0_path).convert('RGB')
        img1 = Image.open(img1_path).convert('RGB')

        # Resize if target_size specified
        if target_size is not None:
            target_h, target_w = target_size
            img0 = img0.resize((target_w, target_h), Image.BILINEAR)
            img1 = img1.resize((target_w, target_h), Image.BILINEAR)

        # Segment
        masks0 = self.segment_image(img0)
        masks1 = self.segment_image(img1)

        if len(masks0) == 0 or len(masks1) == 0:
            return masks0, masks1, np.zeros((len(masks0), len(masks1)))

        # Extract crops
        crops0 = self.extract_mask_crops(img0, masks0)
        crops1 = self.extract_mask_crops(img1, masks1)

        # Encode (returns L2-normalized embeddings)
        emb0 = self.encode_crops(crops0)
        emb1 = self.encode_crops(crops1)

        # Compute similarity
        # Since embeddings are L2-normalized (unit vectors):
        # cosine_similarity(emb0, emb1) = dot(emb0, emb1.T)
        # This is much faster than sklearn's cosine_similarity
        scores = emb0 @ emb1.T  # Matrix multiplication (dot product)

        return masks0, masks1, scores


def load_and_resize_image(image_path, target_h, target_w):
    """
    Load and resize image to target size.

    Args:
        image_path: Path to image
        target_h: Target height
        target_w: Target width

    Returns:
        PIL Image (RGB), resized
    """
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((target_w, target_h), Image.BILINEAR)
    return img_resized
