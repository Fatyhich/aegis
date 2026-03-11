"""
Mask-based image cropping with transparent background.

Critical feature: Extract regions using the actual mask shape with RGBA transparency,
not just bbox cropping. This provides better semantic representation for encoders.
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional


class MaskCropper:
    """Utility class for mask-based image cropping."""

    @staticmethod
    def crop_by_mask(
        image: Image.Image,
        mask: np.ndarray,
        bbox: Tuple[float, float, float, float],
        target_size: Optional[Tuple[int, int]] = None,
        padding: int = 0
    ) -> Image.Image:
        """
        Extract image region using mask with transparent background.

        Args:
            image: PIL Image (RGB)
            mask: Binary mask array (H, W) with True/1 for object pixels
            bbox: Bounding box (x, y, w, h)
            target_size: Optional target size (width, height) for output
            padding: Additional padding around bbox in pixels

        Returns:
            PIL Image (RGBA) with transparent background
        """
        # Convert image to numpy array
        img_array = np.array(image)

        # Extract bbox coordinates
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Apply padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.width, x + w + padding)
        y2 = min(image.height, y + h + padding)

        # Crop bbox region
        crop_rgb = img_array[y1:y2, x1:x2]

        # Crop corresponding mask region
        # Handle case where mask might be full image size or bbox size
        if mask.shape == img_array.shape[:2]:
            # Mask is full image size
            crop_mask = mask[y1:y2, x1:x2]
        else:
            # Assume mask is already cropped to bbox
            # Resize to match crop dimensions
            crop_mask = mask
            if crop_mask.shape != crop_rgb.shape[:2]:
                # Resize mask to match crop
                from PIL import Image as PILImage
                mask_img = PILImage.fromarray((crop_mask * 255).astype(np.uint8))
                mask_img = mask_img.resize((crop_rgb.shape[1], crop_rgb.shape[0]),
                                          PILImage.NEAREST)
                crop_mask = np.array(mask_img) > 0

        # Create RGBA image
        h_crop, w_crop = crop_rgb.shape[:2]
        rgba = np.zeros((h_crop, w_crop, 4), dtype=np.uint8)

        # Copy RGB channels
        rgba[:, :, :3] = crop_rgb

        # Set alpha channel based on mask
        rgba[:, :, 3] = (crop_mask * 255).astype(np.uint8)

        # Convert to PIL Image
        rgba_image = Image.fromarray(rgba, mode='RGBA')

        # Resize if target size specified
        if target_size is not None:
            rgba_image = MaskCropper._resize_with_padding(rgba_image, target_size)

        return rgba_image

    @staticmethod
    def _resize_with_padding(
        image: Image.Image,
        target_size: Tuple[int, int]
    ) -> Image.Image:
        """
        Resize image to target size with padding to maintain aspect ratio.

        Args:
            image: PIL Image (RGBA)
            target_size: Target size (width, height)

        Returns:
            PIL Image (RGBA) resized with transparent padding
        """
        target_w, target_h = target_size

        # Calculate scaling factor to fit within target size
        scale = min(target_w / image.width, target_h / image.height)

        # Resize maintaining aspect ratio
        new_w = int(image.width * scale)
        new_h = int(image.height * scale)
        resized = image.resize((new_w, new_h), Image.LANCZOS)

        # Create transparent canvas
        canvas = Image.new('RGBA', target_size, (0, 0, 0, 0))

        # Paste resized image in center
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))

        return canvas

    @staticmethod
    def prepare_for_encoder(
        image: Image.Image,
        mask: np.ndarray,
        bbox: Tuple[float, float, float, float],
        encoder_type: str,
        target_size: Optional[Tuple[int, int]] = None,
        padding: int = 0
    ) -> Image.Image:
        """
        Prepare masked crop for specific encoder type.

        Args:
            image: PIL Image (RGB)
            mask: Binary mask array
            bbox: Bounding box (x, y, w, h)
            encoder_type: 'dinov2' or 'naradio'
            target_size: Optional override for target size
            padding: Additional padding around bbox

        Returns:
            PIL Image (RGBA) ready for encoder
        """
        # Default target sizes for encoders
        default_sizes = {
            'dinov2': (224, 224),
            'naradio': (512, 512),
            'florence': None  # Florence handles dynamic sizes
        }

        if target_size is None:
            target_size = default_sizes.get(encoder_type.lower())

        return MaskCropper.crop_by_mask(
            image, mask, bbox, target_size, padding
        )

    @staticmethod
    def rgba_to_rgb(image: Image.Image, background_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """
        Convert RGBA image to RGB by compositing on background color.

        Args:
            image: PIL Image (RGBA)
            background_color: RGB tuple for background (default: white)

        Returns:
            PIL Image (RGB)
        """
        if image.mode != 'RGBA':
            return image.convert('RGB')

        # Create background
        bg = Image.new('RGB', image.size, background_color)

        # Composite RGBA over background
        bg.paste(image, mask=image.split()[3])  # Use alpha channel as mask

        return bg
