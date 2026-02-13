"""
Unified mask processing with configurable encoders.
Supports DINOv2 and NaRadIO visual encoders, with optional Florence-2 captions.

IMPORTANT: Uses mask-based cropping with transparent background (RGBA)
instead of simple bbox cropping for better semantic representation.
"""

import sys
from pathlib import Path
from tqdm.auto import tqdm
from encoders.florence import get_florence_caption
from encoders.dinov2 import get_dinov2_embedding, get_dinov2_embeddings_batch
from encoders.naradio import (get_naradio_embedding, get_naradio_embeddings_batch,
                               naradio_zero_shot_classify)

# Import MaskCropper from vizenc_baseline
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from vizenc_baseline.preprocessing import MaskCropper
    USE_MASK_CROPPING = True
except ImportError:
    print("Warning: MaskCropper not found, falling back to bbox cropping")
    USE_MASK_CROPPING = False

# Default categories to filter out (people, shadows, etc.)
DEFAULT_EXCLUDED_CATEGORIES = {
    "pedestrian", "person", "human", "walker", "man", "woman",
    "shadow", "people", "child", "kid"
}


def process_masks_with_features(image, masks, config, models):
    """
    Universal mask processing function supporting multiple encoder configurations.

    IMPORTANT: Now uses mask-based cropping with transparent background (RGBA)
    instead of simple bbox cropping. This provides better semantic representation.

    Args:
        image: PIL Image
        masks: List of mask dictionaries from SAM
        config: Configuration dictionary with keys:
            - encoder: 'dinov2' or 'naradio'
            - use_florence: bool
            - use_zero_shot: bool (NaRadIO only)
            - zero_shot_labels: list of str (NaRadIO only)
            - use_batch: bool
            - batch_size: int
        models: Dictionary with model instances:
            - florence_model: Florence-2 model (if use_florence=True)
            - florence_processor: Florence-2 processor
            - florence_device: device
            - visual_encoder: DINOv2 model or NaRadIO encoder
            - visual_processor: DINOv2 processor (if encoder='dinov2')
            - visual_device: device (if encoder='dinov2')

    Returns:
        List of masks with added 'description' and 'embedding' fields
    """
    encoder_type = config.get('encoder', 'dinov2')

    # Extract crops using mask-based cropping
    crops = []
    for mask_data in masks:
        if USE_MASK_CROPPING:
            # Use mask-based cropping with transparent background
            crop = MaskCropper.prepare_for_encoder(
                image=image,
                mask=mask_data['segmentation'],
                bbox=mask_data['bbox'],
                encoder_type=encoder_type,
                padding=0
            )
        else:
            # Fallback to simple bbox cropping
            x, y, w, h = mask_data['bbox']
            crop = image.crop((x, y, x + w, y + h))

        crops.append(crop)

    # Generate descriptions
    if config.get('use_florence', False) or config.get('use_zero_shot', False):
        print("Generating descriptions...")
        _generate_descriptions(masks, crops, config, models)

    # Extract visual embeddings
    print("Extracting embeddings...")
    _extract_embeddings(masks, crops, config, models)

    return masks


def _extract_category_from_caption(caption):
    """
    Extract category from Florence-2 caption.

    Examples:
        "A red building with windows" -> "building"
        "The grassy ground" -> "ground"
        "A tall tree in the park" -> "tree"
    """
    if not caption:
        return 'unknown'

    # Common category keywords to look for
    category_keywords = [
        'building', 'tree', 'grass', 'ground', 'sky', 'wall', 'floor',
        'ceiling', 'window', 'door', 'column', 'statue', 'bench',
        'car', 'vehicle', 'bicycle', 'road', 'path', 'sidewalk',
        'water', 'fountain', 'person', 'people', 'shadow', 'sign',
        'pole', 'lamp', 'light', 'plant', 'bush', 'flower', 'rock',
        'stone', 'concrete', 'brick', 'metal', 'glass', 'pavement'
    ]

    caption_lower = caption.lower()

    # Find first matching keyword
    for keyword in category_keywords:
        if keyword in caption_lower:
            return keyword

    # Fallback: use first noun-like word (simple heuristic)
    words = caption_lower.split()
    # Skip articles and prepositions
    skip_words = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'with', 'and', 'is', 'are'}
    for word in words:
        # Clean word
        word = word.strip('.,!?;:')
        if word and word not in skip_words and len(word) > 2:
            return word

    return 'object'


def _generate_descriptions(masks, crops, config, models):
    """Generate descriptions using Florence-2 and/or NaRadIO zero-shot."""
    use_florence = config.get('use_florence', False)
    use_zero_shot = config.get('use_zero_shot', False)
    encoder_type = config['encoder']

    # Zero-shot labels for NaRadIO
    zero_shot_labels = config.get('zero_shot_labels', [
        "person", "building", "tree", "grass", "sky", "ground",
        "column", "statue", "bench", "wall", "window", "door",
        "car", "bicycle", "road", "path", "water", "fountain"
    ])

    for mask_data, crop in tqdm(zip(masks, crops), total=len(masks)):
        descriptions = []

        # Zero-shot classification (NaRadIO only)
        if use_zero_shot and encoder_type == 'naradio':
            category, confidence = naradio_zero_shot_classify(
                crop, models['visual_encoder'], zero_shot_labels
            )
            mask_data['category'] = category
            mask_data['category_confidence'] = confidence
            descriptions.append(f"{category} ({confidence:.2f})")

        # Florence-2 caption
        if use_florence:
            caption = get_florence_caption(
                crop,
                models['florence_model'],
                models['florence_processor'],
                models['florence_device']
            )
            mask_data['caption'] = caption
            descriptions.append(caption)

            # Extract category from caption (first noun/word) if no zero-shot
            if 'category' not in mask_data:
                # Simple extraction: first word of caption
                category_from_caption = _extract_category_from_caption(caption)
                mask_data['category'] = category_from_caption

        # Combined description
        if use_zero_shot and use_florence and encoder_type == 'naradio':
            mask_data['description'] = f"{category}: {caption}"
        elif descriptions:
            mask_data['description'] = " | ".join(descriptions)
        else:
            mask_data['description'] = "no description"

        # Set default category if still missing
        if 'category' not in mask_data:
            mask_data['category'] = 'unknown'


def _extract_embeddings(masks, crops, config, models):
    """Extract visual embeddings using DINOv2 or NaRadIO."""
    encoder_type = config['encoder']
    use_batch = config.get('use_batch', False)
    batch_size = config.get('batch_size', 8)

    if encoder_type == 'dinov2':
        # DINOv2 embeddings
        if use_batch:
            embeddings = get_dinov2_embeddings_batch(
                crops,
                models['visual_encoder'],
                models['visual_processor'],
                models['visual_device'],
                batch_size
            )
            for mask_data, embedding in zip(masks, embeddings):
                mask_data['embedding'] = embedding
        else:
            for mask_data, crop in tqdm(zip(masks, crops), total=len(masks)):
                mask_data['embedding'] = get_dinov2_embedding(
                    crop,
                    models['visual_encoder'],
                    models['visual_processor'],
                    models['visual_device']
                )

    elif encoder_type == 'naradio':
        # NaRadIO embeddings
        if use_batch:
            embeddings = get_naradio_embeddings_batch(
                crops,
                models['visual_encoder'],
                batch_size
            )
            for mask_data, embedding in zip(masks, embeddings):
                mask_data['embedding'] = embedding
        else:
            for mask_data, crop in tqdm(zip(masks, crops), total=len(masks)):
                mask_data['embedding'] = get_naradio_embedding(
                    crop,
                    models['visual_encoder']
                )

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Use 'dinov2' or 'naradio'.")


def filter_masks(masks, image_size, config):
    """
    Filter masks based on category and size.

    Args:
        masks: List of mask dictionaries with 'category' and 'bbox' fields
        image_size: Tuple (width, height) of the original image
        config: Configuration dictionary with keys:
            - filtering: bool - enable/disable filtering
            - excluded_categories: set/list of categories to filter out
            - min_mask_ratio: float - minimum mask area as fraction of image (0.0-1.0)

    Returns:
        Filtered list of masks
    """
    if not config.get('filtering', False):
        return masks

    img_width, img_height = image_size
    img_area = img_width * img_height

    # Get filter settings
    excluded = set(config.get('excluded_categories', DEFAULT_EXCLUDED_CATEGORIES))
    min_ratio = config.get('min_mask_ratio', 0.10)  # 10% by default

    filtered = []
    removed_by_category = 0
    removed_by_size = 0

    for mask in masks:
        # Filter by category (if available)
        category = mask.get('category', '').lower()
        if category in excluded:
            removed_by_category += 1
            continue

        # Filter by size
        x, y, w, h = mask['bbox']
        mask_area = w * h
        # print(f"DEBUG mask_area = {mask_area} and img_area = {img_area}")
        if mask_area / img_area < min_ratio:
            removed_by_size += 1
            continue

        filtered.append(mask)

    print(f"Filtering: {len(masks)} -> {len(filtered)} masks "
          f"(removed {removed_by_category} by category, {removed_by_size} by size)")

    return filtered
