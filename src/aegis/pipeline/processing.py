"""
Unified mask processing with configurable encoders.
Supports DINOv2 and NaRadIO visual encoders, with optional Florence-2 captions.
"""

from tqdm.auto import tqdm
from aegis.encoders.florence import get_florence_caption
from aegis.encoders.dinov2 import get_dinov2_embedding, get_dinov2_embeddings_batch
from aegis.encoders.naradio import (get_naradio_embedding, get_naradio_embeddings_batch,
                                     naradio_zero_shot_classify)
from aegis.preprocessing import MaskCropper

DEFAULT_EXCLUDED_CATEGORIES = {
    "pedestrian", "person", "human", "walker", "man", "woman",
    "shadow", "people", "child", "kid"
}


def process_masks_with_features(image, masks, config, models):
    encoder_type = config.get('encoder', 'dinov2')

    crops = []
    for mask_data in masks:
        crop = MaskCropper.prepare_for_encoder(
            image=image,
            mask=mask_data['segmentation'],
            bbox=mask_data['bbox'],
            encoder_type=encoder_type,
            padding=0
        )
        crops.append(crop)

    if config.get('use_florence', False) or config.get('use_zero_shot', False):
        print("Generating descriptions...")
        _generate_descriptions(masks, crops, config, models)

    print("Extracting embeddings...")
    _extract_embeddings(masks, crops, config, models)

    return masks


def _extract_category_from_caption(caption):
    if not caption:
        return 'unknown'

    category_keywords = [
        'building', 'tree', 'grass', 'ground', 'sky', 'wall', 'floor',
        'ceiling', 'window', 'door', 'column', 'statue', 'bench',
        'car', 'vehicle', 'bicycle', 'road', 'path', 'sidewalk',
        'water', 'fountain', 'person', 'people', 'shadow', 'sign',
        'pole', 'lamp', 'light', 'plant', 'bush', 'flower', 'rock',
        'stone', 'concrete', 'brick', 'metal', 'glass', 'pavement'
    ]

    caption_lower = caption.lower()
    for keyword in category_keywords:
        if keyword in caption_lower:
            return keyword

    words = caption_lower.split()
    skip_words = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'with', 'and', 'is', 'are'}
    for word in words:
        word = word.strip('.,!?;:')
        if word and word not in skip_words and len(word) > 2:
            return word

    return 'object'


def _generate_descriptions(masks, crops, config, models):
    use_florence = config.get('use_florence', False)
    use_zero_shot = config.get('use_zero_shot', False)
    encoder_type = config['encoder']

    zero_shot_labels = config.get('zero_shot_labels', [
        "person", "building", "tree", "grass", "sky", "ground",
        "column", "statue", "bench", "wall", "window", "door",
        "car", "bicycle", "road", "path", "water", "fountain"
    ])

    for mask_data, crop in tqdm(zip(masks, crops), total=len(masks)):
        descriptions = []

        if use_zero_shot and encoder_type == 'naradio':
            category, confidence = naradio_zero_shot_classify(
                crop, models['visual_encoder'], zero_shot_labels
            )
            mask_data['category'] = category
            mask_data['category_confidence'] = confidence
            descriptions.append(f"{category} ({confidence:.2f})")

        if use_florence:
            caption = get_florence_caption(
                crop,
                models['florence_model'],
                models['florence_processor'],
                models['florence_device']
            )
            mask_data['caption'] = caption
            descriptions.append(caption)

            if 'category' not in mask_data:
                mask_data['category'] = _extract_category_from_caption(caption)

        if use_zero_shot and use_florence and encoder_type == 'naradio':
            mask_data['description'] = f"{category}: {caption}"
        elif descriptions:
            mask_data['description'] = " | ".join(descriptions)
        else:
            mask_data['description'] = "no description"

        if 'category' not in mask_data:
            mask_data['category'] = 'unknown'


def _extract_embeddings(masks, crops, config, models):
    encoder_type = config['encoder']
    use_batch = config.get('use_batch', False)
    batch_size = config.get('batch_size', 8)

    if encoder_type == 'dinov2':
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
    if not config.get('filtering', False):
        return masks

    img_width, img_height = image_size
    img_area = img_width * img_height

    excluded = set(config.get('excluded_categories', DEFAULT_EXCLUDED_CATEGORIES))
    min_ratio = config.get('min_mask_ratio', 0.10)

    filtered = []
    removed_by_category = 0
    removed_by_size = 0

    for mask in masks:
        category = mask.get('category', '').lower()
        if category in excluded:
            removed_by_category += 1
            continue

        x, y, w, h = mask['bbox']
        mask_area = w * h
        if mask_area / img_area < min_ratio:
            removed_by_size += 1
            continue

        filtered.append(mask)

    print(f"Filtering: {len(masks)} -> {len(filtered)} masks "
          f"(removed {removed_by_category} by category, {removed_by_size} by size)")

    return filtered
