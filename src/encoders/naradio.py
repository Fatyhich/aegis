"""
NaRadIO (RADIO + NACLIP) visual encoder with language alignment.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_naradio_encoder(project_dir,
                         input_resolution=(512, 512),
                         model_version="radio_v2.5-b",
                         lang_model="siglip",
                         device="cuda"):
    """
    Load NaRadIO encoder from RayFronts repository.

    Args:
        project_dir: Path to project root (where RayFronts is located)
        input_resolution: Tuple (height, width) for input images
        model_version: RADIO model version ("radio_v2.5-b", "radio_v2.5-l", "radio_v2.5-g")
        lang_model: Language model to use ("clip" or "siglip")
        device: Device to load model on ('cuda', 'cpu', 'mps')

    Returns:
        NARadioEncoder instance
    """
    # Add RayFronts to Python path
    rayfronts_path = Path(project_dir) / "RayFronts"
    if str(rayfronts_path) not in sys.path:
        sys.path.insert(0, str(rayfronts_path))

    from rayfronts.image_encoders.naradio import NARadioEncoder

    encoder = NARadioEncoder(
        input_resolution=list(input_resolution),
        model_version=model_version,
        lang_model=lang_model,
        device=device
    )

    return encoder


def get_naradio_embedding(image_crop, encoder):
    """
    Extract global embedding vector from NaRadIO for a single image crop.

    Args:
        image_crop: PIL Image
        encoder: NARadioEncoder instance

    Returns:
        numpy array of shape (D,) - embedding vector
    """
    # NaRadIO requires fixed input size - resize crop to match encoder resolution
    target_size = tuple(encoder.input_resolution)  # (H, W)
    image_crop = image_crop.resize((target_size[1], target_size[0]))  # PIL uses (W, H)

    # Convert PIL to numpy to tensor (0-1 normalized)
    img_array = np.array(image_crop).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(encoder.device)

    with torch.no_grad():
        # encode_image_to_vector returns [B, C]
        embedding = encoder.encode_image_to_vector(img_tensor)

    return embedding[0].cpu().numpy()


def get_naradio_embeddings_batch(image_crops, encoder, batch_size=8):
    """
    Extract embeddings for multiple image crops in batches.

    Args:
        image_crops: List of PIL Images
        encoder: NARadioEncoder instance
        batch_size: Number of images to process at once

    Returns:
        List of numpy arrays - embedding vectors
    """
    # NaRadIO requires fixed input size
    target_size = tuple(encoder.input_resolution)  # (H, W)

    embeddings = []

    for i in range(0, len(image_crops), batch_size):
        batch_crops = image_crops[i:i + batch_size]

        # Convert batch of PIL images to tensor
        batch_tensors = []
        for crop in batch_crops:
            # Resize to fixed size
            crop_resized = crop.resize((target_size[1], target_size[0]))  # PIL uses (W, H)
            img_array = np.array(crop_resized).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            batch_tensors.append(img_tensor)

        batch_tensor = torch.stack(batch_tensors).to(encoder.device)  # [B, C, H, W]

        with torch.no_grad():
            batch_embeddings = encoder.encode_image_to_vector(batch_tensor)

        embeddings.extend(batch_embeddings.cpu().numpy())

    return embeddings


def naradio_zero_shot_classify(image_crop, encoder, candidate_labels):
    """
    Zero-shot classification using NaRadIO language alignment.

    Args:
        image_crop: PIL Image
        encoder: NARadioEncoder instance
        candidate_labels: List of text labels (e.g., ["person", "building", "tree"])

    Returns:
        Tuple of (best_label, confidence_score)
    """
    # NaRadIO requires fixed input size - resize crop to match encoder resolution
    target_size = tuple(encoder.input_resolution)  # (H, W)
    image_crop = image_crop.resize((target_size[1], target_size[0]))  # PIL uses (W, H)

    # Visual embedding
    img_array = np.array(image_crop).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(encoder.device)

    with torch.no_grad():
        visual_emb = encoder.encode_image_to_vector(img_tensor)
        # Align visual features with language space (768 -> 1152 for SigLIP)
        # This is needed when return_radio_features=True (default)
        visual_emb_aligned = encoder.align_global_features_with_language(visual_emb)
        text_embs = encoder.encode_labels(candidate_labels)

    # Cosine similarity between language-aligned visual and text embeddings
    similarities = cosine_similarity(visual_emb_aligned.cpu().numpy(), text_embs.cpu().numpy())[0]
    best_idx = similarities.argmax()

    return candidate_labels[best_idx], float(similarities[best_idx])
