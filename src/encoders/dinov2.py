"""
DINOv2 visual encoder for extracting image embeddings.
"""

import torch
from transformers import AutoImageProcessor, AutoModel


def load_dinov2_model(model_name="facebook/dinov2-base", device="cuda"):
    """
    Load DINOv2 model and processor.

    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ('cuda', 'cpu', 'mps')

    Returns:
        Tuple of (model, processor)
    """
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    return model, processor


def get_dinov2_embedding(image_crop, model, processor, device):
    """
    Extract CLS token embedding from DINOv2 for a single image crop.

    Args:
        image_crop: PIL Image
        model: DINOv2 model
        processor: DINOv2 processor
        device: Device where model is loaded

    Returns:
        numpy array of shape (D,) - embedding vector
    """
    inputs = processor(images=image_crop, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract CLS token (first token in sequence)
    return outputs.last_hidden_state[0, 0].cpu().numpy()


def get_dinov2_embeddings_batch(image_crops, model, processor, device, batch_size=8):
    """
    Extract embeddings for multiple image crops in batches.

    Args:
        image_crops: List of PIL Images
        model: DINOv2 model
        processor: DINOv2 processor
        device: Device where model is loaded
        batch_size: Number of images to process at once

    Returns:
        List of numpy arrays - embedding vectors
    """
    embeddings = []

    for i in range(0, len(image_crops), batch_size):
        batch = image_crops[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # CLS tokens for entire batch
        batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
        embeddings.extend(batch_embeddings)

    return embeddings
