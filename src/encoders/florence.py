"""
Florence-2 model for image captioning.
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM


def load_florence_model(model_name="microsoft/Florence-2-large-ft",
                        device="cuda",
                        dtype=torch.float16):
    """
    Load Florence-2 model and processor.

    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ('cuda', 'cpu', 'mps')
        dtype: Model dtype (torch.float16, torch.float32, torch.bfloat16)

    Returns:
        Tuple of (model, processor)
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager"
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    return model, processor


def get_florence_caption(image_crop, model, processor, device,
                         task="<CAPTION>"):
    """
    Generate caption for an image crop using Florence-2.

    Args:
        image_crop: PIL Image to caption
        model: Florence-2 model
        processor: Florence-2 processor
        device: Device where model is loaded
        task: Florence-2 task type (default: "<CAPTION>")

    Returns:
        str: Generated caption
    """
    inputs = processor(text=task, images=image_crop, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3,
        use_cache=False
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text, task=task,
        image_size=(image_crop.width, image_crop.height)
    )

    return parsed[task]
