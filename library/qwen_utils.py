import os
import random
import torch
from diffusers import (
    QwenImagePipeline,
    AutoencoderKLQwenImage,
    QwenImageTransformer2DModel,
)
from transformers import Qwen2Model, CLIPTokenizer
from PIL import Image
import numpy as np
from . import train_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def load_qwen_text_encoder_and_tokenizer(model_name_or_path, torch_dtype, device):
    logger.info("Loading Qwen2Model (Text Encoder) and CLIPTokenizer")
    text_encoder = Qwen2Model.from_pretrained(
        model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        model_name_or_path,
        subfolder="tokenizer"
    )
    text_encoder.to(device)
    return text_encoder, tokenizer


def load_qwen_vae(
    model_name_or_path,
    torch_dtype,
    device,
):
    logger.info("Loading AutoencoderKLQwenImage")
    vae = AutoencoderKLQwenImage.from_pretrained(
        model_name_or_path,
        subfolder="vae",
        torch_dtype=torch_dtype,
    )
    vae.to(device)
    return vae


def load_qwen_transformer(
    model_name_or_path,
    torch_dtype,
    device,
):
    logger.info("Loading QwenImageTransformer2DModel")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch_dtype,
    )
    transformer.to(device)
    return transformer


def sample_images(accelerator, args, epoch, global_step, text_encoder, vae, unet, tokenizer):
    if not args.sample_prompts:
        return

    logger.info(f"Generating samples for epoch {epoch} step {global_step}")

    # Create a new pipeline for sampling
    # We don't need the scheduler from the training arguments, the pipeline will create its own.
    pipeline = QwenImagePipeline(
        vae=vae,
        text_encoder=text_encoder,
        transformer=unet,
        tokenizer=tokenizer,
        scheduler=None, # Will be created internally
    )
    pipeline.to(accelerator.device)

    prompts = train_util.load_prompts(args.sample_prompts)
    with torch.no_grad(), accelerator.autocast():
        for i, prompt_data in enumerate(prompts):
            prompt = prompt_data.get("prompt", "")
            negative_prompt = prompt_data.get("negative_prompt", "")
            seed = prompt_data.get("seed")
            if seed is None:
                seed = random.randint(0, 2**32 - 1)

            generator = torch.Generator(device=accelerator.device).manual_seed(seed)

            logger.info(f"Generating image for prompt: {prompt}")

            steps = prompt_data.get("steps", 30)
            guidance_scale = prompt_data.get("guidance_scale", 7.5)

            image = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]

            # save image
            output_dir = os.path.join(args.output_dir, "sample")
            os.makedirs(output_dir, exist_ok=True)

            filename = f"epoch-{epoch:06d}-step-{global_step:06d}-{i:02d}-{seed}.png"
            image.save(os.path.join(output_dir, filename))

    pipeline.to("cpu")
    train_util.clean_memory_on_device(accelerator.device)
