import os
import random
import torch
from diffusers import (
    QwenImagePipeline,
    AutoencoderKLQwenImage,
    QwenImageTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import Qwen2Model, CLIPTokenizer
from PIL import Image
import numpy as np
from . import train_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def load_qwen_text_encoder(model_name_or_path, torch_dtype, device, custom_text_encoder_path=None):
    text_encoder_path = custom_text_encoder_path if custom_text_encoder_path is not None else model_name_or_path
    subfolder = "text_encoder" if custom_text_encoder_path is None else None

    logger.info(f"Loading Qwen2Model (Text Encoder) from: {text_encoder_path}")
    text_encoder = Qwen2Model.from_pretrained(
        text_encoder_path,
        subfolder=subfolder,
        torch_dtype=torch_dtype,
    )
    text_encoder.to(device)
    return text_encoder


def load_qwen_vae(
    model_name_or_path,
    torch_dtype,
    device,
    custom_vae_path=None,
):
    vae_path = custom_vae_path if custom_vae_path is not None else model_name_or_path
    subfolder = "vae" if custom_vae_path is None else None

    logger.info(f"Loading AutoencoderKLQwenImage from: {vae_path}")
    vae = AutoencoderKLQwenImage.from_pretrained(
        vae_path,
        subfolder=subfolder,
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

    # Create a new pipeline for sampling from the standalone components
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    pipeline = QwenImagePipeline(
        vae=vae,
        text_encoder=text_encoder,
        transformer=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
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
