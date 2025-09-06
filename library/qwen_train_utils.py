import argparse
import os
import time
import torch
from accelerate import Accelerator
from accelerate.state import PartialState
from diffusers import QwenImagePipeline, AutoencoderKLQwenImage, QwenImageTransformer2DModel
from PIL import Image
import numpy as np
from tqdm import tqdm

from . import train_util
from .device_utils import clean_memory_on_device

import logging

logger = logging.getLogger(__name__)


def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    transformer: QwenImageTransformer2DModel,
    vae: AutoencoderKLQwenImage,
    text_encoding_pipeline: QwenImagePipeline,
    num_update_steps_per_epoch: int,
):
    if not args.sample_prompts:
        return

    sample_this_step = False
    if global_step == 0:
        if args.sample_at_first:
            sample_this_step = True
    elif global_step > 0:
        if args.sample_every_n_epochs is not None:
            if num_update_steps_per_epoch > 0 and global_step % num_update_steps_per_epoch == 0:
                current_epoch = global_step // num_update_steps_per_epoch
                if current_epoch > 0 and current_epoch % args.sample_every_n_epochs == 0:
                    sample_this_step = True
                    logger.info(f"Sampling for epoch {current_epoch}")
        elif args.sample_every_n_steps is not None:
            if global_step % args.sample_every_n_steps == 0:
                sample_this_step = True
                logger.info(f"Sampling for step {global_step}")

    if not sample_this_step:
        return

    logger.info("")
    logger.info(f"Generating sample images at step: {global_step}")

    if not os.path.isfile(args.sample_prompts):
        logger.error(f"Sample prompts file not found: {args.sample_prompts}")
        return

    prompts = train_util.load_prompts(args.sample_prompts)
    save_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(save_dir, exist_ok=True)

    unwrapped_transformer = accelerator.unwrap_model(transformer)
    unwrapped_vae = accelerator.unwrap_model(vae)

    pipeline = QwenImagePipeline(
        transformer=unwrapped_transformer,
        vae=unwrapped_vae,
        text_encoder=text_encoding_pipeline.text_encoder,
        tokenizer=text_encoding_pipeline.tokenizer,
        scheduler=text_encoding_pipeline.scheduler,
    )
    pipeline.to(accelerator.device)

    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    with torch.no_grad(), accelerator.autocast():
        for i, prompt_dict in enumerate(prompts):
            prompt = prompt_dict.get("prompt", "")
            negative_prompt = prompt_dict.get("negative_prompt", "")
            height = prompt_dict.get("height", args.resolution[0])
            width = prompt_dict.get("width", args.resolution[1])
            seed = prompt_dict.get("seed")
            guidance_scale = prompt_dict.get("guidance_scale", 7.5)
            num_inference_steps = prompt_dict.get("num_inference_steps", 50)

            generator = torch.Generator(device=accelerator.device).manual_seed(seed) if seed is not None else None

            logger.info(f"Generating image for prompt: {prompt}")

            image = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

            ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
            num_suffix = f"e{epoch:06d}" if epoch is not None else f"{global_step:06d}"
            seed_suffix = "" if seed is None else f"_{seed}"
            img_filename = f"{args.output_name or 'sample'}_{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
            image.save(os.path.join(save_dir, img_filename))

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    del pipeline
    clean_memory_on_device(accelerator.device)
