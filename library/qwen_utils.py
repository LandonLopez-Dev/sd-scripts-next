import os
import random
import math
import torch
from diffusers import (
    QwenImagePipeline,
    AutoencoderKLQwenImage,
    QwenImageTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import AutoModel, Qwen2Model, CLIPTokenizer
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

    logger.info(f"Loading Qwen-VL Model from: {text_encoder_path}")
    qwen_vl = AutoModel.from_pretrained(
        text_encoder_path,
        subfolder=subfolder,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    text_encoder = qwen_vl.language_model
    text_encoder.to(device)

    # Manually move the rotary embedding buffer to the correct device, as it might not be moved automatically.
    # This is a workaround for a potential issue in the Qwen model implementation when used with gradient checkpointing.
    if hasattr(text_encoder, "rotary_emb") and hasattr(text_encoder.rotary_emb, "inv_freq"):
        text_encoder.rotary_emb.inv_freq = text_encoder.rotary_emb.inv_freq.to(device)

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
    accelerator=None,
    args=None,
):
    logger.info("Loading QwenImageTransformer2DModel")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch_dtype,
    )
    transformer.to(device)

    # Enable gradient checkpointing for Qwen transformer when requested or when using grad accumulation
    try:
        enable_ckpt = False
        if accelerator is not None and getattr(accelerator, "gradient_accumulation_steps", 1) > 1:
            enable_ckpt = True
        if args is not None and getattr(args, "gradient_checkpointing", False):
            enable_ckpt = True
        if enable_ckpt:
            # Match HF convention if available
            if hasattr(transformer, "enable_input_require_grads"):
                transformer.enable_input_require_grads()
            # diffusers models often expose gradient_checkpointing_enable
            if hasattr(transformer, "gradient_checkpointing_enable"):
                transformer.gradient_checkpointing_enable()
            else:
                # Fallback: set a common flag if present
                if hasattr(transformer, "gradient_checkpointing"):
                    transformer.gradient_checkpointing = True
            logger.info("Enabled gradient checkpointing for Qwen transformer")
    except Exception as e:
        logger.warning(f"Failed to enable gradient checkpointing for Qwen transformer: {e}")

    if args is not None and getattr(args, "use_qfloat8_on_demand", False):
        logger.info("Quantizing Qwen transformer on demand to qfloat8")
        transformer = quantize_qwen_transformer_on_demand(transformer, accelerator.device, torch_dtype)

    return transformer


def quantize_qwen_transformer_on_demand(transformer, device, dtype):
    try:
        from optimum.quanto import quantize, qfloat8, freeze
        from tqdm import tqdm
    except ImportError:
        raise ImportError("optimum and quanto are required for on-demand quantization. Please install them.")

    logger.info("Quantizing transformer blocks to qfloat8...")
    all_blocks = list(transformer.transformer_blocks)
    for block in tqdm(all_blocks, desc="Quantizing blocks"):
        block.to(device, dtype=dtype)
        quantize(block, weights=qfloat8)
        freeze(block)
        block.to("cpu")

    logger.info("Quantizing top-level transformer...")
    transformer.to(device, dtype=dtype)
    quantize(transformer, weights=qfloat8)
    freeze(transformer)

    logger.info("Quantization complete.")
    return transformer


def sample_images(accelerator, args, epoch, global_step, text_encoder, vae, unet, tokenizer):
    # Align sampling cadence with SD3/FLUX
    if not args.sample_prompts:
        return

    steps = global_step
    if steps == 0:
        if not getattr(args, "sample_at_first", False):
            return
    else:
        if getattr(args, "sample_every_n_steps", None) is None and getattr(args, "sample_every_n_epochs", None) is None:
            return
        if getattr(args, "sample_every_n_epochs", None) is not None:
            # ignore sample_every_n_steps when epoch-based sampling is configured
            if epoch is None or epoch % args.sample_every_n_epochs != 0:
                return
        else:
            # Only sample during training steps (not end-of-epoch) and at the configured interval
            if steps % args.sample_every_n_steps != 0 or epoch is not None:
                return

    logger.info("")
    logger.info(f"generating sample images at step / サンプル画像生成 ステップ: {steps}")

    # Create a new pipeline for sampling from the standalone components
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
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
            prompt = prompt_data.get("prompt")
            negative_prompt = prompt_data.get("negative_prompt", "")
            seed = prompt_data.get("seed")
            if seed is None:
                seed = random.randint(0, 2**32 - 1)

            generator = torch.Generator(device=accelerator.device).manual_seed(seed)

            logger.info(f"Generating image for prompt: {prompt}")

            num_infer_steps = prompt_data.get("steps", 25)
            guidance_scale = prompt_data.get("guidance_scale", 4.0)

            image = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_infer_steps,
                true_cfg_scale=guidance_scale,
                generator=generator,
            ).images[0]

            # save image
            output_dir = os.path.join(args.output_dir, "sample")
            os.makedirs(output_dir, exist_ok=True)

            num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
            filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{seed}.png"
            image.save(os.path.join(output_dir, filename))

    pipeline.to("cpu")
    train_util.clean_memory_on_device(accelerator.device)
