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
    # For on-demand qfloat8 on Windows we will keep the transformer on CPU until after quantization.
    transformer.to(device)

    # Enable gradient checkpointing for Qwen transformer when requested
    # We explicitly target the `transformer_blocks` module, as this was identified as the robust
    # solution from analyzing other working training implementations.
    if args is not None and getattr(args, "gradient_checkpointing", False):
        if hasattr(transformer, "transformer_blocks") and isinstance(transformer.transformer_blocks, torch.nn.ModuleList):
            logger.info("Enabling gradient checkpointing on Qwen transformer_blocks")
            # This is the more robust way to enable it for this architecture
            transformer.transformer_blocks.gradient_checkpointing_enable()
        else:
            # Fallback for older diffusers versions or different model structures
            logger.info("Enabling gradient checkpointing on Qwen transformer (fallback)")
            transformer.enable_gradient_checkpointing()

    if args is not None and getattr(args, "use_qfloat8_on_demand", False):
        logger.info("Quantizing Qwen transformer on demand to qfloat8")
        transformer = quantize_qwen_transformer_on_demand(transformer, accelerator.device, torch_dtype)
        # Defer warmup until after first explicit move to CUDA to avoid hidden device transitions here.
        try:
            setattr(transformer, "_defer_move_device", accelerator.device)
            setattr(transformer, "_defer_move_dtype", torch_dtype)
            logger.info("Deferred moving quantized transformer to CUDA until first forward.")
        except Exception:
            pass

        # On Windows with qfloat8, disable gradient checkpointing for the transformer to avoid potential deadlocks
        try:
            import os as _os
            if _os.name == "nt":
                # Also disable torch.compile on Windows which can interact poorly with quanto quantization
                try:
                    if hasattr(torch, "_dynamo"):
                        torch._dynamo.config.suppress_errors = True
                        torch._dynamo.reset()
                except Exception:
                    pass
                if hasattr(transformer, "gradient_checkpointing_disable"):
                    transformer.gradient_checkpointing_disable()
                elif hasattr(transformer, "gradient_checkpointing"):
                    transformer.gradient_checkpointing = False
                logger.info("Disabled gradient checkpointing (and reset torch.compile) for Qwen transformer due to --use_qfloat8_on_demand on Windows.")
        except Exception as e:
            logger.warning(f"Failed to disable gradient checkpointing (non-fatal): {e}")

    return transformer


def quantize_qwen_transformer_on_demand(transformer, device, dtype):
    try:
        from quanto import quantize, qfloat8, freeze
        from tqdm import tqdm
    except ImportError:
        raise ImportError("optimum and quanto are required for on-demand quantization. Please install them.")

    logger.info("Gathering modules for on-demand quantization...")
    modules_to_quantize = []
    modules_to_quantize.extend(list(transformer.transformer_blocks))

    other_module_names = ["to_patch_embedding", "pos_embed", "norm_out", "to_final_layer"]
    for name in other_module_names:
        if hasattr(transformer, name):
            module = getattr(transformer, name)
            if module is not None:
                modules_to_quantize.append(module)

    # Perform quantization on CPU to avoid CUDA deadlocks on Windows, then move back to target device
    target_device = device
    transformer.to("cpu")

    logger.info(f"Quantizing {len(modules_to_quantize)} modules to qfloat8 on device: cpu...")
    # Ensure eval mode to avoid hooks/state changes during quantization which may deadlock on Windows + CUDA
    was_training = transformer.training
    try:
        transformer.eval()
        for module in tqdm(modules_to_quantize, desc="Quantizing modules"):
            quantize(module, weights=qfloat8)
            freeze(module)
    finally:
        if was_training:
            transformer.train()

    # Defer moving the quantized model to CUDA to avoid Windows hangs; keep on CPU until first forward
    try:
        logger.info("Quantization complete. Keeping transformer on CPU to defer CUDA initialization.")
    except Exception:
        pass
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
    # Normalize modules in case lists/tuples were passed from callers
    if isinstance(text_encoder, (list, tuple)):
        if len(text_encoder) == 0:
            raise ValueError("text_encoder list/tuple is empty; expected a text encoder module")
        text_encoder = text_encoder[0]
    if isinstance(unet, (list, tuple)):
        if len(unet) == 0:
            raise ValueError("unet list/tuple is empty; expected a transformer module")
        unet = unet[0]
    if isinstance(tokenizer, (list, tuple)):
        if len(tokenizer) == 0:
            tokenizer = None
        else:
            tokenizer = tokenizer[0]

    # Unwrap models like sd3/flux to operate on base modules during sampling
    try:
        unet = accelerator.unwrap_model(unet)
    except Exception:
        pass
    try:
        if text_encoder is not None:
            text_encoder = accelerator.unwrap_model(text_encoder)
    except Exception:
        pass

    # Temporarily switch UNet to eval and optionally disable grad checkpointing during sampling (sd3/flux style)
    was_training = getattr(unet, "training", False)
    ckpt_prev_state = None
    try:
        if hasattr(unet, "gradient_checkpointing_disable"):
            # Some diffusers models expose enable/disable methods
            ckpt_prev_state = True
            try:
                if hasattr(unet, "gradient_checkpointing"):  # remember bool flag if present
                    ckpt_prev_state = bool(getattr(unet, "gradient_checkpointing"))
            except Exception:
                pass
            try:
                unet.gradient_checkpointing_disable()
            except Exception:
                pass
        elif hasattr(unet, "gradient_checkpointing"):
            # Fall back to toggling the attribute
            try:
                ckpt_prev_state = bool(unet.gradient_checkpointing)
                unet.gradient_checkpointing = False
            except Exception:
                pass
    except Exception:
        pass

    try:
        unet.eval()
    except Exception:
        pass

    pipeline = QwenImagePipeline(
        vae=vae,
        text_encoder=text_encoder,
        transformer=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )
    # Ensure prompt max length during sampling matches training clamp to avoid internal cache shape drift
    try:
        if hasattr(pipeline, "encode_prompt"):
            pipeline.default_max_sequence_length = 512  # custom attribute used by our wrapper below if any
    except Exception:
        pass
    # Avoid moving the quantized transformer; move only VAE and text encoder when on-demand qfloat8 is used.
    if getattr(args, "use_qfloat8_on_demand", False):
        vae.to(accelerator.device)
        text_encoder.to(accelerator.device)
    else:
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
    # Explicitly delete pipeline to free any references promptly
    del pipeline

    # Attempt to clear any internal caches/state in the Qwen transformer to avoid shape drift after sampling
    try:
        # Common patterns across diffusers models
        if hasattr(unet, "clear_kv_cache") and callable(getattr(unet, "clear_kv_cache")):
            unet.clear_kv_cache()
        if hasattr(unet, "_clear_cache") and callable(getattr(unet, "_clear_cache")):
            unet._clear_cache()
        # Some implementations keep attention caches or rotary caches per-block
        if hasattr(unet, "transformer_blocks"):
            for blk in unet.transformer_blocks:
                for attr_name in ("kv_cache", "_kv_cache", "attn_cache", "cache", "_attn_bias", "attn_bias"):
                    if hasattr(blk, attr_name):
                        try:
                            setattr(blk, attr_name, None)
                        except Exception:
                            pass
    except Exception:
        pass

    # Restore UNet states after sampling
    try:
        if was_training:
            unet.train()
    except Exception:
        pass
    try:
        if ckpt_prev_state is not None:
            if hasattr(unet, "gradient_checkpointing_enable") and ckpt_prev_state:
                try:
                    unet.gradient_checkpointing_enable()
                except Exception:
                    pass
            elif hasattr(unet, "gradient_checkpointing"):
                try:
                    unet.gradient_checkpointing = bool(ckpt_prev_state)
                except Exception:
                    pass
    except Exception:
        pass

    train_util.clean_memory_on_device(accelerator.device)
