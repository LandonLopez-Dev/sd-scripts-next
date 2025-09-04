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
from tqdm import tqdm

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, height // 2 * width // 2, num_channels_latents * 4)
    return latents


def _unpack_latents(latents, batch_size, height, width, num_channels_latents):
    h = height // 2
    w = width // 2
    latents = latents.view(batch_size, h, w, num_channels_latents, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, num_channels_latents, height, width)
    latents = latents.unsqueeze(1)
    return latents


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
    logger.info(f"generating sample images at step {global_step} (manual loop)")

    # Unwrap models
    try:
        unet = accelerator.unwrap_model(unet)
        vae = accelerator.unwrap_model(vae)
        if text_encoder is not None:
            text_encoder = accelerator.unwrap_model(text_encoder)
    except Exception:
        pass

    # Normalize modules in case lists/tuples were passed from callers
    if isinstance(text_encoder, (list, tuple)):
        text_encoder = text_encoder[0] if len(text_encoder) > 0 else None
    if isinstance(unet, (list, tuple)):
        unet = unet[0] if len(unet) > 0 else None
    if isinstance(tokenizer, (list, tuple)):
        tokenizer = tokenizer[0] if len(tokenizer) > 0 else None
    if unet is None or tokenizer is None or text_encoder is None or vae is None:
        logger.error("unet, tokenizer, text_encoder, or vae is None, cannot generate sample images.")
        return

    # Model state management

    # Move models to GPU for sampling
    unet.to(accelerator.device)
    text_encoder.to(accelerator.device)

    # Scheduler setup
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

    prompts = train_util.load_prompts(args.sample_prompts)
    with torch.no_grad(), accelerator.autocast():
        for i, prompt_data in enumerate(prompts):
            prompt = prompt_data.get("prompt", "")
            negative_prompt = prompt_data.get("negative_prompt", "")
            seed = prompt_data.get("seed", random.randint(0, 2**32 - 1))
            num_inference_steps = prompt_data.get("steps", 25)
            guidance_scale = prompt_data.get("guidance_scale", 4.0)
            height = prompt_data.get("height", 1024)
            width = prompt_data.get("width", 1024)
            generator = torch.Generator(device=accelerator.device).manual_seed(seed)

            logger.info(f"Generating image for prompt: {prompt}")

            # Prompt Encoding
            max_len = tokenizer.model_max_length if hasattr(tokenizer, "model_max_length") else 512
            text_inputs = tokenizer(
                [prompt], padding="max_length", max_length=max_len, truncation=True, return_tensors="pt"
            )
            cond_input_ids = text_inputs.input_ids.to(accelerator.device)
            cond_attention_mask = text_inputs.attention_mask.to(accelerator.device)
            cond_embeds = text_encoder(cond_input_ids, attention_mask=cond_attention_mask)[0]

            uncond_inputs = tokenizer(
                [negative_prompt], padding="max_length", max_length=max_len, truncation=True, return_tensors="pt"
            )
            uncond_input_ids = uncond_inputs.input_ids.to(accelerator.device)
            uncond_attention_mask = uncond_inputs.attention_mask.to(accelerator.device)
            uncond_embeds = text_encoder(uncond_input_ids, attention_mask=uncond_attention_mask)[0]

            prompt_embeds = torch.cat([uncond_embeds, cond_embeds])
            prompt_embeds_mask = torch.cat([uncond_attention_mask, cond_attention_mask])

            # Scheduler timesteps
            scheduler.set_timesteps(num_inference_steps, device=accelerator.device, mu=math.log(3))

            # Latent preparation
            vae_scale_factor = vae.config.scale_factor if hasattr(vae, "config") and hasattr(vae.config, "scale_factor") else 8
            latent_height = height // vae_scale_factor
            latent_width = width // vae_scale_factor
            num_channels_latents = vae.config.latent_channels
            shape = (1, 1, num_channels_latents, latent_height, latent_width)
            latents = torch.randn(shape, generator=generator, device=accelerator.device, dtype=unet.dtype)

            if hasattr(scheduler, "init_noise_sigma"):
                latents = latents * scheduler.init_noise_sigma

            # Denoising loop
            for t in tqdm(scheduler.timesteps):
                latent_model_input = torch.cat([latents] * 2)

                packed_latents = _pack_latents(
                    latent_model_input, 2, num_channels_latents, latent_height, latent_width
                )

                img_shapes = [[(1, latent_height // 2, latent_width // 2)]] * 2
                txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

                model_pred_packed = unet(
                    hidden_states=packed_latents,
                    timestep=t.float(),
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]

                pred_uncond, pred_cond = model_pred_packed.chunk(2)
                model_pred_packed = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                model_pred = _unpack_latents(
                    model_pred_packed, 1, latent_height, latent_width, num_channels_latents
                )

                latents = scheduler.step(model_pred, t, latents).prev_sample

            # VAE Decoding
            vae.to(accelerator.device)
            if hasattr(vae.config, "scaling_factor"):
                latents = latents / vae.config.scaling_factor
            image_5d = vae.decode(latents).sample
            image = image_5d.squeeze(2)
            vae.to("cpu")

            # Post-processing
            image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
            image = image.cpu().permute(1, 2, 0).numpy()
            image = (image * 255).round().astype("uint8")
            pil_image = Image.fromarray(image)

            # Save image
            output_dir = os.path.join(args.output_dir, "sample")
            os.makedirs(output_dir, exist_ok=True)
            num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
            filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{seed}.png"
            pil_image.save(os.path.join(output_dir, filename))

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

    # Restore model states

    # Move models back to CPU
    text_encoder.to("cpu")

    # Final cleanup
    train_util.clean_memory_on_device(accelerator.device)
