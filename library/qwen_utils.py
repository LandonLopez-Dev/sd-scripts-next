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


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.org_module = [org_module]

        if self.org_module[0].__class__.__name__ == "Conv2d":
            in_dim = self.org_module[0].in_channels
            out_dim = self.org_module[0].out_channels
        else:
            in_dim = self.org_module[0].in_features
            out_dim = self.org_module[0].out_features

        # if limit_rank:
        #   self.lora_dim = min(lora_dim, in_dim, out_dim)
        #   if self.lora_dim != lora_dim:
        #     logger.info(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")
        # else:
        self.lora_dim = lora_dim

        if self.org_module[0].__class__.__name__ == "Conv2d":
            kernel_size = self.org_module[0].kernel_size
            stride = self.org_module[0].stride
            padding = self.org_module[0].padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward
        # Keep reference to org_module for device/dtype alignment in forward

    def forward(self, x):
        # Ensure input and LoRA weights are on the same device/dtype as the original (wrapped) module
        # Determine the base module's device and dtype from its weight
        base_weight = getattr(self.org_module[0], 'weight', None)
        if base_weight is not None:
            target_device = base_weight.device
            target_dtype = base_weight.dtype
        else:
            # Fallback: use input's current device/dtype
            target_device = x.device
            target_dtype = x.dtype

        # Move input to base module's device/dtype
        if x.device != target_device or x.dtype != target_dtype:
            x = x.to(target_device, dtype=target_dtype)

        # Move LoRA layers to match the base module's device/dtype when needed
        if self.lora_down.weight.device != target_device or self.lora_down.weight.dtype != target_dtype:
            self.lora_down.to(target_device, dtype=target_dtype)
            self.lora_up.to(target_device, dtype=target_dtype)

        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        lx = self.lora_down(x)

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)  # for Text Encoder
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
            lx = lx * mask

            # scaling for rank dropout: treat as if the rank is changed
            # maskから計算することも考えられるが、augmentation的な効果を期待してrank_dropoutを用いる
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
        else:
            scale = self.scale

        lx = self.lora_up(lx)

        return org_forwarded + lx * self.multiplier * scale
