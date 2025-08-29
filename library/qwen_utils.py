import torch
from diffusers import (
    QwenImagePipeline,
    AutoencoderKLQwenImage,
    QwenImageTransformer2DModel,
)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def load_qwen_pipeline(
    model_name_or_path,
    torch_dtype,
    device,
):
    logger.info("Loading QwenImagePipeline")
    pipeline = QwenImagePipeline.from_pretrained(
        model_name_or_path,
        transformer=None,
        vae=None,
        torch_dtype=torch_dtype,
    )
    pipeline.to(device)
    return pipeline


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
