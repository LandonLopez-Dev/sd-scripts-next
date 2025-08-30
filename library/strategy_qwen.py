import os
from typing import Any, List, Optional, Tuple, Union
import torch
import numpy as np
from transformers import CLIPTokenizer

from library import train_util
from library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class QwenTokenizeStrategy(TokenizeStrategy):
    def __init__(self, tokenizer_cache_dir: Optional[str] = None) -> None:
        # This is a bit of a hack. The Qwen tokenizer is a CLIPTokenizer.
        # We load it here independently. In the future, it should be passed from the trainer.
        # For now, this works because the model uses a standard CLIP tokenizer.
        self.tokenizer = self._load_tokenizer("openai/clip-vit-large-patch14", tokenizer_cache_dir=tokenizer_cache_dir)

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text
        tokens = self.tokenizer(text, max_length=1024, padding="max_length", truncation=True, return_tensors="pt")
        return [tokens["input_ids"], tokens["attention_mask"]]


class QwenTextEncodingStrategy(TextEncodingStrategy):
    def __init__(self) -> None:
        pass

    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        text_encoder = models[0] # The text encoder is passed in the models list
        input_ids, attention_mask = tokens

        # The Qwen text_encoder is a Qwen2Model, which returns BaseModelOutputWithPast.
        # The first element is the last_hidden_state.
        prompt_embeds = text_encoder(
            input_ids=input_ids.to(text_encoder.device),
            attention_mask=attention_mask.to(text_encoder.device),
        )[0]

        # The prompt_embeds_mask is the attention_mask
        prompt_embeds_mask = attention_mask

        return [prompt_embeds, prompt_embeds_mask]


class QwenTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    QWEN_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_qwen_te.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + QwenTextEncoderOutputsCachingStrategy.QWEN_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str):
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = np.load(npz_path)
            if "prompt_embeds" not in npz:
                return False
            if "prompt_embeds_mask" not in npz:
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        prompt_embeds = data["prompt_embeds"]
        prompt_embeds_mask = data["prompt_embeds_mask"]
        return [prompt_embeds, prompt_embeds_mask]

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], text_encoding_strategy: TextEncodingStrategy, infos: List
    ):
        captions = [info.caption for info in infos]

        tokens = tokenize_strategy.tokenize(captions)
        with torch.no_grad():
            prompt_embeds, prompt_embeds_mask = text_encoding_strategy.encode_tokens(tokenize_strategy, models, tokens)

        if prompt_embeds.dtype == torch.bfloat16:
            prompt_embeds = prompt_embeds.float()
        if prompt_embeds_mask.dtype == torch.bfloat16:
            prompt_embeds_mask = prompt_embeds_mask.float()

        prompt_embeds = prompt_embeds.cpu().numpy()
        prompt_embeds_mask = prompt_embeds_mask.cpu().numpy()

        for i, info in enumerate(infos):
            prompt_embeds_i = prompt_embeds[i]
            prompt_embeds_mask_i = prompt_embeds_mask[i]

            if self.cache_to_disk:
                np.savez(
                    info.text_encoder_outputs_npz,
                    prompt_embeds=prompt_embeds_i,
                    prompt_embeds_mask=prompt_embeds_mask_i,
                )
            else:
                info.text_encoder_outputs = (prompt_embeds_i, prompt_embeds_mask_i)


class QwenLatentsCachingStrategy(LatentsCachingStrategy):
    QWEN_LATENTS_NPZ_SUFFIX = "_qwen.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    @property
    def cache_suffix(self) -> str:
        return QwenLatentsCachingStrategy.QWEN_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return (
            os.path.splitext(absolute_path)[0]
            + f"_{image_size[0]:04d}x{image_size[1]:04d}"
            + QwenLatentsCachingStrategy.QWEN_LATENTS_NPZ_SUFFIX
        )

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True)

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso)  # support multi-resolution

    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        encode_by_vae = lambda img_tensor: vae.encode(img_tensor.unsqueeze(2)).latent_dist.sample().to("cpu")
        vae_device = vae.device
        vae_dtype = vae.dtype

        self._default_cache_batch_latents(
            encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=True
        )

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae.device)
