import argparse
import torch
from accelerate import Accelerator

from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

import train_network
from library import (
    qwen_utils,
    strategy_qwen,
    train_util,
)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class QwenNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_scale_factor = 8

    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)
        if not args.network_train_unet_only:
            logger.warning("Qwen LoRA does not support text encoder training yet. Please use --network_train_unet_only.")
        train_dataset_group.verify_bucket_reso_steps(32)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, args, weight_dtype, accelerator):
        text_encoder = qwen_utils.load_qwen_text_encoder(
            args.pretrained_model_name_or_path, weight_dtype, "cpu", custom_text_encoder_path=args.text_encoder_path
        )
        vae = qwen_utils.load_qwen_vae(args.pretrained_model_name_or_path, weight_dtype, "cpu", custom_vae_path=args.vae)
        unet = qwen_utils.load_qwen_transformer(args.pretrained_model_name_or_path, weight_dtype, "cpu")

        return "qwen-v1", [text_encoder], vae, unet

    def get_tokenize_strategy(self, args):
        return strategy_qwen.QwenTokenizeStrategy(args.pretrained_model_name_or_path, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_qwen.QwenTokenizeStrategy):
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_qwen.QwenLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )

    def get_text_encoding_strategy(self, args):
        return strategy_qwen.QwenTextEncodingStrategy()

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_qwen.QwenTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk,
                args.text_encoder_batch_size,
                args.skip_cache_check,
            )
        return None

    def get_noise_scheduler(self, args, device):
        from diffusers import FlowMatchEulerDiscreteScheduler

        return FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="scheduler",
        )

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if not args.cache_text_encoder_outputs:
            return

        if not args.lowram:
            logger.info("move vae and unet to cpu to save memory")
            org_vae_device = vae.device
            org_unet_device = unet.device
            vae.to("cpu")
            unet.to("cpu")
            clean_memory_on_device(accelerator.device)

        text_encoder = text_encoders[0]
        text_encoder.to(accelerator.device)

        with torch.no_grad(), accelerator.autocast():
            dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

        accelerator.wait_for_everyone()

        text_encoder.to("cpu")
        clean_memory_on_device(accelerator.device)

        if not args.lowram:
            logger.info("move vae and unet back to original device")
            vae.to(org_vae_device)
            unet.to(org_unet_device)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        if hasattr(text_encoder, "embed_tokens"):
            text_encoder.embed_tokens.requires_grad_(True)

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizers, text_encoder, unet):
        logger.info("Generating samples with Qwen-specific pipeline...")
        qwen_utils.sample_images(accelerator, args, epoch, global_step, text_encoder, vae, unet, tokenizers[0])

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        bsz = latents.shape[0]

        # Ensure latents layout matches Qwen pipeline expectations: (B, 1, C, H, W)
        # Commonly our VAE produces (B, C, 1, H, W). If so, permute to (B, 1, C, H, W).
        if latents.dim() == 5 and latents.shape[2] == 1 and latents.shape[1] != 1:
            # Already (B, C, 1, H, W) -> (B, 1, C, H, W)
            latents = latents.permute(0, 2, 1, 3, 4).contiguous()
        
        noise = torch.randn_like(latents, device=latents.device, dtype=weight_dtype)

        u = torch.rand(bsz, device=latents.device)
        indices = (u * noise_scheduler.config.num_train_timesteps).long().to(device=latents.device)
        timesteps = noise_scheduler.timesteps.to(device=latents.device)[indices]

        sigmas = noise_scheduler.sigmas.to(device=latents.device, dtype=latents.dtype)[indices]
        sigmas = sigmas.view(bsz, 1, 1, 1, 1)

        noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

        # Extract dims assuming (B, 1, C, H, W)
        _, _, num_channels_latents, h, w = noisy_model_input.shape

        # QwenImageTransformer2DModel does not expose _pack_latents. Implement locally, matching the pipeline logic.
        def _pack_latents(latents, batch_size, num_channels_latents, height, width):
            latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
            latents = latents.permute(0, 2, 4, 1, 3, 5)
            latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
            return latents

        packed_noisy_model_input = _pack_latents(
            noisy_model_input,
            bsz,
            num_channels_latents,
            h,
            w,
        )

        img_shapes = [[(1, h // 2, w // 2)]] * bsz

        prompt_embeds, prompt_embeds_mask = text_encoder_conds
        # Ensure text condition tensors match transformer input device/dtype
        pe_device = packed_noisy_model_input.device
        pe_dtype = packed_noisy_model_input.dtype
        if prompt_embeds.device != pe_device or prompt_embeds.dtype != pe_dtype:
            prompt_embeds = prompt_embeds.to(device=pe_device, dtype=pe_dtype)
        if prompt_embeds_mask.device != pe_device:
            prompt_embeds_mask = prompt_embeds_mask.to(device=pe_device)

        # Clamp text sequence length to a safe maximum like the pipeline (default 512)
        max_sequence_length = 512
        if prompt_embeds.dim() == 3 and prompt_embeds.shape[1] > max_sequence_length:
            prompt_embeds = prompt_embeds[:, :max_sequence_length]
        if prompt_embeds_mask.dim() == 2 and prompt_embeds_mask.shape[1] > max_sequence_length:
            prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        # Compute per-sample text lengths and trim sequences to the batch max valid length to avoid rotary mismatch
        if prompt_embeds_mask.dtype != torch.bool:
            prompt_embeds_mask = prompt_embeds_mask.to(torch.bool)
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        max_valid_len = int(max(txt_seq_lens)) if len(txt_seq_lens) > 0 else prompt_embeds.shape[1]
        # Ensure at least length 1 to avoid empty slices
        max_valid_len = max(1, min(max_valid_len, prompt_embeds.shape[1]))
        # Slice both embeddings and mask to the max valid length in this batch
        if prompt_embeds.shape[1] != max_valid_len:
            prompt_embeds = prompt_embeds[:, :max_valid_len]
        if prompt_embeds_mask.shape[1] != max_valid_len:
            prompt_embeds_mask = prompt_embeds_mask[:, :max_valid_len]

        # Also ensure timesteps reside on same device
        timesteps = timesteps.to(device=pe_device)

        # Ensure the Qwen transformer (unet) is on the same device as inputs
        try:
            unet_param_device = next(unet.parameters()).device
        except StopIteration:
            unet_param_device = pe_device
        if unet_param_device != pe_device:
            logger.info(f"Moving Qwen transformer to device {pe_device} (was {unet_param_device}) to match inputs")
            unet.to(pe_device)

        model_pred = unet(
            hidden_states=packed_noisy_model_input,
            timestep=timesteps / 1000,
            guidance=None,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]

        # Implement local unpack to avoid relying on pipeline private methods
        def _unpack_latents(latents, height, width, vae_scale_factor):
            batch_size, num_patches, channels = latents.shape
            # Make height/width divisible by 2 and VAE scale factor like in pipeline
            height = 2 * (int(height) // (vae_scale_factor * 2))
            width = 2 * (int(width) // (vae_scale_factor * 2))
            latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
            latents = latents.permute(0, 3, 1, 4, 2, 5)
            latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)
            return latents

        model_pred = _unpack_latents(
            model_pred,
            height=h * self.vae_scale_factor,
            width=w * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )

        target = noise - latents
        # Return target in (B, C, 1, H, W) like the rest of the training code expects
        target = target.permute(0, 2, 1, 3, 4)

        return model_pred, target, timesteps, None


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="path to the text encoder model to use, if different from the main model",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = QwenNetworkTrainer()
    trainer.train(args)
