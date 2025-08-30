import argparse
import torch
from accelerate import Accelerator
import copy

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
        self.vae_scale_factor = 8  # 2 ** len(vae.config.block_out_channels)
        self.pipeline = None

    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)
        if not args.network_train_unet_only:
            logger.warning("Qwen LoRA does not support text encoder training yet. Please use --network_train_unet_only.")
        train_dataset_group.verify_bucket_reso_steps(32)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, args, weight_dtype, accelerator):
        # Store the pipeline for strategies.
        # Return deep copies of the VAE and UNet to the trainer, to avoid issues
        # where moving the model to another device corrupts the pipeline's internal state.
        self.pipeline = qwen_utils.load_qwen_pipeline(
            args.pretrained_model_name_or_path,
            weight_dtype,
            "cpu",  # load to cpu to save memory
        )
        text_encoder = self.pipeline.text_encoder
        vae = copy.deepcopy(self.pipeline.vae)
        unet = copy.deepcopy(self.pipeline.transformer)

        return "qwen-v1", [text_encoder], vae, unet

    def get_tokenize_strategy(self, args):
        return strategy_qwen.QwenTokenizeStrategy(args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_qwen.QwenTokenizeStrategy):
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_qwen.QwenLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )

    def get_text_encoding_strategy(self, args):
        return strategy_qwen.QwenTextEncodingStrategy(self)

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_qwen.QwenTextEncoderOutputsCachingStrategy(
                self,
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
        noise = torch.randn_like(latents, device=accelerator.device, dtype=weight_dtype)

        u = torch.rand(bsz, device=accelerator.device)
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

        sigmas = noise_scheduler.sigmas[indices].to(device=latents.device, dtype=latents.dtype)
        sigmas = sigmas.view(bsz, 1, 1, 1, 1)

        noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

        packed_noisy_model_input = unet._pack_latents(
            noisy_model_input,
            bsz,
            noisy_model_input.shape[2],
            noisy_model_input.shape[3],
            noisy_model_input.shape[4],
        )

        img_shapes = [(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2)] * bsz

        prompt_embeds, prompt_embeds_mask = text_encoder_conds
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

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

        model_pred = unet._unpack_latents(
            model_pred,
            height=noisy_model_input.shape[3] * self.vae_scale_factor,
            width=noisy_model_input.shape[4] * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )

        # For flow matching, the target is the velocity from noise to data, which is noise - latents.
        # This is different from standard diffusion models where the target is the noise itself.
        target = noise - latents
        target = target.permute(0, 2, 1, 3, 4)

        return model_pred, target, timesteps, None


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = QwenNetworkTrainer()
    trainer.train(args)
