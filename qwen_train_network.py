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

        target = noise - latents
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
