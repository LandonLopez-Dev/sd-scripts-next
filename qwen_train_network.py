import argparse
import copy
from copy import deepcopy
import logging
import os
import shutil
import math
import time
import gc

import torch
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
import datasets
import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import (
    AutoencoderKLQwenImage,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
import library.config_util as config_util
from torch.utils.data import DataLoader
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import transformers
import library.strategy_base as strategy_base
from library import qwen_train_utils

logger = get_logger(__name__, log_level="INFO")


class SimpleFlowMatchScheduler:
    def __init__(self, num_train_timesteps: int = 1000, sigma_max: float = 1.0, sigma_min: float = 0.01):
        self.num_train_timesteps = num_train_timesteps
        self.sigmas = torch.linspace(sigma_max, sigma_min, num_train_timesteps)
        self.timesteps = torch.arange(0, num_train_timesteps).flip(0)

    @property
    def config(self):
        # Create a mock config object for compatibility
        class MockConfig:
            def __init__(self, num_train_timesteps):
                self.num_train_timesteps = num_train_timesteps
        return MockConfig(self.num_train_timesteps)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_config", type=str, default=None, help="path to dataset config file")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument(
        "--output_name", type=str, default=None, help="base name of trained model file / 学習後のモデルの拡張子を除くファイル名"
    )
    parser.add_argument("--logging_dir", type=str, default=None, help="enable logging and output TensorBoard log to this directory")
    parser.add_argument("--log_prefix", type=str, default=None, help="add prefix for each log directory")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--log_with", type=str, default="tensorboard")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--network_dim", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="scheduler to use for learning rate / 学習率のスケジューラ: linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup, adafactor",
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=1600, help="training steps / 学習ステップ数")
    parser.add_argument(
        "--max_train_epochs",
        type=int,
        default=None,
        help="training epochs (overrides max_train_steps) / 学習エポック数（max_train_stepsを上書きします）",
    )
    parser.add_argument("--tracker_project_name", type=str, default="qwen-lora")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--save_every_n_steps", type=int, default=300)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--enable_bucket", action="store_true")
    parser.add_argument("--min_bucket_reso", type=int, default=256)
    parser.add_argument("--max_bucket_reso", type=int, default=1024)
    parser.add_argument("--bucket_reso_steps", type=int, default=64)
    parser.add_argument("--bucket_no_upscale", action="store_true")
    parser.add_argument("--debug_dataset", action="store_true")
    parser.add_argument("--max_data_loader_n_workers", type=int, default=0)
    parser.add_argument("--resolution", type=str, default="512,512")
    parser.add_argument(
        "--sample_at_first", action="store_true", help="generate sample images before training / 学習前にサンプル出力する"
    )
    parser.add_argument(
        "--sample_every_n_steps",
        type=int,
        default=None,
        help="generate sample images every N steps / 学習中のモデルで指定ステップごとにサンプル出力する",
    )
    parser.add_argument(
        "--sample_every_n_epochs",
        type=int,
        default=None,
        help="generate sample images every N epochs (overwrites n_steps) / 学習中のモデルで指定エポックごとにサンプル出力する（ステップ数指定を上書きします）",
    )
    parser.add_argument(
        "--sample_prompts",
        type=str,
        default=None,
        help="file for prompts to generate sample images / 学習中モデルのサンプル出力用プロンプトのファイル",
    )

    parser.add_argument(
        "--noise_offset",
        type=float,
        default=None,
        help="enable noise offset with this value (if enabled, around 0.1 is recommended) / Noise offsetを有効にしてこの値を設定する（有効にする場合は0.1程度を推奨）",
    )
    parser.add_argument(
        "--loss_weighting_scheme",
        type=str,
        default="none",
        choices=["none", "snr", "snr_trunc"],
        help="Loss weighting scheme. 'snr_trunc' is recommended for quality improvement.",
    )

    parser.add_argument('--num_train_timesteps', type=int, default=1000)
    parser.add_argument('--sigma_max', type=float, default=1.0)
    parser.add_argument('--sigma_min', type=float, default=0.01)

    args = parser.parse_args()
    if args.resolution:
        args.resolution = tuple(map(int, args.resolution.split(',')))
    return args


def main():
    args = parse_args()

    if args.dataset_config:
        user_config = config_util.load_user_config(args.dataset_config)
    else:
        user_config = {"datasets": []}

    if args.logging_dir is not None:
        log_prefix = "qwen_" if args.log_prefix is None else args.log_prefix
        logging_dir = os.path.join(args.logging_dir, log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime()))
    else:
        logging_dir = None

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.log_with,
        project_dir=logging_dir,
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
    text_encoding_pipeline = QwenImagePipeline.from_pretrained(
        args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
    )
    text_encoding_pipeline.text_encoder.model.visual = None
    gc.collect()
    torch.cuda.empty_cache()
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer", )
    lora_config = LoraConfig(
        r=args.network_dim,
        lora_alpha=args.network_dim,
        init_lora_weights="gaussian",
        target_modules="all-linear"
    )
    noise_scheduler = SimpleFlowMatchScheduler(
        num_train_timesteps=args.num_train_timesteps,
        sigma_max=args.sigma_max,
        sigma_min=args.sigma_min
    )
    transformer.to(accelerator.device, dtype=weight_dtype)
    transformer.add_adapter(lora_config)
    text_encoding_pipeline.to(accelerator.device)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    transformer.train()
    optimizer_cls = torch.optim.AdamW
    for n, param in transformer.named_parameters():
        if 'lora' not in n:
            param.requires_grad = False
        else:
            param.requires_grad = True
            logger.debug(n)
    logger.info(f"{sum([p.numel() for p in transformer.parameters() if p.requires_grad]) / 1000000:.5f} parameters")
    lora_layers = filter(lambda p: p.requires_grad, transformer.parameters())

    transformer.enable_gradient_checkpointing()
    optimizer = optimizer_cls(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Build dataset from user_config using standard config_util pipeline
    sanitizer = config_util.ConfigSanitizer(support_dreambooth=True, support_finetuning=True, support_controlnet=False, support_dropout=True)
    blueprint = config_util.BlueprintGenerator(sanitizer).generate(user_config, args)
    train_dataset_group, _ = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    # Prevent dataset from trying to tokenize by setting a no-op caching strategy
    class _NoTextEncoderCache(strategy_base.TextEncoderOutputsCachingStrategy):
        def __init__(self):
            super().__init__(cache_to_disk=False, batch_size=None, skip_disk_cache_validity_check=True, is_partial=False)
        def get_outputs_npz_path(self, image_abs_path: str) -> str: raise NotImplementedError
        def load_outputs_npz(self, npz_path: str): raise NotImplementedError
        def is_disk_cached_outputs_expected(self, npz_path: str) -> bool: return False
        def cache_batch_outputs(self, tokenize_strategy, models, text_encoding_strategy, batch):
            return None

    # Set strategies so BaseDataset doesn't attempt tokenization/caching
    strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(_NoTextEncoderCache())
    # TokenizeStrategy must be set (dataset references it), but Qwen path encodes prompts directly.
    # Set a dummy strategy that won't be used since tokenization is disabled by the above caching strategy.
    class _DummyTokenizeStrategy(strategy_base.TokenizeStrategy):
        def tokenize(self, text):
            # Return a structure compatible with dataset expectations but unused.
            # Expecting list of token tensors per text model; return empty to indicate no tokens.
            return []
        def tokenize_with_weights(self, text):
            return [], []
    strategy_base.TokenizeStrategy.set_strategy(_DummyTokenizeStrategy())

    # Dataset needs to capture current strategies for worker processes
    train_dataset_group.set_current_strategies()

    train_dataloader = DataLoader(
        train_dataset_group,
        batch_size=1,  # batching is handled inside the dataset
        shuffle=True,
        collate_fn=lambda examples: examples[0],
        num_workers=args.max_data_loader_n_workers,
        pin_memory=True,
    )

    if args.max_train_steps == 0:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.max_train_steps = args.max_train_epochs * num_update_steps_per_epoch
    else:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.max_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer, optimizer, _, lr_scheduler = accelerator.prepare(
        transformer, optimizer, deepcopy(train_dataloader), lr_scheduler
    )

    initial_global_step = 0

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor = 2 ** len(vae.temperal_downsample)

    if args.sample_prompts is not None and args.sample_at_first:
        qwen_train_utils.sample_images(
            accelerator, args, 0, 0, transformer, vae, text_encoding_pipeline, 0
        )

    for epoch in range(args.max_train_epochs):
        train_loss = 0.0
        epoch_total_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                img = batch["images"]
                prompts = batch["captions"]
                with torch.no_grad():
                    pixel_values = img.to(dtype=weight_dtype).to(accelerator.device)
                    pixel_values = pixel_values.unsqueeze(2)

                    pixel_latents = vae.encode(pixel_values).latent_dist.sample()
                    pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)

                    latents_mean = (
                        torch.tensor(vae.config.latents_mean)
                        .view(1, 1, vae.config.z_dim, 1, 1)
                        .to(pixel_latents.device, pixel_latents.dtype)
                    )
                    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 1, vae.config.z_dim, 1, 1).to(
                        pixel_latents.device, pixel_latents.dtype
                    )
                    pixel_latents = (pixel_latents - latents_mean) * latents_std

                    bsz = pixel_latents.shape[0]
                    noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)
                    if args.noise_offset and args.noise_offset > 0:
                        # Add noise offset scaled by the standard deviation of the noise
                        noise = noise + args.noise_offset * torch.randn_like(noise)
                    timesteps = torch.randint(0, noise_scheduler_copy.config.num_train_timesteps, (bsz,), device=pixel_latents.device).long()

                sigmas = noise_scheduler_copy.sigmas[timesteps].to(device=pixel_latents.device, dtype=pixel_latents.dtype)
                while len(sigmas.shape) < pixel_latents.ndim:
                    sigmas = sigmas.unsqueeze(-1)
                noisy_model_input = pixel_latents + sigmas * noise
                # Concatenate across channels.
                # pack the latents.
                packed_noisy_model_input = QwenImagePipeline._pack_latents(
                    noisy_model_input,
                    bsz,
                    noisy_model_input.shape[2],
                    noisy_model_input.shape[3],
                    noisy_model_input.shape[4],
                )
                # latent image ids for RoPE.
                img_shapes = [(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2)] * bsz
                with torch.no_grad():
                    prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                        prompt=prompts,
                        device=packed_noisy_model_input.device,
                        num_images_per_prompt=1,
                        max_sequence_length=1024,
                    )
                    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]
                model_pred = QwenImagePipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[3] * vae_scale_factor,
                    width=noisy_model_input.shape[4] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.loss_weighting_scheme, sigmas=sigmas)
                # noise prediction loss
                target = noise.detach()
                target = target.permute(0, 2, 1, 3, 4)
                # Calculate per-element loss (the squared error)
                loss_per_element = (weighting.float() * (model_pred.float() - target.float()) ** 2)
                loss = torch.mean(loss_per_element.reshape(target.shape[0], -1), 1)
                loss = loss.mean()
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                epoch_total_loss += avg_loss.item()

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                average_loss = epoch_total_loss / (step + 1)
                logs_for_accelerator = {
                    "loss/step": train_loss,
                    "loss/current": avg_loss.item(),
                    "loss/average": average_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                accelerator.log(logs_for_accelerator, step=global_step)

                train_loss = 0.0

                if global_step > 0 and global_step % args.save_every_n_steps == 0:
                    if accelerator.is_main_process:
                        model_name = args.output_name if args.output_name is not None else "qwen-lora"
                        save_path = os.path.join(args.output_dir, f"{model_name}-step{global_step:08d}")

                        if args.checkpoints_total_limit is not None:
                            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith(f"{model_name}-step")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("step")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint_path = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint_path)

                        logger.info(f"saving checkpoint: {save_path}")
                        os.makedirs(save_path, exist_ok=True)
                        unwrapped_transformer = unwrap_model(transformer)
                        lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_transformer))

                        QwenImagePipeline.save_lora_weights(
                            save_path,
                            transformer_lora_layers=lora_state_dict,
                            safe_serialization=True,
                        )

                if args.sample_prompts is not None:
                    qwen_train_utils.sample_images(
                        accelerator,
                        args,
                        epoch,
                        global_step,
                        transformer,
                        vae,
                        text_encoding_pipeline,
                        num_update_steps_per_epoch,
                    )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()