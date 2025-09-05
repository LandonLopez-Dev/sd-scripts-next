from copy import deepcopy
import logging
import math
import os
import shutil
import random
import argparse
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import (
    AutoencoderKLQwenImage,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from library import train_util, config_util
from library.utils import setup_logging, add_logging_arguments


def image_resize(img, max_size=1024):
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    return img.resize((new_w, new_h), Image.BICUBIC)


def crop_to_aspect_ratio(image, ratio="16:9"):
    width, height = image.size
    ratio_map = {"16:9": (16, 9), "4:3": (4, 3), "1:1": (1, 1)}
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h

    current_ratio = width / height

    if current_ratio > target_ratio_value:
        new_width = int(height * target_ratio_value)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio_value)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    cropped_img = image.crop(crop_box)
    return cropped_img


def transform_image(image, resolution, random_crop_aspect_ratio=False):
    if random_crop_aspect_ratio:
        ratio = random.choice(["16:9", "default", "1:1", "4:3"])
        if ratio != "default":
            image = crop_to_aspect_ratio(image, ratio)

    image = image_resize(image, resolution)
    w, h = image.size
    new_w = (w // 32) * 32
    new_h = (h // 32) * 32
    image = image.resize((new_w, new_h), Image.BICUBIC)

    # normalize to [-1, 1]
    np_image = np.array(image, dtype=np.float32)
    np_image = (np_image / 127.5) - 1.0

    return torch.from_numpy(np_image).permute(2, 0, 1)


class QwenDataset(Dataset):
    def __init__(self, train_data_dir, resolution=1024, caption_dropout_rate=0.0):
        self.train_data_dir = train_data_dir
        self.resolution = resolution
        self.caption_dropout_rate = caption_dropout_rate

        self.image_paths = []
        for root, _, files in os.walk(self.train_data_dir):
            for file in files:
                if file.endswith((".jpg", ".png", ".jpeg")):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        caption_path = os.path.splitext(image_path)[0] + ".txt"
        if os.path.exists(caption_path):
            with open(caption_path, "r") as f:
                caption = f.read()
        else:
            caption = ""

        if random.random() < self.caption_dropout_rate:
            caption = ""

        return {"image": image, "caption": caption}


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)

    parser.add_argument(
        "--network_dim",
        type=int,
        default=16,
        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Limit the number of checkpoints to save. / 保存するチェックポイントの数を制限します。",
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default=None,
        choices=["tensorboard", "wandb", "all"],
        help="Logging with tensorboard, wandb, or all. / tensorboard, wandb, またはすべてでロギングします。",
    )
    parser.add_argument(
        "--caption_dropout_rate",
        type=float,
        default=0.0,
        help="Rate of caption dropout. / キャプションのドロップアウト率。",
    )
    parser.add_argument(
        "--random_crop_aspect_ratio",
        action="store_true",
        help="Enable random aspect ratio cropping. / ランダムなアスペクト比でのクロッピングを有効にします。",
    )
    return parser


logger = get_logger(__name__, log_level="INFO")


def main():
    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    setup_logging(args, reset=True)

    if args.seed is None:
        args.seed = random.randint(0, 2**32)
    set_seed(args.seed)

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.log_with,
        project_config=accelerator_project_config,
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

    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    text_encoding_pipeline = QwenImagePipeline.from_pretrained(
        args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
    )
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
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    transformer.to(accelerator.device, dtype=weight_dtype)
    transformer.add_adapter(lora_config)
    text_encoding_pipeline.to(accelerator.device)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    transformer.train()
    for n, param in transformer.named_parameters():
        if 'lora' not in n:
            param.requires_grad = False
            pass
        else:
            param.requires_grad = True
            print(n)
    print(sum([p.numel() for p in transformer.parameters() if p.requires_grad]) / 1000000, 'parameters')
    lora_layers = filter(lambda p: p.requires_grad, transformer.parameters())

    transformer.enable_gradient_checkpointing()
    _, _, optimizer = train_util.get_optimizer(args, lora_layers)

    # Dataloader
    dataset = QwenDataset(args.train_data_dir, resolution=args.resolution, caption_dropout_rate=args.caption_dropout_rate)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.max_data_loader_n_workers,
    )

    # calculate training steps and epochs
    if args.max_train_steps is None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)
    global_step = 0
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    initial_global_step = 0

    if accelerator.is_main_process:
        accelerator.init_trackers(args.wandb_run_name or "qwen-lora-training", {"test": None})

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num train images * repeats / 学習画像の数×繰り返し回数: {len(dataset)}")
    logger.info(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    logger.info(f"  num epochs / epoch数: {num_train_epochs}")
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
    for epoch in range(num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                images = batch["image"]
                prompts = batch["caption"]

                # Preprocess images
                pixel_values = torch.stack(
                    [transform_image(image, args.resolution, args.random_crop_aspect_ratio) for image in images]
                )
                pixel_values = pixel_values.to(dtype=weight_dtype).to(accelerator.device)
                pixel_values = pixel_values.unsqueeze(2)

                with torch.no_grad():
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
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme="none",
                        batch_size=bsz,
                        logit_mean=0.0,
                        logit_std=1.0,
                        mode_scale=1.29,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

                packed_noisy_model_input = QwenImagePipeline._pack_latents(
                    noisy_model_input,
                    bsz,
                    noisy_model_input.shape[2],
                    noisy_model_input.shape[3],
                    noisy_model_input.shape[4],
                )

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

                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)

                target = noise - pixel_latents
                target = target.permute(0, 2, 1, 3, 4)

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step > 0 and global_step % args.save_every_n_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                        unwrapped_transformer = unwrap_model(transformer)
                        lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_transformer)
                        )

                        QwenImagePipeline.save_lora_weights(
                            save_path,
                            lora_state_dict,
                            safe_serialization=True,
                        )
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()