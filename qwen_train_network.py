import argparse
import torch
from accelerate import Accelerator
from diffusers import QwenImagePipeline

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
import os

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
        unet = qwen_utils.load_qwen_transformer(
            args.pretrained_model_name_or_path,
            weight_dtype,
            "cpu",
            accelerator=accelerator,
            args=args,
        )

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

        move_unet = True
        if not args.lowram:
            # On-demand qfloat8 quantization on Windows can hang when moving UNet between devices.
            if getattr(args, "use_qfloat8_on_demand", False):
                logger.info("Skipping moving UNet to CPU due to --use_qfloat8_on_demand; keeping it on its device during TE caching.")
                move_unet = False

            logger.info("move vae and{} unet to cpu to save memory".format("" if move_unet else " (skip)"))
            org_vae_device = vae.device
            org_unet_device = unet.device
            vae.to("cpu")
            if move_unet:
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
            logger.info("move vae and{} unet back to original device".format("" if move_unet else " (skip)"))
            vae.to(org_vae_device)
            if move_unet:
                unet.to(org_unet_device)

    def cache_latents_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if not args.cache_latents:
            return

        move_unet = True
        if not args.lowram:
            if getattr(args, "use_qfloat8_on_demand", False):
                logger.info("Skipping moving UNet to CPU due to --use_qfloat8_on_demand; keeping it on its device during latents caching.")
                move_unet = False
            logger.info("move text encoder and{} unet to cpu to save memory".format("" if move_unet else " (skip)"))
            org_text_encoder_device = text_encoders[0].device
            org_unet_device = unet.device
            text_encoders[0].to("cpu")
            if move_unet:
                unet.to("cpu")
            clean_memory_on_device(accelerator.device)

        vae.to(accelerator.device, dtype=weight_dtype)

        with torch.no_grad(), accelerator.autocast():
            dataset.new_cache_latents(vae, accelerator)

        accelerator.wait_for_everyone()

        vae.to("cpu")
        clean_memory_on_device(accelerator.device)

        if not args.lowram:
            logger.info("move text encoder and{} unet back to original device".format("" if move_unet else " (skip)"))
            text_encoders[0].to(org_text_encoder_device)
            if move_unet:
                unet.to(org_unet_device)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        if hasattr(text_encoder, "embed_tokens"):
            text_encoder.embed_tokens.requires_grad_(True)

    def is_train_text_encoder(self, args):
        # Qwen LoRA does not support TE training; force-disable to avoid CUDA TE forward + grad ckpt stalls
        import logging as _logging
        _logging.getLogger(__name__).info("Qwen trainer: forcing is_train_text_encoder=False (text encoder will not be trained)")
        return False

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizers, text_encoder, unet):
        qwen_utils.sample_images(accelerator, args, epoch, global_step, text_encoder, vae, unet, tokenizers[0])

    def prepare_unet_with_accelerator(self, args, accelerator: Accelerator, unet: torch.nn.Module) -> torch.nn.Module:
        # Avoid wrapping the Qwen transformer with Accelerator when using on-demand qfloat8 quantization.
        # Some accelerator backends attempt to cast/replicate parameters which can hang with Quanto qfloat8 tensors on Windows.
        if getattr(args, "use_qfloat8_on_demand", False):
            # Just move to the correct device; dtype is already handled by caller.
            logger.info("Skipping accelerator.prepare(unet) due to --use_qfloat8_on_demand; moving model to device directly.")
            # Do not move now; if quantized with deferral, we'll move at first forward safely.
            return unet
        return super().prepare_unet_with_accelerator(args, accelerator, unet)

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
        logger.info("Qwen get_noise_pred_and_target begin")

        # Ensure latents layout matches Qwen pipeline expectations: (B, 1, C, H, W)
        # Commonly our VAE produces (B, C, 1, H, W). If so, permute to (B, 1, C, H, W).
        if latents.dim() == 5 and latents.shape[2] == 1 and latents.shape[1] != 1:
            # Already (B, C, 1, H, W) -> (B, 1, C, H, W)
            latents = latents.permute(0, 2, 1, 3, 4).contiguous()
        logger.info("Qwen preflight: latents layout prepared")

        # Ensure latents are in compute dtype to avoid implicit upcasts and larger buffers
        if latents.dtype != weight_dtype:
            latents = latents.to(dtype=weight_dtype)
        logger.info("Qwen preflight: latents cast")

        # On Windows with qfloat8, build preflight tensors on CPU to avoid first CUDA ops before UNet forward
        cpu_preflight = getattr(args, "use_qfloat8_on_demand", False) and (latents.device.type == "cuda")
        work_device = torch.device("cpu") if cpu_preflight else latents.device

        latents_cpu = latents.to("cpu") if cpu_preflight else latents
        noise = torch.randn_like(latents_cpu, device=work_device, dtype=weight_dtype)
        logger.info("Qwen preflight: noise sampled")

        u = torch.rand(bsz, device=work_device)
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        logger.info("Qwen preflight: timesteps indices computed")
        # Keep scheduler buffers on CPU and only gather the indexed values to GPU to avoid large persistent GPU tensors
        indices_cpu = indices.to("cpu")
        # gather on CPU first to keep everything on host for preflight
        timesteps = noise_scheduler.timesteps[indices_cpu].to(device=work_device)
        logger.info("Qwen preflight: timesteps gathered")

        sigmas = noise_scheduler.sigmas[indices_cpu].to(device=work_device, dtype=latents_cpu.dtype)
        sigmas_5d = sigmas.view(bsz, 1, 1, 1, 1)
        logger.info("Qwen preflight: sigmas prepared")

        noisy_model_input = (1.0 - sigmas_5d) * (latents_cpu if cpu_preflight else latents) + sigmas_5d * noise
        logger.info("Qwen preflight: noisy input built")

        # Extract dims assuming (B, 1, C, H, W)
        _, _, num_channels_latents, h, w = noisy_model_input.shape

        # pack the latents. If we built on CPU, pack on CPU too, then move to device later
        packed_noisy_model_input = QwenImagePipeline._pack_latents(
            noisy_model_input,
            bsz,
            noisy_model_input.shape[2],
            noisy_model_input.shape[3],
            noisy_model_input.shape[4],
        )
        logger.info("Qwen preflight: latents packed")

        img_shapes = [[(1, h // 2, w // 2)]] * bsz

        prompt_embeds, prompt_embeds_mask = text_encoder_conds
        # Ensure text condition tensors and all inputs match the transformer's device/dtype.
        # Prefer moving inputs to the UNet's device instead of moving the UNet, to avoid hangs with qfloat8 on Windows.
        try:
            unet_param_device = next(unet.parameters()).device if any(p.requires_grad or p is not None for p in unet.parameters(recurse=False)) else getattr(unet, 'device', packed_noisy_model_input.device)
        except StopIteration:
            unet_param_device = packed_noisy_model_input.device
        pe_device = unet_param_device
        pe_dtype = packed_noisy_model_input.dtype

        if packed_noisy_model_input.device != pe_device or packed_noisy_model_input.dtype != pe_dtype:
            packed_noisy_model_input = packed_noisy_model_input.to(device=pe_device, dtype=pe_dtype, non_blocking=True)
        if prompt_embeds.device != pe_device or prompt_embeds.dtype != pe_dtype:
            prompt_embeds = prompt_embeds.to(device=pe_device, dtype=pe_dtype, non_blocking=True)
        if prompt_embeds_mask.device != pe_device:
            prompt_embeds_mask = prompt_embeds_mask.to(device=pe_device, non_blocking=True)

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

        # Also ensure timesteps and sigmas reside on same device; if we preflighted on CPU, this is first CUDA hop
        timesteps = timesteps.to(device=pe_device)
        sigmas = sigmas.to(device=pe_device, dtype=pe_dtype)

        # Run Qwen forward under Accelerate's autocast so it respects --mixed_precision
        logger.info(f"Qwen forward start: device={pe_device}, dtype={pe_dtype}, latents={tuple(packed_noisy_model_input.shape)}, txt={tuple(prompt_embeds.shape)}")
        # Perform deferred safe move of the transformer to CUDA just before first use
        if getattr(args, "use_qfloat8_on_demand", False) and hasattr(unet, "_defer_move_device"):
            try:
                dev = getattr(unet, "_defer_move_device")
                dt = getattr(unet, "_defer_move_dtype", None)
                logger.info(f"Deferred move: moving quantized Qwen transformer to {dev} now.")
                unet.to(dev, dtype=dt)
                delattr(unet, "_defer_move_device")
                if hasattr(unet, "_defer_move_dtype"):
                    delattr(unet, "_defer_move_dtype")
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                logger.info("Deferred move completed.")
            except Exception as e:
                logger.warning(f"Deferred move failed (will proceed anyway): {e}")
        # After deferred move, re-align inputs to the UNet's new device/dtype before forward
        try:
            new_device = next(unet.parameters()).device
        except StopIteration:
            new_device = getattr(unet, "device", pe_device)
        try:
            new_dtype = next(unet.parameters()).dtype
        except StopIteration:
            new_dtype = pe_dtype
        # Only cast floating tensors; keep masks as bool
        if packed_noisy_model_input.device != new_device or packed_noisy_model_input.dtype != new_dtype:
            packed_noisy_model_input = packed_noisy_model_input.to(device=new_device, dtype=new_dtype, non_blocking=True)
        if prompt_embeds.device != new_device or prompt_embeds.dtype != new_dtype:
            prompt_embeds = prompt_embeds.to(device=new_device, dtype=new_dtype, non_blocking=True)
        if prompt_embeds_mask.device != new_device:
            prompt_embeds_mask = prompt_embeds_mask.to(device=new_device, non_blocking=True)
        if timesteps.device != new_device:
            timesteps = timesteps.to(device=new_device, non_blocking=True)
        if sigmas.device != new_device or sigmas.dtype != new_dtype:
            sigmas = sigmas.to(device=new_device, dtype=new_dtype, non_blocking=True)
        logger.info(f"Realigned inputs to device={new_device}, dtype={new_dtype} after deferred move.")

        # One-time safe warmup to initialize CUDA kernels for qfloat8 on Windows before the real forward
        if getattr(args, "use_qfloat8_on_demand", False) and new_device.type == "cuda" and not hasattr(unet, "_did_safe_warmup"):
            old_benchmark = None
            old_tf32 = None
            try:
                # Make kernel selection deterministic and simple for the first call
                old_benchmark = torch.backends.cudnn.benchmark
                torch.backends.cudnn.benchmark = False
                if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                    old_tf32 = torch.backends.cuda.matmul.allow_tf32
                    torch.backends.cuda.matmul.allow_tf32 = False
            except Exception:
                pass
            try:
                logger.info("CUDA preflight sync A (before warmup)...")
                # Ensure all pending async copies complete before warmup
                torch.cuda.synchronize(new_device)
                # Perform a tiny CUDA matmul to initialize cuBLAS handles outside the model
                try:
                    a = torch.zeros((1, 1), device=new_device, dtype=torch.float32)
                    b = torch.zeros((1, 1), device=new_device, dtype=torch.float32)
                    _ = a @ b
                    torch.cuda.synchronize(new_device)
                    logger.info("cuBLAS tiny matmul warmup completed.")
                except Exception as _e:
                    logger.warning(f"cuBLAS tiny matmul warmup failed (non-fatal): {_e}")
                logger.info("Skipping Qwen safe warmup on Windows; proceeding to real forward.")
                setattr(unet, "_did_safe_warmup", True)
            except Exception as e:
                logger.warning(f"Qwen safe warmup forward failed (non-fatal): {e}")
            finally:
                try:
                    if old_benchmark is not None:
                        torch.backends.cudnn.benchmark = old_benchmark
                    if old_tf32 is not None:
                        torch.backends.cuda.matmul.allow_tf32 = old_tf32
                except Exception:
                    pass

        # Ensure device is synchronized right before actual forward to avoid stream hazards on first call
        if getattr(args, "use_qfloat8_on_demand", False) and new_device.type == "cuda":
            try:
                logger.info("CUDA preflight sync C (before real forward)...")
                torch.cuda.synchronize(new_device)
            except Exception:
                pass
        if getattr(args, "use_qfloat8_on_demand", False) and new_device.type == "cuda" and not hasattr(unet, "_did_first_forward"):
            try:
                logger.info("Qwen first real forward (fp32, no autocast, eval-mode) start...")
                was_training = getattr(unet, "training", False)
                try:
                    unet.eval()
                except Exception:
                    pass
                # Ensure inputs are contiguous to avoid any fragmented views on first call
                packed_noisy_model_input = packed_noisy_model_input.contiguous()
                prompt_embeds = prompt_embeds.contiguous()
                # Run forward on CUDA with fp32 sigmas but without autocast
                model_pred_packed = unet(
                    hidden_states=packed_noisy_model_input,
                    timestep=sigmas.float(),
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]
                try:
                    if was_training:
                        unet.train()
                except Exception:
                    pass
                setattr(unet, "_did_first_forward", True)
                logger.info("Qwen first real forward (fp32, no autocast) completed.")
            except Exception as e:
                logger.warning(f"Qwen first real forward (no autocast) failed (non-fatal): {e}; attempting CUDA autocast path.")
                with accelerator.autocast():
                    model_pred_packed = unet(
                        hidden_states=packed_noisy_model_input,
                        timestep=sigmas.float(),
                        guidance=None,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        return_dict=False,
                    )[0]
        else:
            with accelerator.autocast():
                model_pred_packed = unet(
                    hidden_states=packed_noisy_model_input,
                    timestep=sigmas,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]
        logger.info("Qwen forward end")

        model_pred_5d = QwenImagePipeline._unpack_latents(
            model_pred_packed,
            height=h * self.vae_scale_factor,
            width=w * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )
        # Free packed tensor ASAP to reduce peak memory
        del model_pred_packed
        # Convert to (B, C, H, W)
        model_pred = model_pred_5d.squeeze(2)
        del model_pred_5d

        # Build target for flow-matching: the clean latents in the same layout as model_pred
        # Convert latents (B, 1, C, H, W) -> (B, C, H, W)
        # These are inputs; ensure they don't hold graph history and reduce precision to weight_dtype when safe
        with torch.no_grad():
            target = latents.permute(0, 2, 1, 3, 4).squeeze(2)
            if target.dtype != weight_dtype:
                target = target.to(dtype=weight_dtype)

        # Free intermediates that are no longer needed to prevent VRAM bloat
        del noisy_model_input
        del sigmas_5d
        del sigmas
        del indices
        if 'indices_cpu' in locals():
            del indices_cpu
        del u
        del noise
        if 'packed_noisy_model_input' in locals():
            del packed_noisy_model_input
        if 'prompt_embeds' in locals():
            # prompt tensors are small compared to image latents, but free them anyway
            del prompt_embeds
        if 'prompt_embeds_mask' in locals():
            del prompt_embeds_mask
        if 'img_shapes' in locals():
            del img_shapes
        if 'txt_seq_lens' in locals():
            del txt_seq_lens
        # Let Accelerate handle cache trimming; still try to hint the allocator
        try:
            clean_memory_on_device(accelerator.device)
        except Exception:
            pass

        logger.info("Qwen get_noise_pred_and_target end")
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
    # Memory-saving arguments
    parser.add_argument(
        "--use_qfloat8_on_demand",
        action="store_true",
        help="[EXPERIMENTAL] quantize the Qwen transformer to qfloat8 on demand to save VRAM",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)

    if os.name == "nt":
        logger.info("Windows detected: Forcing max_data_loader_n_workers=0 and disabling persistent workers to prevent hangs.")
        args.max_data_loader_n_workers = 0
        args.persistent_data_loader_workers = False
        try:
            import torch
            torch.set_num_threads(1)
            logger.info("Set torch.set_num_threads(1) to reduce potential dataloader thread contention on Windows.")
        except Exception:
            pass

    args = train_util.read_config_from_file(args, parser)

    trainer = QwenNetworkTrainer()
    trainer.train(args)
