import toml
import argparse
from library.train_util import (
    DreamBoothSubset,
    FineTuningSubset,
    DreamBoothDataset,
)
from torch.utils.data import DataLoader

class LoraDataset(DreamBoothDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def load_dataset_from_config(config: dict, args: argparse.Namespace):
    if "general" in config:
        general_config = config["general"]
    else:
        general_config = {}

    subsets = []
    for subset_config in config.get("datasets", []):
        is_reg = subset_config.get("is_reg", False)
        if "class_tokens" in subset_config and not is_reg:
            raise ValueError("class_tokens is only for regularization images")

        subset = DreamBoothSubset(
            image_dir=subset_config.get("image_dir"),
            num_repeats=subset_config.get("num_repeats", 1),
            is_reg=is_reg,
            class_tokens=subset_config.get("class_tokens"),
            caption_extension=subset_config.get("caption_extension", ".txt"),
            cache_info=subset_config.get("cache_info", False),
            alpha_mask=subset_config.get("alpha_mask", False),
            shuffle_caption=subset_config.get("shuffle_caption", False),
            caption_separator=subset_config.get("caption_separator", ","),
            keep_tokens=subset_config.get("keep_tokens", 0),
            keep_tokens_separator=subset_config.get("keep_tokens_separator", ""),
            secondary_separator=subset_config.get("secondary_separator"),
            enable_wildcard=subset_config.get("enable_wildcard", False),
            color_aug=subset_config.get("color_aug", False),
            flip_aug=subset_config.get("flip_aug", False),
            face_crop_aug_range=subset_config.get("face_crop_aug_range"),
            random_crop=subset_config.get("random_crop", False),
            caption_dropout_rate=subset_config.get("caption_dropout_rate", 0.0),
            caption_dropout_every_n_epochs=subset_config.get("caption_dropout_every_n_epochs", 0),
            caption_tag_dropout_rate=subset_config.get("caption_tag_dropout_rate", 0.0),
            caption_prefix=subset_config.get("caption_prefix"),
            caption_suffix=subset_config.get("caption_suffix"),
            token_warmup_min=subset_config.get("token_warmup_min", 0),
            token_warmup_step=subset_config.get("token_warmup_step", 0),
            validation_seed=subset_config.get("validation_seed"),
            validation_split=subset_config.get("validation_split", 0.0),
            resize_interpolation=subset_config.get("resize_interpolation"),
        )
        subsets.append(subset)

    dataset = LoraDataset(
        subsets=subsets,
        is_training_dataset=True, # for now, only training
        batch_size=args.train_batch_size,
        resolution=args.resolution,
        network_multiplier=1.0, # will be handled later
        enable_bucket=args.enable_bucket,
        min_bucket_reso=args.min_bucket_reso,
        max_bucket_reso=args.max_bucket_reso,
        bucket_reso_steps=args.bucket_reso_steps,
        bucket_no_upscale=args.bucket_no_upscale,
        prior_loss_weight=1.0,
        debug_dataset=args.debug_dataset,
        validation_split=0.0, # handled per subset
        validation_seed=None, # handled per subset
        resize_interpolation=general_config.get("resize_interpolation"),
    )

    return dataset


def setup_dataloader(config: dict, args: argparse.Namespace):
    dataset = load_dataset_from_config(config, args)
    dataset.make_buckets()

    return DataLoader(
        dataset,
        batch_size=1, # batching is handled inside the dataset
        shuffle=True,
        collate_fn=lambda examples: examples[0],
        num_workers=args.max_data_loader_n_workers,
        pin_memory=True,
    )
