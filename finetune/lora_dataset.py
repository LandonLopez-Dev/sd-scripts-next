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

    datasets_list = config.get("datasets")

    # Support flat config that directly specifies a single directory
    if datasets_list is None and ("train_data_dir" in config or "image_dir" in config or "data_dir" in config):
        # allow flat configs with a single directory
        datasets_list = [{k: v} for k, v in config.items() if k in (
            "train_data_dir",
            "data_dir",
            "image_dir",
            "images",
            "path",
            "dir",
            "num_repeats",
            "caption_extension",
            "cache_info",
            "alpha_mask",
            "shuffle_caption",
            "caption_separator",
            "keep_tokens",
            "keep_tokens_separator",
            "secondary_separator",
            "enable_wildcard",
            "color_aug",
            "flip_aug",
            "face_crop_aug_range",
            "random_crop",
            "caption_dropout_rate",
            "caption_dropout_every_n_epochs",
            "caption_tag_dropout_rate",
            "caption_prefix",
            "caption_suffix",
            "token_warmup_min",
            "token_warmup_step",
            "validation_seed",
            "validation_split",
            "resize_interpolation",
            "is_reg",
            "class_tokens",
        )]
        # collapse to single dict
        merged = {}
        for d in datasets_list:
            merged.update(d)
        datasets_list = [merged]

    # Extract dataset-level overrides (use the first [[datasets]] block if present)
    ds_level = (datasets_list[0] if isinstance(datasets_list, list) and datasets_list else {})

    def add_subset_from_config(subset_config: dict):
        is_reg = subset_config.get("is_reg", False)
        if "class_tokens" in subset_config and is_reg:
            raise ValueError("class_tokens is only for regularization images")

        # Resolve image_dir with backward-compatible keys
        image_dir = subset_config.get("image_dir")
        if image_dir is None:
            # common alternatives used in older configs or other tools
            for alt_key in ("train_data_dir", "data_dir", "images", "path", "dir"):
                if alt_key in subset_config:
                    image_dir = subset_config.get(alt_key)
                    break
        subset = DreamBoothSubset(
            image_dir=image_dir,
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

    # Build subsets, supporting both [[datasets]] and [[datasets.subsets]] schemas
    if isinstance(datasets_list, list):
        for ds in datasets_list:
            if isinstance(ds, dict) and "subsets" in ds and isinstance(ds["subsets"], list):
                for sub in ds["subsets"]:
                    add_subset_from_config(sub)
            else:
                # treat the [[datasets]] item itself as a subset definition
                add_subset_from_config(ds)

    # Apply dataset-level overrides falling back to args/general
    # resolution may be specified as array [w, h] or comma string "w,h"
    resolution = args.resolution
    if isinstance(ds_level.get("resolution"), list) and len(ds_level["resolution"]) == 2:
        resolution = tuple(map(int, ds_level["resolution"]))
    elif isinstance(ds_level.get("resolution"), int):
        resolution = (int(ds_level["resolution"]), int(ds_level["resolution"]))
    elif isinstance(ds_level.get("resolution"), str):
        try:
            resolution = tuple(map(int, ds_level["resolution"].split(",")))
        except Exception:
            pass

    batch_size = int(ds_level.get("batch_size", args.train_batch_size))
    enable_bucket = bool(ds_level.get("enable_bucket", args.enable_bucket))
    min_bucket_reso = int(ds_level.get("min_bucket_reso", args.min_bucket_reso))
    max_bucket_reso = int(ds_level.get("max_bucket_reso", args.max_bucket_reso))
    bucket_reso_steps = int(ds_level.get("bucket_reso_steps", args.bucket_reso_steps))
    bucket_no_upscale = bool(ds_level.get("bucket_no_upscale", args.bucket_no_upscale))

    dataset = LoraDataset(
        subsets=subsets,
        is_training_dataset=True,  # for now, only training
        batch_size=batch_size,
        resolution=resolution,
        network_multiplier=1.0,  # will be handled later
        enable_bucket=enable_bucket,
        min_bucket_reso=min_bucket_reso,
        max_bucket_reso=max_bucket_reso,
        bucket_reso_steps=bucket_reso_steps,
        bucket_no_upscale=bucket_no_upscale,
        prior_loss_weight=1.0,
        debug_dataset=args.debug_dataset,
        validation_split=0.0,  # handled per subset
        validation_seed=None,  # handled per subset
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
