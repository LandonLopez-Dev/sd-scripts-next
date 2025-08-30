import torch
from .lora import LoRAModule, LoRANetwork
from typing import List, Optional
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class QwenLoRANetwork(LoRANetwork):
    # Qwen's attention block class name is Qwen2Attention
    UNET_TARGET_REPLACE_MODULE = ["Qwen2Attention"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["Qwen2Attention"]

    def __init__(self, text_encoder, unet, **kwargs):
        # The base LoRANetwork handles a list of text_encoders
        super().__init__(text_encoder, unet, **kwargs)


def create_network(
    multiplier,
    network_dim,
    network_alpha,
    vae,
    text_encoder,
    unet,
    neuron_dropout=None,
    **kwargs,
):
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = network_dim

    # get dropout argument
    if neuron_dropout is None:
        neuron_dropout = kwargs.pop('dropout', None)
    else:
        # neuron_dropout has priority, so remove dropout from kwargs
        kwargs.pop('dropout', None)

    network = QwenLoRANetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        **kwargs,
    )
    return network
