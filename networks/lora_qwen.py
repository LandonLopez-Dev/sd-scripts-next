import math
import torch
from .lora import LoRAModule, LoRANetwork
from typing import List, Optional, Union, Dict, Type
from transformers import CLIPTextModel

from library.utils import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)


class QwenLoRANetwork(LoRANetwork):
    def __init__(
        self,
        text_encoder: Union[List[CLIPTextModel], CLIPTextModel],
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        **kwargs,
    ):
        # The original __init__ is complex and contains a nested function that cannot be overridden.
        # We must copy and modify it directly.
        super(LoRANetwork, self).__init__() # Call grandparent's init
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None

        logger.info(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
        logger.info(f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}")

        # create module instances
        def create_modules(prefix, root_module: torch.nn.Module) -> List[LoRAModule]:
            loras = []
            for name, module in root_module.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # Broaden matcher to catch Qwen projection names
                    if any(t in name for t in ["to_q", "to_k", "to_v", "to_out.0", "q_proj", "k_proj", "v_proj", "o_proj", "proj", "fc1", "fc2"]):
                        lora_name = prefix + '_' + name.replace('.', '_')
                        loras.append(LoRAModule(
                            lora_name, module, self.multiplier, self.lora_dim, self.alpha,
                            self.dropout, self.rank_dropout, self.module_dropout
                        ))
            return loras

        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

        self.text_encoder_loras = []
        for i, te in enumerate(text_encoders):
            prefix = self.LORA_PREFIX_TEXT_ENCODER
            if len(text_encoders) > 1:
                prefix += f"_{i+1}"
            self.text_encoder_loras.extend(create_modules(prefix, te))

        logger.info(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

        self.unet_loras = create_modules(self.LORA_PREFIX_UNET, unet)
        logger.info(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

        self.block_lr_weight = None
        self.block_lr = False


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
