import torch
from .lora import LoRAModule, LoRANetwork
from typing import List, Optional
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class QwenLoRANetwork(LoRANetwork):
    def __init__(self, text_encoder, unet, **kwargs):
        # text_encoder is a pipeline. get the text_encoder model from it
        if hasattr(text_encoder, "text_encoder"):
            actual_text_encoder = text_encoder.text_encoder
        else:
            actual_text_encoder = text_encoder
        super().__init__(actual_text_encoder, unet, **kwargs)

    def create_modules(
        self,
        is_unet: bool,
        text_encoder_idx: Optional[int],  # None, 1, 2
        root_module: torch.nn.Module,
        target_replace_modules: List[torch.nn.Module],
    ) -> List[LoRAModule]:
        prefix = self.LORA_PREFIX_UNET if is_unet else self.LORA_PREFIX_TEXT_ENCODER
        loras = []
        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ == "Linear":
                        if child_name in ["to_k", "to_q", "to_v"] or child_name.endswith("to_out.0"):
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")
                            lora = LoRAModule(lora_name, child_module, self.multiplier, self.lora_dim, self.alpha)
                            loras.append(lora)
        return loras


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
