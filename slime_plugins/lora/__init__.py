"""
LoRA (Low-Rank Adaptation) plugin for slime.

This plugin provides LoRA functionality for parameter-efficient fine-tuning
in both Megatron and FSDP training backends.
"""

from .lora_layer import LoRALinear, apply_lora_to_model, mark_only_lora_as_trainable
from .lora_config import LoRAConfig

__all__ = ["LoRALinear", "apply_lora_to_model", "mark_only_lora_as_trainable", "LoRAConfig"]
