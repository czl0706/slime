"""LoRA configuration."""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    
    # LoRA rank
    r: int = 8
    
    # LoRA alpha (scaling factor)
    lora_alpha: float = 16.0
    
    # Dropout rate for LoRA layers
    lora_dropout: float = 0.0
    
    # Target modules to apply LoRA (e.g., ["q_proj", "v_proj", "k_proj", "o_proj"])
    target_modules: Optional[List[str]] = None
    
    # Whether to merge LoRA weights into base model at inference time
    merge_weights: bool = False
    
    # Whether to use bias in LoRA layers
    bias: str = "none"  # "none", "all", or "lora_only"
    
    # Initialize LoRA A with kaiming uniform
    init_lora_weights: bool = True
    
    @property
    def scaling(self) -> float:
        """Get the LoRA scaling factor."""
        return self.lora_alpha / self.r if self.r > 0 else 1.0
