"""
Hook for integrating LoRA into FSDP training.

This module provides hooks to apply LoRA to FSDP models after initialization.
"""

from slime_plugins.lora import apply_lora_to_model, mark_only_lora_as_trainable


def apply_lora_to_fsdp_model(model, args):
    """
    Apply LoRA to an FSDP model before wrapping with FSDP.
    
    This should be called after model initialization but before FSDP wrapping.
    
    Args:
        model: HuggingFace model (before FSDP wrapping).
        args: Training arguments.
    
    Returns:
        Modified model with LoRA layers.
    """
    if not getattr(args, 'use_lora', False):
        return model
    
    lora_r = getattr(args, 'lora_r', 8)
    lora_alpha = getattr(args, 'lora_alpha', 16.0)
    lora_dropout = getattr(args, 'lora_dropout', 0.0)
    lora_target_modules = getattr(args, 'lora_target_modules', None)
    
    if lora_target_modules is None:
        # Default target modules for HuggingFace transformer models
        lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    
    print("=" * 80)
    print("Applying LoRA to FSDP model")
    print(f"  - Rank (r): {lora_r}")
    print(f"  - Alpha: {lora_alpha}")
    print(f"  - Dropout: {lora_dropout}")
    print(f"  - Target modules: {lora_target_modules}")
    print("=" * 80)
    
    # Apply LoRA
    apply_lora_to_model(
        model,
        target_modules=lora_target_modules,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        merge_weights=False,
    )
    
    # Mark only LoRA parameters as trainable
    if getattr(args, 'lora_only_trainable', True):
        mark_only_lora_as_trainable(model)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "=" * 80)
    print(f"LoRA applied successfully!")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Trainable %: {100 * trainable_params / total_params:.2f}%")
    print("=" * 80 + "\n")
    
    return model
