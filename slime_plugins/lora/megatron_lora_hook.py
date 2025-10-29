"""
Hook for integrating LoRA into Megatron training.

This module provides hooks to apply LoRA to Megatron models after initialization.
Usage: Set --custom-megatron-init-path to point to this file.
"""

import torch.nn as nn
from slime_plugins.lora import apply_lora_to_model, mark_only_lora_as_trainable


def custom_megatron_init(args):
    """
    Custom initialization hook for Megatron with LoRA.
    
    This function is called during Megatron initialization if specified via
    --custom-megatron-init-path argument.
    
    Args:
        args: Megatron arguments namespace.
    """
    print("=" * 80)
    print("Initializing LoRA for Megatron training")
    print("=" * 80)
    
    # Check if LoRA is enabled
    if not getattr(args, 'use_lora', False):
        print("LoRA not enabled. Skipping LoRA initialization.")
        return
    
    # Get LoRA configuration from args
    lora_r = getattr(args, 'lora_r', 8)
    lora_alpha = getattr(args, 'lora_alpha', 16.0)
    lora_dropout = getattr(args, 'lora_dropout', 0.0)
    lora_target_modules = getattr(args, 'lora_target_modules', None)
    
    if lora_target_modules is None:
        # Default target modules for transformer models
        lora_target_modules = ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2']
    
    print(f"LoRA Configuration:")
    print(f"  - Rank (r): {lora_r}")
    print(f"  - Alpha: {lora_alpha}")
    print(f"  - Dropout: {lora_dropout}")
    print(f"  - Target modules: {lora_target_modules}")
    print("=" * 80)


def apply_lora_to_megatron_model(model, args):
    """
    Apply LoRA to a Megatron model after it's initialized.
    
    This should be called after model initialization but before training begins.
    
    Args:
        model: Megatron model (list of DDP wrapped model chunks).
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
        lora_target_modules = ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2']
    
    # Apply LoRA to each model chunk
    for i, model_chunk in enumerate(model):
        # Get the actual module (unwrap from DDP if needed)
        actual_module = model_chunk.module if hasattr(model_chunk, 'module') else model_chunk
        
        print(f"\nApplying LoRA to model chunk {i}...")
        apply_lora_to_model(
            actual_module,
            target_modules=lora_target_modules,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=False,
        )
        
        # Mark only LoRA parameters as trainable
        if getattr(args, 'lora_only_trainable', True):
            mark_only_lora_as_trainable(actual_module)
    
    # Count trainable parameters
    total_params = sum(p.numel() for model_chunk in model for p in model_chunk.parameters())
    trainable_params = sum(p.numel() for model_chunk in model for p in model_chunk.parameters() if p.requires_grad)
    
    print("\n" + "=" * 80)
    print(f"LoRA applied successfully!")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Trainable %: {100 * trainable_params / total_params:.2f}%")
    print("=" * 80 + "\n")
    
    return model
