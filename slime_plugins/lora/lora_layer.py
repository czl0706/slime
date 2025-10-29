"""LoRA layer implementation."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer that wraps a linear layer.
    
    Implements: h = W0*x + (B*A)*x * scaling
    where W0 is the frozen pretrained weight, and B, A are low-rank matrices.
    
    Supports wrapping:
    - nn.Linear
    - Transformer Engine layers (TEColumnParallelLinear, TERowParallelLinear, etc.)
    """
    
    def __init__(
        self,
        base_layer: nn.Module,  # Changed from nn.Linear to nn.Module
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
        init_lora_weights: bool = True,
    ):
        """
        Args:
            base_layer: The base linear layer to adapt (nn.Linear or TE layer).
            r: Rank of the low-rank adaptation.
            lora_alpha: Scaling factor for LoRA.
            lora_dropout: Dropout rate for LoRA.
            merge_weights: Whether to merge LoRA weights into base layer.
            init_lora_weights: Whether to initialize LoRA weights.
        """
        super().__init__()
        
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.merged = False
        self.merge_weights = merge_weights
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # IMPORTANT: Ensure base_layer.weight retains its TP attributes after wrapping
        # This is critical for weight updates to work correctly
        if hasattr(base_layer, 'weight') and hasattr(base_layer.weight, 'tensor_model_parallel'):
            # Store reference to ensure TP attributes are preserved
            self._base_weight_tp_attrs = {
                'tensor_model_parallel': getattr(base_layer.weight, 'tensor_model_parallel', False),
                'partition_dim': getattr(base_layer.weight, 'partition_dim', -1),
                'partition_stride': getattr(base_layer.weight, 'partition_stride', 1),
                'parallel_mode': getattr(base_layer.weight, 'parallel_mode', None),
            }
        
        # Get input and output features - ALWAYS use weight shape for correct sharding
        # For TE column/row parallel layers, in_features/out_features may return global dims
        # but we need the local/sharded dims to match the actual computation
        if hasattr(base_layer, 'weight'):
            # Use actual weight shape (handles TP sharding correctly)
            out_features, in_features = base_layer.weight.shape
        elif hasattr(base_layer, 'in_features') and hasattr(base_layer, 'out_features'):
            # Fallback for layers without weight attribute
            in_features = base_layer.in_features
            out_features = base_layer.out_features
        else:
            raise ValueError(f"Cannot determine input/output features for {type(base_layer)}")
        
        # Get device and dtype from base layer
        device = None
        dtype = None
        if hasattr(base_layer, 'weight'):
            device = base_layer.weight.device
            dtype = base_layer.weight.dtype
        
        # LoRA matrices - create on same device and dtype as base layer
        self.lora_A = nn.Parameter(torch.zeros(r, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype))
        
        # Copy tensor parallel attributes from base layer weight to LoRA parameters
        # This ensures LoRA params are all-gathered correctly during weight updates
        if hasattr(base_layer, 'weight'):
            base_weight = base_layer.weight
            # Copy TP attributes for lora_A (input dimension sharding)
            if hasattr(base_weight, 'tensor_model_parallel') and base_weight.tensor_model_parallel:
                self.lora_A.tensor_model_parallel = True
                self.lora_A.partition_dim = 1  # lora_A is sharded on input dimension (dim 1: r x in_features)
                self.lora_A.partition_stride = getattr(base_weight, 'partition_stride', 1)
                if hasattr(base_weight, 'parallel_mode'):
                    self.lora_A.parallel_mode = base_weight.parallel_mode
                print(f"[LoRA] Set TP attrs for lora_A: partition_dim=1, tp={self.lora_A.tensor_model_parallel}")
            else:
                # Non-TP layer, mark as non-parallel
                self.lora_A.tensor_model_parallel = False
            
            # Copy TP attributes for lora_B (output dimension sharding)  
            if hasattr(base_weight, 'tensor_model_parallel') and base_weight.tensor_model_parallel:
                self.lora_B.tensor_model_parallel = True
                self.lora_B.partition_dim = 0  # lora_B is sharded on output dimension (dim 0: out_features x r)
                self.lora_B.partition_stride = getattr(base_weight, 'partition_stride', 1)
                if hasattr(base_weight, 'parallel_mode'):
                    self.lora_B.parallel_mode = base_weight.parallel_mode
                print(f"[LoRA] Set TP attrs for lora_B: partition_dim=0, tp={self.lora_B.tensor_model_parallel}")
            else:
                # Non-TP layer, mark as non-parallel
                self.lora_B.tensor_model_parallel = False
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        if init_lora_weights:
            self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA weights."""
        # Initialize A with kaiming uniform (same as nn.Linear)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initialize B to zero so that LoRA starts as identity
        nn.init.zeros_(self.lora_B)
    
    def merge(self):
        """Merge LoRA weights into base layer for inference."""
        if not self.merged and self.r > 0:
            # W_new = W_base + B @ A * scaling
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w
            self.merged = True
    
    def unmerge(self):
        """Separate LoRA weights from base layer."""
        if self.merged and self.r > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= delta_w
            self.merged = False
    
    def forward(self, x: torch.Tensor):
        """Forward pass with LoRA adaptation."""
        # Base layer output
        result = self.base_layer(x)
        
        # Handle Transformer Engine layers that return (output, bias) tuple
        is_tuple_output = isinstance(result, tuple)
        if is_tuple_output:
            base_output, bias = result
        else:
            base_output = result
            bias = None
        
        if self.r > 0 and not self.merged:
            # Add LoRA adaptation: x @ A^T @ B^T * scaling
            lora_output = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t() * self.scaling
            base_output = base_output + lora_output
        
        # Return in same format as base layer
        if is_tuple_output:
            return (base_output, bias)
        else:
            return base_output
    
    def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True):
        """Override named_parameters to ensure TP attributes are preserved on base_layer.weight.
        
        This is critical for weight updates to work correctly with TP-sharded layers.
        When the model is wrapped with LoRALinear, the base_layer.weight must retain
        its TP attributes so that all_gather_params_async() can gather it correctly.
        """
        import torch.distributed as dist
        for name, param in super().named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate):
            # Check if this is the base_layer.weight and needs TP attributes restored
            if name.endswith('base_layer.weight') and hasattr(self, '_base_weight_tp_attrs'):
                # Restore TP attributes that may have been lost during wrapping
                for attr_name, attr_value in self._base_weight_tp_attrs.items():
                    if attr_value is not None:
                        setattr(param, attr_name, attr_value)
                
                # Debug: verify TP attrs are set
                if dist.is_initialized() and dist.get_rank() == 0:
                    tp_enabled = getattr(param, 'tensor_model_parallel', False)
                    partition_dim = getattr(param, 'partition_dim', -1)
                    print(f"[LoRA named_parameters] Restored TP attrs for {name}: tp={tp_enabled}, partition_dim={partition_dim}, shape={param.shape}")
            yield name, param
    
    def extra_repr(self) -> str:
        """Extra representation string."""
        return f"r={self.r}, lora_alpha={self.lora_alpha}, merged={self.merged}"
    
    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """
        Return sharded state dict for Megatron checkpointing.
        
        This method is required for Megatron's distributed checkpointing system.
        We make the LoRA layer transparent - the checkpoint uses the same keys
        as if LoRA wasn't applied. This allows loading non-LoRA checkpoints.
        """
        # Get base layer's sharded state dict if it has one
        # CRITICAL: Don't add 'base_layer.' prefix - keep original keys
        if hasattr(self.base_layer, 'sharded_state_dict'):
            state_dict = self.base_layer.sharded_state_dict(
                prefix=prefix,  # Use the same prefix, not prefix + 'base_layer.'
                sharded_offsets=sharded_offsets,
                metadata=metadata
            )
        else:
            # Fallback: use regular state_dict
            state_dict = {
                f'{prefix}{k}': v 
                for k, v in self.base_layer.state_dict().items()
            }
        
        # Add LoRA parameters (these are not sharded, they're small)
        # Use non-conflicting names
        state_dict[f'{prefix}lora_A'] = self.lora_A
        state_dict[f'{prefix}lora_B'] = self.lora_B
        
        return state_dict
    
    def state_dict(self, *args, **kwargs):
        """
        Override state_dict to be transparent.
        
        Returns base layer state dict with same keys, plus LoRA parameters.
        This allows loading non-LoRA checkpoints into LoRA models.
        """
        state_dict = {}
        
        # Add base layer state dict WITHOUT 'base_layer.' prefix
        base_state_dict = self.base_layer.state_dict(*args, **kwargs)
        for k, v in base_state_dict.items():
            state_dict[k] = v  # Keep original keys
        
        # Add LoRA parameters with distinct names
        state_dict['lora_A'] = self.lora_A
        state_dict['lora_B'] = self.lora_B
        
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to handle loading from non-LoRA checkpoints.
        
        If LoRA parameters are present, load them. If not, just load base layer.
        """
        # Separate LoRA parameters from base layer parameters
        lora_keys = {'lora_A', 'lora_B'}
        base_state_dict = {k: v for k, v in state_dict.items() if k not in lora_keys}
        lora_state_dict = {k: v for k, v in state_dict.items() if k in lora_keys}
        
        # Load base layer state dict
        if base_state_dict:
            # When loading from non-LoRA checkpoint, strict=False to ignore missing LoRA params
            self.base_layer.load_state_dict(base_state_dict, strict=False)
        
        # Load LoRA parameters if present (optional - may not exist in checkpoint)
        if 'lora_A' in lora_state_dict:
            self.lora_A.data.copy_(lora_state_dict['lora_A'])
        if 'lora_B' in lora_state_dict:
            self.lora_B.data.copy_(lora_state_dict['lora_B'])


def apply_lora_to_model(
    model: nn.Module,
    target_modules: list[str],
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    merge_weights: bool = False,
) -> nn.Module:
    """
    Apply LoRA to specified modules in a model.
    
    Args:
        model: The model to apply LoRA to.
        target_modules: List of module names to target (e.g., ["q_proj", "v_proj"]).
        r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: Dropout rate for LoRA.
        merge_weights: Whether to merge weights at inference.
    
    Returns:
        Modified model with LoRA layers.
    """
    # Count how many LoRA layers we actually apply
    applied_count = 0
    
    for name, module in model.named_modules():
        # Check if this module matches any target pattern
        if any(target in name for target in target_modules):
            # Check if it's a Linear layer (nn.Linear or Transformer Engine variants)
            is_linear = isinstance(module, nn.Linear)
            
            # Also check for Transformer Engine linear layers
            if not is_linear and hasattr(module, '__class__'):
                module_class_name = module.__class__.__name__
                is_te_linear = any(te_name in module_class_name for te_name in [
                    'TEColumnParallelLinear',
                    'TERowParallelLinear', 
                    'TELayerNormColumnParallelLinear',
                    'TELayerNormLinear',
                ])
                is_linear = is_linear or is_te_linear
            
            if is_linear:
                # Get parent module and attribute name
                *parent_names, attr_name = name.split('.')
                parent = model
                for parent_name in parent_names:
                    parent = getattr(parent, parent_name)
                
                # Replace with LoRA layer
                lora_layer = LoRALinear(
                    base_layer=module,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    merge_weights=merge_weights,
                )
                setattr(parent, attr_name, lora_layer)
                
                applied_count += 1
                print(f"✓ Applied LoRA to: {name} ({module.__class__.__name__})")
    
    if applied_count == 0:
        print(f"⚠️  WARNING: No LoRA layers were applied! Check your target_modules: {target_modules}")
    else:
        print(f"✓ Successfully applied LoRA to {applied_count} layers")
    
    return model


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Get only LoRA parameters from the model."""
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """Freeze all parameters except LoRA parameters."""
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Then unfreeze LoRA parameters
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True


def merge_lora_weights(model: nn.Module) -> None:
    """Merge all LoRA weights in the model."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora_weights(model: nn.Module) -> None:
    """Unmerge all LoRA weights in the model."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()
