# LoRA Plugin for Slime

This plugin provides LoRA (Low-Rank Adaptation) support for parameter-efficient fine-tuning in the slime framework.

## Features

- **Parameter-efficient training**: Train only a small fraction of parameters
- **Compatible with both backends**: Works with Megatron and FSDP training backends
- **Flexible configuration**: Customize rank, alpha, dropout, and target modules
- **Merge/unmerge support**: Optionally merge LoRA weights for inference

## Installation

The LoRA plugin is already included in `slime_plugins/lora/`. No additional installation needed.

## Usage

### Basic Configuration

Add these arguments to your training script:

```bash
# Enable LoRA
--use-lora \
--lora-r 8 \
--lora-alpha 16.0 \
--lora-dropout 0.1 \
--lora-only-trainable
```

### For Megatron Backend

1. **Modify your model initialization** in `slime/backends/megatron_utils/model.py`:

```python
from slime_plugins.lora.megatron_lora_hook import apply_lora_to_megatron_model

def initialize_model_and_optimizer(args, role="actor"):
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(args, role)
    
    # Apply LoRA after model initialization
    if getattr(args, 'use_lora', False):
        model = apply_lora_to_megatron_model(model, args)
    
    # ... rest of initialization
```

2. **Use custom init hook** (alternative method):

```bash
--custom-megatron-init-path slime_plugins/lora/megatron_lora_hook.py:custom_megatron_init
```

### For FSDP Backend

Modify `slime/backends/fsdp_utils/actor.py` to apply LoRA before FSDP wrapping:

```python
from slime_plugins.lora.fsdp_lora_hook import apply_lora_to_fsdp_model

def init(self, args, role, wandb_run_id, with_ref=False):
    # ... load model
    model = AutoModelForCausalLM.from_pretrained(...)
    
    # Apply LoRA before FSDP wrapping
    if getattr(args, 'use_lora', False):
        model = apply_lora_to_fsdp_model(model, args)
    
    # Then wrap with FSDP
    self.model = FSDP(model)
```

### Configuration Arguments

Add these to `slime/utils/arguments.py`:

```python
def add_lora_arguments(parser):
    parser.add_argument('--use-lora', action='store_true', help='Enable LoRA training')
    parser.add_argument('--lora-r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=float, default=16.0, help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.0, help='LoRA dropout')
    parser.add_argument('--lora-target-modules', type=str, nargs='+', default=None,
                       help='Target modules for LoRA (e.g., q_proj v_proj)')
    parser.add_argument('--lora-only-trainable', action='store_true', default=True,
                       help='Only train LoRA parameters')
    return parser
```

## Example Training Script

```bash
#!/bin/bash

# Megatron backend with LoRA
PYTHONPATH=/root/Megatron-LM python train.py \
   --train-backend megatron \
   --use-lora \
   --lora-r 8 \
   --lora-alpha 16.0 \
   --lora-dropout 0.1 \
   --lora-target-modules linear_qkv linear_proj \
   --lora-only-trainable \
   --lr 2e-4 \
   --global-batch-size 128 \
   # ... other args
```

## Target Modules

### Megatron Models
Common target modules:
- `linear_qkv`: Query, Key, Value projections
- `linear_proj`: Output projection in attention
- `linear_fc1`: First FC layer in MLP
- `linear_fc2`: Second FC layer in MLP

### HuggingFace Models
Common target modules:
- `q_proj`, `k_proj`, `v_proj`, `o_proj`: Attention projections
- `gate_proj`, `up_proj`, `down_proj`: MLP layers

## Advanced Usage

### Merge LoRA Weights for Inference

```python
from slime_plugins.lora.lora_layer import merge_lora_weights

# After training, merge weights for faster inference
merge_lora_weights(model)
```

### Save Only LoRA Parameters

```python
from slime_plugins.lora.lora_layer import get_lora_parameters

lora_params = get_lora_parameters(model)
torch.save({'lora': [p.cpu() for p in lora_params]}, 'lora_weights.pt')
```

## Parameter Efficiency Example

For a 7B parameter model with LoRA:
- Original parameters: 7,000,000,000
- LoRA parameters (r=8, targeting 4 modules): ~33,000,000 (0.47%)
- Memory savings: ~95% less GPU memory for gradients and optimizer states

## Tips

1. **Start with lower rank**: r=8 or r=16 often works well
2. **Adjust learning rate**: LoRA typically needs higher LR (e.g., 2e-4 vs 1e-5)
3. **Target key modules**: Focus on attention projections for best results
4. **Scale alpha with rank**: Common ratio is alpha = 2 * r

## Troubleshooting

**Q: Training is slower than expected**
- A: LoRA adds small computation overhead. Consider reducing dropout or using fewer target modules.

**Q: Loss is not decreasing**
- A: Try increasing learning rate or LoRA rank. Ensure target modules are correct.

**Q: OOM errors**
- A: While LoRA saves memory, the base model still needs to be loaded. Consider gradient checkpointing.

## References

- LoRA paper: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Slime framework: [THUDM/slime](https://github.com/THUDM/slime)
