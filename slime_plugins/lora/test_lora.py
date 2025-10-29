"""
Simple test script to verify LoRA implementation.

Run this to check if LoRA is working correctly:
    python slime_plugins/lora/test_lora.py
"""

import torch
import torch.nn as nn
from lora_layer import LoRALinear, apply_lora_to_model, mark_only_lora_as_trainable


class SimpleModel(nn.Module):
    """Simple model for testing LoRA."""
    
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 512)
        self.k_proj = nn.Linear(512, 512)
        self.v_proj = nn.Linear(512, 512)
        self.o_proj = nn.Linear(512, 512)
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 512)
    
    def forward(self, x):
        # Simple forward pass
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_out = self.o_proj(q + k + v)
        mlp_out = self.fc2(torch.relu(self.fc1(attn_out)))
        return mlp_out


def test_lora_basic():
    """Test basic LoRA functionality."""
    print("=" * 80)
    print("Test 1: Basic LoRA Layer")
    print("=" * 80)
    
    # Create a linear layer
    base_layer = nn.Linear(512, 512)
    
    # Wrap with LoRA
    lora_layer = LoRALinear(base_layer, r=8, lora_alpha=16.0)
    
    # Test forward pass
    x = torch.randn(4, 32, 512)
    output = lora_layer(x)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ LoRA rank: {lora_layer.r}")
    print(f"✓ LoRA scaling: {lora_layer.scaling}")
    
    # Check parameters
    base_params = sum(p.numel() for p in base_layer.parameters() if p.requires_grad)
    lora_params = lora_layer.lora_A.numel() + lora_layer.lora_B.numel()
    
    print(f"✓ Base layer trainable params: {base_params} (should be 0)")
    print(f"✓ LoRA params: {lora_params}")
    print(f"✓ Param ratio: {lora_params / (base_params + lora_params) * 100:.2f}%")
    
    assert base_params == 0, "Base layer should be frozen!"
    print("✓ Test 1 PASSED!\n")


def test_lora_model():
    """Test applying LoRA to a full model."""
    print("=" * 80)
    print("Test 2: Apply LoRA to Model")
    print("=" * 80)
    
    model = SimpleModel()
    
    # Count original parameters
    original_params = sum(p.numel() for p in model.parameters())
    original_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Before LoRA:")
    print(f"  Total params: {original_params:,}")
    print(f"  Trainable params: {original_trainable:,}")
    
    # Apply LoRA
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    apply_lora_to_model(model, target_modules, r=8, lora_alpha=16.0)
    mark_only_lora_as_trainable(model)
    
    # Count parameters after LoRA
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nAfter LoRA:")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params / total_params * 100:.2f}%")
    
    # Test forward pass
    x = torch.randn(2, 16, 512)
    output = model(x)
    
    print(f"\n✓ Forward pass successful")
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    
    # Verify only LoRA params are trainable
    assert trainable_params < total_params * 0.1, "Too many trainable params!"
    print("✓ Test 2 PASSED!\n")


def test_lora_merge():
    """Test merging LoRA weights."""
    print("=" * 80)
    print("Test 3: Merge LoRA Weights")
    print("=" * 80)
    
    # Create layer with LoRA
    base_layer = nn.Linear(512, 512)
    original_weight = base_layer.weight.data.clone()
    
    lora_layer = LoRALinear(base_layer, r=8, lora_alpha=16.0)
    
    # Forward pass before merge
    x = torch.randn(4, 512)
    output_before = lora_layer(x)
    
    # Merge weights
    lora_layer.merge()
    print("✓ Weights merged")
    
    # Forward pass after merge
    output_after = lora_layer(x)
    
    # Check if outputs are the same
    diff = (output_before - output_after).abs().max().item()
    print(f"✓ Max difference in output: {diff:.6e}")
    
    assert diff < 1e-5, "Outputs should be identical after merge!"
    
    # Check weight was actually modified
    weight_diff = (original_weight - base_layer.weight.data).abs().sum().item()
    print(f"✓ Weight modification: {weight_diff:.6e}")
    
    assert weight_diff > 0, "Weights should be modified after merge!"
    
    # Test unmerge
    lora_layer.unmerge()
    print("✓ Weights unmerged")
    
    # Check weight is back to original
    weight_diff_after = (original_weight - base_layer.weight.data).abs().sum().item()
    print(f"✓ Weight difference after unmerge: {weight_diff_after:.6e}")
    
    assert weight_diff_after < 1e-5, "Weights should be back to original after unmerge!"
    print("✓ Test 3 PASSED!\n")


def test_lora_gradient():
    """Test that gradients flow correctly through LoRA."""
    print("=" * 80)
    print("Test 4: Gradient Flow")
    print("=" * 80)
    
    base_layer = nn.Linear(128, 128)
    lora_layer = LoRALinear(base_layer, r=4, lora_alpha=8.0)
    
    # Forward and backward
    x = torch.randn(8, 128)
    output = lora_layer(x)
    loss = output.mean()
    loss.backward()
    
    # Check gradients
    print(f"✓ Loss: {loss.item():.6f}")
    
    # Base layer should not have gradients
    assert base_layer.weight.grad is None or (base_layer.weight.grad == 0).all(), \
        "Base layer should not receive gradients!"
    print("✓ Base layer frozen (no gradients)")
    
    # LoRA layers should have gradients
    assert lora_layer.lora_A.grad is not None, "LoRA A should have gradients!"
    assert lora_layer.lora_B.grad is not None, "LoRA B should have gradients!"
    
    grad_a_norm = lora_layer.lora_A.grad.norm().item()
    grad_b_norm = lora_layer.lora_B.grad.norm().item()
    
    print(f"✓ LoRA A gradient norm: {grad_a_norm:.6f}")
    print(f"✓ LoRA B gradient norm: {grad_b_norm:.6f}")
    
    assert grad_a_norm > 0, "LoRA A should have non-zero gradients!"
    assert grad_b_norm > 0, "LoRA B should have non-zero gradients!"
    
    print("✓ Test 4 PASSED!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("LoRA Implementation Test Suite")
    print("=" * 80 + "\n")
    
    try:
        test_lora_basic()
        test_lora_model()
        test_lora_merge()
        test_lora_gradient()
        
        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nLoRA implementation is working correctly!")
        print("You can now integrate it into your training pipeline.")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
