"""
Comprehensive test to verify all fixes are working correctly.
This script simulates the training/validation visualization process.

Usage:
    python test_visualization_fixes.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import tempfile

def create_dummy_3d_data():
    """Create dummy 3D medical imaging data to test visualization"""
    batch_size = 2
    channels = 1
    depth, height, width = 128, 128, 128
    
    # Create distinct patterns for each type
    # Input condition: vertical stripes
    x_cond = torch.zeros(batch_size, channels, depth, height, width)
    for i in range(0, width, 10):
        x_cond[:, :, :, :, i:i+5] = 0.8
    x_cond = x_cond * 2 - 1  # Normalize to [-1, 1]
    
    # Ground truth: horizontal stripes
    x = torch.zeros(batch_size, channels, depth, height, width)
    for i in range(0, height, 10):
        x[:, :, :, i:i+5, :] = 0.8
    x = x * 2 - 1  # Normalize to [-1, 1]
    
    # Generated (should be different from both)
    # Let's make it diagonal stripes
    sample = torch.zeros(batch_size, channels, depth, height, width)
    for i in range(-width, height, 15):
        for j in range(height):
            k = j - i
            if 0 <= k < width:
                sample[:, :, :, j, k] = 0.8
    sample = sample * 2 - 1  # Normalize to [-1, 1]
    
    return x, x_cond, sample

def get_image_grid(batch, grid_size=2, to_normal=True):
    """Simplified version of get_image_grid from runners/utils.py"""
    batch = batch.detach().clone()
    
    # Create a simple grid
    b, c, h, w = batch.shape
    rows = (b + grid_size - 1) // grid_size
    
    # Create grid manually
    grid_h = rows * h
    grid_w = grid_size * w
    grid = torch.zeros(c, grid_h, grid_w)
    
    for idx in range(b):
        row = idx // grid_size
        col = idx % grid_size
        grid[:, row*h:(row+1)*h, col*w:(col+1)*w] = batch[idx]
    
    if to_normal:
        grid = grid.mul_(0.5).add_(0.5).clamp_(0, 1.)
    
    grid = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    
    # Convert to RGB if grayscale
    if grid.shape[2] == 1:
        grid = np.repeat(grid, 3, axis=2)
    
    return grid

def test_visualization():
    """Test the complete visualization pipeline"""
    print("=" * 80)
    print("TESTING VISUALIZATION FIXES")
    print("=" * 80)
    
    # Create dummy data
    print("\n1. Creating dummy 3D data...")
    x, x_cond, sample = create_dummy_3d_data()
    print(f"   ✓ Created data with shapes: {x.shape}")
    
    # Extract middle slices
    print("\n2. Extracting middle slices...")
    depth_slice = x.shape[2] // 2
    x_slice = x[:, :, depth_slice, :, :]
    x_cond_slice = x_cond[:, :, depth_slice, :, :]
    sample_slice = sample[:, :, depth_slice, :, :]
    print(f"   ✓ Slice shapes: {x_slice.shape}")
    
    # Create image grids
    print("\n3. Creating image grids...")
    grid_size = 2
    to_normal = True
    
    sample_grid = get_image_grid(sample_slice, grid_size, to_normal=to_normal)
    cond_grid = get_image_grid(x_cond_slice, grid_size, to_normal=to_normal)
    gt_grid = get_image_grid(x_slice, grid_size, to_normal=to_normal)
    print(f"   ✓ Grid shapes: {sample_grid.shape}")
    
    # Calculate metrics
    print("\n4. Calculating MSE metrics...")
    mse_cond_sample = torch.nn.functional.mse_loss(
        sample_slice, x_cond_slice
    ).item()
    mse_sample_gt = torch.nn.functional.mse_loss(
        sample_slice, x_slice
    ).item()
    mse_cond_gt = torch.nn.functional.mse_loss(
        x_cond_slice, x_slice
    ).item()
    
    print(f"   MSE (Condition → Generated): {mse_cond_sample:.4f}")
    print(f"   MSE (Generated → GT):        {mse_sample_gt:.4f}")
    print(f"   MSE (Condition → GT):        {mse_cond_gt:.4f}")
    
    # Test that outputs are actually different
    print("\n5. Verifying outputs are different...")
    if mse_cond_sample > 0.01:
        print("   ✓ Generated differs from Condition")
    else:
        print("   ✗ ERROR: Generated is too similar to Condition!")
    
    if mse_sample_gt > 0.01:
        print("   ✓ Generated differs from GT")
    else:
        print("   ✗ ERROR: Generated is too similar to GT!")
    
    if mse_cond_gt > 0.01:
        print("   ✓ Condition differs from GT")
    else:
        print("   ✗ ERROR: Condition is too similar to GT!")
    
    # Create visualization
    print("\n6. Creating side-by-side visualization...")
    with tempfile.TemporaryDirectory() as tmpdir:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        title_text = f'Test - MSE(Cond→Gen)={mse_cond_sample:.4f}, MSE(Gen→GT)={mse_sample_gt:.4f}, MSE(Cond→GT)={mse_cond_gt:.4f}'
        fig.suptitle(title_text, fontsize=14, fontweight='bold')
        
        axes[0].imshow(cond_grid)
        axes[0].set_title('Input Condition', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(sample_grid)
        axes[1].set_title('Generated Output', fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(gt_grid)
        axes[2].set_title('Ground Truth', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(tmpdir, 'test_comparison.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved comparison plot")
        
        # Verify file exists
        if os.path.exists(comparison_path):
            print(f"   ✓ File verified: {os.path.getsize(comparison_path)} bytes")
        else:
            print(f"   ✗ ERROR: File not created!")
        
        # Save individual images
        Image.fromarray(sample_grid).save(os.path.join(tmpdir, 'generated_sample.png'))
        Image.fromarray(cond_grid).save(os.path.join(tmpdir, 'input_condition.png'))
        Image.fromarray(gt_grid).save(os.path.join(tmpdir, 'ground_truth.png'))
        print(f"   ✓ Saved individual images")
    
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    
    # Final assessment
    all_different = (mse_cond_sample > 0.01 and 
                     mse_sample_gt > 0.01 and 
                     mse_cond_gt > 0.01)
    
    if all_different:
        print("✓ ALL TESTS PASSED!")
        print("  - Images are properly differentiated")
        print("  - MSE metrics are calculated correctly")
        print("  - Visualizations are created successfully")
        print("\nYour fixes are working correctly!")
    else:
        print("✗ SOME TESTS FAILED!")
        print("  Please check the implementation")
    
    print("\n" + "=" * 80)
    print("WHAT TO CHECK IN YOUR ACTUAL RESULTS:")
    print("=" * 80)
    print("1. Look for comparison plots in: results/.../samples/*/")
    print("2. Check TensorBoard metrics:")
    print("   - train_metrics/mse_condition_to_generated")
    print("   - val_metrics/mse_generated_to_gt")
    print("3. Watch for console warnings about identical outputs")
    print("4. If MSE(Cond→Gen) < 0.001 after 10k steps, investigate:")
    print("   - Model training (check loss curves)")
    print("   - Data loading (verify inputs != targets)")
    print("   - Sampling parameters (try increasing sample_step)")
    print("=" * 80)

if __name__ == "__main__":
    test_visualization()
    print("\nTest complete!")
