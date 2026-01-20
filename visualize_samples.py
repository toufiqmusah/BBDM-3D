"""
Visual comparison tool for debugging identical input/output issue.
This script loads and displays sample images side-by-side with pixel difference maps.

Usage:
    python visualize_samples.py --sample_dir results/.../samples/1000/val_sample
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def load_and_compare(sample_dir):
    """
    Load sample images and create detailed comparison visualization
    """
    sample_path = Path(sample_dir)
    
    # Try to load the new filenames first, then fall back to old ones
    try:
        condition_path = sample_path / 'input_condition.png'
        if not condition_path.exists():
            condition_path = sample_path / 'condition.png'
        
        generated_path = sample_path / 'generated_sample.png'
        if not generated_path.exists():
            generated_path = sample_path / 'skip_sample.png'
        
        gt_path = sample_path / 'ground_truth.png'
        
        # Load images
        condition = np.array(Image.open(condition_path))
        generated = np.array(Image.open(generated_path))
        ground_truth = np.array(Image.open(gt_path))
        
        print(f"Loaded images from: {sample_path}")
        print(f"  - Condition shape: {condition.shape}")
        print(f"  - Generated shape: {generated.shape}")
        print(f"  - Ground truth shape: {ground_truth.shape}")
        
    except Exception as e:
        print(f"Error loading images: {e}")
        print(f"Looking in: {sample_path}")
        print(f"Available files: {list(sample_path.glob('*.png'))}")
        return
    
    # Calculate difference maps
    diff_cond_gen = np.abs(condition.astype(float) - generated.astype(float))
    diff_gen_gt = np.abs(generated.astype(float) - ground_truth.astype(float))
    diff_cond_gt = np.abs(condition.astype(float) - ground_truth.astype(float))
    
    # Calculate statistics
    mse_cond_gen = np.mean((condition.astype(float) - generated.astype(float)) ** 2)
    mse_gen_gt = np.mean((generated.astype(float) - ground_truth.astype(float)) ** 2)
    mse_cond_gt = np.mean((condition.astype(float) - ground_truth.astype(float)) ** 2)
    
    max_diff_cond_gen = np.max(diff_cond_gen)
    mean_diff_cond_gen = np.mean(diff_cond_gen)
    
    print("\n" + "="*80)
    print("PIXEL DIFFERENCE STATISTICS")
    print("="*80)
    print(f"MSE (Condition → Generated): {mse_cond_gen:.4f}")
    print(f"MSE (Generated → GT):        {mse_gen_gt:.4f}")
    print(f"MSE (Condition → GT):        {mse_cond_gt:.4f}")
    print(f"\nMax pixel difference (Condition vs Generated): {max_diff_cond_gen:.2f}")
    print(f"Mean pixel difference (Condition vs Generated): {mean_diff_cond_gen:.2f}")
    
    # Diagnosis
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    if mse_cond_gen < 1.0:
        print("⚠️  CRITICAL: Generated output is nearly IDENTICAL to input!")
        print("   - The model is not transforming the input properly")
        print("   - This indicates the model hasn't learned yet or isn't working")
        if mean_diff_cond_gen < 1.0:
            print("   - Average pixel difference is less than 1 - extremely similar!")
    elif mse_cond_gen < 10.0:
        print("⚠️  WARNING: Generated output is very similar to input")
        print("   - The model is making small changes but may need more training")
    elif mse_cond_gen < 100.0:
        print("✓ Generated output differs moderately from input")
        print("  - Model is learning but may need more training")
    else:
        print("✓ Generated output significantly differs from input")
        print("  - Model appears to be working correctly")
    
    if mse_gen_gt < mse_cond_gt:
        improvement = ((mse_cond_gt - mse_gen_gt) / mse_cond_gt) * 100
        print(f"\n✓ Model improves over input by {improvement:.1f}%")
    else:
        degradation = ((mse_gen_gt - mse_cond_gt) / mse_cond_gt) * 100
        print(f"\n⚠️  Model output is worse than input by {degradation:.1f}%")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Top row: Original images
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(condition)
    ax1.set_title(f'Input Condition', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(generated)
    ax2.set_title(f'Generated Output', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(ground_truth)
    ax3.set_title(f'Ground Truth', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Bottom row: Difference maps
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(diff_cond_gen, cmap='hot', vmin=0, vmax=255)
    ax4.set_title(f'|Condition - Generated|\nMSE: {mse_cond_gen:.2f}, Mean: {mean_diff_cond_gen:.2f}', 
                  fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.imshow(diff_gen_gt, cmap='hot', vmin=0, vmax=255)
    ax5.set_title(f'|Generated - GT|\nMSE: {mse_gen_gt:.2f}', 
                  fontsize=12, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    ax6 = plt.subplot(2, 3, 6)
    im6 = ax6.imshow(diff_cond_gt, cmap='hot', vmin=0, vmax=255)
    ax6.set_title(f'|Condition - GT|\nMSE: {mse_cond_gt:.2f} (baseline)', 
                  fontsize=12, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    
    fig.suptitle(f'Sample Analysis: {sample_path.name}', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save analysis
    output_path = sample_path / 'detailed_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved detailed analysis to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual comparison tool for sample outputs")
    parser.add_argument("--sample_dir", type=str, required=True,
                       help="Path to sample directory (e.g., results/.../samples/1000/val_sample)")
    
    args = parser.parse_args()
    
    load_and_compare(args.sample_dir)
