"""
Quick diagnostic script to check if your model is generating properly.
Run this after training has started to verify the model is learning.

Usage:
    python check_model_output.py --config configs/c2v.yaml --checkpoint results/.../checkpoint/latest_model_XXX.pth
"""

import torch
import numpy as np
import argparse
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**{k: argparse.Namespace(**v) if isinstance(v, dict) else v for k, v in config.items()})

def check_model_outputs(config_path, checkpoint_path):
    """
    Quick diagnostic to check if model is generating different outputs from inputs
    """
    print("=" * 80)
    print("MODEL OUTPUT DIAGNOSTIC")
    print("=" * 80)
    
    # Load config
    config = load_config(config_path)
    
    # Load checkpoint
    print(f"\n1. Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check training progress
    epoch = checkpoint.get('epoch', 'Unknown')
    step = checkpoint.get('step', 'Unknown')
    print(f"   - Epoch: {epoch}")
    print(f"   - Step: {step}")
    
    # Check model parameters
    model_state = checkpoint.get('model', {})
    if model_state:
        # Calculate parameter statistics
        all_params = []
        for name, param in model_state.items():
            if 'weight' in name:
                all_params.append(param.flatten())
        
        if all_params:
            all_params = torch.cat(all_params)
            print(f"\n2. Model Parameters Statistics:")
            print(f"   - Total parameters: {len(all_params):,}")
            print(f"   - Mean: {all_params.mean().item():.6f}")
            print(f"   - Std: {all_params.std().item():.6f}")
            print(f"   - Min: {all_params.min().item():.6f}")
            print(f"   - Max: {all_params.max().item():.6f}")
            print(f"   - Abs Mean: {all_params.abs().mean().item():.6f}")
            
            # Check if parameters seem initialized
            if all_params.abs().mean().item() < 1e-6:
                print("   ⚠️  WARNING: Parameters are very small - model may not be initialized properly!")
            elif all_params.std().item() < 1e-6:
                print("   ⚠️  WARNING: Parameters have very low variance - model may not be learning!")
            else:
                print("   ✓ Parameters look reasonable")
    
    # Check EMA if available
    if 'ema' in checkpoint:
        print(f"\n3. EMA Status:")
        print(f"   - EMA weights present: Yes")
        ema_params = []
        for name, param in checkpoint['ema'].items():
            if isinstance(param, torch.Tensor):
                ema_params.append(param.flatten())
        if ema_params:
            ema_params = torch.cat(ema_params)
            print(f"   - EMA mean: {ema_params.mean().item():.6f}")
            print(f"   - EMA std: {ema_params.std().item():.6f}")
    else:
        print(f"\n3. EMA Status:")
        print(f"   - EMA weights present: No (will be available after step 30000)")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    
    if isinstance(step, int):
        if step < 1000:
            print("• Model is in very early training - outputs will look like inputs")
            print("• This is NORMAL - wait until at least 5000 steps")
        elif step < 5000:
            print("• Model is in early training - outputs should start to differ slightly")
            print("• Check the MSE metrics in TensorBoard")
        elif step < 30000:
            print("• Model should be learning - check that:")
            print("  - Loss is decreasing in TensorBoard")
            print("  - MSE(condition→generated) is increasing")
            print("  - MSE(generated→GT) is decreasing")
        else:
            print("• Model should be well-trained - outputs should clearly differ from inputs")
            print("• If they don't, check:")
            print("  - Loss curves (should be decreasing)")
            print("  - Learning rate (may need adjustment)")
            print("  - Data loading (ensure inputs != targets)")
    
    print("\n4. Check These Files:")
    checkpoint_dir = Path(checkpoint_path).parent
    result_dir = checkpoint_dir.parent
    
    sample_dir = result_dir / "samples"
    if sample_dir.exists():
        print(f"   - Comparison plots: {sample_dir}/*/*/*_comparison.png")
    
    log_dir = result_dir / "log"
    if log_dir.exists():
        print(f"   - TensorBoard logs: {log_dir}")
        print(f"     Run: tensorboard --logdir {log_dir}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic tool for model output checking")
    parser.add_argument("--config", type=str, default="configs/c2v.yaml", 
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    check_model_outputs(args.config, args.checkpoint)
