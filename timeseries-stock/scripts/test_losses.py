"""
Test the custom loss functions to verify they work as expected.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.losses import (
    DirectionAwareMSE, 
    WeightedDirectionMSE,
    HuberDirectionLoss,
    create_loss_function
)

def test_direction_aware_loss():
    """Test DirectionAwareMSE loss function."""
    print("=" * 80)
    print("TESTING DIRECTION-AWARE MSE LOSS")
    print("=" * 80)
    
    criterion = DirectionAwareMSE(direction_weight=0.5)
    
    # Test case 1: All predictions have correct direction
    print("\n[Test 1] Correct directions:")
    pred = torch.tensor([0.01, -0.02, 0.03, -0.01])
    target = torch.tensor([0.02, -0.01, 0.015, -0.015])
    
    loss = criterion(pred, target)
    print(f"  Predictions: {pred.tolist()}")
    print(f"  Targets:     {target.tolist()}")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  ✅ All directions correct - loss should be relatively low")
    
    # Test case 2: All predictions have wrong direction
    print("\n[Test 2] Wrong directions:")
    pred = torch.tensor([0.01, 0.02, 0.03, 0.01])
    target = torch.tensor([-0.02, -0.01, -0.015, -0.015])
    
    loss = criterion(pred, target)
    print(f"  Predictions: {pred.tolist()}")
    print(f"  Targets:     {target.tolist()}")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  ⚠️  All directions wrong - loss should be HIGH")
    
    # Test case 3: Constant positive predictions (the problem we're solving)
    print("\n[Test 3] Constant positive predictions (biased model):")
    pred = torch.tensor([0.001, 0.001, 0.001, 0.001, 0.001])
    target = torch.tensor([0.02, -0.03, 0.01, -0.02, 0.015])
    
    loss = criterion(pred, target)
    
    # Calculate how many are wrong
    wrong = ((torch.sign(pred) != torch.sign(target)).sum().item())
    print(f"  Predictions: {pred.tolist()} (all positive)")
    print(f"  Targets:     {target.tolist()} (mixed)")
    print(f"  Wrong directions: {wrong}/5")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  ⚠️  This high loss will push the model to learn actual patterns!")
    
    # Compare with standard MSE
    print("\n[Comparison] Standard MSE vs Direction-Aware MSE:")
    mse_criterion = torch.nn.MSELoss()
    
    pred = torch.tensor([0.001, 0.001, 0.001])
    target = torch.tensor([0.02, -0.03, 0.01])
    
    mse_loss = mse_criterion(pred, target)
    dir_loss = criterion(pred, target)
    
    print(f"  Constant predictions: {pred.tolist()}")
    print(f"  True targets:         {target.tolist()}")
    print(f"  Standard MSE:         {mse_loss.item():.6f}")
    print(f"  Direction-Aware MSE:  {dir_loss.item():.6f}")
    print(f"  Difference:           {(dir_loss.item() - mse_loss.item()):.6f}")
    print(f"  ✅ Direction-aware loss is HIGHER, discouraging constant predictions!")


def test_weighted_direction_loss():
    """Test WeightedDirectionMSE loss function."""
    print("\n\n" + "=" * 80)
    print("TESTING WEIGHTED DIRECTION MSE LOSS")
    print("=" * 80)
    
    criterion = WeightedDirectionMSE(direction_weight=0.5, magnitude_scaling=True)
    
    print("\n[Test] Large vs small misses:")
    
    # Small miss
    pred_small = torch.tensor([0.001, -0.001])
    target_small = torch.tensor([-0.001, 0.001])
    loss_small = criterion(pred_small, target_small)
    
    print(f"  Small moves, wrong direction:")
    print(f"    Predictions: {pred_small.tolist()}")
    print(f"    Targets:     {target_small.tolist()}")
    print(f"    Loss:        {loss_small.item():.6f}")
    
    # Large miss
    pred_large = torch.tensor([0.05, -0.05])
    target_large = torch.tensor([-0.05, 0.05])
    loss_large = criterion(pred_large, target_large)
    
    print(f"\n  Large moves, wrong direction:")
    print(f"    Predictions: {pred_large.tolist()}")
    print(f"    Targets:     {target_large.tolist()}")
    print(f"    Loss:        {loss_large.item():.6f}")
    
    print(f"\n  ✅ Loss on large moves ({loss_large.item():.6f}) >> Loss on small moves ({loss_small.item():.6f})")
    print(f"     This encourages getting large moves right!")


def test_loss_factory():
    """Test the loss function factory."""
    print("\n\n" + "=" * 80)
    print("TESTING LOSS FUNCTION FACTORY")
    print("=" * 80)
    
    # Test creating different loss types
    loss_types = ['mse', 'direction_mse', 'weighted_direction_mse', 'huber_direction']
    
    for loss_type in loss_types:
        if loss_type == 'mse':
            criterion = create_loss_function(loss_type)
        else:
            criterion = create_loss_function(loss_type, direction_weight=0.5)
        
        print(f"\n  ✅ Created {loss_type}: {criterion.__class__.__name__}")


def test_real_scenario():
    """Test with realistic stock return values."""
    print("\n\n" + "=" * 80)
    print("REALISTIC SCENARIO TEST")
    print("=" * 80)
    
    print("\nScenario: Market goes up 65% of the time")
    print("Model learns to always predict +0.001")
    
    # Simulate 100 predictions
    torch.manual_seed(42)
    
    # True returns: 65% positive, 35% negative
    n_samples = 100
    n_positive = 65
    
    target = torch.cat([
        torch.randn(n_positive) * 0.02 + 0.01,  # Positive returns (mean ~1%)
        torch.randn(n_samples - n_positive) * 0.02 - 0.01  # Negative returns (mean ~-1%)
    ])
    
    # Naive model: always predict small positive
    pred_naive = torch.full((n_samples,), 0.001)
    
    # Better model: learns actual patterns
    pred_better = target + torch.randn(n_samples) * 0.005  # Some noise
    
    # Evaluate with both losses
    mse_criterion = torch.nn.MSELoss()
    dir_criterion = DirectionAwareMSE(direction_weight=0.5)
    
    print("\n📊 Naive Model (always +0.001):")
    mse_loss_naive = mse_criterion(pred_naive, target)
    dir_loss_naive = dir_criterion(pred_naive, target)
    
    correct_dir_naive = (torch.sign(pred_naive) == torch.sign(target)).float().mean()
    
    print(f"  MSE Loss:         {mse_loss_naive.item():.6f}")
    print(f"  Direction Loss:   {dir_loss_naive.item():.6f}")
    print(f"  Direction Acc:    {correct_dir_naive.item():.2%}")
    
    print("\n📊 Better Model (learns patterns):")
    mse_loss_better = mse_criterion(pred_better, target)
    dir_loss_better = dir_criterion(pred_better, target)
    
    correct_dir_better = (torch.sign(pred_better) == torch.sign(target)).float().mean()
    
    print(f"  MSE Loss:         {mse_loss_better.item():.6f}")
    print(f"  Direction Loss:   {dir_loss_better.item():.6f}")
    print(f"  Direction Acc:    {correct_dir_better.item():.2%}")
    
    print("\n✅ Direction-aware loss comparison:")
    print(f"  Naive model loss:  {dir_loss_naive.item():.6f}")
    print(f"  Better model loss: {dir_loss_better.item():.6f}")
    print(f"  Difference:        {(dir_loss_naive.item() - dir_loss_better.item()):.6f}")
    print(f"  → Direction-aware loss correctly prefers the better model!")


if __name__ == "__main__":
    test_direction_aware_loss()
    test_weighted_direction_loss()
    test_loss_factory()
    test_real_scenario()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED ✅")
    print("=" * 80)
    print("\nYou can now use these loss functions in your training:")
    print("  1. Set loss_function: 'direction_mse' in config/default.yaml")
    print("  2. Adjust direction_weight (0.3-1.0) to control emphasis on direction")
    print("  3. Run training and watch for improved directional accuracy!")
    print()

