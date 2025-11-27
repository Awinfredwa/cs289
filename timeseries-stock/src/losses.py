"""
Custom loss functions for stock prediction.

This module provides specialized loss functions that address common issues
in financial time series prediction, such as constant predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionAwareMSE(nn.Module):
    """
    Hybrid loss that combines MSE with directional penalty.
    
    This loss function forces the model to learn both magnitude AND direction
    of returns, preventing it from just predicting a constant positive value.
    
    Loss = MSE(pred, target) + direction_weight * DirectionPenalty(pred, target)
    
    Where DirectionPenalty penalizes predictions that have the wrong sign
    (predicting up when actual is down, or vice versa).
    
    Args:
        direction_weight: Weight for the directional penalty term (default: 0.5)
        
    Example:
        >>> criterion = DirectionAwareMSE(direction_weight=0.5)
        >>> pred = torch.tensor([0.01, -0.02, 0.03])
        >>> target = torch.tensor([0.02, 0.01, 0.015])
        >>> loss = criterion(pred, target)
    """
    def __init__(self, direction_weight: float = 0.5):
        super().__init__()
        self.direction_weight = direction_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute direction-aware MSE loss.
        
        Args:
            pred: Predicted returns (batch_size,) or (batch_size, 1)
            target: Actual returns (batch_size,) or (batch_size, 1)
            
        Returns:
            Combined loss value
        """
        # Ensure same shape
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Standard MSE loss for magnitude
        mse_loss = F.mse_loss(pred, target)
        
        # Directional penalty: penalize wrong direction predictions
        pred_sign = torch.sign(pred)
        target_sign = torch.sign(target)
        
        # Wrong direction when signs don't match (and neither is zero)
        wrong_direction = (pred_sign != target_sign).float()
        
        # Average penalty for wrong directions
        direction_loss = wrong_direction.mean()
        
        # Combined loss
        total_loss = mse_loss + self.direction_weight * direction_loss
        
        return total_loss


class WeightedDirectionMSE(nn.Module):
    """
    MSE loss with weighted directional penalty based on magnitude.
    
    This variant applies stronger penalties for getting the direction wrong
    on large moves, while being more lenient on small moves near zero.
    
    Args:
        direction_weight: Base weight for directional penalty (default: 0.5)
        magnitude_scaling: Whether to scale penalty by magnitude (default: True)
    """
    def __init__(self, direction_weight: float = 0.5, magnitude_scaling: bool = True):
        super().__init__()
        self.direction_weight = direction_weight
        self.magnitude_scaling = magnitude_scaling
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute magnitude-weighted direction-aware MSE loss."""
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Standard MSE loss
        mse_loss = F.mse_loss(pred, target)
        
        # Directional component
        pred_sign = torch.sign(pred)
        target_sign = torch.sign(target)
        wrong_direction = (pred_sign != target_sign).float()
        
        if self.magnitude_scaling:
            # Weight penalty by target magnitude (bigger moves = bigger penalty)
            magnitude_weight = torch.abs(target)
            direction_loss = (wrong_direction * magnitude_weight).mean()
        else:
            direction_loss = wrong_direction.mean()
        
        total_loss = mse_loss + self.direction_weight * direction_loss
        
        return total_loss


class AsymmetricMSE(nn.Module):
    """
    Asymmetric MSE that penalizes certain errors more heavily.
    
    For example, you can penalize underestimating positive returns more heavily
    than overestimating them (to avoid missing profitable opportunities).
    
    Args:
        overpredict_weight: Weight for errors where pred > target (default: 1.0)
        underpredict_weight: Weight for errors where pred < target (default: 1.0)
    """
    def __init__(self, overpredict_weight: float = 1.0, underpredict_weight: float = 1.0):
        super().__init__()
        self.overpredict_weight = overpredict_weight
        self.underpredict_weight = underpredict_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute asymmetric MSE loss."""
        pred = pred.view(-1)
        target = target.view(-1)
        
        squared_errors = (pred - target) ** 2
        
        # Apply different weights based on over/under prediction
        weights = torch.where(
            pred > target,
            torch.tensor(self.overpredict_weight, device=pred.device),
            torch.tensor(self.underpredict_weight, device=pred.device)
        )
        
        weighted_loss = (weights * squared_errors).mean()
        
        return weighted_loss


class HuberDirectionLoss(nn.Module):
    """
    Huber loss (robust to outliers) combined with directional penalty.
    
    Huber loss is less sensitive to outliers than MSE, which can be helpful
    for financial data with extreme events (crashes, rallies).
    
    Args:
        delta: Threshold for Huber loss (default: 0.1)
        direction_weight: Weight for directional penalty (default: 0.5)
    """
    def __init__(self, delta: float = 0.1, direction_weight: float = 0.5):
        super().__init__()
        self.delta = delta
        self.direction_weight = direction_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss with directional penalty."""
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Huber loss
        huber_loss = F.huber_loss(pred, target, delta=self.delta)
        
        # Directional penalty
        pred_sign = torch.sign(pred)
        target_sign = torch.sign(target)
        wrong_direction = (pred_sign != target_sign).float()
        direction_loss = wrong_direction.mean()
        
        total_loss = huber_loss + self.direction_weight * direction_loss
        
        return total_loss


def create_loss_function(loss_type: str = 'mse', **kwargs):
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss function
            - 'mse': Standard MSE
            - 'direction_mse': Direction-aware MSE
            - 'weighted_direction_mse': Magnitude-weighted direction MSE
            - 'asymmetric_mse': Asymmetric MSE
            - 'huber_direction': Huber loss with direction
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function module
        
    Example:
        >>> criterion = create_loss_function('direction_mse', direction_weight=0.5)
    """
    loss_functions = {
        'mse': nn.MSELoss,
        'direction_mse': DirectionAwareMSE,
        'weighted_direction_mse': WeightedDirectionMSE,
        'asymmetric_mse': AsymmetricMSE,
        'huber_direction': HuberDirectionLoss,
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from {list(loss_functions.keys())}")
    
    return loss_functions[loss_type](**kwargs)

