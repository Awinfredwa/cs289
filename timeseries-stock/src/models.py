"""Neural network models for time-series prediction."""

import torch
import torch.nn as nn
from typing import Optional


class LSTMModel(nn.Module):
    """
    LSTM-based model for time-series prediction.
    
    Args:
        input_size: Number of features per time step
        hidden_size: Size of LSTM hidden state
        num_layers: Number of stacked LSTM layers
        output_size: Size of output (1 for binary classification or regression)
        dropout: Dropout rate (applied between LSTM layers if num_layers > 1)
        task: 'classification' or 'regression'
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int = 1,
        dropout: float = 0.1,
        task: str = 'classification'
    ):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.task = task
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # No activation - CrossEntropyLoss expects raw logits
        self.activation = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, window_size, input_size)
        
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward
        # lstm_out: (batch_size, window_size, hidden_size)
        # h_n, c_n: (num_layers, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the last time step's output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Fully connected layer
        out = self.fc(last_output)  # (batch_size, output_size)
        
        # For binary classification with CrossEntropyLoss, return (batch, 2)
        # For regression, squeeze to (batch,)
        if self.task == 'classification' and out.shape[-1] > 1:
            return out  # (batch_size, num_classes)
        else:
            return out.squeeze(-1)  # (batch_size,)


class CNN1DModel(nn.Module):
    """
    1D CNN-based model for time-series prediction.
    
    Args:
        input_size: Number of features per time step
        hidden_size: Number of filters in conv layers
        output_size: Size of output (1 for binary classification or regression)
        dropout: Dropout rate
        task: 'classification' or 'regression'
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        dropout: float = 0.1,
        task: str = 'classification'
    ):
        super(CNN1DModel, self).__init__()
        
        self.task = task
        
        # Conv layers (over time dimension)
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size * 2,
            kernel_size=3,
            padding=1
        )
        
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
        # No activation - CrossEntropyLoss expects raw logits
        self.activation = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, window_size, input_size)
        
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Transpose for Conv1d: (batch_size, input_size, window_size)
        x = x.transpose(1, 2)
        
        # First conv block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Global pooling: (batch_size, channels, 1)
        x = self.pool(x)
        
        # Flatten: (batch_size, channels)
        x = x.squeeze(-1)
        
        # Fully connected
        out = self.fc(x)
        
        # For binary classification with CrossEntropyLoss, return (batch, 2)
        # For regression, squeeze to (batch,)
        if self.task == 'classification' and out.shape[-1] > 1:
            return out  # (batch_size, num_classes)
        else:
            return out.squeeze(-1)  # (batch_size,)


def create_model(
    model_type: str,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    task: str
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: 'lstm' or 'cnn1d'
        input_size: Number of input features
        hidden_size: Hidden layer size
        num_layers: Number of layers (LSTM only)
        dropout: Dropout rate
        task: 'classification' or 'regression'
    
    Returns:
        Model instance
    """
    # For classification, use 2 output neurons (for CrossEntropyLoss)
    # For regression, use 1 output neuron
    output_size = 2 if task == 'classification' else 1
    
    if model_type == 'lstm':
        return LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
            task=task
        )
    elif model_type == 'cnn1d':
        return CNN1DModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout,
            task=task
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

