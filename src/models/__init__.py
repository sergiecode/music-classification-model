"""
Base Music Classification Model
==============================

Author: Sergie Code - Software Engineer & YouTube Programming Educator
Project: AI Tools for Musicians
Date: August 29, 2025

This module defines the base architecture for music classification models
that work with preprocessed audio features and spectrograms.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class MusicClassificationModel(nn.Module):
    """
    Base class for music classification models.
    
    This model combines spectrogram-based CNN features with traditional
    audio features for multi-task classification including genre, mood,
    BPM, and key prediction.
    """
    
    def __init__(
        self,
        num_genres: int = 10,
        num_moods: int = 4,
        num_keys: int = 12,
        feature_size: int = 103,
        spectrogram_shape: Tuple[int, int] = (128, None),
        dropout_rate: float = 0.3
    ):
        """
        Initialize the Music Classification Model.
        
        Args:
            num_genres: Number of genre classes
            num_moods: Number of mood classes  
            num_keys: Number of musical keys (typically 12)
            feature_size: Size of feature vector from preprocessing (103 features)
            spectrogram_shape: Shape of mel spectrogram (n_mels, time_frames)
            dropout_rate: Dropout rate for regularization
        """
        super(MusicClassificationModel, self).__init__()
        
        self.num_genres = num_genres
        self.num_moods = num_moods
        self.num_keys = num_keys
        self.feature_size = feature_size
        self.spectrogram_shape = spectrogram_shape
        
        # Spectrogram CNN branch
        self.spectrogram_cnn = SpectrogramCNN(
            input_channels=1,
            n_mels=spectrogram_shape[0]
        )
        
        # Feature MLP branch
        self.feature_mlp = FeatureMLP(
            input_size=feature_size,
            hidden_size=256,
            dropout_rate=dropout_rate
        )
        
        # Get feature dimensions
        cnn_features = self.spectrogram_cnn.get_output_size()
        mlp_features = self.feature_mlp.get_output_size()
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(cnn_features + mlp_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Task-specific heads
        self.genre_head = nn.Linear(256, num_genres)
        self.mood_head = nn.Linear(256, num_moods)
        self.bpm_head = nn.Linear(256, 1)  # Regression for BPM
        self.key_head = nn.Linear(256, num_keys)
        
    def forward(
        self, 
        spectrogram: torch.Tensor, 
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            spectrogram: Mel spectrogram tensor (batch_size, 1, n_mels, time_frames)
            features: Feature vector tensor (batch_size, feature_size)
            
        Returns:
            Dictionary containing predictions for each task
        """
        # Process spectrogram through CNN
        spec_features = self.spectrogram_cnn(spectrogram)
        
        # Process features through MLP
        feat_features = self.feature_mlp(features)
        
        # Fuse features
        combined = torch.cat([spec_features, feat_features], dim=1)
        fused = self.fusion(combined)
        
        # Generate predictions for each task
        predictions = {
            'genre': self.genre_head(fused),
            'mood': self.mood_head(fused),
            'bpm': self.bpm_head(fused).squeeze(-1),  # Remove last dimension for regression
            'key': self.key_head(fused)
        }
        
        return predictions


class SpectrogramCNN(nn.Module):
    """
    CNN for processing mel spectrograms.
    
    This network extracts temporal and spectral features from mel spectrograms
    using convolutional layers with adaptive pooling to handle variable lengths.
    """
    
    def __init__(self, input_channels: int = 1, n_mels: int = 128):
        super(SpectrogramCNN, self).__init__()
        
        self.input_channels = input_channels
        self.n_mels = n_mels
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.25),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.25),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.25)
        )
        
        # Adaptive pooling to handle variable length inputs
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Final feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.
        
        Args:
            x: Input spectrogram (batch_size, channels, n_mels, time_frames)
            
        Returns:
            Feature vector (batch_size, 256)
        """
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)
        
        # Flatten and extract features
        x = x.view(x.size(0), -1)
        x = self.feature_extractor(x)
        
        return x
    
    def get_output_size(self) -> int:
        """Return the output feature size."""
        return 256


class FeatureMLP(nn.Module):
    """
    MLP for processing traditional audio features.
    
    This network processes the 103 features extracted by the preprocessing
    pipeline including temporal, spectral, harmonic, rhythmic, and statistical features.
    """
    
    def __init__(
        self, 
        input_size: int = 103, 
        hidden_size: int = 256, 
        dropout_rate: float = 0.3
    ):
        super(FeatureMLP, self).__init__()
        
        self.input_size = input_size
        
        # Feature processing layers
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.
        
        Args:
            x: Feature vector (batch_size, input_size)
            
        Returns:
            Processed features (batch_size, 64)
        """
        return self.layers(x)
    
    def get_output_size(self) -> int:
        """Return the output feature size."""
        return 64


class RecurrentMusicModel(nn.Module):
    """
    Alternative RNN-based model for sequence modeling of audio features.
    
    This model uses LSTM layers to capture temporal dependencies in
    audio feature sequences over time.
    """
    
    def __init__(
        self,
        num_genres: int = 10,
        num_moods: int = 4,
        num_keys: int = 12,
        feature_size: int = 103,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.3
    ):
        super(RecurrentMusicModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Feature processing
        lstm_output_size = hidden_size * 2  # Bidirectional
        self.feature_processor = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Task-specific heads
        self.genre_head = nn.Linear(128, num_genres)
        self.mood_head = nn.Linear(128, num_moods)
        self.bpm_head = nn.Linear(128, 1)
        self.key_head = nn.Linear(128, num_keys)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through RNN model.
        
        Args:
            x: Sequence of feature vectors (batch_size, seq_length, feature_size)
            
        Returns:
            Dictionary containing predictions for each task
        """
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output for classification
        last_output = lstm_out[:, -1, :]
        
        # Process features
        features = self.feature_processor(last_output)
        
        # Generate predictions
        predictions = {
            'genre': self.genre_head(features),
            'mood': self.mood_head(features),
            'bpm': self.bpm_head(features).squeeze(-1),
            'key': self.key_head(features)
        }
        
        return predictions


def create_model(model_type: str = "cnn", **kwargs) -> nn.Module:
    """
    Factory function to create different model architectures.
    
    Args:
        model_type: Type of model to create ("cnn" or "rnn")
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized model
    """
    if model_type == "cnn":
        return MusicClassificationModel(**kwargs)
    elif model_type == "rnn":
        return RecurrentMusicModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Example usage and testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_model("cnn")
    model.to(device)
    
    # Test with dummy data
    batch_size = 4
    n_mels = 128
    time_frames = 300
    feature_size = 103
    
    # Dummy inputs
    spectrogram = torch.randn(batch_size, 1, n_mels, time_frames).to(device)
    features = torch.randn(batch_size, feature_size).to(device)
    
    # Forward pass
    with torch.no_grad():
        predictions = model(spectrogram, features)
    
    print("Model output shapes:")
    for task, output in predictions.items():
        print(f"{task}: {output.shape}")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
