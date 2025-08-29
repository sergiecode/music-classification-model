"""
Basic Tests for Music Classification Model
==========================================

Author: Sergie Code - Software Engineer & YouTube Programming Educator
Project: AI Tools for Musicians
Date: August 29, 2025

Basic tests to ensure model components work correctly.
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import create_model, MusicClassificationModel, SpectrogramCNN, FeatureMLP


def test_model_creation():
    """Test that models can be created successfully."""
    print("Testing model creation...")
    
    # Test CNN model
    cnn_model = create_model("cnn", num_genres=5, num_moods=3, num_keys=12)
    assert isinstance(cnn_model, MusicClassificationModel)
    print("✅ CNN model created successfully")
    
    # Test RNN model
    rnn_model = create_model("rnn", num_genres=5, num_moods=3, num_keys=12)
    assert rnn_model is not None
    print("✅ RNN model created successfully")


def test_model_forward_pass():
    """Test that models can perform forward passes."""
    print("Testing model forward pass...")
    
    device = torch.device("cpu")
    model = create_model("cnn", num_genres=5, num_moods=3, num_keys=12)
    model.to(device)
    
    # Create dummy inputs
    batch_size = 2
    spectrogram = torch.randn(batch_size, 1, 128, 200).to(device)
    features = torch.randn(batch_size, 103).to(device)
    
    # Forward pass
    with torch.no_grad():
        predictions = model(spectrogram, features)
    
    # Check output shapes
    assert predictions['genre'].shape == (batch_size, 5)
    assert predictions['mood'].shape == (batch_size, 3)
    assert predictions['bpm'].shape == (batch_size,)
    assert predictions['key'].shape == (batch_size, 12)
    
    print("✅ Model forward pass successful")


def test_individual_components():
    """Test individual model components."""
    print("Testing individual components...")
    
    # Test SpectrogramCNN
    cnn = SpectrogramCNN(input_channels=1, n_mels=128)
    dummy_spec = torch.randn(2, 1, 128, 200)
    cnn_output = cnn(dummy_spec)
    assert cnn_output.shape == (2, 256)
    print("✅ SpectrogramCNN working")
    
    # Test FeatureMLP
    mlp = FeatureMLP(input_size=103, hidden_size=256)
    dummy_features = torch.randn(2, 103)
    mlp_output = mlp(dummy_features)
    assert mlp_output.shape == (2, 64)
    print("✅ FeatureMLP working")


def test_variable_length_spectrograms():
    """Test that model handles variable-length spectrograms."""
    print("Testing variable-length spectrograms...")
    
    model = create_model("cnn", num_genres=3, num_moods=2, num_keys=12)
    
    # Test different lengths
    lengths = [100, 200, 300]
    for length in lengths:
        spectrogram = torch.randn(1, 1, 128, length)
        features = torch.randn(1, 103)
        
        with torch.no_grad():
            predictions = model(spectrogram, features)
        
        assert predictions['genre'].shape == (1, 3)
    
    print("✅ Variable-length spectrograms handled correctly")


def test_model_parameters():
    """Test model parameter counting."""
    print("Testing model parameters...")
    
    model = create_model("cnn")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert total_params > 0
    assert trainable_params == total_params  # All parameters should be trainable
    
    print(f"✅ Model has {total_params:,} parameters (all trainable)")


def run_all_tests():
    """Run all tests."""
    print("🧪 Running Music Classification Model Tests")
    print("=" * 50)
    
    try:
        test_model_creation()
        test_model_forward_pass()
        test_individual_components()
        test_variable_length_spectrograms()
        test_model_parameters()
        
        print("\n🎉 All tests passed!")
        print("✅ Models are working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise e


if __name__ == "__main__":
    run_all_tests()
