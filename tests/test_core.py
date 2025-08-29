"""
Core Functionality Tests
========================

Author: Sergie Code - Software Engineer & YouTube Programming Educator
Project: AI Tools for Musicians
Date: August 29, 202def run_all_core_tests():
    """Run all core tests."""
    print("Music Classification Model - Core Tests")
    print("=" * 50)
    print("Author: Sergie Code - Software Engineer & YouTube Programming Educator")
    print("Project: AI Tools for Musicians")
    print("=" * 50)sic tests to ensure core functionality works perfectly.
"""

import sys
import torch
import tempfile
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import create_model

from models import create_model
from training import create_dummy_data_loaders, MusicClassificationTrainer
from utils import save_model_for_api


def test_model_creation():
    """Test basic model creation."""
    print("✅ Testing model creation...")
    
    # Test CNN model
    cnn_model = create_model("cnn", num_genres=5, num_moods=3, num_keys=12)
    assert cnn_model is not None
    
    # Test RNN model  
    rnn_model = create_model("rnn", num_genres=5, num_moods=3, num_keys=12)
    assert rnn_model is not None
    
    print("   ✅ CNN and RNN models created successfully")


def test_model_inference():
    """Test model inference works."""
    print("✅ Testing model inference...")
    
    model = create_model("cnn", num_genres=5, num_moods=3, num_keys=12)
    model.eval()
    
    # Create dummy inputs
    spectrogram = torch.randn(2, 1, 128, 200)
    features = torch.randn(2, 103)
    
    # Run inference
    with torch.no_grad():
        predictions = model(spectrogram, features)
    
    # Check outputs
    assert 'genre' in predictions
    assert 'mood' in predictions
    assert 'bpm' in predictions
    assert 'key' in predictions
    
    # Check shapes
    assert predictions['genre'].shape == (2, 5)
    assert predictions['mood'].shape == (2, 3)
    assert predictions['bpm'].shape == (2,)
    assert predictions['key'].shape == (2, 12)
    
    print("   ✅ Model inference working correctly")


def test_training_setup():
    """Test training setup works."""
    print("✅ Testing training setup...")
    
    model = create_model("cnn", num_genres=5, num_moods=3, num_keys=12)
    train_loader, val_loader = create_dummy_data_loaders(batch_size=4)
    device = torch.device("cpu")
    
    # Create trainer
    trainer = MusicClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    assert trainer.model is not None
    assert trainer.optimizer is not None
    
    print("   ✅ Training setup successful")


def test_short_training():
    """Test a very short training run."""
    print("✅ Testing short training run...")
    
    model = create_model("cnn", num_genres=5, num_moods=3, num_keys=12)
    train_loader, val_loader = create_dummy_data_loaders(batch_size=4)
    device = torch.device("cpu")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = MusicClassificationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            model_save_dir=temp_dir,
            log_dir=temp_dir
        )
        
        # Train for just 1 epoch
        history = trainer.train(num_epochs=1, save_every=1)
        
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 1
        
        # Check model was saved
        assert os.path.exists(os.path.join(temp_dir, 'best_model.pth'))
    
    print("   ✅ Short training run completed successfully")


def test_model_export():
    """Test model export for API."""
    print("✅ Testing model export...")
    
    model = create_model("cnn", num_genres=3, num_moods=2, num_keys=12)
    
    class_mappings = {
        'genres': ['rock', 'pop', 'jazz'],
        'moods': ['happy', 'sad'],
        'keys': ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    }
    
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        save_path = f.name
    
    try:
        save_model_for_api(
            model=model,
            class_mappings=class_mappings,
            save_path=save_path
        )
        
        # Check file exists and can be loaded
        assert os.path.exists(save_path)
        checkpoint = torch.load(save_path, map_location='cpu')
        assert 'model_state_dict' in checkpoint
        assert 'class_mappings' in checkpoint
        
        print("   ✅ Model export successful")
        
    finally:
        if os.path.exists(save_path):
            os.unlink(save_path)


def test_variable_length_handling():
    """Test handling of variable-length inputs."""
    print("✅ Testing variable-length input handling...")
    
    model = create_model("cnn", num_genres=3, num_moods=2, num_keys=12)
    model.eval()
    
    # Test different spectrogram lengths
    lengths = [100, 200, 300, 500]
    
    for length in lengths:
        spectrogram = torch.randn(1, 1, 128, length)
        features = torch.randn(1, 103)
        
        with torch.no_grad():
            predictions = model(spectrogram, features)
        
        # Should always work regardless of length
        assert predictions['genre'].shape == (1, 3)
        assert predictions['mood'].shape == (1, 2)
    
    print("   ✅ Variable-length inputs handled correctly")


def run_all_core_tests():
    """Run all core functionality tests."""
    print("🎵 Music Classification Model - Core Tests")
    print("=" * 50)
    print("Author: Sergie Code - Software Engineer & YouTube Programming Educator")
    print("Project: AI Tools for Musicians")
    print("=" * 50)
    
    try:
        test_model_creation()
        test_model_inference()
        test_training_setup()
        test_short_training()
        test_model_export()
        test_variable_length_handling()
        
        print("\nAll core tests passed!")
        print("Your music classification model is working perfectly!")
        print("Ready for your YouTube content!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_core_tests()
    exit(0 if success else 1)
