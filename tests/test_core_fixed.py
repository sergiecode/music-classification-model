"""
Core Functionality Tests
========================

Author: Sergie Code - Software Engineer & YouTube Programming Educator
Project: AI Tools for Musicians
Date: August 29, 2025

Basic tests to ensure core functionality works perfectly.
"""

import sys
import torch
import tempfile
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import create_model
from training import create_dummy_data_loaders, MusicClassificationTrainer
from utils import save_model_for_api
from data import MusicDataset


def test_model_creation():
    """Test creating different model architectures."""
    print("Testing model creation...")
    
    # Test CNN model
    cnn_model = create_model("cnn", num_genres=5, num_moods=3, num_keys=12)
    assert cnn_model is not None
    
    # Test RNN model
    rnn_model = create_model("rnn", num_genres=5, num_moods=3, num_keys=12)
    assert rnn_model is not None
    
    print("   CNN and RNN models created successfully")


def test_model_inference():
    """Test model inference with dummy data."""
    print("Testing model inference...")
    
    model = create_model("cnn", num_genres=5, num_moods=3, num_keys=12)
    
    # Create dummy input
    batch_size = 2
    spectrogram = torch.randn(batch_size, 1, 128, 300)
    features = torch.randn(batch_size, 103)
    
    # Test inference
    model.eval()
    with torch.no_grad():
        outputs = model(spectrogram, features)
    
    # Check outputs
    assert 'genre' in outputs
    assert 'mood' in outputs
    assert 'bpm' in outputs
    assert 'key' in outputs
    
    assert outputs['genre'].shape == (batch_size, 5)
    assert outputs['mood'].shape == (batch_size, 3)
    assert outputs['bpm'].shape == (batch_size,)  # BPM is scalar output
    assert outputs['key'].shape == (batch_size, 12)
    
    print("   Model inference working correctly")


def test_training_setup():
    """Test training setup with dummy data."""
    print("Testing training setup...")
    
    model = create_model("cnn", num_genres=5, num_moods=3, num_keys=12)
    train_loader, val_loader = create_dummy_data_loaders(batch_size=4)
    device = torch.device("cpu")
    
    trainer = MusicClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    assert trainer.model is not None
    assert trainer.optimizer is not None
    
    print("   Training setup successful")


def test_short_training():
    """Test a very short training run."""
    print("Testing short training run...")
    
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
    
    print("   Short training run completed successfully")


def test_model_export():
    """Test exporting model for API use."""
    print("Testing model export...")
    
    model = create_model("cnn", num_genres=5, num_moods=3, num_keys=12)
    
    # Create class mappings
    class_mappings = {
        'genre': {0: 'rock', 1: 'pop', 2: 'jazz', 3: 'classical', 4: 'electronic'},
        'mood': {0: 'happy', 1: 'sad', 2: 'energetic'},
        'key': {i: f'key_{i}' for i in range(12)}
    }
    
    save_path = 'test_model_export.pth'
    
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
        
        print("   Model export successful")
        
    finally:
        if os.path.exists(save_path):
            os.unlink(save_path)


def test_variable_length_handling():
    """Test handling of variable-length inputs."""
    print("Testing variable-length input handling...")
    
    model = create_model("cnn", num_genres=5, num_moods=3, num_keys=12)
    
    # Create inputs with different lengths
    batch_data = [
        {
            'spectrogram': torch.randn(1, 128, 200),  # Shorter
            'features': torch.randn(103),
            'genre': torch.tensor(0),
            'mood': torch.tensor(1),
            'bpm': torch.tensor(120.0),
            'key': torch.tensor(3),
            'filename': 'test1.wav'
        },
        {
            'spectrogram': torch.randn(1, 128, 500),  # Longer
            'features': torch.randn(103),
            'genre': torch.tensor(2),
            'mood': torch.tensor(0),
            'bpm': torch.tensor(140.0),
            'key': torch.tensor(7),
            'filename': 'test2.wav'
        }
    ]
    
    # Use collate function from data module
    from data import collate_fn
    
    # Collate the batch data
    collated = collate_fn(batch_data)
    
    # Test with collated data
    model.eval()
    with torch.no_grad():
        outputs = model(collated['spectrogram'], collated['features'])
    
    # Should work without errors
    assert 'genre' in outputs
    assert outputs['genre'].shape[0] == 2  # Batch size
    
    print("   Variable-length inputs handled correctly")


def run_all_core_tests():
    """Run all core tests."""
    print("Music Classification Model - Core Tests")
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
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_core_tests()
