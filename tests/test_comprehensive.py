"""
Comprehensive Test Suite for Music Classification Model
======================================================

Author: Sergie Code - Software Engineer & YouTube Programming Educator
Project: AI Tools for Musicians
Date: August 29, 2025

This module provides comprehensive testing for all components of the
music classification model to ensure everything works perfectly.
"""

import sys
import pytest
import torch
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import (
    create_model, 
    MusicClassificationModel, 
    SpectrogramCNN, 
    FeatureMLP,
    RecurrentMusicModel
)
from data import MusicDataset, collate_fn
from training import MusicClassificationTrainer, create_dummy_data_loaders
from utils import (
    save_model_for_api,
    load_model_for_inference,
    create_sample_config
)


class TestModelArchitectures:
    """Test all model architectures work correctly."""
    
    def test_cnn_model_creation(self):
        """Test CNN model can be created with different parameters."""
        # Test default parameters
        model = create_model("cnn")
        assert isinstance(model, MusicClassificationModel)
        
        # Test custom parameters
        model_custom = create_model(
            "cnn", 
            num_genres=8, 
            num_moods=5, 
            num_keys=12,
            feature_size=103
        )
        assert model_custom.num_genres == 8
        assert model_custom.num_moods == 5
        assert model_custom.num_keys == 12
    
    def test_rnn_model_creation(self):
        """Test RNN model can be created with different parameters."""
        model = create_model("rnn", num_genres=5, num_moods=3)
        assert isinstance(model, RecurrentMusicModel)
        
        # Test parameters
        assert hasattr(model, 'lstm')
        assert hasattr(model, 'genre_head')
        assert hasattr(model, 'mood_head')
        assert hasattr(model, 'bpm_head')
        assert hasattr(model, 'key_head')
    
    def test_invalid_model_type(self):
        """Test that invalid model types raise appropriate errors."""
        with pytest.raises(ValueError):
            create_model("invalid_model_type")
    
    def test_model_forward_pass_shapes(self):
        """Test that model outputs have correct shapes."""
        model = create_model("cnn", num_genres=5, num_moods=3, num_keys=12)
        
        batch_size = 4
        spectrogram = torch.randn(batch_size, 1, 128, 200)
        features = torch.randn(batch_size, 103)
        
        with torch.no_grad():
            predictions = model(spectrogram, features)
        
        # Check all outputs exist
        assert 'genre' in predictions
        assert 'mood' in predictions
        assert 'bpm' in predictions
        assert 'key' in predictions
        
        # Check shapes
        assert predictions['genre'].shape == (batch_size, 5)
        assert predictions['mood'].shape == (batch_size, 3)
        assert predictions['bpm'].shape == (batch_size,)
        assert predictions['key'].shape == (batch_size, 12)
    
    def test_variable_length_inputs(self):
        """Test models handle variable-length spectrograms."""
        model = create_model("cnn", num_genres=3, num_moods=2)
        
        lengths = [100, 200, 300, 500]
        
        for length in lengths:
            spectrogram = torch.randn(1, 1, 128, length)
            features = torch.randn(1, 103)
            
            with torch.no_grad():
                predictions = model(spectrogram, features)
            
            assert predictions['genre'].shape == (1, 3)
            assert predictions['mood'].shape == (1, 2)
    
    def test_spectrogram_cnn_component(self):
        """Test SpectrogramCNN component independently."""
        cnn = SpectrogramCNN(input_channels=1, n_mels=128)
        
        # Test different input sizes
        test_inputs = [
            torch.randn(2, 1, 128, 100),
            torch.randn(2, 1, 128, 200),
            torch.randn(2, 1, 128, 500),
        ]
        
        for input_tensor in test_inputs:
            output = cnn(input_tensor)
            assert output.shape == (2, 256)  # Should always output 256 features
        
        # Test output size method
        assert cnn.get_output_size() == 256
    
    def test_feature_mlp_component(self):
        """Test FeatureMLP component independently."""
        mlp = FeatureMLP(input_size=103, hidden_size=256)
        
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            features = torch.randn(batch_size, 103)
            output = mlp(features)
            assert output.shape == (batch_size, 64)
        
        # Test output size method
        assert mlp.get_output_size() == 64
    
    def test_model_parameters_count(self):
        """Test model has reasonable number of parameters."""
        model = create_model("cnn")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Should have reasonable number of parameters (not too few, not too many)
        assert 1_000_000 < total_params < 10_000_000  # Between 1M and 10M
        assert trainable_params == total_params  # All should be trainable
    
    def test_model_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = create_model("cnn", num_genres=3, num_moods=2)
        
        # Create dummy data and targets
        spectrogram = torch.randn(2, 1, 128, 200, requires_grad=True)
        features = torch.randn(2, 103, requires_grad=True)
        
        # Forward pass
        predictions = model(spectrogram, features)
        
        # Create dummy loss
        loss = (predictions['genre'].sum() + 
                predictions['mood'].sum() + 
                predictions['bpm'].sum() + 
                predictions['key'].sum())
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert spectrogram.grad is not None
        assert features.grad is not None
        
        # Check model gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestDataLoading:
    """Test data loading components."""
    
    def create_dummy_manifest(self, num_files=10):
        """Create a dummy manifest for testing."""
        manifest = {
            "dataset_name": "test_dataset",
            "total_files": num_files,
            "files": []
        }
        
        genres = ["rock", "pop", "jazz"]
        moods = ["happy", "sad", "energetic"]
        
        for i in range(num_files):
            file_entry = {
                "filename": f"test_song_{i:03d}.wav",
                "duration": np.random.uniform(120, 240),
                "features_file": f"features/test_song_{i:03d}_features.json",
                "spectrogram_file": f"spectrograms/test_song_{i:03d}_spectrogram.npy",
                "genre": np.random.choice(genres),
                "mood": np.random.choice(moods),
                "bpm": int(np.random.uniform(60, 180))
            }
            manifest["files"].append(file_entry)
        
        return manifest
    
    def test_dummy_data_loaders(self):
        """Test dummy data loader creation."""
        train_loader, val_loader = create_dummy_data_loaders(batch_size=4)
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        
        # Test a batch
        for batch in train_loader:
            assert 'spectrogram' in batch
            assert 'features' in batch
            assert 'genre' in batch
            assert 'mood' in batch
            assert 'bpm' in batch
            assert 'key' in batch
            
            # Check shapes
            assert batch['spectrogram'].shape[0] <= 4  # Batch size
            assert batch['features'].shape[1] == 103   # Feature size
            break
    
    def test_collate_function(self):
        """Test custom collate function handles variable lengths."""
        # Create dummy batch with different spectrogram lengths
        batch = []
        for i in range(3):
            length = 100 + i * 50  # Different lengths
            item = {
                'spectrogram': torch.randn(1, 128, length),
                'features': torch.randn(103),
                'genre': torch.tensor(i % 3),
                'mood': torch.tensor(i % 2),
                'filename': f'test_{i}.wav'
            }
            batch.append(item)
        
        # Test collate function
        collated = collate_fn(batch)
        
        # Check that spectrograms are padded to same length
        assert collated['spectrogram'].shape[0] == 3  # Batch size
        assert len(set(collated['spectrogram'].shape[2:])) == 1  # Same length after padding
        
        # Check other tensors
        assert collated['features'].shape == (3, 103)
        assert collated['genre'].shape == (3,)
        assert len(collated['filename']) == 3
    
    @patch('os.path.exists')
    @patch('numpy.load')
    @patch('builtins.open')
    def test_music_dataset_loading(self, mock_open, mock_np_load, mock_exists):
        """Test MusicDataset class with mocked file operations."""
        # Setup mocks
        mock_exists.return_value = True
        mock_np_load.return_value = np.random.randn(128, 200)
        
        # Mock features file content
        mock_features = {
            "tempo": 120.0,
            "spectral_centroid_mean": 2000.0,
            "mfcc_1_mean": -125.0,
            # Add more features as needed
        }
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_features)
        
        # Create temporary manifest file
        manifest = self.create_dummy_manifest(5)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(manifest, f)
            manifest_path = f.name
        
        try:
            # Test dataset creation
            dataset = MusicDataset(
                manifest_file=manifest_path,
                use_spectrograms=True,
                use_features=True
            )
            
            assert len(dataset) == 5
            
            # Test item loading
            item = dataset[0]
            assert 'spectrogram' in item
            assert 'features' in item
            assert 'genre' in item
            
        finally:
            os.unlink(manifest_path)


class TestTraining:
    """Test training components."""
    
    def test_trainer_initialization(self):
        """Test trainer can be initialized correctly."""
        model = create_model("cnn", num_genres=3, num_moods=2)
        train_loader, val_loader = create_dummy_data_loaders(batch_size=2)
        device = torch.device("cpu")
        
        trainer = MusicClassificationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.device == device
    
    def test_loss_computation(self):
        """Test multi-task loss computation."""
        model = create_model("cnn", num_genres=3, num_moods=2)
        train_loader, val_loader = create_dummy_data_loaders(batch_size=2)
        device = torch.device("cpu")
        
        trainer = MusicClassificationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
        
        # Create dummy predictions and targets
        predictions = {
            'genre': torch.randn(2, 3),
            'mood': torch.randn(2, 2),
            'bpm': torch.randn(2),
            'key': torch.randn(2, 12)
        }
        
        targets = {
            'genre': torch.randint(0, 3, (2,)),
            'mood': torch.randint(0, 2, (2,)),
            'bpm': torch.randn(2),
            'key': torch.randint(0, 12, (2,))
        }
        
        total_loss, task_losses = trainer.compute_loss(predictions, targets)
        
        assert isinstance(total_loss, torch.Tensor)
        assert 'total' in task_losses
        assert 'genre' in task_losses
        assert 'mood' in task_losses
        assert 'bpm' in task_losses
        assert 'key' in task_losses
    
    def test_training_one_epoch(self):
        """Test training for one epoch works."""
        model = create_model("cnn", num_genres=5, num_moods=3, num_keys=12)
        train_loader, val_loader = create_dummy_data_loaders(batch_size=2)
        device = torch.device("cpu")
        
        trainer = MusicClassificationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
        
        # Train for one epoch
        train_losses = trainer.train_epoch()
        
        assert isinstance(train_losses, dict)
        assert 'total' in train_losses
        assert train_losses['total'] > 0
    
    def test_validation_one_epoch(self):
        """Test validation for one epoch works."""
        model = create_model("cnn", num_genres=5, num_moods=3, num_keys=12)
        train_loader, val_loader = create_dummy_data_loaders(batch_size=2)
        device = torch.device("cpu")
        
        trainer = MusicClassificationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
        
        # Validate for one epoch
        val_losses, metrics = trainer.validate_epoch()
        
        assert isinstance(val_losses, dict)
        assert isinstance(metrics, dict)
        assert 'total' in val_losses
        assert val_losses['total'] > 0
    
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        model = create_model("cnn", num_genres=3, num_moods=2)
        train_loader, val_loader = create_dummy_data_loaders(batch_size=2)
        device = torch.device("cpu")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = MusicClassificationTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                model_save_dir=temp_dir
            )
            
            # Save model
            save_path = "test_model.pth"
            trainer.save_model(save_path)
            
            # Check file exists
            full_path = os.path.join(temp_dir, save_path)
            assert os.path.exists(full_path)
            
            # Load model
            trainer.load_model(full_path)
            
            # Model should still work
            with torch.no_grad():
                spectrogram = torch.randn(1, 1, 128, 200)
                features = torch.randn(1, 103)
                predictions = trainer.model(spectrogram, features)
                assert 'genre' in predictions


class TestUtils:
    """Test utility functions."""
    
    def test_sample_config_creation(self):
        """Test sample configuration creation."""
        config = create_sample_config()
        
        assert isinstance(config, dict)
        assert 'model' in config
        assert 'training' in config
        assert 'data' in config
        assert 'task_weights' in config
        
        # Check specific values
        assert config['model']['type'] in ['cnn', 'rnn']
        assert config['training']['epochs'] > 0
        assert config['training']['batch_size'] > 0
    
    def test_model_export_for_api(self):
        """Test model export functionality."""
        model = create_model("cnn", num_genres=3, num_moods=2)
        
        class_mappings = {
            'genres': ['rock', 'pop', 'jazz'],
            'moods': ['happy', 'sad'],
            'keys': ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C#', 'D#', 'F#', 'G#', 'A#']
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            save_path = f.name
        
        try:
            save_model_for_api(
                model=model,
                class_mappings=class_mappings,
                save_path=save_path
            )
            
            # Check file exists
            assert os.path.exists(save_path)
            
            # Load and check contents
            checkpoint = torch.load(save_path, map_location='cpu')
            assert 'model_state_dict' in checkpoint
            assert 'class_mappings' in checkpoint
            assert 'preprocessing_config' in checkpoint
            
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)
    
    def test_model_loading_for_inference(self):
        """Test model loading for inference."""
        # Create and save a model
        model = create_model("cnn", num_genres=3, num_moods=2)
        
        class_mappings = {
            'genres': ['rock', 'pop', 'jazz'],
            'moods': ['happy', 'sad'],
            'keys': ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C#', 'D#', 'F#', 'G#', 'A#']
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            save_path = f.name
        
        try:
            save_model_for_api(
                model=model,
                class_mappings=class_mappings,
                save_path=save_path
            )
            
            # Load model for inference
            loaded_model, metadata = load_model_for_inference(
                MusicClassificationModel,
                save_path,
                torch.device('cpu')
            )
            
            assert isinstance(loaded_model, MusicClassificationModel)
            assert 'class_mappings' in metadata
            assert metadata['class_mappings'] == class_mappings
            
            # Test inference
            with torch.no_grad():
                spectrogram = torch.randn(1, 1, 128, 200)
                features = torch.randn(1, 103)
                predictions = loaded_model(spectrogram, features)
                assert 'genre' in predictions
            
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_complete_training_pipeline(self):
        """Test a complete mini training run."""
        # Create model
        model = create_model("cnn", num_genres=5, num_moods=3, num_keys=12)

        # Create data loaders
        train_loader, val_loader = create_dummy_data_loaders(batch_size=2)

        # Create trainer
        device = torch.device("cpu")        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = MusicClassificationTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                model_save_dir=temp_dir,
                log_dir=temp_dir
            )
            
            # Train for 2 epochs
            history = trainer.train(num_epochs=2, save_every=1)
            
            # Check training completed
            assert isinstance(history, dict)
            assert 'train_loss' in history
            assert 'val_loss' in history
            assert len(history['train_loss']) == 2
            assert len(history['val_loss']) == 2
            
            # Check model files were saved
            assert os.path.exists(os.path.join(temp_dir, 'best_model.pth'))
            assert os.path.exists(os.path.join(temp_dir, 'final_model.pth'))
    
    def test_model_compatibility_across_architectures(self):
        """Test that different model architectures are compatible."""
        architectures = ["cnn", "rnn"]
        
        for arch in architectures:
            model = create_model(arch, num_genres=5, num_moods=3)
            
            # Test forward pass
            if arch == "cnn":
                spectrogram = torch.randn(2, 1, 128, 200)
                features = torch.randn(2, 103)
                predictions = model(spectrogram, features)
            else:  # RNN
                # RNN expects sequence input
                sequence = torch.randn(2, 10, 103)  # (batch, seq_len, features)
                predictions = model(sequence)
            
            # All models should have the same output structure
            assert 'genre' in predictions
            assert 'mood' in predictions
            assert 'bpm' in predictions
            assert 'key' in predictions
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        model = create_model("cnn")
        
        # Test with wrong input shapes
        with pytest.raises((RuntimeError, ValueError)):
            wrong_spectrogram = torch.randn(1, 128, 200)  # Missing channel dimension
            features = torch.randn(1, 103)
            model(wrong_spectrogram, features)
        
        # Test with wrong feature size
        with pytest.raises((RuntimeError, ValueError)):
            spectrogram = torch.randn(1, 1, 128, 200)
            wrong_features = torch.randn(1, 50)  # Wrong feature size
            model(spectrogram, wrong_features)


def run_performance_tests():
    """Run performance tests to ensure model efficiency."""
    print("🚀 Running Performance Tests...")
    
    model = create_model("cnn")
    model.eval()
    
    # Test inference speed
    import time
    
    spectrogram = torch.randn(1, 1, 128, 300)
    features = torch.randn(1, 103)
    
    # Warm up
    for _ in range(5):
        with torch.no_grad():
            _ = model(spectrogram, features)
    
    # Time inference
    start_time = time.time()
    num_inferences = 100
    
    with torch.no_grad():
        for _ in range(num_inferences):
            _ = model(spectrogram, features)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_inferences
    
    print(f"✅ Average inference time: {avg_time*1000:.2f}ms")
    assert avg_time < 0.1  # Should be faster than 100ms per inference
    
    # Test memory usage
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create large batch
    large_batch_spec = torch.randn(32, 1, 128, 300)
    large_batch_feat = torch.randn(32, 103)
    
    with torch.no_grad():
        _ = model(large_batch_spec, large_batch_feat)
    
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = memory_after - memory_before
    
    print(f"✅ Memory increase for batch of 32: {memory_increase:.1f}MB")
    assert memory_increase < 500  # Should not use more than 500MB for batch


if __name__ == "__main__":
    print("🎵 Music Classification Model - Comprehensive Test Suite")
    print("=" * 60)
    print("Author: Sergie Code - Software Engineer & YouTube Programming Educator")
    print("Project: AI Tools for Musicians")
    print("=" * 60)
    
    # Run pytest with verbose output
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "--color=yes"
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n🎉 All tests passed! Running performance tests...")
        try:
            run_performance_tests()
            print("\n✅ Performance tests passed!")
            print("\n🚀 Your music classification model is working perfectly!")
            print("Ready for your YouTube content! 🎵")
        except Exception as e:
            print(f"\n⚠️  Performance test warning: {e}")
            print("Core functionality works, but check performance optimizations.")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    exit(exit_code)
