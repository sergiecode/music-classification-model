#!/usr/bin/env python3
"""
Music Classification Example Script
===================================

Author: Sergie Code - Software Engineer & YouTube Programming Educator
Project: AI Tools for Musicians
Date: August 29, 2025

This script demonstrates how to use the music classification model
with dummy data for testing and learning purposes.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import create_model
from training import MusicClassificationTrainer, create_dummy_data_loaders
from utils import save_model_for_api


def create_dummy_manifest():
    """Create a dummy manifest file for testing."""
    dummy_manifest = {
        "dataset_name": "dummy_music_dataset",
        "created_by": "music-classification-model-example",
        "total_files": 100,
        "total_errors": 0,
        "processing_date": "2025-08-29T12:00:00Z",
        "statistics": {
            "total_duration_hours": 8.3,
            "average_duration_seconds": 180.0,
            "sample_rates": [22050],
            "file_formats": ["wav"]
        },
        "files": []
    }
    
    # Generate dummy file entries
    genres = ["rock", "pop", "jazz", "classical", "electronic"]
    moods = ["energetic", "calm", "happy", "melancholic"]
    
    for i in range(100):
        file_entry = {
            "filename": f"dummy_song_{i:03d}.wav",
            "duration": np.random.uniform(120, 240),
            "sample_rate": 22050,
            "features_file": f"data/features/dummy_song_{i:03d}_features.json",
            "spectrogram_file": f"data/spectrograms/dummy_song_{i:03d}_spectrogram.npy",
            "file_size_mb": np.random.uniform(8, 15),
            "genre": np.random.choice(genres),
            "mood": np.random.choice(moods),
            "bpm": int(np.random.uniform(60, 180))
        }
        dummy_manifest["files"].append(file_entry)
    
    return dummy_manifest


def run_training_example():
    """Run a complete training example with dummy data."""
    print("🎵 Music Classification Model - Training Example")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Model configuration
    num_genres = 5
    num_moods = 4
    num_keys = 12
    
    print(f"\nModel Configuration:")
    print(f"- Genres: {num_genres}")
    print(f"- Moods: {num_moods}")
    print(f"- Keys: {num_keys}")
    print(f"- Features: 103")
    
    # Create model
    print("\n📱 Creating CNN model...")
    model = create_model(
        model_type="cnn",
        num_genres=num_genres,
        num_moods=num_moods,
        num_keys=num_keys
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create dummy data loaders
    print("\n📊 Creating dummy data loaders...")
    train_loader, val_loader = create_dummy_data_loaders(batch_size=8)
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create trainer
    print("\n🏋️ Setting up trainer...")
    trainer = MusicClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001
    )
    
    # Train model
    print("\n🚀 Starting training (5 epochs for demo)...")
    history = trainer.train(
        num_epochs=5,
        save_every=2,
        early_stopping_patience=10
    )
    
    print("\n✅ Training completed!")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    
    return model, trainer, history


def run_inference_example(model):
    """Run inference example with dummy data."""
    print("\n🔮 Running Inference Example")
    print("=" * 30)
    
    model.eval()
    device = next(model.parameters()).device
    
    # Create dummy input data
    batch_size = 1
    n_mels = 128
    time_frames = 300
    feature_size = 103
    
    # Dummy spectrogram and features
    dummy_spectrogram = torch.randn(batch_size, 1, n_mels, time_frames).to(device)
    dummy_features = torch.randn(batch_size, feature_size).to(device)
    
    print(f"Input spectrogram shape: {dummy_spectrogram.shape}")
    print(f"Input features shape: {dummy_features.shape}")
    
    # Run inference
    with torch.no_grad():
        predictions = model(dummy_spectrogram, dummy_features)
    
    print("\n📈 Predictions:")
    
    # Genre prediction
    genre_probs = torch.softmax(predictions['genre'], dim=1)
    genre_pred = torch.argmax(genre_probs, dim=1)
    print(f"Genre: Class {genre_pred.item()} (confidence: {genre_probs[0, genre_pred].item():.3f})")
    
    # Mood prediction
    mood_probs = torch.softmax(predictions['mood'], dim=1)
    mood_pred = torch.argmax(mood_probs, dim=1)
    print(f"Mood: Class {mood_pred.item()} (confidence: {mood_probs[0, mood_pred].item():.3f})")
    
    # BPM prediction
    bpm_pred = predictions['bpm']
    print(f"BPM: {bpm_pred.item():.1f}")
    
    # Key prediction
    key_probs = torch.softmax(predictions['key'], dim=1)
    key_pred = torch.argmax(key_probs, dim=1)
    print(f"Key: Class {key_pred.item()} (confidence: {key_probs[0, key_pred].item():.3f})")
    
    return predictions


def run_model_export_example(model):
    """Demonstrate model export for API integration."""
    print("\n💾 Model Export Example")
    print("=" * 25)
    
    # Define class mappings (these would come from your training data)
    class_mappings = {
        'genres': ['rock', 'pop', 'jazz', 'classical', 'electronic'],
        'moods': ['energetic', 'calm', 'happy', 'melancholic'],
        'keys': ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    }
    
    # Save model for API
    save_path = "models/example_model_for_api.pth"
    save_model_for_api(
        model=model,
        class_mappings=class_mappings,
        save_path=save_path
    )
    
    print(f"✅ Model exported to: {save_path}")
    print("This model can now be used with the music-classification-api!")
    
    return save_path


def show_integration_example():
    """Show how this integrates with the preprocessing pipeline."""
    print("\n🔗 Integration with Preprocessing Pipeline")
    print("=" * 45)
    
    print("1. Preprocessing Pipeline (music-classification-preprocessing):")
    print("   - Processes audio files")
    print("   - Extracts 103 features per file")
    print("   - Generates mel spectrograms")
    print("   - Creates manifest.json with file metadata")
    
    print("\n2. This Model Repository (music-classification-model):")
    print("   - Loads data using manifest.json")
    print("   - Trains CNN/RNN models on features + spectrograms")
    print("   - Supports multi-task learning (genre, mood, BPM, key)")
    print("   - Exports trained models for API use")
    
    print("\n3. API Repository (music-classification-api - future):")
    print("   - Loads trained model from this repository")
    print("   - Uses preprocessing components for real-time inference")
    print("   - Provides REST API for music classification")
    
    print("\n📂 Expected File Structure from Preprocessing:")
    print("data/")
    print("├── processed/")
    print("│   ├── features/")
    print("│   │   ├── song1_features.json")
    print("│   │   └── song2_features.json")
    print("│   ├── spectrograms/")
    print("│   │   ├── song1_spectrogram.npy")
    print("│   │   └── song2_spectrogram.npy")
    print("│   └── dataset_manifest.json")


def main():
    """Main example script."""
    print("🎼 Welcome to Sergie Code's Music Classification Model!")
    print("This example demonstrates the complete training and inference pipeline.")
    print("\nFor educational purposes - AI Tools for Musicians series")
    print("=" * 60)
    
    # Show integration info
    show_integration_example()
    
    # Create dummy manifest for reference
    print("\n📝 Creating dummy manifest file...")
    dummy_manifest = create_dummy_manifest()
    manifest_path = "examples/dummy_manifest.json"
    
    # Create examples directory
    import os
    os.makedirs("examples", exist_ok=True)
    
    with open(manifest_path, 'w') as f:
        json.dump(dummy_manifest, f, indent=2)
    print(f"Dummy manifest saved to: {manifest_path}")
    
    # Run training example
    model, trainer, history = run_training_example()
    
    # Run inference example
    predictions = run_inference_example(model)
    
    # Export model
    model_path = run_model_export_example(model)
    
    print("\n🎉 Example completed successfully!")
    print("\nNext steps:")
    print("1. Replace dummy data with real preprocessed music data")
    print("2. Adjust model hyperparameters for your dataset")
    print("3. Train for more epochs with real data")
    print("4. Use the exported model with the music-classification-api")
    
    print(f"\n📊 Training Summary:")
    print(f"- Training loss: {history['train_loss'][-1]:.4f}")
    print(f"- Validation loss: {history['val_loss'][-1]:.4f}")
    print(f"- Model saved at: {model_path}")
    
    print("\nHappy coding! 🎵")


if __name__ == "__main__":
    main()
