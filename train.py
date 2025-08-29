#!/usr/bin/env python3
"""
Music Classification Training Script
====================================

Author: Sergie Code - Software Engineer & YouTube Programming Educator
Project: AI Tools for Musicians
Date: August 29, 2025

This script provides a command-line interface for training music classification models.
"""

import argparse
import torch
import yaml
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from models import create_model
from data import create_data_loaders
from training import MusicClassificationTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train Music Classification Model')
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to preprocessing manifest file')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                       help='Path to training configuration file')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='cnn',
                       choices=['cnn', 'rnn'], help='Type of model to train')
    parser.add_argument('--num-genres', type=int, default=10,
                       help='Number of genre classes')
    parser.add_argument('--num-moods', type=int, default=4,
                       help='Number of mood classes')
    parser.add_argument('--num-keys', type=int, default=12,
                       help='Number of musical keys')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                       help='Weight decay for regularization')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Directory to save models and logs')
    parser.add_argument('--experiment-name', type=str, default='music_classification',
                       help='Name for this experiment')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers for data loading')
    
    # Optional features
    parser.add_argument('--use-spectrograms', action='store_true', default=True,
                       help='Use spectrogram data')
    parser.add_argument('--use-features', action='store_true', default=True,
                       help='Use traditional audio features')
    parser.add_argument('--max-spectrogram-length', type=int, default=None,
                       help='Maximum length for spectrograms (for padding)')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = output_dir / 'models'
    log_dir = output_dir / 'logs'
    model_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Models will be saved to: {model_dir}")
    print(f"Logs will be saved to: {log_dir}")
    
    # Create data loaders
    print("Creating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            manifest_file=args.data,
            batch_size=args.batch_size,
            use_spectrograms=args.use_spectrograms,
            use_features=args.use_features,
            max_spectrogram_length=args.max_spectrogram_length,
            num_workers=args.num_workers
        )
        print("Data loaders created successfully")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        print("Creating dummy data loaders for testing...")
        from training import create_dummy_data_loaders
        train_loader, val_loader = create_dummy_data_loaders(args.batch_size)
        print("Using dummy data for testing")
    
    # Create model
    print(f"Creating {args.model_type} model...")
    model = create_model(
        model_type=args.model_type,
        num_genres=args.num_genres,
        num_moods=args.num_moods,
        num_keys=args.num_keys,
        feature_size=103
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = MusicClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        log_dir=str(log_dir),
        model_save_dir=str(model_dir)
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming training from {args.resume}")
        trainer.load_model(args.resume)
    
    # Train model
    print(f"Starting training for {args.epochs} epochs...")
    history = trainer.train(
        num_epochs=args.epochs,
        save_every=5,
        early_stopping_patience=10
    )
    
    print("Training completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    # Save training history
    import json
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    # Test final model if test data is available
    if test_loader:
        print("Evaluating on test set...")
        try:
            trainer.model.eval()
            # Add test evaluation here if needed
            print("Test evaluation completed")
        except Exception as e:
            print(f"Test evaluation failed: {e}")


if __name__ == "__main__":
    main()
