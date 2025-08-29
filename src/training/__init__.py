"""
Music Classification Training Module
====================================

Author: Sergie Code - Software Engineer & YouTube Programming Educ           # BPM regression loss
        if 'bpm' in predictions and 'bpm' in targets:
            bpm_loss = self.criterion_regression(predictions['bpm'], targets['bpm'])
            losses['bpm'] = bpm_loss.item()
            total_loss += self.task_weights['bpm'] * bpm_loss           losses['bpm'] = bpm_loss.item()# BPM regression loss
        if 'bpm' in predictions and 'bpm' in targets:
            bpm_loss = self.criterion_regression(predictions['bpm'], targets['bpm'])
            losses['bpm'] = bpm_loss.item()
            total_loss += self.task_weights['bpm'] * bpm_loss
Project: AI Tools for Musicians
Date: August 29, 2025

This module provides training functionality for music classification models.
Supports multi-task learning for genre, mood, BPM, and key prediction.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MusicClassificationTrainer:
    """
    Trainer class for music classification models.
    
    Handles multi-task training with support for:
    - Genre classification
    - Mood classification  
    - BPM regression
    - Key classification
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        task_weights: Optional[Dict[str, float]] = None,
        log_dir: str = "logs",
        model_save_dir: str = "models"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on (CPU or CUDA)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            task_weights: Weights for different tasks in loss computation
            log_dir: Directory for tensorboard logs
            model_save_dir: Directory to save model checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.log_dir = log_dir
        self.model_save_dir = model_save_dir
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Task weights for multi-task learning
        self.task_weights = task_weights or {
            'genre': 1.0,
            'mood': 1.0,
            'bpm': 0.1,  # Lower weight for regression task
            'key': 0.8
        }
        
        # Loss functions
        self.criterion_classification = nn.CrossEntropyLoss()
        self.criterion_regression = nn.MSELoss()
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"Trainer initialized - Device: {device}")
        logger.info(f"Task weights: {self.task_weights}")
    
    def compute_loss(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Model predictions for each task
            targets: Ground truth targets for each task
            
        Returns:
            Total loss and individual task losses
        """
        losses = {}
        total_loss = 0.0
        
        # Genre classification loss
        if 'genre' in predictions and 'genre' in targets:
            genre_loss = self.criterion_classification(predictions['genre'], targets['genre'])
            losses['genre'] = genre_loss.item()
            total_loss += self.task_weights['genre'] * genre_loss
        
        # Mood classification loss
        if 'mood' in predictions and 'mood' in targets:
            mood_loss = self.criterion_classification(predictions['mood'], targets['mood'])
            losses['mood'] = mood_loss.item()
            total_loss += self.task_weights['mood'] * mood_loss
        
        # BPM regression loss
        if 'bpm' in predictions and 'bpm' in targets:
            bpm_loss = self.criterion_regression(predictions['bpm'], targets['bpm'])
            losses['bpm'] = bpm_loss.item()
            total_loss += self.task_weights['bpm'] * bpm_loss
        
        # Key classification loss
        if 'key' in predictions and 'key' in targets:
            key_loss = self.criterion_classification(predictions['key'], targets['key'])
            losses['key'] = key_loss.item()
            total_loss += self.task_weights['key'] * key_loss
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_losses = {
            'total': 0.0,
            'genre': 0.0,
            'mood': 0.0,
            'bpm': 0.0,
            'key': 0.0
        }
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} Training")
        
        for batch in progress_bar:
            # Move data to device
            spectrograms = batch.get('spectrogram', torch.zeros(1, 1, 128, 100)).to(self.device)
            features = batch.get('features', torch.zeros(1, 103)).to(self.device)
            
            # Prepare targets
            targets = {}
            for task in ['genre', 'mood', 'bpm', 'key']:
                if task in batch:
                    targets[task] = batch[task].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(spectrograms, features)
            
            # Compute loss
            loss, task_losses = self.compute_loss(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Accumulate losses
            for task, task_loss in task_losses.items():
                total_losses[task] += task_loss
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{task_losses['total']:.4f}",
                'genre': f"{task_losses.get('genre', 0):.3f}",
                'mood': f"{task_losses.get('mood', 0):.3f}",
                'bpm': f"{task_losses.get('bpm', 0):.3f}",
                'key': f"{task_losses.get('key', 0):.3f}"
            })
        
        # Average losses
        avg_losses = {task: loss / num_batches for task, loss in total_losses.items()}
        
        return avg_losses
    
    def validate_epoch(self) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_losses = {
            'total': 0.0,
            'genre': 0.0,
            'mood': 0.0,
            'bpm': 0.0,
            'key': 0.0
        }
        
        # For metrics computation
        all_predictions = {
            'genre': [],
            'mood': [],
            'bpm': [],
            'key': []
        }
        all_targets = {
            'genre': [],
            'mood': [],
            'bpm': [],
            'key': []
        }
        
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for batch in progress_bar:
                # Move data to device
                spectrograms = batch.get('spectrogram', torch.zeros(1, 1, 128, 100)).to(self.device)
                features = batch.get('features', torch.zeros(1, 103)).to(self.device)
                
                # Prepare targets
                targets = {}
                for task in ['genre', 'mood', 'bpm', 'key']:
                    if task in batch:
                        targets[task] = batch[task].to(self.device)
                
                # Forward pass
                predictions = self.model(spectrograms, features)
                
                # Compute loss
                loss, task_losses = self.compute_loss(predictions, targets)
                
                # Accumulate losses
                for task, task_loss in task_losses.items():
                    total_losses[task] += task_loss
                num_batches += 1
                
                # Collect predictions and targets for metrics
                for task in ['genre', 'mood', 'bpm', 'key']:
                    if task in predictions and task in targets:
                        if task == 'bpm':
                            # Regression task
                            all_predictions[task].extend(predictions[task].cpu().numpy())
                            all_targets[task].extend(targets[task].cpu().numpy())
                        else:
                            # Classification tasks
                            pred_classes = torch.argmax(predictions[task], dim=1)
                            all_predictions[task].extend(pred_classes.cpu().numpy())
                            all_targets[task].extend(targets[task].cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'val_loss': f"{task_losses['total']:.4f}"
                })
        
        # Average losses
        avg_losses = {task: loss / num_batches for task, loss in total_losses.items()}
        
        # Compute metrics
        metrics = self.compute_metrics(all_predictions, all_targets)
        
        return avg_losses, metrics
    
    def compute_metrics(
        self, 
        predictions: Dict[str, List], 
        targets: Dict[str, List]
    ) -> Dict[str, Any]:
        """Compute evaluation metrics for each task."""
        metrics = {}
        
        for task in ['genre', 'mood', 'key']:
            if predictions[task] and targets[task]:
                # Classification metrics
                accuracy = accuracy_score(targets[task], predictions[task])
                metrics[f'{task}_accuracy'] = accuracy
                
                # Classification report (if we have enough data)
                if len(set(targets[task])) > 1:
                    report = classification_report(
                        targets[task], 
                        predictions[task], 
                        output_dict=True,
                        zero_division=0
                    )
                    metrics[f'{task}_f1'] = report.get('weighted avg', {}).get('f1-score', 0.0)
        
        # BPM regression metrics
        if predictions['bpm'] and targets['bpm']:
            mae = mean_absolute_error(targets['bpm'], predictions['bpm'])
            mse = np.mean((np.array(targets['bpm']) - np.array(predictions['bpm'])) ** 2)
            metrics['bpm_mae'] = mae
            metrics['bpm_mse'] = mse
        
        return metrics
    
    def train(
        self, 
        num_epochs: int, 
        save_every: int = 5,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save model every N epochs
            early_stopping_patience: Stop training if no improvement for N epochs
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        no_improvement_count = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training
            train_losses = self.train_epoch()
            
            # Validation
            val_losses, val_metrics = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_losses['total'])
            
            # Record history
            history['train_loss'].append(train_losses['total'])
            history['val_loss'].append(val_losses['total'])
            
            # Tensorboard logging
            for task, loss in train_losses.items():
                self.writer.add_scalar(f'Loss/Train_{task}', loss, epoch)
            
            for task, loss in val_losses.items():
                self.writer.add_scalar(f'Loss/Val_{task}', loss, epoch)
            
            for metric, value in val_metrics.items():
                self.writer.add_scalar(f'Metrics/{metric}', value, epoch)
            
            # Logging
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - "
                       f"Train Loss: {train_losses['total']:.4f}, "
                       f"Val Loss: {val_losses['total']:.4f}, "
                       f"Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_model('best_model.pth')
                no_improvement_count = 0
                logger.info(f"New best model saved with val loss: {self.best_val_loss:.4f}")
            else:
                no_improvement_count += 1
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_model(f'checkpoint_epoch_{epoch + 1}.pth')
            
            # Early stopping
            if no_improvement_count >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        self.save_model('final_model.pth')
        
        logger.info("Training completed!")
        return history
    
    def save_model(self, filename: str):
        """Save model checkpoint."""
        filepath = os.path.join(self.model_save_dir, filename)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'task_weights': self.task_weights
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.task_weights = checkpoint.get('task_weights', self.task_weights)
        
        logger.info(f"Model loaded from {filepath}")


def create_dummy_data_loaders(batch_size: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create dummy data loaders for testing training pipeline.
    
    Args:
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import TensorDataset
    
    # Create dummy data
    num_samples = 100
    n_mels = 128
    time_frames = 300
    feature_size = 103
    
    # Dummy spectrograms and features
    spectrograms = torch.randn(num_samples, 1, n_mels, time_frames)
    features = torch.randn(num_samples, feature_size)
    
    # Dummy labels
    genres = torch.randint(0, 5, (num_samples,))  # 5 genres
    moods = torch.randint(0, 3, (num_samples,))   # 3 moods
    bpms = torch.rand(num_samples) * 100 + 60     # BPM between 60-160
    keys = torch.randint(0, 12, (num_samples,))   # 12 keys
    
    # Create dataset
    dataset = TensorDataset(spectrograms, features, genres, moods, bpms, keys)
    
    # Split into train and validation
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders with custom collate function
    def dummy_collate_fn(batch):
        spectrograms, features, genres, moods, bpms, keys = zip(*batch)
        return {
            'spectrogram': torch.stack(spectrograms),
            'features': torch.stack(features),
            'genre': torch.stack(genres),
            'mood': torch.stack(moods),
            'bpm': torch.stack(bpms),
            'key': torch.stack(keys)
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dummy_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dummy_collate_fn
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example training with dummy data
    print("Music Classification Trainer - Testing with dummy data")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Import model (this would be from the models module)
    import sys
    sys.path.append('..')
    from models import create_model
    
    # Create model
    model = create_model("cnn", num_genres=5, num_moods=3, num_keys=12)
    
    # Create dummy data loaders
    train_loader, val_loader = create_dummy_data_loaders(batch_size=4)
    
    # Create trainer
    trainer = MusicClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001
    )
    
    # Train for a few epochs
    print("Starting dummy training...")
    history = trainer.train(num_epochs=3, save_every=2, early_stopping_patience=5)
    
    print("Training completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
