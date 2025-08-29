"""
Utility Functions for Music Classification
==========================================

Author: Sergie Code - Software Engineer & YouTube Programming Educator
Project: AI Tools for Musicians
Date: August 29, 2025

This module provides utility functions for model evaluation, visualization,
and data processing.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple, Any
import json


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on
        class_names: Dictionary mapping task names to class names
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
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
    
    total_loss = 0.0
    num_batches = 0
    
    criterion_classification = torch.nn.CrossEntropyLoss()
    criterion_regression = torch.nn.MSELoss()
    
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            spectrograms = batch.get('spectrogram', torch.zeros(1, 1, 128, 100)).to(device)
            features = batch.get('features', torch.zeros(1, 103)).to(device)
            
            # Forward pass
            predictions = model(spectrograms, features)
            
            # Compute loss
            batch_loss = 0.0
            
            for task in ['genre', 'mood', 'bpm', 'key']:
                if task in predictions and task in batch:
                    targets = batch[task].to(device)
                    
                    if task == 'bpm':
                        # Regression task
                        task_loss = criterion_regression(predictions[task], targets)
                        all_predictions[task].extend(predictions[task].cpu().numpy())
                        all_targets[task].extend(targets.cpu().numpy())
                    else:
                        # Classification tasks
                        task_loss = criterion_classification(predictions[task], targets)
                        pred_classes = torch.argmax(predictions[task], dim=1)
                        all_predictions[task].extend(pred_classes.cpu().numpy())
                        all_targets[task].extend(targets.cpu().numpy())
                    
                    batch_loss += task_loss.item()
            
            total_loss += batch_loss
            num_batches += 1
    
    # Compute metrics
    metrics = {
        'average_loss': total_loss / num_batches if num_batches > 0 else 0.0
    }
    
    # Classification metrics
    for task in ['genre', 'mood', 'key']:
        if all_predictions[task] and all_targets[task]:
            # Accuracy
            correct = sum(p == t for p, t in zip(all_predictions[task], all_targets[task]))
            accuracy = correct / len(all_predictions[task])
            metrics[f'{task}_accuracy'] = accuracy
            
            # Classification report
            if len(set(all_targets[task])) > 1:
                report = classification_report(
                    all_targets[task],
                    all_predictions[task],
                    target_names=class_names.get(task, None),
                    output_dict=True,
                    zero_division=0
                )
                metrics[f'{task}_classification_report'] = report
                metrics[f'{task}_f1_score'] = report.get('weighted avg', {}).get('f1-score', 0.0)
    
    # Regression metrics for BPM
    if all_predictions['bpm'] and all_targets['bpm']:
        pred_bpm = np.array(all_predictions['bpm'])
        target_bpm = np.array(all_targets['bpm'])
        
        mae = np.mean(np.abs(pred_bpm - target_bpm))
        mse = np.mean((pred_bpm - target_bpm) ** 2)
        rmse = np.sqrt(mse)
        
        metrics['bpm_mae'] = mae
        metrics['bpm_mse'] = mse
        metrics['bpm_rmse'] = rmse
        
        # Correlation
        if len(pred_bpm) > 1:
            correlation = np.corrcoef(pred_bpm, target_bpm)[0, 1]
            metrics['bpm_correlation'] = correlation
    
    return metrics


def plot_confusion_matrices(
    predictions: Dict[str, List],
    targets: Dict[str, List],
    class_names: Dict[str, List[str]],
    save_path: str = None
):
    """
    Plot confusion matrices for classification tasks.
    
    Args:
        predictions: Dictionary of predictions for each task
        targets: Dictionary of targets for each task
        class_names: Dictionary mapping task names to class names
        save_path: Path to save the plot
    """
    classification_tasks = ['genre', 'mood', 'key']
    available_tasks = [task for task in classification_tasks 
                      if predictions[task] and targets[task]]
    
    if not available_tasks:
        print("No classification tasks available for confusion matrix plotting")
        return
    
    fig, axes = plt.subplots(1, len(available_tasks), figsize=(5 * len(available_tasks), 4))
    if len(available_tasks) == 1:
        axes = [axes]
    
    for i, task in enumerate(available_tasks):
        cm = confusion_matrix(targets[task], predictions[task])
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names.get(task, range(cm.shape[1])),
            yticklabels=class_names.get(task, range(cm.shape[0])),
            ax=axes[i]
        )
        
        axes[i].set_title(f'{task.capitalize()} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to {save_path}")
    
    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str = None
):
    """
    Plot training history (loss curves).
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    if 'train_loss' in history and 'val_loss' in history:
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
    
    # Accuracy plot (if available)
    if 'train_accuracy' in history and 'val_accuracy' in history:
        epochs = range(1, len(history['train_accuracy']) + 1)
        axes[1].plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy')
        axes[1].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    else:
        # Remove empty subplot
        fig.delaxes(axes[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_bpm_regression(
    predictions: List[float],
    targets: List[float],
    save_path: str = None
):
    """
    Plot BPM regression results.
    
    Args:
        predictions: Predicted BPM values
        targets: Target BPM values
        save_path: Path to save the plot
    """
    if not predictions or not targets:
        print("No BPM data available for plotting")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Scatter plot
    axes[0].scatter(targets, predictions, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    axes[0].set_xlabel('Actual BPM')
    axes[0].set_ylabel('Predicted BPM')
    axes[0].set_title('BPM Prediction vs Actual')
    axes[0].legend()
    axes[0].grid(True)
    
    # Residuals plot
    residuals = np.array(predictions) - np.array(targets)
    axes[1].scatter(targets, residuals, alpha=0.6)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Actual BPM')
    axes[1].set_ylabel('Residuals (Predicted - Actual)')
    axes[1].set_title('BPM Prediction Residuals')
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"BPM regression plot saved to {save_path}")
    
    plt.show()


def save_model_for_api(
    model: torch.nn.Module,
    class_mappings: Dict[str, List[str]],
    save_path: str,
    preprocessing_config: Dict[str, Any] = None
):
    """
    Save a trained model in the format expected by the API.
    
    Args:
        model: Trained PyTorch model
        class_mappings: Dictionary mapping task names to class labels
        save_path: Path to save the model
        preprocessing_config: Configuration used for preprocessing
    """
    if preprocessing_config is None:
        preprocessing_config = {
            'sample_rate': 22050,
            'n_mels': 128,
            'hop_length': 512,
            'feature_count': 103
        }
    
    # Save complete model information
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model.__class__.__name__,
        'feature_size': 103,
        'spectrogram_shape': (128, None),  # Variable length
        'class_mappings': class_mappings,
        'preprocessing_config': preprocessing_config,
        'model_info': {
            'created_by': 'Sergie Code - Music Classification Model',
            'framework': 'PyTorch',
            'tasks': ['genre', 'mood', 'bpm', 'key']
        }
    }, save_path)
    
    print(f"Model saved for API integration at: {save_path}")


def load_model_for_inference(
    model_class,
    model_path: str,
    device: torch.device
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load a saved model for inference.
    
    Args:
        model_class: Model class to instantiate
        model_path: Path to saved model
        device: Device to load model on
        
    Returns:
        Loaded model and metadata
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    class_mappings = checkpoint.get('class_mappings', {})
    
    # Create model instance
    model = model_class(
        num_genres=len(class_mappings.get('genres', ['unknown'])),
        num_moods=len(class_mappings.get('moods', ['unknown'])),
        num_keys=len(class_mappings.get('keys', list(range(12)))),
        feature_size=checkpoint.get('feature_size', 103)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    metadata = {
        'class_mappings': class_mappings,
        'preprocessing_config': checkpoint.get('preprocessing_config', {}),
        'model_info': checkpoint.get('model_info', {})
    }
    
    return model, metadata


def create_sample_config():
    """Create a sample training configuration file."""
    config = {
        'model': {
            'type': 'cnn',
            'num_genres': 10,
            'num_moods': 4,
            'num_keys': 12,
            'dropout_rate': 0.3
        },
        'training': {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'early_stopping_patience': 10
        },
        'data': {
            'use_spectrograms': True,
            'use_features': True,
            'max_spectrogram_length': None,
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        },
        'task_weights': {
            'genre': 1.0,
            'mood': 1.0,
            'bpm': 0.1,
            'key': 0.8
        }
    }
    
    return config


if __name__ == "__main__":
    # Example usage
    print("Music Classification Utilities")
    
    # Create sample config
    config = create_sample_config()
    print("Sample configuration:")
    print(json.dumps(config, indent=2))
    
    # Create dummy data for testing visualization
    dummy_predictions = {
        'genre': [0, 1, 2, 0, 1] * 10,
        'mood': [0, 1, 0, 1, 2] * 10,
        'bpm': np.random.normal(120, 20, 50).tolist(),
        'key': [0, 1, 2, 3, 4] * 10
    }
    
    dummy_targets = {
        'genre': [0, 1, 1, 0, 2] * 10,
        'mood': [0, 0, 1, 1, 2] * 10,
        'bpm': np.random.normal(125, 15, 50).tolist(),
        'key': [0, 2, 1, 3, 5] * 10
    }
    
    class_names = {
        'genre': ['rock', 'pop', 'jazz'],
        'mood': ['happy', 'sad', 'energetic'],
        'key': ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    }
    
    print("\nVisualization functions ready for use with real data!")
    print("Example: plot_confusion_matrices(predictions, targets, class_names)")
