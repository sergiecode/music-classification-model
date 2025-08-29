"""
Music Classification Model Package
==================================

Author: Sergie Code - Software Engineer & YouTube Programming Educator
Project: AI Tools for Musicians
Date: August 29, 2025

This package provides PyTorch-based models for automatic music classification
including genre, mood, BPM, and key prediction.
"""

__version__ = "1.0.0"
__author__ = "Sergie Code"
__email__ = "sergiecode@example.com"

# Import main components for easy access
from .models import create_model, MusicClassificationModel
from .data import MusicDataset, create_data_loaders
from .training import MusicClassificationTrainer

__all__ = [
    'create_model',
    'MusicClassificationModel', 
    'MusicDataset',
    'create_data_loaders',
    'MusicClassificationTrainer'
]
