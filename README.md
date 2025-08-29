# 🎵 Music Classification Model

**Author**: Sergie Code - Software Engineer & YouTube Programming Educator  
**Project**: AI Tools for Musicians  
**Date**: August 29, 2025  

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive PyTorch-based machine learning project for automatic music classification and tagging. This model can predict genre, mood, BPM, and musical key from audio features and spectrograms.

## 🎯 What This Project Does

This project implements deep learning models for **multi-task music classification**, capable of predicting:

- **🎼 Genre Classification**: Rock, Pop, Jazz, Classical, Electronic, etc.
- **😊 Mood Classification**: Happy, Sad, Energetic, Calm, etc.  
- **🥁 BPM Prediction**: Tempo estimation (beats per minute)
- **🎹 Key Detection**: Musical key prediction (C, C#, D, etc.)

### Key Features

- **Dual Input Architecture**: Combines traditional audio features (103 features) with mel spectrograms
- **Multi-Task Learning**: Trains on multiple classification/regression tasks simultaneously
- **Flexible Models**: Supports both CNN and RNN architectures
- **Integration Ready**: Designed to work seamlessly with preprocessing and API components
- **Production Ready**: Includes model export functionality for API deployment

## 🧠 How the Model Works

### Architecture Overview

The model uses a **dual-branch architecture** that processes two types of input:

1. **Spectrogram Branch (CNN)**:
   - Processes mel spectrograms (128 mel bins × variable time frames)
   - Uses convolutional layers to extract temporal and spectral patterns
   - Applies adaptive pooling to handle variable-length audio

2. **Feature Branch (MLP)**:
   - Processes 103 traditional audio features from preprocessing pipeline
   - Includes temporal, spectral, harmonic, rhythmic, and statistical features
   - Uses fully connected layers with dropout for regularization

3. **Fusion Layer**:
   - Combines features from both branches
   - Feeds into task-specific prediction heads

### Training Process

```
Audio Files → Preprocessing → Features + Spectrograms → Model Training → Trained Model
```

The model is trained using **multi-task learning** with weighted losses:
- Classification tasks use CrossEntropyLoss
- BPM regression uses MSELoss
- Task weights can be adjusted based on importance

## 🔗 Integration with Other Projects

This project is part of a three-repository pipeline:

### 1. **music-classification-preprocessing** (Input)
- Processes raw audio files
- Extracts 103 audio features per file
- Generates mel spectrograms
- Creates manifest files with metadata

### 2. **music-classification-model** (This Repository)
- Loads preprocessed data using manifest files
- Trains deep learning models
- Exports trained models for API use

### 3. **music-classification-api** (Output)
- Loads trained models from this repository
- Provides REST API for real-time classification
- Integrates preprocessing for live audio analysis

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (optional, but recommended)

### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd music-classification-model

# Install required packages
pip install -r requirements.txt
```

### Dependencies Include

- **PyTorch 2.0+**: Deep learning framework
- **librosa**: Audio processing
- **scikit-learn**: Machine learning utilities
- **matplotlib/seaborn**: Visualization
- **tensorboard**: Training monitoring
- **tqdm**: Progress bars

## 🚀 Quick Start

### 1. Run the Example Script

```bash
# Run complete example with dummy data
python examples/run_example.py
```

This will:
- Create a CNN model
- Train on dummy data for 5 epochs
- Run inference examples
- Export model for API use

### 2. Train with Real Data

First, ensure you have preprocessed data from the `music-classification-preprocessing` repository:

```bash
# Train model with real preprocessed data
python train.py \
    --data /path/to/dataset_manifest.json \
    --epochs 50 \
    --batch-size 32 \
    --output-dir ./results \
    --experiment-name my_music_model
```

### 3. Custom Training Configuration

```bash
# Train with custom settings
python train.py \
    --data /path/to/manifest.json \
    --model-type cnn \
    --num-genres 8 \
    --num-moods 5 \
    --epochs 100 \
    --learning-rate 0.0005 \
    --batch-size 64 \
    --use-spectrograms \
    --use-features
```

## 📊 Training and Testing

### Training Script Options

```bash
python train.py --help
```

**Key Parameters**:
- `--data`: Path to preprocessing manifest file
- `--model-type`: Choose 'cnn' or 'rnn'
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate for optimizer
- `--output-dir`: Directory for saved models and logs

### Example Training Commands

```bash
# Basic training
python train.py --data data/manifest.json --epochs 50

# Advanced training with GPU
python train.py \
    --data data/large_dataset_manifest.json \
    --model-type cnn \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --device cuda \
    --num-workers 8 \
    --output-dir ./experiments/experiment_1

# Resume training from checkpoint
python train.py \
    --data data/manifest.json \
    --resume ./experiments/experiment_1/models/checkpoint_epoch_25.pth \
    --epochs 50
```

### Model Evaluation

```python
# Load and evaluate trained model
import torch
from src.models import create_model
from src.utils import evaluate_model

# Load model
model = create_model("cnn")
checkpoint = torch.load("models/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate on test data
metrics = evaluate_model(model, test_loader, device, class_names)
print(f"Genre Accuracy: {metrics['genre_accuracy']:.3f}")
print(f"BPM MAE: {metrics['bpm_mae']:.2f}")
```

## 📂 Project Structure

```
music-classification-model/
├── src/
│   ├── models/
│   │   └── __init__.py          # Model architectures (CNN, RNN)
│   ├── data/
│   │   └── __init__.py          # Data loading and processing
│   ├── training/
│   │   └── __init__.py          # Training loops and utilities
│   └── utils/
│       └── __init__.py          # Evaluation and visualization
├── examples/
│   └── run_example.py           # Complete example script
├── data/                        # Data directory (preprocessed files)
├── models/                      # Saved model checkpoints
├── tests/                       # Unit tests
├── train.py                     # Main training script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🎛️ Model Configuration

### CNN Model (Default)

```python
# Spectrogram CNN branch
- Conv2d layers with BatchNorm and MaxPool
- Adaptive pooling for variable-length inputs
- Feature extraction to 256 dimensions

# Feature MLP branch  
- Fully connected layers with dropout
- Processes 103 audio features
- Output: 64 dimensions

# Fusion and prediction heads
- Combined features → 512 → 256 dimensions
- Task-specific heads for genre, mood, BPM, key
```

### RNN Model (Alternative)

```python
# LSTM for temporal modeling
- Bidirectional LSTM layers
- Processes feature sequences over time
- Suitable for time-series analysis
```

## 📈 Expected Data Format

### Input from Preprocessing Pipeline

The model expects data processed by the `music-classification-preprocessing` repository:

#### 1. Manifest File (`dataset_manifest.json`)
```json
{
  "dataset_name": "my_music_dataset",
  "total_files": 1000,
  "files": [
    {
      "filename": "song1.wav",
      "duration": 180.5,
      "features_file": "data/features/song1_features.json",
      "spectrogram_file": "data/spectrograms/song1_spectrogram.npy",
      "genre": "rock",
      "mood": "energetic", 
      "bpm": 120
    }
  ]
}
```

#### 2. Feature Files (`*_features.json`)
```json
{
  "tempo": 120.5,
  "spectral_centroid_mean": 2048.5,
  "mfcc_1_mean": -125.8,
  "chroma_1_mean": 0.25,
  // ... 103 total features
}
```

#### 3. Spectrogram Files (`*_spectrogram.npy`)
- NumPy arrays with shape `(128, time_frames)`
- Mel-scale spectrograms with 128 mel bins
- Variable time length based on audio duration

## 🔧 Model Export for API

### Export Trained Model

```python
from src.utils import save_model_for_api

# Define class mappings
class_mappings = {
    'genres': ['rock', 'pop', 'jazz', 'classical'],
    'moods': ['happy', 'sad', 'energetic', 'calm'],
    'keys': ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
}

# Export model for API
save_model_for_api(
    model=trained_model,
    class_mappings=class_mappings,
    save_path="models/api_model.pth"
)
```

### Exported Model Contents

The exported model includes:
- Model state dictionary
- Architecture information
- Class mappings for each task
- Preprocessing configuration
- Metadata for API integration

## 📊 Performance Monitoring

### Tensorboard Logging

```bash
# Start tensorboard to monitor training
tensorboard --logdir ./experiments/experiment_1/logs

# View in browser at http://localhost:6006
```

### Training Metrics

The training process logs:
- **Loss curves**: Training and validation loss for each task
- **Accuracy metrics**: Genre, mood, and key classification accuracy
- **Regression metrics**: BPM prediction MAE and MSE
- **Learning rate**: Adaptive learning rate changes

## 🔮 Inference Example

```python
import torch
from src.models import create_model

# Load trained model
model = create_model("cnn")
checkpoint = torch.load("models/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare input data (from preprocessing pipeline)
spectrogram = torch.randn(1, 1, 128, 300)  # Batch, channels, mels, time
features = torch.randn(1, 103)              # Batch, features

# Run inference
with torch.no_grad():
    predictions = model(spectrogram, features)

# Get predictions
genre_pred = torch.argmax(predictions['genre'], dim=1)
mood_pred = torch.argmax(predictions['mood'], dim=1)
bpm_pred = predictions['bpm']
key_pred = torch.argmax(predictions['key'], dim=1)

print(f"Predicted Genre: {genre_pred.item()}")
print(f"Predicted BPM: {bpm_pred.item():.1f}")
```

## 🧪 Testing

### Run Unit Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_models.py -v

# Run with coverage
python -m pytest tests/ --cov=src
```

### Test Model Training

```bash
# Quick training test with dummy data
python examples/run_example.py

# Verify model can overfit small dataset
python train.py --data examples/small_dataset.json --epochs 10 --batch-size 4
```

## 🎓 Educational Content

This project is part of **Sergie Code's YouTube channel** focusing on AI tools for musicians. The codebase is designed to be:

- **Educational**: Clear, well-commented code with explanations
- **Modular**: Easy to understand and modify individual components
- **Progressive**: Builds from simple concepts to advanced techniques
- **Practical**: Real-world applicable for music technology projects

### Learning Path

1. **Start with**: `examples/run_example.py` - Complete working example
2. **Understand**: `src/models/__init__.py` - Model architectures
3. **Explore**: `src/training/__init__.py` - Training procedures
4. **Experiment**: Modify hyperparameters and model architectures
5. **Deploy**: Export models for the API component

## 🤝 Contributing

This project welcomes contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with clear documentation
4. Add tests if applicable
5. Submit a pull request

### Code Style

- Follow PEP 8 formatting
- Add type hints where appropriate
- Include docstrings for all functions
- Comment complex algorithms

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Librosa**: Excellent audio processing library
- **PyTorch**: Powerful deep learning framework
- **Music Information Retrieval Community**: Research and datasets
- **YouTube Subscribers**: Feedback and feature requests

## 📞 Support

- **YouTube Channel**: [Sergie Code](https://youtube.com/@sergiecode)
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and ideas

---

**Happy coding and music making! 🎵**

Built with ❤️ by [Sergie Code](https://github.com/sergiecode) for the AI Tools for Musicians series.
