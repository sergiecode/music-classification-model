# 🎵 Music Classification Model

**Author**: Sergie Code - Software Engineer & YouTube Programming Educator  
**Project**: AI Tools for Musicians  
**Date**: August 29, 2025  

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive PyTorch-based machine learning project for automatic music classification and tagging. This model can predict genre, mood, BPM, and musical key from audio features and spectrograms.

## 📋 Table of Contents

- [What This Project Does](#-what-this-project-does)
- [How the Model Works](#-how-the-model-works)
- [Integration with Other Projects](#-integration-with-other-projects)
- [🚀 API Integration Guide](#-api-integration-guide)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training and Testing](#-training-and-testing)
- [Project Structure](#-project-structure)
- [Model Configuration](#️-model-configuration)
- [Educational Content](#-educational-content)
- [Contributing](#-contributing)
- [Support](#-support)

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

## 🚀 API Integration Guide

This section provides everything you need to integrate the trained models with your **music-classification-api** project for real-time music classification.

### API Integration Overview

```
Audio File → API Endpoint → Preprocessing → Model Inference → JSON Response
```

The API integration involves:
1. **Model Export**: Export trained models from this repository
2. **Model Loading**: Load models in the API for inference
3. **Preprocessing Integration**: Use preprocessing pipeline for real-time audio
4. **Prediction Serving**: Serve predictions via REST API

### Step 1: Export Model for API

After training your model, export it for API use:

```python
from src.utils import save_model_for_api

# Define your class mappings
class_mappings = {
    'genre': {
        0: 'rock', 1: 'pop', 2: 'jazz', 3: 'classical', 
        4: 'electronic', 5: 'hip_hop', 6: 'country', 7: 'blues'
    },
    'mood': {
        0: 'happy', 1: 'sad', 2: 'energetic', 3: 'calm', 4: 'aggressive'
    },
    'key': {
        0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
        6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
    }
}

# Export model with class mappings
save_model_for_api(
    model=trained_model,
    class_mappings=class_mappings,
    save_path="models/api_model.pth"
)
```

### Step 2: API Model Loading Code

Use this code in your **music-classification-api** project:

```python
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import numpy as np

class MusicClassificationAPI:
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize the API with a trained model.
        
        Args:
            model_path: Path to the exported model (.pth file)
            device: Device to run inference on ("cpu" or "cuda")
        """
        self.device = torch.device(device)
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load the trained model and class mappings."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration
        config = checkpoint['model_config']
        self.class_mappings = checkpoint['class_mappings']
        
        # Recreate model architecture
        from models import create_model  # Your model creation function
        self.model = create_model(
            model_type=config['model_type'],
            num_genres=config['num_genres'],
            num_moods=config['num_moods'],
            num_keys=config['num_keys']
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, spectrogram: np.ndarray, features: np.ndarray) -> Dict[str, Any]:
        """
        Make predictions on audio data.
        
        Args:
            spectrogram: Mel spectrogram (shape: [n_mels, time_frames])
            features: Audio features (shape: [103])
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        # Convert to tensors
        spec_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0)  # [1, 1, n_mels, time]
        feat_tensor = torch.FloatTensor(features).unsqueeze(0)  # [1, 103]
        
        # Move to device
        spec_tensor = spec_tensor.to(self.device)
        feat_tensor = feat_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(spec_tensor, feat_tensor)
        
        # Process predictions
        predictions = {}
        
        # Genre prediction
        genre_probs = F.softmax(outputs['genre'], dim=1)
        genre_pred = torch.argmax(genre_probs, dim=1).item()
        predictions['genre'] = {
            'label': self.class_mappings['genre'][genre_pred],
            'confidence': genre_probs[0, genre_pred].item(),
            'all_probabilities': {
                self.class_mappings['genre'][i]: genre_probs[0, i].item()
                for i in range(len(self.class_mappings['genre']))
            }
        }
        
        # Mood prediction
        mood_probs = F.softmax(outputs['mood'], dim=1)
        mood_pred = torch.argmax(mood_probs, dim=1).item()
        predictions['mood'] = {
            'label': self.class_mappings['mood'][mood_pred],
            'confidence': mood_probs[0, mood_pred].item(),
            'all_probabilities': {
                self.class_mappings['mood'][i]: mood_probs[0, i].item()
                for i in range(len(self.class_mappings['mood']))
            }
        }
        
        # BPM prediction
        bpm_value = outputs['bpm'].item()
        predictions['bpm'] = {
            'value': round(bpm_value, 1),
            'range': self._get_bpm_range(bpm_value)
        }
        
        # Key prediction
        key_probs = F.softmax(outputs['key'], dim=1)
        key_pred = torch.argmax(key_probs, dim=1).item()
        predictions['key'] = {
            'label': self.class_mappings['key'][key_pred],
            'confidence': key_probs[0, key_pred].item(),
            'all_probabilities': {
                self.class_mappings['key'][i]: key_probs[0, i].item()
                for i in range(len(self.class_mappings['key']))
            }
        }
        
        return predictions
    
    def _get_bpm_range(self, bpm: float) -> str:
        """Categorize BPM into ranges."""
        if bpm < 60:
            return "very_slow"
        elif bpm < 90:
            return "slow"
        elif bpm < 120:
            return "moderate"
        elif bpm < 140:
            return "fast"
        else:
            return "very_fast"
```

### Step 3: FastAPI Integration Example

Create endpoints in your **music-classification-api**:

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import librosa
import numpy as np
from typing import Optional

app = FastAPI(title="Music Classification API")

# Initialize model
classifier = MusicClassificationAPI("models/api_model.pth")

@app.post("/classify/")
async def classify_music(
    file: UploadFile = File(...),
    include_probabilities: bool = False
):
    """
    Classify an uploaded audio file.
    
    Args:
        file: Audio file (MP3, WAV, etc.)
        include_probabilities: Whether to include all class probabilities
        
    Returns:
        JSON with classification results
    """
    try:
        # Read audio file
        audio_data = await file.read()
        
        # Process with librosa (integrate with your preprocessing)
        y, sr = librosa.load(io.BytesIO(audio_data), sr=22050)
        
        # Extract features (use your preprocessing pipeline)
        features = extract_features(y, sr)  # Your 103 features
        spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, fmax=8000
        )
        
        # Make prediction
        predictions = classifier.predict(spectrogram, features)
        
        # Format response
        response = {
            "filename": file.filename,
            "predictions": {
                "genre": {
                    "label": predictions['genre']['label'],
                    "confidence": predictions['genre']['confidence']
                },
                "mood": {
                    "label": predictions['mood']['label'], 
                    "confidence": predictions['mood']['confidence']
                },
                "bpm": predictions['bpm'],
                "key": {
                    "label": predictions['key']['label'],
                    "confidence": predictions['key']['confidence']
                }
            }
        }
        
        # Add probabilities if requested
        if include_probabilities:
            response["predictions"]["genre"]["all_probabilities"] = predictions['genre']['all_probabilities']
            response["predictions"]["mood"]["all_probabilities"] = predictions['mood']['all_probabilities']
            response["predictions"]["key"]["all_probabilities"] = predictions['key']['all_probabilities']
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/health/")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": classifier.model is not None}

@app.get("/info/")
async def model_info():
    """Get model information."""
    return {
        "model_type": "CNN",
        "supported_tasks": ["genre", "mood", "bpm", "key"],
        "genres": list(classifier.class_mappings['genre'].values()),
        "moods": list(classifier.class_mappings['mood'].values()),
        "keys": list(classifier.class_mappings['key'].values())
    }
```

### Step 4: API Requirements

Add these to your **music-classification-api** `requirements.txt`:

```txt
fastapi>=0.68.0
uvicorn[standard]>=0.15.0
torch>=2.0.0
torchaudio>=2.0.0
librosa>=0.9.0
numpy>=1.21.0
python-multipart>=0.0.5
```

### Step 5: Running the API

```bash
# In your music-classification-api directory
pip install -r requirements.txt

# Copy the exported model
cp ../music-classification-model/models/api_model.pth ./models/

# Run the API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 6: API Usage Examples

```bash
# Test the API
curl -X POST "http://localhost:8000/classify/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_song.mp3"

# With probabilities
curl -X POST "http://localhost:8000/classify/?include_probabilities=true" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_song.mp3"

# Health check
curl -X GET "http://localhost:8000/health/"
```

### Expected API Response

```json
{
    "filename": "test_song.mp3",
    "predictions": {
        "genre": {
            "label": "rock",
            "confidence": 0.87
        },
        "mood": {
            "label": "energetic",
            "confidence": 0.92
        },
        "bpm": {
            "value": 128.5,
            "range": "fast"
        },
        "key": {
            "label": "A",
            "confidence": 0.73
        }
    }
}
```

### Integration with Preprocessing Pipeline

Your API should integrate the preprocessing components:

```python
# In your API project, create a preprocessing module
from music_classification_preprocessing import AudioProcessor

class APIPreprocessor:
    def __init__(self):
        self.processor = AudioProcessor()
    
    def process_audio(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Process audio and extract features + spectrogram."""
        features = self.processor.extract_features(audio_path)
        spectrogram = self.processor.extract_spectrogram(audio_path)
        
        return spectrogram, features
```

### Deployment Considerations

1. **Model Size**: Exported models are typically 10-50MB
2. **Memory Usage**: ~500MB RAM for inference
3. **Performance**: CPU inference ~100ms, GPU ~20ms per file
4. **Scaling**: Use multiple workers for concurrent requests
5. **Caching**: Cache preprocessing results for repeated files

### Error Handling

Implement robust error handling in your API:

```python
class ClassificationError(Exception):
    pass

class ModelLoadError(Exception):
    pass

class PreprocessingError(Exception):
    pass

### Error Handling

Implement robust error handling in your API:

```python
class ClassificationError(Exception):
    pass

class ModelLoadError(Exception):
    pass

class PreprocessingError(Exception):
    pass

# Handle different error types with appropriate HTTP status codes
```

### Recommended API Project Structure

```
music-classification-api/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py        # MusicClassificationAPI class
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── processor.py         # Audio preprocessing
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── classification.py    # Classification endpoints
│   │   └── health.py           # Health check endpoints
│   └── utils/
│       ├── __init__.py
│       └── helpers.py          # Utility functions
├── models/
│   └── api_model.pth           # Exported model from this repo
├── tests/
│   ├── test_classification.py
│   └── test_endpoints.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── .env
└── README.md
```

### Docker Deployment

Create a `Dockerfile` for your API:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy model from this repository
COPY models/api_model.pth ./models/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Performance Optimization Tips

1. **Model Optimization**:
   ```python
   # Use TorchScript for faster inference
   traced_model = torch.jit.trace(model, (sample_spec, sample_features))
   torch.jit.save(traced_model, "models/traced_model.pt")
   ```

2. **Batch Processing**:
   ```python
   # Process multiple files in batches for better throughput
   def batch_classify(self, files_batch: List[UploadFile]):
       # Implementation for batch processing
   ```

3. **Async Processing**:
   ```python
   import asyncio
   
   async def async_classify(self, audio_data: bytes):
       # Non-blocking classification
   ```

### Testing Your API Integration

Create comprehensive tests for your API:

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_classify_endpoint():
    with open("test_audio.mp3", "rb") as f:
        response = client.post(
            "/classify/",
            files={"file": ("test.mp3", f, "audio/mpeg")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "genre" in data["predictions"]
    assert "confidence" in data["predictions"]["genre"]

def test_health_endpoint():
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

### API Documentation

Your API will automatically generate documentation at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Monitoring and Logging

Add monitoring to your API:

```python
import logging
from prometheus_client import Counter, Histogram

# Metrics
REQUESTS_TOTAL = Counter('requests_total', 'Total requests')
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

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

## 📚 Quick Reference for API Integration

### Essential Commands for API Setup

```bash
# 1. Export model after training
python -c "
from src.utils import save_model_for_api
from src.models import create_model
import torch

# Load your trained model
model = create_model('cnn', num_genres=8, num_moods=5, num_keys=12)
checkpoint = torch.load('models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Define class mappings
class_mappings = {
    'genre': {0: 'rock', 1: 'pop', 2: 'jazz', 3: 'classical', 4: 'electronic', 5: 'hip_hop', 6: 'country', 7: 'blues'},
    'mood': {0: 'happy', 1: 'sad', 2: 'energetic', 3: 'calm', 4: 'aggressive'},
    'key': {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}
}

# Export for API
save_model_for_api(model, class_mappings, 'models/api_model.pth')
print('Model exported successfully!')
"

# 2. Copy model to your API project
cp models/api_model.pth ../music-classification-api/models/

# 3. Setup API project
cd ../music-classification-api
pip install fastapi uvicorn torch librosa
uvicorn main:app --reload
```

### Key Integration Points

| Component | This Repository | Your API Repository |
|-----------|----------------|-------------------|
| **Model Export** | `src/utils.save_model_for_api()` | Load with `torch.load()` |
| **Preprocessing** | Uses manifest files | Integrate `librosa` + your preprocessing |
| **Model Loading** | N/A | `MusicClassificationAPI` class |
| **Inference** | Training pipeline | REST API endpoints |
| **Class Mappings** | Exported with model | Used for human-readable outputs |

### Expected File Flow

```
Training (This Repo) → API Deployment
├── models/best_model.pth → models/api_model.pth
├── Class mappings → JSON responses
├── Model architecture → Inference code
└── Preprocessing logic → Real-time processing
```

### API Endpoints to Implement

```python
POST   /classify/              # Main classification endpoint
GET    /health/                # Health check
GET    /info/                  # Model information
POST   /classify/batch/        # Batch processing (optional)
GET    /metrics/              # Performance metrics (optional)
```

### Common Integration Issues & Solutions

1. **Model Loading Errors**:
   ```python
   # Ensure model architecture matches
   checkpoint = torch.load(model_path, map_location='cpu')
   config = checkpoint['model_config']  # Use saved config
   ```

2. **Feature Dimension Mismatch**:
   ```python
   # Always use 103 features as expected by model
   assert features.shape[-1] == 103, f"Expected 103 features, got {features.shape[-1]}"
   ```

3. **Class Mapping Issues**:
   ```python
   # Use saved class mappings from checkpoint
   class_mappings = checkpoint['class_mappings']
   ```

4. **Performance Optimization**:
   ```python
   # Use torch.jit for faster inference
   model = torch.jit.trace(model, (sample_spec, sample_features))
   ```

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

## 🙏 Acknowledgments

- **Librosa**: Excellent audio processing library
- **PyTorch**: Powerful deep learning framework
- **Music Information Retrieval Community**: Research and datasets
- **YouTube Subscribers**: Feedback and feature requests

## 📞 Support

- 📸 Instagram: https://www.instagram.com/sergiecode

- 🧑🏼‍💻 LinkedIn: https://www.linkedin.com/in/sergiecode/

- 📽️Youtube: https://www.youtube.com/@SergieCode

- 😺 Github: https://github.com/sergiecode

- 👤 Facebook: https://www.facebook.com/sergiecodeok

- 🎞️ Tiktok: https://www.tiktok.com/@sergiecode

- 🕊️Twitter: https://twitter.com/sergiecode

- 🧵Threads: https://www.threads.net/@sergiecode

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and ideas

---

**Happy coding and music making! 🎵**

Built with ❤️ by [Sergie Code](https://github.com/sergiecode) for the AI Tools for Musicians series.
