#!/usr/bin/env python3
"""
Quick verification script for the music classification model project.
"""

import sys
import os

def main():
    print("🎵 Music Classification Model - Quick Verification")
    print("=" * 50)
    
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    # Test PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    # Test other key dependencies
    dependencies = [
        ('numpy', 'NumPy'),
        ('librosa', 'Librosa'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'TQDM')
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} not found")
    
    # Test our model imports
    print("\nTesting project modules...")
    try:
        sys.path.insert(0, 'src')
        from models import create_model
        print("✅ Models module imported")
        
        # Test model creation
        model = create_model("cnn", num_genres=5, num_moods=3, num_keys=12)
        print("✅ CNN model created successfully")
        
        # Test model forward pass
        dummy_spec = torch.randn(1, 1, 128, 200)
        dummy_features = torch.randn(1, 103)
        
        with torch.no_grad():
            predictions = model(dummy_spec, dummy_features)
        
        print("✅ Model forward pass successful")
        print(f"Output shapes: genre={predictions['genre'].shape}, bpm={predictions['bpm'].shape}")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False
    
    print("\n🎉 All verification tests passed!")
    print("🚀 Project is ready for use!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
