#!/usr/bin/env python3
"""
Test Runner for Music Classification Model
==========================================

Author: Sergie Code - Software Engineer & YouTube Programming Educator
Project: AI Tools for Musicians
Date: August 29, 2025

Simple test runner to verify the application works perfectly.
"""

import sys
import subprocess
from pathlib import Path

def run_core_tests():
    """Run core functionality tests."""
    print("Running Core Functionality Tests...")
    print("=" * 50)
    
    # Run the core tests
    result = subprocess.run([
        sys.executable, 
        str(Path(__file__).parent / "test_core.py")
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Core tests passed!")
        print(result.stdout)
        return True
    else:
        print("Core tests failed!")
        print(result.stdout)
        print(result.stderr)
        return False

def run_model_tests():
    """Run model architecture tests."""
    print("\nRunning Model Architecture Tests...")
    print("=" * 50)
    
    # Run pytest on model tests
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        str(Path(__file__).parent / "test_models.py"), 
        "-v"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Model tests passed!")
        return True
    else:
        print("Model tests failed!")
        print(result.stdout)
        print(result.stderr)
        return False

def check_project_structure():
    """Check that all essential files exist."""
    print("\nChecking Project Structure...")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent
    
    essential_files = [
        "src/models/__init__.py",
        "src/data/__init__.py", 
        "src/training/__init__.py",
        "src/utils/__init__.py",
        "examples/run_example.py",
        "requirements.txt",
        "README.md",
        "config/training_config.yaml"
    ]
    
    all_exist = True
    for file_path in essential_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"[OK] {file_path}")
        else:
            print(f"[MISSING] {file_path}")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests and checks."""
    print("Music Classification Model - Test Suite")
    print("=" * 60)
    print("Author: Sergie Code - Software Engineer & YouTube Programming Educator")
    print("Project: AI Tools for Musicians")
    print("=" * 60)
    
    success = True
    
    # Check project structure
    if not check_project_structure():
        print("\nProject structure check failed!")
        success = False
    
    # Run core tests
    if not run_core_tests():
        print("\nCore functionality tests failed!")
        success = False
    
    # Run model tests
    if not run_model_tests():
        print("\nModel architecture tests failed!")
        success = False
    
    # Final result
    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED! The app works perfect!")
        print("[OK] Your music classification model is ready for YouTube!")
        print("[OK] All components are working correctly")
        print("[OK] Training pipeline is functional")
        print("[OK] Models can be exported for API use")
        print("[OK] Integration with preprocessing pipeline is verified")
    else:
        print("Some tests failed. Please check the output above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
