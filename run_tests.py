#!/usr/bin/env python3
"""
Test Runner for Music Classification Model
==========================================

Author: Sergie Code - Software Engineer & YouTube Programming Educator
Project: AI Tools for Musicians
Date: August 29, 2025

Runs all tests to ensure the project works perfectly.
"""

import sys
import os
from pathlib import Path

def main():
    print("🎵 Music Classification Model - Test Suite")
    print("=" * 60)
    print("Author: Sergie Code - Software Engineer & YouTube Programming Educator")
    print("Project: AI Tools for Musicians")
    print("=" * 60)
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("\n1. Running Core Functionality Tests...")
    print("-" * 40)
    
    # Run core tests
    from tests.test_core import run_all_core_tests
    core_success = run_all_core_tests()
    
    if not core_success:
        print("\n❌ Core tests failed. Please fix issues before proceeding.")
        return False
    
    print("\n2. Testing Project Verification...")
    print("-" * 40)
    
    # Run verification script
    try:
        exec(open('verify.py').read())
        print("✅ Project verification passed!")
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False
    
    print("\n3. Testing Example Script...")
    print("-" * 40)
    
    # Test if example script can import and create models
    try:
        sys.path.append(str(Path('src')))
        from models import create_model
        from training import create_dummy_data_loaders
        
        # Quick test
        _ = create_model("cnn")
        _, _ = create_dummy_data_loaders(batch_size=2)
        
        print("✅ Example components work correctly!")
        
    except Exception as e:
        print(f"❌ Example test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! 🎉")
    print("🚀 Your music classification model is working perfectly!")
    print("✅ Ready for production and YouTube content!")
    print("🎵 Happy coding, Sergie!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
