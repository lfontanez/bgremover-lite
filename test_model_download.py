#!/usr/bin/env python3
"""
Test script for U²-Net Model Download Module

This script tests the CMake model download functionality by:
1. Checking if the model files exist
2. Verifying file sizes
3. Testing the download module integration
"""

import os
import sys
import subprocess
import hashlib
from pathlib import Path

def run_cmake_test():
    """Run CMake to test model download functionality."""
    print("=== Testing U²-Net Model Download Module ===\n")
    
    # Create test build directory
    test_dir = Path("test_build")
    test_dir.mkdir(exist_ok=True)
    os.chdir(test_dir)
    
    # Run CMake with model download enabled
    print("🧪 Running CMake configuration...")
    result = subprocess.run([
        "cmake", "..",
        "-DU2NET_DOWNLOAD_MODELS=ON",
        "-DU2NET_CLEAN_CACHE=ON"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ CMake configuration failed:")
        print(result.stderr)
        return False
    
    print("✅ CMake configuration successful")
    print(result.stdout)
    
    # Check if models directory was created
    models_dir = Path("../models")
    if models_dir.exists():
        print(f"✅ Models directory created: {models_dir}")
        
        # Check for model files
        for model_name in ["u2net.onnx", "u2netp.onnx"]:
            model_file = models_dir / model_name
            if model_file.exists():
                size_mb = model_file.stat().st_size / (1024 * 1024)
                print(f"✅ {model_name}: {size_mb:.1f} MB")
            else:
                print(f"❌ {model_name}: Not found")
                return False
    else:
        print("❌ Models directory not created")
        return False
    
    return True

def check_manual_models():
    """Check if models exist for manual testing."""
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("📁 Models directory not found - creating...")
        models_dir.mkdir(exist_ok=True)
        print("💡 Please manually place u2net.onnx and u2netp.onnx in the models/ directory")
        return False
    
    print("=== Manual Model Check ===")
    all_found = True
    
    for model_name in ["u2net.onnx", "u2netp.onnx"]:
        model_file = models_dir / model_name
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"✅ {model_name}: {size_mb:.1f} MB")
            
            # Calculate SHA-256 for verification
            try:
                with open(model_file, 'rb') as f:
                    sha256_hash = hashlib.sha256(f.read()).hexdigest()
                print(f"   SHA-256: {sha256_hash[:16]}...")
            except Exception as e:
                print(f"   SHA-256 calculation failed: {e}")
        else:
            print(f"❌ {model_name}: Not found")
            all_found = False
    
    return all_found

def main():
    """Main test function."""
    print("U²-Net Model Download Module Test")
    print("=" * 50)
    
    # Check if CMake is available
    try:
        subprocess.run(["cmake", "--version"], capture_output=True, check=True)
        print("✅ CMake is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ CMake not found - please install CMake 3.16+")
        return 1
    
    # Check manual models first
    manual_models_ok = check_manual_models()
    
    if manual_models_ok:
        print("\n✅ Manual models are available - module integration should work")
    else:
        print("\n⚠️  Manual models not found - will test download functionality")
    
    # Test CMake integration
    if len(sys.argv) > 1 and sys.argv[1] == "--cmake-test":
        if run_cmake_test():
            print("\n🎉 CMake model download test passed!")
            return 0
        else:
            print("\n❌ CMake model download test failed")
            return 1
    
    print("\n💡 To test download functionality, run:")
    print("   python test_model_download.py --cmake-test")
    print("\n💡 To test manual models only, run:")
    print("   python test_model_download.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
