#!/bin/bash

# Background Remover Lite - Build Script
# This script builds the background remover project and downloads ONNX Runtime automatically

set -e  # Exit on any error

echo "=== Background Remover Lite Build Script ==="

# Check if OpenCV is installed
if ! pkg-config --exists opencv4; then
    echo "Error: OpenCV4 is not installed or not found in PKG_CONFIG_PATH"
    echo "Please install OpenCV4 development packages:"
    echo "  Ubuntu/Debian: sudo apt install libopencv-dev"
    echo "  CentOS/RHEL: sudo yum install opencv-devel"
    echo "  macOS: brew install opencv"
    exit 1
fi

# Check for CUDA
echo "Checking for CUDA..."
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits
    CUDA_AVAILABLE="true"
else
    echo "No NVIDIA GPU detected"
    CUDA_AVAILABLE="false"
fi

# Create and enter build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake ..

# Build the project
echo "Building the project..."
make -j$(nproc)

echo "=== Build Complete! ==="
echo "The executables have been created in the build directory."
echo ""
echo "Available executables:"
if [ -f "./bgremover" ]; then
    echo "  📱 CPU Version: ./bgremover"
    echo "  Usage: ./bgremover or ./bgremover <video_file>"
fi
if [ -f "./bgremover_gpu" ]; then
    echo "  🚀 GPU Version: ./bgremover_gpu"
    echo "  Usage: ./bgremover_gpu or ./bgremover_gpu <video_file>"
fi
echo ""
echo "Optional: Copy executables to the root directory:"
echo "  cp build/bgremover .           # CPU version"
if [ -f "./bgremover_gpu" ]; then
    echo "  cp build/bgremover_gpu .     # GPU version"
fi