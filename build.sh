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
echo "The executable has been created in the build directory."
echo "You can run it with: ./bgremover"
echo ""
echo "Optional: Copy the executable to the root directory:"
echo "  cp build/bgremover ."
echo "  Then run: ./bgremover"