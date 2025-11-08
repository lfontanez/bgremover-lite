#!/bin/bash

# CUDA Environment Setup Script
# This script sets up the correct CUDA environment variables for the build

# Check if CUDA 12.8 exists
if [ -d "/usr/local/cuda-12.8" ]; then
    export CUDA_PATH="/usr/local/cuda-12.8"
    export CUDA_HOME="/usr/local/cuda-12.8"
    export PATH="/usr/local/cuda-12.8/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"
    echo "✅ Using CUDA 12.8 from /usr/local/cuda-12.8"
elif [ -d "/usr/local/cuda" ]; then
    # Fallback to /usr/local/cuda if it exists
    export CUDA_PATH="/usr/local/cuda"
    export CUDA_HOME="/usr/local/cuda"
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    echo "✅ Using CUDA from /usr/local/cuda"
else
    echo "❌ No suitable CUDA installation found"
    echo "Available CUDA installations:"
    ls -d /usr/local/cuda-* 2>/dev/null || echo "  None found"
    exit 1
fi

echo "CUDA_PATH: $CUDA_PATH"
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Verify NVCC can find CUDA headers
if [ -f "$CUDA_PATH/include/cuda_runtime.h" ]; then
    echo "✅ CUDA headers found at: $CUDA_PATH/include/cuda_runtime.h"
else
    echo "❌ CUDA headers not found at: $CUDA_PATH/include/cuda_runtime.h"
    echo "Looking for alternative locations..."
    find /usr /opt -name "cuda_runtime.h" 2>/dev/null
fi

echo ""
echo "CUDA environment is now configured. Run your build command with:"
echo "source ./setup_cuda_env.sh && ./build.sh"