#!/bin/bash

# BGRemover Lite - System Check Script
# Analyzes your system and provides build recommendations

echo "🔍 BGRemover Lite - System Analysis"
echo "=================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

check_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
        return 0
    else
        echo -e "${RED}❌ $3${NC}"
        return 1
    fi
}

warning_result() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

info_result() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check CPU
echo "🖥️  CPU Analysis"
cpu_info=$(lscpu | grep "Model name" | sed 's/Model name:\s*//')
info_result "CPU: $cpu_info"

cpu_cores=$(nproc)
info_result "CPU Cores: $cpu_cores"

if [ $cpu_cores -ge 4 ]; then
    check_result 0 "Sufficient CPU cores" "Insufficient CPU cores"
else
    check_result 1 "Sufficient CPU cores" "Insufficient CPU cores (4+ recommended)"
fi
echo ""

# Check Memory
echo "💾 Memory Analysis"
total_mem=$(free -g | awk '/^Mem:/{print $2}')
info_result "Total RAM: ${total_mem}GB"

if [ $total_mem -ge 8 ]; then
    check_result 0 "Sufficient RAM (8GB+)" "Insufficient RAM"
elif [ $total_mem -ge 4 ]; then
    warning_result "Moderate RAM (4GB) - GPU version recommended"
else
    check_result 1 "Sufficient RAM (8GB+)" "Insufficient RAM (4GB+ needed)"
fi
echo ""

# Check GPU
echo "🎮 GPU Analysis"
gpu_found=false

if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_found=true
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n1)
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)
    gpu_memory_gb=$((gpu_memory / 1024))
    
    info_result "GPU: $gpu_name"
    info_result "GPU Memory: ${gpu_memory_gb}GB"
    
    if [ $gpu_memory_gb -ge 12 ]; then
        check_result 0 "Excellent GPU (12GB+ VRAM)" "Insufficient GPU memory"
        gpu_recommendation="🟢 EXCELLENT - Perfect for 1080p real-time processing"
    elif [ $gpu_memory_gb -ge 6 ]; then
        check_result 0 "Good GPU (6GB+ VRAM)" "Insufficient GPU memory"
        gpu_recommendation="🟡 GOOD - Suitable for 1080p processing"
    else
        check_result 0 "Moderate GPU (<6GB VRAM)" "Insufficient GPU memory"
        gpu_recommendation="🟠 MODERATE - May struggle with 1080p, try 720p"
    fi
else
    check_result 1 "NVIDIA GPU detected" "No NVIDIA GPU found"
    gpu_recommendation="🔴 NO GPU - CPU version only"
fi
echo ""

# Check CUDA
echo "🚀 CUDA Analysis"
if command -v nvcc >/dev/null 2>&1; then
    cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    info_result "CUDA Version: $cuda_version"
    
    if [[ $cuda_version == "12."* ]] || [[ $cuda_version == "11."* ]]; then
        check_result 0 "CUDA version compatible" "CUDA version too old"
        cuda_recommendation="✅ CUDA $cuda_version - Ready for GPU build"
    else
        warning_result "CUDA version $cuda_version - May have compatibility issues"
        cuda_recommendation="⚠️  CUDA $cuda_version - Consider updating to 11.0+"
    fi
else
    check_result 1 "CUDA toolkit found" "CUDA toolkit not found"
    cuda_recommendation="❌ No CUDA - GPU build not possible"
fi
echo ""

# Check Conda
echo "🐍 Conda Environment"
if command -v conda >/dev/null 2>&1; then
    check_result 0 "Conda available" "Conda not found"
    
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        info_result "Current environment: $CONDA_DEFAULT_ENV"
    else
        info_result "Not in a Conda environment"
        warning_result "Consider using Conda for easier dependency management"
    fi
else
    check_result 1 "Conda available" "Conda not found"
    info_result "Conda not required - system Python is fine"
fi
echo ""

# Check build tools
echo "🔧 Build Tools"
if command -v cmake >/dev/null 2>&1; then
    cmake_version=$(cmake --version | head -n1 | awk '{print $3}')
    check_result 0 "CMake $cmake_version found" "CMake not found"
else
    check_result 1 "CMake found" "CMake not found"
fi

if command -v make >/dev/null 2>&1; then
    check_result 0 "Make found" "Make not found"
else
    check_result 1 "Make found" "Make not found"
fi
echo ""

# Final recommendation
echo "🎯 BUILD RECOMMENDATION"
echo "======================"
echo ""

if [ "$gpu_found" = true ] && command -v nvcc >/dev/null 2>&1; then
    echo -e "${GREEN}🎮 GPU BUILD RECOMMENDED${NC}"
    echo "• Your system supports GPU acceleration"
    echo "• $gpu_recommendation"
    echo "• $cuda_recommendation"
    echo ""
    echo "📋 Recommended Build Steps:"
    echo "1. conda create -n opencv_cuda12 python=3.11 opencv cudatoolkit=12.1"
    echo "2. conda activate opencv_cuda12"
    echo "3. ./build.sh"
    echo ""
    echo "📱 Expected Results:"
    echo "• CPU version: ./build/bgremover (1-5 FPS)"
    echo "• GPU version: ./build/bgremover_gpu (25-30 FPS at 1080p)"
    echo ""
    echo "🎯 Use GPU version for production, CPU for testing"
else
    echo -e "${YELLOW}📱 CPU BUILD RECOMMENDED${NC}"
    echo "• No NVIDIA GPU or CUDA detected"
    echo "• CPU version will be built automatically"
    echo "• Performance: 1-5 FPS (suitable for testing)"
    echo ""
    echo "📋 Recommended Build Steps:"
    echo "1. ./build.sh"
    echo ""
    echo "📱 Expected Results:"
    echo "• CPU version: ./build/bgremover (1-5 FPS)"
    echo "• GPU version: Not available (no GPU)"
    echo ""
    echo "💡 For better performance, consider a system with NVIDIA GPU"
fi

echo ""
echo "🔍 Quick Build Test:"
echo "./build.sh && ./build/bgremover --help"

echo ""
echo "📚 For detailed instructions, see README.md or BUILD.md"