#!/bin/bash

# Background Remover Lite - Enhanced Build Script
# This script builds the background remover project with comprehensive CUDA detection and configuration

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo ""
    echo "=== $1 ==="
}

# Global variables
CUDA_AVAILABLE=false
NVCC_AVAILABLE=false
OPENCV_CUDA_SUPPORT=false
PYTHON_OPENCV_ISSUES=false
CUDA_ARCH_BIN=""
GPU_NAME=""

# CUDA Architecture mapping
get_cuda_arch_bin() {
    local gpu_name="$1"
    local gpu_name_upper=$(echo "$gpu_name" | tr '[:lower:]' '[:upper:]')
    
    # Architecture mapping based on GPU name
    case "$gpu_name_upper" in
        *"RTX 4090"*) echo "8.9";;
        *"RTX 4080"*) echo "8.9";;
        *"RTX 4070"*) echo "8.9";;
        *"RTX 3090"*) echo "8.6";;
        *"RTX 3080"*) echo "8.6";;
        *"RTX 3070"*) echo "8.6";;
        *"RTX 3060"*) echo "8.6";;
        *"RTX 2080 TI"*) echo "7.5";;
        *"RTX 2080"*) echo "7.5";;
        *"RTX 2070"*) echo "7.5";;
        *"RTX 2060"*) echo "7.5";;
        *"GTX 1080 TI"*) echo "6.1";;
        *"GTX 1080"*) echo "6.1";;
        *"GTX 1070"*) echo "6.1";;
        *"GTX 1060"*) echo "6.1";;
        *"GTX 980 TI"*) echo "5.2";;
        *"GTX 980"*) echo "5.2";;
        *"GTX 970"*) echo "5.2";;
        *"Tesla V100"*) echo "7.0";;
        *"Tesla T4"*) echo "7.5";;
        *"Quadro RTX 4000"*) echo "7.5";;
        *"Quadro RTX 3000"*) echo "7.5";;
        *)
            log_warning "Unknown GPU architecture: $gpu_name"
            echo "6.1"  # Default to Pascal
            ;;
    esac
}

# Check NVIDIA driver and GPU
check_nvidia_environment() {
    log_header "NVIDIA Environment Detection"
    
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log_error "nvidia-smi not found. NVIDIA drivers may not be installed."
        log_info "Install NVIDIA drivers for your GPU:"
        echo "  Ubuntu/Debian: sudo apt install nvidia-driver-470"
        echo "  CentOS/RHEL: sudo yum install nvidia-driver"
        echo "  Or visit: https://www.nvidia.com/drivers"
        return 1
    fi
    
    log_success "nvidia-smi found"
    
    # Get GPU name
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n1)
    if [[ -z "$GPU_NAME" ]]; then
        log_error "Failed to detect GPU name"
        return 1
    fi
    
    log_info "Detected GPU: $GPU_NAME"
    CUDA_AVAILABLE=true
    return 0
}

# Check NVCC availability
check_nvcc() {
    log_header "CUDA Toolkit Detection"
    
    if ! command -v nvcc >/dev/null 2>&1; then
        log_warning "nvcc (CUDA compiler) not found"
        log_info "Install CUDA Toolkit:"
        echo "  Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit"
        echo "  Or download from: https://developer.nvidia.com/cuda-downloads"
        return 1
    fi
    
    local nvcc_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    log_success "NVCC found (version $nvcc_version)"
    NVCC_AVAILABLE=true
    return 0
}

# Check OpenCV CUDA support
check_opencv_cuda() {
    log_header "OpenCV CUDA Support Detection"
    
    if ! pkg-config --exists opencv4; then
        log_error "OpenCV4 is not installed or not found in PKG_CONFIG_PATH"
        return 1
    fi
    
    # Get OpenCV version
    local opencv_version=$(pkg-config --modversion opencv4)
    log_info "OpenCV version: $opencv_version"
    
    # Check if OpenCV has CUDA modules
    if python3 -c "
import sys
try:
    import cv2
    cuda_count = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, 'cuda') else 0
    print(cuda_count > 0)
except:
    print(False)
" 2>/dev/null | grep -q "True"; then
        log_success "OpenCV CUDA support detected!"
        OPENCV_CUDA_SUPPORT=true
        return 0
    else
        log_warning "OpenCV does not have CUDA support"
        log_info "To enable CUDA support, rebuild OpenCV:"
        echo "  Ubuntu/Debian:"
        echo "    sudo apt remove libopencv-dev python3-opencv"
        echo "    sudo apt install libopencv-dev python3-opencv"
        echo "    # Then rebuild from source with CUDA flags"
        echo ""
        echo "  Or install pre-built OpenCV with CUDA:"
        echo "    pip install opencv-contrib-python"
        return 1
    fi
}

# Check Python environment for OpenCV shadowing
check_python_opencv_shadowing() {
    log_header "Python OpenCV Environment Check"
    
    if ! command -v python3 >/dev/null 2>&1; then
        log_warning "Python3 not found"
        return 0
    fi
    
    # Get Python OpenCV info
    local python_opencv_path=$(python3 -c "import cv2; print(cv2.__file__)" 2>/dev/null || echo "")
    local python_opencv_version=$(python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "Not found")
    
    # Get system OpenCV info
    local system_opencv_version=$(pkg-config --modversion opencv4 2>/dev/null || echo "Not found")
    
    log_info "Python OpenCV path: $python_opencv_path"
    log_info "Python OpenCV version: $python_opencv_version"
    log_info "System OpenCV version: $system_opencv_version"
    
    # Check for shadowing
    if [[ "$python_opencv_path" == *"site-packages"* ]] && [[ "$system_opencv_version" != "Not found" ]]; then
        if [[ "$python_opencv_version" != "$system_opencv_version" ]]; then
            log_warning "Potential OpenCV version mismatch detected!"
            log_warning "System OpenCV ($system_opencv_version) vs Python OpenCV ($python_opencv_version)"
            log_info "This may cause module shadowing issues."
            log_info "Solutions:"
            echo "  1. Use system OpenCV: pip uninstall opencv-python opencv-contrib-python"
            echo "  2. Use same version: Install matching version via pip"
            echo "  3. Use virtual environment"
            PYTHON_OPENCV_ISSUES=true
        fi
    fi
    
    return 0
}

# Main build function
main() {
    log_header "Background Remover Lite - Enhanced Build Script"
    
    # Environment checks
    check_nvidia_environment
    check_nvcc
    check_opencv_cuda
    check_python_opencv_shadowing
    
    # Create build directory
    log_header "Build Configuration"
    mkdir -p build
    cd build
    
    # Determine CUDA architecture
    if [[ "$CUDA_AVAILABLE" == "true" ]] && [[ -n "$GPU_NAME" ]]; then
        CUDA_ARCH_BIN=$(get_cuda_arch_bin "$GPU_NAME")
        log_info "CUDA architecture set to: $CUDA_ARCH_BIN"
    else
        log_warning "CUDA not available, building CPU-only version"
    fi
    
    # Configure CMake
    log_info "Configuring with CMake..."
    
    # Set CMake flags based on detection
    CMAKE_FLAGS=""
    
    if [[ "$OPENCV_CUDA_SUPPORT" == "true" ]]; then
        CMAKE_FLAGS="-DWITH_CUDA=ON -DCUDA_ARCH_BIN=$CUDA_ARCH_BIN"
        log_info "CUDA support enabled in CMake"
    else
        CMAKE_FLAGS="-DWITH_CUDA=OFF"
        log_info "CUDA support disabled"
    fi
    
    # Run CMake
    if cmake $CMAKE_FLAGS ..; then
        log_success "CMake configuration successful"
    else
        log_error "CMake configuration failed"
        log_info "Troubleshooting steps:"
        echo "  1. Ensure all dependencies are installed"
        echo "  2. Check CUDA toolkit installation"
        echo "  3. Verify OpenCV CUDA support"
        echo "  4. Review CMake output above for specific errors"
        exit 1
    fi
    
    # Build the project
    log_info "Building the project..."
    if make -j$(nproc); then
        log_success "Build successful"
    else
        log_error "Build failed"
        exit 1
    fi
    
    # Results
    log_header "Build Results"
    
    echo "Available executables:"
    if [[ -f "./bgremover" ]]; then
        log_success "  📱 CPU Version: ./bgremover"
        echo "     Usage: ./bgremover or ./bgremover <video_file>"
    fi
    if [[ -f "./bgremover_gpu" ]]; then
        log_success "  🚀 GPU Version: ./bgremover_gpu"
        echo "     Usage: ./bgremover_gpu or ./bgremover_gpu <video_file>"
    fi
    
    echo ""
    echo "Optional: Copy executables to the root directory:"
    echo "  cp build/bgremover .           # CPU version"
    if [[ -f "./bgremover_gpu" ]]; then
        echo "  cp build/bgremover_gpu .     # GPU version"
    fi
    
    # Final status
    log_header "Environment Summary"
    echo "CUDA Available: $(if [[ "$CUDA_AVAILABLE" == "true" ]]; then echo "✅"; else echo "❌"; fi)"
    echo "NVCC Available: $(if [[ "$NVCC_AVAILABLE" == "true" ]]; then echo "✅"; else echo "❌"; fi)"
    echo "OpenCV CUDA: $(if [[ "$OPENCV_CUDA_SUPPORT" == "true" ]]; then echo "✅"; else echo "❌"; fi)"
    echo "Python Issues: $(if [[ "$PYTHON_OPENCV_ISSUES" == "true" ]]; then echo "⚠️"; else echo "✅"; fi)"
    
    if [[ "$CUDA_AVAILABLE" == "true" ]] && [[ "$OPENCV_CUDA_SUPPORT" == "true" ]]; then
        echo ""
        log_success "🎉 Ready for GPU-accelerated background blur!"
    else
        echo ""
        log_warning "⚠️  Running in CPU-only mode. For GPU acceleration:"
        echo "  1. Install NVIDIA drivers"
        echo "  2. Install CUDA Toolkit"
        echo "  3. Rebuild OpenCV with CUDA support"
    fi
}

# Run main function
main "$@"
