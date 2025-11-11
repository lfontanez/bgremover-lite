#!/bin/bash

# Enhanced Build Script for BGRemover Lite
# Supports GPU acceleration and automatic dependency resolution

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
WHITE='\033[1;37m'
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

# Enhanced CUDA Architecture mapping for modern GPUs
get_cuda_arch_bin() {
    local gpu_name="$1"
    local gpu_name_upper=$(echo "$gpu_name" | tr '[:lower:]' '[:upper:]')
    
    # Architecture mapping based on GPU name - comprehensive coverage
    case "$gpu_name_upper" in
        # RTX 40 Series (Ada Lovelace)
        *"RTX 4090"*) echo "8.9";;
        *"RTX 4080 SUPER"*) echo "8.9";;
        *"RTX 4080"*) echo "8.9";;
        *"RTX 4070 TI SUPER"*) echo "8.9";;
        *"4070 TI SUPER"*|"4070 Ti SUPER"*) echo "8.9";;
        *"RTX 4070 TI"*) echo "8.9";;
        *"RTX 4070 SUPER"*) echo "8.9";;
        *"RTX 4070"*) echo "8.9";;
        *"RTX 4060 TI"*) echo "8.9";;
        *"RTX 4060"*) echo "8.9";;
        *"RTX 4050"*) echo "8.9";;
        
        # RTX 30 Series (Ampere)
        *"RTX 3090 TI"*) echo "8.6";;
        *"RTX 3090"*) echo "8.6";;
        *"RTX 3080 TI"*) echo "8.6";;
        *"RTX 3080"*) echo "8.6";;
        *"RTX 3070 TI"*) echo "8.6";;
        *"RTX 3070"*) echo "8.6";;
        *"RTX 3060 TI"*) echo "8.6";;
        *"RTX 3060"*) echo "8.6";;
        *"RTX 3050"*) echo "8.6";;
        
        # RTX 20 Series (Turing)
        *"RTX 2080 TI"*) echo "7.5";;
        *"RTX 2080 SUPER"*) echo "7.5";;
        *"RTX 2080"*) echo "7.5";;
        *"RTX 2070 SUPER"*) echo "7.5";;
        *"RTX 2070"*) echo "7.5";;
        *"RTX 2060 SUPER"*) echo "7.5";;
        *"RTX 2060"*) echo "7.5";;
        
        # GTX 16 Series (Turing)
        *"GTX 1660 TI"*) echo "7.5";;
        *"GTX 1660 SUPER"*) echo "7.5";;
        *"GTX 1660"*) echo "7.5";;
        *"GTX 1650 TI"*) echo "7.5";;
        *"GTX 1650"*) echo "7.5";;
        
        # GTX 10 Series (Pascal)
        *"GTX 1080 TI"*) echo "6.1";;
        *"GTX 1080"*) echo "6.1";;
        *"GTX 1070 TI"*) echo "6.1";;
        *"GTX 1070"*) echo "6.1";;
        *"GTX 1060"*) echo "6.1";;
        *"GTX 1050 TI"*) echo "6.1";;
        *"GTX 1050"*) echo "6.1";;
        
        # Professional GPUs
        *"Tesla V100"*) echo "7.0";;
        *"Tesla T4"*) echo "7.5";;
        *"Tesla A100"*) echo "8.0";;
        *"RTX A6000"*) echo "8.6";;
        *"RTX A5000"*) echo "8.6";;
        *"RTX A4000"*) echo "8.6";;
        *"RTX A2000"*) echo "8.6";;
        *)
            log_warning "Unknown GPU architecture: $gpu_name"
            echo "6.1"  # Default to Pascal
            ;;
    esac
}

# Enhanced NVIDIA environment detection
check_nvidia_environment() {
    log_header "NVIDIA Environment Detection"
    
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log_error "nvidia-smi not found. NVIDIA drivers may not be installed."
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
    
    # Get driver version
    local driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -n1)
    log_info "Driver Version: $driver_version"
    
    # Get GPU memory
    local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)
    if [[ -n "$gpu_memory" ]]; then
        local memory_gb=$((gpu_memory / 1024))
        log_info "GPU Memory: ${memory_gb}GB"
    fi
    
    CUDA_AVAILABLE=true
    return 0
}

# Enhanced NVCC verification
check_nvcc() {
    log_header "CUDA Toolkit Detection"
    
    if ! command -v nvcc >/dev/null 2>&1; then
        log_error "nvcc (CUDA compiler) not found"
        return 1
    fi
    
    local nvcc_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    log_success "NVCC found (version $nvcc_version)"
    
    NVCC_AVAILABLE=true
    return 0
}

# Enhanced OpenCV CUDA detection
check_opencv_cuda() {
    log_header "OpenCV CUDA Support Detection"
    
    # Check multiple Python environments
    local cuda_opencv_found=false
    
    for python_exe in "python3" "python"; do
        local cuda_result=$($python_exe -c "
try:
    import cv2
    if hasattr(cv2, 'cuda'):
        if hasattr(cv2.cuda, 'getCudaEnabledDeviceCount'):
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            print(f'CUDA:{cuda_count}:{cv2.__version__}')
        else:
            print('CUDA:0:no_getCudaEnabledDeviceCount')
    else:
        print('CUDA:0:no_cuda_module')
except Exception as e:
    print(f'CUDA:0:ERROR:{str(e)}')
" 2>/dev/null)
        
        if [[ -n "$cuda_result" ]]; then
            local cuda_count=$(echo "$cuda_result" | cut -d':' -f2)
            local opencv_version=$(echo "$cuda_result" | cut -d':' -f3)
            
            if [[ "$cuda_count" != "0" ]]; then
                log_success "✅ CUDA-enabled OpenCV found in $python_exe environment!"
                log_info "OpenCV version: $opencv_version"
                log_info "CUDA devices: $cuda_count"
                cuda_opencv_found=true
                break
            fi
        fi
    done
    
    if [[ "$cuda_opencv_found" == "true" ]]; then
        log_success "🎉 OpenCV CUDA support is available!"
        OPENCV_CUDA_SUPPORT=true
    else
        log_error "❌ OpenCV CUDA support not available"
        log_info "Install CUDA-enabled OpenCV:"
        echo "  conda create -n opencv_cuda12 python=3.12"
        echo "  conda install -c conda-forge -y glib gtk3 gstreamer gst-plugins-base protobuf absl-py"
        echo "  conda install -c nvidia cuda-toolkit cudnn"
        echo "  conda activate opencv_cuda12"
    fi
    
    return 0
}

# Setup CUDA environment
setup_cuda_environment() {
    log_header "CUDA Environment Setup"
    
    # Check for CUDA 12.8 first and ensure it's used
    if [[ -d "/usr/local/cuda-12.8" ]]; then
        export CUDA_PATH="/usr/local/cuda-12.8"
        export CUDA_HOME="/usr/local/cuda-12.8"
        export CUDA_ROOT="$CUDA_PATH"
        log_success "✅ Found CUDA 12.8"
    else
        log_error "CUDA 12.8 not found at /usr/local/cuda-12.8"
        return 1
    fi
    
    if [[ -n "$CUDA_PATH" ]]; then
        # Set explicit CUDA environment variables
        export PATH="$CUDA_PATH/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
        export CUDA_PATH="$CUDA_PATH"
        export CUDA_HOME="$CUDA_PATH"
        export CUDA_ROOT="$CUDA_PATH"
        
        # Disable old CUDA toolkit to prevent conflicts
        export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "/usr/lib/nvidia-cuda-toolkit/bin" | tr '\n' ':' | sed 's/:$//')
        export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "/usr/lib/nvidia-cuda-toolkit" | tr '\n' ':' | sed 's/:$//')
        
        # Verify CUDA 12.8 NVCC is accessible
        if [[ -x "$CUDA_PATH/bin/nvcc" ]]; then
            local nvcc_version=$($CUDA_PATH/bin/nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
            log_success "✅ NVCC version: $nvcc_version"
        else
            log_error "❌ NVCC not found at $CUDA_PATH/bin/nvcc"
            return 1
        fi
        
        log_info "CUDA environment configured to use: $CUDA_PATH"
        log_info "PATH contains: $(echo $PATH | tr ':' '\n' | grep cuda)"
    fi
    
    return 0
}

# Main build function
main() {
    log_header "Background Remover Lite - Enhanced Build Script"
    
    # Check if we're in a Conda environment
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        log_info "Running in Conda environment: $CONDA_DEFAULT_ENV"
        # Respect the current environment - don't override it
        if [[ "$CONDA_DEFAULT_ENV" == "opencv_cuda12" ]]; then
            log_success "✅ Using CUDA-enabled OpenCV environment"
        elif [[ "$CONDA_DEFAULT_ENV" == "opencv_cpu" ]]; then
            log_info "Using CPU-only OpenCV environment"
        else
            log_info "Using custom Conda environment: $CONDA_DEFAULT_ENV"
        fi
    else
        log_info "Running in system environment"
        log_info "Using system OpenCV or conda-forge OpenCV"
    fi
    
    # Environment checks
    check_nvidia_environment
    setup_cuda_environment
    check_nvcc
    check_opencv_cuda
    
    # Clean build directory to avoid cached CMake configuration
    log_header "Build Configuration"
    if [[ -d "build" ]]; then
        log_info "Cleaning existing build directory..."
        rm -rf build
    fi
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
    
    CMAKE_FLAGS="-DU2NET_DOWNLOAD_MODELS=ON"
    
    # Use OpenCV's built-in CUDA support (already available in conda environment)
    log_info "Building with system/conda-forge OpenCV"
    # Don't override OpenCV's CUDA settings - let it use its built-in support
    log_info "Using system OpenCV - GPU acceleration via ONNX Runtime will still be available"
    
    # Display final CMake configuration
    log_info "CMake configuration command: cmake $CMAKE_FLAGS .."
    
    # Run CMake
    if cmake $CMAKE_FLAGS ..; then
        log_success "CMake configuration successful"
    else
        log_error "CMake configuration failed"
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
    
    echo "🏗️  Build Summary:"
    echo ""
    
    if [[ -f "./bgremover" ]]; then
        log_success "  📱 CPU Version: ./bgremover"
        echo "     • Works on any system with x64 CPU"
        echo "     • Performance: 1-5 FPS"
        echo "     • Perfect for testing and fallback"
    fi
    
    if [[ -f "./bgremover_gpu" ]]; then
        log_success "  🚀 GPU Version: ./bgremover_gpu"
        echo "     • Requires NVIDIA GPU with CUDA"
        echo "     • Performance: 25-30 FPS at 1080p"
        echo "     • Recommended for production use"
    fi
    
    echo ""
    echo "📋 Usage Examples:"
    if [[ -f "./bgremover" ]]; then
        echo "  • CPU: ./bgremover --help"
    fi
    if [[ -f "./bgremover_gpu" ]]; then
        echo "  • GPU: ./build/bgremover_gpu --help"
    fi
    
    if [[ ! -f "./bgremover" ]] && [[ ! -f "./bgremover_gpu" ]]; then
        log_error "❌ No executables were created"
        log_info "   Check the build logs above for errors"
        exit 1
    fi
    
    # Final status
    log_header "Environment Summary"
    
    local cuda_status=$(if [[ "$CUDA_AVAILABLE" == "true" ]]; then echo "✅"; else echo "❌"; fi)
    local nvcc_status=$(if [[ "$NVCC_AVAILABLE" == "true" ]]; then echo "✅"; else echo "❌"; fi)
    local opencv_cuda_status=$(if [[ "$OPENCV_CUDA_SUPPORT" == "true" ]]; then echo "✅"; else echo "❌"; fi)
    
    echo "  CUDA Available: $cuda_status"
    echo "  NVCC Available: $nvcc_status"
    echo "  OpenCV CUDA: $opencv_cuda_status"
    
    if [[ -n "$GPU_NAME" ]]; then
        echo "  GPU: $GPU_NAME"
    fi
    
    # Overall assessment
    echo ""
    if [[ "$CUDA_AVAILABLE" == "true" ]] && [[ "$OPENCV_CUDA_SUPPORT" == "true" ]]; then
        log_success "🎉 All systems ready for GPU-accelerated background blur!"
    elif [[ "$CUDA_AVAILABLE" == "true" ]]; then
        log_warning "⚠️  CUDA available but OpenCV CUDA support missing"
    else
        log_error "❌ Running in CPU-only mode"
    fi
}

# Run main function
main "$@"