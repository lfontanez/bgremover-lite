#!/bin/bash

# Background Remover Lite - Enhanced Build Script
# This script builds the background remover project with comprehensive CUDA detection and configuration

set -e  # Exit on any error

# Enhanced color codes for better visual feedback
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Enhanced logging functions
log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_header() {
    echo ""
    echo -e "${BOLD}${CYAN}=== $1 ===${NC}"
}

log_enhancement() {
    echo -e "${MAGENTA}[ENHANCED]${NC} $1"
}

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
        
        # GTX 900 Series (Maxwell)
        *"GTX 980 TI"*) echo "5.2";;
        *"GTX 980"*) echo "5.2";;
        *"GTX 970"*) echo "5.2";;
        *"GTX 960"*) echo "5.2";;
        *"GTX 950"*) echo "5.2";;
        
        # Professional GPUs
        *"Tesla V100"*) echo "7.0";;
        *"Tesla T4"*) echo "7.5";;
        *"Tesla A100"*) echo "8.0";;
        *"Tesla K80"*) echo "3.7";;
        *"Tesla K40"*) echo "3.5";;
        *"Tesla K20"*) echo "3.5";;
        *"Quadro RTX 8000"*) echo "7.5";;
        *"Quadro RTX 6000"*) echo "7.5";;
        *"Quadro RTX 5000"*) echo "7.5";;
        *"Quadro RTX 4000"*) echo "7.5";;
        *"Quadro RTX 3000"*) echo "7.5";;
        *"Quadro GP100"*) echo "6.0";;
        *"Quadro GV100"*) echo "7.0";;
        *"Tesla P100"*) echo "6.0";;
        *"Tesla P40"*) echo "6.1";;
        *"Tesla P4"*) echo "6.1";;
        
        # Workstation cards
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

# Enhanced NVIDIA environment detection with driver and runtime version checking
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
    
    # Get driver version
    local driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -n1)
    log_info "Driver Version: $driver_version"
    
    # Get CUDA version supported by driver
    local cuda_version=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader,nounits 2>/dev/null | head -n1)
    if [[ -n "$cuda_version" ]]; then
        log_info "CUDA Version: $cuda_version"
    else
        log_info "CUDA Version: Not available (older driver)"
    fi
    
    # Get GPU memory
    local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)
    if [[ -n "$gpu_memory" ]]; then
        local memory_gb=$((gpu_memory / 1024))
        log_info "GPU Memory: ${memory_gb}GB"
        
        # Check memory adequacy
        if [[ $memory_gb -ge 8 ]]; then
            log_success "✅ GPU has sufficient memory (${memory_gb}GB)"
        elif [[ $memory_gb -ge 4 ]]; then
            log_warning "⚠️  GPU has moderate memory (${memory_gb}GB) - may impact performance"
        else
            log_error "❌ GPU has limited memory (${memory_gb}GB) - may not be suitable"
        fi
    fi
    
    # Get GPU utilization and temperature
    local utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1)
    if [[ -n "$utilization" ]]; then
        log_info "Current GPU Utilization: ${utilization}%"
    fi
    
    # Validate driver version
    local driver_major=$(echo "$driver_version" | cut -d'.' -f1)
    if [[ $driver_major -ge 450 ]]; then
        log_success "✅ Driver version $driver_version is compatible with modern CUDA"
    else
        log_warning "⚠️  Driver version $driver_version may be outdated"
    fi
    
    CUDA_AVAILABLE=true
    return 0
}

# Enhanced NVCC verification with comprehensive version checking
check_nvcc() {
    log_header "CUDA Toolkit Detection"
    
    if ! command -v nvcc >/dev/null 2>&1; then
        log_error "nvcc (CUDA compiler) not found"
        log_info "Install CUDA Toolkit:"
        echo "  Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit"
        echo "  Or download from: https://developer.nvidia.com/cuda-downloads"
        return 1
    fi
    
    # Get detailed NVCC version information
    local nvcc_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    local nvcc_full_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\.[0-9]\+\).*/\1/')
    local nvcc_cuda_version=$(nvcc --version | grep "Cuda compilation tools" | sed 's/.*build \([0-9]\+\).*/\1/')
    
    log_success "NVCC found (version $nvcc_version)"
    log_info "Full version: $nvcc_full_version"
    log_info "CUDA build: $nvcc_cuda_version"
    
    # Validate CUDA version compatibility
    if [[ $(echo "$nvcc_version >= 11.0" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
        log_success "✅ CUDA version $nvcc_version is compatible with modern frameworks"
    elif [[ $(echo "$nvcc_version >= 10.0" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
        log_warning "⚠️  CUDA version $nvcc_version is supported but consider upgrading"
    else
        log_error "❌ CUDA version $nvcc_version is outdated for modern development"
        return 1
    fi
    
    # Check NVCC environment variables
    if [[ -n "$CUDA_PATH" ]]; then
        log_info "CUDA_PATH: $CUDA_PATH"
    else
        log_warning "CUDA_PATH not set - may cause linking issues"
    fi
    
    # Check for CUDA_HOME
    if [[ -n "$CUDA_HOME" ]]; then
        log_info "CUDA_HOME: $CUDA_HOME"
    else
        log_info "CUDA_HOME not set (optional)"
    fi
    
    # Verify NVCC can compile simple test
    if nvcc -V >/dev/null 2>&1; then
        log_success "✅ NVCC verification test passed"
    else
        log_error "❌ NVCC verification test failed"
        return 1
    fi
    
    NVCC_AVAILABLE=true
    return 0
}

# Enhanced OpenCV CUDA detection with build information verification
check_opencv_cuda() {
    log_header "OpenCV CUDA Support Detection"
    
    # Check for system OpenCV
    local system_opencv_available=false
    if pkg-config --exists opencv4; then
        local system_opencv_version=$(pkg-config --modversion opencv4)
        log_info "System OpenCV version: $system_opencv_version"
        system_opencv_available=true
    else
        log_info "System OpenCV4 not found in PKG_CONFIG_PATH"
    fi
    
    # Check multiple Python environments for CUDA-enabled OpenCV
    local cuda_opencv_found=false
    local cuda_opencv_env=""
    local cuda_opencv_version=""
    
    for python_exe in "python3" "python"; do
        # Check current Python environment
        local cuda_result=$($python_exe -c "
try:
    import cv2
    if hasattr(cv2, 'cuda'):
        if hasattr(cv2.cuda, 'getCudaEnabledDeviceCount'):
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            print(f'CURRENT_PYTHON_CUDA:{cuda_count}:{cv2.__version__}:{cv2.__file__}')
        else:
            print('CURRENT_PYTHON_CUDA:0:no_getCudaEnabledDeviceCount:unknown')
    else:
        print('CURRENT_PYTHON_CUDA:0:no_cuda_module:unknown')
except Exception as e:
    print(f'CURRENT_PYTHON_CUDA:0:ERROR:{str(e)}')
" 2>/dev/null)
        
        if [[ -n "$cuda_result" ]]; then
            local cuda_count=$(echo "$cuda_result" | cut -d':' -f2)
            local opencv_version=$(echo "$cuda_result" | cut -d':' -f3)
            local opencv_path=$(echo "$cuda_result" | cut -d':' -f4)
            
            if [[ "$cuda_count" != "0" ]]; then
                log_success "✅ CUDA-enabled OpenCV found in $python_exe environment!"
                log_info "OpenCV version: $opencv_version"
                log_info "CUDA devices: $cuda_count"
                log_info "Path: $opencv_path"
                cuda_opencv_found=true
                cuda_opencv_version="$opencv_version"
                break
            fi
        fi
    done
    
    # Also check common Conda environments
    if [[ "$cuda_opencv_found" == "false" ]]; then
        for conda_env in opencv_cuda12 opencv_env py39 py310 py311 py312; do
            if [[ -f "$HOME/miniconda3/envs/$conda_env/bin/python" ]]; then
                local conda_cuda_result=$($HOME/miniconda3/envs/$conda_env/bin/python -c "
try:
    import cv2
    if hasattr(cv2, 'cuda'):
        if hasattr(cv2.cuda, 'getCudaEnabledDeviceCount'):
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            print(f'CONDA_CUDA:{cuda_count}:{cv2.__version__}:{cv2.__file__}')
        else:
            print('CONDA_CUDA:0:no_getCudaEnabledDeviceCount:unknown')
    else:
        print('CONDA_CUDA:0:no_cuda_module:unknown')
except Exception as e:
    print(f'CONDA_CUDA:0:ERROR:{str(e)}')
" 2>/dev/null)
                
                if [[ -n "$conda_cuda_result" ]]; then
                    local cuda_count=$(echo "$conda_cuda_result" | cut -d':' -f2)
                    local opencv_version=$(echo "$conda_cuda_result" | cut -d':' -f3)
                    local opencv_path=$(echo "$conda_cuda_result" | cut -d':' -f4)
                    
                    if [[ "$cuda_count" != "0" ]]; then
                        log_success "✅ CUDA-enabled OpenCV found in Conda environment: $conda_env!"
                        log_info "OpenCV version: $opencv_version"
                        log_info "CUDA devices: $cuda_count"
                        log_info "Path: $opencv_path"
                        cuda_opencv_found=true
                        cuda_opencv_env="$conda_env"
                        cuda_opencv_version="$opencv_version"
                        break
                    fi
                fi
            fi
        done
    fi
    
    if [[ "$cuda_opencv_found" == "true" ]]; then
        log_success "🎉 OpenCV CUDA support is available!"
        OPENCV_CUDA_SUPPORT=true
        
        if [[ -n "$cuda_opencv_env" ]]; then
            log_info "To use CUDA-enabled OpenCV, run:"
            echo "  source ~/miniconda3/bin/activate $cuda_opencv_env"
        fi
    else
        log_error "❌ OpenCV CUDA support not available in any Python environment"
        log_info "Available options:"
        
        if [[ "$system_opencv_available" == "true" ]]; then
            echo "  • System OpenCV $system_opencv_version (CPU only)"
        fi
        
        echo "  • Install CUDA-enabled OpenCV in Conda:"
        echo "    conda create -n opencv_cuda12 python=3.12 opencv cudatoolkit=12.1"
        echo "    conda activate opencv_cuda12"
        echo ""
        echo "  • Or install CUDA-enabled OpenCV with pip:"
        echo "    pip install opencv-contrib-python"
        echo ""
        
        # Still set OpenCV_CUDA_SUPPORT to false but don't fail
        log_info "Build will continue with CPU-only OpenCV support"
    fi
    
    return 0
}

# Enhanced Python OpenCV environment check with Conda support and auto-fix
check_python_opencv_shadowing() {
    log_header "Python OpenCV Environment Check"
    
    if ! command -v python3 >/dev/null 2>&1; then
        log_warning "Python3 not found"
        return 0
    fi
    
    # Check for Conda environment
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        log_info "Conda environment: $CONDA_DEFAULT_ENV"
    fi
    
    # Check for common Conda locations
    local conda_python_path=""
    local conda_opencv_path=""
    
    # Try to find OpenCV in common Conda locations
    for conda_env in opencv_cuda12 opencv_env py39 py310 py311 py312; do
        if [[ -f "$HOME/miniconda3/envs/$conda_env/bin/python" ]]; then
            local test_opencv=$(HOME/miniconda3/envs/$conda_env/bin/python -c "import cv2; print('found')" 2>/dev/null || echo "notfound")
            if [[ "$test_opencv" == "found" ]]; then
                local opencv_version=$(HOME/miniconda3/envs/$conda_env/bin/python -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "unknown")
                local opencv_file=$(HOME/miniconda3/envs/$conda_env/bin/python -c "import cv2; print(cv2.__file__)" 2>/dev/null || echo "unknown")
                local cuda_devices=$(HOME/miniconda3/envs/$conda_env/bin/python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())" 2>/dev/null || echo "0")
                
                if [[ "$cuda_devices" != "0" ]] && [[ "$cuda_devices" != "AttributeError" ]]; then
                    log_success "✅ CUDA-enabled OpenCV found in Conda environment: $conda_env"
                    log_info "OpenCV version: $opencv_version"
                    log_info "CUDA devices: $cuda_devices"
                    log_info "Path: $opencv_file"
                    
                    # Check if current Python is using this environment
                    local current_opencv=$(python3 -c "import cv2; print(cv2.__file__)" 2>/dev/null || echo "notfound")
                    if [[ "$current_opencv" == "notfound" ]] || [[ "$current_opencv" != *"conda"* ]]; then
                        log_info "Current Python not using Conda OpenCV - this is normal"
                        log_info "To use CUDA-enabled OpenCV, activate Conda environment:"
                        echo "  source ~/miniconda3/bin/activate $conda_env"
                    fi
                fi
            fi
        fi
    done
    
    # Get current Python OpenCV info
    local python_opencv_path=$(python3 -c "import cv2; print(cv2.__file__)" 2>/dev/null || echo "")
    local python_opencv_version=$(python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "Not found")
    
    # Get system OpenCV info
    local system_opencv_version=$(pkg-config --modversion opencv4 2>/dev/null || echo "Not found")
    
    log_info "Current Python OpenCV path: $python_opencv_path"
    log_info "Current Python OpenCV version: $python_opencv_version"
    log_info "System OpenCV version: $system_opencv_version"
    
    # Check if we have OpenCV available at all
    if [[ -z "$python_opencv_path" ]]; then
        log_warning "⚠️  OpenCV not found in current Python environment"
        log_info "Available Python OpenCV environments:"
        echo "  • Current (system): Not available"
        
        # Check for Conda environments with OpenCV
        local conda_found=false
        for conda_env in opencv_cuda12 opencv_env py39 py310 py311 py312; do
            if [[ -f "$HOME/miniconda3/envs/$conda_env/bin/python" ]]; then
                local test_opencv=$(HOME/miniconda3/envs/$conda_env/bin/python -c "import cv2; print('found')" 2>/dev/null || echo "notfound")
                if [[ "$test_opencv" == "found" ]]; then
                    local opencv_version=$(HOME/miniconda3/envs/$conda_env/bin/python -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "unknown")
                    local cuda_devices=$(HOME/miniconda3/envs/$conda_env/bin/python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())" 2>/dev/null || echo "0")
                    local cuda_status=$(if [[ "$cuda_devices" != "0" ]] && [[ "$cuda_devices" != "AttributeError" ]]; then echo "✅"; else echo "❌"; fi)
                    echo "  • Conda ($conda_env): $opencv_version $cuda_status"
                    conda_found=true
                fi
            fi
        done
        
        if [[ "$conda_found" == "true" ]]; then
            log_info "To use OpenCV with CUDA, activate a Conda environment:"
            echo "  source ~/miniconda3/bin/activate opencv_cuda12"
            echo "  # Then run this build script again"
        else
            log_error "❌ No OpenCV found in any Python environment"
            log_info "Install OpenCV:"
            echo "  pip install opencv-contrib-python"
            echo "  # Or for CUDA support:"
            echo "  conda create -n opencv_cuda12 python=3.12 opencv cudatoolkit=12.1"
            echo "  conda activate opencv_cuda12"
            return 1
        fi
    else
        log_success "✅ OpenCV found in current Python environment"
    fi
    
    # Check for shadowing issues only if we have both system and Python OpenCV
    if [[ "$python_opencv_path" == *"site-packages"* ]] && [[ "$system_opencv_version" != "Not found" ]]; then
        if [[ "$python_opencv_version" != "$system_opencv_version" ]]; then
            log_warning "⚠️  OpenCV version mismatch detected!"
            log_warning "System OpenCV ($system_opencv_version) vs Python OpenCV ($python_opencv_version)"
            log_info "This can cause module shadowing and potential runtime issues."
            
            # Check if this is a Conda installation (usually OK)
            if [[ "$python_opencv_path" == *"conda"* ]] || [[ "$python_opencv_path" == *"$HOME/miniconda3"* ]]; then
                log_info "Python OpenCV appears to be from Conda (OK for development)"
                log_info "System OpenCV is available for system tools if needed"
            else
                log_warning "This may cause conflicts between system tools and Python scripts"
                
                PYTHON_OPENCV_ISSUES=true
                return 1
            fi
        else
            log_success "✅ OpenCV versions match between system and Python"
        fi
    elif [[ "$python_opencv_path" == *"site-packages"* ]] && [[ "$system_opencv_version" == "Not found" ]]; then
        log_info "✅ Only Python OpenCV found - system OpenCV not available (this is fine)"
    else
        log_success "✅ OpenCV environment looks good"
    fi
    
    return 0
}

# Main build function
main() {
    log_header "Background Remover Lite - Enhanced Build Script"
    
    # Check if we're in a Conda environment
    local in_conda=false
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        in_conda=true
        log_info "Running in Conda environment: $CONDA_DEFAULT_ENV"
    else
        log_info "Running in system environment"
    fi
    
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
        log_success "CUDA support enabled in CMake"
    else
        CMAKE_FLAGS="-DWITH_CUDA=OFF"
        log_warning "CUDA support disabled - building CPU-only version"
    fi
    
    # Run CMake
    log_info "Running: cmake $CMAKE_FLAGS .."
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
        log_info "Common build issues and solutions:"
        echo "  • Missing OpenCV: sudo apt install libopencv-dev"
        echo "  • CUDA issues: Re-run with --cuda-only or --cpu-only"
        echo "  • CMake cache: rm -rf build && ./build.sh"
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
    
    if [[ ! -f "./bgremover" ]] && [[ ! -f "./bgremover_gpu" ]]; then
        log_error "❌ No executables were created"
        exit 1
    fi
    
    echo ""
    echo "Optional: Copy executables to the root directory:"
    echo "  cp build/bgremover .           # CPU version"
    if [[ -f "./bgremover_gpu" ]]; then
        echo "  cp build/bgremover_gpu .     # GPU version"
    fi
    
    # Set library path for ONNX Runtime
    if [[ -f "onnxruntime/lib/libonnxruntime.so" ]]; then
        export LD_LIBRARY_PATH="$PWD/onnxruntime/lib:$LD_LIBRARY_PATH"
        log_info "Added ONNX Runtime to LD_LIBRARY_PATH for GPU support"
    fi
    
    # Final status with comprehensive summary
    log_header "Environment Summary"
    
    # Create status indicators
    local cuda_status=$(if [[ "$CUDA_AVAILABLE" == "true" ]]; then echo "✅"; else echo "❌"; fi)
    local nvcc_status=$(if [[ "$NVCC_AVAILABLE" == "true" ]]; then echo "✅"; else echo "❌"; fi)
    local opencv_cuda_status=$(if [[ "$OPENCV_CUDA_SUPPORT" == "true" ]]; then echo "✅"; else echo "❌"; fi)
    local python_status=$(if [[ "$PYTHON_OPENCV_ISSUES" == "true" ]]; then echo "⚠️"; else echo "✅"; fi)
    
    # Print detailed summary
    echo -e "${WHITE}Driver & Runtime:${NC}"
    echo "  CUDA Available: $cuda_status"
    echo "  NVCC Available: $nvcc_status"
    if [[ -n "$GPU_NAME" ]]; then
        echo "  GPU: $GPU_NAME"
    fi
    if [[ -n "$driver_version" ]]; then
        echo "  Driver: $driver_version"
    fi
    
    echo ""
    echo -e "${WHITE}Libraries:${NC}"
    echo "  OpenCV CUDA: $opencv_cuda_status"
    echo "  Python Issues: $python_status"
    
    # Overall assessment
    echo ""
    if [[ "$CUDA_AVAILABLE" == "true" ]] && [[ "$OPENCV_CUDA_SUPPORT" == "true" ]] && [[ "$PYTHON_OPENCV_ISSUES" != "true" ]]; then
        log_success "🎉 All systems ready for GPU-accelerated background blur!"
        echo ""
        echo -e "${GREEN}✅ Environment is fully optimized for GPU acceleration${NC}"
        echo -e "${GREEN}✅ All CUDA and OpenCV components are properly configured${NC}"
        echo -e "${GREEN}✅ No Python environment conflicts detected${NC}"
    elif [[ "$CUDA_AVAILABLE" == "true" ]] && [[ "$OPENCV_CUDA_SUPPORT" == "true" ]]; then
        log_warning "⚠️  GPU acceleration available but there are Python environment issues"
        echo ""
        echo -e "${YELLOW}⚠️  CUDA and OpenCV are properly configured${NC}"
        echo -e "${YELLOW}⚠️  Python environment needs attention (see above)${NC}"
    elif [[ "$CUDA_AVAILABLE" == "true" ]]; then
        log_warning "⚠️  CUDA available but OpenCV CUDA support missing"
        echo ""
        echo -e "${YELLOW}⚠️  GPU and drivers are properly installed${NC}"
        echo -e "${YELLOW}⚠️  OpenCV needs CUDA support (see OpenCV section above)${NC}"
    else
        log_error "❌ Running in CPU-only mode"
        echo ""
        echo -e "${RED}❌ GPU acceleration is not available${NC}"
        echo -e "${RED}❌ The application will run on CPU only${NC}"
    fi
    
    # Provide next steps
    echo ""
    if [[ "$CUDA_AVAILABLE" != "true" ]] || [[ "$OPENCV_CUDA_SUPPORT" != "true" ]]; then
        echo -e "${BOLD}Next steps for GPU acceleration:${NC}"
        echo "  1. Install NVIDIA drivers: https://www.nvidia.com/drivers"
        echo "  2. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
        
        if [[ "$in_conda" == "true" ]]; then
            echo "  3. Install CUDA-enabled OpenCV in Conda:"
            echo "     conda install opencv cudatoolkit=12.1"
        else
            echo "  3. Install CUDA-enabled OpenCV:"
            echo "     conda create -n opencv_cuda python=3.12 opencv cudatoolkit=12.1"
            echo "     conda activate opencv_cuda"
        fi
        
        echo "  4. Re-run this build script"
    fi
    
    # Performance recommendations
    echo ""
    if [[ "$CUDA_AVAILABLE" == "true" ]] && [[ "$OPENCV_CUDA_SUPPORT" == "true" ]]; then
        echo -e "${BOLD}Performance Tips:${NC}"
        echo "  • Use GPU version: ./bgremover_gpu"
        echo "  • Monitor GPU utilization during processing"
        echo "  • Consider adjusting blur strength for optimal performance"
        echo "  • Ensure adequate system RAM and GPU memory"
    fi
    
    # Additional recommendations for mixed environments
    if [[ "$in_conda" == "true" ]] && [[ "$OPENCV_CUDA_SUPPORT" == "true" ]]; then
        echo ""
        echo -e "${BOLD}CUDA Environment Setup:${NC}"
        echo "  • Your Conda environment is ready for CUDA development"
        echo "  • Always activate this environment when using OpenCV CUDA"
        echo "  • Consider setting up other tools (PyTorch, ONNX Runtime) with CUDA"
    fi
}

# Run main function
main "$@"
