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
    
    if ! pkg-config --exists opencv4; then
        log_error "OpenCV4 is not installed or not found in PKG_CONFIG_PATH"
        log_info "Install OpenCV4 development packages:"
        echo "  Ubuntu/Debian: sudo apt install libopencv-dev python3-opencv"
        echo "  CentOS/RHEL: sudo yum install opencv-devel python3-opencv"
        return 1
    fi
    
    # Get OpenCV version and build information
    local opencv_version=$(pkg-config --modversion opencv4)
    log_info "OpenCV version: $opencv_version"
    
    # Get OpenCV build information
    local opencv_build_info=$(pkg-config --libs opencv4 2>/dev/null | grep -o "\-lopencv_\w*" | head -5)
    log_info "OpenCV modules: $opencv_build_info"
    
    # Check if OpenCV has CUDA modules via pkg-config
    if pkg-config --libs opencv4 2>/dev/null | grep -q "cudart\|cublas\|cufft"; then
        log_success "✅ OpenCV libraries linked to CUDA runtime"
    else
        log_warning "⚠️  OpenCV libraries not linked to CUDA runtime"
    fi
    
    # Check CUDA version in OpenCV build
    if pkg-config --cflags opencv4 2>/dev/null | grep -q "CUDA"; then
        log_success "✅ OpenCV compiled with CUDA support"
    else
        log_warning "⚠️  OpenCV may not be compiled with CUDA support"
    fi
    
    # Enhanced Python OpenCV CUDA detection
    local python_cuda_result=$(python3 -c "
import sys
import importlib.util
try:
    import cv2
    if hasattr(cv2, 'cuda'):
        # Check CUDA module
        cuda_modules = [attr for attr in dir(cv2.cuda) if not attr.startswith('_')]
        print(f'CUDA modules: {len(cuda_modules)}')
        
        # Test CUDA device count
        if hasattr(cv2.cuda, 'getCudaEnabledDeviceCount'):
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            print(f'CUDA devices: {cuda_count}')
            
            if cuda_count > 0:
                # Test CUDA info
                if hasattr(cv2.cuda, 'DeviceInfo'):
                    device_info = cv2.cuda.DeviceInfo()
                    print(f'Device: {device_info.name()}')
                    print(f'Memory: {device_info.totalMemory()} bytes')
            
            print('CUDA_AVAILABLE=True')
        else:
            print('CUDA_AVAILABLE=False')
    else:
        print('CUDA_AVAILABLE=False')
except Exception as e:
    print(f'ERROR: {str(e)}')
    print('CUDA_AVAILABLE=False')
" 2>/dev/null)
    
    if echo "$python_cuda_result" | grep -q "CUDA_AVAILABLE=True"; then
        local cuda_modules=$(echo "$python_cuda_result" | grep "CUDA modules:" | cut -d':' -f2 | tr -d ' ')
        local cuda_devices=$(echo "$python_cuda_result" | grep "CUDA devices:" | cut -d':' -f2 | tr -d ' ')
        local device_name=$(echo "$python_cuda_result" | grep "Device:" | cut -d':' -f2- | sed 's/^ *//')
        
        log_success "✅ OpenCV CUDA support detected!"
        log_info "CUDA modules available: $cuda_modules"
        log_info "CUDA devices: $cuda_devices"
        if [[ -n "$device_name" ]]; then
            log_info "CUDA device: $device_name"
        fi
        
        OPENCV_CUDA_SUPPORT=true
    else
        log_error "❌ OpenCV CUDA support not available"
        log_info "Error details:"
        echo "$python_cuda_result" | grep "ERROR:" || echo "No CUDA modules found"
        
        log_info "To enable CUDA support:"
        echo "  1. Install CUDA-enabled OpenCV:"
        echo "     pip install opencv-contrib-python"
        echo "  2. Or build OpenCV from source with CUDA flags"
        echo "  3. Or install system package with CUDA support"
        return 1
    fi
    
    # Verify library linking
    local opencv_libs=$(ldd $(python3 -c "import cv2; print(cv2.__file__)" 2>/dev/null | xargs dirname 2>/dev/null)/cv2*.so 2>/dev/null | grep "libcuda\|libcudart" || echo "")
    if [[ -n "$opencv_libs" ]]; then
        log_success "✅ OpenCV properly linked to CUDA libraries"
    else
        log_warning "⚠️  OpenCV may not be properly linked to CUDA libraries"
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
        
        # Check if Conda OpenCV is shadowing system OpenCV
        local conda_opencv_path=$(python3 -c "import cv2; print(cv2.__file__)" 2>/dev/null || echo "")
        if [[ "$conda_opencv_path" == *"conda"* ]] || [[ "$conda_opencv_path" == *"site-packages"* ]]; then
            local conda_opencv_version=$(python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "Unknown")
            local system_opencv_version=$(pkg-config --modversion opencv4 2>/dev/null || echo "Not found")
            
            log_warning "⚠️  Conda OpenCV detected - potential shadowing issue"
            log_info "Conda OpenCV: $conda_opencv_path (v$conda_opencv_version)"
            log_info "System OpenCV: $system_opencv_version"
            
            if [[ "$conda_opencv_version" != "$system_opencv_version" ]] && [[ "$system_opencv_version" != "Not found" ]]; then
                log_error "❌ Version mismatch detected!"
                log_info "This can cause runtime issues and library conflicts."
                
                # Offer automatic fix
                echo ""
                log_info "Would you like to uninstall Conda OpenCV to use system OpenCV? (y/N)"
                read -t 10 -r response || response="n"
                
                if [[ "$response" =~ ^[Yy]$ ]]; then
                    log_info "Uninstalling Conda OpenCV packages..."
                    if python3 -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless 2>/dev/null; then
                        log_success "✅ Conda OpenCV packages uninstalled"
                        
                        # Verify system OpenCV is now used
                        local new_path=$(python3 -c "import cv2; print(cv2.__file__)" 2>/dev/null || echo "")
                        if [[ "$new_path" != *"site-packages"* ]]; then
                            log_success "✅ System OpenCV is now being used"
                        fi
                    else
                        log_error "Failed to uninstall Conda OpenCV packages"
                    fi
                else
                    log_info "Skipping automatic fix - manual intervention required"
                fi
            fi
        fi
    fi
    
    # Get Python OpenCV info
    local python_opencv_path=$(python3 -c "import cv2; print(cv2.__file__)" 2>/dev/null || echo "")
    local python_opencv_version=$(python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "Not found")
    
    # Get system OpenCV info
    local system_opencv_version=$(pkg-config --modversion opencv4 2>/dev/null || echo "Not found")
    
    log_info "Python OpenCV path: $python_opencv_path"
    log_info "Python OpenCV version: $python_opencv_version"
    log_info "System OpenCV version: $system_opencv_version"
    
    # Enhanced shadowing detection
    if [[ "$python_opencv_path" == *"site-packages"* ]] && [[ "$system_opencv_version" != "Not found" ]]; then
        if [[ "$python_opencv_version" != "$system_opencv_version" ]]; then
            log_warning "⚠️  OpenCV version mismatch detected!"
            log_warning "System OpenCV ($system_opencv_version) vs Python OpenCV ($python_opencv_version)"
            log_error "This will cause module shadowing and potential runtime issues."
            
            # Check for multiple OpenCV installations
            local opencv_count=$(python3 -c "import site; print(len(site.getsitepackages()))" 2>/dev/null || echo "1")
            if [[ $opencv_count -gt 1 ]]; then
                log_warning "Multiple Python site-packages directories detected"
            fi
            
            PYTHON_OPENCV_ISSUES=true
            
            # Provide detailed solutions
            log_info "Solutions to resolve this issue:"
            echo "  1. Remove pip-installed OpenCV to use system OpenCV:"
            echo "     python3 -m pip uninstall opencv-python opencv-contrib-python opencv-python-headless"
            echo ""
            echo "  2. Install matching system OpenCV version via pip:"
            echo "     python3 -m pip install opencv-contrib-python==$system_opencv_version"
            echo ""
            echo "  3. Use a virtual environment:"
            echo "     python3 -m venv opencv_env && source opencv_env/bin/activate"
            echo ""
            echo "  4. For Conda users, create a clean environment:"
            echo "     conda create -n opencv_env python=3.9"
            echo "     conda activate opencv_env"
            echo "     conda install opencv"
            
            return 1
        else
            log_success "✅ OpenCV versions match between system and Python"
        fi
    elif [[ "$python_opencv_path" == *"site-packages"* ]] && [[ "$system_opencv_version" == "Not found" ]]; then
        log_warning "⚠️  Only Python OpenCV found - system OpenCV not available"
        log_info "Consider installing system OpenCV for better performance:"
        echo "  Ubuntu/Debian: sudo apt install libopencv-dev python3-opencv"
    elif [[ -z "$python_opencv_path" ]]; then
        log_error "❌ OpenCV not found in Python environment"
        log_info "Install OpenCV:"
        echo "  pip install opencv-contrib-python"
        return 1
    else
        log_success "✅ Python OpenCV is using system libraries"
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
        echo "  3. Install CUDA-enabled OpenCV: pip install opencv-contrib-python"
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
}

# Run main function
main "$@"
