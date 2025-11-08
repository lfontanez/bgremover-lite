#!/bin/bash

# Enhanced build script that demonstrates the new model download module
# This script shows how to use the comprehensive model download functionality

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
    echo -e "${GREEN}=== $1 ===${NC}"
}

# Test the model download module
test_model_download() {
    log_header "Testing Model Download Module"
    
    if command -v python3 >/dev/null 2>&1; then
        log_info "Running model download test..."
        if python3 test_model_download.py --cmake-test; then
            log_success "Model download test passed!"
            return 0
        else
            log_warning "Model download test failed, but this is expected without internet"
            return 1
        fi
    else
        log_warning "Python3 not found, skipping model download test"
        return 1
    fi
}

# Check model files
check_model_files() {
    log_header "Checking Model Files"
    
    local models_dir="models"
    local all_good=true
    
    if [ ! -d "$models_dir" ]; then
        log_warning "Models directory not found: $models_dir"
        log_info "The build will attempt to download models automatically"
        return 1
    fi
    
    for model in "u2net.onnx" "u2netp.onnx"; do
        if [ -f "$models_dir/$model" ]; then
            local size_mb=$(du -h "$models_dir/$model" | cut -f1)
            log_success "$model: $size_mb"
        else
            log_error "$model: Not found"
            all_good=false
        fi
    done
    
    if [ "$all_good" = true ]; then
        log_success "All model files are available!"
        return 0
    else
        log_warning "Some model files are missing - build will download them"
        return 1
    fi
}

# Main build function
main() {
    log_header "U²-Net Build with Enhanced Model Download"
    
    log_info "This build demonstrates the comprehensive model download module"
    log_info "Features:"
    echo "  • Multiple download sources (GitHub, Hugging Face, direct)"
    echo "  • SHA-256 integrity verification"
    echo "  • Persistent caching in ~/.cache/u2net"
    echo "  • Offline build support"
    echo "  • Progress indication for large files"
    echo "  • Comprehensive error handling"
    
    # Check existing models
    models_ok=false
    if check_model_files; then
        models_ok=true
    fi
    
    # Test download module if possible
    test_model_download
    
    # Create build directory
    log_header "Building Project"
    mkdir -p build
    cd build
    
    # Configure with CMake
    log_info "Configuring with CMake..."
    local cmake_args=(
        -DCMAKE_BUILD_TYPE=Release
        -DU2NET_DOWNLOAD_MODELS=ON
        -DU2NET_CLEAN_CACHE=OFF
    )
    
    # Add offline mode if no internet
    if ! ping -c 1 google.com >/dev/null 2>&1; then
        log_info "No internet detected - enabling offline mode"
        cmake_args+=(-DU2NET_OFFLINE_MODE=ON)
    fi
    
    log_info "CMake args: ${cmake_args[*]}"
    
    if cmake .. "${cmake_args[@]}"; then
        log_success "CMake configuration successful"
    else
        log_error "CMake configuration failed"
        exit 1
    fi
    
    # Build
    log_info "Building project..."
    if make -j$(nproc); then
        log_success "Build successful"
    else
        log_error "Build failed"
        exit 1
    fi
    
    # Results
    log_header "Build Results"
    
    if [ -f "./bgremover" ]; then
        log_success "CPU version: ./bgremover"
    fi
    
    if [ -f "./bgremover_gpu" ]; then
        log_success "GPU version: ./bgremover_gpu"
    fi
    
    # Model cache info
    log_info "Model cache location:"
    if [ -n "$XDG_CACHE_HOME" ]; then
        echo "  $XDG_CACHE_HOME/u2net"
    elif [ "$OSTYPE" = "msys" ] || [ "$OSTYPE" = "cygwin" ]; then
        echo "  $LOCALAPPDATA/u2net/cache"
    else
        echo "  ~/.cache/u2net"
    fi
    
    log_success "Build completed successfully!"
    
    # Usage instructions
    log_header "Usage"
    echo "CPU version:"
    echo "  ./bgremover                    # Webcam"
    echo "  ./bgremover video.mp4         # Video file"
    echo ""
    echo "GPU version:"
    echo "  ./bgremover_gpu               # Webcam"
    echo "  ./bgremover_gpu video.mp4    # Video file"
    echo ""
    echo "Virtual camera:"
    echo "  ./bgremover_gpu --vcam        # Send to /dev/video2"
    echo "  ./bgremover_gpu --vcam-device /dev/video3  # Custom device"
}

# Run main function
main "$@"
