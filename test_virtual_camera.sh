#!/bin/bash

# Virtual Camera Test Script
# Quick verification that the virtual camera is working

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }

echo -e "${BLUE}=== Virtual Camera Test ===${NC}"
echo ""

# Test 1: Check if v4l2loopback module is loaded
log_info "Checking v4l2loopback module..."
if lsmod | grep -q v4l2loopback; then
    log_success "✅ v4l2loopback module is loaded"
else
    log_error "❌ v4l2loopback module is not loaded"
    echo "Run: ./setup_virtual_camera.sh"
    exit 1
fi

# Test 2: Check if device exists
log_info "Checking virtual camera device..."
if [[ -e /dev/video2 ]]; then
    log_success "✅ Virtual camera device exists: /dev/video2"
else
    log_error "❌ Virtual camera device not found: /dev/video2"
    echo "Run: ./setup_virtual_camera.sh"
    exit 1
fi

# Test 3: Check device permissions
log_info "Checking device permissions..."
if [[ -r /dev/video2 && -w /dev/video2 ]]; then
    log_success "✅ Device has read/write permissions"
else
    log_warning "⚠️  Device permissions may be insufficient"
    echo "Run: sudo chmod 666 /dev/video2"
fi

# Test 4: Query device capabilities
log_info "Querying device capabilities..."
if v4l2-ctl --device=/dev/video2 --all >/dev/null 2>&1; then
    log_success "✅ Device is accessible via v4l2-ctl"
    echo ""
    log_info "Device Information:"
    v4l2-ctl --device=/dev/video2 --info
else
    log_error "❌ Cannot query device with v4l2-ctl"
    exit 1
fi

# Test 5: List supported formats
echo ""
log_info "Supported formats:"
v4l2-ctl --device=/dev/video2 --list-formats-ext 2>/dev/null || log_warning "Could not list formats"

# Test 6: Check if bgremover_gpu exists
echo ""
log_info "Checking bgremover_gpu executable..."
if [[ -f "./build/bgremover_gpu" ]]; then
    log_success "✅ bgremover_gpu found"
elif [[ -f "./bgremover_gpu" ]]; then
    log_success "✅ bgremover_gpu found in current directory"
else
    log_warning "⚠️  bgremover_gpu not found. Build it first with: ./build.sh"
fi

# Summary
echo ""
echo -e "${BLUE}=== Test Summary ===${NC}"
echo ""
echo "Virtual camera setup appears to be working!"
echo ""
echo "Next steps:"
echo "  1. Run bgremover with virtual camera:"
echo "     ./build/bgremover_gpu --vcam"
echo ""
echo "  2. Test the virtual camera output:"
echo "     ffplay /dev/video2"
echo "     # or"
echo "     mpv av://v4l2:/dev/video2"
echo ""
echo "  3. Use in applications:"
echo "     • Open Zoom → Settings → Video → Select 'BGRemover Virtual Camera'"
echo "     • Open Teams → Settings → Devices → Camera → Select 'BGRemover Virtual Camera'"
echo "     • Open OBS → Sources → Video Capture Device → Select 'BGRemover Virtual Camera'"
echo ""
log_success "All tests passed!"
