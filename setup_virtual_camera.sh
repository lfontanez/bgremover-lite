#!/bin/bash

# Virtual Camera Setup Script for BGRemover Lite
# This script installs and configures v4l2loopback for virtual camera functionality

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log_error "This script should NOT be run as root (it will use sudo when needed)"
   exit 1
fi

log_header "Virtual Camera Setup for BGRemover Lite"

# Step 1: Install v4l2loopback
log_info "Installing v4l2loopback kernel module..."

if command -v apt >/dev/null 2>&1; then
    # Debian/Ubuntu
    sudo apt update
    sudo apt install -y v4l2loopback-dkms v4l2loopback-utils
elif command -v dnf >/dev/null 2>&1; then
    # Fedora
    sudo dnf install -y v4l2loopback
elif command -v pacman >/dev/null 2>&1; then
    # Arch Linux
    sudo pacman -S --noconfirm v4l2loopback-dkms
else
    log_error "Unsupported package manager. Please install v4l2loopback manually."
    exit 1
fi

log_success "v4l2loopback installed"

# Step 2: Remove existing module if loaded
log_info "Removing existing v4l2loopback module (if loaded)..."
sudo modprobe -r v4l2loopback 2>/dev/null || true

# Step 3: Load module with proper configuration
log_info "Loading v4l2loopback module with WebRTC-compatible settings..."

# Configuration:
# - exclusive_caps=1: Required for Chrome/Zoom/Teams WebRTC compatibility
# - video_nr=2: Creates /dev/video2 (avoids conflict with built-in webcams)
# - card_label: Friendly name shown in applications
# - max_buffers=2: Minimal buffering for low latency

sudo modprobe v4l2loopback \
    exclusive_caps=1 \
    video_nr=2 \
    card_label="BGRemover Virtual Camera" \
    max_buffers=2

if [[ $? -eq 0 ]]; then
    log_success "v4l2loopback module loaded successfully"
else
    log_error "Failed to load v4l2loopback module"
    exit 1
fi

# Step 4: Verify device creation
if [[ -e /dev/video2 ]]; then
    log_success "Virtual camera device created: /dev/video2"
else
    log_error "Virtual camera device not found at /dev/video2"
    exit 1
fi

# Step 5: Set proper permissions
log_info "Setting device permissions..."
sudo chmod 666 /dev/video2
log_success "Device permissions set"

# Step 6: Make module load persistent across reboots
log_info "Configuring module to load on boot..."

# Create modprobe configuration
sudo tee /etc/modprobe.d/v4l2loopback.conf > /dev/null <<EOF
# BGRemover Virtual Camera Configuration
# Loaded automatically on boot
options v4l2loopback exclusive_caps=1 video_nr=2 card_label="BGRemover Virtual Camera" max_buffers=2
EOF

# Add to modules to load on boot
if ! grep -q "v4l2loopback" /etc/modules 2>/dev/null; then
    echo "v4l2loopback" | sudo tee -a /etc/modules > /dev/null
    log_success "Module configured to load on boot"
else
    log_info "Module already configured for boot loading"
fi

# Step 7: Verify installation
log_header "Verification"

log_info "Checking v4l2loopback module status..."
if lsmod | grep -q v4l2loopback; then
    log_success "✅ v4l2loopback module is loaded"
else
    log_error "❌ v4l2loopback module is not loaded"
    exit 1
fi

log_info "Checking virtual camera device..."
if v4l2-ctl --device=/dev/video2 --all >/dev/null 2>&1; then
    log_success "✅ Virtual camera device is accessible"
    
    # Show device info
    echo ""
    log_info "Device Information:"
    v4l2-ctl --device=/dev/video2 --info
else
    log_error "❌ Virtual camera device is not accessible"
    exit 1
fi

# Step 8: Test with v4l2-ctl
log_info "Testing device capabilities..."
v4l2-ctl --device=/dev/video2 --list-formats-ext 2>/dev/null || true

# Final summary
log_header "Setup Complete!"

echo ""
echo "Virtual camera is ready to use!"
echo ""
echo "Device: /dev/video2"
echo "Name: BGRemover Virtual Camera"
echo ""
echo "Usage:"
echo "  ./bgremover_gpu --vcam                    # Use default /dev/video2"
echo "  ./bgremover_gpu --vcam-device /dev/video3 # Use custom device"
echo ""
echo "The virtual camera will appear in:"
echo "  • Zoom (Settings → Video → Camera)"
echo "  • Microsoft Teams (Settings → Devices → Camera)"
echo "  • Google Meet (Settings → Video)"
echo "  • OBS Studio (Sources → Video Capture Device)"
echo "  • Discord (User Settings → Voice & Video)"
echo ""
echo "To verify the camera is working:"
echo "  ffplay /dev/video2"
echo "  # or"
echo "  mpv av://v4l2:/dev/video2"
echo ""
log_success "Setup completed successfully!"
