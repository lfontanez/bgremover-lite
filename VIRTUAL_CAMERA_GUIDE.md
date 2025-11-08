# Virtual Camera Setup and Usage Guide

Complete guide for using BGRemover Lite as a virtual camera in video conferencing applications.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Usage Examples](#usage-examples)
- [Application-Specific Guides](#application-specific-guides)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## Overview

BGRemover Lite can output processed video to a virtual camera device, making it available to any application that uses webcams (Zoom, Teams, OBS, Discord, etc.). This is achieved using the `v4l2loopback` kernel module on Linux.

### How It Works

```
Physical Webcam → BGRemover GPU Processing → Virtual Camera Device → Video Apps
     /dev/video0         (Background Blur)           /dev/video2         (Zoom/Teams/etc)
```

### Key Features

- **Real-time Processing**: 30 FPS GPU-accelerated background removal
- **Universal Compatibility**: Works with any WebRTC-compatible application
- **Low Latency**: Minimal delay between input and output
- **Dual Output**: Display window + virtual camera simultaneously
- **Flexible Configuration**: Custom device paths and settings

## Quick Start

### 1. Install and Configure Virtual Camera

```bash
# Run the automated setup script
./setup_virtual_camera.sh
```

This script will:
- Install v4l2loopback kernel module
- Configure it with WebRTC-compatible settings
- Create `/dev/video2` device
- Set proper permissions
- Configure automatic loading on boot

### 2. Build BGRemover with Virtual Camera Support

```bash
# Build the project (if not already built)
./build.sh

# The virtual camera support is automatically included
```

### 3. Run with Virtual Camera

```bash
# Start with virtual camera enabled
./build/bgremover_gpu --vcam

# Or with custom device
./build/bgremover_gpu --vcam-device /dev/video3
```

### 4. Use in Applications

Open your video conferencing app and select "BGRemover Virtual Camera" from the camera list.

## Detailed Setup

### Prerequisites

- Linux system (Ubuntu 22.04+ recommended)
- Kernel headers installed: `sudo apt install linux-headers-$(uname -r)`
- DKMS installed: `sudo apt install dkms`
- Root/sudo access for module installation

### Manual Installation

If you prefer manual setup or the script doesn't work:

#### Step 1: Install v4l2loopback

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install v4l2loopback-dkms v4l2loopback-utils

# Fedora
sudo dnf install v4l2loopback

# Arch Linux
sudo pacman -S v4l2loopback-dkms
```

#### Step 2: Load the Module

```bash
# Load with WebRTC-compatible settings
sudo modprobe v4l2loopback \
    exclusive_caps=1 \
    video_nr=2 \
    card_label="BGRemover Virtual Camera" \
    max_buffers=2
```

**Parameter Explanation:**
- `exclusive_caps=1`: Required for Chrome/Zoom/Teams WebRTC compatibility
- `video_nr=2`: Creates `/dev/video2` (avoids conflict with built-in webcams)
- `card_label`: Friendly name shown in applications
- `max_buffers=2`: Minimal buffering for low latency

#### Step 3: Set Permissions

```bash
sudo chmod 666 /dev/video2
```

#### Step 4: Make Persistent (Optional)

To load the module automatically on boot:

```bash
# Create modprobe configuration
sudo tee /etc/modprobe.d/v4l2loopback.conf > /dev/null <<EOF
options v4l2loopback exclusive_caps=1 video_nr=2 card_label="BGRemover Virtual Camera" max_buffers=2
EOF

# Add to modules list
echo "v4l2loopback" | sudo tee -a /etc/modules
```

### Verification

```bash
# Check if module is loaded
lsmod | grep v4l2loopback

# Check if device exists
ls -la /dev/video2

# Query device capabilities
v4l2-ctl --device=/dev/video2 --all

# Run automated test
./test_virtual_camera.sh
```

## Usage Examples

### Basic Usage

```bash
# Use default webcam (/dev/video0) and output to virtual camera (/dev/video2)
./build/bgremover_gpu --vcam
```

### Custom Input Source

```bash
# Use video file as input
./build/bgremover_gpu path/to/video.mp4 --vcam

# Use specific webcam device
./build/bgremover_gpu /dev/video1 --vcam
```

### Custom Virtual Camera Device

```bash
# Output to /dev/video3 instead of /dev/video2
./build/bgremover_gpu --vcam-device /dev/video3

# Combined with video file
./build/bgremover_gpu video.mp4 --vcam-device /dev/video3
```

### Testing Virtual Camera Output

```bash
# View with ffplay
ffplay /dev/video2

# View with mpv
mpv av://v4l2:/dev/video2

# View with VLC
vlc v4l2:///dev/video2

# Check format details
v4l2-ctl --device=/dev/video2 --list-formats-ext
```

## Application-Specific Guides

### Zoom

1. Open Zoom
2. Click your profile picture → **Settings**
3. Navigate to **Video** tab
4. Under **Camera**, select **BGRemover Virtual Camera**
5. You should see the processed video with blurred background

**Tips:**
- Disable Zoom's built-in virtual background for best performance
- Enable HD video for better quality
- Turn off "Mirror my video" if needed

### Microsoft Teams

1. Open Microsoft Teams
2. Click your profile picture → **Settings**
3. Navigate to **Devices** section
4. Under **Camera**, select **BGRemover Virtual Camera**
5. Preview should show processed video

**Tips:**
- Disable Teams' background effects
- Enable "HD video" in settings
- Test in a meeting preview first

### Google Meet

1. Open Google Meet
2. Start or join a meeting
3. Click the three dots (More options)
4. Select **Settings**
5. Go to **Video** tab
6. Select **BGRemover Virtual Camera**

**Tips:**
- Works best in Chrome/Chromium browsers
- Disable Meet's built-in background blur
- Check "Send HD video" for better quality

### OBS Studio

1. Open OBS Studio
2. In **Sources** panel, click **+**
3. Select **Video Capture Device**
4. Name it (e.g., "BGRemover Camera")
5. In **Device** dropdown, select **BGRemover Virtual Camera**
6. Click **OK**

**Tips:**
- Set resolution to match your webcam (e.g., 1920x1080)
- Use "Custom" FPS and set to 30
- Add as a scene source for streaming/recording

### Discord

1. Open Discord
2. Click the gear icon (User Settings)
3. Navigate to **Voice & Video**
4. Under **Camera**, select **BGRemover Virtual Camera**
5. Test in video preview

**Tips:**
- Works in both desktop and web versions
- Disable Discord's noise suppression for better performance
- Use "Go Live" feature for screen sharing with camera

### Skype

1. Open Skype
2. Click your profile picture → **Settings**
3. Navigate to **Audio & Video**
4. Under **Camera**, select **BGRemover Virtual Camera**
5. Preview should show processed video

### Browser-Based Apps (WebRTC)

Most modern video conferencing apps use WebRTC and will automatically detect the virtual camera:

- **Jitsi Meet**: Settings → Select camera
- **Whereby**: Camera settings → Select device
- **Google Duo**: Settings → Video → Select camera
- **Facebook Messenger**: Video settings → Select camera

## Troubleshooting

### Device Not Found

**Problem**: `/dev/video2` doesn't exist

**Solutions**:
```bash
# Check if module is loaded
lsmod | grep v4l2loopback

# If not loaded, load it
sudo modprobe v4l2loopback exclusive_caps=1 video_nr=2 card_label="BGRemover Virtual Camera"

# Check for errors
dmesg | grep v4l2loopback

# Verify device creation
ls -la /dev/video*
```

### Permission Denied

**Problem**: Cannot write to `/dev/video2`

**Solutions**:
```bash
# Set permissions
sudo chmod 666 /dev/video2

# Or add user to video group
sudo usermod -a -G video $USER
# Then log out and log back in

# Check current permissions
ls -la /dev/video2
```

### Not Showing in Applications

**Problem**: Virtual camera doesn't appear in app's camera list

**Solutions**:

1. **Check exclusive_caps setting**:
```bash
# Reload module with exclusive_caps=1
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback exclusive_caps=1 video_nr=2 card_label="BGRemover Virtual Camera"
```

2. **Restart the application**: Close and reopen the video app

3. **Check browser permissions** (for web apps):
   - Chrome: Settings → Privacy and security → Site settings → Camera
   - Firefox: Preferences → Privacy & Security → Permissions → Camera

4. **Verify device is accessible**:
```bash
v4l2-ctl --device=/dev/video2 --all
```

### Module Not Loading on Boot

**Problem**: v4l2loopback doesn't load automatically after reboot

**Solutions**:

1. **Check configuration files**:
```bash
# Verify modprobe config
cat /etc/modprobe.d/v4l2loopback.conf

# Verify modules list
cat /etc/modules | grep v4l2loopback
```

2. **Manually add to boot**:
```bash
# Add to modules
echo "v4l2loopback" | sudo tee -a /etc/modules

# Create modprobe config
sudo tee /etc/modprobe.d/v4l2loopback.conf > /dev/null <<EOF
options v4l2loopback exclusive_caps=1 video_nr=2 card_label="BGRemover Virtual Camera" max_buffers=2
EOF

# Update initramfs
sudo update-initramfs -u
```

### Poor Performance / Lag

**Problem**: Virtual camera output is laggy or low FPS

**Solutions**:

1. **Check GPU acceleration**:
```bash
# Verify CUDA is being used
./build/bgremover_gpu --vcam
# Look for "🚀 GPU acceleration enabled!" message
```

2. **Monitor GPU usage**:
```bash
nvidia-smi --loop-ms=1000
# Should show ~20-40% GPU utilization
```

3. **Reduce resolution** (if needed):
   - Lower your webcam resolution in the source app
   - BGRemover will process faster at lower resolutions

4. **Close other GPU applications**:
   - Check for other apps using GPU
   - Close unnecessary browser tabs with hardware acceleration

### Black Screen in Applications

**Problem**: Virtual camera shows black screen in apps

**Solutions**:

1. **Verify BGRemover is running**:
```bash
# Check if process is running
ps aux | grep bgremover_gpu
```

2. **Test virtual camera directly**:
```bash
# Should show video output
ffplay /dev/video2
```

3. **Check device format**:
```bash
v4l2-ctl --device=/dev/video2 --list-formats-ext
# Should show YUYV format
```

4. **Restart BGRemover**:
```bash
# Kill existing process
pkill bgremover_gpu

# Start fresh
./build/bgremover_gpu --vcam
```

### Color Issues

**Problem**: Colors look wrong or washed out

**Solutions**:

1. **Verify format conversion**: The V4L2Output class converts BGR to YUYV
2. **Check application settings**: Some apps have color correction settings
3. **Test with different apps**: Verify if issue is app-specific

## Advanced Configuration

### Multiple Virtual Cameras

Create multiple virtual camera devices:

```bash
# Load module with multiple devices
sudo modprobe v4l2loopback \
    exclusive_caps=1,1 \
    video_nr=2,3 \
    card_label="BGRemover Camera 1","BGRemover Camera 2" \
    max_buffers=2,2

# Use different devices
./build/bgremover_gpu --vcam-device /dev/video2  # Terminal 1
./build/bgremover_gpu --vcam-device /dev/video3  # Terminal 2
```

### Custom Resolution

The virtual camera automatically matches your input resolution. To force a specific resolution:

1. Modify `v4l2_output.hpp` to set custom width/height
2. Or resize your input video before processing

### Format Options

Current implementation uses YUYV (YUV 4:2:2) format for maximum compatibility. Other formats can be added:

- **MJPEG**: Better compression, higher CPU usage
- **RGB24**: No conversion needed, larger bandwidth
- **NV12**: GPU-friendly, requires different conversion

### Performance Tuning

```bash
# Increase buffer size for smoother playback (at cost of latency)
sudo modprobe v4l2loopback exclusive_caps=1 video_nr=2 max_buffers=4

# Decrease for lower latency (may cause frame drops)
sudo modprobe v4l2loopback exclusive_caps=1 video_nr=2 max_buffers=1
```

### Systemd Service (Optional)

Create a systemd service to auto-start BGRemover:

```bash
# Create service file
sudo tee /etc/systemd/system/bgremover.service > /dev/null <<EOF
[Unit]
Description=BGRemover Virtual Camera
After=network.target

[Service]
Type=simple
User=$USER
ExecStart=/path/to/bgremover-lite/build/bgremover_gpu --vcam
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl enable bgremover.service
sudo systemctl start bgremover.service
```

## Technical Details

### V4L2 API

The virtual camera uses the Video4Linux2 (V4L2) API:

- **Device Type**: V4L2_CAP_VIDEO_OUTPUT
- **Pixel Format**: V4L2_PIX_FMT_YUYV (YUV 4:2:2)
- **Buffer Type**: V4L2_BUF_TYPE_VIDEO_OUTPUT
- **Memory**: Userspace buffers (no DMA)

### Color Space Conversion

BGR (OpenCV) → YUYV (V4L2):
```
For each 2 pixels:
  Y1 = 0.299*R1 + 0.587*G1 + 0.114*B1
  Y2 = 0.299*R2 + 0.587*G2 + 0.114*B2
  U = -0.147*R_avg - 0.289*G_avg + 0.436*B_avg + 128
  V = 0.615*R_avg - 0.515*G_avg - 0.100*B_avg + 128
```

### Performance Impact

- **CPU Overhead**: ~2-3ms per frame for format conversion
- **Memory**: Minimal (single frame buffer)
- **Latency**: <5ms additional latency
- **Total Pipeline**: Input → GPU Processing (9ms) → Conversion (2ms) → V4L2 Output (1ms) = ~12ms

## FAQ

**Q: Can I use this on Windows or macOS?**
A: Currently Linux-only. Windows would require DirectShow virtual camera, macOS would require AVFoundation virtual camera.

**Q: Does this work with hardware encoding?**
A: Yes, applications can use hardware encoding (H.264/H.265) on the virtual camera output.

**Q: Can I record the virtual camera output?**
A: Yes, use OBS, ffmpeg, or any recording software that supports V4L2 devices.

**Q: What's the maximum resolution supported?**
A: Limited by your GPU and webcam. Tested up to 1920x1080 @ 30 FPS.

**Q: Can I use multiple physical webcams?**
A: Yes, specify different input devices and output to different virtual cameras.

**Q: Does this work in VMs?**
A: Yes, if the VM has GPU passthrough and access to /dev/video* devices.

## Resources

- [v4l2loopback GitHub](https://github.com/umlaeute/v4l2loopback)
- [V4L2 API Documentation](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/v4l2.html)
- [WebRTC Camera Requirements](https://webrtc.org/getting-started/media-devices)

## Support

If you encounter issues:

1. Run the test script: `./test_virtual_camera.sh`
2. Check logs: `dmesg | grep v4l2loopback`
3. Verify GPU acceleration: `nvidia-smi`
4. Open an issue on GitHub with full error output

---

**Happy video conferencing with GPU-accelerated background removal!** 🚀📹
