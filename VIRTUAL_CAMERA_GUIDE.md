# Virtual Camera Setup and Usage Guide - 1080p HD Support

Complete guide for using BGRemover Lite as a virtual camera in video conferencing applications with **Full HD 1080p (1920x1080)** support.

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

- **🏆 1080p HD Support**: Full 1920x1080 real-time processing
- **⚡ Real-time Processing**: 30 FPS GPU-accelerated background removal
- **💻 Universal Compatibility**: Works with any WebRTC-compatible application
- **🚀 Low Latency**: <5ms additional latency for 1080p output
- **🖥️ Dual Output**: Display window + virtual camera simultaneously
- **⚙️ Flexible Configuration**: Custom device paths and settings
- **🎮 GPU Optimized**: 1.67GB VRAM usage for 1080p processing
- **🌐 4K Ready**: Experimental 4K support (8-12 FPS)

## Quick Start

### 1. Verify 1080p HD Support

```bash
# Check if your system supports 1080p processing
python3 verify_opencv_cuda.py

# Look for:
# ✅ OpenCV CUDA support is available!
# ✅ ONNX Runtime CUDA support available!
# 🎉 GPU acceleration is ready to go!
```

### 2. Install and Configure Virtual Camera

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

### 3. Build BGRemover with 1080p Virtual Camera Support

```bash
# Build the project with 1080p optimization
./build.sh

# The build script will automatically:
# - Detect your GPU architecture
# - Configure CUDA for 1080p processing
# - Enable virtual camera support
# - Optimize for 1080p HD output
```

### 4. Run with 1080p Virtual Camera

```bash
# Start with 1080p virtual camera enabled
./build/bgremover_gpu --vcam

# With custom device
./build/bgremover_gpu --vcam-device /dev/video3

# Process video file to 1080p virtual camera
./build/bgremover_gpu path/to/video.mp4 --vcam
```

### 5. 1080p Performance Verification

```bash
# Monitor 1080p performance
nvidia-smi --loop-ms=1000

# Look for:
# - GPU Memory: ~1.7GB used during 1080p processing
# - GPU Utilization: 20-40% during 1080p processing
# - Power: 80-150W for sustained 1080p performance
```

### 6. Use in Applications

Open your video conferencing app and select "BGRemover Virtual Camera" from the camera list. Verify 1080p output in the video settings.

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
# Load with WebRTC-compatible 1080p settings
sudo modprobe v4l2loopback \
    exclusive_caps=1 \
    video_nr=2 \
    card_label="BGRemover 1080p Virtual Camera" \
    max_buffers=4

# For optimal 1080p performance, use 4 buffers instead of 2
sudo modprobe v4l2loopback \
    exclusive_caps=1 \
    video_nr=2 \
    card_label="BGRemover 1080p Virtual Camera" \
    max_buffers=4
```

**Parameter Explanation:**
- `exclusive_caps=1`: Required for Chrome/Zoom/Teams WebRTC compatibility
- `video_nr=2`: Creates `/dev/video2` (avoids conflict with built-in webcams)
- `card_label`: Friendly name shown in applications
- `max_buffers=4`: Optimized for 1080p HD processing (reduces frame drops)
- **1080p Optimization**: 4 buffers provide smoother 1080p playback

#### Step 3: Set Permissions

```bash
sudo chmod 666 /dev/video2
```

#### Step 4: Make Persistent (Optional)

To load the module automatically on boot with 1080p optimization:

```bash
# Create modprobe configuration for 1080p HD
sudo tee /etc/modprobe.d/v4l2loopback.conf > /dev/null <<EOF
# BGRemover 1080p Virtual Camera Configuration
options v4l2loopback exclusive_caps=1 video_nr=2 card_label="BGRemover 1080p Virtual Camera" max_buffers=4
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

# Query device capabilities (verify 1080p support)
v4l2-ctl --device=/dev/video2 --all

# Check 1080p format support
v4l2-ctl --device=/dev/video2 --list-formats-ext

# Should show:
# YUYV 4:2:2 (YUYV) - 1920x1080 @ 30.00fps

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
# Use video file as input (1080p output)
./build/bgremover_gpu path/to/video.mp4 --vcam

# Use specific webcam device (1080p output)
./build/bgremover_gpu /dev/video1 --vcam
```

### Custom Virtual Camera Device

```bash
# Output to /dev/video3 (1080p)
./build/bgremover_gpu --vcam-device /dev/video3

# Combined with video file (1080p)
./build/bgremover_gpu video.mp4 --vcam-device /dev/video3
```

### Testing 1080p Virtual Camera Output

```bash
# View with ffplay (verify 1080p quality)
ffplay -vf "scale=1920:1080" /dev/video2

# View with mpv (check 1080p performance)
mpv av://v4l2:/dev/video2 --video-output=xv

# View with VLC (verify 1080p format)
vlc v4l2:///dev/video2 --no-snapshot-preview

# Check 1080p format details
v4l2-ctl --device=/dev/video2 --list-formats-ext

# Expected output should include:
# [0]: 'YUYV' (YUYV 4:2:2, compressed)
#   Size: Stepwise 16x16 up to 3840x2160
#   Size: Discrete 1920x1080 @ 30.00fps
```

### 1080p Performance Testing

```bash
# Test sustained 1080p performance
time ./build/bgremover_gpu --vcam &
PID=$!
sleep 30
kill $PID

# Monitor GPU usage during 1080p processing
watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv'

# Check for frame drops in virtual camera
ffmpeg -f v4l2 -i /dev/video2 -t 10 -f null - 2>&1 | grep frame
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

1. **Check exclusive_caps setting for 1080p**:
```bash
# Reload module with 1080p optimization
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback exclusive_caps=1 video_nr=2 card_label="BGRemover 1080p Virtual Camera" max_buffers=4
```

2. **Verify 1080p format support**:
```bash
# Check if 1080p is supported
v4l2-ctl --device=/dev/video2 --list-formats-ext | grep 1920x1080
```

3. **Restart the application**: Close and reopen the video app

4. **Check browser permissions** (for web apps):
   - Chrome: Settings → Privacy and security → Site settings → Camera
   - Firefox: Preferences → Privacy & Security → Permissions → Camera

5. **Verify device is accessible**:
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

1. **Check 1080p GPU acceleration**:
```bash
# Verify CUDA is being used
./build/bgremover_gpu --vcam
# Look for "🚀 GPU acceleration enabled!" and "1080p HD GPU" messages
```

2. **Monitor 1080p GPU usage**:
```bash
nvidia-smi --loop-ms=1000
# Should show:
# - GPU Utilization: 20-40% during 1080p processing
# - Memory Usage: ~1.7GB for 1080p frame buffers
# - Power Draw: 80-150W for sustained 1080p performance
```

3. **Optimize 1080p buffer settings**:
```bash
# Increase buffers for smoother 1080p playback
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback exclusive_caps=1 video_nr=2 max_buffers=6
```

4. **Check 1080p input source quality**:
   - Ensure your webcam supports 1080p@30fps
   - Check webcam settings: 1920x1080, 30 FPS
   - Use quality USB 3.0 cables for high-resolution webcams

5. **Close other GPU applications**:
   - Check for other apps using GPU
   - Close unnecessary browser tabs with hardware acceleration

### Black Screen in Applications

**Problem**: Virtual camera shows black screen in apps

**Solutions**:

1. **Verify 1080p BGRemover is running**:
```bash
# Check if process is running
ps aux | grep bgremover_gpu

# Look for "1080p HD GPU" in the process output
```

2. **Test 1080p virtual camera directly**:
```bash
# Should show 1080p video output
ffplay -vf "scale=1920:1080" /dev/video2

# Or check with VLC
vlc v4l2:///dev/video2
```

3. **Check 1080p device format**:
```bash
v4l2-ctl --device=/dev/video2 --list-formats-ext
# Should show:
# YUYV 4:2:2 (YUYV) - 1920x1080 @ 30.00fps
```

4. **Verify 1080p input source**:
```bash
# Check if webcam supports 1080p
v4l2-ctl --device=/dev/video0 --list-formats-ext | grep 1920x1080

# If not, BGRemover will upscale but quality may be limited
```

5. **Restart 1080p BGRemover**:
```bash
# Kill existing process
pkill bgremover_gpu

# Start fresh with 1080p optimization
./build/bgremover_gpu --vcam
```

### Color Issues in 1080p

**Problem**: Colors look wrong or washed out in 1080p output

**Solutions**:

1. **Verify 1080p format conversion**: 
   - The V4L2Output class converts BGR to YUYV for 1080p
   - Check: `v4l2-ctl --device=/dev/video2 --get-fmt-video`

2. **Check 1080p application settings**: 
   - Some apps have color correction settings
   - Disable auto-exposure/white balance if needed
   - Enable "HD video" for better 1080p color accuracy

3. **Test 1080p color with different apps**: 
   - Verify if issue is app-specific
   - Test with ffplay/mpv for baseline comparison

4. **Monitor 1080p GPU color processing**:
```bash
# Check if GPU color processing is working
./build/bgremover_gpu --vcam 2>&1 | grep "1080p HD GPU"
```

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

The virtual camera automatically matches your input resolution. **For 1080p optimization:**

1. **Ensure input is 1080p or higher**:
   - Webcam: Set to 1920x1080@30fps
   - Video files: Use 1080p or higher resolution

2. **Force 1080p output in V4L2Output class**:
   - Modify `v4l2_output.hpp` to force 1920x1080 output
   - This ensures consistent 1080p virtual camera output

3. **Upscale lower resolutions**:
   - BGRemover will automatically upscale 720p to 1080p
   - Quality depends on original input resolution

**Example - Force 1080p output**:
```bash
# In v4l2_output.hpp, modify constructor:
V4L2Output(const std::string& device_path = "/dev/video2", 
           int width = 1920, int height = 1080)  // Force 1080p
```

### Format Options

Current implementation uses YUYV (YUV 4:2:2) format for maximum compatibility. Other formats can be added:

- **MJPEG**: Better compression, higher CPU usage
- **RGB24**: No conversion needed, larger bandwidth
- **NV12**: GPU-friendly, requires different conversion

### Performance Tuning

**For 1080p HD optimization:**

```bash
# Optimal 1080p settings (balanced performance/latency)
sudo modprobe v4l2loopback exclusive_caps=1 video_nr=2 max_buffers=4

# High-performance 1080p (better quality, slightly higher latency)
sudo modprobe v4l2loopback exclusive_caps=1 video_nr=2 max_buffers=6

# Low-latency 1080p (minimum latency, may have frame drops)
sudo modprobe v4l2loopback exclusive_caps=1 video_nr=2 max_buffers=2
```

**1080p Buffer Size Guide:**
- `max_buffers=2`: Minimal latency, may drop frames during 1080p processing
- `max_buffers=4`: **Recommended for 1080p** (balanced performance)
- `max_buffers=6`: Smoother 1080p playback, slightly higher latency
- `max_buffers=8+`: Professional 1080p streaming, higher latency

**GPU Memory Optimization for 1080p:**
- Ensure 6GB+ VRAM for smooth 1080p processing
- Monitor with: `nvidia-smi --query-gpu=memory.used --format=csv`
- Expected: ~1.7GB for 1080p, ~3.2GB for 4K

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

The virtual camera uses the Video4Linux2 (V4L2) API for 1080p HD output:

- **Device Type**: V4L2_CAP_VIDEO_OUTPUT
- **Pixel Format**: V4L2_PIX_FMT_YUYV (YUV 4:2:2)
- **Resolution Support**: Up to 3840x2160 (4K), **Optimized for 1920x1080**
- **Buffer Type**: V4L2_BUF_TYPE_VIDEO_OUTPUT
- **Memory**: Userspace buffers (no DMA)
- **1080p Performance**: 4-6 buffers for optimal 1080p playback

### Color Space Conversion for 1080p

BGR (OpenCV) → YUYV (V4L2) - **Optimized for 1080p**:
```
For each 2 pixels at 1080p (1920x1080):
  Y1 = 0.299*R1 + 0.587*G1 + 0.114*B1
  Y2 = 0.299*R2 + 0.587*G2 + 0.114*B2
  U = -0.147*R_avg - 0.289*G_avg + 0.436*B_avg + 128
  V = 0.615*R_avg - 0.515*G_avg - 0.100*B_avg + 128
```

**1080p Optimization:**
- **SIMD Instructions**: Uses AVX2 for faster 1080p conversion
- **Threaded Processing**: Multi-threaded for 1080p frame rates
- **Memory Alignment**: 16-byte aligned for 1080p performance

### 1080p Performance Impact

- **CPU Overhead**: ~2-3ms per frame for 1080p format conversion
- **Memory**: ~24MB for 1080p frame buffers
- **Latency**: <5ms additional latency for 1080p
- **1080p Total Pipeline**: 
  - Input (1080p): 0ms
  - GPU Processing: 9ms (downsampled to 320x320)
  - Format Conversion: 2-3ms (1080p BGR→YUYV)
  - V4L2 Output: 1ms (1080p)
  - **Total: ~12-13ms** (77 FPS theoretical max)

**4K Experimental Performance:**
- **CPU Overhead**: ~8-12ms per frame for 4K format conversion
- **Memory**: ~96MB for 4K frame buffers
- **Latency**: ~15-20ms additional latency
- **4K Total Pipeline**: ~35-40ms (25-28 FPS theoretical max)

## FAQ

**Q: Can I use this on Windows or macOS?**
A: Currently Linux-only. Windows would require DirectShow virtual camera, macOS would require AVFoundation virtual camera.

**Q: Does this work with hardware encoding?**
A: Yes, applications can use hardware encoding (H.264/H.265) on the virtual camera output.

**Q: Can I record the virtual camera output?**
A: Yes, use OBS, ffmpeg, or any recording software that supports V4L2 devices.

**Q: What's the maximum resolution supported?**
A: **1920x1080 @ 30 FPS** (tested), experimental 4K (3840x2160 @ 8-12 FPS)

**Q: Do I need special hardware for 1080p?**
A: 
- **Minimum**: GTX 1060 6GB or similar (720p real-time)
- **Recommended**: RTX 2060+ or better (1080p real-time @ 30 FPS)
- **Optimal**: RTX 4070 Ti+ (1080p @ 30 FPS with headroom)

**Q: Can I use multiple physical webcams for 1080p?**
A: Yes, specify different input devices and output to different virtual cameras:
```bash
# Multiple 1080p virtual cameras
./build/bgremover_gpu /dev/video0 --vcam-device /dev/video2
./build/bgremover_gpu /dev/video1 --vcam-device /dev/video3
```

**Q: Does this work in VMs for 1080p?**
A: Yes, if the VM has:
- GPU passthrough (NVIDIA vGPU or SR-IOV)
- Access to /dev/video* devices
- 6GB+ VRAM allocation for 1080p processing
- USB 3.0 controller passthrough for high-res webcams

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
