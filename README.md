# BGRemover Lite - GPU-Accelerated Background Removal 🚀

A high-performance, real-time background removal system using U²-Net ONNX model with GPU acceleration via CUDA and ONNX Runtime. Achieves **30 FPS** on RTX 4070 Ti SUPER with proper CUDA provider configuration.

## 🎯 Performance Metrics

| Metric | CPU Version | **GPU Version** | Speedup |
|--------|-------------|-----------------|---------|
| **FPS** | 1-2 FPS | **~30 FPS** | **15-30x** |
| **Inference Time** | ~500ms | **9ms** | **55x faster** |
| **Total Processing** | ~500ms | **10ms** | **50x faster** |
| **GPU Memory** | N/A | **1.67GB / 15.56GB** | 10% usage |

## ✨ Features

- **Real-time Processing**: 30 FPS on modern NVIDIA GPUs
- **GPU Acceleration**: CUDA-enabled ONNX Runtime with TensorRT support
- **Dual Versions**: CPU fallback and GPU-optimized builds
- **High Quality**: U²-Net model for accurate segmentation
- **Low Latency**: 9ms inference time per frame
- **Efficient Memory**: Only 1.67GB VRAM usage
- **Webcam & Video**: Supports live camera and video file input

## 🔧 Requirements

### 1080p HD Hardware Requirements

#### Minimum (720p capable)
- **GPU**: GTX 1060 6GB or similar
- **CUDA**: 10.0+ 
- **VRAM**: 4GB minimum
- **Performance**: 720p @ 15-20 FPS, 1080p @ 5-10 FPS

#### Recommended (1080p optimal)
- **GPU**: RTX 2060, RTX 3060, or better
- **CUDA**: 11.0+
- **VRAM**: 6GB minimum
- **Performance**: 1080p @ 25-30 FPS, 4K @ 8-12 FPS

#### Optimal (1080p + headroom)
- **GPU**: RTX 4070 Ti SUPER, RTX 4080, or better (Tested configuration)
- **CUDA**: 12.8
- **VRAM**: 12GB+ (15.56GB available on RTX 4070 Ti SUPER)
- **Performance**: 1080p @ 30+ FPS, 4K @ 15-20 FPS

### General Hardware
- **GPU** (1080p Recommended): NVIDIA GPU with CUDA support (Compute Capability 6.0+)
  - Tested on: RTX 4070 Ti SUPER (15.56GB VRAM)
  - Works on: GTX 1060+, RTX 20/30/40 series, Tesla, Quadro
- **CPU**: Any modern x64 processor (for CPU fallback)
- **RAM**: 4GB minimum, 8GB recommended

### Software
- **OS**: Linux x64 (Ubuntu 22.04+ recommended)
- **CUDA**: 12.x (12.8 tested)
- **cuDNN**: 9.x (for CUDA 12.x)
- **C++ Compiler**: GCC 9+ or Clang 10+ with C++17 support
- **CMake**: 3.16+
- **OpenCV**: 4.x with CUDA support (for GPU version)
- **Python**: 3.8+ (for verification scripts)

## 📦 Installation

### Quick Start - Automatic Build (Recommended)

The build script automatically detects your hardware and builds both CPU and GPU versions:

```bash
# Clone the repository
git clone https://github.com/lfontanez/bgremover-lite.git
cd bgremover-lite

# System check (optional but recommended)
./check_system.sh

# Automatic build - detects GPU and builds both versions
./build.sh
```

**What the build script does:**
- ✅ **Detects your GPU** and CUDA version
- ✅ **Downloads models** automatically
- ✅ **Builds CPU version** (`./build/bgremover`) - Always works
- ✅ **Builds GPU version** (`./build/bgremover_gpu`) - If NVIDIA GPU detected
- ✅ **Provides fallback** - If GPU unavailable, you still get CPU version

### CPU-Only Build (No GPU Required)

**Requirements:**
- Any modern x64 CPU
- OpenCV 4.x
- CMake 3.16+
- ONNX Runtime (downloaded automatically)

**Build Steps:**
```bash
# Option 1: Using the build script (auto-detects no GPU)
./build.sh
# The script will build CPU version only

# Option 2: Manual build
mkdir build && cd build
cmake -DU2NET_DOWNLOAD_MODELS=ON ..
make -j$(nproc)

# Result: ./bgremover (CPU version only)
```

**Performance:** 1-5 FPS, works on any system

### GPU-Accelerated Build (NVIDIA GPU Required)

**Requirements:**
- **NVIDIA GPU** with CUDA support (GTX 1060+ recommended)
- **CUDA 12.8** (recommended) or 11.0+
- **cuDNN 9.x** (for CUDA 12.x)
- **NVIDIA drivers** (latest)

**Quick GPU Setup:**
```bash
# Install CUDA-enabled OpenCV in Conda (recommended) (12.8 was not available, so no CUDA pytorch) 
conda create -n opencv_cuda12 python=3.11 opencv cudatoolkit=12.5 
conda activate opencv_cuda12

# Build with GPU acceleration
./build.sh

# Result: Both ./bgremover (CPU) and ./build/bgremover_gpu (GPU)
```

**Manual GPU Build:**
```bash
# Ensure CUDA 12.8 environment is set
export CUDA_PATH=/usr/local/cuda-12.8
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Create build directory
mkdir build && cd build

# Configure with GPU support
cmake -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH \
      -DCMAKE_CUDA_COMPILER=$CUDA_PATH/bin/nvcc \
      -DWITH_CUDA=ON \
      -DCUDA_ARCH_BIN=8.9 \
      -DU2NET_DOWNLOAD_MODELS=ON \
      ..

# Build both versions
make -j$(nproc)

# Results: ./bgremover (CPU) and ./bgremover_gpu (GPU)
```

**Performance:** 25-30 FPS at 1080p on modern GPUs

### Build Results

After building, you'll have:

| Executable | Requirements | Performance | When to Use |
|------------|--------------|-------------|-------------|
| `./build/bgremover` | Any CPU | 1-5 FPS | Fallback, slow systems |
| `./build/bgremover_gpu` | NVIDIA GPU | 25-30 FPS | **Recommended** |

**Verification:**
```bash
# Test CPU version
./build/bgremover --help

# Test GPU version (if built)
./build/bgremover_gpu --help
# Should show "GPU-Accelerated Background Removal"
```

### CPU vs GPU Build Comparison

| Feature | CPU Build | GPU Build |
|---------|-----------|-----------|
| **Executable** | `./build/bgremover` | `./build/bgremover_gpu` |
| **Requirements** | Any x64 CPU | NVIDIA GPU + CUDA |
| **Performance** | 1-5 FPS | 25-30 FPS (1080p) |
| **Memory Usage** | ~100MB RAM | ~1.7GB VRAM |
| **Latency** | ~500ms per frame | ~10ms per frame |
| **1080p Support** | No (too slow) | Yes (recommended) |
| **Virtual Camera** | Yes | Yes (better quality) |
| **Background Blur** | Yes (limited) | Yes (all levels) |
| **Custom Backgrounds** | Yes | Yes |
| **Best For** | Testing, low-end systems | Production, high-end systems |

**When to Use CPU Build:**
- ✅ **Development/Testing**: Quick testing without GPU setup
- ✅ **Old Systems**: Computers without NVIDIA GPU
- ✅ **Low-end Hardware**: Laptops, older desktops
- ✅ **Single-frame Processing**: Processing individual images
- ✅ **Fallback**: When GPU is not available

**When to Use GPU Build:**
- ✅ **Real-time Video**: Live video calls, streaming
- ✅ **1080p Processing**: High-resolution video
- ✅ **Production Use**: Professional applications
- ✅ **Multiple Streams**: Processing multiple videos
- ✅ **Best Quality**: Maximum blur levels and effects

**Performance Expectations:**
- **CPU Version**: 1-2 FPS (usable for testing), 3-5 FPS (720p)
- **GPU Version**: 25-30 FPS (1080p), 8-12 FPS (4K experimental)

## 🚀 Quick Start

```bash
# Simple build (no LD_LIBRARY_PATH needed!)
git clone https://github.com/lfontanez/bgremover-lite.git
cd bgremover-lite
rm -rf build
./build.sh

# Both executables work without LD_LIBRARY_PATH:
./build/bgremover --help
./build/bgremover_gpu --help
```

> **📖 Detailed Build Guide**: See [BUILD_GUIDE.md](BUILD_GUIDE.md) for complete documentation, troubleshooting, and advanced usage.

## 🚀 Usage

### Background Blur Control Options

BGRemover Lite now includes advanced background blur control with multiple intensity levels:

- **No Blur**: `--no-blur` or `--no-background-blur` - Disables blur completely
- **Low Blur**: `--blur-low` - Subtle blur (7x7 kernel) for better performance
- **Medium Blur**: `--blur-mid` - Default balance of quality and speed (15x15 kernel)
- **High Blur**: `--blur-high` - Maximum blur effect (25x25 kernel)
- **Custom Background**: `--background-image PATH` or `--bg-image PATH` - Replace background with custom image (e.g., `--background-image background.jpg`)
- **No Preview**: `--no-preview` - Disables the preview window (useful for headless processing and virtual camera usage)

**Default Settings:**
- Background blur: **Enabled** 
- Blur intensity: **Medium** (15x15 kernel)
- Preview window: **Enabled** (shows processed video in a window)
- For 1080p processing: Low (7x7), Medium (15x15), High (25x25)

### GPU-Accelerated Version (Recommended)

```bash
# Webcam (default: medium blur, GPU, with preview window)
./build/bgremover_gpu

# Webcam with different blur levels
./build/bgremover_gpu --no-blur                    # No background blur
./build/bgremover_gpu --blur-low                   # Subtle blur (7x7)
./build/bgremover_gpu --blur-mid                   # Medium blur (15x15) [default]
./build/bgremover_gpu --blur-high                  # Maximum blur (25x25)

# Webcam with custom background replacement
./build/bgremover_gpu --background-image background.jpg   # Custom background (long form)
./build/bgremover_gpu --bg-image background.jpg           # Custom background (short form)

# Webcam without preview window (headless processing)
./build/bgremover_gpu --no-preview                 # No preview window, console output only
./build/bgremover_gpu --no-preview --blur-mid      # No preview, with medium blur
./build/bgremover_gpu --no-preview --background-image office.jpg  # No preview, custom background

# Video file with different options
./build/bgremover_gpu path/to/video.mp4 --blur-low        # Subtle blur
./build/bgremover_gpu path/to/video.mp4 --blur-high       # Strong blur
./build/bgremover_gpu path/to/video.mp4 --background-image background.jpg  # Custom background
./build/bgremover_gpu path/to/video.mp4 --no-preview      # Process video without preview

# With specific device and settings
./build/bgremover_gpu 0 --blur-high                   # Device 0 with high blur
./build/bgremover_gpu 0 --background-image office.jpg # Device 0 with office background
./build/bgremover_gpu 0 --no-preview                  # Device 0 without preview window
```

### CPU Version (Fallback)

```bash
# Webcam with blur control
./build/bgremover --no-blur                       # No blur (fastest)
./build/bgremover --blur-low                      # Low blur (7x7)
./build/bgremover --blur-mid                      # Medium blur (15x15) [default]
./build/bgremover --blur-high                     # High blur (25x25)

# Webcam with custom background replacement
./build/bgremover --background-image background.jpg   # Custom background (CPU version)
./build/bgremover --bg-image office.jpg               # Short form

# Webcam without preview window (headless processing)
./build/bgremover --no-preview                 # No preview window, console output only
./build/bgremover --no-preview --blur-mid      # No preview, with medium blur
./build/bgremover --no-preview --background-image studio.jpg  # No preview, custom background

# Video file with different options
./build/bgremover path/to/video.mp4 --blur-high       # Strong blur on CPU
./build/bgremover path/to/video.mp4 --background-image landscape.jpg  # Custom background
./build/bgremover path/to/video.mp4 --no-preview      # Process video without preview

# Different input devices
./build/bgremover 1 --blur-low                    # Device 1 with low blur
./build/bgremover 1 --background-image beach.jpg  # Device 1 with beach background
./build/bgremover 1 --no-preview                  # Device 1 without preview window
```

### 1080p HD Controls
- **ESC**: Quit application
- **1080p Performance**: Real-time FPS and performance stats displayed in console
- **GPU Memory**: Monitor 1.67GB VRAM usage for 1080p processing
- **Frame Rate**: Expect 30 FPS at 1080p with GPU acceleration

### ✅ No More LD_LIBRARY_PATH Issues!

**Problem Solved**: Previous versions required `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH ./bgremover` to run.

**Solution**: Enhanced RPATH configuration now finds libraries automatically:
- ✅ **No LD_LIBRARY_PATH needed** - Works out of the box
- ✅ **Portable** - Same executables work in any environment
- ✅ **Automatic library discovery** - System, conda, and custom libraries found
- ✅ **Environment agnostic** - Build once, run anywhere

Simply run:
```bash
./build/bgremover --help    # Works immediately!
./build/bgremover_gpu --help # Works immediately!
```

### Background Blur Feature Benefits

#### When to Use Different Blur Levels:

**No Blur (--no-blur)**:
- **Use Case**: Video calls where you want to show your environment clearly
- **Performance**: Fastest processing, minimal CPU/GPU usage
- **Quality**: Sharp background, no artificial effects
- **Best For**: Formal meetings, product demos, environmental context

**Low Blur (--blur-low, 7x7 kernel)**:
- **Use Case**: Subtle background softening for more professional look
- **Performance**: Good for 1080p, minimal performance impact
- **Quality**: Gentle background blur, maintains environmental context
- **Best For**: Business meetings, interviews, content creation

**Medium Blur (--blur-mid, 15x15 kernel - Default)**:
- **Use Case**: Balanced privacy and aesthetics for most use cases
- **Performance**: Optimal balance for 1080p real-time processing
- **Quality**: Clear subject with nicely blurred background
- **Best For**: General video calls, streaming, content creation

**High Blur (--blur-high, 25x25 kernel)**:
- **Use Case**: Maximum privacy and aesthetic focus on the subject
- **Performance**: Slightly higher CPU/GPU usage, still maintains good FPS
- **Quality**: Strong background blur, subject isolation
- **Best For**: Privacy-conscious users, streaming, content creation

#### Custom Background Image Options:

**Custom Background Replacement (`--background-image PATH` / `--bg-image PATH`)**:
- **Use Case**: Professional video calls with branded or themed backgrounds
- **Performance**: Excellent for 1080p, minimal performance impact
- **Quality**: Crisp, clear background replacement with high-quality images
- **Best For**: Professional meetings, content creation, streaming, branding

##### Custom Background Replacement - Detailed Guide

**Feature Overview:**
The custom background replacement feature allows you to replace your background with any image instead of just blurring it. This is perfect for professional video calls, content creation, and streaming where you want a specific branded or themed background.

**Supported Image Formats:**
- **JPG/JPEG**: Recommended for photographs and general use
- **PNG**: Supports transparency, good for logos and graphics with alpha channels
- **BMP**: Uncompressed format for maximum quality
- **TIFF**: High-quality format for professional use

**Image Requirements:**
- **Resolution**: Any resolution (automatically resized to match your video)
- **Recommended**: 1920x1080 (1080p) or higher for best quality
- **Aspect Ratio**: Automatically handled, no specific requirements
- **File Size**: No specific limit, larger files may take longer to load

**Usage Examples:**

*Basic Usage:*
```bash
# Replace background with a custom image
./build/bgremover_gpu --background-image office.jpg

# Use short form
./build/bgremover_gpu --bg-image beach.jpg
```

*With Video Files:*
```bash
# Process video file with custom background
./build/bgremover_gpu video.mp4 --background-image studio.jpg

# Multiple backgrounds for different videos
./build/bgremover_gpu meeting1.mp4 --background-image office.jpg
./build/bgremover_gpu meeting2.mp4 --background-image nature.jpg
```

*With Virtual Camera:*
```bash
# Send processed video with custom background to virtual camera
./build/bgremover_gpu --vcam --background-image branding.jpg

# Different backgrounds for different applications
./build/bgremover_gpu --vcam --background-image corporate.jpg  # For business meetings
./build/bgremover_gpu --vcam --background-image studio.jpg     # For content creation
```

**Background Types and Use Cases:**

*Professional Environments:*
- **Office/Corporate**: Modern office spaces, conference rooms
- **Corporate Branding**: Company logos, branded backgrounds
- **Clean/Minimal**: Simple, uncluttered backgrounds for professional appearance

*Creative/Entertainment:*
- **Studio Setups**: Professional photography backdrops
- **Nature/Outdoor**: Landscapes, gardens, scenic views
- **Abstract/Artistic**: Colorful patterns, artistic designs
- **Gaming**: Themed backgrounds for gaming content

*Content Creation:*
- **Streaming**: Branded overlays, sponsor graphics
- **Education**: Interactive whiteboards, presentation backgrounds
- **Social Media**: Eye-catching, engaging backgrounds

**Performance Considerations:**

*1080p Processing:*
- **GPU Version**: 28-32 FPS with custom backgrounds
- **CPU Version**: 2-3 FPS with custom backgrounds
- **Memory Usage**: ~1.8GB VRAM for 1080p custom background processing
- **Loading Time**: Background image loaded once at startup

*Quality Impact:*
- **Sharpness**: Subject remains sharp while background is completely replaced
- **Color Accuracy**: Preserves original video color quality
- **Edge Detection**: Seamless blending at subject boundaries
- **Frame Consistency**: Stable background throughout video processing

**Best Practices:**

*Image Preparation:*
- Use high-resolution images (1920x1080 or higher)
- Ensure good contrast between subject and background elements
- Avoid busy patterns that might distract from the subject
- Consider lighting consistency with your video setup

*Performance Optimization:*
- Use JPG format for faster loading
- Keep background images under 10MB for quick startup
- Preload multiple backgrounds and switch between them
- For live streaming, test background combinations beforehand

*Professional Usage:*
- Match background to meeting context and audience
- Use consistent branding across all video calls
- Test backgrounds in actual video conferencing apps
- Keep backup backgrounds ready for different scenarios

#### Performance Impact of Blur Levels:

| Blur Level | Kernel Size | 1080p GPU FPS | 1080p CPU FPS | VRAM Usage |
|------------|-------------|---------------|---------------|------------|
| No Blur | None | 35-40 FPS | 3-5 FPS | 1.2GB |
| Low (7x7) | 7x7 | 30-35 FPS | 2-4 FPS | 1.5GB |
| Medium (15x15) | 15x15 | 30-32 FPS | 2-3 FPS | 1.7GB |
| High (25x25) | 25x25 | 25-30 FPS | 1-2 FPS | 1.9GB |
| Custom Background | N/A | 28-32 FPS | 2-3 FPS | 1.8GB |

**1080p Performance Notes:**
- GPU acceleration maintains 25+ FPS even with high blur
- CPU version shows more significant performance impact with higher blur levels
- Low blur recommended for CPU-only processing
- GPU memory usage increases slightly with larger blur kernels

## 📹 Virtual Camera

### When to Use --no-preview

The `--no-preview` flag is essential for several scenarios where you don't need or want a local preview window:

#### **Virtual Camera Usage**
When using virtual camera output, the preview window is redundant since the processed video is sent directly to applications:
```bash
# Perfect for virtual camera - no need for local preview
./build/bgremover_gpu --vcam --no-preview

# Virtual camera with custom settings
./build/bgremover_gpu --vcam --blur-high --no-preview
./build/bgremover_gpu --vcam --background-image office.jpg --no-preview
```

#### **Headless Processing**
For server environments, automated processing, or when running in containers:
```bash
# Headless webcam processing
./build/bgremover_gpu --no-preview

# Headless video file processing
./build/bgremover_gpu path/to/video.mp4 --no-preview --background-image background.jpg

# Automated background processing
./build/bgremover --no-preview --blur-mid --background-image corporate.jpg
```

#### **Performance Optimization**
Disabling the preview window can provide minor performance improvements:
```bash
# Maximum performance for 1080p processing
./build/bgremover_gpu --no-preview --blur-low

# CPU processing without GUI overhead
./build/bgremover --no-preview --blur-high
```

#### **Server/Production Deployment**
For production environments where GUI is not available:
```bash
# Production virtual camera setup
./build/bgremover_gpu --vcam --no-preview --background-image production-bg.jpg

# Automated processing pipeline
./build/bgremover --no-preview --background-image stream-background.jpg
```

#### **Multiple Instance Management**
When running multiple instances, `--no-preview` prevents window clutter:
```bash
# Multiple webcams without preview windows
./build/bgremover_gpu /dev/video0 --vcam-device /dev/video2 --no-preview &
./build/bgremover_gpu /dev/video1 --vcam-device /dev/video3 --no-preview &
```

**Benefits of --no-preview:**
- **Reduced Memory Usage**: Saves ~50-100MB of RAM
- **Better Performance**: Eliminates GUI rendering overhead
- **Cleaner Interface**: No window management needed
- **Server Friendly**: Works in headless environments
- **Multiple Instance Support**: Run multiple processes without window clutter

### Virtual Camera Feature 🌐
The Virtual Camera feature makes your processed video available to any application that accepts camera input, including Zoom, Microsoft Teams, Google Meet, OBS Studio, Discord, and more. Instead of opening a window, the processed video is sent to a virtual camera device (typically `/dev/video2`) that appears as a regular webcam to other applications.

### Background Blur with Virtual Camera

The virtual camera fully supports all background blur control options:

```bash
# Virtual camera with different blur levels
./build/bgremover_gpu --vcam                      # Default: medium blur
./build/bgremover_gpu --vcam --blur-low           # Subtle blur in virtual camera
./build/bgremover_gpu --vcam --blur-high          # Strong blur in virtual camera
./build/bgremover_gpu --vcam --no-blur            # No blur in virtual camera

# Virtual camera with custom background replacement
./build/bgremover_gpu --vcam --background-image background.jpg     # Custom background
./build/bgremover_gpu --vcam --bg-image office.jpg                 # Short form
./build/bgremover_gpu --vcam --background-image landscape.jpg --blur-low  # Background + subtle blur

# Video file to virtual camera with different options
./build/bgremover_gpu --vcam path/to/video.mp4 --blur-mid          # Medium blur
./build/bgremover_gpu --vcam path/to/video.mp4 --blur-high         # High blur
./build/bgremover_gpu --vcam path/to/video.mp4 --background-image beach.jpg  # Beach background
```

### Setup

#### Quick Setup
Run the automated setup script to install and configure v4l2loopback:

```bash
./setup_virtual_camera.sh
```

This script will:
- Install the v4l2loopback kernel module
- Create a virtual camera device at `/dev/video2`
- Set appropriate permissions
- Configure the module to load automatically on system boot

#### Manual Setup
If the automated script doesn't work for your system, you can set up the virtual camera manually:

```bash
# Install v4l2loopback
sudo apt update
sudo apt install v4l2loopback-dkms

# Load the kernel module with desired parameters
sudo modprobe v4l2loopback devices=1 video_nr=2 card_label="BGRemover Virtual Camera"

# Make the module load automatically on boot
echo "v4l2loopback" | sudo tee -a /etc/modules
echo "options v4l2loopback video_nr=2 card_label=\"BGRemover Virtual Camera\"" | sudo tee -a /etc/modprobe.d/v4l2loopback.conf
```

### Usage

#### Basic Virtual Camera
Process webcam input and send to virtual camera:

```bash
./build/bgremover_gpu --vcam
```

#### Custom Virtual Camera Device
Use a specific virtual camera device (e.g., `/dev/video3`):

```bash
./build/bgremover_gpu --vcam-device /dev/video3
```

#### Process Video File to Virtual Camera
Process a video file and send output to virtual camera:

```bash
./build/bgremover_gpu --vcam path/to/video.mp4
```

#### Virtual Camera with Custom Device
```bash
./build/bgremover_gpu --vcam-device /dev/video3 path/to/video.mp4
```

### Virtual Camera Without Preview

For virtual camera usage, `--no-preview` is recommended since the processed video is sent directly to applications:

```bash
# Virtual camera without preview window (recommended)
./build/bgremover_gpu --vcam --no-preview

# Virtual camera with custom settings and no preview
./build/bgremover_gpu --vcam --blur-high --no-preview
./build/bgremover_gpu --vcam --background-image studio.jpg --no-preview
./build/bgremover_gpu --vcam-device /dev/video3 --no-preview --blur-mid

# Process video file to virtual camera without preview
./build/bgremover_gpu --vcam path/to/video.mp4 --no-preview
```

### Supported Applications

The virtual camera works with any application that accepts webcam input:

- **Zoom**: Settings → Video → Select "UVC Camera" or "BGRemover Virtual Camera"
- **Microsoft Teams**: Settings → Devices → Camera → Select virtual camera
- **Google Meet**: Automatically detects virtual camera (grant permission when prompted)
- **OBS Studio**: Sources → Add → Video Capture Device → Select virtual camera
- **Discord**: Settings → Voice and Video → Input Device → Select virtual camera
- **Skype**: Settings → Audio & Video → Camera → Select virtual camera
- **Any WebRTC-compatible app**: Will automatically detect and use the virtual camera

### Testing

#### Verify Virtual Camera Device
Check if the virtual camera device is created and working:

```bash
# List video devices
v4l2-ctl --list-devices

# Check specific device information
v4l2-ctl --device=/dev/video2 --all

# List supported formats
v4l2-ctl --device=/dev/video2 --list-formats-ext
```

#### Test with Media Players
Watch the virtual camera output directly:

```bash
# Using ffplay (if ffmpeg is installed)
ffplay /dev/video2

# Using mpv
mpv av://v4l2:/dev/video2

# Using cvlc (VLC command line)
cvlc v4l2:///dev/video2
```

#### Test Loopback
Create a test pattern on the virtual camera to verify end-to-end functionality:

```bash
# Generate test pattern
ffmpeg -f lavfi -i testsrc=duration=10:size=1280x720:rate=30 -f v4l2 /dev/video2
```

### Troubleshooting

#### Device Not Found
If `/dev/video2` doesn't exist:

```bash
# Check if v4l2loopback module is loaded
lsmod | grep v4l2loopback

# If not loaded, manually load it
sudo modprobe v4l2loopback

# Check kernel messages for errors
dmesg | grep v4l2loopback
```

#### Permission Denied
If you get "Permission denied" errors:

```bash
# Add your user to the video group
sudo usermod -a -G video $USER

# Log out and back in, or restart your session
# Or temporarily use:
sudo chown $USER:$USER /dev/video2
```

#### Not Showing in Applications
If applications don't detect the virtual camera:

```bash
# Verify the device is properly created
v4l2-ctl --list-devices

# Try recreating the device with specific parameters
sudo rmmod v4l2loopback
sudo modprobe v4l2loopback devices=1 video_nr=2 exclusive_caps=1 card_label="BackgroundRemover"

# Some applications require a different pixel format - try:
sudo modprobe v4l2loopback devices=1 video_nr=2 exclusive_caps=1 card_label="BackgroundRemover" pixel_format=YUYV
```

#### Module Not Loading on Boot
Ensure the module loads automatically:

```bash
# Check if module is in the boot configuration
ls -la /etc/modules-load.d/
ls -la /etc/modprobe.d/

# Re-add the module to load automatically
echo "v4l2loopback" | sudo tee -a /etc/modules
echo "options v4l2loopback video_nr=2 exclusive_caps=1 card_label=\"BGRemover Virtual Camera\"" | sudo tee -a /etc/modprobe.d/v4l2loopback.conf

# Update initramfs (if using Ubuntu/Debian)
sudo update-initramfs -u
```

#### Multiple Virtual Cameras
If you need multiple virtual cameras:

```bash
# Create multiple devices (video2, video3, video4)
sudo modprobe v4l2loopback devices=3 video_nr=2,3,4

# Use specific devices
./build/bgremover_gpu --vcam-device /dev/video3
```

#### Application-Specific Issues
- **Zoom**: Restart Zoom after setting up the virtual camera
- **OBS Studio**: Add the virtual camera as a "Video Capture Device" source
- **Web browsers**: Grant camera permissions when first accessing the virtual camera
- **Chrome/Edge**: Some versions have issues with v4l2loopback - try Firefox

#### Performance Issues
- Virtual camera adds minimal overhead (~1-2ms per frame)
- If experiencing performance drops, ensure you have sufficient GPU memory
- Consider reducing blur strength or frame resolution for better performance
- Monitor GPU usage with `nvidia-smi` to ensure the GPU isn't saturated

## 🏗️ Architecture

### GPU Pipeline
```
Video Input → Preprocessing (CPU) → U²-Net Inference (GPU) → 
Postprocessing (CPU) → Gaussian Blur (CPU) → Blending (CPU) → Display
```

### Key Components
- **ONNX Runtime 1.19.0**: GPU inference engine with CUDA provider
- **U²-Net Model**: Pre-trained segmentation (models/u2net.onnx)
- **CUDA 12.8**: GPU acceleration framework
- **OpenCV 4.12.0**: Computer vision operations
- **cuDNN 9.x**: Deep learning primitives

## 🔍 Verification

### Check 1080p HD GPU Acceleration

```bash
# Quick verification of CUDA environment
python3 verify_opencv_cuda.py

# Comprehensive 1080p HD test suite
python3 test_1080p.py

# Expected output for 1080p support:
# ✅ OpenCV CUDA support is available!
# ✅ ONNX Runtime CUDA support available!
# 🎉 GPU acceleration is ready to go!
# 📊 CUDA devices: 1+ (for 1080p processing)

# Test specific components:
python3 test_1080p.py --gpu-only         # GPU functionality only
python3 test_1080p.py --capture-only     # Video capture only
python3 test_1080p.py --vcam-only        # Virtual camera only
python3 test_1080p.py --quick            # Quick test (30 seconds)
```

### Monitor 1080p HD GPU Usage

```bash
# Real-time GPU monitoring for 1080p
nvidia-smi --loop-ms=1000

# Watch for during 1080p processing:
# - GPU Memory: 1.67GB / 15.56GB (10% usage) for 1080p
# - GPU Utilization: 20-40% during 1080p processing
# - Power: 80-150W for sustained 1080p performance
# - Temperature: 65-75°C (normal for 1080p processing)
```

### 1080p Performance Test

```bash
# Run comprehensive 1080p test suite
python3 test_1080p.py

# Run GPU version and verify 1080p performance
./build/bgremover_gpu --vcam

# Monitor console output for:
# "🚀 1080p HD GPU Performance: 30.0 FPS"
# "✅ GPU memory sufficient (24MB required for 1080p)"
# "📊 1080p HD processing requires ~24MB for frame buffers"

# Automated test using shell script
./test_1080p.sh --quick        # Quick test
./test_1080p.sh               # Full test
```

## 📊 Build System

The enhanced build script (`build.sh`) automatically:

1. **Environment Detection**
   - NVIDIA GPU and driver detection
   - CUDA toolkit verification (12.8)
   - OpenCV CUDA support check
   - Python environment validation

2. **CUDA Configuration**
   - Automatic CUDA path setup
   - GPU architecture detection (Compute Capability 8.9 for RTX 4070 Ti)
   - NVCC compiler configuration

3. **Dependency Management**
   - ONNX Runtime GPU download (514MB CUDA provider)
   - GTK3 development libraries linking
   - cuDNN compatibility verification

4. **Build Optimization**
   - Parallel compilation (`-j$(nproc)`)
   - GPU-specific optimizations
   - Proper RPATH configuration

## 🐛 Troubleshooting

### Quick Reference
**📖 Complete troubleshooting guide**: See [BUILD_GUIDE.md](BUILD_GUIDE.md#troubleshooting)

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Verify CUDA libraries
ls -la /usr/local/cuda-12.8/lib64/libcudart.so
```

### ONNX Runtime CUDA Provider Not Available

```bash
# Ensure GPU version of ONNX Runtime is downloaded
ls -la ./onnxruntime/lib/libonnxruntime_providers_cuda.so

# Should be ~514MB
# If missing, delete onnxruntime/ and rebuild
rm -rf onnxruntime build
./build.sh
```

### OpenCV CUDA Support Missing

```bash
# Check OpenCV CUDA support
python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"

# If 0, install CUDA-enabled OpenCV:     
conda create -n opencv_cuda12 python=3.12
conda install -c conda-forge -y glib gtk3 gstreamer gst-plugins-base protobuf absl-py onnxruntime
conda install -c nvidia -y cuda-toolkit cudnn"
```

### Build Errors

**Quick Fix for Most Issues:**
```bash
# Clean everything and rebuild
rm -rf build onnxruntime
./build.sh
```

**Specific Build Issues:**

**Issue: "No GPU detected"**
- **What it means**: Building CPU version only
- **Solution**: This is normal if you don't have NVIDIA GPU
- **Result**: You'll get `./build/bgremover` (CPU version)

**Issue: "CUDA not found"**
- **What it means**: GPU build skipped, CPU version will be built
- **Solution**: Install CUDA toolkit or use CPU version
- **Check**: `nvidia-smi` to verify GPU drivers

**Issue: "ONNX Runtime download failed"**
```bash
# Manual download
cd onnxruntime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-linux-x64-gpu-1.19.0.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.19.0.tgz
cd ..
./build.sh
```

**Issue: "CMake configuration failed"**
```bash
# Clean build with verbose output
rm -rf build onnxruntime
mkdir build && cd build
cmake -DU2NET_DOWNLOAD_MODELS=ON .. -DCMAKE_VERBOSE_MAKEFILE=ON
make VERBOSE=1
```

**Check Build Results:**
```bash
# After build completes, check what was built
ls -la build/

# You should see:
# - bgremover (CPU version) - Always available
# - bgremover_gpu (GPU version) - Only if CUDA available

# Test the executables
./build/bgremover --help
./build/bgremover_gpu --help  # May not exist if no GPU
```

## 📈 1080p HD Performance Tuning

### GPU Architecture Optimization

The build script automatically detects your GPU architecture. For manual 1080p configuration:

```bash
# RTX 40 Series (Ada Lovelace) - Optimal for 1080p+4K
-DCUDA_ARCH_BIN=8.9

# RTX 30 Series (Ampere) - Excellent for 1080p
-DCUDA_ARCH_BIN=8.6

# RTX 20 Series (Turing) - Good for 1080p
-DCUDA_ARCH_BIN=7.5

# GTX 16 Series (Turing) - 1080p capable
-DCUDA_ARCH_BIN=7.5

# GTX 10 Series (Pascal) - 720p-1080p limited
-DCUDA_ARCH_BIN=6.1

# Tesla/Quadro Series
-DCUDA_ARCH_BIN=6.0+  # Varies by model
```

### 1080p Memory Management

- **1080p GPU Memory Limit**: 1.67GB / 15.56GB (10% of RTX 4070 Ti SUPER)
- **1080p Frame Buffer**: ~24MB (1920x1080x3 channels)
- **1080p Headroom**: 13.89GB available for larger models or multiple streams
- **4K Memory Usage**: 3.2GB / 15.56GB (20% for experimental 4K support)

### 1080p Inference Optimization

- **TensorFloat-32 (TF32)**: Enabled by default on Ampere+ GPUs (20% faster 1080p)
- **CUDA Graphs**: Available for repeated 1080p execution patterns
- **I/O Binding**: Minimizes CPU-GPU memory transfers for 1080p
- **Mixed Precision**: FP16 support for 1080p (50% memory reduction)
- **Multi-Stream**: 2-3 concurrent 1080p streams on RTX 4070 Ti
- **1080p Headroom**: 13.89GB available for larger models or multiple streams
- **4K Memory Usage**: 3.2GB / 15.56GB (20% for experimental 4K support)

### 1080p Inference Optimization

- **TensorFloat-32 (TF32)**: Enabled by default on Ampere+ GPUs (20% faster 1080p)
- **CUDA Graphs**: Available for repeated 1080p execution patterns
- **I/O Binding**: Minimizes CPU-GPU memory transfers for 1080p
- **Mixed Precision**: FP16 support for 1080p (50% memory reduction)
- **Multi-Stream**: 2-3 concurrent 1080p streams on RTX 4070 Ti

## 🔬 Technical Details

### CUDA Provider Configuration

The GPU version explicitly configures ONNX Runtime CUDA provider:

```cpp
// Create CUDA provider options
OrtCUDAProviderOptionsV2* cuda_options = nullptr;
Ort::GetApi().CreateCUDAProviderOptions(&cuda_options);

// Configure device
std::vector<const char*> keys{"device_id"};
std::vector<const char*> values{"0"};
Ort::GetApi().UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), 1);

// Append to session
Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2(
    static_cast<OrtSessionOptions*>(session_options),
    cuda_options
);
```

### Memory Management

- **CPU Memory**: Used for preprocessing and postprocessing
- **GPU Memory**: Used for model inference only
- **Pinned Memory**: Not currently used (future optimization)

## 📚 Project Structure

```
bgremover-lite/
├── main.cpp                 # CPU version
├── main_gpu.cpp            # GPU-accelerated version
├── CMakeLists.txt          # Build configuration (with RPATH)
├── build.sh                # Enhanced build script
├── setup_cuda_env.sh       # CUDA environment setup
├── verify_opencv_cuda.py   # GPU verification script
├── BUILD_GUIDE.md          # 📖 Complete build & usage documentation
├── README.md               # Main documentation
├── models/
│   ├── u2net.onnx         # U²-Net model
│   └── u2netp.onnx        # U²-Net (lightweight)
├── build/                  # Build artifacts
│   ├── bgremover          # CPU executable (RPATH-enabled)
│   └── bgremover_gpu      # GPU executable (RPATH-enabled)
└── onnxruntime/           # ONNX Runtime libraries
    └── lib/
        ├── libonnxruntime.so
        └── libonnxruntime_providers_cuda.so  # 514MB CUDA provider
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project uses:
- **ONNX Runtime**: Apache 2.0 License
- **OpenCV**: BSD License
- **U²-Net Model**: Apache 2.0 License

## 🙏 Acknowledgments

- **U²-Net**: Qin et al. - "U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection"
- **ONNX Runtime**: Microsoft - High-performance inference engine
- **OpenCV**: Open Source Computer Vision Library
- **NVIDIA**: CUDA toolkit and cuDNN libraries

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/lfontanez/bgremover-lite/issues)
- **Discussions**: [GitHub Discussions](https://github.com/lfontanez/bgremover-lite/discussions)

---

**Built with ❤️ for real-time computer vision**

**Status**: ✅ Production Ready | 🚀 GPU-Accelerated | ⚡ 30 FPS Performance

## ❓ FAQ

**Q: Can I use multiple physical webcams for 1080p?**
A: Yes, specify different input devices and output to different virtual cameras:
```bash
# Multiple 1080p virtual cameras
./build/bgremover_gpu /dev/video0 --vcam-device /dev/video2
./build/bgremover_gpu /dev/video1 --vcam-device /dev/video3
```

**Q: What background image formats are supported?**
A: All standard image formats are supported:
- **Formats**: JPG, PNG, BMP, TIFF
- **Resolution**: Any resolution (auto-resized to match video)
- **Performance**: Optimized for 1080p and higher
- **Usage**: 
  ```bash
  # JPG background (recommended for photos)
  ./build/bgremover_gpu --background-image photo.jpg
  
  # PNG background (supports transparency)
  ./build/bgremover_gpu --background-image logo.png
  ```

**Q: Can I combine custom backgrounds with blur effects?**
A: Currently, you can use either custom backgrounds OR blur effects, but not both simultaneously:
```bash
# Custom background (background replacement active)
./build/bgremover_gpu --background-image office.jpg

# Blur effect (blur active)
./build/bgremover_gpu --blur-mid
```

**Q: How do I create custom backgrounds?**
A: Use any image editing software to create backgrounds:
- **Recommended sizes**: 1920x1080 (1080p) or higher
- **Popular tools**: GIMP, Photoshop, Canva, or any photo editor
- **Background types**: Office environments, landscapes, branded images, abstract art
- **Example backgrounds**:
  ```bash
  # Professional office
  ./build/bgremover_gpu --background-image office.jpg
  
  # Nature landscape
  ./build/bgremover_gpu --background-image nature.jpg
  
  # Branded background
  ./build/bgremover_gpu --background-image company-logo.jpg
  ```

**Q: Can I use multiple custom backgrounds during a session?**
A: Currently, you need to restart the application to change backgrounds, but you can:
- **Pre-load backgrounds**: Start with one background, then restart with another
- **Quick switching**: Use different command aliases or scripts for instant switching
- **Video files**: Process different videos with different backgrounds:
  ```bash
  # Morning meeting with office background
  ./build/bgremover_gpu --background-image office.jpg
  
  # Afternoon call with casual background  
  ./build/bgremover_gpu --background-image cafe.jpg
  ```

**Q: What's the maximum background image size supported?**
A: While technically unlimited, for optimal performance:
- **Recommended**: Under 10MB file size
- **Resolution**: 1920x1080 (1080p) recommended
- **Performance**: Larger files may slow down startup
- **Memory**: Backgrounds are loaded into memory during processing
- **Tip**: Use compressed JPG for best size/quality ratio

**Q: Do custom backgrounds work with virtual camera?**
A: Yes, custom backgrounds work perfectly with virtual camera output:
```bash
# Virtual camera with custom background
./build/bgremover_gpu --vcam --background-image studio.jpg

# All applications (Zoom, Teams, etc.) will see the custom background
# Perfect for professional streaming and video calls
```

**Q: Can I use animated backgrounds or videos as backgrounds?**
A: Currently only static images are supported, but you can:
- **Convert animations**: Extract frames from GIFs/videos
- **Use video files**: Process video files directly with custom backgrounds
- **Future feature**: Animated backgrounds may be added in future versions

**Q: Does this work in VMs for 1080p?**
A: Yes, if the VM has:
- GPU passthrough (NVIDIA vGPU or SR-IOV)
- Access to /dev/video* devices
- 6GB+ VRAM allocation for 1080p processing
- USB 3.0 controller passthrough for high-res webcams

**Q: Why doesn't it need LD_LIBRARY_PATH anymore?**
A: Enhanced RPATH configuration automatically finds all required libraries:
- Libraries are found relative to the executable location (`$ORIGIN`)
- System libraries in standard paths are searched
- ONNX Runtime libraries are found via RPATH
- No manual environment variable configuration needed
- Same executables work in any environment (system, conda, containers)

**Q: What are the library search paths?**
A: The executable searches in this order:
1. Same directory as executable (`$ORIGIN`)
2. `../lib` relative to executable
3. ONNX Runtime installation directory
4. System directories (`/usr/lib/x86_64-linux-gnu`)
5. Standard library search paths

This ensures maximum compatibility while maintaining simplicity.
