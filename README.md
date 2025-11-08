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

### Hardware
- **GPU** (Recommended): NVIDIA GPU with CUDA support (Compute Capability 6.0+)
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

### Quick Start (Conda Environment - Recommended)

```bash
# Create and activate conda environment with CUDA-enabled OpenCV
conda create -n opencv_cuda12 python=3.11 opencv cudatoolkit=12.1
conda activate opencv_cuda12

# Clone the repository
git clone https://github.com/lfontanez/bgremover-lite.git
cd bgremover-lite

# Build with GPU acceleration
./build.sh
```

### Manual Build

```bash
# Ensure CUDA environment is set
export CUDA_PATH=/usr/local/cuda-12.8
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Create build directory
mkdir build && cd build

# Configure with CUDA support
cmake -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH \
      -DCMAKE_CUDA_COMPILER=$CUDA_PATH/bin/nvcc \
      -DWITH_CUDA=ON \
      -DCUDA_ARCH_BIN=8.9 \
      ..

# Build
make -j$(nproc)
```

## 🚀 Usage

### GPU-Accelerated Version (Recommended)

```bash
# Webcam (default)
./build/bgremover_gpu

# Video file
./build/bgremover_gpu path/to/video.mp4

# With specific device
./build/bgremover_gpu 0  # Device 0 (webcam)
```

### CPU Version (Fallback)

```bash
# Webcam
./build/bgremover

# Video file
./build/bgremover path/to/video.mp4
```

### Controls
- **ESC**: Quit application
- Real-time FPS and performance stats displayed in console

## 📹 Virtual Camera

### Virtual Camera Feature 🌐
The Virtual Camera feature makes your processed video available to any application that accepts camera input, including Zoom, Microsoft Teams, Google Meet, OBS Studio, Discord, and more. Instead of opening a window, the processed video is sent to a virtual camera device (typically `/dev/video2`) that appears as a regular webcam to other applications.

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
sudo modprobe v4l2loopback devices=1 video_nr=2 card_label="BackgroundRemover"

# Some applications require a different pixel format - try:
sudo modprobe v4l2loopback devices=1 video_nr=2 card_label="BackgroundRemover" pixel_format=YUYV
```

#### Module Not Loading on Boot
Ensure the module loads automatically:

```bash
# Check if module is in the boot configuration
ls -la /etc/modules-load.d/
ls -la /etc/modprobe.d/

# Re-add the module to load automatically
echo "v4l2loopback" | sudo tee -a /etc/modules
echo "options v4l2loopback video_nr=2 card_label=\"BGRemover Virtual Camera\"" | sudo tee -a /etc/modprobe.d/v4l2loopback.conf

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

### Check GPU Acceleration

```bash
# Verify CUDA environment
python3 verify_opencv_cuda.py

# Expected output:
# ✅ OpenCV CUDA support is available!
# ✅ ONNX Runtime CUDA support available!
# 🎉 GPU acceleration is ready to go!
```

### Monitor GPU Usage

```bash
# Real-time GPU monitoring
nvidia-smi --loop-ms=1000

# Watch for:
# - GPU Memory: ~1.7GB used during inference
# - GPU Utilization: 20-40% during processing
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
conda create -n opencv_cuda12 python=3.11 opencv cudatoolkit=12.1
conda activate opencv_cuda12
```

### Build Errors

```bash
# Clean build
rm -rf build onnxruntime
source ~/miniconda3/bin/activate opencv_cuda12
./build.sh

# Check CMake output for specific errors
# Common issues:
# - Missing CUDA_PATH environment variable
# - Incompatible cuDNN version
# - Missing GTK3 development libraries
```

## 📈 Performance Tuning

### GPU Architecture Optimization

The build script automatically detects your GPU architecture. For manual configuration:

```bash
# RTX 40 Series (Ada Lovelace)
-DCUDA_ARCH_BIN=8.9

# RTX 30 Series (Ampere)
-DCUDA_ARCH_BIN=8.6

# RTX 20 Series (Turing)
-DCUDA_ARCH_BIN=7.5

# GTX 10 Series (Pascal)
-DCUDA_ARCH_BIN=6.1
```

### Memory Management

- **GPU Memory Limit**: Configurable in main_gpu.cpp
- **Current Usage**: 1.67GB / 15.56GB (10% of RTX 4070 Ti SUPER)
- **Headroom**: 13.89GB available for larger models

### Inference Optimization

- **TensorFloat-32 (TF32)**: Enabled by default on Ampere+ GPUs
- **CUDA Graphs**: Available for repeated execution patterns
- **I/O Binding**: Minimizes CPU-GPU memory transfers

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
├── CMakeLists.txt          # Build configuration
├── build.sh                # Enhanced build script
├── setup_cuda_env.sh       # CUDA environment setup
├── verify_opencv_cuda.py   # GPU verification script
├── models/
│   ├── u2net.onnx         # U²-Net model
│   └── u2netp.onnx        # U²-Net (lightweight)
├── build/                  # Build artifacts
│   ├── bgremover          # CPU executable
│   └── bgremover_gpu      # GPU executable
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
