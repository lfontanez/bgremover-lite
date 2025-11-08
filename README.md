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