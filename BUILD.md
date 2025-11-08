# Build Guide - CPU vs GPU

Quick reference for building BGRemover Lite with CPU and/or GPU acceleration.

## 🚀 Quick Build (Recommended)

**Automatically builds both CPU and GPU versions:**

```bash
git clone https://github.com/lfontanez/bgremover-lite.git
cd bgremover-lite

# Check your system first (optional but recommended)
./check_system.sh

# Automatic build - detects GPU and builds both versions
./build.sh
```

**What happens:**
- ✅ Detects your hardware automatically
- ✅ Downloads required models
- ✅ Builds CPU version (always)
- ✅ Builds GPU version (if NVIDIA GPU detected)

**Results:**
- `./build/bgremover` - CPU version (1-5 FPS)
- `./build/bgremover_gpu` - GPU version (25-30 FPS at 1080p)

## 📱 CPU-Only Build

**Requirements:** Any modern x64 CPU

**Use case:** Testing, development, systems without NVIDIA GPU

```bash
# Automatic (no GPU detected)
./build.sh

# Manual
mkdir build && cd build
cmake -DU2NET_DOWNLOAD_MODELS=ON ..
make -j$(nproc)

# Result: ./bgremover
```

## 🎮 GPU-Accelerated Build

**Requirements:** NVIDIA GPU + CUDA + Drivers

**Use case:** Production, real-time video processing, 1080p

### Quick GPU Setup

```bash
# Install CUDA-enabled OpenCV (recommended)
conda create -n opencv_cuda12 python=3.11 opencv cudatoolkit=12.1
conda activate opencv_cuda12

# Build with GPU support
./build.sh

# Results: ./bgremover (CPU) + ./build/bgremover_gpu (GPU)
```

### Manual GPU Build

```bash
# Set CUDA environment
export CUDA_PATH=/usr/local/cuda-12.8
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Build
mkdir build && cd build
cmake -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH \
      -DWITH_CUDA=ON \
      -DU2NET_DOWNLOAD_MODELS=ON ..
make -j$(nproc)
```

## 🔧 Build Verification

**Check what was built:**

```bash
ls -la build/
# Expected files:
# - bgremover (CPU version)
# - bgremover_gpu (GPU version - if CUDA available)

# Test executables
./build/bgremover --help
./build/bgremover_gpu --help
```

## ⚡ Performance Comparison

| Version | Requirements | Performance | Use Case |
|---------|--------------|-------------|----------|
| **CPU** | Any x64 CPU | 1-5 FPS | Testing, old systems |
| **GPU** | NVIDIA GPU | 25-30 FPS | Production, 1080p |

## 🐛 Common Build Issues

**"No GPU detected"**
- ✅ Normal if no NVIDIA GPU
- ✅ CPU version still works
- 🖥️ Use `./build/bgremover`

**"CUDA not found"**
- ✅ Install CUDA toolkit or use CPU version
- 📊 Check: `nvidia-smi`

**"Build failed"**
- 🔄 Clean rebuild: `rm -rf build onnxruntime && ./build.sh`

## 📋 Requirements Summary

### CPU Build
- **CPU:** Any x64 processor
- **Memory:** 4GB RAM
- **Storage:** 2GB free space
- **Time:** 5-10 minutes

### GPU Build
- **GPU:** NVIDIA GTX 1060+ (recommended RTX 2060+)
- **VRAM:** 6GB minimum (12GB+ recommended)
- **CUDA:** 12.8 (recommended) or 11.0+
- **Drivers:** Latest NVIDIA drivers
- **Memory:** 8GB RAM
- **Storage:** 3GB free space
- **Time:** 10-15 minutes

## 🎯 Recommended Build Strategy

1. **First time:** Run `./build.sh` - it handles everything
2. **Check results:** Verify both executables are built
3. **Test CPU version:** `./build/bgremover --help`
4. **Test GPU version:** `./build/bgremover_gpu --help`
5. **Use GPU version:** For production, real-time use
6. **Use CPU version:** For testing, fallback, old systems

---

**Need help?** See README.md for detailed troubleshooting or run `./build.sh --help` for build options.