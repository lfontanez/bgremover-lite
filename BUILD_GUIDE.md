# BGRemover Lite - Build & Usage Guide

## 🚀 Quick Start

```bash
cd /home/lfontanez/dev/bgremover-lite
rm -rf build
./build.sh
./build/bgremover --help
./build/bgremover_gpu --help
```

## 📋 Table of Contents

1. [LD_LIBRARY_PATH Issue & Solution](#ld_library_path-issue--solution)
2. [Build Methods](#build-methods)
3. [Usage Instructions](#usage-instructions)
4. [RPATH Configuration](#rpath-configuration)
5. [Troubleshooting](#troubleshooting)
6. [Performance Notes](#performance-notes)

---

## 🔧 LD_LIBRARY_PATH Issue & Solution

### **The Problem**

When building with conda environments or system libraries, executables couldn't find their required shared libraries at runtime, requiring:

```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH ./build/bgremover
```

### **The Solution: Enhanced RPATH Configuration**

The build system now uses **RPATH** (Runtime Library Path) configuration to automatically find libraries without LD_LIBRARY_PATH. This was implemented in `CMakeLists.txt`:

```cmake
# Enhanced RPATH setup - libraries found automatically
set(RUNTIME_LIBRARY_PATHS)
list(APPEND RUNTIME_LIBRARY_PATHS "$ORIGIN")              # Same directory as executable
list(APPEND RUNTIME_LIBRARY_PATHS "$ORIGIN/../lib")       # ../lib relative to executable
list(APPEND RUNTIME_LIBRARY_PATHS "${ONNXRUNTIME_LIB_DIR}")  # ONNX Runtime library directory
list(APPEND RUNTIME_LIBRARY_PATHS "/usr/lib/x86_64-linux-gnu")  # System libraries
list(APPEND RUNTIME_LIBRARY_PATHS "/lib/x86_64-linux-gnu")      # System libraries

# Set RPATH properties for all executables
set_target_properties(bgremover bgremover_gpu test_model debug_colors test_performance test_gpu
    PROPERTIES
    BUILD_RPATH "${RUNTIME_RPATH}"
    INSTALL_RPATH "${RUNTIME_RPATH}"
    BUILD_WITH_INSTALL_RPATH FALSE
    SKIP_BUILD_RPATH FALSE
)
```

### **How RPATH Works**

- **`$ORIGIN`** - Looks for libraries relative to the executable location
- **Multiple search paths** - System libraries, ONNX Runtime, conda libraries
- **Portable** - Works in any environment without LD_LIBRARY_PATH
- **Automatic** - No manual configuration needed

---

## 🏗️ Build Methods

### **Method 1: System Environment (Recommended)**

**Default method - works on any system:**

```bash
cd /home/lfontanez/dev/bgremover-lite
rm -rf build
./build.sh
```

**What it does:**
- Uses system OpenCV from conda-forge
- Downloads ONNX Runtime automatically
- Builds both CPU and GPU versions
- Configures RPATH for library discovery

### **Method 2: Conda Environment Build**

**For users who prefer conda environments:**

```bash
cd /home/lfontanez/dev/bgremover-lite
rm -rf build

# Set conda environment variables
export CONDA_DEFAULT_ENV=opencv_cpu
export PATH=$HOME/miniconda3/envs/opencv_cpu/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/opencv_cpu/lib:$LD_LIBRARY_PATH

./build.sh
```

**Or using conda activation (if available):**

```bash
conda activate opencv_cpu
cd /home/lfontanez/dev/bgremover-lite
rm -rf build
./build.sh
```

### **Method 3: Clean Environment**

**For debugging or clean builds:**

```bash
cd /home/lfontanez/dev/bgremover-lite
rm -rf build onnxruntime
unset CONDA_DEFAULT_ENV
export PATH="$HOME/miniconda3/bin:$PATH"
./build.sh
```

---

## 📖 Usage Instructions

### **Build Results**

After successful build, you'll have:

| Executable | Location | Requirements | Performance | Usage |
|------------|----------|--------------|-------------|-------|
| **CPU Version** | `./build/bgremover` | Any x64 CPU | 1-5 FPS | Testing, fallback |
| **GPU Version** | `./build/bgremover_gpu` | NVIDIA GPU + CUDA | 25-30 FPS (1080p) | **Recommended** |

### **Running Executables**

**No LD_LIBRARY_PATH needed!**

```bash
# CPU version
./build/bgremover --help
./build/bgremover                    # Webcam with background blur
./build/bgremover --blur-high        # Strong background blur
./build/bgremover --background-image background.jpg  # Custom background
./build/bgremover --no-preview       # No preview window (headless)

# GPU version
./build/bgremover_gpu --help
./build/bgremover_gpu                # GPU-accelerated webcam
./build/bgremover_gpu --vcam         # Virtual camera output
./build/bgremover_gpu --blur-high    # High blur with GPU
./build/bgremover_gpu --no-preview   # No preview window (headless)
```

### **Background Blur Control Options**

| Option | Description | Kernel Size | Performance Impact |
|--------|-------------|-------------|-------------------|
| `--no-blur` | Disable blur completely | None | Fastest |
| `--blur-low` | Subtle background blur | 7x7 | Minimal |
| `--blur-mid` | Balanced blur (default) | 15x15 | Medium |
| `--blur-high` | Maximum background blur | 25x25 | Higher |
| `--background-image PATH` | Replace with custom image | N/A | Minimal |
| `--no-preview` | Disable preview window | N/A | Better performance |

### **Virtual Camera Support**

Send processed video to virtual camera device:

```bash
# Enable virtual camera output
./build/bgremover_gpu --vcam                    # Default device: /dev/video2
./build/bgremover_gpu --vcam --vcam-device /dev/video3  # Custom device

# Virtual camera with different settings
./build/bgremover_gpu --vcam --blur-high        # High blur in virtual camera
./build/bgremover_gpu --vcam --background-image office.jpg  # Custom background

# Virtual camera without preview (recommended for streaming)
./build/bgremover_gpu --vcam --no-preview       # No preview window
./build/bgremover_gpu --vcam --blur-high --no-preview  # High blur, no preview
./build/bgremover_gpu --vcam --background-image studio.jpg --no-preview  # Custom background, no preview

# Headless processing examples
./build/bgremover_gpu --no-preview              # Headless webcam processing
./build/bgremover --no-preview --blur-mid       # CPU version without preview
./build/bgremover_gpu path/to/video.mp4 --no-preview --background-image background.jpg  # Process video without preview
```

---

## 🔗 RPATH Configuration Details

### **Library Search Order**

The executable searches for libraries in this order:

1. **`$ORIGIN`** - Directory containing the executable
2. **`$ORIGIN/../lib`** - `../lib` relative to executable
3. **ONNX Runtime directory** - Where ONNX Runtime libraries are stored
4. **System directories** - `/usr/lib/x86_64-linux-gnu`, `/lib/x86_64-linux-gnu`
5. **Standard system paths** - Default library search paths

### **Verification**

Check if RPATH is working:

```bash
# View library dependencies
ldd ./build/bgremover | grep -E "(onnxruntime|opencv)"

# Expected output shows libraries found via RPATH:
# libonnxruntime.so.1 => /home/lfontanez/dev/bgremover-lite/onnxruntime/lib/libonnxruntime.so.1
# libopencv_highgui.so.412 => /home/lfontanez/miniconda3/lib/libopencv_highgui.so.412
```

### **Build Output**

Look for this in the build log:

```
🔗 RPATH configured for runtime library discovery:
   Runtime library paths: $ORIGIN;$ORIGIN/../lib;/home/lfontanez/dev/bgremover-lite/onnxruntime/lib;/usr/lib/x86_64-linux-gnu;/lib/x86_64-linux-gnu
   Combined RPATH: $ORIGIN:$ORIGIN/../lib:/home/lfontanez/dev/bgremover-lite/onnxruntime/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu
```

---

## 🐛 Troubleshooting

### **Common Issues**

#### 1. Build Fails with CUDA Conflict
**Error:** `CMAKE_DISABLE_FIND_PACKAGE_CUDA is enabled. A REQUIRED package cannot be disabled.`

**Solution:**
```bash
# Clean build
rm -rf build onnxruntime

# Use system environment (no conda)
unset CONDA_DEFAULT_ENV
./build.sh
```

#### 2. Executables Don't Work After Build
**Error:** `./build/bgremover: error while loading shared libraries`

**Solution:** This should be fixed by RPATH. If it still occurs:
```bash
# Check library dependencies
ldd ./build/bgremover

# Clean rebuild
rm -rf build
./build.sh
```

#### 3. OpenCV Not Found
**Error:** `Could not find OpenCV`

**Solution:**
```bash
# Install OpenCV
conda install -c conda-forge opencv=4.12.0

# Or ensure conda-forge is in channels
conda config --add channels conda-forge
conda install opencv
```

#### 4. CUDA Not Available
**Error:** `CUDA 12.8 not found at /usr/local/cuda-12.8`

**Solution:** Install CUDA 12.8 or use CPU-only build:
```bash
# CPU-only build
export CONDA_DEFAULT_ENV=opencv_cpu
./build.sh
# Only ./build/bgremover will be created
```

### **Verification Commands**

Check build status:

```bash
# Verify executables exist
ls -la build/bgremover*

# Test without LD_LIBRARY_PATH
./build/bgremover --help
./build/bgremover_gpu --help

# Check library dependencies
ldd build/bgremover | head -5
ldd build/bgremover_gpu | head -5
```

### **Clean Rebuild Process**

If you encounter any issues:

```bash
# Complete clean rebuild
cd /home/lfontanez/dev/bgremover-lite
rm -rf build onnxruntime models
unset CONDA_DEFAULT_ENV
export PATH="$HOME/miniconda3/bin:$PATH"
./build.sh
```

---

## ⚡ Performance Notes

### **Hardware Requirements**

#### CPU Version
- **Requirements**: Any modern x64 CPU
- **Performance**: 1-5 FPS
- **Memory**: ~100MB RAM
- **Use Case**: Testing, fallback, systems without GPU

#### GPU Version
- **Requirements**: NVIDIA GPU with CUDA support
- **Recommended**: GTX 1060+ or RTX series
- **Performance**: 25-30 FPS at 1080p
- **Memory**: ~1.7GB VRAM
- **Use Case**: Production, real-time processing, high-resolution video

### **Performance Optimization**

#### For 1080p Processing
- **Low Blur**: `--blur-low` for best FPS
- **Medium Blur**: `--blur-mid` balanced quality/performance
- **High Blur**: `--blur-high` maximum quality, still maintains good FPS

#### For Virtual Camera
- **Minimal overhead**: ~1-2ms per frame
- **Best quality**: Use GPU version for virtual camera output
- **Device setup**: Requires v4l2loopback module for Linux

### **Memory Usage**

- **1080p processing**: ~1.67GB VRAM (RTX 4070 Ti SUPER)
- **ONNX Runtime**: ~514MB (downloaded automatically)
- **U²-Net models**: ~175MB cached in `~/.cache/u2net/`

---

## 🔄 Environment Management

### **Best Practices**

1. **Use system build** for most users - no conda environment needed
2. **Conda environment** only if you have specific OpenCV requirements
3. **Clean build** if you encounter issues
4. **Verify RPATH** is working after build

### **Environment Variables**

| Variable | Purpose | When to Set |
|----------|---------|-------------|
| `CONDA_DEFAULT_ENV` | Activate conda environment | Optional |
| `LD_LIBRARY_PATH` | Library search path | Not needed (RPATH handles this) |
| `CUDA_PATH` | CUDA toolkit location | Auto-configured |
| `PATH` | Executable search path | Auto-configured |

---

## 📚 Additional Resources

### **Build System Components**
- **CMakeLists.txt** - Build configuration with RPATH
- **build.sh** - Enhanced build script with environment detection
- **cmake/ModelDownload.cmake** - Automatic model download
- **onnxruntime/** - Downloaded ONNX Runtime with GPU support

### **Model Files**
- **~/.cache/u2net/u2net.onnx** - Main U²-Net model (175MB)
- **~/.cache/u2net/u2netp.onnx** - U²-Net Portrait model (4.5MB)
- **models/** - Local copy in build directory

### **Output Files**
- **build/bgremover** - CPU executable
- **build/bgremover_gpu** - GPU executable
- **build/models/** - Model files copied to build directory
- **onnxruntime/** - ONNX Runtime libraries and headers

---

## ✅ Summary

**The LD_LIBRARY_PATH issue is permanently resolved** through enhanced RPATH configuration. Users can now:

1. **Build easily**: `./build.sh` in any environment
2. **Run anywhere**: No LD_LIBRARY_PATH manipulation needed
3. **Get good performance**: 30 FPS GPU, 1-5 FPS CPU
4. **Use flexible options**: Multiple blur levels, custom backgrounds, virtual camera

**Result**: A portable, high-performance background removal system that works out of the box! 🚀
