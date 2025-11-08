# BGRemover Lite - Quick Reference

## 🚀 Build & Run (30 seconds)

```bash
# Build (no LD_LIBRARY_PATH needed!)
rm -rf build && ./build.sh

# Run (works immediately!)
./build/bgremover --help
./build/bgremover_gpu --help
```

## 🔧 Problem Solved: LD_LIBRARY_PATH

**Before**: Required `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH ./build/bgremover`

**After**: Enhanced RPATH finds libraries automatically - no LD_LIBRARY_PATH needed!

## 📋 Available Commands

| Command | Description | Performance |
|---------|-------------|-------------|
| `./build/bgremover` | CPU background blur | 1-5 FPS |
| `./build/bgremover_gpu` | GPU background blur | 25-30 FPS |
| `--blur-low/--mid/--high` | Blur intensity levels | Varies |
| `--background-image PATH` | Custom background | Minimal impact |
| `--vcam` | Virtual camera output | +1-2ms latency |

## 🎯 Most Common Usage

```bash
# GPU version (recommended)
./build/bgremover_gpu

# With high blur
./build/bgremover_gpu --blur-high

# Virtual camera
./build/bgremover_gpu --vcam

# Custom background
./build/bgremover_gpu --background-image office.jpg
```

## 📖 Complete Documentation

**Detailed guide**: [BUILD_GUIDE.md](BUILD_GUIDE.md)
- Complete build methods
- Troubleshooting guide
- Performance optimization
- Environment management
- RPATH configuration details

## ✅ Build Status

- ✅ **No LD_LIBRARY_PATH required** - RPATH enabled
- ✅ **GPU acceleration** - CUDA + ONNX Runtime
- ✅ **Virtual camera support** - v4l2loopback
- ✅ **Custom backgrounds** - Image replacement
- ✅ **Multiple blur levels** - Performance tuning
- ✅ **Cross-environment** - System/conda compatible