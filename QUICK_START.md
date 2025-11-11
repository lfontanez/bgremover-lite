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
| `--no-preview` | Disable preview window | Better performance |

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

## 🖥️ No Preview Usage Examples

```bash
# Virtual camera without preview (recommended for streaming)
./build/bgremover_gpu --vcam --no-preview

# Virtual camera with custom settings and no preview
./build/bgremover_gpu --vcam --blur-high --no-preview
./build/bgremover_gpu --vcam --background-image studio.jpg --no-preview

# Headless webcam processing (server/automation)
./build/bgremover_gpu --no-preview

# Headless video file processing
./build/bgremover path/to/video.mp4 --no-preview --background-image background.jpg

# Maximum performance for 1080p processing
./build/bgremover_gpu --no-preview --blur-low

# CPU processing without GUI overhead
./build/bgremover --no-preview --blur-high
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
