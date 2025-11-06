# Background Remover Lite

A lightweight C++ background removal tool using ONNX Runtime and OpenCV.

## Features

- Real-time background removal from video streams
- Supports webcam input or video files
- Uses pre-trained U²-Net model for high-quality segmentation
- Fast inference with ONNX Runtime optimization

## Requirements

- C++17 compatible compiler
- CMake 3.16+
- OpenCV 4.x development libraries
- Internet connection (for first build to download ONNX Runtime)

## Installation

### Quick Start (Recommended)

```bash
# Make the build script executable (if needed)
chmod +x build.sh

# Run the build script
./build.sh
```

### Manual Build

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
make -j$(nproc)

# Run
./bgremover
```

## Usage

### Webcam
```bash
./build/bgremover
```

### Video File
```bash
./build/bgremover path/to/video.mp4
```

### Exit
Press `ESC` to quit the application.

## How It Works

1. **Model Download**: During the first build, ONNX Runtime is automatically downloaded
2. **Real-time Processing**: Captures frames from video source
3. **Segmentation**: Uses U²-Net model to generate foreground mask
4. **Compositing**: Applies Gaussian blur to background and composites with foreground

## Dependencies

- **OpenCV**: Computer vision library
- **ONNX Runtime**: Deep learning inference engine
- **U²-Net Model**: Pre-trained segmentation model (included)

## Build Process

The build script automatically:
1. Checks for required dependencies (OpenCV)
2. Downloads ONNX Runtime (Linux x64, version 1.19.0)
3. Compiles the project with proper linking
4. Creates an executable ready to run

## Platform Support

Currently optimized for **Linux x64**. 
- Windows and macOS builds can be supported by extending the CMake download logic
- For other platforms, ONNX Runtime can be installed manually and CMake paths adjusted

## License

This project uses:
- ONNX Runtime (Apache 2.0 License)
- OpenCV (BSD License)
- U²-Net Model (various open source licenses)

## Troubleshooting

### Build Issues
- Ensure OpenCV development packages are installed
- Check that CMake can find your C++17 compiler
- Verify internet connection for ONNX Runtime download

### Runtime Issues
- Ensure camera permissions for webcam access
- Check that model files are properly downloaded
- Verify OpenCV can access video sources

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request