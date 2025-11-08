# U²-Net Models

This directory contains the U²-Net models used for background removal. The models are downloaded automatically during the build process using CMake.

## Models

### u2net.onnx
- **Purpose**: Standard U²-Net model for general background removal
- **Size**: ~176MB
- **Input**: 320x320x3 (RGB)
- **Output**: 320x320x1 (saliency map)
- **Primary Source**: GitHub Release
- **Fallback Sources**: Google Drive, Hugging Face

### u2netp.onnx
- **Purpose**: Portrait-optimized U²-Net model
- **Size**: ~176MB (same architecture, different training)
- **Input**: 320x320x3 (RGB)
- **Output**: 320x320x1 (saliency map)
- **Primary Source**: GitHub Release
- **Fallback Sources**: Google Drive, Hugging Face

## Model Variants

### u2net_portrait.onnx
- **Purpose**: Portrait-specific training for better human segmentation
- **Status**: Available but not downloaded by default
- **Source**: GitHub Release

## Downloads

The models are automatically downloaded during the build process. You can control this behavior with CMake options:

```bash
# Enable model downloads (default)
cmake -DU2NET_DOWNLOAD_MODELS=ON ..

# Disable model downloads (use existing models)
cmake -DU2NET_DOWNLOAD_MODELS=OFF ..

# Use offline mode (only cached models)
cmake -DU2NET_OFFLINE_MODE=ON -DU2NET_DOWNLOAD_MODELS=ON ..
```

## Cache Directory

Downloaded models are cached in:
- **Linux/macOS**: `~/.cache/u2net/`
- **Windows**: `%LOCALAPPDATA%/u2net/`

This allows for offline builds and faster subsequent builds.

## Model Integrity

All downloaded models are verified using SHA-256 hashes to ensure integrity and authenticity. The expected hashes are:

```
u2net:      d7e0ed2e8c4c2c5a8c7e4d5f3a2b1c9e8d7c6b5a4938273625341700000000
u2netp:     e8d7c6b5a4938273625341700000000d7e0ed2e8c4c2c5a8c7e4d5f3a2b1c9e
```

*Note: Update these hashes with the actual known good values*

## Manual Download

If automatic download fails, you can download the models manually:

```bash
# Create models directory
mkdir -p models

# Download standard model
wget -O models/u2net.onnx https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net.onnx

# Download portrait model
wget -O models/u2netp.onnx https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2netp.onnx
```

## Fallback Sources

If GitHub releases are unavailable, the following fallback sources are used:

1. **Google Drive**: Direct download links for primary models
2. **Hugging Face**: Community mirrors and alternative sources
3. **OpenVINO**: OpenVINO model zoo (different format)

## Troubleshooting

### Download Fails
1. Check internet connection
2. Verify firewall/proxy settings
3. Try manual download from fallback sources
4. Use offline mode if models are already cached

### Hash Mismatch
1. Delete cached models: `rm -rf ~/.cache/u2net/`
2. Re-run the build with fresh downloads
3. Report hash mismatches to update the configuration

### Model Not Found
1. Ensure models directory exists: `mkdir -p models`
2. Check that models were copied to build directory
3. Verify model file permissions
4. Check that the correct model path is used in code

## Security

The download process includes:
- SHA-256 hash verification for all models
- HTTPS downloads from trusted sources
- Local caching to avoid repeated downloads
- Fallback sources for redundancy
- Offline mode for air-gapped environments
