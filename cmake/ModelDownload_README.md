# U²-Net Model Download Module

## Overview

This CMake module provides comprehensive download and verification functionality for U²-Net ONNX models at build time. It supports multiple download sources, SHA-256 verification, persistent caching, and offline builds.

## Features

- **Multiple Download Sources**: GitHub releases, Hugging Face, and direct repository URLs
- **SHA-256 Integrity Verification**: Ensures downloaded models are authentic and uncorrupted
- **Persistent Caching**: Models are cached in `~/.cache/u2net` (or equivalent platform-specific location)
- **Offline Build Support**: Uses cached models when no internet connection is available
- **Progress Indication**: Shows download progress for large files
- **Comprehensive Error Handling**: Retry logic and detailed error messages
- **Cross-Platform Support**: Works on Windows, Linux, and macOS

## Usage

### Basic Usage

```cmake
# Include the module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ModelDownload)

# Download models automatically
u2net_download_models()

# Verify model integrity
u2net_verify_model_integrity("u2net")
u2net_verify_model_integrity("u2netp")

# Get model paths
u2net_get_model_path("u2net" U2NET_MODEL_PATH)
u2net_get_model_path("u2netp" U2NETP_MODEL_PATH)
```

### Configuration Options

- `U2NET_DOWNLOAD_MODELS`: Enable/disable model download (default: ON)
- `U2NET_OFFLINE_MODE`: Use only cached models (default: OFF)
- `U2NET_CLEAN_CACHE`: Clean cache before downloading (default: OFF)
- `U2NET_MODEL_CACHE_DIR`: Custom cache directory path

### Manual Model Placement

If you prefer to place models manually, simply put them in the `models/` directory:

```
models/
├── u2net.onnx
└── u2netp.onnx
```

## API Reference

### Functions

#### `u2net_download_models()`
Downloads all required U²-Net models with retry logic and progress indication.

#### `u2net_verify_model_integrity(model_name)`
Verifies model integrity using SHA-256 hash comparison.

#### `u2net_get_model_path(model_name output_var)`
Returns the full path to a specific model file.

#### `u2net_get_all_model_paths(output_var)`
Returns all available model paths as a list.

#### `u2net_models_available(available_var)`
Checks if all required models are available.

#### `u2net_clean_cache()`
Removes cached models from the cache directory.

#### `u2net_show_model_info()`
Displays detailed information about available models.

## Cache Directory

The module uses platform-appropriate cache directories:

- **Linux**: `~/.cache/u2net`
- **macOS**: `~/Library/Caches/u2net` (if `XDG_CACHE_HOME` not set)
- **Windows**: `%LOCALAPPDATA%\u2net\cache`

## Download Sources

The module tries multiple sources in order:

1. **GitHub Releases** (primary): `https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/{model}.onnx`
2. **Hugging Face** (secondary): `https://huggingface.co/datasets/ak семейr/u2net/resolve/main/{model}.onnx`
3. **Direct Repository** (fallback): `https://raw.githubusercontent.com/xuebinqin/U-2-Net/master/{model}.onnx`

## Model Files

- **u2net.onnx**: ~167MB - Full precision U²-Net model
- **u2netp.onnx**: ~4MB - Lightweight U²-Net Portrait model

## Error Handling

The module provides detailed error messages and retry logic:

- Automatic retry (3 attempts by default)
- Graceful fallback to cached models
- Clear error messages for troubleshooting
- Build continues with warnings if models are missing

## Examples

### Enable with custom cache directory:

```cmake
set(U2NET_MODEL_CACHE_DIR "/custom/cache/path" CACHE PATH "Custom cache directory")
u2net_download_models()
```

### Check model availability before building:

```cmake
u2net_models_available(models_available)
if(NOT models_available)
    message(FATAL_ERROR "Required U²-Net models not found!")
endif()
```

### Display model information:

```cmake
u2net_show_model_info()
```

## Troubleshooting

### Download fails

1. Check internet connectivity
2. Verify firewall/proxy settings
3. Try manual model placement in `models/` directory
4. Use offline mode if you have cached models

### SHA-256 verification fails

1. Delete cache and try again: `u2net_clean_cache()`
2. Check if model files are corrupted
3. Use manual placement as fallback

### Build fails due to missing models

1. Ensure models are in `models/` directory
2. Check file permissions
3. Verify CMake can access the files

## Integration

This module is designed to be easily integrated into existing CMake projects. Simply include it in your main `CMakeLists.txt` and use the provided functions.
