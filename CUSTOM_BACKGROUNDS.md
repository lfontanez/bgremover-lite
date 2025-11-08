# Custom Background Replacement Guide

## Overview

The Custom Background Replacement feature allows you to replace your real background with any image during video processing. This feature is perfect for professional video calls, content creation, streaming, and any scenario where you want a specific branded or themed background.

## Quick Start

```bash
# Basic usage with custom background
./build/bgremover_gpu --background-image my_background.jpg

# Short form option
./build/bgremover_gpu --bg-image office.jpg

# With virtual camera for video conferencing apps
./build/bgremover_gpu --vcam --background-image studio.jpg
```

## Supported Image Formats

| Format | Extension | Best For | Notes |
|--------|-----------|----------|-------|
| **JPEG** | .jpg, .jpeg | Photos, general use | Compressed, good quality |
| **PNG** | .png | Graphics, logos | Supports transparency |
| **BMP** | .bmp | Uncompressed images | Large file size |
| **TIFF** | .tif, .tiff | Professional use | High quality, large files |

## Image Requirements

### Resolution
- **Any resolution supported**: Images are automatically resized
- **Recommended**: 1920x1080 (1080p) or higher for best quality
- **Common sizes**: 1080p, 1440p, 4K
- **Auto-scaling**: Small images are upscaled, large images are downscaled

### File Size
- **Recommended**: Under 10MB for fast loading
- **No hard limit**: Larger files work but may slow startup
- **Performance impact**: Larger files = longer initial load time

### Aspect Ratio
- **Flexible**: Any aspect ratio supported
- **Auto-handled**: No need to crop or resize manually
- **16:9 recommended**: Matches most video formats

## Usage Examples

### Basic Video Processing

```bash
# Webcam with custom background
./build/bgremover_gpu --background-image office.jpg

# Video file with custom background
./build/bgremover_gpu meeting_recording.mp4 --background-image conference_room.jpg

# Specific input device
./build/bgremover_gpu /dev/video1 --background-image studio.jpg
```

### Professional Video Conferencing

```bash
# Corporate meeting setup
./build/bgremover_gpu --vcam --background-image corporate_office.jpg

# Client presentation
./build/bgremover_gpu --vcam --background-image branded_studio.jpg

# Team collaboration
./build/bgremover_gpu --vcam --background-image modern_office.jpg
```

### Content Creation & Streaming

```bash
# YouTube/TikTok content
./build/bgremover_gpu --vcam --background-image studio_green.jpg

# Podcast recording
./build/bgremover_gpu --background-image podcast_studio.jpg

# Gaming stream
./build/bgremover_gpu --vcam --background-image gaming_setup.jpg
```

### Multiple Backgrounds

```bash
# Different backgrounds for different purposes
./build/bgremover_gpu --background-image morning_office.jpg    # Morning meetings
./build/bgremover_gpu --background-image casual_café.jpg      # Informal calls
./build/bgremover_gpu --background-image professional_studio.jpg # Client meetings
```

## Background Types & Recommendations

### Professional Environments

#### Corporate Office
- **Use case**: Business meetings, client calls
- **Style**: Clean, modern, professional
- **Colors**: Neutral tones, corporate colors
- **Elements**: Office furniture, city views, modern architecture

#### Conference Room
- **Use case**: Formal presentations, team meetings
- **Style**: Professional, authoritative
- **Colors**: Blues, grays, white
- **Elements**: Conference tables, screens, professional lighting

#### Home Office
- **Use case**: Remote work, casual business
- **Style**: Comfortable yet professional
- **Colors**: Warm, inviting
- **Elements**: Desk, bookshelf, plants, natural light

### Creative & Entertainment

#### Photography Studio
- **Use case**: Content creation, artistic projects
- **Style**: Clean, minimalist, professional
- **Colors**: White, gray, black
- **Elements**: Backdrop stands, lighting equipment

#### Nature & Outdoor
- **Use case**: Relaxed meetings, wellness content
- **Style**: Natural, calming
- **Colors**: Greens, blues, earth tones
- **Elements**: Landscapes, gardens, outdoor scenes

#### Abstract & Artistic
- **Use case**: Creative projects, artistic content
- **Style**: Bold, creative, eye-catching
- **Colors**: Vibrant, artistic
- **Elements**: Patterns, textures, artistic designs

### Branded & Themed

#### Company Branding
- **Use case**: Corporate communications, marketing
- **Style**: Branded, consistent
- **Colors**: Company colors
- **Elements**: Logos, brand elements, company messaging

#### Seasonal Themes
- **Use case**: Holiday content, seasonal campaigns
- **Style**: Themed, festive
- **Colors**: Seasonal colors
- **Elements**: Holiday decorations, seasonal imagery

## Performance Optimization

### For GPU Users (Recommended)

```bash
# 1080p processing with custom background
./build/bgremover_gpu --background-image high_res_background.jpg
# Performance: 28-32 FPS
# VRAM Usage: ~1.8GB
```

### For CPU Users

```bash
# CPU processing with custom background
./build/bgremover_gpu --background-image optimized_background.jpg
# Performance: 2-3 FPS
# Consider using lower resolution backgrounds for better performance
```

### Optimization Tips

1. **Image Format**: Use JPG for best compression/quality ratio
2. **Resolution**: Match your video resolution (1080p for 1080p video)
3. **File Size**: Keep backgrounds under 10MB for fast loading
4. **Complexity**: Simple backgrounds process faster than complex ones
5. **Color Depth**: Standard 24-bit color is sufficient

## Advanced Usage

### Scripting Multiple Backgrounds

Create a script for quick background switching:

```bash
#!/bin/bash
# background_switcher.sh

case "$1" in
    "office")
        ./build/bgremover_gpu --background-image office.jpg --vcam
        ;;
    "studio")
        ./build/bgremover_gpu --background-image studio.jpg --vcam
        ;;
    "casual")
        ./build/bgremover_gpu --background-image cafe.jpg --vcam
        ;;
    *)
        echo "Usage: $0 {office|studio|casual}"
        ;;
esac
```

### Batch Processing Multiple Videos

```bash
# Process multiple videos with different backgrounds
./build/bgremover_gpu meeting1.mp4 --background-image office.jpg
./build/bgremover_gpu interview.mp4 --background-image studio.jpg
./build/bgremover_gpu presentation.mp4 --background-image conference.jpg
```

### Virtual Camera with Background Rotation

```bash
# Set up virtual camera with different backgrounds for different apps
./build/bgremover_gpu --vcam --background-image zoom_background.jpg    # For Zoom
./build/bgremover_gpu --vcam --background-image teams_background.jpg   # For Teams
./build/bgremover_gpu --vcam --background-image meet_background.jpg    # For Google Meet
```

## Troubleshooting

### Common Issues

#### "Failed to load background image"
**Causes:**
- File path doesn't exist
- Unsupported file format
- Corrupted image file
- Insufficient permissions

**Solutions:**
```bash
# Check if file exists
ls -la background.jpg

# Try different format
./build/bgremover_gpu --background-image background.png

# Check permissions
chmod 644 background.jpg
```

#### Poor Quality Output
**Causes:**
- Low resolution background image
- Highly compressed image
- Aspect ratio mismatch

**Solutions:**
- Use higher resolution background (1920x1080+)
- Use less compression (higher JPG quality)
- Ensure background has good contrast with subject

#### Performance Issues
**Causes:**
- Very large background files
- Complex background images
- Insufficient GPU memory

**Solutions:**
- Reduce background file size
- Simplify background design
- Close other GPU applications

### Performance Monitoring

```bash
# Monitor GPU usage during background processing
nvidia-smi --loop-ms=1000

# Check FPS output in console
# Look for: "GPU Performance: 28-32 FPS" with custom backgrounds
```

## Best Practices

### Image Preparation
1. **Resolution**: Use 1920x1080 or higher
2. **File Size**: Keep under 10MB for optimal performance
3. **Format**: JPG for photos, PNG for graphics with transparency
4. **Quality**: Use high-quality compression settings
5. **Contrast**: Ensure good contrast between subject and background elements

### Professional Usage
1. **Consistency**: Use consistent backgrounds across similar meetings
2. **Context**: Match background to meeting purpose and audience
3. **Branding**: Incorporate company branding where appropriate
4. **Testing**: Always test backgrounds before important meetings
5. **Backup**: Keep multiple backgrounds ready for different scenarios

### Content Creation
1. **Engagement**: Use eye-catching backgrounds for content
2. **Branding**: Consistent backgrounds for brand recognition
3. **Variety**: Rotate backgrounds to keep content fresh
4. **Quality**: High-resolution backgrounds for professional appearance
5. **Copyright**: Ensure you have rights to use background images

## Technical Details

### Processing Pipeline
1. **Image Loading**: Background image loaded at startup
2. **Auto-scaling**: Resized to match video resolution
3. **Real-time Blending**: Combined with processed video frame-by-frame
4. **Output**: Final composited image displayed/sent to virtual camera

### Memory Usage
- **1080p Background**: ~6MB in memory
- **Processing Buffer**: ~24MB for 1080p frame
- **Total VRAM**: ~1.8GB for 1080p custom background processing
- **CPU Memory**: ~100MB for background storage

### Supported Resolutions
- **720p**: 1280x720 (HD)
- **1080p**: 1920x1080 (Full HD)
- **1440p**: 2560x1440 (2K/QHD)
- **4K**: 3840x2160 (Ultra HD) - Experimental
- **Any Resolution**: Automatically scaled

## Future Enhancements

### Planned Features
- **Animated backgrounds**: Support for GIF and video backgrounds
- **Background effects**: Blur, tint, and other effects on custom backgrounds
- **Background mixing**: Blend multiple backgrounds
- **Real-time switching**: Change backgrounds without restart
- **Background templates**: Built-in professional background library

### Experimental Features
- **AI-generated backgrounds**: Automatically generated backgrounds
- **Background effects**: Dynamic lighting and effects
- **Green screen mode**: Traditional chroma key replacement
- **Background blurring**: Combine custom background with blur effects

## Examples & Templates

### Professional Background Collection
```bash
# Modern office
./build/bgremover_gpu --background-image modern_office_1.jpg

# Conference room
./build/bgremover_gpu --background-image conference_room.jpg

# Co-working space
./build/bgremover_gpu --background-image coworking_space.jpg

# Home office
./build/bgremover_gpu --background-image home_office.jpg
```

### Creative Background Collection
```bash
# Photography studio
./build/bgremover_gpu --background-image photo_studio_white.jpg

# Nature scene
./build/bgremover_gpu --background-image forest_clear.jpg

# Abstract art
./build/bgremover_gpu --background-image abstract_blue.jpg

# Cityscape
./build/bgremover_gpu --background-image city_night.jpg
```

### Seasonal Backgrounds
```bash
# Spring theme
./build/bgremover_gpu --background-image spring_garden.jpg

# Summer theme
./build/bgremover_gpu --background-image beach_sunset.jpg

# Fall theme
./build/bgremover_gpu --background-image autumn_leaves.jpg

# Winter theme
./build/bgremover_gpu --background-image winter_mountains.jpg
```

---

**Ready to enhance your video calls with custom backgrounds!** 🎬✨

For more information, see the main README.md or run `./build/bgremover_gpu --help` for command-line options.