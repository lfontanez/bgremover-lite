#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bgremover");

// Function to display help and usage information
void showUsage(const std::string& program_name) {
    std::cout << "BGRemover Lite - CPU Background Removal\n";
    std::cout << "Usage: " << program_name << " [OPTIONS] [video_source]\n";
    std::cout << "\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help              Show this help message\n";
    std::cout << "  --no-blur               Disable background blur\n";
    std::cout << "  --no-background-blur    Disable background blur (alternative)\n";
    std::cout << "  --no-preview            Disable preview window\n";
    std::cout << "  --blur-low              Use low blur intensity (7x7 kernel)\n";
    std::cout << "  --blur-mid              Use medium blur intensity (15x15 kernel) [default]\n";
    std::cout << "  --blur-high             Use high blur intensity (25x25 kernel)\n";
    std::cout << "  --background-image PATH # Replace background with image (e.g. --background-image background.jpg)\n";
    std::cout << "  --bg-image PATH         # Short form for background image\n";
    std::cout << "\n";
    std::cout << "Arguments:\n";
    std::cout << "  video_source            Video file path or device number (default: 0 for webcam)\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << "                    # Use webcam with default settings\n";
    std::cout << "  " << program_name << " --no-blur          # Disable background blur\n";
    std::cout << "  " << program_name << " --blur-high        # Use high blur intensity\n";
    std::cout << "  " << program_name << " --background-image background.jpg  # Use custom background\n";
    std::cout << "  " << program_name << " video.mp4          # Process video file\n";
    std::cout << "  " << program_name << " 1                  # Use device 1 as input\n";
}

// Function to show current settings
void showCurrentSettings(bool blur_enabled, const std::string& blur_level, const std::string& background_image, bool show_preview) {
    std::cout << "Current settings:\n";
    if (!background_image.empty()) {
        std::cout << "  Background replacement: " << background_image << " (ENABLED)\n";
    } else {
        std::cout << "  Background blur: " << (blur_enabled ? "Enabled" : "Disabled") << "\n";
        if (blur_enabled) {
            std::cout << "  Blur intensity: " << blur_level << "\n";
            Size kernel_size;
            if (blur_level == "low") kernel_size = Size(7, 7);
            else if (blur_level == "high") kernel_size = Size(25, 25);
            else kernel_size = Size(15, 15);  // mid
            std::cout << "  Kernel size: " << kernel_size.width << "x" << kernel_size.height << "\n";
        }
    }
    std::cout << "  Preview window: " << (show_preview ? "Enabled" : "Disabled") << "\n";
    std::cout << "\n";
}

// Background replacement blend - use custom background image
Mat replace_background(const Mat& frame, const Mat& mask, const Mat& background) {
    if (background.empty()) {
        return frame;  // Fallback to original frame
    }
    
    // Resize background to match frame size
    Mat background_resized;
    resize(background, background_resized, frame.size());
    
    // Clean, simple mask (0 or 255)
    Mat mask_clean = (mask > 0.5);
    
    // Create output by combining person (from frame) and background
    Mat output = background_resized.clone();
    
    // Copy person from original frame using mask
    frame.copyTo(output, mask_clean);
    
    return output;
}

// Run inference and return properly isolated mask
Mat run_inference(Ort::Session& session, const Mat& img) {
    Mat resized;
    resize(img, resized, Size(320, 320));
    resized.convertTo(resized, CV_32F, 1.0f / 255.0f);
    Mat blob = dnn::blobFromImage(resized);

    array<int64_t, 4> shape = {1, 3, 320, 320};
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, blob.ptr<float>(), blob.total(),
        shape.data(), shape.size());

    // ✅ Correct way for ONNX Runtime >= 1.15
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
    auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
    input_names.push_back(input_name_alloc.get());
    output_names.push_back(output_name_alloc.get());

    auto outputs = session.Run(Ort::RunOptions{nullptr},
                               input_names.data(), &input_tensor, 1,
                               output_names.data(), 1);

    float* data = outputs.front().GetTensorMutableData<float>();
    Mat mask(320, 320, CV_32F, data);
    resize(mask, mask, img.size());

    // ✅ FIXED: Proper mask isolation
    normalize(mask, mask, 0.0, 1.0, NORM_MINMAX);
    
    // Sharpen the separation between person and background
    threshold(mask, mask, 0.4, 1.0, THRESH_BINARY);
    
    // Apply soft edges to prevent harsh borders
    GaussianBlur(mask, mask, Size(7, 7), 0);
    
    return mask;
}

int main(int argc, char** argv) {
    // Default command-line arguments
    std::string source = "0";
    bool show_preview = true;
    bool blur_enabled = true;
    std::string blur_level = "mid";
    std::string background_image = "";
    
    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            showUsage(argv[0]);
            return 0;
        } else if (arg == "--no-blur" || arg == "--no-background-blur") {
            blur_enabled = false;
        } else if (arg == "--no-preview") {
            show_preview = false;
        } else if (arg == "--blur-low") {
            blur_level = "low";
        } else if (arg == "--blur-mid") {
            blur_level = "mid";
        } else if (arg == "--blur-high") {
            blur_level = "high";
        } else if ((arg == "--background-image" || arg == "--bg-image") && i + 1 < argc) {
            background_image = argv[++i];
        } else if (i == 1 && arg != "--no-blur" && arg != "--no-background-blur" && 
                   arg != "--blur-low" && arg != "--blur-mid" && arg != "--blur-high" &&
                   arg != "--background-image" && arg != "--bg-image" &&
                   arg != "-h" && arg != "--help") {
            source = arg;  // This is the video source
        }
    }
    
    // Load background image if specified
    Mat background_mat;
    if (!background_image.empty()) {
        background_mat = imread(background_image, IMREAD_COLOR);
        if (background_mat.empty()) {
            cerr << "❌ Failed to load background image: " << background_image << "\n";
            return 1;
        } else {
            cout << "✅ Loaded background image: " << background_image 
                 << " (" << background_mat.cols << "x" << background_mat.rows << ")\n";
        }
    }
    
    // Show current settings
    showCurrentSettings(blur_enabled, blur_level, background_image, show_preview);
    
    VideoCapture cap;
    if (source == "0") {
        std::cout << "Attempting to open webcam (device 0)...\n";
        cap.open(0);
    } else {
        std::cout << "Opening video file: " << source << "...\n";
        cap.open(source);
    }
    if (!cap.isOpened()) { 
        std::cerr << "Cannot open video source: " << source << "\n";
        return 1; 
    }
    std::cout << "Video source opened successfully!\n";
    
    // Get video properties
    double fps = cap.get(CAP_PROP_FPS);
    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    
    // Performance advisory for high resolutions
    if (width * height >= 1920 * 1080) {
        std::cout << "🔍 High resolution detected (" << width << "x" << height << ") - consider GPU version for optimal performance\n";
    } else if (width * height >= 1280 * 720) {
        std::cout << "📺 HD resolution detected (" << width << "x" << height << ")\n";
    }
    
    std::cout << "Video properties - FPS: " << fps << ", Resolution: " 
         << width << "x" << height << "\n";

    std::cout << "Loading U²-Net model...\n";
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    Ort::Session session(env, "models/u2net.onnx", opts);
    std::cout << "Model loaded successfully!\n";

    std::cout << "Press ESC to quit\n";
    Mat frame;
    bool first_frame = true;
    
    // Pre-allocate buffers for efficient 1080p processing
    Mat mask, output, blurred;
    
    // Get blur kernel size
    Size blur_kernel;
    if (!blur_enabled) {
        blur_kernel = Size(0, 0);
    } else if (blur_level == "low") {
        blur_kernel = Size(7, 7);
    } else if (blur_level == "high") {
        blur_kernel = Size(25, 25);
    } else {
        blur_kernel = Size(15, 15);  // mid
    }
    
    while (cap.read(frame)) {
        // Ensure consistent frame size for 1080p processing
        if (frame.cols != width || frame.rows != height) {
            resize(frame, frame, Size(width, height));
            if (first_frame) {
                std::cout << "📐 Resized input to: " << width << "x" << height << " for processing" << endl;
            }
        }
        
        // Run inference on downsampled frame (U²-Net processes at 320x320)
        Mat downsampled;
        resize(frame, downsampled, Size(320, 320));
        Mat mask_320 = run_inference(session, downsampled);
        
        // Resize mask back to full resolution
        resize(mask_320, mask, frame.size());
        
        // Apply blur only if enabled
        if (blur_enabled && blur_kernel.width > 0) {
            // Pre-allocate blurred frame for efficiency
            if (first_frame || blurred.empty() || blurred.size() != frame.size()) {
                blurred.create(frame.size(), frame.type());
            }
            // Apply Gaussian blur with specified kernel
            GaussianBlur(frame, blurred, blur_kernel, 0);
        }
        
        // Ensure output frame is properly allocated
        if (first_frame || output.empty() || output.size() != frame.size()) {
            output.create(frame.size(), frame.type());
        }
        
        // Optimized pixel-level blend
        Mat mask_clean = (mask > 0.5);
        if (!background_image.empty()) {
            // Use background replacement
            output = replace_background(frame, mask, background_mat);
        } else if (blur_enabled) {
            // Use blur effect
            frame.copyTo(output, mask_clean);
            output.setTo(blurred, ~mask_clean);
        } else {
            // No blur - just show original frame
            frame.copyTo(output);
        }

        // Create window title with processing settings
        std::string processing_info;
        if (!background_image.empty()) {
            processing_info = " (Custom Background)";
        } else if (blur_enabled) {
            processing_info = blur_level == "low" ? " (Low Blur)" : 
                             blur_level == "high" ? " (High Blur)" : " (Mid Blur)";
        } else {
            processing_info = " (No Blur)";
        }
        std::string window_title = "Background Removed (CPU)" + processing_info;
        if (show_preview) {
            imshow(window_title, output);
            if (waitKey(1) == 27) break;  // ESC
        }
        
        first_frame = false;
    }

    // Cleanup
    cap.release();
    if (show_preview) {
        destroyAllWindows();
    }
    
    // Release pre-allocated matrices
    frame.release();
    mask.release();
    output.release();
    blurred.release();
    
    std::cout << "🧹 CPU processing cleanup completed" << endl;
    
    return 0;
}
