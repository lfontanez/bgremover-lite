#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <memory>
#include <string>
#include <iomanip>
#include <sstream>

using namespace cv;
using namespace std;

// Global stats file stream
std::ofstream stats_file_stream;

// Logging level control
enum class LogLevel {
    QUIET = 0,
    NORMAL = 1,
    VERBOSE = 2
};

LogLevel current_log_level = LogLevel::NORMAL;

// Logging helper functions
void logMessage(LogLevel level, const std::string& message) {
    if (level <= current_log_level) {
        std::cout << message << std::endl;
    }
}

void logError(const std::string& message) {
    std::cerr << "❌ " << message << std::endl;
}

void logSuccess(const std::string& message) {
    if (current_log_level >= LogLevel::NORMAL) {
        std::cout << "✅ " << message << std::endl;
    }
}

void logWarning(const std::string& message) {
    if (current_log_level >= LogLevel::NORMAL) {
        std::cout << "⚠️  " << message << std::endl;
    }
}

void logInfo(const std::string& message) {
    if (current_log_level >= LogLevel::VERBOSE) {
        std::cout << "ℹ️  " << message << std::endl;
    }
}

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
    std::cout << "  -q, --quiet             Minimal output (only errors)\n";
    std::cout << "  --verbose               Detailed output (current behavior)\n";
    std::cout << "  --stats-file PATH       Save performance stats to file\n";
    std::cout << "  --overlay-stats         Show real-time stats as video overlay\n";
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
    std::cout << "  " << program_name << " --quiet            # Minimal console output\n";
    std::cout << "  " << program_name << " --stats-file stats.txt  # Save performance stats\n";
    std::cout << "  " << program_name << " --overlay-stats    # Show stats on video\n";
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

// Function to draw stats overlay on the video frame
void drawStatsOverlay(cv::Mat& frame, double fps, const std::string& blur_level, const std::string& background_image) {
    // Create semi-transparent background rectangle
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, cv::Point(10, 10), cv::Point(350, 120), cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(frame, 0.7, overlay, 0.3, 0, frame);
    
    // Set text properties
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.6;
    cv::Scalar text_color(255, 255, 255); // White text
    int thickness = 1;
    int line_type = cv::LINE_AA;
    
    int y_offset = 30;
    
    // Draw FPS
    std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
    cv::putText(frame, fps_text, cv::Point(20, y_offset), font, font_scale, text_color, thickness, line_type);
    y_offset += 20;
    
    // Draw processing mode
    std::string mode_text = "Mode: CPU";
    cv::putText(frame, mode_text, cv::Point(20, y_offset), font, font_scale, text_color, thickness, line_type);
    y_offset += 20;
    
    // Draw blur level or background type
    std::string effect_text;
    if (!background_image.empty()) {
        effect_text = "Effect: Custom Background";
    } else {
        effect_text = "Effect: " + blur_level + " blur";
    }
    cv::putText(frame, effect_text, cv::Point(20, y_offset), font, font_scale, text_color, thickness, line_type);
    y_offset += 20;
    
    // Draw memory info (CPU mode - N/A)
    cv::putText(frame, "GPU Memory: N/A", cv::Point(20, y_offset), font, font_scale, text_color, thickness, line_type);
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
    bool quiet_mode = false;
    bool verbose_mode = true;
    std::string stats_file = "";
    bool overlay_stats = false;
    
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
        } else if (arg == "-q" || arg == "--quiet") {
            quiet_mode = true;
            verbose_mode = false;
        } else if (arg == "--verbose") {
            verbose_mode = true;
            quiet_mode = false;
        } else if (arg == "--stats-file" && i + 1 < argc) {
            stats_file = argv[++i];
        } else if (arg == "--overlay-stats") {
            overlay_stats = true;
        } else if (i == 1 && arg != "--no-blur" && arg != "--no-background-blur" && 
                   arg != "--blur-low" && arg != "--blur-mid" && arg != "--blur-high" &&
                   arg != "--background-image" && arg != "--bg-image" &&
                   arg != "-q" && arg != "--quiet" && arg != "--verbose" &&
                   arg != "--stats-file" && arg != "--overlay-stats" &&
                   arg != "-h" && arg != "--help") {
            source = arg;  // This is the video source
        }
    }
    
    // Set logging level based on flags
    if (quiet_mode) {
        current_log_level = LogLevel::QUIET;
    } else if (verbose_mode) {
        current_log_level = LogLevel::VERBOSE;
    } else {
        current_log_level = LogLevel::NORMAL;
    }
    
    // Open stats file if specified
    if (!stats_file.empty()) {
        stats_file_stream.open(stats_file, std::ios::out | std::ios::trunc);
        if (stats_file_stream.is_open()) {
            // Write CSV header (matching GPU version format)
            stats_file_stream << "Timestamp,FPS,GPU_Memory_Used_GB,GPU_Memory_Total_GB,Processing_Time_ms,Frame_Count,Mode\n";
            logSuccess("Stats file opened: " + stats_file);
        } else {
            logWarning("Failed to open stats file: " + stats_file);
        }
    }
    
    // Load background image if specified
    Mat background_mat;
    if (!background_image.empty()) {
        background_mat = imread(background_image, IMREAD_COLOR);
        if (background_mat.empty()) {
            logError("Failed to load background image: " + background_image);
            return 1;
        } else {
            logSuccess("Loaded background image: " + background_image + 
                      " (" + std::to_string(background_mat.cols) + "x" + 
                      std::to_string(background_mat.rows) + ")");
        }
    }
    
    // Show current settings (only if not in quiet mode)
    if (!quiet_mode) {
        showCurrentSettings(blur_enabled, blur_level, background_image, show_preview);
    }
    
    VideoCapture cap;
    if (source == "0") {
        logInfo("Attempting to open webcam (device 0)...");
        cap.open(0);
    } else {
        logInfo("Opening video file: " + source + "...");
        cap.open(source);
    }
    if (!cap.isOpened()) { 
        logError("Cannot open video source: " + source);
        return 1; 
    }
    logSuccess("Video source opened successfully!");
    
    // Get video properties
    double fps = cap.get(CAP_PROP_FPS);
    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    
    // Performance advisory for high resolutions
    if (width * height >= 1920 * 1080) {
        logInfo("High resolution detected (" + std::to_string(width) + "x" + 
               std::to_string(height) + ") - consider GPU version for optimal performance");
    } else if (width * height >= 1280 * 720) {
        logInfo("HD resolution detected (" + std::to_string(width) + "x" + 
               std::to_string(height) + ")");
    }
    
    logInfo("Video properties - FPS: " + std::to_string(fps) + ", Resolution: " + 
           std::to_string(width) + "x" + std::to_string(height));

    logInfo("Loading U²-Net model...");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    Ort::Session session(env, "models/u2net.onnx", opts);
    logSuccess("Model loaded successfully!");

    if (!quiet_mode) {
        std::cout << "Press ESC to quit\n";
    }
    Mat frame;
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
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
                logInfo("Resized input to: " + std::to_string(width) + "x" + 
                       std::to_string(height) + " for processing");
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
        
        // Optimized pixel-level blend with comprehensive safety checks
        Mat mask_clean = (mask > 0.5);
        if (!background_image.empty()) {
            // Use background replacement
            output = replace_background(frame, mask, background_mat);
        } else if (blur_enabled) {
            // Ensure mask_clean is valid
            if (mask_clean.empty() || mask_clean.type() != CV_8U) {
                // Fallback: use original frame without processing
                frame.copyTo(output);
                logWarning("Invalid mask detected - using original frame");
            } else {
                // Use blur effect with safe mask operations
                frame.copyTo(output, mask_clean);
                
                // Safety check for blurred frame and create inverted mask safely
                if (!blurred.empty() && blurred.size() == output.size() && blurred.type() == output.type()) {
                    // Create inverted mask safely
                    Mat inverted_mask;
                    bitwise_not(mask_clean, inverted_mask);
                    output.setTo(blurred, inverted_mask);
                } else {
                    // Fallback: use original frame for background
                    Mat inverted_mask;
                    bitwise_not(mask_clean, inverted_mask);
                    frame.copyTo(output, inverted_mask);
                }
            }
        } else {
            // No blur - just show original frame
            frame.copyTo(output);
        }

        // Draw stats overlay if enabled
        if (overlay_stats) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
            double fps_real = (frame_count * 1000.0) / duration.count();
            drawStatsOverlay(output, fps_real, blur_level, background_image);
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
        
        // Enhanced performance monitoring with logging level control
        frame_count++;
        if (frame_count % 10 == 0) {  // More frequent updates
            auto current_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
            double fps_real = (frame_count * 1000.0) / duration.count();
            
            // Add processing info to performance output
            std::string processing_info;
            if (!background_image.empty()) {
                processing_info = " [Custom Background]";
            } else if (blur_enabled) {
                processing_info = blur_level == "low" ? " [Low Blur]" : 
                                 blur_level == "high" ? " [High Blur]" : " [Mid Blur]";
            } else {
                processing_info = " [No Blur]";
            }
            
            // Add preview info to performance output
            if (!show_preview) {
                processing_info += " [No Preview]";
            }
            
            logMessage(LogLevel::NORMAL, "⚡ CPU Performance: " + 
                      std::to_string(fps_real) + " FPS (" + std::to_string(frame_count) + 
                      " frames in " + std::to_string(duration.count()) + "ms)" + 
                      processing_info);
            
            // Write to stats file if open
            if (stats_file_stream.is_open()) {
                // Get current timestamp
                auto now = std::chrono::system_clock::now();
                auto now_time = std::chrono::system_clock::to_time_t(now);
                std::stringstream timestamp_ss;
                timestamp_ss << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S");
                
                // Write CSV line (matching GPU version format with GPU memory as 0.0 for CPU)
                stats_file_stream << timestamp_ss.str() << ","
                                 << std::fixed << std::setprecision(2) << fps_real << ","
                                 << std::fixed << std::setprecision(3) << 0.0 << ","
                                 << std::fixed << std::setprecision(3) << 0.0 << ","
                                 << duration.count() << ","
                                 << frame_count << ","
                                 << "CPU"
                                 << "\n";
                stats_file_stream.flush(); // Ensure data is written immediately
            }
            
            // Reset for next measurement
            frame_count = 0;
            start_time = current_time;
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
    
    // Close stats file if open
    if (stats_file_stream.is_open()) {
        stats_file_stream.close();
        logInfo("Stats file closed: " + stats_file);
    }
    
    logMessage(LogLevel::NORMAL, "🧹 CPU processing cleanup completed");
    
    return 0;
}
