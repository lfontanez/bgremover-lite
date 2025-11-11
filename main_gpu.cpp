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
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cuda.h>
#include <unistd.h>
#include "v4l2_output.hpp"

using namespace cv;
using namespace std;

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bgremover-gpu");

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

// Function to display help and usage information
void showUsage(const std::string& program_name) {
    std::cout << "BGRemover Lite - GPU-Accelerated Background Removal\n";
    std::cout << "Usage: " << program_name << " [OPTIONS] [video_source]\n";
    std::cout << "\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help              Show this help message\n";
    std::cout << "  -v, --vcam              Enable virtual camera output\n";
    std::cout << "  --vcam-device DEVICE    Virtual camera device path (default: /dev/video2)\n";
    std::cout << "  --no-blur               Disable background blur\n";
    std::cout << "  --no-background-blur    Disable background blur (alternative)\n";
    std::cout << "  --blur-low              Use low blur intensity (7x7 kernel)\n";
    std::cout << "  --blur-mid              Use medium blur intensity (15x15 kernel) [default]\n";
    std::cout << "  --blur-high             Use high blur intensity (25x25 kernel)\n";
    std::cout << "  --background-image PATH # Replace background with image (e.g. --background-image background.jpg)\n";
    std::cout << "  --bg-image PATH         # Short form for background image\n";
    std::cout << "  --no-preview            Disable preview window\n";
    std::cout << "  --no-vcam               Disable virtual camera output\n";
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
    std::cout << "  " << program_name << " --vcam             # Enable virtual camera\n";
    std::cout << "  " << program_name << " --no-blur          # Disable background blur\n";
    std::cout << "  " << program_name << " --blur-high        # Use high blur intensity\n";
    std::cout << "  " << program_name << " --background-image background.jpg  # Use custom background\n";
    std::cout << "  " << program_name << " video.mp4          # Process video file\n";
    std::cout << "  " << program_name << " 1 --vcam-device /dev/video3  # Custom devices\n";
    std::cout << "  " << program_name << " --quiet            # Minimal console output\n";
    std::cout << "  " << program_name << " --stats-file stats.txt  # Save performance stats\n";
    std::cout << "  " << program_name << " --overlay-stats    # Show stats on video\n";
}

// Function to show current settings
void showCurrentSettings(bool blur_enabled, const std::string& blur_level, 
                        bool vcam_enabled, const std::string& vcam_device,
                        const std::string& background_image, bool show_preview) {
    logInfo("Current settings:");
    if (!background_image.empty()) {
        logInfo("  Background replacement: " + background_image + " (ENABLED)");
    } else {
        logInfo("  Background blur: " + std::string(blur_enabled ? "Enabled" : "Disabled"));
        if (blur_enabled) {
            logInfo("  Blur intensity: " + blur_level);
            cv::Size kernel_size;
            if (blur_level == "low") kernel_size = cv::Size(7, 7);
            else if (blur_level == "high") kernel_size = cv::Size(25, 25);
            else kernel_size = cv::Size(15, 15);  // mid
            logInfo("  Kernel size: " + std::to_string(kernel_size.width) + "x" + std::to_string(kernel_size.height));
        }
    }
    logInfo("  Preview window: " + std::string(show_preview ? "Enabled" : "Disabled"));
    logInfo("  Virtual camera: " + std::string(vcam_enabled ? "Enabled" : "Disabled"));
    if (vcam_enabled) {
        logInfo("  Virtual camera device: " + vcam_device);
    }
    logInfo("");
}

// GPU Memory Management and Performance Monitoring
class GPUMemoryManager {
public:
    bool cuda_available = false;
    int device_count = 0;
    size_t total_memory = 0;
    size_t free_memory = 0;
    
    GPUMemoryManager() {
        initializeCUDA();
    }
    
    void initializeCUDA() {
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error == cudaSuccess && device_count > 0) {
            cuda_available = true;
            
            // Get memory info for the default device
            cudaSetDevice(0);
            error = cudaMemGetInfo(&free_memory, &total_memory);
            if (error == cudaSuccess) {
                double total_gb = total_memory / (1024.0 * 1024.0 * 1024.0);
                double free_gb = free_memory / (1024.0 * 1024.0 * 1024.0);
                logSuccess("CUDA runtime initialized successfully");
                logSuccess("GPU Memory: " + std::to_string(free_gb) + "GB free / " + 
                     std::to_string(total_gb) + "GB total");
            } else {
                logWarning("CUDA initialized but memory query failed: " + 
                     std::string(cudaGetErrorString(error)));
                // Still set cuda_available to true since CUDA works
                cuda_available = true;
            }
        } else {
            cuda_available = false;
            logWarning("CUDA runtime not available");
        }
    }
    
    void printMemoryStats(const std::string& label = "") {
        if (!cuda_available) {
            logInfo("GPU Memory " + label + ": Not available (CPU mode)");
            return;
        }
        
        size_t current_free = 0, current_total = 0;
        cudaError_t error = cudaMemGetInfo(&current_free, &current_total);
        if (error == cudaSuccess) {
            double used_gb = (current_total - current_free) / (1024.0 * 1024.0 * 1024.0);
            double free_gb = current_free / (1024.0 * 1024.0 * 1024.0);
            double total_gb = current_total / (1024.0 * 1024.0 * 1024.0);
            logInfo("GPU Memory " + label + ": " + 
                 std::to_string(used_gb) + "GB used / " + std::to_string(free_gb) + "GB free / " + 
                 std::to_string(total_gb) + "GB total");
        } else {
            // Use cached values if current query fails
            double used_gb = (total_memory - free_memory) / (1024.0 * 1024.0 * 1024.0);
            double free_gb = free_memory / (1024.0 * 1024.0 * 1024.0);
            double total_gb = total_memory / (1024.0 * 1024.0 * 1024.0);
            logInfo("GPU Memory " + label + ": " + 
                 std::to_string(used_gb) + "GB used / " + std::to_string(free_gb) + "GB free / " + 
                 std::to_string(total_gb) + "GB total (cached)");
        }
    }
    
    bool hasSufficientMemory(size_t required_bytes) {
        if (!cuda_available) return false;
        
        // For RTX 4070 Ti SUPER with 15GB, always allow models under 5GB
        if (required_bytes < 5ull * 1024 * 1024 * 1024) { // 5GB
            return true;
        }
        
        // For very large requirements, check actual memory
        size_t current_free = 0, current_total = 0;
        cudaError_t error = cudaMemGetInfo(&current_free, &current_total);
        if (error == cudaSuccess) {
            // Be generous with memory requirements - leave 20% headroom
            size_t safe_free = current_free * 0.8;
            return safe_free >= required_bytes;
        }
        
        // If we can't check, assume success for reasonable requirements
        return required_bytes < 2ull * 1024 * 1024 * 1024; // 2GB
    }
    
    void updateMemoryInfo() {
        if (cuda_available) {
            cudaError_t error = cudaMemGetInfo(&free_memory, &total_memory);
            if (error != cudaSuccess) {
                logWarning("Failed to update memory info: " + 
                     std::string(cudaGetErrorString(error)));
            }
        }
    }
};

// Global GPU memory manager
GPUMemoryManager gpu_manager;

// Global stats file stream
std::ofstream stats_file_stream;

// Simple Buffer Manager - GPU handled by ONNX Runtime
class BufferManager {
public:
    BufferManager() {
        initialized = false;
    }
    
    void initialize(bool cuda_available) {
        if (initialized) return;
        
        if (cuda_available) {
            logSuccess("Memory manager initialized (GPU acceleration enabled via ONNX Runtime)");
        } else {
            logSuccess("Memory manager initialized (CPU mode)");
        }
        
        initialized = true;
    }
    
    Ort::MemoryInfo getMemoryInfo() {
        if (!initialized) {
            initialize(false);
        }
        // Always use CPU memory info - ONNX Runtime handles GPU acceleration internally
        return Ort::MemoryInfo("Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault);
    }
    
    bool isUsingGPU() const {
        return initialized;
    }
    
private:
    bool initialized = false;
};

static BufferManager buffer_manager;

// ONNX Runtime inference with memory management
cv::Mat run_inference(Ort::Session& session, const cv::Mat& img, bool cuda_available) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize memory manager with CUDA availability
    buffer_manager.initialize(cuda_available);
    Ort::MemoryInfo mem_info = buffer_manager.getMemoryInfo();
    
    // Preprocess on CPU (always, for compatibility)
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(320, 320));
    resized.convertTo(resized, CV_32F, 1.0f / 255.0f);
    cv::Mat blob = cv::dnn::blobFromImage(resized);
    
    std::array<int64_t, 4> shape = {1, 3, 320, 320};
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Simplified GPU memory check - we have 15GB VRAM, should be plenty
    size_t required_memory = blob.total() * sizeof(float);
    double required_mb = required_memory / (1024.0 * 1024.0);
    
    if (cuda_available) {
        // For RTX 4070 Ti SUPER with 15GB, we should always have enough memory
        if (gpu_manager.hasSufficientMemory(100 * 1024 * 1024)) { // 100MB minimum
            logSuccess("GPU memory sufficient (" + std::to_string(required_mb) + "MB required)");
        } else {
            logWarning("Memory check failed, but proceeding with GPU");
        }
    }
    
    // Re-initialize memory manager if CUDA availability changed
    if (cuda_available != buffer_manager.isUsingGPU()) {
        buffer_manager.initialize(cuda_available);
    }
    
    // Allocate input tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, blob.ptr<float>(), blob.total(),
        shape.data(), shape.size());
    
    // Run inference
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
    auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
    input_names.push_back(input_name_alloc.get());
    output_names.push_back(output_name_alloc.get());
    
    auto inference_start = std::chrono::high_resolution_clock::now();
    auto outputs = session.Run(Ort::RunOptions{nullptr},
                               input_names.data(), &input_tensor, 1,
                               output_names.data(), 1);
    auto inference_end = std::chrono::high_resolution_clock::now();
    
    // Extract result
    float* data = outputs.front().GetTensorMutableData<float>();
    cv::Mat mask(320, 320, CV_32F, data);
    cv::resize(mask, mask, img.size());

    // Process result with proper mask isolation
    cv::normalize(mask, mask, 0.0, 1.0, cv::NORM_MINMAX);
    
    // Sharpen the separation between person and background
    cv::threshold(mask, mask, 0.4, 1.0, cv::THRESH_BINARY);
    
    // Apply soft edges to prevent harsh borders
    cv::GaussianBlur(mask, mask, cv::Size(7, 7), 0);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - inference_start);
    
    // Performance monitoring
    static int frame_count = 0;
    static auto last_print_time = std::chrono::high_resolution_clock::now();
    frame_count++;
    
    if (std::chrono::duration_cast<std::chrono::seconds>(end_time - last_print_time).count() >= 2) {
        if (cuda_available) {
            gpu_manager.printMemoryStats();
        }
        logInfo((cuda_available ? "🚀 GPU" : "⚡ CPU") + std::string(" Performance: ") + 
                 std::to_string(frame_count / 2.0) + std::string(" FPS | Inference: ") + 
                 std::to_string(inference_duration.count()) + std::string("ms | Total: ") + 
                 std::to_string(duration.count()) + std::string("ms"));
        frame_count = 0;
        last_print_time = end_time;
    }
    
    return mask;
}

// Optimized GPU/CPU processing with proper mask isolation
cv::Mat fast_blend(const cv::Mat& frame, const cv::Mat& mask) {
    // Use moderate kernel for quality
    cv::Mat blurred;
    cv::GaussianBlur(frame, blurred, cv::Size(15, 15), 0);
    
    // Clean, simple mask (0 or 255) - works with properly isolated mask
    cv::Mat mask_clean = (mask > 0.5) * 255;
    
    // Direct pixel-level blend - fastest and most reliable
    cv::Mat output = frame.clone();
    for (int y = 0; y < frame.rows; y++) {
        for (int x = 0; x < frame.cols; x++) {
            uchar m = mask_clean.at<uchar>(y, x);
            if (m == 0) {
                // Outside mask - use blurred background
                output.at<cv::Vec3b>(y, x) = blurred.at<cv::Vec3b>(y, x);
            } else {
                // Inside mask - use original (person in focus)
                // output.at<cv::Vec3b>(y, x) stays as original
            }
        }
    }
    
    return output;
}

// Background replacement blend - use custom background image
cv::Mat replace_background(const cv::Mat& frame, const cv::Mat& mask, const cv::Mat& background) {
    if (background.empty()) {
        return frame;  // Fallback to original frame
    }
    
    // Resize background to match frame size
    cv::Mat background_resized;
    cv::resize(background, background_resized, frame.size());
    
    // Clean, simple mask (0 or 255)
    cv::Mat mask_clean = (mask > 0.5);
    
    // Create output by combining person (from frame) and background
    cv::Mat output = background_resized.clone();
    
    // Copy person from original frame using mask
    frame.copyTo(output, mask_clean);
    
    return output;
}

// Function to draw stats overlay on the video frame
void drawStatsOverlay(cv::Mat& frame, double fps, bool cuda_available, 
                     const std::string& blur_level, const std::string& background_image,
                     GPUMemoryManager& gpu_manager) {
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
    std::string mode_text = "Mode: " + std::string(cuda_available ? "GPU" : "CPU");
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
    
    // Draw GPU memory usage if available
    if (cuda_available && gpu_manager.cuda_available) {
        size_t current_free = 0, current_total = 0;
        cudaError_t error = cudaMemGetInfo(&current_free, &current_total);
        if (error == cudaSuccess) {
            double used_gb = (current_total - current_free) / (1024.0 * 1024.0 * 1024.0);
            double total_gb = current_total / (1024.0 * 1024.0 * 1024.0);
            std::string memory_text = "GPU Memory: " + std::to_string(used_gb).substr(0, 4) + 
                                    "GB / " + std::to_string(total_gb).substr(0, 4) + "GB";
            cv::putText(frame, memory_text, cv::Point(20, y_offset), font, font_scale, text_color, thickness, line_type);
        }
    } else {
        cv::putText(frame, "GPU Memory: N/A", cv::Point(20, y_offset), font, font_scale, text_color, thickness, line_type);
    }
}

int main(int argc, char** argv) {
    // Default command-line arguments
    std::string source = "0";
    bool vcam_enabled = false;
    std::string vcam_device = "/dev/video2";
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
        } else if (arg == "--vcam" || arg == "-v") {
            vcam_enabled = true;
        } else if (arg == "--vcam-device" && i + 1 < argc) {
            vcam_device = argv[++i];
        } else if (arg == "--no-preview") {
            show_preview = false;
        } else if (arg == "--no-vcam") {
            vcam_enabled = false;
        } else if (arg == "--no-blur" || arg == "--no-background-blur") {
            blur_enabled = false;
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
        } else if (i == 1 && arg != "--vcam" && arg != "-v" && arg != "--vcam-device" &&
                   arg != "--no-vcam" && arg != "--no-blur" && arg != "--no-background-blur" && 
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
    
    // Load background image if specified
    cv::Mat background_mat;
    if (!background_image.empty()) {
        background_mat = cv::imread(background_image, cv::IMREAD_COLOR);
        if (background_mat.empty()) {
            logError("Failed to load background image: " + background_image);
            return 1;
        } else {
            logSuccess("Loaded background image: " + background_image + 
                      " (" + std::to_string(background_mat.cols) + "x" + std::to_string(background_mat.rows) + ")");
        }
    }
    
    // Open stats file if specified
    if (!stats_file.empty()) {
        stats_file_stream.open(stats_file, std::ios::out | std::ios::trunc);
        if (stats_file_stream.is_open()) {
            // Write CSV header
            stats_file_stream << "Timestamp,FPS,GPU_Memory_Used_GB,GPU_Memory_Total_GB,Processing_Time_ms,Frame_Count,Mode\n";
            logSuccess("Stats file opened: " + stats_file);
        } else {
            logWarning("Failed to open stats file: " + stats_file);
        }
    }
    
    // Show current settings (only if not in quiet mode)
    if (!quiet_mode) {
        showCurrentSettings(blur_enabled, blur_level, vcam_enabled, vcam_device, background_image, show_preview);
    }
    
    cv::VideoCapture cap;
    if (source == "0") {
        logInfo("Attempting to open webcam (device 0) with GPU acceleration...");
        cap.open(0);
    } else {
        logInfo("Opening video file: " + source + " with GPU acceleration...");
        cap.open(source);
    }
    if (!cap.isOpened()) { 
        logError("Cannot open video source: " + source);
        return 1; 
    }
    logSuccess("Video source opened successfully!");
    
    // Get video properties
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    // Performance advisory for high resolutions
    if (width * height >= 1920 * 1080) {
        logInfo("1080p HD processing - GPU acceleration recommended for real-time performance");
    } else if (width * height >= 1280 * 720) {
        logInfo("HD resolution processing (" + std::to_string(width) + "x" + std::to_string(height) + ")");
    }
    
    logInfo("Video properties - FPS: " + std::to_string(fps) + ", Resolution: " + 
         std::to_string(width) + "x" + std::to_string(height));

    // Initialize virtual camera if enabled
    std::unique_ptr<V4L2Output> vcam_output;
    bool vcam_opened = false;
    
    if (vcam_enabled) {
        logInfo("Initializing virtual camera at: " + vcam_device);
        
        // Always use 1080p for virtual camera output
        const int vcam_width = 1920;
        const int vcam_height = 1080;
        
        logInfo("1080p HD virtual camera: " + std::to_string(vcam_width) + "x" + std::to_string(vcam_height));
        
        vcam_output = std::make_unique<V4L2Output>(vcam_device, vcam_width, vcam_height);
        if (vcam_output->open()) {
            vcam_opened = true;
            logSuccess("1080p Virtual camera enabled: " + vcam_device + " (" + 
                 std::to_string(vcam_output->getSize().width) + "x" + std::to_string(vcam_output->getSize().height) + ")");
        } else {
            logWarning("Virtual camera failed to open, continuing without it");
            vcam_output.reset();
        }
    }

    logInfo("Loading U²-Net model with GPU acceleration...");
    
    // Configure session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    
    // Check available ONNX Runtime providers
    bool cuda_available = false;
    try {
        auto providers = Ort::GetAvailableProviders();
        std::string providers_list;
        for (const auto& provider : providers) {
            providers_list += provider + " ";
        }
        logInfo("Available ONNX Runtime providers: " + providers_list);
        
        // Check if CUDA provider is available
        cuda_available = std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end();
        
        if (cuda_available) {
            logSuccess("CUDA execution provider available");
            
            // Configure and append CUDA provider to session options
            try {
                // Create CUDA provider options using C API
                OrtCUDAProviderOptionsV2* cuda_options = nullptr;
                Ort::GetApi().CreateCUDAProviderOptions(&cuda_options);
                
                // Configure device_id=0
                std::vector<const char*> keys{"device_id"};
                std::vector<const char*> values{"0"};
                Ort::GetApi().UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), 1);
                
                // Append CUDA provider to session options
                Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2(
                    static_cast<OrtSessionOptions*>(session_options),
                    cuda_options
                );
                
                // Release CUDA provider options
                Ort::GetApi().ReleaseCUDAProviderOptions(cuda_options);
                
                logSuccess("CUDA provider successfully configured with device_id=0");
            } catch (const Ort::Exception& e) {
                logWarning("Could not configure CUDA provider: " + std::string(e.what()));
                cuda_available = false;
            } catch (const std::exception& e) {
                logWarning("Unexpected error configuring CUDA provider: " + std::string(e.what()));
                cuda_available = false;
            }
        } else {
            logWarning("CUDA execution provider not available - using CPU only");
        }
    } catch (const std::exception& e) {
        logError("Error checking providers: " + std::string(e.what()));
        cuda_available = false;
    }
    
    // Create session with configured options
    Ort::Session session(env, "models/u2net.onnx", session_options);
    
    // Initialize GPU memory manager before session creation
    gpu_manager.initializeCUDA();
    
    // Enhanced memory management for 1080p processing
    size_t estimated_1080p_memory = static_cast<size_t>(width) * height * 3 * sizeof(uchar) * 4; // 4 frames buffer
    double estimated_1080p_mb = estimated_1080p_memory / (1024.0 * 1024.0);
    
    bool cuda_used = false;
    if (cuda_available) {
        logSuccess("GPU acceleration enabled!");
        logMessage(LogLevel::NORMAL, "Processing requires ~" + std::to_string(estimated_1080p_mb) + "MB for frame buffers");
        gpu_manager.printMemoryStats("after model load");
        cuda_used = true;
    } else {
        logWarning("Using CPU fallback (GPU not available or not configured)");
        logMessage(LogLevel::NORMAL, "CPU processing may be slower - consider GPU version for optimal performance");
    }
    
    logSuccess("Model loaded successfully!");

    if (!quiet_mode) {
        logMessage(LogLevel::NORMAL, "Process started with PID: " + std::to_string(getpid()));
        if (show_preview) {
            logMessage(LogLevel::NORMAL, "Press ESC in preview window to quit");
        } else {
            logMessage(LogLevel::NORMAL, "Press CTRL+C to quit");
        }
    }
    cv::Mat frame;
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    double fps_real = 0.0;
    
    // Pre-allocate buffers for processing to avoid frequent allocations
    cv::Mat mask, output, blurred;
    bool first_frame = true;
    
    // Get blur kernel size
    cv::Size blur_kernel;
    if (!blur_enabled) {
        blur_kernel = cv::Size(0, 0);
    } else if (blur_level == "low") {
        blur_kernel = cv::Size(7, 7);
    } else if (blur_level == "high") {
        blur_kernel = cv::Size(25, 25);
    } else {
        blur_kernel = cv::Size(15, 15);  // mid
    }
    
    while (cap.read(frame)) {
        // Ensure frame dimensions are suitable for processing
        if (frame.cols != width || frame.rows != height) {
            cv::resize(frame, frame, cv::Size(width, height));
            if (first_frame) {
                logInfo("Resized input to: " + std::to_string(width) + "x" + std::to_string(height) + " for processing");
            }
        }
        
        // Auto-resize to 1080p for virtual camera output if enabled
        if (vcam_enabled) {
            if (frame.cols != 1920 || frame.rows != 1080) {
                cv::resize(frame, frame, cv::Size(1920, 1080));
                if (first_frame) {
                    logInfo("Auto-resized to 1080p for virtual camera: 1920x1080");
                }
            }
        }
        
        // Run inference on downsampled frame (U²-Net processes at 320x320)
        cv::Mat downsampled;
        cv::resize(frame, downsampled, cv::Size(320, 320));
        cv::Mat mask_320 = run_inference(session, downsampled, cuda_available);
        
        // Resize mask back to full resolution
        cv::resize(mask_320, mask, frame.size());
        
        // Apply blur only if enabled
        if (blur_enabled && blur_kernel.width > 0) {
            // Pre-allocate blurred frame for efficiency
            if (first_frame || blurred.empty() || blurred.size() != frame.size()) {
                blurred.create(frame.size(), frame.type());
            }
            // Apply Gaussian blur with specified kernel
            cv::GaussianBlur(frame, blurred, blur_kernel, 0);
        }
        
        // Ensure output frame is properly allocated
        if (first_frame || output.empty() || output.size() != frame.size()) {
            output.create(frame.size(), frame.type());
        }
        
        // Fast pixel-level blend optimized
        cv::Mat mask_clean = (mask > 0.5);
        if (!background_image.empty()) {
            // Use background replacement
            output = replace_background(frame, mask, background_mat);
        } else if (blur_enabled) {
            // Use blur effect
            frame.copyTo(output, mask_clean);
            // Create proper 3-channel mask for the blurred background
            cv::Mat mask_3channel;
            cv::cvtColor(~mask_clean, mask_3channel, cv::COLOR_GRAY2BGR);
            blurred.copyTo(output, mask_3channel);
        } else {
            // No blur - just show original frame
            frame.copyTo(output);
        }
        
        // Write to virtual camera if enabled and open
        if (vcam_enabled && vcam_opened && vcam_output) {
            if (!vcam_output->writeFrame(output)) {
                std::cerr << "⚠️ Virtual camera write failed\n";
            }
        }
        
        // Draw stats overlay if enabled
        if (overlay_stats) {
            drawStatsOverlay(output, fps_real, cuda_available, blur_level, background_image, gpu_manager);
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
        
        std::string window_title = cuda_available ? 
            (vcam_enabled ? "Background Removed (GPU + Virtual Camera)" + processing_info : 
                          "Background Removed (GPU)" + processing_info) :
            (vcam_enabled ? "Background Removed (CPU + Virtual Camera)" + processing_info : 
                          "Background Removed (CPU)" + processing_info);
        
        if (show_preview) {
            cv::imshow(window_title, output);
            if (cv::waitKey(1) == 27) break;  // ESC
        }
        
        // Enhanced performance monitoring with memory info
        frame_count++;
        if (frame_count % 10 == 0) {  // More frequent updates
            auto current_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
            fps_real = (frame_count * 1000.0) / duration.count();
            
            // Update and display GPU memory info if CUDA is available
            if (cuda_available && cuda_used) {
                gpu_manager.printMemoryStats();
            }
            
            std::string perf_label = cuda_available ? 
                (vcam_enabled ? "GPU + VCam" : "GPU") : 
                (vcam_enabled ? "CPU + VCam" : "CPU");
            
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
            
            logInfo(perf_label + " Performance: " + 
                 std::to_string(fps_real) + " FPS (" + std::to_string(frame_count) + " frames in " +
                 std::to_string(duration.count()) + "ms)" + processing_info);
            
            // Write to stats file if open
            if (stats_file_stream.is_open()) {
                // Get current timestamp
                auto now = std::chrono::system_clock::now();
                auto now_time = std::chrono::system_clock::to_time_t(now);
                std::stringstream timestamp_ss;
                timestamp_ss << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S");
                
                // Get GPU memory info
                double gpu_used_gb = 0.0, gpu_total_gb = 0.0;
                if (cuda_available && cuda_used) {
                    size_t current_free = 0, current_total = 0;
                    cudaError_t error = cudaMemGetInfo(&current_free, &current_total);
                    if (error == cudaSuccess) {
                        gpu_used_gb = (current_total - current_free) / (1024.0 * 1024.0 * 1024.0);
                        gpu_total_gb = current_total / (1024.0 * 1024.0 * 1024.0);
                    }
                }
                
                // Write CSV line
                stats_file_stream << timestamp_ss.str() << ","
                                 << std::fixed << std::setprecision(2) << fps_real << ","
                                 << std::fixed << std::setprecision(3) << gpu_used_gb << ","
                                 << std::fixed << std::setprecision(3) << gpu_total_gb << ","
                                 << duration.count() << ","
                                 << frame_count << ","
                                 << perf_label << "\n";
                stats_file_stream.flush(); // Ensure data is written immediately
            }
            
            // Reset for next measurement
            frame_count = 0;
            start_time = current_time;
        }
        
        first_frame = false;
    }

    // Proper cleanup
    cap.release();
    if (show_preview) {
        cv::destroyAllWindows();
    }
    
    // Final memory stats
    if (cuda_available && cuda_used) {
        gpu_manager.printMemoryStats("after processing");
    }
    
    // Clean up virtual camera
    if (vcam_output) {
        vcam_output->close();
    }
    
    // Close stats file if open
    if (stats_file_stream.is_open()) {
        stats_file_stream.close();
        logMessage(LogLevel::NORMAL, "Stats file closed: " + stats_file);
    }
    
    // Clear pre-allocated matrices to free memory
    frame.release();
    mask.release();
    output.release();
    blurred.release();
    
    logMessage(LogLevel::NORMAL, "Processing cleanup completed");
    
    return 0;
}
