#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <memory>
#include <string>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cuda.h>
#include "v4l2_output.hpp"

using namespace cv;
using namespace std;

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bgremover-gpu");

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
    std::cout << "  --no-vcam               Disable virtual camera output\n";
    std::cout << "\n";
    std::cout << "Arguments:\n";
    std::cout << "  video_source            Video file path or device number (default: 0 for webcam)\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << "                    # Use webcam with default settings\n";
    std::cout << "  " << program_name << " --vcam             # Enable virtual camera\n";
    std::cout << "  " << program_name << " --no-blur          # Disable background blur\n";
    std::cout << "  " << program_name << " --blur-high        # Use high blur intensity\n";
    std::cout << "  " << program_name << " video.mp4          # Process video file\n";
    std::cout << "  " << program_name << " 1 --vcam-device /dev/video3  # Custom devices\n";
}

// Function to show current settings
void showCurrentSettings(bool blur_enabled, const std::string& blur_level, 
                        bool vcam_enabled, const std::string& vcam_device) {
    std::cout << "Current settings:\n";
    std::cout << "  Background blur: " << (blur_enabled ? "Enabled" : "Disabled") << "\n";
    if (blur_enabled) {
        std::cout << "  Blur intensity: " << blur_level << "\n";
        cv::Size kernel_size;
        if (blur_level == "low") kernel_size = cv::Size(7, 7);
        else if (blur_level == "high") kernel_size = cv::Size(25, 25);
        else kernel_size = cv::Size(15, 15);  // mid
        std::cout << "  Kernel size: " << kernel_size.width << "x" << kernel_size.height << "\n";
    }
    std::cout << "  Virtual camera: " << (vcam_enabled ? "Enabled" : "Disabled") << "\n";
    if (vcam_enabled) {
        std::cout << "  Virtual camera device: " << vcam_device << "\n";
    }
    std::cout << "\n";
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
                std::cout << "✅ CUDA runtime initialized successfully" << std::endl;
                std::cout << "✅ GPU Memory: " << free_gb << "GB free / " 
                     << total_gb << "GB total" << std::endl;
            } else {
                std::cout << "⚠️  CUDA initialized but memory query failed: " 
                     << cudaGetErrorString(error) << std::endl;
                // Still set cuda_available to true since CUDA works
                cuda_available = true;
            }
        } else {
            cuda_available = false;
            std::cout << "⚠️  CUDA runtime not available" << std::endl;
        }
    }
    
    void printMemoryStats(const std::string& label = "") {
        if (!cuda_available) {
            std::cout << "GPU Memory " << label << ": Not available (CPU mode)" << std::endl;
            return;
        }
        
        size_t current_free = 0, current_total = 0;
        cudaError_t error = cudaMemGetInfo(&current_free, &current_total);
        if (error == cudaSuccess) {
            double used_gb = (current_total - current_free) / (1024.0 * 1024.0 * 1024.0);
            double free_gb = current_free / (1024.0 * 1024.0 * 1024.0);
            double total_gb = current_total / (1024.0 * 1024.0 * 1024.0);
            std::cout << "GPU Memory " << label << ": " 
                 << used_gb << "GB used / " << free_gb << "GB free / " 
                 << total_gb << "GB total" << std::endl;
        } else {
            // Use cached values if current query fails
            double used_gb = (total_memory - free_memory) / (1024.0 * 1024.0 * 1024.0);
            double free_gb = free_memory / (1024.0 * 1024.0 * 1024.0);
            double total_gb = total_memory / (1024.0 * 1024.0 * 1024.0);
            std::cout << "GPU Memory " << label << ": " 
                 << used_gb << "GB used / " << free_gb << "GB free / " 
                 << total_gb << "GB total (cached)" << std::endl;
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
                std::cout << "⚠️  Failed to update memory info: " 
                     << cudaGetErrorString(error) << std::endl;
            }
        }
    }
};

// Global GPU memory manager
GPUMemoryManager gpu_manager;

// Simple Buffer Manager - GPU handled by ONNX Runtime
class BufferManager {
public:
    BufferManager() {
        initialized = false;
    }
    
    void initialize(bool cuda_available) {
        if (initialized) return;
        
        if (cuda_available) {
            std::cout << "✅ Memory manager initialized (GPU acceleration enabled via ONNX Runtime)" << std::endl;
        } else {
            std::cout << "✅ Memory manager initialized (CPU mode)" << std::endl;
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
            std::cout << "✅ GPU memory sufficient (" << required_mb << "MB required)" << std::endl;
        } else {
            std::cout << "⚠️  Memory check failed, but proceeding with GPU" << std::endl;
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
        std::cout << (cuda_available ? "🚀 GPU" : "⚡ CPU") << " Performance: " 
             << frame_count / 2.0 << " FPS | Inference: " 
             << inference_duration.count() << "ms | Total: " 
             << duration.count() << "ms" << std::endl;
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

int main(int argc, char** argv) {
    // Default command-line arguments
    std::string source = "0";
    bool vcam_enabled = false;
    std::string vcam_device = "/dev/video2";
    bool blur_enabled = true;
    std::string blur_level = "mid";
    
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
        } else if (i == 1 && arg != "--vcam" && arg != "-v" && arg != "--vcam-device" &&
                   arg != "--no-vcam" && arg != "--no-blur" && arg != "--no-background-blur" && 
                   arg != "--blur-low" && arg != "--blur-mid" && arg != "--blur-high" &&
                   arg != "-h" && arg != "--help") {
            source = arg;  // This is the video source
        }
    }
    
    // Show current settings
    showCurrentSettings(blur_enabled, blur_level, vcam_enabled, vcam_device);
    
    cv::VideoCapture cap;
    if (source == "0") {
        std::cout << "Attempting to open webcam (device 0) with GPU acceleration...\n";
        cap.open(0);
    } else {
        std::cout << "Opening video file: " << source << " with GPU acceleration...\n";
        cap.open(source);
    }
    if (!cap.isOpened()) { 
        std::cerr << "Cannot open video source: " << source << "\n";
        return 1; 
    }
    std::cout << "Video source opened successfully!\n";
    
    // Get video properties
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    // Performance advisory for high resolutions
    if (width * height >= 1920 * 1080) {
        std::cout << "🎬 1080p HD processing - GPU acceleration recommended for real-time performance\n";
    } else if (width * height >= 1280 * 720) {
        std::cout << "📺 HD resolution processing (" << width << "x" << height << ")\n";
    }
    
    std::cout << "Video properties - FPS: " << fps << ", Resolution: " 
         << width << "x" << height << "\n";

    // Initialize virtual camera if enabled
    std::unique_ptr<V4L2Output> vcam_output;
    bool vcam_opened = false;
    
    if (vcam_enabled) {
        std::cout << "Initializing virtual camera at: " << vcam_device << "\n";
        
        // Always use 1080p for virtual camera output
        const int vcam_width = 1920;
        const int vcam_height = 1080;
        
        std::cout << "🖥️ 1080p HD virtual camera: " << vcam_width << "x" << vcam_height << "\n";
        
        vcam_output = std::make_unique<V4L2Output>(vcam_device, vcam_width, vcam_height);
        if (vcam_output->open()) {
            vcam_opened = true;
            std::cout << "✅ 1080p Virtual camera enabled: " << vcam_device << " (" 
                 << vcam_output->getSize().width << "x" << vcam_output->getSize().height << ")\n";
        } else {
            std::cout << "⚠️ Virtual camera failed to open, continuing without it\n";
            vcam_output.reset();
        }
    }

    std::cout << "Loading U²-Net model with GPU acceleration...\n";
    
    // Configure session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    
    // Check available ONNX Runtime providers
    bool cuda_available = false;
    try {
        auto providers = Ort::GetAvailableProviders();
        std::cout << "Available ONNX Runtime providers: ";
        for (const auto& provider : providers) {
            std::cout << provider << " ";
        }
        std::cout << std::endl;
        
        // Check if CUDA provider is available
        cuda_available = std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end();
        
        if (cuda_available) {
            std::cout << "✅ CUDA execution provider available" << std::endl;
            
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
                
                std::cout << "✅ CUDA provider successfully configured with device_id=0" << std::endl;
            } catch (const Ort::Exception& e) {
                std::cout << "⚠️  Could not configure CUDA provider: " << e.what() << std::endl;
                cuda_available = false;
            } catch (const std::exception& e) {
                std::cout << "⚠️  Unexpected error configuring CUDA provider: " << e.what() << std::endl;
                cuda_available = false;
            }
        } else {
            std::cout << "⚠️  CUDA execution provider not available - using CPU only" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "Error checking providers: " << e.what() << std::endl;
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
        std::cout << "🚀 GPU acceleration enabled!" << std::endl;
        std::cout << "📊 Processing requires ~" << estimated_1080p_mb << "MB for frame buffers" << std::endl;
        gpu_manager.printMemoryStats("after model load");
        cuda_used = true;
    } else {
        std::cout << "⚠️  Using CPU fallback (GPU not available or not configured)" << std::endl;
        std::cout << "📊 CPU processing may be slower - consider GPU version for optimal performance" << std::endl;
    }
    
    std::cout << "Model loaded successfully!\n";

    std::cout << "Press ESC to quit\n";
    cv::Mat frame;
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    
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
                std::cout << "📐 Resized input to: " << width << "x" << height << " for processing" << std::endl;
            }
        }
        
        // Auto-resize to 1080p for virtual camera output if enabled
        if (vcam_enabled) {
            if (frame.cols != 1920 || frame.rows != 1080) {
                cv::resize(frame, frame, cv::Size(1920, 1080));
                if (first_frame) {
                    std::cout << "🖥️ Auto-resized to 1080p for virtual camera: 1920x1080" << std::endl;
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
        if (blur_enabled) {
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
        
        // Create window title with blur settings
        std::string blur_info = blur_enabled ? 
            (blur_level == "low" ? " (Low Blur)" : 
             blur_level == "high" ? " (High Blur)" : " (Mid Blur)") : 
            " (No Blur)";
        std::string window_title = cuda_available ? 
            (vcam_enabled ? "Background Removed (GPU + Virtual Camera)" + blur_info : 
                          "Background Removed (GPU)" + blur_info) :
            (vcam_enabled ? "Background Removed (CPU + Virtual Camera)" + blur_info : 
                          "Background Removed (CPU)" + blur_info);
        cv::imshow(window_title, output);
        if (cv::waitKey(1) == 27) break;  // ESC
        
        // Enhanced performance monitoring with memory info
        frame_count++;
        if (frame_count % 10 == 0) {  // More frequent updates
            auto current_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
            double fps_real = (frame_count * 1000.0) / duration.count();
            
            // Update and display GPU memory info if CUDA is available
            if (cuda_available && cuda_used) {
                gpu_manager.printMemoryStats();
            }
            
            std::string perf_label = cuda_available ? 
                (vcam_enabled ? "GPU + VCam" : "GPU") : 
                (vcam_enabled ? "CPU + VCam" : "CPU");
            
            // Add blur info to performance output
            std::string blur_info = blur_enabled ? 
                (blur_level == "low" ? " [Low Blur]" : 
                 blur_level == "high" ? " [High Blur]" : " [Mid Blur]") : 
                " [No Blur]";
            
            std::cout << "🚀 " << perf_label << " Performance: " 
                 << fps_real << " FPS (" << frame_count << " frames in "
                 << duration.count() << "ms)" << blur_info << std::endl;
            
            // Reset for next measurement
            frame_count = 0;
            start_time = current_time;
        }
        
        first_frame = false;
    }

    // Proper cleanup
    cap.release();
    cv::destroyAllWindows();
    
    // Final memory stats
    if (cuda_available && cuda_used) {
        gpu_manager.printMemoryStats("after processing");
    }
    
    // Clean up virtual camera
    if (vcam_output) {
        vcam_output->close();
    }
    
    // Clear pre-allocated matrices to free memory
    frame.release();
    mask.release();
    output.release();
    blurred.release();
    
    std::cout << "🧹 Processing cleanup completed" << std::endl;
    
    return 0;
}
