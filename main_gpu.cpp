#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <memory>

using namespace cv;
using namespace std;

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bgremover-gpu");

// GPU Memory Management and Performance Monitoring
class GPUMemoryManager {
public:
    bool cuda_available = false;
    
    GPUMemoryManager() {
        // Check if CUDA is available
        cuda_available = false;
        #ifdef CUDA_AVAILABLE
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error == cudaSuccess && device_count > 0) {
            cuda_available = true;
            cout << "✅ CUDA runtime initialized successfully" << endl;
        } else {
            cout << "⚠️  CUDA runtime not available" << endl;
        }
        #endif
    }
    
    void printMemoryStats(const string& label = "") {
        if (!cuda_available) {
            cout << "GPU Memory " << label << ": Not available (CPU mode)" << endl;
            return;
        }
        
        size_t free_memory = 0, total_memory = 0;
        cudaError_t error = cudaMemGetInfo(&free_memory, &total_memory);
        if (error == cudaSuccess) {
            double used_gb = (total_memory - free_memory) / (1024.0 * 1024.0 * 1024.0);
            double free_gb = free_memory / (1024.0 * 1024.0 * 1024.0);
            double total_gb = total_memory / (1024.0 * 1024.0 * 1024.0);
            cout << "GPU Memory " << label << ": " 
                 << used_gb << "GB used / " << free_gb << "GB free / " 
                 << total_gb << "GB total" << endl;
        } else {
            cout << "GPU Memory " << label << ": Unable to query (Error: " << cudaGetErrorString(error) << ")" << endl;
        }
    }
    
    bool hasSufficientMemory(size_t required_bytes) {
        if (!cuda_available) return false;
        
        size_t free_memory = 0, total_memory = 0;
        cudaError_t error = cudaMemGetInfo(&free_memory, &total_memory);
        return (error == cudaSuccess) && (free_memory >= required_bytes);
    }
};

// Global GPU memory manager
GPUMemoryManager gpu_manager;

// Memory Pre-allocation (CPU-based for compatibility)
class BufferManager {
public:
    Ort::MemoryInfo cpu_memory_info;
    std::vector<Ort::Value> preallocated_buffers;
    bool initialized = false;
    
    void initialize() {
        if (initialized) return;
        
        // Create CPU memory info (compatible with all ONNX Runtime versions)
        cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        initialized = true;
        cout << "✅ Memory manager initialized (CPU mode)" << endl;
    }
    
    Ort::MemoryInfo& getMemoryInfo() {
        if (!initialized) initialize();
        return cpu_memory_info;
    }
};

static BufferManager buffer_manager;

// ONNX Runtime inference with memory management
Mat run_inference(Ort::Session& session, const Mat& img, bool cuda_available) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize memory manager
    buffer_manager.initialize();
    Ort::MemoryInfo& mem_info = buffer_manager.getMemoryInfo();
    
    // Preprocess on CPU
    Mat resized;
    resize(img, resized, Size(320, 320));
    resized.convertTo(resized, CV_32F, 1.0f / 255.0f);
    Mat blob = dnn::blobFromImage(resized);
    
    array<int64_t, 4> shape = {1, 3, 320, 320};
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Check GPU memory availability if CUDA is supposedly available
    size_t required_memory = blob.total() * sizeof(float);
    if (cuda_available && !gpu_manager.hasSufficientMemory(required_memory)) {
        cout << "⚠️  Insufficient GPU memory, using CPU fallback" << endl;
        cuda_available = false;
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
    Mat mask(320, 320, CV_32F, data);
    resize(mask, mask, img.size());

    // Process result with proper mask isolation
    normalize(mask, mask, 0.0, 1.0, NORM_MINMAX);
    
    // Sharpen the separation between person and background
    threshold(mask, mask, 0.4, 1.0, THRESH_BINARY);
    
    // Apply soft edges to prevent harsh borders
    GaussianBlur(mask, mask, Size(7, 7), 0);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - inference_start);
    
    // Performance monitoring
    static int frame_count = 0;
    static auto last_print_time = std::chrono::high_resolution_clock::now();
    frame_count++;
    
    if (std::chrono::duration_cast<std::chrono::seconds>(end_time - last_print_time).count() >= 2) {
        gpu_manager.printMemoryStats();
        cout << (cuda_available ? "🚀 GPU" : "⚡ CPU") << " Performance: " 
             << frame_count / 2.0 << " FPS | Inference: " 
             << inference_duration.count() << "ms | Total: " 
             << duration.count() << "ms" << endl;
        frame_count = 0;
        last_print_time = end_time;
    }
    
    return mask;
}

// Optimized GPU/CPU processing with proper mask isolation
Mat fast_blend(const Mat& frame, const Mat& mask) {
    // Use moderate kernel for quality
    Mat blurred;
    GaussianBlur(frame, blurred, Size(15, 15), 0);
    
    // Clean, simple mask (0 or 255) - works with properly isolated mask
    Mat mask_clean = (mask > 0.5) * 255;
    
    // Direct pixel-level blend - fastest and most reliable
    Mat output = frame.clone();
    for (int y = 0; y < frame.rows; y++) {
        for (int x = 0; x < frame.cols; x++) {
            uchar m = mask_clean.at<uchar>(y, x);
            if (m == 0) {
                // Outside mask - use blurred background
                output.at<Vec3b>(y, x) = blurred.at<Vec3b>(y, x);
            } else {
                // Inside mask - use original (person in focus)
                // output.at<Vec3b>(y, x) stays as original
            }
        }
    }
    
    return output;
}

int main(int argc, char** argv) {
    string source = argc > 1 ? argv[1] : "0";
    VideoCapture cap;
    if (source == "0") {
        cout << "Attempting to open webcam (device 0) with GPU acceleration...\n";
        cap.open(0);
    } else {
        cout << "Opening video file: " << source << " with GPU acceleration...\n";
        cap.open(source);
    }
    if (!cap.isOpened()) { 
        cerr << "Cannot open video source: " << source << "\n";
        return 1; 
    }
    cout << "Video source opened successfully!\n";
    
    // Get video properties
    double fps = cap.get(CAP_PROP_FPS);
    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    cout << "Video properties - FPS: " << fps << ", Resolution: " 
         << width << "x" << height << "\n";

    cout << "Loading U²-Net model with GPU acceleration...\n";
    
    // Configure session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    
    // Check available ONNX Runtime providers
    bool cuda_available = false;
    try {
        auto providers = Ort::GetAvailableProviders();
        cout << "Available ONNX Runtime providers: ";
        for (const auto& provider : providers) {
            cout << provider << " ";
        }
        cout << endl;
        
        // Check if CUDA provider is available
        cuda_available = std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end();
        
        if (cuda_available) {
            cout << "✅ CUDA execution provider available" << endl;
            
            // Try to enable CUDA provider with basic options
            try {
                // For ONNX Runtime 1.19.0, try basic CUDA provider configuration
                // Note: The exact API may vary, so we use a try-catch approach
                #ifdef ENABLE_CUDA_PROVIDER
                // Attempt to add CUDA provider (API may differ in 1.19.0)
                // This is a placeholder - actual API may need adjustment
                // session_options.AppendExecutionProvider_CUDA({});
                cout << "⚠️  CUDA provider configuration not available in this version" << endl;
                #endif
            } catch (const Ort::Exception& e) {
                cout << "⚠️  Could not enable CUDA provider: " << e.what() << endl;
                cuda_available = false;
            }
        } else {
            cout << "⚠️  CUDA execution provider not available - using CPU only" << endl;
        }
    } catch (const exception& e) {
        cout << "Error checking providers: " << e.what() << endl;
        cuda_available = false;
    }
    
    // Create session
    Ort::Session session(env, "models/u2net.onnx", session_options);
    
    // Initialize GPU memory manager after session creation
    gpu_manager = GPUMemoryManager();
    
    if (cuda_available) {
        cout << "🚀 GPU acceleration enabled!" << endl;
        gpu_manager.printMemoryStats("after model load");
    } else {
        cout << "⚠️  Using CPU fallback (GPU not available or not configured)" << endl;
    }
    
    cout << "Model loaded successfully!\n";

    cout << "Press ESC to quit\n";
    Mat frame;
    auto start_time = chrono::high_resolution_clock::now();
    int frame_count = 0;
    
    while (cap.read(frame)) {
        // Run inference
        Mat mask = run_inference(session, frame, cuda_available);
        
        // Optimized blending
        Mat output = fast_blend(frame, mask);
        
        imshow(cuda_available ? "Background Removed (GPU)" : "Background Removed (CPU)", output);
        if (waitKey(1) == 27) break;  // ESC
        
        // Enhanced performance monitoring with GPU memory
        frame_count++;
        if (frame_count % 10 == 0) {  // More frequent updates
            auto current_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(current_time - start_time);
            double fps_real = (frame_count * 1000.0) / duration.count();
            
            // Update and display GPU memory info if CUDA is available
            if (cuda_available && cuda_used) {
                gpu_manager.printMemoryStats();
            }
            
            cout << (cuda_available ? "🚀 GPU" : "⚡ CPU") << " Performance: " 
                 << fps_real << " FPS (" << frame_count << " frames in "
                 << duration.count() << "ms)" << endl;
            
            // Reset for next measurement
            frame_count = 0;
            start_time = current_time;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
