#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <dlfcn.h>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <memory>

using namespace cv;
using namespace std;

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bgremover-gpu");

// GPU Memory Management and Performance Monitoring
class GPUMemoryManager {
public:
    size_t total_memory = 0;
    size_t free_memory = 0;
    size_t used_memory = 0;
    
    void updateMemoryInfo() {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        free_memory = free;
        total_memory = total;
        used_memory = total - free;
    }
    
    void printMemoryStats(const string& label = "") {
        updateMemoryInfo();
        double used_gb = used_memory / (1024.0 * 1024.0 * 1024.0);
        double free_gb = free_memory / (1024.0 * 1024.0 * 1024.0);
        double total_gb = total_memory / (1024.0 * 1024.0 * 1024.0);
        cout << "GPU Memory " << label << ": " 
             << used_gb << "GB used / " << free_gb << "GB free / " 
             << total_gb << "GB total" << endl;
    }
    
    bool hasSufficientMemory(size_t required_bytes) {
        updateMemoryInfo();
        return free_memory >= required_bytes;
    }
};

// Global GPU memory manager
GPUMemoryManager gpu_manager;

// GPU Memory Pre-allocation
class GPUBufferManager {
public:
    Ort::MemoryInfo gpu_memory_info;
    std::vector<Ort::Value> preallocated_buffers;
    bool initialized = false;
    
    void initialize(int device_id = 0) {
        if (initialized) return;
        
        // Create GPU memory info
        try {
            gpu_memory_info = Ort::MemoryInfo::CreateCuda(OrtArenaAllocator, OrtMemTypeDefault, device_id);
        } catch (const Ort::Exception& e) {
            // Fallback to CPU if CUDA memory not available
            gpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            cout << "⚠️  Using CPU memory fallback: " << e.what() << endl;
        }
        initialized = true;
        cout << "✅ GPU memory manager initialized" << endl;
    }
    
    Ort::MemoryInfo& getMemoryInfo() {
        if (!initialized) initialize();
        return gpu_memory_info;
    }
};

static GPUBufferManager buffer_manager;

// GPU-accelerated ONNX inference with optimized memory management
Mat run_gpu_inference(Ort::Session& session, const Mat& img, bool cuda_available) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize GPU memory manager
    buffer_manager.initialize(0);
    Ort::MemoryInfo& gpu_mem_info = buffer_manager.getMemoryInfo();
    
    // Preprocess on CPU
    Mat resized;
    resize(img, resized, Size(320, 320));
    resized.convertTo(resized, CV_32F, 1.0f / 255.0f);
    Mat blob = dnn::blobFromImage(resized);
    
    array<int64_t, 4> shape = {1, 3, 320, 320};
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Check GPU memory availability
    size_t required_memory = blob.total() * sizeof(float);
    if (!gpu_manager.hasSufficientMemory(required_memory)) {
        cout << "⚠️  Insufficient GPU memory, using CPU fallback" << endl;
        cuda_available = false;
    }
    
    if (cuda_available && gpu_mem_info.mem_type == OrtMemType::OrtMemTypeDefault) {
        cout << "🚀 Using GPU execution with CUDA provider" << endl;
    }
    
    // Allocate input tensor on GPU if available
    Ort::Value input_tensor;
    if (cuda_available && gpu_mem_info.mem_type == OrtMemType::OrtMemTypeDefault) {
        input_tensor = Ort::Value::CreateTensor<float>(
            gpu_mem_info, blob.ptr<float>(), blob.total(),
            shape.data(), shape.size());
    } else {
        // Use CPU memory
        input_tensor = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
            blob.ptr<float>(), blob.total(),
            shape.data(), shape.size());
    }
    
    // Run inference on GPU (if available)
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
    
    // Extract result - try to keep on GPU if possible
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
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    
    // Initialize ONNX Runtime SessionOptions with CUDA support
    Ort::SessionOptions session_options;
    
    // Enhanced CUDA provider library detection with multiple paths
    vector<string> cuda_lib_paths = {
        "./onnxruntime/lib/libonnxruntime_providers_shared.so",
        "/usr/local/onnxruntime/lib/libonnxruntime_providers_shared.so",
        "/opt/onnxruntime/lib/libonnxruntime_providers_shared.so",
        "./libonnxruntime_providers_shared.so"
    };
    
    bool cuda_provider_available = false;
    string found_cuda_lib = "";
    
    // Check for CUDA provider library in multiple locations
    for (const auto& cuda_lib_path : cuda_lib_paths) {
        std::ifstream cuda_lib_file(cuda_lib_path);
        if (cuda_lib_file.good()) {
            cout << "✅ Found CUDA provider library at: " << cuda_lib_path << endl;
            cuda_provider_available = true;
            found_cuda_lib = cuda_lib_path;
            cuda_lib_file.close();
            break;
        }
    }
    
    if (cuda_provider_available) {
        // Add library path to system library path
        string lib_dir = found_cuda_lib.substr(0, found_cuda_lib.find_last_of("/"));
        setenv("LD_LIBRARY_PATH", (lib_dir + ":" + (getenv("LD_LIBRARY_PATH") ?: "")).c_str(), 1);
        
        // Try to load the CUDA provider library
        string lib_name = found_cuda_lib.substr(found_cuda_lib.find_last_of("/") + 1);
        void* cuda_lib = dlopen(lib_name.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (cuda_lib) {
            cout << "✅ CUDA provider library loaded successfully" << endl;
            dlclose(cuda_lib);
        } else {
            cout << "❌ Failed to load CUDA provider library: " << dlerror() << endl;
            cuda_provider_available = false;
        }
    } else {
        cout << "❌ CUDA provider library not found in any expected location" << endl;
    }
    
    // Check if CUDA is available in ONNX Runtime
    bool cuda_available = false;
    try {
        auto providers = Ort::GetAvailableProviders();
        cuda_available = std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end();
        cout << "Available ONNX Runtime providers: ";
        for (const auto& provider : providers) {
            cout << provider << " ";
        }
        cout << endl;
        
        // Initialize GPU memory manager
        try {
            cudaGetDeviceCount(reinterpret_cast<int*>(&cuda_available));
            if (cuda_available) {
                gpu_manager.updateMemoryInfo();
                cout << "✅ CUDA runtime initialized successfully" << endl;
            }
        } catch (...) {
            cout << "⚠️  CUDA runtime not available" << endl;
            cuda_available = false;
        }
        
        // If CUDA is available and provider library exists, explicitly enable it with optimized settings
        if (cuda_available && cuda_provider_available) {
            try {
                // Try to use CUDA provider with proper configuration
                // First, try the V2 API if available with memory pool optimization
                try {
                    #ifdef ORT_API_VERSION
                    #if ORT_API_VERSION >= 14
                    // Use V2 API for newer ONNX Runtime versions with memory optimization
                    Ort::CUDAProviderOptionsV2 cuda_options_v2;
                    // Enable memory arena and other optimizations
                    cuda_options_v2.enable_cuda_memory_arena = true;
                    cuda_options_v2.enable_cuda_mem_pattern = true;
                    cuda_options_v2.enable_duplicate_workspace = false;
                    cuda_options_v2.cudnn_conv_use_max_workspace = true;
                    session_options.AppendExecutionProvider_CUDA_V2(cuda_options_v2);
                    cout << "✅ CUDA execution provider V2 with memory optimization enabled" << endl;
                    #else
                    // Use legacy API for older versions
                    Ort::CUDAProviderOptions cuda_options;
                    // Legacy options (limited optimization)
                    session_options.AppendExecutionProvider_CUDA(cuda_options);
                    cout << "✅ CUDA execution provider enabled (legacy API)" << endl;
                    #endif
                    #else
                    // Default to legacy API
                    Ort::CUDAProviderOptions cuda_options;
                    session_options.AppendExecutionProvider_CUDA(cuda_options);
                    cout << "✅ CUDA execution provider enabled (default)" << endl;
                    #endif
                    
                    // Configure session options for optimal performance
                    session_options.SetIntraOpNumThreads(1);
                    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
                    session_options.SetInterOpNumThreads(1);
                    
                } catch (const Ort::Exception& e) {
                    // If V2 API fails, try the original API
                    try {
                        Ort::CUDAProviderOptions cuda_options;
                        session_options.AppendExecutionProvider_CUDA(cuda_options);
                        cout << "✅ CUDA execution provider enabled (fallback)" << endl;
                    } catch (const Ort::Exception& e2) {
                        cerr << "❌ Failed to enable CUDA execution provider: " << e2.what() << endl;
                        cuda_available = false;
                    } catch (const exception& e2) {
                        cerr << "❌ Generic error enabling CUDA provider: " << e2.what() << endl;
                        cuda_available = false;
                    }
                }
                
            } catch (const Ort::Exception& e) {
                cerr << "❌ Failed to enable CUDA execution provider: " << e.what() << endl;
                cuda_available = false;
            } catch (const exception& e) {
                cerr << "❌ Generic error enabling CUDA provider: " << e.what() << endl;
                cuda_available = false;
            }
        } else if (cuda_available && !cuda_provider_available) {
            cout << "⚠️  CUDA runtime available but provider library missing - may use limited fallback" << endl;
        } else {
            cout << "❌ CUDA execution provider not available" << endl;
        }
    } catch (const exception& e) {
        cout << "Error checking CUDA availability: " << e.what() << endl;
        cuda_available = false;
    }
    
    // Create session with proper CUDA support
    Ort::Session session(env, "models/u2net.onnx", session_options);
    
    // Verify CUDA provider is actually being used
    auto used_providers = session.GetExecutionProviders();
    bool cuda_used = false;
    for (const auto& provider : used_providers) {
        cout << "  Using provider: " << provider << endl;
        if (provider.find("CUDA") != string::npos) {
            cuda_used = true;
        }
    }
    
    if (cuda_used) {
        cout << "🚀 CUDA execution provider active - GPU acceleration enabled!" << endl;
        gpu_manager.printMemoryStats("after model load");
    } else {
        cout << "⚠️  CUDA provider not found in active execution providers" << endl;
        if (cuda_provider_available && cuda_available) {
            cout << "⚠️  Session created but CUDA provider may not be used - check configuration" << endl;
        } else {
            cout << "⚠️  Using CPU fallback (CUDA not available or not configured)" << endl;
        }
    }
    
    cout << "Model loaded successfully!\n";

    cout << "Press ESC to quit\n";
    Mat frame;
    auto start_time = chrono::high_resolution_clock::now();
    int frame_count = 0;
    
    while (cap.read(frame)) {
        // Run inference with GPU acceleration (if available)
        Mat mask = run_gpu_inference(session, frame, cuda_available);
        
        // Optimized blending
        Mat output = fast_blend(frame, mask);
        
        imshow("Background Removed (GPU)", output);
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
