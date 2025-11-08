#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <dlfcn.h>
#include <fstream>

using namespace cv;
using namespace std;

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bgremover-gpu");

// GPU-accelerated ONNX inference (CUDA)
Mat run_gpu_inference(Ort::Session& session, const Mat& img, bool cuda_available) {
    // Preprocess on CPU
    Mat resized;
    resize(img, resized, Size(320, 320));
    resized.convertTo(resized, CV_32F, 1.0f / 255.0f);
    Mat blob = dnn::blobFromImage(resized);
    
    array<int64_t, 4> shape = {1, 3, 320, 320};
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Use CPU memory for now (GPU memory would require different preprocessing)
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    if (cuda_available) {
        cout << "🚀 Using GPU execution with CUDA provider" << endl;
    }
    
    auto input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, blob.ptr<float>(), blob.total(),
        shape.data(), shape.size());
    
    // Run inference on GPU (if available)
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
    auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
    input_names.push_back(input_name_alloc.get());
    output_names.push_back(output_name_alloc.get());
    
    auto outputs = session.Run(Ort::RunOptions{nullptr},
                               input_names.data(), &input_tensor, 1,
                               output_names.data(), 1);
    
    // Process result with proper mask isolation
    float* data = outputs.front().GetTensorMutableData<float>();
    Mat mask(320, 320, CV_32F, data);
    resize(mask, mask, img.size());

    // ✅ FIXED: Proper mask isolation for GPU version
    normalize(mask, mask, 0.0, 1.0, NORM_MINMAX);
    
    // Sharpen the separation between person and background
    threshold(mask, mask, 0.4, 1.0, THRESH_BINARY);
    
    // Apply soft edges to prevent harsh borders
    GaussianBlur(mask, mask, Size(7, 7), 0);
    
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
    
    // Check for CUDA provider library
    string cuda_lib_path = "./onnxruntime/lib/libonnxruntime_providers_shared.so";
    bool cuda_provider_available = false;
    
    // Check if CUDA provider library exists
    std::ifstream cuda_lib_file(cuda_lib_path);
    if (cuda_lib_file.good()) {
        cout << "✅ Found CUDA provider library at: " << cuda_lib_path << endl;
        cuda_provider_available = true;
        cuda_lib_file.close();
        
        // Try to load the CUDA provider library
        void* cuda_lib = dlopen("libonnxruntime_providers_shared.so", RTLD_NOW | RTLD_GLOBAL);
        if (cuda_lib) {
            cout << "✅ CUDA provider library loaded successfully" << endl;
            dlclose(cuda_lib);
        } else {
            cout << "❌ Failed to load CUDA provider library: " << dlerror() << endl;
            cuda_provider_available = false;
        }
    } else {
        cout << "❌ CUDA provider library not found at: " << cuda_lib_path << endl;
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
        
        // If CUDA is available and provider library exists, explicitly enable it
        if (cuda_available && cuda_provider_available) {
            try {
                // Try to use CUDA provider with proper configuration
                // First, try the V2 API if available
                try {
                    #ifdef ORT_API_VERSION
                    #if ORT_API_VERSION >= 14
                    // Use V2 API for newer ONNX Runtime versions
                    Ort::CUDAProviderOptionsV2 cuda_options_v2;
                    session_options.AppendExecutionProvider_CUDA_V2(cuda_options_v2);
                    cout << "✅ CUDA execution provider V2 explicitly enabled" << endl;
                    #else
                    // Use legacy API for older versions
                    Ort::CUDAProviderOptions cuda_options;
                    session_options.AppendExecutionProvider_CUDA(cuda_options);
                    cout << "✅ CUDA execution provider explicitly enabled" << endl;
                    #endif
                    #else
                    // Default to legacy API
                    Ort::CUDAProviderOptions cuda_options;
                    session_options.AppendExecutionProvider_CUDA(cuda_options);
                    cout << "✅ CUDA execution provider explicitly enabled" << endl;
                    #endif
                    
                    // Configure session options for optimal performance
                    session_options.SetIntraOpNumThreads(1);
                    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
                    
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
            cout << "⚠️  CUDA available but provider library missing - may use fallback" << endl;
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
        
        // Performance monitoring
        frame_count++;
        if (frame_count % 30 == 0) {
            auto current_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(current_time - start_time);
            double fps_real = (frame_count * 1000.0) / duration.count();
            cout << (cuda_available ? "🚀 GPU" : "⚡ Optimized CPU") << " Performance: " 
                 << fps_real << " FPS" << endl;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
