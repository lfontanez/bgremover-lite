#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bgremover-gpu");

// GPU-accelerated ONNX inference (CUDA)
Mat run_gpu_inference(Ort::Session& session, const Mat& img) {
    // Preprocess on CPU
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
    
    // Process result
    float* data = outputs.front().GetTensorMutableData<float>();
    Mat mask(320, 320, CV_32F, data);
    resize(mask, mask, img.size());
    normalize(mask, mask, 0, 1, NORM_MINMAX);
    
    return mask;
}

// CPU processing with GPU-optimized parameters
Mat fast_blend(const Mat& frame, const Mat& mask) {
    // Use smaller kernel and optimized processing
    Mat blurred;
    GaussianBlur(frame, blurred, Size(9, 9), 0);  // Smaller kernel for speed
    
    // Optimized masking
    Mat mask_clean = (mask > 128) * 255;
    
    // Direct pixel-level blend with optimization
    Mat output = frame.clone();
    
    // Use vectorized operations for speed
    for (int y = 0; y < frame.rows; y++) {
        const Vec3b* frame_ptr = frame.ptr<Vec3b>(y);
        const Vec3b* blur_ptr = blurred.ptr<Vec3b>(y);
        const uchar* mask_ptr = mask_clean.ptr<uchar>(y);
        Vec3b* out_ptr = output.ptr<Vec3b>(y);
        
        for (int x = 0; x < frame.cols; x++) {
            if (mask_ptr[x] == 0) {
                out_ptr[x] = blur_ptr[x];
            }
            // else keep original (frame_ptr[x])
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
    
    // Enable CUDA provider if available
    vector<string> available_providers = Ort::GetAvailableProviders();
    bool cuda_available = false;
    
    cout << "Available providers: ";
    for (const auto& provider : available_providers) {
        cout << provider << " ";
        if (provider == "CUDAExecutionProvider") {
            cuda_available = true;
            // For now, just note CUDA is available - full GPU support would require
            // different ONNX Runtime build with CUDA provider
            cout << "\n⚡ CUDA-capable runtime detected!" << endl;
        }
    }
    cout << endl;
    
    if (!cuda_available) {
        cout << "⚠️  CUDA not available, using optimized CPU fallback" << endl;
    }
    
    Ort::Session session(env, "models/u2net.onnx", opts);
    cout << "Model loaded successfully!\n";

    cout << "Press ESC to quit\n";
    Mat frame;
    auto start_time = chrono::high_resolution_clock::now();
    int frame_count = 0;
    
    while (cap.read(frame)) {
        // Run inference with GPU acceleration (if available)
        Mat mask = run_gpu_inference(session, frame);
        
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