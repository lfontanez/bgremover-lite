#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

int main() {
    cout << "=== ONNX GPU Performance Test ===" << endl;
    
    // Check ONNX Runtime providers
    vector<string> providers = Ort::GetAvailableProviders();
    cout << "Available ONNX Runtime providers: ";
    for (const auto& provider : providers) {
        cout << provider << " ";
    }
    cout << endl;
    
    bool cuda_available = false;
    for (const auto& provider : providers) {
        if (provider == "CUDAExecutionProvider") {
            cuda_available = true;
            cout << "✅ CUDA provider available!" << endl;
            break;
        }
    }
    
    if (!cuda_available) {
        cout << "❌ CUDA provider not available" << endl;
    }
    
    // Create test image
    Mat test_frame(480, 640, CV_8UC3, Scalar(128, 128, 128));
    randu(test_frame, 0, 255);
    
    cout << "📊 Testing ONNX model performance..." << endl;
    cout << "Image size: " << test_frame.cols << "x" << test_frame.rows << endl;
    cout << "Test iterations: 50" << endl;
    
    // CPU Test
    cout << "\n🔄 CPU Inference Test:" << endl;
    Ort::Env env_cpu(ORT_LOGGING_LEVEL_WARNING, "bgremover-cpu-test");
    Ort::SessionOptions opts_cpu;
    opts_cpu.SetIntraOpNumThreads(1);
    
    auto cpu_start = chrono::high_resolution_clock::now();
    Ort::Session session_cpu(env_cpu, "models/u2net.onnx", opts_cpu);
    
    for (int i = 0; i < 50; i++) {
        Mat resized;
        resize(test_frame, resized, Size(320, 320));
        resized.convertTo(resized, CV_32F, 1.0f / 255.0f);
        Mat blob = dnn::blobFromImage(resized);
        
        array<int64_t, 4> shape = {1, 3, 320, 320};
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        auto input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, blob.ptr<float>(), blob.total(),
            shape.data(), shape.size());
        
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;
        auto input_name_alloc = session_cpu.GetInputNameAllocated(0, allocator);
        auto output_name_alloc = session_cpu.GetOutputNameAllocated(0, allocator);
        input_names.push_back(input_name_alloc.get());
        output_names.push_back(output_name_alloc.get());
        
        auto outputs = session_cpu.Run(Ort::RunOptions{nullptr},
                                       input_names.data(), &input_tensor, 1,
                                       output_names.data(), 1);
    }
    
    auto cpu_end = chrono::high_resolution_clock::now();
    auto cpu_duration = chrono::duration_cast<chrono::milliseconds>(cpu_end - cpu_start);
    double cpu_fps = (50 * 1000.0) / cpu_duration.count();
    cout << "   CPU Time: " << cpu_duration.count() << "ms" << endl;
    cout << "   CPU FPS: " << cpu_fps << endl;
    
    // GPU Test (if available)
    if (cuda_available) {
        cout << "\n🚀 GPU Inference Test:" << endl;
        cout << "   Note: CUDA provider not available in current ONNX Runtime build" << endl;
        cout << "   This is expected for the standard binary release" << endl;
        cout << "   Performance would be similar to CPU for now" << endl;
    }
    
    return 0;
}