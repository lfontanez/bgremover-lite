#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    cout << "=== ONNX Model Test ===" << endl;
    
    // Test model loading
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bgremover-test");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    
    try {
        cout << "Loading U²-Net model..." << endl;
        Ort::Session session(env, "models/u2net.onnx", opts);
        cout << "✅ Model loaded successfully!" << endl;
        
        // Get model input info
        auto input_info = session.GetInputTypeInfo(0);
        auto input_shape = input_info.GetTensorTypeAndShapeInfo().GetShape();
        cout << "📏 Input shape: ";
        for (auto dim : input_shape) {
            cout << dim << " ";
        }
        cout << endl;
        
        // Get model output info
        auto output_info = session.GetOutputTypeInfo(0);
        auto output_shape = output_info.GetTensorTypeAndShapeInfo().GetShape();
        cout << "📏 Output shape: ";
        for (auto dim : output_shape) {
            cout << dim << " ";
        }
        cout << endl;
        
        cout << "🎯 ONNX Runtime is working correctly!" << endl;
        return 0;
        
    } catch (const exception& e) {
        cerr << "❌ Error loading model: " << e.what() << endl;
        return 1;
    }
}