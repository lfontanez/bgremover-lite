#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bgremover");

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
    string source = argc > 1 ? argv[1] : "0";
    VideoCapture cap;
    if (source == "0") {
        cout << "Attempting to open webcam (device 0)...\n";
        cap.open(0);
    } else {
        cout << "Opening video file: " << source << "...\n";
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
    
    // Performance advisory for high resolutions
    if (width * height >= 1920 * 1080) {
        cout << "🔍 High resolution detected (" << width << "x" << height << ") - consider GPU version for optimal performance\n";
    } else if (width * height >= 1280 * 720) {
        cout << "📺 HD resolution detected (" << width << "x" << height << ")\n";
    }
    
    cout << "Video properties - FPS: " << fps << ", Resolution: " 
         << width << "x" << height << "\n";

    cout << "Loading U²-Net model...\n";
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    Ort::Session session(env, "models/u2net.onnx", opts);
    cout << "Model loaded successfully!\n";

    cout << "Press ESC to quit\n";
    Mat frame;
    bool first_frame = true;
    
    // Pre-allocate buffers for efficient 1080p processing
    Mat mask, output, blurred;
    
    while (cap.read(frame)) {
        // Ensure consistent frame size for 1080p processing
        if (frame.cols != width || frame.rows != height) {
            resize(frame, frame, Size(width, height));
            if (first_frame) {
                cout << "📐 Resized input to: " << width << "x" << height << " for 1080p processing" << endl;
            }
        }
        
        // Run inference on downsampled frame (U²-Net processes at 320x320)
        Mat downsampled;
        resize(frame, downsampled, Size(320, 320));
        Mat mask_320 = run_inference(session, downsampled);
        
        // Resize mask back to full resolution
        resize(mask_320, mask, frame.size());
        
        // Pre-allocate blurred frame for 1080p efficiency
        if (first_frame || blurred.empty() || blurred.size() != frame.size()) {
            blurred.create(frame.size(), frame.type());
        }
        
        // Apply Gaussian blur to frame
        GaussianBlur(frame, blurred, Size(15, 15), 0);
        
        // Ensure output frame is properly allocated
        if (first_frame || output.empty() || output.size() != frame.size()) {
            output.create(frame.size(), frame.type());
        }
        
        // Optimized pixel-level blend for 1080p
        Mat mask_clean = (mask > 0.5);
        frame.copyTo(output, mask_clean);
        output.setTo(blurred, ~mask_clean);

        string window_title = "1080p Background Removed (CPU)";
        imshow(window_title, output);
        if (waitKey(1) == 27) break;  // ESC
        
        first_frame = false;
    }

    // Cleanup
    cap.release();
    destroyAllWindows();
    
    // Release pre-allocated matrices
    frame.release();
    mask.release();
    output.release();
    blurred.release();
    
    cout << "🧹 1080p CPU processing cleanup completed" << endl;
    
    return 0;
}
