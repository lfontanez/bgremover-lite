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
    while (cap.read(frame)) {
        Mat mask = run_inference(session, frame);

        // Simple, reliable, fast blending - direct pixel operations
        Mat blurred;
        GaussianBlur(frame, blurred, Size(15, 15), 0);
        
        // Clean, simple mask (0 or 255)
        mask = (mask > 128) * 255;  // Simple threshold
        
        // Direct pixel-level blend - most reliable for colors
        Mat output = frame.clone();
        for (int y = 0; y < frame.rows; y++) {
            for (int x = 0; x < frame.cols; x++) {
                uchar m = mask.at<uchar>(y, x);  // 0 or 255
                if (m == 0) {
                    // Outside mask - use blurred
                    output.at<Vec3b>(y, x) = blurred.at<Vec3b>(y, x);
                } else {
                    // Inside mask - use original
                    output.at<Vec3b>(y, x) = frame.at<Vec3b>(y, x);
                }
            }
        }

        imshow("Background Removed", output);
        if (waitKey(1) == 27) break;  // ESC
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
