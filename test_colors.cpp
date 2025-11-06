#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    cout << "=== Color Accuracy Test ===" << endl;
    
    // Create a simple test image with known colors
    Mat test_img(100, 100, CV_8UC3, Scalar(255, 0, 0)); // Pure blue
    Mat test_red(100, 100, CV_8UC3, Scalar(0, 0, 255)); // Pure red  
    Mat test_green(100, 100, CV_8UC3, Scalar(0, 255, 0)); // Pure green
    
    cout << "🧪 Testing color processing pipeline..." << endl;
    
    // Test the same normalization/blending logic as the main app
    Mat blurred;
    GaussianBlur(test_img, blurred, Size(25, 25), 0);
    
    // Create a simple circular mask (center = 1, edges = 0)
    Mat mask(100, 100, CV_8UC1, Scalar(0));
    circle(mask, Point(50, 50), 40, Scalar(255), -1);
    
    // Apply the corrected blending logic
    Mat mask_f, mask_blur;
    mask.convertTo(mask_f, CV_32F);
    normalize(mask_f, mask_f, 0.0, 1.0, NORM_MINMAX);
    GaussianBlur(mask_f, mask_blur, Size(15, 15), 0);
    
    Mat mask3;
    cvtColor(mask_blur, mask3, COLOR_GRAY2BGR);
    
    Mat frame_f, blurred_f;
    test_img.convertTo(frame_f, CV_32F, 1.0 / 255.0);
    blurred.convertTo(blurred_f, CV_32F, 1.0 / 255.0);
    
    Mat output_f = frame_f.mul(mask3) + blurred_f.mul(1.0 - mask3);
    Mat output;
    output_f.convertTo(output, CV_8UC3, 255.0);
    
    // Check that colors are preserved (blue should stay blue, not become blue-tinted)
    Vec3b pixel = output.at<Vec3b>(50, 50); // Center pixel should be mostly blue
    cout << "🎨 Center pixel (B,G,R): " << (int)pixel[0] << "," << (int)pixel[1] << "," << (int)pixel[2] << endl;
    
    if (pixel[0] > 200 && pixel[1] < 50 && pixel[2] < 50) {
        cout << "✅ Colors preserved correctly!" << endl;
    } else {
        cout << "❌ Color distortion detected!" << endl;
    }
    
    return 0;
}