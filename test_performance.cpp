#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

int main() {
    cout << "=== Performance & Color Test ===" << endl;
    
    // Create test images
    Mat test_img(100, 100, CV_8UC3, Scalar(128, 128, 128)); // Gray (neutral)
    Mat test_blue(100, 100, CV_8UC3, Scalar(255, 0, 0));   // Pure blue
    Mat test_red(100, 100, CV_8UC3, Scalar(0, 0, 255));    // Pure red
    
    // Create test mask
    Mat mask(100, 100, CV_8UC1, Scalar(0));
    circle(mask, Point(50, 50), 30, Scalar(255), -1);
    
    cout << "🧪 Testing optimized algorithm performance and color accuracy..." << endl;
    
    // Test each color
    vector<Mat> test_images = {test_blue, test_red, test_img};
    vector<string> color_names = {"Blue", "Red", "Gray"};
    
    auto start = chrono::high_resolution_clock::now();
    
    for (int i = 0; i < test_images.size(); i++) {
        Mat frame = test_images[i].clone();
        
        // Apply the same algorithm as main.cpp
        auto frame_start = chrono::high_resolution_clock::now();
        
        Mat blurred;
        GaussianBlur(frame, blurred, Size(15, 15), 0);
        
        // Clean, simple mask (0 or 255)
        Mat mask_clean = (mask > 128) * 255;
        
        // Direct pixel-level blend - most reliable for colors
        Mat output = frame.clone();
        for (int y = 0; y < frame.rows; y++) {
            for (int x = 0; x < frame.cols; x++) {
                uchar m = mask_clean.at<uchar>(y, x);
                if (m == 0) {
                    output.at<Vec3b>(y, x) = blurred.at<Vec3b>(y, x);
                } else {
                    output.at<Vec3b>(y, x) = frame.at<Vec3b>(y, x);
                }
            }
        }
        
        auto frame_end = chrono::high_resolution_clock::now();
        auto frame_duration = chrono::duration_cast<chrono::milliseconds>(frame_end - frame_start);
        
        // Check center pixel (should match input color)
        Vec3b center_pixel = output.at<Vec3b>(50, 50);
        Vec3b input_pixel = frame.at<Vec3b>(50, 50);
        
        cout << "🎨 " << color_names[i] << " test:" << endl;
        cout << "   Input (B,G,R): " << (int)input_pixel[0] << ", " 
             << (int)input_pixel[1] << ", " << (int)input_pixel[2] << endl;
        cout << "   Output (B,G,R): " << (int)center_pixel[0] << ", " 
             << (int)center_pixel[1] << ", " << (int)center_pixel[2] << endl;
        cout << "   Time: " << frame_duration.count() << "ms" << endl;
        
        // Color accuracy check
        bool color_preserved = true;
        for (int c = 0; c < 3; c++) {
            if (abs(center_pixel[c] - input_pixel[c]) > 10) {
                color_preserved = false;
                break;
            }
        }
        cout << "   " << (color_preserved ? "✅" : "❌") << " Color preserved" << endl;
        cout << endl;
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto total_duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "🏁 Total processing time: " << total_duration.count() << "ms" << endl;
    cout << "🚀 Average per frame: " << total_duration.count() / 3.0 << "ms" << endl;
    cout << "📊 FPS potential: " << (1000.0 / (total_duration.count() / 3.0)) << " FPS" << endl;
    
    return 0;
}