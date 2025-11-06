#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    cout << "=== Color Debug Test ===" << endl;
    
    // Create test image with pure colors
    Mat test_blue(100, 100, CV_8UC3, Scalar(255, 0, 0));   // BGR: (255,0,0) = blue
    Mat test_red(100, 100, CV_8UC3, Scalar(0, 0, 255));    // BGR: (0,0,255) = red
    Mat test_green(100, 100, CV_8UC3, Scalar(0, 255, 0));  // BGR: (0,255,0) = green
    
    // Create circular mask
    Mat mask(100, 100, CV_8UC1, Scalar(0));
    circle(mask, Point(50, 50), 30, Scalar(255), -1);
    
    cout << "🔍 Testing color preservation with new algorithm..." << endl;
    
    // Apply the corrected algorithm
    Mat blurred;
    GaussianBlur(test_blue, blurred, Size(25, 25), 0);
    
    Mat mask_f, mask_blur;
    mask.convertTo(mask_f, CV_32F);
    normalize(mask_f, mask_f, 0.0, 1.0, NORM_MINMAX);
    GaussianBlur(mask_f, mask_blur, Size(15, 15), 0);
    
    // Create 3-channel mask properly
    Mat mask3;
    vector<Mat> channels(3, mask_blur);
    merge(channels, mask3);
    
    // Convert to float 0-1 range
    Mat test_blue_f, blurred_f;
    test_blue.convertTo(test_blue_f, CV_32F, 1.0 / 255.0);
    blurred.convertTo(blurred_f, CV_32F, 1.0 / 255.0);
    
    // Blend
    Mat output_f = test_blue_f.mul(mask3) + blurred_f.mul(1.0 - mask3);
    
    // Range check
    output_f = max(output_f, 0.0);
    output_f = min(output_f, 1.0);
    
    // Convert back
    Mat output;
    output_f.convertTo(output, CV_8UC3, 255.0);
    
    // Check center pixel (should be blue)
    Vec3b center_pixel = output.at<Vec3b>(50, 50);
    cout << "🎨 Center pixel (B,G,R): " << (int)center_pixel[0] << ", " 
         << (int)center_pixel[1] << ", " << (int)center_pixel[2] << endl;
    
    // Check edge pixel (should be blurred blue)
    Vec3b edge_pixel = output.at<Vec3b>(20, 20);
    cout << "🎨 Edge pixel (B,G,R): " << (int)edge_pixel[0] << ", " 
         << (int)edge_pixel[1] << ", " << (int)edge_pixel[2] << endl;
    
    // Verify blue dominance in center
    if (center_pixel[0] > center_pixel[1] + 50 && center_pixel[0] > center_pixel[2] + 50) {
        cout << "✅ Blue color preserved correctly!" << endl;
    } else {
        cout << "❌ Color distortion detected!" << endl;
        cout << "   Expected: Strong blue, weak green/red" << endl;
    }
    
    return 0;
}