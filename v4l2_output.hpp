#ifndef V4L2_OUTPUT_HPP
#define V4L2_OUTPUT_HPP

#include <opencv2/opencv.hpp>
#include <linux/videodev2.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <string>
#include <memory>

/**
 * V4L2Output - A class for writing frames to a V4L2 virtual camera device
 * 
 * This class provides a convenient interface for outputting video frames to
 * v4l2loopback virtual camera devices. It handles device initialization,
 * format configuration (YUYV), and efficient BGR-to-YUYV conversion.
 * 
 * Key features:
 * - RAII pattern for automatic resource cleanup
 * - YUYV format (most compatible with Zoom, Teams, etc.)
 * - Efficient color space conversion
 * - Comprehensive error handling
 * - Device status checking
 */
class V4L2Output {
public:
    /**
     * Constructor
     * 
     * @param device_path Path to the V4L2 device (default: "/dev/video2")
     * @param width Frame width in pixels (default: 1920 for 1080p)
     * @param height Frame height in pixels (default: 1080 for 1080p)
     */
    V4L2Output(const std::string& device_path = "/dev/video2", 
               int width = 1920, int height = 1080)
        : device_path_(device_path), fd_(-1), width_(width), height_(height), 
          is_open_(false) {
    }
    
    /**
     * Destructor - automatically closes device if open
     */
    ~V4L2Output() {
        closeDevice();
    }
    
    // Delete copy and assignment operators (RAII pattern)
    V4L2Output(const V4L2Output&) = delete;
    V4L2Output& operator=(const V4L2Output&) = delete;
    
    /**
     * Open and configure the V4L2 device
     * 
     * @return true if successful, false otherwise
     */
    bool open() {
        if (is_open_) {
            std::cerr << "V4L2Output: Device already open\n";
            return true;
        }
        
        // Open device in non-blocking mode (we'll use blocking I/O for video)
        fd_ = ::open(device_path_.c_str(), O_WRONLY);
        if (fd_ < 0) {
            std::cerr << "V4L2Output: Failed to open device " << device_path_ 
                      << ": " << strerror(errno) << "\n";
            return false;
        }
        
        // Configure device format
        if (!setFormat()) {
            std::cerr << "V4L2Output: Failed to set device format\n";
            closeDevice();
            return false;
        }
        
        is_open_ = true;
        std::cout << "V4L2Output: Successfully opened device " << device_path_ 
                  << " (" << width_ << "x" << height_ << ", YUYV)\n";
        return true;
    }
    
    /**
     * Close the V4L2 device
     */
    void close() {
        closeDevice();
    }
    
    /**
     * Check if device is open
     * 
     * @return true if device is open, false otherwise
     */
    bool isOpen() const {
        return is_open_;
    }
    
    /**
     * Write a frame to the V4L2 device
     * 
     * @param frame Input frame in BGR format (OpenCV Mat)
     * @return true if successful, false otherwise
     */
    bool writeFrame(const cv::Mat& frame) {
        if (!is_open_) {
            std::cerr << "V4L2Output: Device not open\n";
            return false;
        }
        
        if (frame.empty()) {
            std::cerr << "V4L2Output: Empty frame provided\n";
            return false;
        }
        
        if (frame.cols != width_ || frame.rows != height_) {
            std::cerr << "V4L2Output: Frame size mismatch. Expected " 
                      << width_ << "x" << height_ << ", got " 
                      << frame.cols << "x" << frame.rows << "\n";
            return false;
        }
        
        if (frame.type() != CV_8UC3) {
            std::cerr << "V4L2Output: Frame must be 3-channel 8-bit (BGR)\n";
            return false;
        }
        
        // Convert BGR to YUYV
        cv::Mat yuyv_frame;
        if (!convertBGRtoYUYV(frame, yuyv_frame)) {
            std::cerr << "V4L2Output: Failed to convert BGR to YUYV\n";
            return false;
        }
        
        // Write to device
        ssize_t written = write(fd_, yuyv_frame.data, yuyv_frame.total() * yuyv_frame.elemSize());
        if (written < 0) {
            std::cerr << "V4L2Output: Failed to write frame: " << strerror(errno) << "\n";
            return false;
        }
        
        if (static_cast<size_t>(written) != yuyv_frame.total() * yuyv_frame.elemSize()) {
            std::cerr << "V4L2Output: Incomplete frame write. Expected " 
                      << yuyv_frame.total() * yuyv_frame.elemSize() 
                      << " bytes, wrote " << written << " bytes\n";
            return false;
        }
        
        return true;
    }
    
    /**
     * Get device path
     * 
     * @return Path to the device
     */
    std::string getDevicePath() const {
        return device_path_;
    }
    
    /**
     * Get frame dimensions
     * 
     * @return cv::Size object with width and height
     */
    cv::Size getSize() const {
        return cv::Size(width_, height_);
    }

private:
    /**
     * Configure V4L2 device format to YUYV
     * 
     * @return true if successful, false otherwise
     */
    bool setFormat() {
        struct v4l2_format fmt;
        std::memset(&fmt, 0, sizeof(fmt));
        fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
        fmt.fmt.pix.width = width_;
        fmt.fmt.pix.height = height_;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        fmt.fmt.pix.field = V4L2_FIELD_NONE;
        fmt.fmt.pix.colorspace = V4L2_COLORSPACE_DEFAULT;
        
        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
            std::cerr << "V4L2Output: VIDIOC_S_FMT failed: " << strerror(errno) << "\n";
            return false;
        }
        
        // Verify the format was set correctly
        if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
            std::cerr << "V4L2Output: Device does not support YUYV format\n";
            return false;
        }
        
        // Check if dimensions were accepted
        if (fmt.fmt.pix.width != width_ || fmt.fmt.pix.height != height_) {
            std::cerr << "V4L2Output: Device rejected dimensions. Requested " 
                      << width_ << "x" << height_ << ", got " 
                      << fmt.fmt.pix.width << "x" << fmt.fmt.pix.height << "\n";
            return false;
        }
        
        return true;
    }
    
    /**
     * Convert BGR image to YUYV format
     * 
     * YUYV format: 2 bytes per pixel, packed as Y0 U0 Y1 V0
     * For 2 pixels: [Y0][U0][Y1][V0] (4 bytes total)
     * 
     * @param bgr_frame Input BGR frame
     * @param yuyv_frame Output YUYV frame
     * @return true if successful, false otherwise
     */
    bool convertBGRtoYUYV(const cv::Mat& bgr_frame, cv::Mat& yuyv_frame) {
        int width = bgr_frame.cols;
        int height = bgr_frame.rows;
        
        // YUYV is 2 bytes per pixel, stored as single channel with width*2 columns
        yuyv_frame.create(height, width * 2, CV_8UC1);
        
        // Conversion coefficients (BT.601 standard)
        const float wr = 0.299f, wg = 0.587f, wb = 0.114f;
        const float u_max = 0.436f, v_max = 0.615f;
        
        for (int y = 0; y < height; ++y) {
            const cv::Vec3b* bgr_row = bgr_frame.ptr<cv::Vec3b>(y);
            uchar* yuyv_row = yuyv_frame.ptr<uchar>(y);
            
            for (int x = 0; x < width; x += 2) {
                // Process first pixel
                const cv::Vec3b& bgr0 = bgr_row[x];
                float B0 = bgr0[0] / 255.0f;
                float G0 = bgr0[1] / 255.0f;
                float R0 = bgr0[2] / 255.0f;
                
                // Calculate Y0
                float Y0 = wr * R0 + wg * G0 + wb * B0;
                Y0 = std::clamp(Y0, 0.0f, 1.0f);
                uchar y0_value = static_cast<uchar>(Y0 * 255.0f);
                
                // Process second pixel if available
                uchar y1_value = 0;
                if (x + 1 < width) {
                    const cv::Vec3b& bgr1 = bgr_row[x + 1];
                    float B1 = bgr1[0] / 255.0f;
                    float G1 = bgr1[1] / 255.0f;
                    float R1 = bgr1[2] / 255.0f;
                    
                    // Calculate Y1
                    float Y1 = wr * R1 + wg * G1 + wb * B1;
                    Y1 = std::clamp(Y1, 0.0f, 1.0f);
                    y1_value = static_cast<uchar>(Y1 * 255.0f);
                }
                
                // Calculate U and V from the first pixel
                float U = (B0 - Y0) * u_max;
                float V = (R0 - Y0) * v_max;
                
                U = std::clamp(U, -0.436f, 0.436f);
                V = std::clamp(V, -0.615f, 0.615f);
                
                uchar u_value = static_cast<uchar>((U + 0.436f) * 255.0f / (2 * 0.436f));
                uchar v_value = static_cast<uchar>((V + 0.615f) * 255.0f / (2 * 0.615f));
                
                // Store packed YUYV: Y0 U0 Y1 V0
                yuyv_row[x * 2] = y0_value;     // Y0
                yuyv_row[x * 2 + 1] = u_value;  // U0
                yuyv_row[x * 2 + 2] = y1_value; // Y1
                yuyv_row[x * 2 + 3] = v_value;  // V0
            }
        }
        
        return true;
    }
    
    /**
     * Close the device and cleanup resources
     */
    void closeDevice() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
            is_open_ = false;
        }
    }
    
    std::string device_path_;  // Path to V4L2 device
    int fd_;                   // File descriptor for device
    int width_;                // Frame width
    int height_;               // Frame height
    bool is_open_;             // Device open status
};

#endif // V4L2_OUTPUT_HPP
