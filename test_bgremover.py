#!/usr/bin/env python3
"""
Test script for GPU background blur
"""

import cv2
import numpy as np
from bgremover_gpu import BackgroundBlurGPU
import os

def create_test_image():
    """Create a simple test image with a person-like shape"""
    # Create a blue background
    img = np.full((480, 640, 3), (255, 100, 50), dtype=np.uint8)  # Blue background in BGR
    
    # Draw a simple "person" in the center
    cv2.circle(img, (320, 200), 50, (50, 200, 50), -1)  # Head (green in BGR)
    cv2.rectangle(img, (280, 250), (360, 400), (50, 200, 50), -1)  # Body
    cv2.circle(img, (300, 180), 8, (50, 100, 200), -1)  # Eye
    cv2.circle(img, (340, 180), 8, (50, 100, 200), -1)  # Eye
    
    return img

def test_background_blur():
    """Test the background blur functionality"""
    print("🧪 Testing GPU Background Blur")
    
    # Check if model exists
    model_path = "models/u2netp.onnx"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Available models:")
        if os.path.exists("models"):
            for model in os.listdir("models"):
                if model.endswith(".onnx"):
                    print(f"  - {model}")
        return False
    
    try:
        # Create test image
        print("📸 Creating test image...")
        test_img = create_test_image()
        cv2.imwrite("test_input.jpg", test_img)
        print("✅ Test image saved as test_input.jpg")
        
        # Initialize processor
        print("🚀 Initializing background blur processor...")
        processor = BackgroundBlurGPU(model_path=model_path)
        
        # Process test image
        print("⚡ Processing test image...")
        result, mask = processor.process_frame(test_img, blur_strength=51)
        
        # Save results
        cv2.imwrite("test_output.jpg", result)
        cv2.imwrite("test_mask.jpg", mask)
        
        print("✅ Test complete!")
        print(f"   Input: test_input.jpg")
        print(f"   Output: test_output.jpg") 
        print(f"   Mask: test_mask.jpg")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_webcam():
    """Test with webcam (if available)"""
    print("🎥 Testing with webcam...")
    
    # Check if camera is available
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not available")
        return False
    
    cap.release()
    
    try:
        processor = BackgroundBlurGPU()
        processor.process_video(source="0", blur_strength=51, show_mask=True)
        return True
    except Exception as e:
        print(f"❌ Webcam test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== GPU Background Blur Test ===\n")
    
    # Test with generated image
    if test_background_blur():
        print("\n✅ Static image test passed!")
    else:
        print("\n❌ Static image test failed!")
    
    # Ask if user wants to test webcam
    try:
        response = input("\n🎥 Test with webcam? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            test_webcam()
    except KeyboardInterrupt:
        print("\n👋 Test interrupted")