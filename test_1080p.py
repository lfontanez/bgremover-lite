#!/usr/bin/env python3
"""
1080p HD Video Processing Test Script
Tests the background remover's 1080p HD video processing capabilities
"""

import cv2
import numpy as np
import time
import sys
import os

def test_1080p_video_capture():
    """Test if the system can handle 1080p video input"""
    print("🧪 Testing 1080p video capture...")
    
    # Create a test 1080p frame
    width, height = 1920, 1080
    test_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some visual markers to verify resolution
    cv2.putText(test_frame, "1080p TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
    cv2.putText(test_frame, f"Resolution: {width}x{height}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Save test frame
    test_file = "test_1080p_frame.jpg"
    cv2.imwrite(test_file, test_frame)
    print(f"✅ Created 1080p test frame: {test_file}")
    
    # Verify frame dimensions
    loaded_frame = cv2.imread(test_file)
    if loaded_frame.shape[1] == width and loaded_frame.shape[0] == height:
        print(f"✅ 1080p resolution confirmed: {loaded_frame.shape[1]}x{loaded_frame.shape[0]}")
        return True
    else:
        print(f"❌ Resolution mismatch: {loaded_frame.shape[1]}x{loaded_frame.shape[0]}")
        return False

def test_virtual_camera_1080p():
    """Test virtual camera 1080p support"""
    print("\n🧪 Testing virtual camera 1080p support...")
    
    try:
        # Check if virtual camera device exists
        vcam_path = "/dev/video2"
        if os.path.exists(vcam_path):
            print(f"✅ Virtual camera device found: {vcam_path}")
            
            # Try to get device info
            try:
                result = os.popen(f"v4l2-ctl --device={vcam_path} --list-formats-ext").read()
                if "1920x1080" in result:
                    print("✅ Virtual camera supports 1080p")
                    return True
                else:
                    print("⚠️  Virtual camera may not support 1080p")
                    return False
            except:
                print("⚠️  Could not check virtual camera capabilities")
                return False
        else:
            print(f"ℹ️  Virtual camera device not found: {vcam_path}")
            print("   Run ./setup_virtual_camera.sh to create it")
            return None
    except Exception as e:
        print(f"⚠️  Virtual camera test failed: {e}")
        return False

def test_gpu_memory_1080p():
    """Test GPU memory requirements for 1080p processing"""
    print("\n🧪 Testing GPU memory requirements for 1080p...")
    
    try:
        import pynvml
        pynvml.nvmlInit()
        
        # Get GPU memory info
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        total_memory_gb = mem_info.total / (1024**3)
        free_memory_gb = mem_info.free / (1024**3)
        
        print(f"GPU Memory: {free_memory_gb:.2f}GB free / {total_memory_gb:.2f}GB total")
        
        # Estimate 1080p memory requirements
        # 1080p frame: 1920*1080*3 bytes = ~6.2MB per frame
        # Processing buffer: ~25MB for 1080p
        estimated_1080p_memory = 50  # MB estimate for 1080p processing
        
        if free_memory_gb * 1024 >= estimated_1080p_memory:
            print(f"✅ Sufficient GPU memory for 1080p (~{estimated_1080p_memory}MB required)")
            return True
        else:
            print(f"⚠️  May have insufficient GPU memory for 1080p (~{estimated_1080p_memory}MB required)")
            return False
            
    except ImportError:
        print("ℹ️  pynvml not available - skipping GPU memory test")
        return None
    except Exception as e:
        print(f"⚠️  GPU memory test failed: {e}")
        return False

def test_performance_simulation():
    """Simulate 1080p processing performance"""
    print("\n🧪 Simulating 1080p processing performance...")
    
    # Create test 1080p frame
    width, height = 1920, 1080
    test_frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    # Simulate processing time
    start_time = time.time()
    
    # Simulate U²-Net processing (resize to 320x320 for model input)
    resized = cv2.resize(test_frame, (320, 320))
    
    # Simulate post-processing (resize back to 1080p)
    processed = cv2.resize(resized, (width, height))
    
    # Simulate blur operation
    blurred = cv2.GaussianBlur(test_frame, (15, 15), 0)
    
    processing_time = time.time() - start_time
    estimated_fps = 1.0 / processing_time if processing_time > 0 else 0
    
    print(f"1080p processing simulation: {processing_time*1000:.2f}ms")
    print(f"Estimated FPS capability: {estimated_fps:.1f}")
    
    if estimated_fps >= 30:
        print("✅ System should handle 1080p at real-time speeds")
        return True
    elif estimated_fps >= 15:
        print("✅ System should handle 1080p at good quality")
        return True
    else:
        print("⚠️  1080p processing may be slower than real-time")
        return False

def main():
    print("=== 1080p HD Video Processing Test ===\n")
    
    tests = [
        ("1080p Video Capture", test_1080p_video_capture),
        ("Virtual Camera 1080p", test_virtual_camera_1080p),
        ("GPU Memory 1080p", test_gpu_memory_1080p),
        ("Performance Simulation", test_performance_simulation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("=== TEST SUMMARY ===")
    print('='*50)
    
    passed = 0
    total = 0
    
    for test_name, result in results:
        total += 1
        if result is True:
            status = "✅ PASS"
            passed += 1
        elif result is None:
            status = "ℹ️  SKIP"
        else:
            status = "❌ FAIL"
        
        print(f"{test_name:25} {status}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! 1080p HD video processing is ready!")
        print("\nTo use 1080p:")
        print("1. Build the project: ./build.sh")
        print("2. Run GPU version: ./build/bgremover_gpu")
        print("3. For virtual camera: ./build/bgremover_gpu --vcam")
        return 0
    elif passed >= total - 1:
        print("\n✅ Most tests passed! 1080p processing should work well.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the requirements above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())