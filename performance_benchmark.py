#!/usr/bin/env python3
"""
Performance comparison between basic and optimized background blur
"""

import cv2
import numpy as np
import time
import os
from bgremover_gpu import BackgroundBlurGPU
from bgremover_optimized import OptimizedBackgroundBlur

def create_test_scene():
    """Create a more complex test scene"""
    # Create a realistic scene with multiple elements
    scene = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Background - gradient blue
    for y in range(480):
        scene[y, :] = [50 + y//4, 100, 200 - y//8]  # BGR gradient
    
    # Add some "people" shapes
    # Person 1 - center left
    cv2.circle(scene, (200, 200), 40, (100, 200, 100), -1)  # Head
    cv2.rectangle(scene, (160, 240), (240, 380), (100, 200, 100), -1)  # Body
    cv2.circle(scene, (190, 190), 5, (50, 50, 200), -1)  # Eye
    cv2.circle(scene, (210, 190), 5, (50, 50, 200), -1)  # Eye
    
    # Person 2 - center right
    cv2.circle(scene, (400, 180), 35, (150, 180, 120), -1)  # Head
    cv2.rectangle(scene, (370, 215), (430, 350), (150, 180, 120), -1)  # Body
    cv2.circle(scene, (395, 175), 4, (50, 50, 200), -1)  # Eye
    cv2.circle(scene, (405, 175), 4, (50, 50, 200), -1)  # Eye
    
    # Add some background objects
    cv2.rectangle(scene, (50, 50), (150, 100), (200, 100, 50), -1)  # Box
    cv2.circle(scene, (500, 400), 30, (80, 120, 200), -1)  # Ball
    
    return scene

def benchmark_implementations():
    """Benchmark both implementations"""
    print("🏁 Performance Benchmark")
    print("=" * 50)
    
    # Create test scene
    print("📸 Creating test scene...")
    test_scene = create_test_scene()
    cv2.imwrite("benchmark_scene.jpg", test_scene)
    
    results = {}
    
    # Test Basic Implementation
    print("\n⚡ Testing Basic Implementation...")
    try:
        basic_processor = BackgroundBlurGPU("models/u2netp.onnx")
        
        times = []
        for i in range(10):  # Process 10 frames
            start_time = time.time()
            result, mask = basic_processor.process_frame(test_scene, blur_strength=51)
            end_time = time.time()
            times.append(end_time - start_time)
            
            if i == 0:  # Save first result
                cv2.imwrite("benchmark_basic.jpg", result)
        
        results['basic'] = {
            'avg_time': np.mean(times) * 1000,  # Convert to ms
            'min_time': np.min(times) * 1000,
            'max_time': np.max(times) * 1000,
            'fps': 1 / np.mean(times)
        }
        
        print(f"   Average time: {results['basic']['avg_time']:.1f}ms")
        print(f"   FPS: {results['basic']['fps']:.1f}")
        
    except Exception as e:
        print(f"   ❌ Basic implementation failed: {e}")
        results['basic'] = None
    
    # Test Optimized Implementation
    print("\n🚀 Testing Optimized Implementation...")
    try:
        optimized_processor = OptimizedBackgroundBlur("models/u2netp.onnx", cache_size=1)
        
        times = []
        for i in range(10):  # Process 10 frames
            start_time = time.time()
            result, mask, process_time = optimized_processor.process_frame_optimized(
                test_scene, blur_strength=51, threshold=0.5, smooth_kernel=7
            )
            end_time = time.time()
            times.append(end_time - start_time)
            
            if i == 0:  # Save first result
                cv2.imwrite("benchmark_optimized.jpg", result)
        
        results['optimized'] = {
            'avg_time': np.mean(times) * 1000,  # Convert to ms
            'min_time': np.min(times) * 1000,
            'max_time': np.max(times) * 1000,
            'fps': 1 / np.mean(times)
        }
        
        print(f"   Average time: {results['optimized']['avg_time']:.1f}ms")
        print(f"   FPS: {results['optimized']['fps']:.1f}")
        
    except Exception as e:
        print(f"   ❌ Optimized implementation failed: {e}")
        results['optimized'] = None
    
    # Compare results
    print("\n📊 Performance Comparison")
    print("=" * 50)
    
    if results['basic'] and results['optimized']:
        speedup = results['basic']['avg_time'] / results['optimized']['avg_time']
        fps_improvement = results['optimized']['fps'] / results['basic']['fps']
        
        print(f"Basic Implementation:")
        print(f"   Average: {results['basic']['avg_time']:.1f}ms")
        print(f"   Min: {results['basic']['min_time']:.1f}ms")
        print(f"   Max: {results['basic']['max_time']:.1f}ms")
        print(f"   FPS: {results['basic']['fps']:.1f}")
        
        print(f"\nOptimized Implementation:")
        print(f"   Average: {results['optimized']['avg_time']:.1f}ms")
        print(f"   Min: {results['optimized']['min_time']:.1f}ms")
        print(f"   Max: {results['optimized']['max_time']:.1f}ms")
        print(f"   FPS: {results['optimized']['fps']:.1f}")
        
        print(f"\n🎯 Improvements:")
        print(f"   Speedup: {speedup:.2f}x faster")
        print(f"   FPS improvement: {fps_improvement:.2f}x")
        print(f"   Time saved: {results['basic']['avg_time'] - results['optimized']['avg_time']:.1f}ms per frame")
        
        if speedup > 1.1:
            print(f"   ✅ Optimized version is {speedup:.1f}x faster!")
        else:
            print(f"   ⚠️  Similar performance, optimizations may need GPU hardware")
    
    # Feature comparison
    print(f"\n🔧 Feature Comparison")
    print("=" * 50)
    print("Basic Implementation:")
    print("   ✅ GPU acceleration (ONNX Runtime CUDA)")
    print("   ✅ Background blur")
    print("   ✅ Adjustable blur strength")
    print("   ❌ Limited error handling")
    print("   ❌ No performance stats")
    
    print(f"\nOptimized Implementation:")
    print("   ✅ GPU acceleration (ONNX Runtime CUDA/TensorRT)")
    print("   ✅ Advanced background blur (Gaussian, Median, Bilateral)")
    print("   ✅ Multiple blending algorithms (Standard, Soft, Seamless)")
    print("   ✅ Configurable threshold and smoothing")
    print("   ✅ Real-time performance statistics")
    print("   ✅ Enhanced error handling and logging")
    print("   ✅ Automatic model selection")
    print("   ✅ Video saving capability")
    print("   ✅ Interactive controls (mask toggle, stats display)")

if __name__ == "__main__":
    benchmark_implementations()