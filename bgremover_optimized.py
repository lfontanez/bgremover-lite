#!/usr/bin/env python3
"""
Optimized GPU-Accelerated Background Blur with Advanced Features
- Background blur only (person stays sharp)
- Multiple model support
- Adjustable blur strength
- Performance optimization
- Error handling and fallbacks
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import argparse
import os
import sys
from typing import Tuple, Optional, List
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedBackgroundBlur:
    def __init__(self, model_path: str = "models/u2netp.onnx", target_size: int = 320, 
                 use_gpu: bool = True, cache_size: int = 5):
        """
        Initialize the optimized background blur processor
        
        Args:
            model_path: Path to the ONNX segmentation model
            target_size: Input size for the model
            use_gpu: Whether to attempt GPU acceleration
            cache_size: Number of frames to cache for processing
        """
        self.target_size = target_size
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.cache_size = cache_size
        
        # Performance tracking
        self.frame_times = []
        self.total_frames = 0
        
        # Initialize ONNX Runtime
        self.session = self._init_onnx_session()
        
        # Preprocessing parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        
        # Frame cache for batch processing
        self.frame_cache = []
        
        logger.info(f"🚀 Optimized background blur initialized with {model_path}")
        logger.info(f"ONNX Providers: {self.session.get_providers()}")
        
    def _init_onnx_session(self) -> ort.InferenceSession:
        """Initialize ONNX Runtime session with best available provider"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            if self.use_gpu:
                # Try GPU providers first
                providers = [
                    'TensorrtExecutionProvider',
                    'CUDAExecutionProvider', 
                    'CPUExecutionProvider'
                ]
            else:
                # CPU only
                providers = ['CPUExecutionProvider']
            
            session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Log the actual provider being used
            active_provider = session.get_providers()[0]
            if 'CUDA' in active_provider or 'Tensorrt' in active_provider:
                logger.info(f"✅ Using GPU acceleration: {active_provider}")
            else:
                logger.info(f"⚠️  Using CPU execution: {active_provider}")
                
            return session
            
        except Exception as e:
            logger.error(f"Failed to initialize ONNX session: {e}")
            raise
    
    def _get_optimal_model(self) -> str:
        """Select the best available model based on speed/accuracy trade-off"""
        models_dir = Path("models")
        if not models_dir.exists():
            return self.model_path
        
        available_models = list(models_dir.glob("*.onnx"))
        if not available_models:
            return self.model_path
        
        # Prefer u2netp for speed, then u2net, then modnet
        model_priority = ["u2netp.onnx", "u2net.onnx", "modnet.onnx"]
        
        for preferred in model_priority:
            for model in available_models:
                if model.name == preferred:
                    logger.info(f"Selected model: {model.name} (speed optimized)")
                    return str(model)
        
        # Return first available model
        selected = available_models[0]
        logger.info(f"Using available model: {selected.name}")
        return str(selected)
    
    def preprocess_frame_fast(self, frame: np.ndarray) -> np.ndarray:
        """Optimized preprocessing with minimal allocations"""
        # Resize in one step
        resized = cv2.resize(frame, (self.target_size, self.target_size), 
                           interpolation=cv2.INTER_LINEAR)
        
        # BGR to RGB and normalize in one operation
        normalized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Apply normalization
        normalized = (normalized - self.mean) / self.std
        
        # Reorder to NCHW
        blob = normalized.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)
        
        return blob
    
    def run_inference_optimized(self, blob: np.ndarray) -> np.ndarray:
        """Optimized inference with timing"""
        input_name = self.session.get_inputs()[0].name
        
        start_time = time.perf_counter()
        outputs = self.session.run(None, {input_name: blob})
        inference_time = time.perf_counter() - start_time
        
        # Track performance
        self.frame_times.append(inference_time)
        if len(self.frame_times) > 30:  # Keep last 30 frame times
            self.frame_times.pop(0)
        
        mask = outputs[0][0, 0]  # Remove batch and channel dims
        return mask
    
    def create_advanced_mask(self, mask: np.ndarray, original_shape: Tuple[int, int], 
                           threshold: float = 0.5, smooth_kernel: int = 7) -> np.ndarray:
        """Create refined mask with configurable threshold and smoothing"""
        # Resize to original
        mask_resized = cv2.resize(mask, (original_shape[1], original_shape[0]))
        
        # Normalize
        mask_min, mask_max = mask_resized.min(), mask_resized.max()
        if mask_max > mask_min:
            mask_normalized = (mask_resized - mask_min) / (mask_max - mask_min)
        else:
            mask_normalized = np.zeros_like(mask_resized)
        
        # Multi-stage smoothing
        if smooth_kernel > 0:
            # Apply multiple blur passes for smoother edges
            mask_smooth = cv2.GaussianBlur(mask_normalized, (smooth_kernel, smooth_kernel), 0)
            mask_smooth = cv2.GaussianBlur(mask_smooth, (3, 3), 0)
        else:
            mask_smooth = mask_normalized
        
        # Adaptive thresholding
        _, mask_binary = cv2.threshold(mask_smooth, threshold, 1.0, cv2.THRESH_BINARY)
        
        # Convert to uint8
        return (mask_binary * 255).astype(np.uint8)
    
    def create_advanced_blur(self, frame: np.ndarray, blur_strength: int, 
                           blur_type: str = "gaussian") -> np.ndarray:
        """Create blurred background with multiple blur types"""
        # Ensure odd kernel size
        if blur_strength % 2 == 0:
            blur_strength += 1
        
        blur_strength = max(3, blur_strength)  # Minimum kernel size
        
        if blur_type == "gaussian":
            return cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
        elif blur_type == "median":
            return cv2.medianBlur(frame, blur_strength)
        elif blur_type == "bilateral":
            # Edge-preserving blur - good for keeping sharp edges
            return cv2.bilateralFilter(frame, blur_strength//3, blur_strength, blur_strength)
        else:
            return cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
    
    def smart_blend(self, original: np.ndarray, blurred: np.ndarray, 
                   mask: np.ndarray, blend_type: str = "standard") -> np.ndarray:
        """Advanced blending with multiple algorithms"""
        mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
        
        if blend_type == "standard":
            # Standard alpha blend
            result = original * mask_3ch + blurred * (1 - mask_3ch)
            
        elif blend_type == "soft":
            # Soft edge blending
            mask_soft = cv2.GaussianBlur(mask_3ch, (3, 3), 0)
            result = original * mask_soft + blurred * (1 - mask_soft)
            
        elif blend_type == "seamless":
            # Poisson blending for seamless transitions
            try:
                # Create mask for seamless cloning
                mask_u8 = mask.astype(np.uint8)
                result = cv2.seamlessClone(original, blurred, mask_u8, 
                                         (original.shape[1]//2, original.shape[0]//2), 
                                         cv2.NORMAL_CLONE)
            except:
                # Fallback to standard blend
                result = original * mask_3ch + blurred * (1 - mask_3ch)
        else:
            result = original * mask_3ch + blurred * (1 - mask_3ch)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def process_frame_optimized(self, frame: np.ndarray, blur_strength: int = 51,
                              threshold: float = 0.5, smooth_kernel: int = 7,
                              blur_type: str = "gaussian", blend_type: str = "standard") -> Tuple[np.ndarray, np.ndarray]:
        """Process frame with all optimizations"""
        start_time = time.perf_counter()
        original_shape = frame.shape[:2]
        
        # Preprocess
        blob = self.preprocess_frame_fast(frame)
        
        # Inference
        mask_raw = self.run_inference_optimized(blob)
        
        # Postprocess
        mask = self.create_advanced_mask(mask_raw, original_shape, threshold, smooth_kernel)
        
        # Blur background
        blurred_bg = self.create_advanced_blur(frame, blur_strength, blur_type)
        
        # Blend
        result = self.smart_blend(frame, blurred_bg, mask, blend_type)
        
        total_time = time.perf_counter() - start_time
        self.total_frames += 1
        
        return result, mask, total_time
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        if not self.frame_times:
            return {"avg_inference_time": 0, "fps": 0, "total_frames": 0}
        
        avg_inference = np.mean(self.frame_times) * 1000  # Convert to ms
        avg_fps = 1 / np.mean(self.frame_times) if np.mean(self.frame_times) > 0 else 0
        
        return {
            "avg_inference_time": avg_inference,
            "fps": avg_fps,
            "total_frames": self.total_frames,
            "current_provider": self.session.get_providers()[0]
        }
    
    def process_video_advanced(self, source: str = "0", blur_strength: int = 51,
                              threshold: float = 0.5, smooth_kernel: int = 7,
                              blur_type: str = "gaussian", blend_type: str = "standard",
                              show_mask: bool = False, show_stats: bool = True,
                              save_path: Optional[str] = None):
        """Advanced video processing with all features"""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        
        logger.info(f"📹 Video: {width}x{height} @ {fps} FPS")
        
        # Initialize video writer
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            logger.info(f"💾 Saving to: {save_path}")
        
        # Processing loop
        frame_count = 0
        key_frame_interval = 10  # Update stats every 10 frames
        
        logger.info("🎬 Press 'q' to quit, 'm' to toggle mask, 's' to save frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            try:
                # Process frame
                result, mask, process_time = self.process_frame_optimized(
                    frame, blur_strength, threshold, smooth_kernel, blur_type, blend_type
                )
                
                # Display processing info
                if show_stats and frame_count % key_frame_interval == 0:
                    stats = self.get_performance_stats()
                    info_text = f"FPS: {stats['fps']:.1f} | Frame: {frame_count} | Time: {process_time*1000:.1f}ms | Provider: {stats['current_provider']}"
                    
                    # Add text with background
                    cv2.rectangle(result, (5, 5), (800, 35), (0, 0, 0), -1)
                    cv2.putText(result, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Show mask if requested
                display_frame = result
                if show_mask:
                    mask_display = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                    mask_display = cv2.resize(mask_display, (width//4, height//4))
                    cv2.putText(mask_display, "MASK", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Place mask in corner
                    y_offset = 10
                    x_offset = width - mask_display.shape[1] - 10
                    if x_offset > 0 and y_offset > 0:
                        display_frame[y_offset:y_offset+mask_display.shape[0], 
                                     x_offset:x_offset+mask_display.shape[1]] = mask_display
                
                # Show result
                window_name = "Advanced Background Blur"
                cv2.imshow(window_name, display_frame)
                
                # Save frame if writer is active
                if writer:
                    writer.write(result)
                
                # Save individual frame on 's' key
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    cv2.imwrite(f"frame_{frame_count:04d}.jpg", result)
                    logger.info(f"Frame saved: frame_{frame_count:04d}.jpg")
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                # Show original frame with error message
                cv2.putText(frame, f"Error: {str(e)[:50]}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow(window_name, frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                show_mask = not show_mask
                logger.info(f"Mask display: {'ON' if show_mask else 'OFF'}")
            elif key == ord('h'):
                show_stats = not show_stats
                logger.info(f"Stats display: {'ON' if show_stats else 'OFF'}")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Final performance report
        stats = self.get_performance_stats()
        logger.info(f"✅ Processing complete!")
        logger.info(f"   Total frames: {stats['total_frames']}")
        logger.info(f"   Average FPS: {stats['fps']:.1f}")
        logger.info(f"   Average inference time: {stats['avg_inference_time']:.1f}ms")
        logger.info(f"   Provider: {stats['current_provider']}")

def main():
    parser = argparse.ArgumentParser(
        description="Optimized GPU-Accelerated Background Blur",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with webcam
  python bgremover_optimized.py
  
  # Use specific video file
  python bgremover_optimized.py --source video.mp4
  
  # Strong blur with Gaussian smoothing
  python bgremover_optimized.py --blur 75 --smooth 9
  
  # Use bilateral filter for edge preservation
  python bgremover_optimized.py --blur-type bilateral --blend-type soft
  
  # Show real-time statistics
  python bgremover_optimized.py --show-stats
  
  # Save output video
  python bgremover_optimized.py --save output.mp4
        """
    )
    
    parser.add_argument("--source", type=str, default="0",
                       help="Video source (0 for webcam, or video file path)")
    parser.add_argument("--model", type=str,
                       help="Path to ONNX model (auto-selects best available)")
    parser.add_argument("--blur", type=int, default=51,
                       help="Background blur strength (odd number, 3-99)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Segmentation threshold (0.1-0.9)")
    parser.add_argument("--smooth", type=int, default=7,
                       help="Mask smoothing kernel size (0-21, odd numbers)")
    parser.add_argument("--blur-type", choices=["gaussian", "median", "bilateral"], 
                       default="gaussian", help="Background blur algorithm")
    parser.add_argument("--blend-type", choices=["standard", "soft", "seamless"],
                       default="standard", help="Blending algorithm")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU acceleration")
    parser.add_argument("--show-mask", action="store_true",
                       help="Show segmentation mask overlay")
    parser.add_argument("--show-stats", action="store_true",
                       help="Show performance statistics")
    parser.add_argument("--save", type=str,
                       help="Save output video to specified path")
    
    args = parser.parse_args()
    
    try:
        # Select model
        model_path = args.model
        if not model_path:
            # Auto-select best model
            processor = OptimizedBackgroundBlur(cache_size=1)  # Minimal init for model selection
            model_path = processor._get_optimal_model()
        
        # Initialize processor
        processor = OptimizedBackgroundBlur(
            model_path=model_path,
            use_gpu=not args.no_gpu
        )
        
        # Process video
        processor.process_video_advanced(
            source=args.source,
            blur_strength=args.blur,
            threshold=args.threshold,
            smooth_kernel=args.smooth,
            blur_type=args.blur_type,
            blend_type=args.blend_type,
            show_mask=args.show_mask,
            show_stats=args.show_stats,
            save_path=args.save
        )
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if "--verbose" in sys.argv:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())