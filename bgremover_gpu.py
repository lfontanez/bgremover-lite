#!/usr/bin/env python3
"""
GPU-Accelerated Background Blur using ONNX Runtime CUDA
Blurs only the background while keeping the person/foreground sharp
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import argparse
from typing import Tuple, Optional
import os

class BackgroundBlurGPU:
    def __init__(self, model_path: str = "models/u2netp.onnx", target_size: int = 320):
        """
        Initialize the background blur processor
        
        Args:
            model_path: Path to the ONNX segmentation model
            target_size: Input size for the model (320 for u2net/u2netp)
        """
        self.target_size = target_size
        self.model_path = model_path
        
        # Initialize ONNX Runtime with CUDA provider
        self.session = self._init_onnx_session()
        
        # Preprocessing parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        
        print(f"🚀 Background blur initialized with {model_path}")
        print(f"ONNX Providers: {self.session.get_providers()}")
        
    def _init_onnx_session(self) -> ort.InferenceSession:
        """Initialize ONNX Runtime session with CUDA provider"""
        try:
            # Try CUDA first, then CPU fallback
            providers = [
                'TensorrtExecutionProvider',  # Fastest if available
                'CUDAExecutionProvider',      # Good CUDA support
                'CPUExecutionProvider'        # Fallback
            ]
            
            session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Verify we're using GPU if available
            if 'CUDA' in str(session.get_providers()):
                print("✅ Using CUDA acceleration")
            else:
                print("⚠️  Using CPU (CUDA not available)")
                
            return session
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ONNX session: {e}")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for model input
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Preprocessed blob ready for inference
        """
        # Resize to model input size
        resized = cv2.resize(frame, (self.target_size, self.target_size))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        normalized = (normalized - self.mean) / self.std
        
        # Convert to NCHW format
        blob = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        blob = np.expand_dims(blob, axis=0)
        
        # Ensure float32 type
        blob = blob.astype(np.float32)
        
        return blob
    
    def run_inference(self, blob: np.ndarray) -> np.ndarray:
        """
        Run segmentation inference
        
        Args:
            blob: Preprocessed input blob
            
        Returns:
            Segmentation mask
        """
        # Get input name
        input_name = self.session.get_inputs()[0].name
        
        # Run inference
        start_time = time.time()
        outputs = self.session.run(None, {input_name: blob})
        inference_time = time.time() - start_time
        
        # Get the mask (usually the first output)
        mask = outputs[0][0, 0]  # Remove batch and channel dimensions
        
        print(f"⚡ Inference time: {inference_time*1000:.1f}ms")
        
        return mask
    
    def postprocess_mask(self, mask: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess the segmentation mask
        
        Args:
            mask: Raw mask from model
            original_shape: Original frame shape (height, width)
            
        Returns:
            Processed mask (0-255, uint8)
        """
        # Resize mask to original frame size
        mask_resized = cv2.resize(mask, (original_shape[1], original_shape[0]))
        
        # Normalize to [0, 1]
        mask_normalized = (mask_resized - mask_resized.min()) / (mask_resized.max() - mask_resized.min() + 1e-8)
        
        # Apply Gaussian blur to smooth edges
        mask_smooth = cv2.GaussianBlur(mask_normalized, (7, 7), 0)
        
        # Create binary mask (person = 255, background = 0)
        _, mask_binary = cv2.threshold(mask_smooth, 0.5, 1.0, cv2.THRESH_BINARY)
        
        # Convert to uint8
        mask_uint8 = (mask_binary * 255).astype(np.uint8)
        
        return mask_uint8
    
    def create_blurred_background(self, frame: np.ndarray, blur_strength: int = 51) -> np.ndarray:
        """
        Create blurred version of the background
        
        Args:
            frame: Input frame
            blur_strength: Gaussian blur kernel size
            
        Returns:
            Blurred frame
        """
        # Ensure odd kernel size
        if blur_strength % 2 == 0:
            blur_strength += 1
            
        blurred = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
        return blurred
    
    def blend_frames(self, original: np.ndarray, blurred: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Blend original and blurred frames using mask
        
        Args:
            original: Original frame (person sharp)
            blurred: Blurred frame (background blurred)
            mask: Person mask (255 = person, 0 = background)
            
        Returns:
            Final blended frame
        """
        # Create 3-channel mask
        mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
        
        # Blend frames
        result = original * mask_3ch + blurred * (1 - mask_3ch)
        
        return result.astype(np.uint8)
    
    def process_frame(self, frame: np.ndarray, blur_strength: int = 51) -> np.ndarray:
        """
        Process a single frame for background blur
        
        Args:
            frame: Input frame
            blur_strength: Background blur strength
            
        Returns:
            Frame with blurred background
        """
        original_shape = frame.shape[:2]
        
        # Preprocess
        blob = self.preprocess_frame(frame)
        
        # Run inference
        mask = self.run_inference(blob)
        
        # Postprocess mask
        mask_processed = self.postprocess_mask(mask, original_shape)
        
        # Create blurred background
        blurred_bg = self.create_blurred_background(frame, blur_strength)
        
        # Blend frames
        result = self.blend_frames(frame, blurred_bg, mask_processed)
        
        return result, mask_processed
    
    def process_video(self, source: str = "0", blur_strength: int = 51, 
                     show_mask: bool = False, save_path: Optional[str] = None):
        """
        Process video stream with background blur
        
        Args:
            source: Video source (0 for webcam, or video file path)
            blur_strength: Background blur strength
            show_mask: Whether to show the segmentation mask
            save_path: Optional path to save output video
        """
        # Initialize video capture
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video: {width}x{height} @ {fps} FPS")
        
        # Initialize video writer if save path specified
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            print(f"💾 Saving to: {save_path}")
        
        # Processing loop
        frame_count = 0
        total_time = 0
        
        print("🎬 Press 'q' to quit, 'm' to toggle mask, 's' to save frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Process frame
            result, mask = self.process_frame(frame, blur_strength)
            
            # Add processing time overlay
            process_time = time.time() - start_time
            total_time += process_time
            frame_count += 1
            
            # Display processing info
            avg_fps = frame_count / total_time if total_time > 0 else 0
            info_text = f"FPS: {avg_fps:.1f} | Frame: {frame_count} | Blur: {blur_strength}"
            cv2.putText(result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show mask if requested
            display_frame = result
            if show_mask:
                mask_display = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                mask_display = cv2.resize(mask_display, (width//4, height//4))
                cv2.putText(mask_display, "MASK", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1, cv2.LINE_AA)
                # Place mask in top-right corner
                result[10:10+mask_display.shape[0], width-mask_display.shape[1]-10:width-10] = mask_display
                display_frame = result
            
            # Show result
            cv2.imshow("Background Blur (GPU)", display_frame)
            
            # Save frame if writer is active
            if writer:
                writer.write(result)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                show_mask = not show_mask
                print(f"Mask display: {'ON' if show_mask else 'OFF'}")
            elif key == ord('s'):
                cv2.imwrite(f"frame_{frame_count:04d}.jpg", result)
                print(f"Frame saved: frame_{frame_count:04d}.jpg")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Processing complete: {frame_count} frames, avg FPS: {avg_fps:.1f}")

def main():
    parser = argparse.ArgumentParser(description="GPU-Accelerated Background Blur")
    parser.add_argument("--source", type=str, default="0", 
                       help="Video source (0 for webcam, or video file path)")
    parser.add_argument("--model", type=str, default="models/u2netp.onnx",
                       help="Path to ONNX model")
    parser.add_argument("--blur", type=int, default=51,
                       help="Background blur strength (odd number)")
    parser.add_argument("--show-mask", action="store_true",
                       help="Show segmentation mask")
    parser.add_argument("--save", type=str,
                       help="Save output video to specified path")
    
    args = parser.parse_args()
    
    try:
        # Check if model exists
        if not os.path.exists(args.model):
            print(f"❌ Model not found: {args.model}")
            print("Available models in models/ directory:")
            if os.path.exists("models"):
                for model in os.listdir("models"):
                    if model.endswith(".onnx"):
                        print(f"  - {model}")
            return 1
        
        # Initialize and run
        processor = BackgroundBlurGPU(model_path=args.model)
        processor.process_video(
            source=args.source,
            blur_strength=args.blur,
            show_mask=args.show_mask,
            save_path=args.save
        )
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())