#!/usr/bin/env python3
"""
Verify CUDA support in OpenCV, PyTorch, and ONNX Runtime
"""

import sys
import importlib

def check_opencv_cuda():
    """Check if OpenCV has CUDA support."""
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")

        if not hasattr(cv2, "cuda"):
            print("❌ OpenCV built without CUDA support (no cv2.cuda module)")
            return False

        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"CUDA enabled devices: {cuda_count}")

        if cuda_count > 0:
            print("✅ OpenCV CUDA support is available!")
            for i in range(cuda_count):
                try:
                    if hasattr(cv2.cuda, 'getDeviceProperties'):
                        props = cv2.cuda.getDeviceProperties(i)
                        print(f"  Device {i}: {props.name}")
                        print(f"    Compute capability: {props.major}.{props.minor}")
                        print(f"    Total memory: {props.totalGlobalMem / 1024 ** 3:.2f} GB")
                    else:
                        print(f"  Device {i}: CUDA device detected (getDeviceProperties not available)")
                except Exception as e:
                    print(f"  ⚠️  Could not query device {i}: {e}")
            return True
        else:
            print("❌ No CUDA-capable devices detected by OpenCV.")
            return False

    except ImportError:
        print("❌ OpenCV not installed.")
        return False
    except Exception as e:
        print(f"⚠️  OpenCV CUDA check failed: {e}")
        return False


def check_torch_cuda():
    """Check PyTorch CUDA support."""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")

        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        if cuda_available:
            count = torch.cuda.device_count()
            print(f"CUDA devices: {count}")

            for i in range(count):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

            print("✅ PyTorch CUDA is working!")
        else:
            print("❌ PyTorch CUDA not available.")
        return cuda_available

    except ImportError:
        print("❌ PyTorch not installed.")
        return False
    except Exception as e:
        print(f"⚠️  PyTorch CUDA check failed: {e}")
        return False


def check_onnx_cuda():
    """Check ONNX Runtime CUDA support."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"ONNX Runtime providers: {providers}")

        has_cuda = "CUDAExecutionProvider" in providers
        if has_cuda:
            print("✅ ONNX Runtime CUDA support available!")
        else:
            print("❌ ONNX Runtime CUDA support not available.")
        return has_cuda

    except ImportError:
        print("❌ ONNX Runtime not installed.")
        return False
    except Exception as e:
        print(f"⚠️  ONNX Runtime check failed: {e}")
        return False


def main():
    print("=== CUDA Environment Verification ===\n")

    opencv_cuda = check_opencv_cuda()
    print()

    torch_cuda = check_torch_cuda()
    print()

    onnx_cuda = check_onnx_cuda()
    print()

    print("=== Summary ===")
    print(f"OpenCV CUDA:   {'✅' if opencv_cuda else '❌'}")
    print(f"PyTorch CUDA:  {'✅' if torch_cuda else '❌'}")
    print(f"ONNX Runtime:  {'✅' if onnx_cuda else '❌'}")

    if opencv_cuda and (torch_cuda or onnx_cuda):
        print("\n🎉 GPU acceleration is ready to go!")
        return 0
    else:
        print("\n⚠️  One or more CUDA components are missing or misconfigured.")
        return 1


if __name__ == "__main__":
    sys.exit(main())