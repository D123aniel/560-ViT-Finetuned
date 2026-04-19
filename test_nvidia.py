import torch
import sys

# This file checks to see if your system has a compatible NVIDIA GPU and that pytorch can use it

def verify_cuda():
    print("--- Environment Check ---")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # check if CUDA available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    # print info
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    capability = torch.cuda.get_device_capability(current_device)

    print(f"Number of GPUs detected: {device_count}")
    print(f"Current Device ID: {current_device}")
    print(f"Device Name: {device_name}")
    print(f"Compute Capability: {capability[0]}.{capability[1]}")

    # test tenor to GPU
    print("\n--- GPU Test ---")
    try:
        # create a tensor on the CPU, move to gpu, then test operation
        cpu_tensor = torch.randn(3, 3)
        gpu_tensor = cpu_tensor.to("cuda")
        result = gpu_tensor * gpu_tensor
        
        print("Successfully moved tensor to GPU and performed a calculation")
        print(f"Result Device: {result.device}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")

# Check for BF16 support (ideal for 50-series)
print(f"BF16 Support: {torch.cuda.is_bf16_supported()}")

if __name__ == "__main__":
    verify_cuda()

import sys
print(sys.executable)
print(torch.__version__)