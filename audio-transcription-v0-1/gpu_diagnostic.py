#!/usr/bin/env python3
"""
GPU Diagnostic Script for WhisperX Transcription Pipeline
Run this to diagnose GPU detection issues.
"""

import sys
import os

print("=" * 60)
print("GPU DIAGNOSTIC SCRIPT")
print("=" * 60)

# Check Python version
print(f"Python version: {sys.version}")
print()

# Check CUDA environment variables
print("CUDA Environment Variables:")
cuda_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH']
for var in cuda_vars:
    value = os.environ.get(var, 'Not set')
    print(f"  {var}: {value}")
print()

# Check if nvidia-smi works
print("NVIDIA System Management Interface:")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        # Print just the GPU info lines
        for line in lines:
            if 'NVIDIA' in line or 'GeForce' or 'RTX' in line or 'GPU' in line:
                print(f"  {line}")
            if '|' in line and ('MiB' in line or 'W' in line):
                print(f"  {line}")
        print("  ✅ nvidia-smi working correctly")
    else:
        print(f"  ❌ nvidia-smi failed: {result.stderr}")
except FileNotFoundError:
    print("  ❌ nvidia-smi not found - NVIDIA drivers may not be installed")
except Exception as e:
    print(f"  ❌ Error running nvidia-smi: {e}")
print()

# Check PyTorch installation and CUDA support
print("PyTorch Installation:")
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version (PyTorch): {torch.version.cuda}")
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB VRAM)")
        
        print(f"  Current device: {torch.cuda.current_device()}")
        print("  ✅ PyTorch CUDA support working")
    else:
        print("  ❌ PyTorch reports CUDA not available")
        
        # Check if torch was installed with CUDA support
        try:
            # Try to import a CUDA-specific module
            from torch.backends import cuda
            print("  - PyTorch has CUDA backend compiled in")
        except:
            print("  - PyTorch may be CPU-only version")
            
except ImportError:
    print("  ❌ PyTorch not installed")
print()

# Check CUDA runtime version if available
print("CUDA Runtime Version:")
try:
    import subprocess
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if 'release' in line.lower():
                print(f"  {line.strip()}")
        print("  ✅ CUDA toolkit installed")
    else:
        print("  ❌ nvcc not found - CUDA toolkit may not be installed")
except FileNotFoundError:
    print("  ❌ nvcc not found - CUDA toolkit may not be installed")
except Exception as e:
    print(f"  ❌ Error checking CUDA version: {e}")
print()

# Test a simple CUDA operation
print("CUDA Functionality Test:")
try:
    import torch
    if torch.cuda.is_available():
        # Try to create a tensor on GPU
        x = torch.randn(2, 2).cuda()
        y = x + x
        print(f"  ✅ Successfully created and operated on GPU tensor")
        print(f"  ✅ GPU memory available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
        
        # Clear the test tensor
        del x, y
        torch.cuda.empty_cache()
    else:
        print("  ❌ Cannot test CUDA - not available")
except Exception as e:
    print(f"  ❌ CUDA operation failed: {e}")
print()

# Check WhisperX if installed
print("WhisperX Installation:")
try:
    import whisperx
    print("  ✅ WhisperX imported successfully")
    
    # Try to check if WhisperX can see CUDA
    try:
        # This might not work in all versions, but worth trying
        print(f"  WhisperX can access torch.cuda: {torch.cuda.is_available()}")
    except:
        pass
        
except ImportError:
    print("  ❌ WhisperX not installed")
    print("  Install with: pip install git+https://github.com/m-bain/whisperx.git")
print()

print("=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)

# Provide recommendations based on what we found
import torch
if not torch.cuda.is_available():
    print("❌ ISSUE: PyTorch cannot access CUDA")
    print()
    print("POSSIBLE SOLUTIONS:")
    print("1. Check NVIDIA drivers:")
    print("   sudo nvidia-smi")
    print()
    print("2. Reinstall PyTorch with CUDA support:")
    print("   pip uninstall torch torchaudio")
    print("   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("3. Check CUDA compatibility:")
    print("   Your GPU needs to support CUDA compute capability 3.5+")
    print()
    print("4. Check virtual environment:")
    print("   Make sure you're in the correct environment")
    print("   source ~/transcription-env/bin/activate")
    
else:
    print("✅ GPU detection should work!")
    print("If the transcription script still fails, the issue might be:")
    print("1. WhisperX-specific GPU detection")
    print("2. Memory allocation issues")
    print("3. CUDA version compatibility with WhisperX")

print("=" * 60)
