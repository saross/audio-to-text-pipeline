#!/usr/bin/env python3
"""
GPU Test Script for Transcription Pipeline
==========================================

Tests GPU access, model loading, and memory usage for the transcription pipeline.
"""

import torch
import psutil
import os
import sys
import time
from pathlib import Path

def test_cuda_setup():
    """Test basic CUDA setup."""
    print("=" * 60)
    print("CUDA SETUP TEST")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / (1024**3):.1f}GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
        
        return True
    else:
        print("❌ CUDA not available")
        return False

def test_gpu_memory():
    """Test GPU memory allocation."""
    print("\n" + "=" * 60)
    print("GPU MEMORY TEST")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping memory test")
        return False
    
    device = torch.device("cuda:0")
    
    # Get initial memory
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated(0) / (1024**3)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"Total GPU memory: {total_memory:.1f}GB")
    print(f"Initial memory used: {initial_memory:.1f}GB")
    
    # Test memory allocation
    test_sizes = [0.5, 1.0, 2.0, 3.0]  # GB
    
    for size_gb in test_sizes:
        try:
            # Allocate tensor
            elements = int(size_gb * 1024**3 / 4)  # 4 bytes per float32
            tensor = torch.randn(elements, device=device, dtype=torch.float32)
            
            current_memory = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"✅ Allocated {size_gb}GB tensor, total used: {current_memory:.1f}GB")
            
            # Free tensor
            del tensor
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ Failed to allocate {size_gb}GB: Out of memory")
                break
            else:
                print(f"❌ Failed to allocate {size_gb}GB: {e}")
                break
    
    return True

def test_faster_whisper():
    """Test faster-whisper model loading."""
    print("\n" + "=" * 60)
    print("FASTER-WHISPER TEST")
    print("=" * 60)
    
    try:
        from faster_whisper import WhisperModel
        
        print("Loading distil-large-v3 model...")
        start_time = time.time()
        
        model = WhisperModel(
            "distil-large-v3",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "float32"
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f} seconds")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"GPU memory used: {memory_used:.1f}GB")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except ImportError:
        print("❌ faster-whisper not installed")
        return False
    except Exception as e:
        print(f"❌ Error loading faster-whisper: {e}")
        return False

def test_pyannote():
    """Test pyannote.audio model loading."""
    print("\n" + "=" * 60)
    print("PYANNOTE.AUDIO TEST")
    print("=" * 60)
    
    try:
        from pyannote.audio import Pipeline
        
        # Check for Hugging Face token
        hf_token = os.environ.get('HUGGINGFACE_TOKEN')
        if not hf_token:
            print("❌ HUGGINGFACE_TOKEN not set")
            print("Set with: export HUGGINGFACE_TOKEN=your_token_here")
            return False
        
        print("Loading speaker-diarization-3.1 model...")
        start_time = time.time()
        
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f} seconds")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"GPU memory used: {memory_used:.1f}GB")
        
        # Clean up
        del pipeline
        torch.cuda.empty_cache()
        
        return True
        
    except ImportError:
        print("❌ pyannote.audio not installed")
        return False
    except Exception as e:
        print(f"❌ Error loading pyannote.audio: {e}")
        print("This might be due to missing HUGGINGFACE_TOKEN or model access permissions")
        return False

def test_system_resources():
    """Test system resource availability."""
    print("\n" + "=" * 60)
    print("SYSTEM RESOURCES TEST")
    print("=" * 60)
    
    # CPU info
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU cores: {cpu_count}")
    print(f"CPU usage: {cpu_percent}%")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"RAM total: {memory.total / (1024**3):.1f}GB")
    print(f"RAM available: {memory.available / (1024**3):.1f}GB")
    print(f"RAM usage: {memory.percent}%")
    
    # Disk space
    disk = psutil.disk_usage('/')
    print(f"Disk total: {disk.total / (1024**3):.1f}GB")
    print(f"Disk free: {disk.free / (1024**3):.1f}GB")
    print(f"Disk usage: {(disk.used / disk.total) * 100:.1f}%")
    
    return True

def main():
    """Run all tests."""
    print("GPU-Accelerated Transcription Pipeline - System Test")
    print("=" * 60)
    
    results = {}
    
    # Run tests
    results['cuda'] = test_cuda_setup()
    results['memory'] = test_gpu_memory() if results['cuda'] else False
    results['faster_whisper'] = test_faster_whisper()
    results['pyannote'] = test_pyannote()
    results['system'] = test_system_resources()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test.upper()}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! Pipeline ready for use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())