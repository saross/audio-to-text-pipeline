#!/usr/bin/env python3
import subprocess
import sys
import os
import argparse
import time
import shutil
import shlex
import tempfile

def find_whisper_binary():
    """Find the whisper executable"""
    possible_names = [
        "/app/whisper.cpp/build/bin/whisper-cli",  # Correct CMake build location
        "/app/whisper.cpp/build/examples/cli/whisper-cli",  # Alternative location
        "/app/whisper.cpp/build/bin/main",  # Deprecated but might work
        "/app/whisper.cpp/whisper-cli",
        "/app/whisper.cpp/main"
    ]
    
    for path in possible_names:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    
    # If not found, list what's actually there
    print("ERROR: Could not find whisper executable!")
    print("Contents of /app/whisper.cpp/build/bin:")
    try:
        files = os.listdir("/app/whisper.cpp/build/bin")
        for f in files:
            print(f"  {f}")
    except:
        pass
    
    return None

def transcribe(input_file, output_file, model="medium.en", threads=4, prompt_file=None):
    """Run whisper.cpp transcription"""
    
    # Ensure threads is an integer
    try:
        threads = int(threads)
    except (ValueError, TypeError):
        print(f"Warning: Invalid threads value '{threads}', using default 4")
        threads = 4
    
    # Find the whisper binary
    whisper_binary = find_whisper_binary()
    if not whisper_binary:
        print("Fatal error: whisper.cpp binary not found!")
        return 1
    
    print(f"Using whisper binary: {whisper_binary}")
    
    # Map model names to files
    model_path = f"/app/whisper.cpp/models/ggml-{model}.bin"
    
    if not os.path.exists(model_path):
        print(f"Error: Model {model} not found at {model_path}")
        print("Available models:")
        try:
            models_dir = "/app/whisper.cpp/models"
            for f in os.listdir(models_dir):
                if f.endswith(".bin"):
                    print(f"  {f}")
        except:
            pass
        return 1
    
    # Build command
    cmd = [
        whisper_binary,
        "-m", model_path,
        "-f", input_file,
        "-otxt",  # Output as text
        "-pp",    # Print progress
        "-nt",    # No timestamps in output
        "-t", str(threads),
        "-np",    # No prints (suppress transcription output)
    ]
    
    # Specify output file without extension (whisper.cpp adds .txt automatically)
    output_base = output_file.rsplit('.txt', 1)[0] if output_file and output_file.endswith('.txt') else output_file
    if output_base:
        cmd.extend(["-of", output_base])
    
    # Check if CUDA is available and add GPU layers
    try:
        # Test if whisper was built with CUDA support
        test_result = subprocess.run([whisper_binary, "--help"], capture_output=True, text=True)
        if "cuda" in test_result.stdout.lower() or "gpu" in test_result.stdout.lower():
            cmd.extend(["-ngl", "99"])  # Use GPU layers (new syntax)
            print("GPU acceleration enabled (using CUDA)")
            
            # Check if GPU is actually available
            nvidia_test = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
            if nvidia_test.returncode == 0:
                print(f"GPU detected: {nvidia_test.stdout.strip()}")
            else:
                print("WARNING: nvidia-smi not available - GPU may not work properly")
        else:
            print("WARNING: whisper.cpp was not built with CUDA support")
    except Exception as e:
        print(f"WARNING: Could not check GPU support: {e}")
    
    # Add prompt if provided
    temp_prompt_file = None
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
            # Extract just the essential prompt text, removing metadata
            if "TECHNICAL TERMINOLOGY" in prompt_text:
                # Find the section that contains the actual vocabulary guidance
                start = prompt_text.find("TECHNICAL TERMINOLOGY")
                if start > 0:
                    prompt_text = prompt_text[:start].strip()
        
        # Write prompt to a temporary file to avoid command line escaping issues
        temp_prompt_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_prompt_file.write(prompt_text[:2048])  # Whisper.cpp has a prompt length limit
        temp_prompt_file.close()
        
        cmd.extend(["--prompt", temp_prompt_file.name])
        print(f"Using prompt: {len(prompt_text)} characters (truncated to 2048)")
    
    print(f"Running transcription with {model} model...")
    print(f"Input: {input_file}")
    print(f"Model path: {model_path}")
    
    start_time = time.time()
    
    # Run whisper.cpp
    print("Processing... (this may take several minutes)")
    
    # If using the deprecated 'main' binary, we need to handle the deprecation warning
    if whisper_binary.endswith('/main'):
        print("WARNING: Using deprecated 'main' binary. The container should be rebuilt to use whisper-cli.")
        # Run with stderr redirected to handle deprecation warning
        result = subprocess.run(cmd, stderr=subprocess.DEVNULL)
    else:
        result = subprocess.run(cmd, capture_output=False)
    
    # Clean up temporary prompt file
    if temp_prompt_file:
        try:
            os.unlink(temp_prompt_file.name)
        except:
            pass
    
    elapsed = time.time() - start_time
    print(f"\nTranscription completed in {elapsed/60:.1f} minutes")
    
    if result.returncode != 0:
        print(f"Error: Transcription failed with return code {result.returncode}")
        return result.returncode
    
    # Look for output file in multiple possible locations
    base_name = os.path.basename(input_file).rsplit('.', 1)[0]
    possible_outputs = [
        input_file + ".txt",  # Next to input file
        os.path.join(os.path.dirname(input_file), base_name + ".txt"),  # In input dir
        os.path.join(output_dir, base_name + ".txt") if output_file else None,  # In output dir
        base_name + ".txt",  # In current directory
    ]
    
    output_found = None
    for possible in possible_outputs:
        if possible and os.path.exists(possible):
            output_found = possible
            print(f"Found output at: {output_found}")
            break
    
    if output_found:
        if output_file and output_found != output_file:
            # Move to desired location
            shutil.move(output_found, output_file)
            print(f"Output saved to: {output_file}")
        else:
            print(f"Output saved to: {output_found}")
        return 0
    else:
        print(f"Error: Output file not found in any expected location")
        print("Searched locations:")
        for loc in possible_outputs:
            if loc:
                print(f"  - {loc}")
        
        # List files in current directory to help debug
        print("\nFiles in current directory:")
        for f in os.listdir("."):
            if f.endswith(".txt"):
                print(f"  - {f}")
        
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper.cpp transcription wrapper")
    parser.add_argument("input_file", help="Audio file to transcribe")
    parser.add_argument("-o", "--output", help="Output file path", required=True)
    parser.add_argument("-m", "--model", default="medium.en", 
                       choices=["tiny.en", "base.en", "small.en", "medium.en"],
                       help="Model to use (default: medium.en)")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("-p", "--prompt", help="Prompt file to guide transcription")
    
    args = parser.parse_args()
    
    sys.exit(transcribe(args.input_file, args.output, args.model, args.threads, args.prompt))