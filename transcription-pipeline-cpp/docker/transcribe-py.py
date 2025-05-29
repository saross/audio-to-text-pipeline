#!/usr/bin/env python3
import subprocess
import sys
import os
import argparse
import time
import shutil

def transcribe(input_file, output_file, model="medium.en", threads=4, prompt_file=None):
    """Run whisper.cpp transcription"""
    
    # Map model names to files
    if model == "large-v3-q5_0":
        model_path = "/app/whisper.cpp/models/ggml-large-v3-q5_0.bin"
    else:
        model_path = f"/app/whisper.cpp/models/ggml-{model}.bin"
    
    if not os.path.exists(model_path):
        print(f"Error: Model {model} not found at {model_path}")
        return 1
    
    # Build command
    cmd = [
        "/app/whisper.cpp/main",
        "-m", model_path,
        "-f", input_file,
        "-otxt",  # Output as text
        "-pp",    # Print progress
        "-nt",    # No timestamps in output
        "-t", str(threads),
        "--gpu-layers", "99",  # Use GPU for all layers
    ]
    
    # Add prompt if provided
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
            # Extract just the essential prompt text, removing metadata
            if "TECHNICAL TERMINOLOGY" in prompt_text:
                # Find the section that contains the actual vocabulary guidance
                start = prompt_text.find("TECHNICAL TERMINOLOGY")
                if start > 0:
                    prompt_text = prompt_text[:start].strip()
        cmd.extend(["--prompt", prompt_text[:2048]])  # Whisper.cpp has a prompt length limit
        print(f"Using prompt: {len(prompt_text)} characters (truncated to 2048)")
    
    print(f"Running transcription with {model} model...")
    print(f"Input: {input_file}")
    print(f"Model path: {model_path}")
    
    start_time = time.time()
    
    # Run whisper.cpp
    result = subprocess.run(cmd, capture_output=False)
    
    elapsed = time.time() - start_time
    print(f"\nTranscription completed in {elapsed/60:.1f} minutes")
    
    # whisper.cpp creates input_file.txt in the same directory
    expected_output = input_file + ".txt"
    
    if result.returncode == 0 and os.path.exists(expected_output):
        if output_file:
            # Move to desired location
            shutil.move(expected_output, output_file)
            print(f"Output saved to: {output_file}")
        else:
            print(f"Output saved to: {expected_output}")
    else:
        print(f"Error: Expected output file not found: {expected_output}")
        return 1
    
    return result.returncode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper.cpp transcription wrapper")
    parser.add_argument("input_file", help="Audio file to transcribe")
    parser.add_argument("-o", "--output", help="Output file path", required=True)
    parser.add_argument("-m", "--model", default="medium.en", 
                       choices=["tiny.en", "base.en", "small.en", "medium.en", "large-v3-q5_0"],
                       help="Model to use")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("-p", "--prompt", help="Prompt file to guide transcription")
    
    args = parser.parse_args()
    
    sys.exit(transcribe(args.input_file, args.output, args.model, args.threads, args.prompt))