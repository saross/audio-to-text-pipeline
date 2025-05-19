#!/usr/bin/env python3
"""
GPU-Optimised Audio Transcription and Diarisation Script

This script takes an audio file, transcribes it using Whisper medium model on GPU,
and identifies speakers using pyannote.audio on CPU.
Optimised for 6GB VRAM GPUs using full precision and efficient resource allocation.
"""

import os
import sys
import whisper
import torch
import numpy as np
from pyannote.audio import Pipeline
from pyannote.core import Segment
from tqdm import tqdm
import contextlib
import gc
import re

# Step 0: Context manager for suppressing warnings
@contextlib.contextmanager
def suppress_stderr():
    """Context manager to temporarily redirect stderr to /dev/null."""
    old_stderr = sys.stderr
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stderr = os.fdopen(null_fd, 'w')
        yield
    finally:
        sys.stderr = old_stderr

def check_gpu_memory(required_mb=1000):
    """Check if there's enough GPU memory available."""
    if torch.cuda.is_available():
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        free_mb = free_memory / (1024 * 1024)
        if free_mb < required_mb:
            print(f"Warning: Only {free_mb:.0f}MB GPU memory available, {required_mb}MB recommended")
        return free_mb >= required_mb
    return False

def is_gpu_for_display():
    """Check if NVIDIA GPU is being used for display."""
    try:
        # Use glxinfo to check renderer
        import subprocess
        result = subprocess.run(['glxinfo', '|', 'grep', 'renderer'], 
                              shell=True, capture_output=True, text=True)
        return "NVIDIA" in result.stdout
    except:
        # If we can't determine, assume it's not used for display
        return False

def load_whisper_prompt(filepath="whisper_prompt.txt"):
    """Load the initial prompt for guiding Whisper transcription."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        print(f"Loaded Whisper prompt from {filepath} ({len(prompt)} characters)")
        return prompt
    except FileNotFoundError:
        print(f"Whisper prompt file {filepath} not found, using default prompt.")
        return "This is a technical discussion that may include specialized terminology."

def load_term_corrections(filepath="term_corrections.txt"):
    """Load term correction dictionary from a file for post-processing."""
    corrections = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('|')
                    if len(parts) == 2:
                        incorrect, correct = parts
                        corrections[incorrect.strip()] = correct.strip()
        print(f"Loaded {len(corrections)} term corrections from {filepath}")
        return corrections
    except FileNotFoundError:
        print(f"Term corrections file {filepath} not found, no post-processing will be applied.")
        return {}

def apply_term_corrections(text, corrections):
    """Apply the term corrections to the text."""
    if not corrections:
        return text
    
    corrected_text = text
    for incorrect, correct in corrections.items():
        # Case-insensitive replacement with word boundary check
        pattern = re.compile(r'\b' + re.escape(incorrect) + r'\b', re.IGNORECASE)
        corrected_text = pattern.sub(correct, corrected_text)
    
    return corrected_text

def optimised_transcribe_diarise(audio_file, output_file=None, num_speakers=None, prompt_file="whisper_prompt.txt", corrections_file="term_corrections.txt"):
    """
    Process an audio file with GPU-optimised transcription and CPU diarisation.
    
    Args:
        audio_file: Path to the audio file
        output_file: Where to save the transcript (defaults to input_filename_transcript.txt)
        num_speakers: Number of speakers in the recording (if known, otherwise auto-detected)
        prompt_file: Path to the Whisper prompt file
        corrections_file: Path to the term corrections file
    
    Returns:
        Path to the output file
    """
    print(f"Processing file: {audio_file}")
    
    # Set default output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(audio_file)[0]
        output_file = f"{base_name}_transcript.txt"

    # Load prompt and corrections
    whisper_prompt = load_whisper_prompt(prompt_file)
    term_corrections = load_term_corrections(corrections_file)

    # Setup device and model configurations
    if torch.cuda.is_available():
        # Check if GPU is being used for display
        if is_gpu_for_display():
            print("Warning: NVIDIA GPU is being used for display. This may reduce available VRAM.")
            print("For best results, connect your display to integrated graphics.")
        
        device = "cuda"
        # Check available VRAM
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"GPU detected with {vram_mb:.0f}MB VRAM")
    else:
        device = "cpu"
    
    print(f"Using device: {device} with full precision (FP32)")
    
    # Step 1: Load Whisper model
    print("Loading Whisper medium.en model optimized for English...")
    model_size = "medium.en"  # Changed to English-specific model
    
    # Load model on the selected device with full precision
    model = whisper.load_model(model_size, device=device)
    
    # Step 2: Transcribe using Whisper's built-in sequential processing
    print("Transcribing audio (this may take several minutes)...")
    
    # Use tqdm to show a progress bar instead of verbose output
    with tqdm(total=0, desc="Transcription", bar_format='{desc}: {elapsed}') as pbar:
        # Handle the Triton/CUDA compatibility issue by using different strategy for word timestamps
        try:
            transcription = model.transcribe(
                audio_file,
                word_timestamps=True,
                verbose=False,  # Disable verbose output to prevent partial transcription printout
                initial_prompt=whisper_prompt  # Use prompt to guide transcription
            )
        except AttributeError as e:
            # If we get the Triton/CUDA error, try again with a different approach
            if "Cannot set attribute 'src' directly" in str(e):
                print("\nDetected Triton/CUDA compatibility issue. Trying alternative approach...")
                # Move model to CPU temporarily for word timestamp calculation
                model = model.to("cpu")
                transcription = model.transcribe(
                    audio_file,
                    word_timestamps=True,
                    verbose=False,
                    initial_prompt=whisper_prompt # Use prompt for transcription
                )
                # Move model back to GPU if it was on GPU before
                if device == "cuda":
                    model = model.to(device)
            else:
                raise  # Re-raise if it's a different error
                
        pbar.set_description("Transcription completed")
    
    # Apply term corrections if available
    if term_corrections:
        print("Applying technical term corrections...")
        for segment in transcription["segments"]:
            segment["text"] = apply_term_corrections(segment["text"], term_corrections)
    
    # Clear Whisper model from memory before loading diarisation
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    # Step 3: Load pyannote.audio for speaker diarisation on CPU
    print("Loading diarisation model on CPU...")
    hf_token = os.environ.get('HUGGINGFACE_TOKEN')
    if not hf_token:
        print("ERROR: HUGGINGFACE_TOKEN environment variable not set.")
        print("Please set this with a valid token from huggingface.co")
        print("Example: export HUGGINGFACE_TOKEN=your_token_here")
        sys.exit(1)
    
    # Explicitly force CPU for diarisation
    try:
        # Save the original CUDA visibility and temporarily force CPU
        original_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        diarisation = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        # Restore the original CUDA visibility
        if original_cuda_visible_devices:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible_devices
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            
    except Exception as e:
        print(f"Error loading diarisation model: {e}")
        print("Please ensure you have accepted the model license at huggingface.co")
        sys.exit(1)

    # Step 4: Perform speaker diarisation
    print("Performing speaker diarisation on CPU (this may take several minutes)...")
    print(f"Number of speakers: {'auto-detect' if num_speakers is None else num_speakers}")
    
    # Redirect both stdout and stderr to /dev/null during diarisation
    with tqdm(total=0, desc="Diarisation", bar_format='{desc}: {elapsed}') as pbar:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        null_fd = os.open(os.devnull, os.O_WRONLY)
        null_out = os.fdopen(null_fd, 'w')
        
        try:
            sys.stderr = null_out
            sys.stdout = null_out
            diarisation_result = diarisation(audio_file, num_speakers=num_speakers)
        finally:
            sys.stderr = old_stderr
            sys.stdout = old_stdout
            null_out.close()
            
        pbar.set_description("Diarisation completed")
        
    # Step 5: Combine transcription with speaker information
    print("Combining transcription with speaker information...")
    combined_segments = []
    
    # For each segment in the transcription
    for segment in transcription["segments"]:
        # Get the time range for this segment
        start_time = segment["start"]
        end_time = segment["end"]
        segment_range = Segment(start_time, end_time)
        
        # Find which speaker was active during this segment
        speaker_times = {}
        for turn, _, speaker in diarisation_result.itertracks(yield_label=True):
            # Check if this speaker turn overlaps with our segment
            if segment_range.intersects(turn):
                # Calculate the intersection
                overlap_start = max(segment_range.start, turn.start)
                overlap_end = min(segment_range.end, turn.end)
                overlap_duration = overlap_end - overlap_start
                
                # Add to the speaker's total time
                speaker_times[speaker] = speaker_times.get(speaker, 0) + overlap_duration
        
        # Find the speaker who talked the most during this segment
        dominant_speaker = max(speaker_times.items(), key=lambda x: x[1])[0] if speaker_times else "UNKNOWN"
        
        # Create a combined entry with both text and speaker
        combined_segment = {
            "start": start_time,
            "end": end_time,
            "speaker": dominant_speaker,
            "text": segment["text"].strip()
        }
        
        combined_segments.append(combined_segment)
    
    # Step 6: Merge consecutive segments from the same speaker
    print("Merging consecutive segments from the same speaker...")
    merged_segments = []
    if combined_segments:
        current = combined_segments[0].copy()
        for segment in combined_segments[1:]:
            # If this is the same speaker as the previous segment, merge them
            if segment["speaker"] == current["speaker"] and segment["start"] - current["end"] < 1.0:
                current["end"] = segment["end"]
                current["text"] += " " + segment["text"]
            else:
                merged_segments.append(current)
                current = segment.copy()

        # Don't forget to add the last segment
        merged_segments.append(current)

    # Filter out empty UNKNOWN segments
    print("Removing empty UNKNOWN speaker segments...")
    merged_segments = [segment for segment in merged_segments 
                       if not (segment["speaker"] == "UNKNOWN" and not segment["text"].strip())]
    
    # Step 7: Save the transcript
    print(f"Saving transcript to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write a header
        f.write(f"# Transcript of {os.path.basename(audio_file)}\n\n")
        
        # Write each segment with timestamp and speaker
        for segment in merged_segments:
            # Format timestamp as [HH:MM:SS]
            m, s = divmod(segment["start"], 60)
            h, m = divmod(m, 60)
            timestamp = f"[{int(h):02d}:{int(m):02d}:{int(s):02d}]"
            
            # Write the line
            f.write(f"{timestamp} {segment['speaker']}: {segment['text']}\n\n")
    
    print("Transcription and diarisation complete!")
    return output_file

# Main function to run if this script is executed directly
if __name__ == "__main__":
    # Command line argument parsing using argparse
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe and diarise an audio file")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--output", "-o", help="Output file path (default: [input]_transcript.txt)")
    parser.add_argument("--speakers", "-s", type=int, 
                       help="Number of speakers in the recording (if known, improves accuracy)")
    parser.add_argument("--prompt", "-p", help="Path to Whisper prompt file (default: whisper_prompt.txt)")
    parser.add_argument("--corrections", "-c", help="Path to term corrections file (default: term_corrections.txt)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: File '{args.audio_file}' not found")
        sys.exit(1)
    
    # Run the optimised transcription and diarisation process
    transcript_file = optimised_transcribe_diarise(
        args.audio_file, 
        output_file=args.output,
        num_speakers=args.speakers,
        prompt_file=args.prompt if args.prompt else "whisper_prompt.txt",
        corrections_file=args.corrections if args.corrections else "term_corrections.txt"
    )
    print(f"Transcript saved to: {transcript_file}")
