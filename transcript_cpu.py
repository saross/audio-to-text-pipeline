#!/usr/bin/env python3
"""
CPU-Only Audio Transcription and Diarisation Script

This script takes an audio file, transcribes it using Whisper large-v3 model,
identifies speakers using pyannote.audio, and combines the results.
Optimised for highest quality on CPU using Whisper's built-in sequential processing.
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

# Force CPU usage by disabling CUDA visibility
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Set CPU optimization environment variables
os.environ["OMP_NUM_THREADS"] = "4"  # Adjust based on your CPU cores
os.environ["MKL_NUM_THREADS"] = "4"  # Intel Math Kernel Library threads
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # OpenBLAS threads

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

def cpu_transcribe_diarise(audio_file, output_file=None, num_speakers=None):
    """
    Process an audio file with high-quality CPU-based transcription and diarisation.
    
    Args:
        audio_file: Path to the audio file
        output_file: Where to save the transcript (defaults to input_filename_transcript.txt)
        num_speakers: Number of speakers in the recording (if known, otherwise auto-detected)
    
    Returns:
        Path to the output file
    """
    print(f"Processing file: {audio_file}")
    
    # Set default output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(audio_file)[0]
        output_file = f"{base_name}_transcript.txt"

    print("CPU-only mode enabled for highest quality transcription")
    
    # Load Whisper large-v3 model (latest and most accurate version)
    print("Loading Whisper large-v3 model on CPU (this will take a moment)...")
    model = whisper.load_model("large-v3")

    # Transcribe the entire audio file using Whisper's built-in sequential processing
    print("Transcribing audio (this may take a while on CPU)...")
    transcription = model.transcribe(
        audio_file,
        word_timestamps=True,
        language="en",  # You can change this for other languages
        verbose=False
    )
    
    print(f"Transcription complete with {len(transcription['segments'])} segments")
    
    # Clear model from memory before loading diarisation
    del model
    gc.collect()
    
    # Load pyannote.audio for speaker diarisation
    print("Loading diarisation model (this may take a moment)...")
    hf_token = os.environ.get('HUGGINGFACE_TOKEN')
    if not hf_token:
        print("ERROR: HUGGINGFACE_TOKEN environment variable not set.")
        print("Please set this with a valid token from huggingface.co")
        print("Example: export HUGGINGFACE_TOKEN=your_token_here")
        sys.exit(1)
    
    # Load the diarisation pipeline (always on CPU)
    try:
        diarisation = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
    except Exception as e:
        print(f"Error loading diarisation model: {e}")
        print("Please ensure you have accepted the model license at huggingface.co")
        sys.exit(1)

    # Perform speaker diarisation
    print("Performing speaker diarisation (this may take several minutes on CPU)...")
    print(f"Number of speakers: {'auto-detect' if num_speakers is None else num_speakers}")
    
    with tqdm(total=0, desc="Diarisation", bar_format='{desc}: {elapsed}') as pbar:
        with suppress_stderr():
            diarisation_result = diarisation(audio_file, num_speakers=num_speakers)
        pbar.set_description("Diarisation completed")
        
    # Combine transcription with speaker information
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
    
    # Merge consecutive segments from the same speaker
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
    
    # Save the transcript
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
    
    parser = argparse.ArgumentParser(description="CPU-Only high-quality transcription and diarisation")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--output", "-o", help="Output file path (default: [input]_transcript.txt)")
    parser.add_argument("--speakers", "-s", type=int, 
                       help="Number of speakers in the recording (if known, improves accuracy)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: File '{args.audio_file}' not found")
        sys.exit(1)
    
    # Run the CPU-optimized transcription and diarisation process
    transcript_file = cpu_transcribe_diarise(
        args.audio_file, 
        output_file=args.output,
        num_speakers=args.speakers
    )
    print(f"Transcript saved to: {transcript_file}")