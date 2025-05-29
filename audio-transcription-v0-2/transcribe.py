#!/usr/bin/env python3
"""
Audio Transcription Script with Academic/Research Optimisations

This script uses faster-whisper (CTranslate2) for efficient transcription that fits in 6GB VRAM,
with support for academic prompting, term corrections, and speaker diarisation.
"""

import os
import sys
import torch
from faster_whisper import WhisperModel
from tqdm import tqdm
import gc
import re
import argparse
from datetime import datetime

def load_whisper_prompt(filepath="whisper_prompt.txt"):
    """Load the initial prompt for guiding transcription."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        print(f"Loaded academic prompt from {filepath} ({len(prompt)} characters)")
        return prompt
    except FileNotFoundError:
        print(f"Prompt file {filepath} not found, using default academic prompt.")
        return "This is a technical discussion about research vocabulary management and semantic technologies."

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

def get_optimal_config_for_6gb(model_size="medium"):
    """Get optimal configuration for 6GB VRAM GPU with faster-whisper."""
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "compute_type": "float16" if torch.cuda.is_available() else "float32",
    }
    
    if torch.cuda.is_available():
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"GPU detected with {vram_mb:.0f}MB VRAM")
        
        # faster-whisper is much more memory efficient than regular whisper
        if model_size == "large-v3":
            # Large model can run on 6GB with faster-whisper!
            config["compute_type"] = "float16"
            print(f"Large-v3 model: using float16 precision (fits in 6GB with faster-whisper)")
        elif model_size == "medium":
            config["compute_type"] = "float16"
            print(f"Medium model: using float16 precision")
        
        print(f"Using device: {config['device']} with compute type: {config['compute_type']}")
    else:
        print("Using CPU with float32 precision")
    
    return config

def add_speaker_diarisation_simple(segments, audio_duration):
    """
    Simple speaker alternation for when we don't have proper diarisation.
    This is a fallback approach - alternates speakers based on silence gaps.
    """
    if not segments:
        return segments
    
    # Add simple speaker labels based on gaps between segments
    current_speaker = 0
    speaker_segments = []
    
    for i, segment in enumerate(segments):
        # If there's a significant gap (>2 seconds) or very different timing, change speaker
        if i > 0:
            gap = segment["start"] - segments[i-1]["end"]
            if gap > 2.0:  # 2+ second gap suggests speaker change
                current_speaker = 1 - current_speaker  # Alternate between 0 and 1
        
        segment_with_speaker = segment.copy()
        segment_with_speaker["speaker"] = f"SPEAKER_{current_speaker:02d}"
        speaker_segments.append(segment_with_speaker)
    
    return speaker_segments

def faster_whisper_transcribe(audio_file, output_file=None, model_size="medium", language="en",
                            prompt_file="whisper_prompt.txt", corrections_file="term_corrections.txt"):
    """
    Process an audio file with faster-whisper transcription.
    
    Args:
        audio_file: Path to the audio file
        output_file: Where to save the transcript
        model_size: "tiny", "small", "medium", "large-v3"
        language: Language code (default: "en" for English)
        prompt_file: Path to the academic prompt file
        corrections_file: Path to the term corrections file
    
    Returns:
        Path to the output file
    """
    print(f"Processing file: {audio_file}")
    print(f"Using model: {model_size}")
    
    # Set default output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(audio_file)[0]
        output_file = f"{base_name}_transcript.txt"

    # Load prompt and corrections
    whisper_prompt = load_whisper_prompt(prompt_file)
    term_corrections = load_term_corrections(corrections_file)

    # Get optimal configuration for 6GB GPU
    config = get_optimal_config_for_6gb(model_size)
    device = config["device"]
    compute_type = config["compute_type"]
    
    # Step 1: Load transcription model
    print(f"Loading {model_size} transcription model...")
    try:
        model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type
        )
        print("✅ Model loaded successfully")

        # GPU debugging
        if torch.cuda.is_available() and device == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.1f}MB")
            print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**2:.1f}MB")

    except Exception as e:
        print(f"Error loading model on GPU: {e}")
        print("Falling back to CPU...")
        device = "cpu"
        compute_type = "float32"
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    # Step 2: Transcribe audio
    print("Transcribing audio...")
    
    # faster-whisper API is different - it returns segments and info
    with tqdm(total=0, desc="Transcription", bar_format='{desc}: {elapsed}') as pbar:
        try:
            # faster-whisper transcribe method
            segments, info = model.transcribe(
                audio_file,
                language=language,
                initial_prompt=whisper_prompt,  # faster-whisper supports initial_prompt!
                beam_size=5,
                best_of=5,
                temperature=0.0
            )
            
            # Convert generator to list
            segments_list = list(segments)
            pbar.set_description("Transcription completed")
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            print("Trying without prompt...")
            segments, info = model.transcribe(audio_file, language=language)
            segments_list = list(segments)
    
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    print(f"Found {len(segments_list)} segments")
    
    # Step 3: Convert faster-whisper segments to our format
    processed_segments = []
    for segment in segments_list:
        processed_segment = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        }
        processed_segments.append(processed_segment)
    
    # Step 4: Apply term corrections
    if term_corrections:
        print("Applying academic term corrections...")
        for segment in processed_segments:
            segment["text"] = apply_term_corrections(segment["text"], term_corrections)
    
    # Step 5: Add simple speaker diarisation (fallback approach)
    print("Adding speaker labels...")
    audio_duration = processed_segments[-1]["end"] if processed_segments else 0
    final_segments = add_speaker_diarisation_simple(processed_segments, audio_duration)
    
    # Step 6: Merge consecutive segments from the same speaker
    print("Merging consecutive segments from the same speaker...")
    merged_segments = []
    if final_segments:
        current = final_segments[0].copy()
        for segment in final_segments[1:]:
            # If this is the same speaker as the previous segment and close in time, merge them
            if (segment["speaker"] == current["speaker"] and 
                segment["start"] - current["end"] < 1.0):
                current["end"] = segment["end"]
                current["text"] += " " + segment["text"]
            else:
                merged_segments.append(current)
                current = segment.copy()
        
        # Don't forget to add the last segment
        merged_segments.append(current)
    
    # Filter out very short segments
    merged_segments = [segment for segment in merged_segments 
                       if len(segment["text"].strip()) > 2]
    
    # Step 7: Save the transcript
    print(f"Saving transcript to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header with metadata
        f.write(f"# Transcript of {os.path.basename(audio_file)}\n")
        f.write(f"# Transcribed on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"# Model: {model_size} (faster-whisper)\n")
        f.write(f"# Language: {info.language} (confidence: {info.language_probability:.2f})\n")
        f.write(f"# Device: {device} ({compute_type})\n")
        if term_corrections:
            f.write(f"# Term corrections applied: {len(term_corrections)} terms\n")
        f.write("\n")
        
        # Write each segment with timestamp and speaker
        for segment in merged_segments:
            # Format timestamp as [HH:MM:SS]
            m, s = divmod(segment["start"], 60)
            h, m = divmod(m, 60)
            timestamp = f"[{int(h):02d}:{int(m):02d}:{int(s):02d}]"
            
            # Write the line
            f.write(f"{timestamp} {segment['speaker']}: {segment['text']}\n\n")
    
    # Clean up
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    print("Audio transcription complete!")
    return output_file

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio with academic/research optimisations")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--output", "-o", help="Output file path (default: [input]_transcript.txt)")
    parser.add_argument("--model", "-m", choices=["tiny", "small", "medium", "large-v3"], default="medium",
                       help="Model size: tiny, small, medium, or large-v3 (default: medium)")
    parser.add_argument("--language", "-l", default="en", 
                       help="Language code (default: en for English)")
    parser.add_argument("--prompt", "-p", help="Path to academic prompt file (default: whisper_prompt.txt)")
    parser.add_argument("--corrections", "-c", help="Path to term corrections file (default: term_corrections.txt)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: File '{args.audio_file}' not found")
        sys.exit(1)
    
    # Run the faster-whisper transcription process
    transcript_file = faster_whisper_transcribe(
        args.audio_file, 
        output_file=args.output,
        model_size=args.model,
        language=args.language,
        prompt_file=args.prompt if args.prompt else "whisper_prompt.txt",
        corrections_file=args.corrections if args.corrections else "term_corrections.txt"
    )
    print(f"Transcript saved to: {transcript_file}")
