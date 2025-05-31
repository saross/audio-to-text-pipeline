#!/usr/bin/env python3
"""
Audio Preprocessing Script for Speech Transcription and Diarization

PURPOSE:
    Converts MP4 audio files to FLAC format optimized for speech recognition (Whisper)
    and speaker diarization. Applies 7 processing steps to maximize transcription
    accuracy while minimizing file size.

USAGE:
    Single file conversion:
        python3 simple_preprocess.py convert input.mp4 output.flac
    
    Batch processing:
        python3 simple_preprocess.py batch /input/dir /output/dir
    
    Test mode (30-second preview):
        python3 simple_preprocess.py test input.mp4

PROCESSING STEPS:
    1. MP4 to FLAC conversion - Lossless format for processing
    2. Sample rate standardization - 16kHz (Whisper optimal)
    3. Stereo to mono conversion - Better for transcription
    4. Background noise reduction - High-pass filter + noise gate
    5. Volume normalization - EBU R128 standard (-16 LUFS)
    6. Dynamic range compression - Reduces volume variations
    7. Speech enhancement EQ - Boosts speech frequencies

TECHNICAL DETAILS:
    - Uses FFmpeg streaming (no full file loading into memory)
    - FLAC compression level 8 (maximum)
    - Output: 16kHz, mono, 16-bit
    - Typical size reduction: 90%
    
REQUIREMENTS:
    - Python 3.6+
    - FFmpeg with FLAC support
    - No additional Python packages

AUTHOR: Audio transcription preprocessing pipeline
DATE: 2024
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path

def run_ffmpeg(command, description="Processing", show_progress=True):
    """
    Execute FFmpeg command with error handling and optional progress display.
    
    Args:
        command (list): FFmpeg command as list of arguments
        description (str): Human-readable description of the operation
        show_progress (bool): Whether to show FFmpeg's progress output
        
    Returns:
        bool: True if successful, False if failed
        
    Note:
        Uses subprocess.run() without capturing output to avoid memory issues
        with large files. Progress is shown directly in terminal.
    """
    print(f"{description}...")
    try:
        if show_progress:
            # Show FFmpeg progress directly in terminal
            # This avoids buffering issues with large files
            result = subprocess.run(command, check=True)
        else:
            # Hide output for cleaner batch processing
            result = subprocess.run(command, check=True, 
                                  stdout=subprocess.DEVNULL, 
                                  stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: FFmpeg command failed with code {e.returncode}")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def check_file(filepath):
    """
    Validate input file and display basic information.
    
    Args:
        filepath (str): Path to the audio file
        
    Returns:
        bool: True if file exists and is accessible, False otherwise
        
    Note:
        Uses ffprobe to extract duration and format info. Continues even
        if probe fails, as FFmpeg may still be able to process the file.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return False
    
    # Use ffprobe to get file information
    # -v error: Only show errors
    # -show_entries: Select specific fields to display
    # -of default=noprint_wrappers=1: Simple output format
    cmd = ["ffprobe", "-v", "error", "-show_entries", 
           "format=duration,size:stream=codec_type,sample_rate,channels",
           "-of", "default=noprint_wrappers=1", filepath]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = result.stdout
        
        # Parse duration from ffprobe output
        duration_line = [l for l in info.split('\n') if 'duration=' in l]
        if duration_line:
            duration = float(duration_line[0].split('=')[1])
            print(f"Duration: {duration/60:.1f} minutes")
        
        # Display file size in MB
        print(f"Size: {os.path.getsize(filepath) / (1024*1024):.1f} MB")
        return True
    except:
        # ffprobe failed, but file exists - continue anyway
        print(f"Warning: Could not probe file info")
        return True  # Continue anyway

def convert_mp4_to_flac(input_file, output_file, verbose=False, remove_silence=False):
    """
    Convert MP4 to FLAC with all 7 processing steps:
    1. MP4 to FLAC conversion
    2. Sample rate standardisation (16kHz)
    3. Stereo to mono conversion
    4. Background noise reduction
    5. Volume normalisation
    6. Dynamic range compression
    7. Speech enhancement (EQ)
    
    Args:
        input_file: Path to input MP4 file
        output_file: Path to output FLAC file
        verbose: Show detailed processing information
        remove_silence: Enable VAD and silence removal (default: False)
    """
    
    # Check input file
    if not check_file(input_file):
        return False
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Build complex audio filter chain
    # Each filter is carefully tuned for speech transcription
    audio_filters = [
        # STEP 3: Stereo to mono conversion
        # Equal power mixing preserves content from both channels
        "pan=mono|c0=0.5*c0+0.5*c1",
        
        # STEP 4a: High-pass filter to remove low-frequency noise
        # 80Hz cutoff removes rumble, HVAC noise, handling noise
        # Speech fundamental frequency starts ~85Hz (male) to ~255Hz (female)
        "highpass=f=80",
        
        # STEP 4b: Noise gate to reduce background noise
        # threshold=0.02: -34dB gate threshold
        # ratio=2: Gentle 2:1 reduction below threshold
        # attack=10ms: Quick opening for speech
        # release=100ms: Smooth closing to avoid cutting words
        "agate=threshold=0.02:ratio=2:attack=10:release=100",
        
        # STEP 6: Dynamic range compression (gentler for better diarization)
        # threshold=-22dB: Slightly lower threshold
        # ratio=3:1: Gentler compression to preserve speaker characteristics
        # attack=10ms: Slightly slower for more natural sound
        # release=100ms: Slower release preserves speech dynamics
        "acompressor=threshold=-22dB:ratio=3:attack=10:release=100",
        
        # STEP 7: Speech enhancement EQ
        # Carefully tuned frequency adjustments for clarity
        
        # 7a: Reduce low frequency muddiness
        # 100Hz high-shelf, -2dB reduction
        "equalizer=f=100:t=h:width=200:g=-2",
        
        # 7b: Boost speech fundamentals
        # 1kHz Q-filter, +3dB boost for presence
        "equalizer=f=1000:t=q:width=2:g=3",
        
        # 7c: Boost speech clarity/consonants
        # 3kHz Q-filter, +2dB for intelligibility
        "equalizer=f=3000:t=q:width=2:g=2",
        
        # 7d: Reduce high frequency noise
        # 8kHz high-shelf, -3dB to reduce hiss
        "equalizer=f=8000:t=h:width=2000:g=-3",
        
        # 7e: High-frequency boost for Whisper consonant recognition
        # 4kHz high-shelf, +1dB gentle boost for 's', 't', 'k' sounds
        "equalizer=f=4000:t=h:width=2000:g=1",
        
        # STEP 5: Volume normalisation with EBU R128 standard
        # I=-16: Integrated loudness target (LUFS)
        # TP=-1.5: True peak limit (dB)
        # LRA=11: Loudness range target
        "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=summary",
        
        # Final safety limiter to prevent any clipping
        # limit=0.95: Leave 5% headroom
        # Fast attack/release for transparent limiting
        "alimiter=limit=0.95:attack=5:release=50"
    ]
    
    # Add optional silence removal filters
    if remove_silence:
        # Insert VAD and silence removal after noise gate, before compression
        silence_filters = [
            # Remove silence periods longer than 0.5 seconds
            # start_threshold=-35dB: Fairly sensitive to preserve quiet speech
            # stop_threshold=-35dB: Same threshold for consistency
            "silenceremove=start_periods=1:start_duration=0.5:start_threshold=-35dB:stop_periods=-1:stop_duration=0.5:stop_threshold=-35dB",
        ]
        # Insert after noise gate (index 3) but before compression
        audio_filters = audio_filters[:3] + silence_filters + audio_filters[3:]
    
    # Join all filters with commas
    filter_string = ",".join(audio_filters)
    
    # Build FFmpeg command
    command = [
        "ffmpeg",
        "-i", input_file,              # Input file
        "-vn",                         # No video
        "-acodec", "flac",            # FLAC codec (step 1)
        "-ar", "16000",               # 16kHz sample rate (step 2)
        "-sample_fmt", "s16",         # 16-bit samples
        "-compression_level", "8",     # Maximum FLAC compression (8 is max)
        "-af", filter_string,          # Apply all audio filters
        "-y",                         # Overwrite output
        output_file
    ]
    
    if verbose:
        print(f"\nApplying audio processing pipeline:")
        print("1. ✓ MP4 to FLAC conversion")
        print("2. ✓ Sample rate: 16kHz")
        print("3. ✓ Stereo to mono")
        print("4. ✓ Noise reduction (gate + high-pass)")
        if remove_silence:
            print("   ✓ VAD silence removal (>0.5s gaps)")
        print("5. ✓ Volume normalisation")
        print("6. ✓ Dynamic range compression (gentle 3:1)")
        print("7. ✓ Speech enhancement EQ + HF boost")
        print(f"\nCommand: {' '.join(command[:6])}... [complex filter chain] ... {output_file}")
    
    # Check if output file already exists
    if os.path.exists(output_file):
        print(f"⚠️  Overwriting existing file: {output_file}")
    
    # Run conversion
    success = run_ffmpeg(command, f"Converting {os.path.basename(input_file)} to optimized FLAC", 
                        show_progress=not verbose)
    
    if success and os.path.exists(output_file):
        output_size = os.path.getsize(output_file) / (1024*1024)
        print(f"✓ Success: Created {output_file} ({output_size:.1f} MB)")
        return True
    else:
        print(f"✗ Failed to create {output_file}")
        return False

def process_batch(input_dir, output_dir, pattern="*.mp4", dry_run=False, verbose=False, remove_silence=False):
    """Process all matching files in a directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all matching files
    files = list(input_path.glob(pattern))
    if not files:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return
    
    print(f"Found {len(files)} files to process")
    
    # Create output directory
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    successful = 0
    failed = 0
    
    for i, input_file in enumerate(files, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(files)}] Processing: {input_file.name}")
        print(f"{'='*60}")
        
        # Generate output filename
        output_file = output_path / f"{input_file.stem}.flac"
        
        if dry_run:
            print(f"  Would create: {output_file}")
            successful += 1
        else:
            if convert_mp4_to_flac(str(input_file), str(output_file), verbose, remove_silence):
                successful += 1
            else:
                failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description="Simple audio preprocessor for transcription - converts MP4 to FLAC with 7-step optimization"
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Single file conversion
    single_parser = subparsers.add_parser('convert', help='Convert a single file')
    single_parser.add_argument('input', help='Input MP4 file')
    single_parser.add_argument('output', help='Output FLAC file')
    single_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    single_parser.add_argument('--remove-silence', action='store_true', 
                              help='Remove silence gaps longer than 0.5 seconds (VAD)')
    
    # Batch processing
    batch_parser = subparsers.add_parser('batch', help='Process multiple files')
    batch_parser.add_argument('input_dir', help='Input directory')
    batch_parser.add_argument('output_dir', help='Output directory')
    batch_parser.add_argument('-p', '--pattern', default='*.mp4', help='File pattern (default: *.mp4)')
    batch_parser.add_argument('-d', '--dry-run', action='store_true', help='Show what would be done')
    batch_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    batch_parser.add_argument('--remove-silence', action='store_true', 
                             help='Remove silence gaps longer than 0.5 seconds (VAD)')
    
    # Quick test
    test_parser = subparsers.add_parser('test', help='Test with first 30 seconds of a file')
    test_parser.add_argument('input', help='Input MP4 file')
    test_parser.add_argument('-o', '--output', help='Output FLAC file (default: test_output.flac)')
    
    args = parser.parse_args()
    
    if args.command == 'convert':
        # Single file conversion
        success = convert_mp4_to_flac(args.input, args.output, args.verbose, args.remove_silence)
        sys.exit(0 if success else 1)
    
    elif args.command == 'batch':
        # Batch processing
        process_batch(args.input_dir, args.output_dir, args.pattern, args.dry_run, 
                     args.verbose, args.remove_silence)
    
    elif args.command == 'test':
        # Test with first 30 seconds
        print("Testing with first 30 seconds of audio...")
        output = args.output or 'test_output.flac'
        
        # Create temp file for 30-second clip
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            temp_file = tmp.name
        
        # Extract first 30 seconds
        cmd = ["ffmpeg", "-i", args.input, "-t", "30", "-c", "copy", "-y", temp_file]
        if run_ffmpeg(cmd, "Extracting 30-second test clip", show_progress=False):
            # Process the clip
            success = convert_mp4_to_flac(temp_file, output, verbose=True)
            # Clean up
            os.unlink(temp_file)
            sys.exit(0 if success else 1)
        else:
            print("Failed to extract test clip")
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()