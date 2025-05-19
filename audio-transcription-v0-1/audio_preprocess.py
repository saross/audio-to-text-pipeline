#!/usr/bin/env python3
"""
Audio Pre-processor for Transcription and Diarisation

This script performs various audio processing steps to optimize recordings
for speech recognition (Whisper) and speaker diarisation (pyannote.audio).

It improves audio quality through noise reduction, volume normalisation,
dynamic range compression, and other techniques to enhance transcription accuracy.

Requires the same Python environment as the transcription scripts:
- Python 3.8+ with torch, librosa, soundfile, numpy, pydub
- FFmpeg installed for audio format conversion
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
import librosa
import warnings
from tqdm import tqdm
import contextlib
import subprocess
import tempfile
import shutil
from scipy import signal
import pydub
import re
import time

# Suppress warnings
warnings.filterwarnings("ignore")

def log(message, verbose=True):
    """Log a message if verbose mode is enabled."""
    if verbose:
        print(message)

def run_command(command, verbose=True):
    """Run a shell command and capture output."""
    log(f"Running: {' '.join(command)}", verbose)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(command)}")
        print(f"Error: {result.stderr}")
        return False
    return True

def get_audio_duration(file_path):
    """Get the duration of an audio file in seconds using FFmpeg."""
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
               "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting audio duration: {e}")
    return None

def convert_to_flac(input_file, output_file=None, verbose=True):
    """Convert audio file to FLAC format with FFmpeg with progress bar."""
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.flac"
    
    log(f"Converting {input_file} to FLAC format...", verbose)
    
    # Get audio duration for progress calculation
    duration = get_audio_duration(input_file)
    if duration is None:
        log("Couldn't determine audio duration, progress bar will be indeterminate", verbose)
    
    # Create temporary file for FFmpeg progress output
    progress_file = tempfile.mktemp()
    
    # Use FFmpeg for conversion with progress output
    command = [
        "ffmpeg", 
        "-i", input_file, 
        "-c:a", "flac", 
        "-progress", progress_file,
        "-y",  # Overwrite output if exists
        output_file
    ]
    
    try:
        # Start FFmpeg process
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Initialize progress bar
        pbar = tqdm(total=100, desc="Converting to FLAC", disable=not verbose)
        last_progress = 0
        
        # Monitor progress file
        while process.poll() is None:
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    progress_text = f.read()
                    
                    # Parse progress information
                    if duration and "out_time_ms=" in progress_text:
                        time_matches = re.findall(r'out_time_ms=(\d+)', progress_text)
                        if time_matches:
                            # Convert microseconds to seconds
                            current_time = int(time_matches[-1]) / 1000000
                            # Calculate percentage
                            progress = min(100, int(100 * current_time / duration))
                            
                            # Update progress bar
                            if progress > last_progress:
                                pbar.update(progress - last_progress)
                                last_progress = progress
            
            # Avoid heavy CPU usage
            time.sleep(0.1)
        
        # Ensure progress bar reaches 100%
        if last_progress < 100:
            pbar.update(100 - last_progress)
        
        pbar.close()
        
        # Check if process was successful
        if process.returncode != 0:
            stderr = process.stderr.read()
            print(f"Error in FFmpeg: {stderr}")
            return None
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return None
    finally:
        # Clean up temporary files
        if os.path.exists(progress_file):
            os.remove(progress_file)
    
    return output_file

def resample_audio(input_file, output_file=None, target_sr=16000, verbose=True):
    """Resample audio to target sample rate."""
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        ext = os.path.splitext(input_file)[1]
        output_file = f"{base_name}_resampled{ext}"
    
    log(f"Resampling to {target_sr}Hz...", verbose)
    
    # Get input sample rate
    info = sf.info(input_file)
    if info.samplerate == target_sr:
        log(f"File already at {target_sr}Hz, skipping resampling", verbose)
        return input_file
    
    # Use FFmpeg for resampling
    command = [
        "ffmpeg", "-i", input_file,
        "-ar", str(target_sr),
        "-y",
        output_file
    ]
    
    success = run_command(command, verbose)
    return output_file if success else None

def reduce_noise(input_file, output_file=None, strength=0.5, verbose=True):
    """Reduce background noise using spectral gating."""
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        ext = os.path.splitext(input_file)[1]
        output_file = f"{base_name}_noisereduced{ext}"
    
    log("Reducing background noise...", verbose)
    
    # Load audio
    try:
        audio, sr = librosa.load(input_file, sr=None)
    except Exception as e:
        print(f"Error loading audio file for noise reduction: {e}")
        return input_file
    
    # Calculate noise profile from the first 2 seconds
    # (assuming there's some silence or background noise only)
    noise_length = min(int(2 * sr), len(audio) // 4)
    noise_sample = audio[:noise_length]
    
    # Compute noise profile
    noise_fft = np.abs(librosa.stft(noise_sample))
    noise_profile = np.mean(noise_fft, axis=1)
    
    # Apply spectral gating
    audio_stft = librosa.stft(audio)
    audio_power = np.abs(audio_stft) ** 2
    noise_power = np.reshape(noise_profile ** 2, (-1, 1))
    
    # Create mask based on noise profile and apply strength parameter
    mask = audio_power - noise_power * (1 + strength)
    mask = np.maximum(mask, 0)
    mask = mask / (audio_power + 1e-10)
    mask = np.sqrt(mask)
    
    # Apply mask to audio
    audio_denoised = librosa.istft(audio_stft * mask)
    
    # Save output
    try:
        sf.write(output_file, audio_denoised, sr)
    except Exception as e:
        print(f"Error saving denoised audio: {e}")
        return input_file
    
    return output_file

def normalize_volume(input_file, output_file=None, target_level=-18, verbose=True):
    """Normalize audio volume to target RMS level (in dB)."""
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        ext = os.path.splitext(input_file)[1]
        output_file = f"{base_name}_normalized{ext}"
    
    log(f"Normalizing volume to {target_level}dB RMS...", verbose)
    
    # Use FFmpeg for normalization
    command = [
        "ffmpeg", "-i", input_file,
        "-filter:a", f"loudnorm=I={target_level}:LRA=11:TP=-1.5",
        "-y",
        output_file
    ]
    
    success = run_command(command, verbose)
    return output_file if success else None

def apply_compression(input_file, output_file=None, threshold=-20, ratio=3, verbose=True):
    """Apply dynamic range compression."""
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        ext = os.path.splitext(input_file)[1]
        output_file = f"{base_name}_compressed{ext}"
    
    log(f"Applying dynamic range compression (threshold: {threshold}dB, ratio: {ratio}:1)...", verbose)
    
    # Use FFmpeg for compression
    command = [
        "ffmpeg", "-i", input_file,
        "-filter:a", f"acompressor=threshold={threshold}dB:ratio={ratio}:attack=200:release=1000",
        "-y",
        output_file
    ]
    
    success = run_command(command, verbose)
    return output_file if success else None

def enhance_speech(input_file, output_file=None, verbose=True):
    """Enhance speech frequencies with EQ."""
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        ext = os.path.splitext(input_file)[1]
        output_file = f"{base_name}_enhanced{ext}"
    
    log("Enhancing speech frequencies...", verbose)
    
    # Use FFmpeg with multi-band EQ to enhance speech clarity
    # Slightly boost 1-3kHz range, reduce below 100Hz, apply gentle high shelf above 7kHz
    command = [
        "ffmpeg", "-i", input_file,
        "-filter:a", "equalizer=f=100:width_type=h:width=100:g=-3, equalizer=f=1000:width_type=q:width=1:g=2, equalizer=f=2500:width_type=q:width=1:g=3, equalizer=f=7000:width_type=h:width=1000:g=-2",
        "-y",
        output_file
    ]
    
    success = run_command(command, verbose)
    return output_file if success else None

def trim_silence(input_file, output_file=None, threshold=0.02, min_silence_duration=1.0, verbose=True):
    """Trim leading and trailing silence."""
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        ext = os.path.splitext(input_file)[1]
        output_file = f"{base_name}_trimmed{ext}"
    
    log("Trimming leading and trailing silence...", verbose)
    
    # Load audio
    try:
        audio, sr = librosa.load(input_file, sr=None)
    except Exception as e:
        print(f"Error loading audio file for silence trimming: {e}")
        return input_file
    
    # Find non-silent segments
    non_silent = librosa.effects.split(
        audio, 
        top_db=20,  # Default value, adjust if needed
        frame_length=2048,
        hop_length=512
    )
    
    if len(non_silent) > 0:
        # Get start and end of non-silent audio
        start_sample = max(0, non_silent[0][0] - int(sr * 0.5))  # Add half second buffer
        end_sample = min(len(audio), non_silent[-1][1] + int(sr * 0.5))  # Add half second buffer
        
        # Trim the audio
        trimmed_audio = audio[start_sample:end_sample]
        
        # Save trimmed audio
        try:
            sf.write(output_file, trimmed_audio, sr)
        except Exception as e:
            print(f"Error saving trimmed audio: {e}")
            return input_file
    else:
        log("No non-silent segments detected, keeping original audio", verbose)
        shutil.copy(input_file, output_file)
    
    return output_file

def convert_to_mono(input_file, output_file=None, verbose=True):
    """Convert stereo audio to mono."""
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        ext = os.path.splitext(input_file)[1]
        output_file = f"{base_name}_mono{ext}"
    
    # Check if already mono
    info = sf.info(input_file)
    if info.channels == 1:
        log("Audio is already mono, skipping conversion", verbose)
        return input_file
    
    log("Converting stereo to mono...", verbose)
    
    # Use FFmpeg for mono conversion
    command = [
        "ffmpeg", "-i", input_file,
        "-ac", "1",
        "-y",
        output_file
    ]
    
    success = run_command(command, verbose)
    return output_file if success else None

def process_audio(input_file, output_file=None, options=None, verbose=True):
    """
    Process audio with selected options.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output file (optional)
        options: Dictionary of processing options
        verbose: Whether to print progress
    
    Returns:
        Path to processed audio file
    """
    if options is None:
        options = {
            'convert_to_flac': True,
            'resample': True,
            'reduce_noise': True,
            'normalize_volume': True,
            'apply_compression': True,
            'enhance_speech': True,
            'trim_silence': True,
            'convert_to_mono': True
        }
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return None
    
    # Set default output file if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_processed.flac"
    
    # Create a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        current_file = input_file
        
        # Apply each processing step in sequence
        
        if options.get('convert_to_flac', True):
            temp_output = os.path.join(temp_dir, "audio_flac.flac")
            result = convert_to_flac(current_file, temp_output, verbose)
            if result:
                current_file = result
        
        if options.get('resample', True):
            temp_output = os.path.join(temp_dir, "audio_resampled.flac")
            result = resample_audio(current_file, temp_output, verbose=verbose)
            if result:
                current_file = result
        
        if options.get('convert_to_mono', True):
            temp_output = os.path.join(temp_dir, "audio_mono.flac")
            result = convert_to_mono(current_file, temp_output, verbose)
            if result:
                current_file = result
        
        if options.get('reduce_noise', True):
            temp_output = os.path.join(temp_dir, "audio_denoised.flac")
            result = reduce_noise(current_file, temp_output, verbose=verbose)
            if result:
                current_file = result
        
        if options.get('normalize_volume', True):
            temp_output = os.path.join(temp_dir, "audio_normalized.flac")
            result = normalize_volume(current_file, temp_output, verbose=verbose)
            if result:
                current_file = result
        
        if options.get('apply_compression', True):
            temp_output = os.path.join(temp_dir, "audio_compressed.flac")
            result = apply_compression(current_file, temp_output, verbose=verbose)
            if result:
                current_file = result
        
        if options.get('enhance_speech', True):
            temp_output = os.path.join(temp_dir, "audio_enhanced.flac")
            result = enhance_speech(current_file, temp_output, verbose=verbose)
            if result:
                current_file = result
        
        if options.get('trim_silence', True):
            temp_output = os.path.join(temp_dir, "audio_trimmed.flac")
            result = trim_silence(current_file, temp_output, verbose=verbose)
            if result:
                current_file = result
        
        # Copy final result to output file
        try:
            shutil.copy(current_file, output_file)
            log(f"Audio processing complete! Output saved to: {output_file}", verbose)
            return output_file
        except Exception as e:
            print(f"Error saving final output: {e}")
            return None

def main():
    """Parse arguments and process audio file."""
    parser = argparse.ArgumentParser(description="Process audio for optimal transcription")
    parser.add_argument("input_file", help="Path to input audio file")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--no-flac", action="store_true", help="Skip FLAC conversion")
    parser.add_argument("--no-resample", action="store_true", help="Skip resampling to 16kHz")
    parser.add_argument("--no-noise-reduction", action="store_true", help="Skip noise reduction")
    parser.add_argument("--no-normalize", action="store_true", help="Skip volume normalization")
    parser.add_argument("--no-compression", action="store_true", help="Skip dynamic range compression")
    parser.add_argument("--no-enhance", action="store_true", help="Skip speech enhancement")
    parser.add_argument("--no-trim", action="store_true", help="Skip silence trimming")
    parser.add_argument("--no-mono", action="store_true", help="Skip conversion to mono")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    
    args = parser.parse_args()
    
    # Configure processing options based on arguments
    options = {
        'convert_to_flac': not args.no_flac,
        'resample': not args.no_resample,
        'reduce_noise': not args.no_noise_reduction,
        'normalize_volume': not args.no_normalize,
        'apply_compression': not args.no_compression,
        'enhance_speech': not args.no_enhance,
        'trim_silence': not args.no_trim,
        'convert_to_mono': not args.no_mono
    }
    
    # Process the audio
    process_audio(args.input_file, args.output, options, verbose=not args.quiet)

if __name__ == "__main__":
    main()