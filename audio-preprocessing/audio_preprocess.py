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
import threading
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

def log(message, verbose=True):
    """Log a message if verbose mode is enabled."""
    if verbose:
        print(message, flush=True)  # Added flush=True for immediate output

def run_command(command, verbose=True, timeout=300):
    """Run a shell command and capture output with timeout."""
    log(f"Running: {' '.join(command)}", verbose)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"Error executing command: {' '.join(command)}")
            print(f"Error: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds: {' '.join(command)}")
        return False
    except Exception as e:
        print(f"Exception running command: {e}")
        return False

def get_audio_duration(file_path):
    """Get the duration of an audio file in seconds using FFmpeg."""
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
               "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting audio duration: {e}")
    return None

def diagnose_audio_file(file_path, verbose=True):
    """Diagnose audio file properties to help troubleshoot issues."""
    log(f"Diagnosing audio file: {file_path}", verbose)
    
    if not os.path.exists(file_path):
        print(f"ERROR: File does not exist: {file_path}")
        return False
    
    # Check file size
    try:
        file_size = os.path.getsize(file_path)
        log(f"File size: {file_size / (1024*1024):.1f} MB", verbose)
        
        if file_size == 0:
            print("ERROR: File is empty (0 bytes)")
            return False
            
        if file_size > 10 * 1024 * 1024 * 1024:  # > 10GB
            print("WARNING: File is very large (>10GB), processing may be slow")
            
    except Exception as e:
        print(f"ERROR: Cannot check file size: {e}")
        return False
    
    # Check with ffprobe
    try:
        cmd = ["ffprobe", "-v", "error", "-show_format", "-show_streams", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"ERROR: ffprobe failed: {result.stderr}")
            return False
            
        # Parse basic info
        output = result.stdout
        log("FFprobe analysis successful", verbose)
        
        # Extract key information
        for line in output.split('\n'):
            if 'duration=' in line:
                duration = line.split('=')[1]
                try:
                    dur_seconds = float(duration)
                    log(f"Duration: {dur_seconds/3600:.2f} hours ({dur_seconds:.0f} seconds)", verbose)
                except:
                    log(f"Duration: {duration}", verbose)
            elif 'codec_name=' in line:
                codec = line.split('=')[1]
                log(f"Codec: {codec}", verbose)
            elif 'sample_rate=' in line:
                sr = line.split('=')[1]
                log(f"Sample rate: {sr} Hz", verbose)
            elif 'channels=' in line:
                channels = line.split('=')[1]
                log(f"Channels: {channels}", verbose)
                
    except subprocess.TimeoutExpired:
        print("ERROR: ffprobe timed out - file may be corrupted or very large")
        return False
    except Exception as e:
        print(f"ERROR: ffprobe analysis failed: {e}")
        return False
    
    # Test a quick conversion of first 10 seconds to verify FFmpeg works
    try:
        log("Testing quick conversion of first 10 seconds...", verbose)
        test_output = tempfile.mktemp(suffix=".flac")
        
        cmd = ["ffmpeg", "-i", file_path, "-t", "10", "-c:a", "flac", "-y", 
               "-hide_banner", "-loglevel", "error", test_output]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and os.path.exists(test_output):
            test_size = os.path.getsize(test_output)
            log(f"Quick test successful - 10s sample: {test_size/1024:.1f} KB", verbose)
            os.remove(test_output)
            return True
        else:
            print(f"ERROR: Quick conversion test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"ERROR: Quick conversion test failed: {e}")
        return False

def convert_to_flac(input_file, output_file=None, verbose=True):
    """Convert audio file to FLAC format with FFmpeg - minimal working command with progress."""
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.flac"
    
    log(f"Converting {input_file} to FLAC format...", verbose)
    
    # Get duration for timeout estimation
    duration = get_audio_duration(input_file)
    if duration:
        # More realistic timeout: 5 minutes per hour of audio, minimum 15 minutes, maximum 45 minutes
        timeout_seconds = max(900, min(2700, int(duration * 300 / 3600)))
        log(f"Audio duration: {duration/60:.1f} minutes, using {timeout_seconds/60:.1f} minute timeout", verbose)
    else:
        timeout_seconds = 1200  # Default 20 minutes
        log("Could not determine audio duration, using 20 minute timeout", verbose)
    
    # Use minimal FFmpeg command that we know works
    command = [
        "ffmpeg", 
        "-nostdin",  # Prevent hanging on stdin
        "-i", input_file, 
        "-y",  # Overwrite output if exists
        output_file
    ]
    
    log(f"Running FFmpeg conversion with progress indicators...", verbose)
    if verbose:
        log(f"Command: {' '.join(command)}", verbose)
        log("Progress will show: frame=X fps=X size=X time=X speed=X", verbose)
        print("-" * 60)
    
    try:
        if verbose:
            # Show progress by letting FFmpeg output directly to terminal
            # but capture stderr for error handling
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr into stdout
                text=True,
                universal_newlines=True
            )
            
            # Read output line by line and show progress
            stderr_content = []
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    # Show progress lines (contain frame= and speed=)
                    if 'frame=' in line and 'speed=' in line:
                        # Clean up the line and show it
                        print(f"\r{line}", end='', flush=True)
                    elif 'error' in line.lower() or 'warning' in line.lower():
                        # Capture error/warning lines
                        stderr_content.append(line)
                        print(f"\n{line}")
            
            # Final newline after progress
            print()
            
            return_code = process.poll()
            
        else:
            # Silent mode - just run without progress
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                timeout=timeout_seconds
            )
            return_code = result.returncode
            stderr_content = [result.stderr] if result.stderr else []
        
        if return_code != 0:
            print(f"FFmpeg failed with return code {return_code}")
            print(f"Command: {' '.join(command)}")
            if stderr_content:
                print("Error details:")
                for line in stderr_content:
                    print(f"  {line}")
            return None
        
        # Verify output file was created and has reasonable size
        if not os.path.exists(output_file):
            print(f"Output file {output_file} was not created")
            return None
            
        output_size = os.path.getsize(output_file)
        if output_size < 1024:  # Less than 1KB is suspicious
            print(f"Output file suspiciously small: {output_size} bytes")
            return None
        
        if verbose:
            print("-" * 60)
        log(f"Conversion successful! Output size: {output_size/(1024*1024):.1f}MB", verbose)
        return output_file
        
    except subprocess.TimeoutExpired:
        print(f"\nFFmpeg conversion timed out after {timeout_seconds/60:.1f} minutes")
        print("This is unexpected with the minimal command. Possible issues:")
        print("- Very large input file")
        print("- System resource constraints") 
        print("- Corrupted input file")
        return None
        
    except Exception as e:
        print(f"Unexpected error during conversion: {e}")
        return None

def resample_audio(input_file, output_file=None, target_sr=16000, verbose=True):
    """Resample audio to target sample rate with speech optimization."""
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        ext = os.path.splitext(input_file)[1]
        output_file = f"{base_name}_resampled{ext}"
    
    log(f"Resampling to {target_sr}Hz with speech optimization...", verbose)
    
    # Check input sample rate using FFprobe (more reliable than soundfile)
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate", 
               "-of", "default=noprint_wrappers=1:nokey=1", input_file]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            current_sr = int(result.stdout.strip())
            if current_sr == target_sr:
                log(f"File already at {target_sr}Hz, but applying speech optimization anyway", verbose)
            else:
                log(f"Current sample rate: {current_sr}Hz, resampling to {target_sr}Hz", verbose)
        else:
            log(f"Could not determine sample rate, proceeding with optimization", verbose)
            
    except Exception as e:
        log(f"Error checking sample rate: {e}, proceeding with optimization", verbose)
    
    # Use FFmpeg for resampling with speech optimization
    command = [
        "ffmpeg", "-i", input_file,
        "-ar", str(target_sr),           # Sample rate
        "-ac", "1",                      # Force mono
        "-sample_fmt", "s16",            # 16-bit depth (not 24-bit)
        "-compression_level", "8",       # Maximum FLAC compression
        "-hide_banner", "-loglevel", "error",
        "-y",
        output_file
    ]
    
    log(f"Optimizing for speech recognition (16kHz, 16-bit, mono, compressed)", verbose)
    success = run_command(command, verbose, timeout=300)
    
    # Verify the output was created and has reasonable size
    if success and os.path.exists(output_file):
        try:
            input_size = os.path.getsize(input_file)
            output_size = os.path.getsize(output_file)
            compression_ratio = input_size / output_size if output_size > 0 else 0
            
            if output_size > 1024:  # At least 1KB
                log(f"Speech optimization successful: {output_size/(1024*1024):.1f}MB (was {input_size/(1024*1024):.1f}MB, {compression_ratio:.1f}x smaller)", verbose)
                return output_file
            else:
                log("Speech optimization failed: output file too small", verbose)
        except:
            log("Speech optimization failed: could not verify output", verbose)
    
    log("Speech optimization failed, returning original file", verbose)
    return input_file

def reduce_noise(input_file, output_file=None, strength=0.5, verbose=True):
    """Reduce background noise using spectral gating with memory optimization."""
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        ext = os.path.splitext(input_file)[1]
        output_file = f"{base_name}_noisereduced{ext}"
    
    log("Reducing background noise...", verbose)
    
    # Check file size first - skip noise reduction for very large files
    try:
        file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
        if file_size_mb > 1024:  # Skip for files larger than 1GB (increased from 500MB)
            log(f"File too large ({file_size_mb:.1f}MB) for noise reduction, skipping...", verbose)
            return input_file
    except Exception as e:
        print(f"Error checking file size: {e}")
        return input_file
    
    # Load audio with error handling
    try:
        # Load with a maximum duration to prevent memory issues
        duration = get_audio_duration(input_file)
        if duration and duration > 10800:  # More than 3 hours (increased from 1 hour)
            log(f"Audio file too long ({duration/3600:.1f} hours) for noise reduction, skipping...", verbose)
            return input_file
            
        audio, sr = librosa.load(input_file, sr=None, duration=min(duration or 10800, 10800))
    except Exception as e:
        print(f"Error loading audio file for noise reduction: {e}")
        return input_file
    
    try:
        # Calculate noise profile from the first 2 seconds
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
        sf.write(output_file, audio_denoised, sr)
        
    except Exception as e:
        print(f"Error during noise reduction: {e}")
        return input_file
    
    return output_file

def normalize_volume(input_file, output_file=None, target_level=-18, verbose=True):
    """Normalize audio volume to target RMS level (in dB)."""
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        ext = os.path.splitext(input_file)[1]
        output_file = f"{base_name}_normalized{ext}"
    
    log(f"Normalizing volume to {target_level}dB RMS...", verbose)
    
    # Use FFmpeg for normalization with timeout
    command = [
        "ffmpeg", "-i", input_file,
        "-filter:a", f"loudnorm=I={target_level}:LRA=11:TP=-1.5",
        "-hide_banner", "-loglevel", "error",
        "-y",
        output_file
    ]
    
    success = run_command(command, verbose, timeout=600)
    return output_file if success else input_file

def apply_compression(input_file, output_file=None, threshold=-20, ratio=3, verbose=True):
    """Apply dynamic range compression."""
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        ext = os.path.splitext(input_file)[1]
        output_file = f"{base_name}_compressed{ext}"
    
    log(f"Applying dynamic range compression (threshold: {threshold}dB, ratio: {ratio}:1)...", verbose)
    
    # Use FFmpeg for compression with timeout
    command = [
        "ffmpeg", "-i", input_file,
        "-filter:a", f"acompressor=threshold={threshold}dB:ratio={ratio}:attack=200:release=1000",
        "-hide_banner", "-loglevel", "error",
        "-y",
        output_file
    ]
    
    success = run_command(command, verbose, timeout=300)
    return output_file if success else input_file

def enhance_speech(input_file, output_file=None, verbose=True):
    """Enhance speech frequencies with EQ."""
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        ext = os.path.splitext(input_file)[1]
        output_file = f"{base_name}_enhanced{ext}"
    
    log("Enhancing speech frequencies...", verbose)
    
    # Use FFmpeg with multi-band EQ to enhance speech clarity
    command = [
        "ffmpeg", "-i", input_file,
        "-filter:a", "equalizer=f=100:width_type=h:width=100:g=-3,equalizer=f=1000:width_type=q:width=1:g=2,equalizer=f=2500:width_type=q:width=1:g=3,equalizer=f=7000:width_type=h:width=1000:g=-2",
        "-hide_banner", "-loglevel", "error",
        "-y",
        output_file
    ]
    
    success = run_command(command, verbose, timeout=300)
    return output_file if success else input_file

def trim_silence(input_file, output_file=None, threshold=0.02, min_silence_duration=1.0, verbose=True):
    """Trim leading and trailing silence."""
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        ext = os.path.splitext(input_file)[1]
        output_file = f"{base_name}_trimmed{ext}"
    
    log("Trimming leading and trailing silence...", verbose)
    
    # Check file size first
    try:
        file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
        if file_size_mb > 1536:  # Skip for files larger than 1.5GB (increased from 500MB)
            log(f"File too large ({file_size_mb:.1f}MB) for silence trimming, skipping...", verbose)
            return input_file
    except Exception as e:
        print(f"Error checking file size: {e}")
        return input_file
    
    # Load audio with error handling
    try:
        duration = get_audio_duration(input_file)
        if duration and duration > 10800:  # More than 3 hours (increased from 1 hour)
            log(f"Audio file too long ({duration/3600:.1f} hours) for silence trimming, skipping...", verbose)
            return input_file
            
        audio, sr = librosa.load(input_file, sr=None, duration=min(duration or 5400, 5400))
    except Exception as e:
        print(f"Error loading audio file for silence trimming: {e}")
        return input_file
    
    try:
        # Find non-silent segments
        non_silent = librosa.effects.split(
            audio, 
            top_db=20,
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
            sf.write(output_file, trimmed_audio, sr)
        else:
            log("No non-silent segments detected, keeping original audio", verbose)
            shutil.copy(input_file, output_file)
            
    except Exception as e:
        print(f"Error during silence trimming: {e}")
        return input_file
    
    return output_file

def convert_to_mono(input_file, output_file=None, verbose=True):
    """Convert stereo audio to mono."""
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        ext = os.path.splitext(input_file)[1]
        output_file = f"{base_name}_mono{ext}"
    
    # Check if already mono
    try:
        info = sf.info(input_file)
        if info.channels == 1:
            log("Audio is already mono, skipping conversion", verbose)
            return input_file
    except Exception as e:
        print(f"Error checking audio channels: {e}")
        return input_file
    
    log("Converting stereo to mono...", verbose)
    
    # Use FFmpeg for mono conversion with timeout
    command = [
        "ffmpeg", "-i", input_file,
        "-ac", "1",
        "-hide_banner", "-loglevel", "error",
        "-y",
        output_file
    ]
    
    success = run_command(command, verbose, timeout=300)
    return output_file if success else input_file

def process_audio(input_file, output_file=None, options=None, verbose=True, diagnose_only=False):
    """
    Process audio with selected options.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output file (optional)
        options: Dictionary of processing options
        verbose: Whether to print progress
        diagnose_only: If True, only run diagnostics and exit
    
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
    
    # Always run diagnostics first
    log("=" * 60, verbose)
    log("AUDIO FILE DIAGNOSIS", verbose)
    log("=" * 60, verbose)
    
    diagnosis_success = diagnose_audio_file(input_file, verbose)
    
    if not diagnosis_success:
        print("ERROR: Audio file diagnosis failed. Cannot proceed with processing.")
        return None
    
    if diagnose_only:
        log("Diagnosis complete. Exiting (diagnose-only mode).", verbose)
        return input_file
    
    log("=" * 60, verbose)
    log("STARTING AUDIO PROCESSING", verbose)
    log("=" * 60, verbose)
    
    # Set default output file if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_processed.flac"
    
    # Check available disk space
    try:
        input_size = os.path.getsize(input_file)
        free_space = shutil.disk_usage(os.path.dirname(output_file))[2]
        if free_space < input_size * 5:  # Need at least 5x the input file size
            print(f"Warning: Low disk space. Available: {free_space/(1024**3):.1f}GB, recommended: {input_size*5/(1024**3):.1f}GB")
    except Exception as e:
        print(f"Warning: Could not check disk space: {e}")
    
    # Create a temporary directory for intermediate files
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            current_file = input_file
            step_count = sum(options.values())
            current_step = 0
            
            # Apply each processing step in sequence
            
            if options.get('convert_to_flac', True):
                current_step += 1
                if verbose:
                    print(f"Step {current_step}/{step_count}: Converting to FLAC")
                temp_output = os.path.join(temp_dir, "audio_flac.flac")
                result = convert_to_flac(current_file, temp_output, verbose)
                if result and os.path.exists(result):
                    current_file = result
                else:
                    print("FLAC conversion failed, continuing with original file")
            
            if options.get('resample', True):
                current_step += 1
                if verbose:
                    print(f"Step {current_step}/{step_count}: Resampling")
                temp_output = os.path.join(temp_dir, "audio_resampled.flac")
                result = resample_audio(current_file, temp_output, verbose=verbose)
                if result and os.path.exists(result):
                    current_file = result
            
            if options.get('convert_to_mono', True):
                current_step += 1
                if verbose:
                    print(f"Step {current_step}/{step_count}: Converting to mono")
                temp_output = os.path.join(temp_dir, "audio_mono.flac")
                result = convert_to_mono(current_file, temp_output, verbose)
                if result and os.path.exists(result):
                    current_file = result
            
            if options.get('reduce_noise', True):
                current_step += 1
                if verbose:
                    print(f"Step {current_step}/{step_count}: Reducing noise")
                temp_output = os.path.join(temp_dir, "audio_denoised.flac")
                result = reduce_noise(current_file, temp_output, verbose=verbose)
                if result and os.path.exists(result):
                    current_file = result
            
            if options.get('normalize_volume', True):
                current_step += 1
                if verbose:
                    print(f"Step {current_step}/{step_count}: Normalizing volume")
                temp_output = os.path.join(temp_dir, "audio_normalized.flac")
                result = normalize_volume(current_file, temp_output, verbose=verbose)
                if result and os.path.exists(result):
                    current_file = result
            
            if options.get('apply_compression', True):
                current_step += 1
                if verbose:
                    print(f"Step {current_step}/{step_count}: Applying compression")
                temp_output = os.path.join(temp_dir, "audio_compressed.flac")
                result = apply_compression(current_file, temp_output, verbose=verbose)
                if result and os.path.exists(result):
                    current_file = result
            
            if options.get('enhance_speech', True):
                current_step += 1
                if verbose:
                    print(f"Step {current_step}/{step_count}: Enhancing speech")
                temp_output = os.path.join(temp_dir, "audio_enhanced.flac")
                result = enhance_speech(current_file, temp_output, verbose=verbose)
                if result and os.path.exists(result):
                    current_file = result
            
            if options.get('trim_silence', True):
                current_step += 1
                if verbose:
                    print(f"Step {current_step}/{step_count}: Trimming silence")
                temp_output = os.path.join(temp_dir, "audio_trimmed.flac")
                result = trim_silence(current_file, temp_output, verbose=verbose)
                if result and os.path.exists(result):
                    current_file = result
            
            # Copy final result to output file
            try:
                shutil.copy(current_file, output_file)
                log(f"Audio processing complete! Output saved to: {output_file}", verbose)
                return output_file
            except Exception as e:
                print(f"Error saving final output: {e}")
                return None
                
    except Exception as e:
        print(f"Error during processing: {e}")
        return None

def main():
    """Parse arguments and process audio file."""
    parser = argparse.ArgumentParser(description="Process audio for optimal transcription")
    parser.add_argument("input_file", help="Path to input audio file")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--diagnose", action="store_true", help="Only run diagnostics, don't process")
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
    result = process_audio(args.input_file, args.output, options, 
                          verbose=not args.quiet, diagnose_only=args.diagnose)
    if result is None:
        sys.exit(1)

if __name__ == "__main__":
    main()