#!/usr/bin/env python3
"""
GPU-Accelerated Transcription Pipeline (Fixed for pyannote issues)
==================================================================

A containerized transcription pipeline using faster-whisper and pyannote.audio
optimized for 45-90 minute technical interviews with 6GB GPU memory constraints.
"""

import os
import sys
import argparse
import time
import re
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import psutil
from faster_whisper import WhisperModel
import soundfile as sf

def check_gpu_memory():
    """Check available GPU memory."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_used = torch.cuda.memory_allocated(0) / (1024**3)
        gpu_free = gpu_memory - gpu_used
        print(f"GPU Memory: {gpu_free:.1f}GB free / {gpu_memory:.1f}GB total")
        return gpu_free
    else:
        print("CUDA not available")
        return 0

def load_prompt(prompt_file: str) -> str:
    """Load the whisper prompt from file."""
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        print(f"Loaded prompt: {len(prompt)} characters")
        return prompt
    except FileNotFoundError:
        print(f"Warning: Prompt file {prompt_file} not found, using empty prompt")
        return ""

def load_term_corrections(corrections_file: str) -> Dict[str, str]:
    """Load term corrections from file."""
    corrections = {}
    try:
        with open(corrections_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '|' in line:
                        incorrect, correct = line.split('|', 1)
                        corrections[incorrect.strip()] = correct.strip()
        
        print(f"Loaded {len(corrections)} term corrections")
        return corrections
    except FileNotFoundError:
        print(f"Warning: Corrections file {corrections_file} not found")
        return {}

def apply_term_corrections(text: str, corrections: Dict[str, str]) -> str:
    """Apply term corrections to text."""
    if not corrections:
        return text
    
    corrected_text = text
    corrections_applied = 0
    
    for incorrect, correct in corrections.items():
        # Use word boundaries to avoid partial matches
        pattern = re.compile(r'\b' + re.escape(incorrect) + r'\b', re.IGNORECASE)
        if pattern.search(corrected_text):
            corrected_text = pattern.sub(correct, corrected_text)
            corrections_applied += 1
    
    if corrections_applied > 0:
        print(f"Applied {corrections_applied} term corrections")
    
    return corrected_text

def run_diarisation(audio_file: str, num_speakers: Optional[int] = None) -> List[Dict]:
    """Run speaker diarisation using pyannote.audio."""
    print("=" * 60)
    print("RUNNING SPEAKER DIARISATION")
    print("=" * 60)
    
    # Import pyannote only when needed
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        print(f"Error importing pyannote.audio: {e}")
        print("Diarisation not available. Continue with --no-diarization flag.")
        return []
    except Exception as e:
        print(f"Unexpected error importing pyannote.audio: {e}")
        print("This might be due to package compatibility issues.")
        print("Continue with --no-diarization flag.")
        return []
    
    # Check for Hugging Face token
    hf_token = os.environ.get('HUGGINGFACE_TOKEN')
    if not hf_token:
        print("Warning: HUGGINGFACE_TOKEN not set. Diarisation may fail.")
        print("Set token with: export HUGGINGFACE_TOKEN=your_token_here")
    
    try:
        # Initialize diarisation pipeline
        print("Loading pyannote.audio diarisation model...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        # Run diarisation
        print(f"Processing: {audio_file}")
        diarisation = pipeline(audio_file, num_speakers=num_speakers)
        
        # Convert to list of segments
        segments = []
        for turn, _, speaker in diarisation.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        
        print(f"Found {len(set(seg['speaker'] for seg in segments))} speakers in {len(segments)} segments")
        
        # Clear GPU memory
        del pipeline
        torch.cuda.empty_cache()
        gc.collect()
        
        return segments
        
    except Exception as e:
        print(f"Diarisation failed: {e}")
        print("Continuing without speaker diarisation...")
        return []

def run_transcription(audio_file: str, model_name: str, prompt: str) -> List[Dict]:
    """Run transcription using faster-whisper."""
    print("=" * 60)
    print("RUNNING TRANSCRIPTION")
    print("=" * 60)
    
    # Map distil-whisper models to their full Hugging Face paths
    model_mapping = {
        "distil-large-v3": "distil-whisper/distil-large-v3",
        "distil-large-v2": "distil-whisper/distil-large-v2",
    }
    
    # Use mapped name if available, otherwise use the original name
    actual_model_name = model_mapping.get(model_name, model_name)
    
    print(f"Loading {model_name} model (using: {actual_model_name})...")
    
    # Initialize model with GPU acceleration
    model = WhisperModel(
        actual_model_name,
        device="cuda",
        compute_type="float16",  # Optimize for memory
        download_root="./models/"  # Cache models in container
    )
    
    print(f"Model loaded. Starting transcription...")
    print(f"Using prompt: {len(prompt)} characters")
    
    # Run transcription - faster-whisper handles long-form automatically
    segments, info = model.transcribe(
        audio_file,
        language="en",
        initial_prompt=prompt,
        condition_on_previous_text=False,  # Avoid hallucination
        temperature=0.0,  # Deterministic output
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        vad_filter=True,  # Use voice activity detection
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=5,  # Better quality
        best_of=5,   # Better quality
        patience=1.0
    )
    
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    
    # Convert generator to list
    transcription_segments = []
    for segment in segments:
        transcription_segments.append({
            'start': segment.start,
            'end': segment.end,
            'text': segment.text.strip(),
            'speaker': 'UNKNOWN'  # Will be updated by diarisation
        })
    
    print(f"Transcribed {len(transcription_segments)} segments")
    
    # Clear GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return transcription_segments

def align_speakers_and_transcription(diarisation_segments: List[Dict], 
                                   transcription_segments: List[Dict]) -> List[Dict]:
    """Align speaker diarisation with transcription segments."""
    if not diarisation_segments:
        print("No diarisation data available, using generic speaker labels")
        return transcription_segments
    
    print("Aligning speakers with transcription...")
    
    aligned_segments = []
    
    for trans_seg in transcription_segments:
        trans_start = trans_seg['start']
        trans_end = trans_seg['end']
        trans_mid = (trans_start + trans_end) / 2
        
        # Find overlapping speaker segments
        best_speaker = 'UNKNOWN'
        best_overlap = 0
        
        for dia_seg in diarisation_segments:
            dia_start = dia_seg['start']
            dia_end = dia_seg['end']
            
            # Calculate overlap
            overlap_start = max(trans_start, dia_start)
            overlap_end = min(trans_end, dia_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if overlap_duration > best_overlap:
                best_overlap = overlap_duration
                best_speaker = dia_seg['speaker']
        
        # If no overlap found, use the speaker segment containing the midpoint
        if best_speaker == 'UNKNOWN':
            for dia_seg in diarisation_segments:
                if dia_seg['start'] <= trans_mid <= dia_seg['end']:
                    best_speaker = dia_seg['speaker']
                    break
        
        aligned_segments.append({
            'start': trans_seg['start'],
            'end': trans_seg['end'],
            'text': trans_seg['text'],
            'speaker': best_speaker
        })
    
    # Count speakers
    speakers = set(seg['speaker'] for seg in aligned_segments)
    print(f"Aligned transcription with {len(speakers)} speakers")
    
    return aligned_segments

def format_transcript(segments: List[Dict], corrections: Dict[str, str], model_name: str = "distil-large-v3") -> str:
    """Format the final transcript with timestamps and corrections."""
    print("Formatting final transcript...")
    
    lines = []
    lines.append("# Audio Transcription")
    lines.append("# Generated using GPU-accelerated pipeline")
    lines.append(f"# Model: {model_name}")
    lines.append("")
    
    for segment in segments:
        # Apply term corrections
        corrected_text = apply_term_corrections(segment['text'], corrections)
        
        # Format timestamp
        start_time = segment['start']
        minutes = int(start_time // 60)
        seconds = int(start_time % 60)
        timestamp = f"[{minutes:02d}:{seconds:02d}]"
        
        # Format line
        speaker = segment['speaker']
        line = f"{timestamp} {speaker}: {corrected_text}"
        lines.append(line)
        lines.append("")  # Empty line between segments
    
    return "\n".join(lines)

def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="GPU-Accelerated Transcription Pipeline")
    parser.add_argument("input_file", nargs='?', help="Path to input audio file")
    parser.add_argument("-o", "--output", help="Output transcript file (default: input_name.txt)")
    parser.add_argument("--model", default="medium.en", 
                      choices=["distil-large-v3", "large-v3", "large", "medium.en", "medium", "small.en", "small", "base", "tiny"],
                      help="Whisper model to use (default: medium.en)")
    parser.add_argument("--speakers", type=int, help="Number of speakers (auto-detect if not specified)")
    parser.add_argument("--no-diarization", action="store_true", help="Skip speaker diarisation")
    parser.add_argument("--gpu-check", action="store_true", help="Check GPU status and exit")
    
    args = parser.parse_args()
    
    # GPU check mode
    if args.gpu_check:
        print("GPU Status:")
        check_gpu_memory()
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        return
    
    # Validate input file (only required when not in gpu-check mode)
    if not args.input_file:
        print("Error: Input file is required (unless using --gpu-check)")
        parser.print_help()
        sys.exit(1)
        
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Set output file
    if args.output:
        output_file = args.output
    else:
        input_path = Path(args.input_file)
        output_file = input_path.with_suffix('.txt')
    
    # Load configuration
    prompt = load_prompt('config/whisper_prompt.txt')
    corrections = load_term_corrections('config/term_corrections.txt')
    
    print("=" * 60)
    print("GPU-ACCELERATED TRANSCRIPTION PIPELINE")
    print("=" * 60)
    print(f"Input: {args.input_file}")
    print(f"Output: {output_file}")
    print(f"Model: {args.model}")
    print(f"Speakers: {'auto-detect' if not args.speakers else args.speakers}")
    print(f"Diarization: {'disabled' if args.no_diarization else 'enabled'}")
    print()
    
    # Check GPU status
    gpu_memory = check_gpu_memory()
    if gpu_memory < 4:
        print("Warning: Less than 4GB GPU memory available. Performance may be impacted.")
    
    start_time = time.time()
    
    try:
        # Step 1: Speaker diarisation (if enabled)
        diarisation_segments = []
        if not args.no_diarization:
            diarisation_segments = run_diarisation(args.input_file, args.speakers)
        
        # Step 2: Transcription
        transcription_segments = run_transcription(args.input_file, args.model, prompt)
        
        # Step 3: Align speakers with transcription
        aligned_segments = align_speakers_and_transcription(diarisation_segments, transcription_segments)
        
        # Step 4: Format final transcript
        transcript = format_transcript(aligned_segments, corrections, args.model)
        
        # Step 5: Save output
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(transcript)
        
        # Report results
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("=" * 60)
        print("TRANSCRIPTION COMPLETE")
        print("=" * 60)
        print(f"Processing time: {processing_time/60:.1f} minutes")
        print(f"Output saved to: {output_file}")
        print(f"Transcript length: {len(transcript)} characters")
        print(f"Number of segments: {len(aligned_segments)}")
        
        # Final GPU memory check
        check_gpu_memory()
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()