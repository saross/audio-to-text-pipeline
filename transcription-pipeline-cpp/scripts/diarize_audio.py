#!/usr/bin/env python3
"""
Standalone Speaker Diarization Script

PURPOSE:
    Performs speaker diarization on audio files to identify who spoke when.
    Designed to work with the existing whisper.cpp transcription pipeline.
    
USAGE:
    python3 diarize_audio.py <audio_file> [options]
    
    Options:
        -o, --output FILE       Output file (default: input_name_diarization.json)
        -n, --num-speakers INT  Number of speakers (default: auto-detect)
        --min-speakers INT      Minimum speakers for auto-detection (default: 1)
        --max-speakers INT      Maximum speakers for auto-detection (default: 10)
        --merge-threshold FLOAT Speaker merging threshold (default: 0.5)
        --format FORMAT         Output format: json, rttm, or simple (default: json)
        --device DEVICE         Computing device: cuda, cpu, or auto (default: auto)
        --show-plot             Display visualization of diarization

REQUIREMENTS:
    - pyannote.audio
    - torch
    - torchaudio
    - scipy
    - A Hugging Face access token (for pyannote models)

OUTPUTS:
    JSON format with speaker segments:
    {
        "speakers": ["SPEAKER_0", "SPEAKER_1"],
        "segments": [
            {"speaker": "SPEAKER_0", "start": 0.5, "end": 10.2},
            {"speaker": "SPEAKER_1", "start": 10.2, "end": 15.7},
            ...
        ]
    }

AUTHOR: Audio transcription diarization module
DATE: 2024
"""

import os
import sys
import json
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

try:
    import torch
    import torchaudio
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook
except ImportError as e:
    print("Error: Missing required dependencies")
    print("Please install: pip install pyannote.audio torch torchaudio")
    print(f"Details: {e}")
    sys.exit(1)


class SimpleDiarizer:
    """Simple wrapper around pyannote.audio for speaker diarization"""
    
    def __init__(self, device='auto', hf_token=None):
        """
        Initialize diarizer with device and authentication token
        
        Args:
            device: 'cuda', 'cpu', or 'auto'
            hf_token: Hugging Face access token (required for pyannote models)
        """
        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Additional GPU debugging
        if self.device.type == 'cuda':
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Initialize pipeline
        try:
            # Use the pretrained speaker diarization pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            
            # Move pipeline to device
            self.pipeline.to(self.device)
            
        except Exception as e:
            print(f"Error loading diarization model: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure you have accepted the model conditions at:")
            print("   https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("2. Set your Hugging Face token:")
            print("   export HF_TOKEN=your_token_here")
            print("3. Or pass it with: --hf-token your_token_here")
            sys.exit(1)
    
    def diarize(self, audio_file, num_speakers=None, min_speakers=1, max_speakers=10):
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_file: Path to audio file
            num_speakers: Exact number of speakers (None for auto-detect)
            min_speakers: Minimum speakers for auto-detection
            max_speakers: Maximum speakers for auto-detection
            
        Returns:
            Diarization object with speaker segments
        """
        print(f"Processing: {audio_file}")
        
        # Get audio duration for progress tracking
        try:
            info = torchaudio.info(audio_file)
            duration_seconds = info.num_frames / info.sample_rate
            duration_str = f"{int(duration_seconds//60)}:{int(duration_seconds%60):02d}"
            print(f"Audio duration: {duration_str} ({duration_seconds:.1f} seconds)")
        except:
            duration_seconds = None
            print("Audio duration: Unknown")
        
        # Progress hook for tracking
        with ProgressHook() as hook:
            # Set pipeline parameters
            if num_speakers is not None:
                # Known number of speakers
                print(f"Using fixed number of speakers: {num_speakers}")
                print("\nDiarization progress:")
                sys.stdout.flush()
                diarization = self.pipeline(audio_file, num_speakers=num_speakers, hook=hook)
            else:
                # Auto-detect number of speakers
                print(f"Auto-detecting speakers (min: {min_speakers}, max: {max_speakers})")
                print("\nDiarization progress:")
                sys.stdout.flush()
                diarization = self.pipeline(
                    audio_file, 
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    hook=hook
                )
        
        print()  # New line after progress
        return diarization
    
    def format_output(self, diarization, format='json'):
        """
        Format diarization output
        
        Args:
            diarization: Pyannote diarization object
            format: Output format (json, rttm, simple)
            
        Returns:
            Formatted output string
        """
        if format == 'json':
            # Extract unique speakers and build segments list
            speakers = set()
            segments = []
            
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                speakers.add(speaker)
                segments.append({
                    "speaker": speaker,
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "duration": round(segment.duration, 2)
                })
            
            # Sort by start time
            segments.sort(key=lambda x: x['start'])
            
            return json.dumps({
                "speakers": sorted(list(speakers)),
                "num_speakers": len(speakers),
                "total_segments": len(segments),
                "segments": segments
            }, indent=2)
            
        elif format == 'rttm':
            # RTTM format (standard for diarization evaluation)
            lines = []
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                lines.append(
                    f"SPEAKER {Path(audio_file).stem} 1 "
                    f"{segment.start:.3f} {segment.duration:.3f} "
                    f"<NA> <NA> {speaker} <NA> <NA>"
                )
            return '\n'.join(lines)
            
        elif format == 'simple':
            # Simple human-readable format
            lines = []
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                start_min = int(segment.start // 60)
                start_sec = int(segment.start % 60)
                end_min = int(segment.end // 60)
                end_sec = int(segment.end % 60)
                lines.append(
                    f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}] {speaker}"
                )
            return '\n'.join(lines)
        
        else:
            raise ValueError(f"Unknown format: {format}")


def visualize_diarization(diarization, audio_file):
    """Display a simple visualization of speaker segments"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(15, 4))
        
        # Get unique speakers and assign colors
        speakers = sorted(set(segment[2] for segment in diarization.itertracks()))
        colors = plt.cm.Set3(range(len(speakers)))
        speaker_colors = dict(zip(speakers, colors))
        
        # Plot segments
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            rect = patches.Rectangle(
                (segment.start, 0), segment.duration, 1,
                linewidth=1, edgecolor='black',
                facecolor=speaker_colors[speaker],
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add speaker label if segment is long enough
            if segment.duration > 5:
                ax.text(
                    segment.start + segment.duration/2, 0.5,
                    speaker, ha='center', va='center'
                )
        
        # Set labels and title
        ax.set_xlim(0, max(segment.end for segment, _ in diarization.itertracks()))
        ax.set_ylim(0, 1)
        ax.set_xlabel('Time (seconds)')
        ax.set_yticks([])
        ax.set_title(f'Speaker Diarization: {Path(audio_file).name}')
        
        # Add legend
        legend_elements = [
            patches.Patch(facecolor=color, label=speaker, alpha=0.8)
            for speaker, color in speaker_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error creating visualization: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Perform speaker diarization on audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-n', '--num-speakers', type=int, 
                       help='Number of speakers (default: auto-detect)')
    parser.add_argument('--min-speakers', type=int, default=1,
                       help='Minimum speakers for auto-detection')
    parser.add_argument('--max-speakers', type=int, default=10,
                       help='Maximum speakers for auto-detection')
    parser.add_argument('--format', choices=['json', 'rttm', 'simple'], 
                       default='json', help='Output format')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'],
                       default='auto', help='Computing device')
    parser.add_argument('--hf-token', help='Hugging Face access token')
    parser.add_argument('--show-plot', action='store_true',
                       help='Display visualization')
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # Get HF token from environment if not provided
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')
    if not hf_token:
        print("Error: Hugging Face token required")
        print("Set environment variable: export HF_TOKEN=your_token_here")
        print("Or use: --hf-token your_token_here")
        print("\nGet your token from: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    # Set output file if not specified
    if not args.output:
        base_name = Path(args.audio_file).stem
        extension = 'json' if args.format == 'json' else 'txt'
        args.output = f"{base_name}_diarization.{extension}"
    
    # Initialize diarizer
    diarizer = SimpleDiarizer(device=args.device, hf_token=hf_token)
    
    # Perform diarization
    try:
        diarization = diarizer.diarize(
            args.audio_file,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers
        )
        
        # Format output
        output = diarizer.format_output(diarization, format=args.format)
        
        # Save to file
        with open(args.output, 'w') as f:
            f.write(output)
        
        print(f"\nDiarization complete!")
        print(f"Output saved to: {args.output}")
        
        # Print summary
        speakers = sorted(set(speaker for _, _, speaker in diarization.itertracks(yield_label=True)))
        print(f"Detected {len(speakers)} speakers: {', '.join(speakers)}")
        
        # Show visualization if requested
        if args.show_plot:
            visualize_diarization(diarization, args.audio_file)
            
    except Exception as e:
        print(f"Error during diarization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()