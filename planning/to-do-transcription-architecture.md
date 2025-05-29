# Transcription Pipeline Modular Architecture Plan

## Overview

This document outlines the planned architecture for refactoring the transcription pipeline into a modular Python structure while preserving the excellent shell wrapper integration. The architecture supports the planned migration to WhisperX and implementation of additional improvements.

## Directory Structure

```
transcription_pipeline/
├── __init__.py
├── config.py                  # Configuration handling
├── models/                    # Model implementations
│   ├── __init__.py
│   ├── whisper_model.py       # Current implementation
│   ├── whisperx_model.py      # Future implementation
│   └── diarisation.py         # Speaker diarisation
├── processors/                # Audio/text processing
│   ├── __init__.py
│   ├── preprocess.py          # Audio preprocessing logic
│   ├── postprocess.py         # Term corrections, cleanup
│   └── segment_merger.py      # Merging, filtering segments
├── utils/                     # Helper functions
│   ├── __init__.py
│   ├── audio_utils.py         # Audio file handling
│   └── device_utils.py        # GPU/CPU management
├── formats/                   # Output formats
│   ├── __init__.py
│   ├── txt_format.py          # Standard transcript format
│   ├── srt_format.py          # Subtitle format
│   └── json_format.py         # JSON format
├── cli/                       # Command-line interfaces
│   ├── __init__.py
│   └── main.py                # Main Python entry point
└── scripts/                   # Shell wrappers (keep & refine existing)
    ├── setup.sh               # Environment setup
    ├── preprocess.sh          # Audio preprocessing wrapper
    ├── transcribe.sh          # Main transcription wrapper
    └── env_vars.sh            # Environment variables
```

## Component Responsibilities

### 1. Configuration Module (`config.py`)
- Parses command-line arguments
- Loads configuration from files
- Defines default values
- Validates settings
- Maintains configuration types for different pipeline stages

```python
# Example config.py structure
class TranscriptionConfig:
    def __init__(self, args=None, config_file=None):
        # Set defaults
        self.model_type = "whisper"  # or "whisperx"
        self.model_size = "medium.en"
        self.device = "auto"  # auto, cuda, cpu
        self.language = "en"
        # Etc.
        
        # Load from file if provided
        if config_file:
            self._load_from_file(config_file)
            
        # Override with command line args
        if args:
            self._apply_args(args)
    
    def _load_from_file(self, config_file):
        # Load YAML/JSON config
        pass
        
    def _apply_args(self, args):
        # Override config with CLI args
        pass
```

### 2. Model Modules

#### `whisper_model.py`
- Current Whisper implementation
- Device management (GPU/CPU)
- Implements common model interface

```python
def transcribe(audio_file, config):
    """
    Transcribe audio using Whisper.
    
    Args:
        audio_file: Path to audio file
        config: TranscriptionConfig object
        
    Returns:
        Dictionary with transcription segments
    """
    # Initialize model on appropriate device
    device = get_device(config)
    model = whisper.load_model(config.model_size, device=device)
    
    # Rest of current implementation...
    return transcription
```

#### `whisperx_model.py`
- WhisperX implementation with same interface
- Potentially handles diarisation directly

```python
def transcribe(audio_file, config):
    """
    Transcribe audio using WhisperX.
    
    Args:
        audio_file: Path to audio file
        config: TranscriptionConfig object
        
    Returns:
        Dictionary with transcription segments
    """
    # Initialize WhisperX
    device = get_device(config)
    model = whisperx.load_model(config.model_size, device)
    
    # Transcribe with WhisperX
    result = model.transcribe(audio_file, batch_size=16)
    
    # Add diarisation if configured
    if config.diarize:
        diarize_model = whisperx.load_diarization_model(device)
        diarization = diarize_model({"audio": audio_file})
        result = whisperx.assign_speakers(result["segments"], diarization)
    
    return result
```

#### `diarisation.py`
- Speaker diarisation using pyannote.audio
- Handles token authentication
- CPU optimization

```python
def diarise(audio_file, config):
    """
    Perform speaker diarisation.
    
    Args:
        audio_file: Path to audio file
        config: DiarisationConfig object
        
    Returns:
        Speaker turns with timestamps
    """
    # Get token from config or environment
    token = config.huggingface_token or os.environ.get('HUGGINGFACE_TOKEN')
    
    # Initialize pipeline
    diarisation = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token
    )
    
    # Run diarisation
    return diarisation(audio_file, num_speakers=config.num_speakers)
```

### 3. Processing Modules

#### `preprocess.py`
- Existing audio preprocessing (mostly preserved)
- Noise reduction, normalisation, etc.

#### `postprocess.py`
- Term corrections
- Text formatting
- Speaker name mapping

```python
def apply_term_corrections(text, corrections):
    """Apply the term corrections to the text."""
    if not corrections:
        return text
    
    corrected_text = text
    for incorrect, correct in corrections.items():
        pattern = re.compile(r'\b' + re.escape(incorrect) + r'\b', re.IGNORECASE)
        corrected_text = pattern.sub(correct, corrected_text)
    
    return corrected_text

def map_speaker_names(segments, name_map=None):
    """Map SPEAKER_XX to actual names if provided."""
    if not name_map:
        return segments
        
    for segment in segments:
        if segment["speaker"] in name_map:
            segment["speaker"] = name_map[segment["speaker"]]
            
    return segments
```

#### `segment_merger.py`
- Combines transcription and diarisation
- Merges consecutive segments
- Filters empty segments
- Implements speech rate detection

```python
def merge_segments(segments, max_gap=1.0):
    """Merge consecutive segments from the same speaker."""
    if not segments:
        return []
        
    merged = []
    current = segments[0].copy()
    
    for segment in segments[1:]:
        if (segment["speaker"] == current["speaker"] and 
                segment["start"] - current["end"] < max_gap):
            current["end"] = segment["end"]
            current["text"] += " " + segment["text"]
        else:
            merged.append(current)
            current = segment.copy()
            
    merged.append(current)
    return merged

def filter_empty_segments(segments):
    """Remove empty UNKNOWN segments."""
    return [segment for segment in segments 
            if not (segment["speaker"] == "UNKNOWN" and not segment["text"].strip())]
```

### 4. Output Format Modules

#### Base format interface
```python
class OutputFormat:
    def write(self, segments, output_file):
        """Write segments to the output file."""
        raise NotImplementedError()
```

#### `txt_format.py`
- Standard transcript format with timestamps

```python
class TxtFormat(OutputFormat):
    def write(self, segments, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"# Transcript\n\n")
            
            # Write each segment
            for segment in segments:
                # Format timestamp as [HH:MM:SS]
                m, s = divmod(segment["start"], 60)
                h, m = divmod(m, 60)
                timestamp = f"[{int(h):02d}:{int(m):02d}:{int(s):02d}]"
                
                # Write the line
                f.write(f"{timestamp} {segment['speaker']}: {segment['text']}\n\n")
```

#### `srt_format.py`
- Subtitle format for videos

#### `json_format.py`
- Structured JSON format for programmatic use

### 5. CLI Module

```python
# cli/main.py
def parse_arguments():
    parser = argparse.ArgumentParser(description="Transcribe audio with speaker diarisation")
    parser.add_argument("input_file", help="Path to audio file")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--speakers", "-s", type=int, help="Number of speakers")
    parser.add_argument("--model", choices=["whisper", "whisperx"], default="whisper")
    parser.add_argument("--format", choices=["txt", "srt", "json"], default="txt")
    # Add more arguments
    return parser.parse_args()

def main():
    args = parse_arguments()
    config = config.TranscriptionConfig(args)
    
    # Preprocessing if needed
    if args.preprocess:
        from transcription_pipeline.processors import preprocess
        audio_file = preprocess.process_audio(args.input_file, config.preprocess)
    else:
        audio_file = args.input_file
    
    # Select model based on config
    if config.model_type == "whisperx":
        from transcription_pipeline.models import whisperx_model as model
    else:
        from transcription_pipeline.models import whisper_model as model
    
    # Run transcription
    transcription = model.transcribe(audio_file, config)
    
    # Handle diarisation if needed (for Whisper model)
    if config.model_type == "whisper" and config.diarize:
        from transcription_pipeline.models import diarisation
        diarisation_result = diarisation.diarise(audio_file, config)
        
        # Combine results
        from transcription_pipeline.processors import segment_merger
        segments = segment_merger.combine(transcription, diarisation_result)
        segments = segment_merger.merge_segments(segments)
        segments = segment_merger.filter_empty_segments(segments)
    else:
        # For WhisperX with built-in diarisation, just use segments directly
        segments = transcription["segments"]
    
    # Apply post-processing
    from transcription_pipeline.processors import postprocess
    segments = postprocess.process(segments, config)
    
    # Format and save output
    from transcription_pipeline.formats import get_format
    formatter = get_format(config.format)
    formatter.write(segments, config.output_file)
```

### 6. Shell Wrappers

The existing shell wrappers would be simplified to focus on:
- Environment setup
- Environment variables
- Basic argument forwarding to Python CLI
- Timing and feedback

Example simplified `transcribe.sh`:

```bash
#!/bin/bash

# ====================================================================
# AUDIO TRANSCRIPTION PIPELINE
# ====================================================================

# Activate environment
source ~/transcription-env/bin/activate

# Load environment variables
source "$(dirname "$0")/env_vars.sh"

# Record start time
START_TIME=$(date +%s)

# Run Python transcription module with all arguments
python -m transcription_pipeline.cli.main "$@"
EXIT_CODE=$?

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

if [ $EXIT_CODE -eq 0 ]; then
    echo "======================================================================"
    echo "Transcription completed successfully!"
    echo "Processing time: ${MINUTES}m ${SECONDS}s"
    echo "======================================================================"
else
    echo "======================================================================"
    echo "ERROR: Transcription failed."
    echo "======================================================================"
fi

exit $EXIT_CODE
```

## Implementation Strategy

### Phase 1: Refactor Current Pipeline
1. Create the directory structure
2. Move relevant code into the modules
3. Implement configuration handling
4. Adapt existing shell scripts to call new Python CLI

### Phase 2: Add WhisperX Support
1. Implement WhisperX model module
2. Add configuration options to switch between models
3. Test with both pipelines

### Phase 3: Implement Improvements
1. Add multiple output formats
2. Implement named speaker mapping
3. Add confidence indicators
4. Implement paragraph breaks based on silence
5. Add other improvements from the to-do list

### Phase 4: Performance Optimization
1. Profile and optimize performance
2. Add caching for intermediate results
3. Implement parallel processing where possible

## Shell and Python Integration

The shell wrappers provide several advantages that should be preserved:
- Environment setup and management
- User-friendly documentation and error messages
- System integration and resource management
- Pipeline orchestration

The Python modules handle:
- Core processing logic
- Complex configuration
- Data manipulation
- Format conversion

This hybrid approach maintains the best of both worlds: the accessibility and system integration of shell scripts with the modularity and power of Python.

## WhisperX Integration

WhisperX offers several advantages:
- Faster processing
- Better speaker diarisation integration
- More accurate word-level timestamps
- Better handling of overlapping speech

The modular architecture allows for a gradual transition:
1. First implement the basic WhisperX functionality
2. Test it alongside the existing Whisper implementation
3. Add WhisperX-specific optimizations and features
4. Make WhisperX the default when ready

Both models will implement the same interface, allowing for easy switching between them.

## Conclusion

This architecture provides a clean migration path from the current monolithic script to a modular, maintainable system. It preserves the strengths of the current implementation while enabling future improvements and the planned transition to WhisperX.

The separation of concerns allows for individual components to be developed, tested, and optimized independently, making the system more robust and extensible.
