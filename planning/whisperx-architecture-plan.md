# WhisperX Migration and Architecture Plan for Academic/Research Transcription

## Overview

This document outlines the architectural design for migrating the audio transcription pipeline from Whisper to WhisperX, with a specific focus on academic and research content. The system is designed for transcribing interviews, meetings, panel discussions, and workshops in technical and academic settings.

## Goals

1. **WhisperX Integration**: Replace Whisper with WhisperX for improved performance and accuracy
2. **Academic/Research Focus**: Optimize for technical terminology, multi-speaker discussions, and research contexts
3. **Hardware Adaptability**: Support optimal performance across various hardware configurations (6GB GPU, higher-end GPUs, CPU, Raspberry Pi)
4. **Portability**: Ensure system works on different platforms including cloud VMs and ARM-based devices
5. **Support Future Improvements**: Design with consideration for containerisation, testing, and workflow management

## System Architecture

```
transcription_pipeline/
├── __init__.py               # Package initialization & version
├── config.py                 # Simplified configuration management
├── models/                   # Model implementations
│   ├── __init__.py
│   ├── whisperx_model.py     # WhisperX implementation
│   └── diarisation.py        # Speaker identification capabilities
├── processors/               # Processing modules
│   ├── __init__.py
│   ├── preprocess.py         # Audio preprocessing (from existing)
│   ├── postprocess.py        # Technical term corrections, academic vocabulary
│   └── segment_merger.py     # Discussion flow handling
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── audio_utils.py        # Audio file handling
│   └── device_utils.py       # Hardware detection & optimization
├── formats/                  # Output formatters
│   ├── __init__.py
│   ├── txt_format.py         # Text transcript format
│   └── json_format.py        # JSON format for programmatic analysis
├── cli/                      # Command-line interface
│   ├── __init__.py
│   └── main.py               # Entry point with argument parsing
└── scripts/                  # Shell wrappers
    ├── setup.sh              # Installation & environment setup
    ├── preprocess.sh         # Audio preprocessing wrapper
    ├── transcribe.sh         # Main transcription wrapper
    └── env_vars.sh           # Environment variables
```

## Component Responsibilities

### Configuration (`config.py`)

The configuration module centralises all system settings and provides:
- Command-line argument parsing
- Configuration file loading (YAML/JSON)
- Environment variable support
- Hardware-specific automatic optimisation
- Default value management
- Configuration validation

```python
class TranscriptionConfig:
    def __init__(self, args=None, config_file=None):
        # Set defaults
        self.model_type = "whisperx"
        self.model_size = "medium"
        self.device = "auto"  # auto, cuda, cpu
        self.language = "en"
        self.batch_size = 4
        self.output_format = "txt"  # txt or json
        self.diarize = True
        self.num_speakers = None  # Auto-detect
        self.academic_mode = True  # Enable academic/technical term handling
        
        # Load from file if provided
        if config_file:
            self._load_from_file(config_file)
            
        # Override with command line args
        if args:
            self._apply_args(args)
            
        # Auto-detect and optimize for hardware
        if self.device == "auto":
            self._configure_for_hardware()
```

### Models

#### `models/whisperx_model.py`
Implements the WhisperX integration with hardware-aware configuration:
- Dynamic model loading based on available resources
- Precision control (FP32, FP16, INT8) for memory efficiency
- Batch size optimisation
- Graceful degradation for memory constraints
- Special handling for technical terminology and academic speech patterns

```python
class WhisperXModel:
    def __init__(self, config):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = config.get("model_size", "medium")
        self.compute_type = config.get("compute_type", "float16" if self.device == "cuda" else "float32")
        self.batch_size = config.get("batch_size", 4)
        
        # Use a more suitable language model for academic/technical content
        self.academic_mode = config.get("academic_mode", True)
        
        # Load appropriate model for hardware
        self.model = self._load_model()
        
    def transcribe(self, audio_file):
        # Transcription with hardware-optimized settings
        # Additional prompting for academic/technical content if enabled
```

#### `models/diarisation.py`
Handles speaker identification with WhisperX integration:
- Speaker segmentation optimized for meeting/interview settings
- Speaker counting (auto or manual)
- Integration with WhisperX's native capabilities
- Speaker identity mapping
- Handling of overlapping speech in discussion scenarios

### Processors

#### `processors/preprocess.py`
Maintains the existing audio preprocessing functionality:
- Format conversion
- Sample rate standardisation
- Noise reduction
- Normalisation
- Speech enhancement

#### `processors/postprocess.py`
Handles text processing after transcription with academic/research focus:
- Technical term corrections from external files
- Academic vocabulary handling
- Research citation and reference formatting
- Handling mathematical expressions and technical notation

```python
def load_academic_terms(filepath="academic_terms.txt"):
    """Load specialized academic and technical terminology."""
    terms = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('|')
                    if len(parts) == 2:
                        incorrect, correct = parts
                        terms[incorrect.strip()] = correct.strip()
        print(f"Loaded {len(terms)} academic terms from {filepath}")
        return terms
    except FileNotFoundError:
        print(f"Academic terms file {filepath} not found.")
        return {}

def correct_technical_terms(text, technical_terms, general_terms):
    """Prioritize technical term corrections over general corrections."""
    # First apply technical/academic term corrections
    if technical_terms:
        for incorrect, correct in technical_terms.items():
            pattern = re.compile(r'\b' + re.escape(incorrect) + r'\b', re.IGNORECASE)
            text = pattern.sub(correct, text)
    
    # Then apply general term corrections
    if general_terms:
        for incorrect, correct in general_terms.items():
            # Don't override technical terms that have already been corrected
            if any(re.search(r'\b' + re.escape(correct_tech) + r'\b', text, re.IGNORECASE) 
                  for _, correct_tech in technical_terms.items()):
                continue
                
            pattern = re.compile(r'\b' + re.escape(incorrect) + r'\b', re.IGNORECASE)
            text = pattern.sub(correct, text)
            
    return text
```

#### `processors/segment_merger.py`
Manages transcript segment operations with specific handling for academic discussions:
- Merging consecutive segments from the same speaker
- Handling question-answer patterns
- Identifying and preserving important discussion transitions
- Managing interruptions and overlapping speech common in group discussions

```python
def detect_discussion_pattern(segments):
    """Identify question-answer patterns and discussion dynamics."""
    qa_pairs = []
    for i in range(len(segments) - 1):
        # Check if this might be a question (ends with ? or has question indicators)
        current = segments[i]
        next_seg = segments[i+1]
        
        # Check for question patterns
        is_question = (current["text"].strip().endswith("?") or 
                     re.search(r'\b(what|how|why|when|where|who|could you|can you)\b', 
                               current["text"], re.IGNORECASE))
        
        # Different speakers and within reasonable time
        if (is_question and 
            current["speaker"] != next_seg["speaker"] and
            next_seg["start"] - current["end"] < 3.0):
            
            qa_pairs.append((i, i+1))  # Record the Q&A pair indices
    
    return qa_pairs
```

### Utils

#### `utils/device_utils.py`
Provides hardware detection and optimisation:
- GPU memory detection
- CPU core counting
- Platform detection (x86, ARM)
- Optimal configuration generation

```python
def get_optimal_config(minimum_required_mb=6000):
    """Detect available hardware and return optimal configuration."""
    config = {"device": "cpu", "batch_size": 1, "precision": "float32"}
    
    if torch.cuda.is_available():
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        
        # Scale batch size and model precision based on available VRAM
        if vram_mb >= 24000:  # High-end GPU
            config = {"device": "cuda", "batch_size": 16, "precision": "float16", 
                      "model_size": "large-v3"}
        elif vram_mb >= 12000:  # Mid-range GPU
            config = {"device": "cuda", "batch_size": 8, "precision": "float16", 
                      "model_size": "large-v3"}
        elif vram_mb >= 6000:  # Minimum 6GB GPU (your setup)
            config = {"device": "cuda", "batch_size": 4, "precision": "float16", 
                      "model_size": "medium"}
        else:
            # CPU fallback with optimized settings
            config = {"device": "cpu", "batch_size": 1, "precision": "float32", 
                      "model_size": "tiny"}
            
    return config
```

#### `utils/audio_utils.py`
Provides audio file handling functionality:
- Format detection
- Duration calculation
- Audio loading optimisations
- Voice quality assessment for academic content

### Formats

#### `formats/txt_format.py`
Implements research-friendly transcript format with timestamps, speakers, and enhanced academic content formatting.

```python
class TxtFormat:
    def write(self, segments, output_file, metadata=None):
        """Write segments to the output file in text format optimized for research content."""
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write metadata header
            if metadata:
                f.write(f"# Transcript of {metadata.get('filename', 'audio')}\n")
                f.write(f"# Transcribed on: {metadata.get('date', datetime.now().strftime('%Y-%m-%d %H:%M'))}\n")
                f.write(f"# Duration: {metadata.get('duration', 0):.2f} seconds\n")
                f.write(f"# Speakers: {metadata.get('num_speakers', 'auto-detected')}\n")
                if metadata.get('context'):
                    f.write(f"# Context: {metadata.get('context')}\n")
                f.write("\n")
            
            # Write each segment with timestamp and speaker
            for segment in segments:
                # Format timestamp as [HH:MM:SS]
                m, s = divmod(segment["start"], 60)
                h, m = divmod(m, 60)
                timestamp = f"[{int(h):02d}:{int(m):02d}:{int(s):02d}]"
                
                # Write the line with proper formatting for academic content
                speaker_label = segment['speaker']
                
                # Check if this is a question for special formatting
                if segment.get('is_question'):
                    f.write(f"{timestamp} {speaker_label} (Question): {segment['text']}\n\n")
                else:
                    f.write(f"{timestamp} {speaker_label}: {segment['text']}\n\n")
```

#### `formats/json_format.py`
Implements structured JSON output for programmatic analysis of research discussions.

```python
class JsonFormat:
    def write(self, segments, output_file, metadata=None):
        """Write segments to the output file in JSON format for programmatic analysis."""
        output = {
            "metadata": metadata or {},
            "segments": segments,
            # Add discussion analysis
            "statistics": self._generate_statistics(segments),
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
    def _generate_statistics(self, segments):
        """Generate statistics about the transcript useful for research analysis."""
        stats = {
            "speaker_talk_time": {},
            "speaker_turn_count": {},
            "total_duration": 0,
        }
        
        # Calculate talk time and turn counts for each speaker
        for segment in segments:
            speaker = segment["speaker"]
            duration = segment["end"] - segment["start"]
            
            stats["speaker_talk_time"][speaker] = stats["speaker_talk_time"].get(speaker, 0) + duration
            stats["speaker_turn_count"][speaker] = stats["speaker_turn_count"].get(speaker, 0) + 1
            stats["total_duration"] += duration
            
        # Calculate participation percentages
        for speaker in stats["speaker_talk_time"]:
            stats["speaker_talk_time"][speaker + "_percent"] = (
                stats["speaker_talk_time"][speaker] / stats["total_duration"] * 100
            )
            
        return stats
```

### CLI

#### `cli/main.py`
Provides the command-line interface with research-specific options:
- Argument parsing
- Pipeline orchestration
- Academic mode toggle
- Technical term dictionary specification

```python
def parse_arguments():
    """Parse command line arguments for academic/research transcription."""
    parser = argparse.ArgumentParser(
        description="Transcribe academic/research audio with WhisperX"
    )
    
    parser.add_argument("input_file", help="Path to the audio file")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--speakers", "-s", type=int, 
                      help="Number of speakers in the recording (if known)")
    
    # Research-specific options
    parser.add_argument("--academic-terms", help="Path to academic/technical terms dictionary")
    parser.add_argument("--context", help="Add research context for better transcription")
    parser.add_argument("--speaker-names", help="Comma-separated list of speaker names in order")
    
    # Format options
    parser.add_argument("--format", choices=["txt", "json"], default="txt",
                      help="Output format (default: txt)")
    
    # Hardware options
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                      help="Compute device to use (default: auto)")
    parser.add_argument("--model", choices=["tiny", "small", "medium", "large-v3"], 
                      default="medium", help="Model size (default: medium)")
    
    return parser.parse_args()
```

## Hardware Configuration Strategy

The system is designed to adapt to different hardware environments with research-optimized profiles:

```python
# Configuration for different hardware tiers
HARDWARE_TIERS = {
    "raspberry_pi": {
        "model_size": "tiny",
        "batch_size": 1,
        "precision": "float32",
        "device": "cpu",
        "vad_filter": True,  # Voice Activity Detection to save processing
        "compute_type": "int8"  # Use quantization
    },
    "laptop_cpu": {
        "model_size": "small",
        "batch_size": 1,
        "precision": "float32",
        "device": "cpu"
    },
    "gpu_6gb": {  # Your current setup
        "model_size": "medium",
        "batch_size": 4,
        "precision": "float16",
        "device": "cuda"
    },
    "gpu_12gb": {
        "model_size": "large-v3",
        "batch_size": 8,
        "precision": "float16",
        "device": "cuda"
    },
    "gpu_24gb": {
        "model_size": "large-v3",
        "batch_size": 16,
        "precision": "float16",
        "device": "cuda",
        "beam_size": 5  # Better quality with more memory
    },
    "cloud_vm": {
        "model_size": "large-v3",
        "batch_size": 16,
        "precision": "float16",
        "device": "cuda",
        "beam_size": 5
    }
}
```

## Design Principles

### 1. Research Content Optimization

- Technical vocabulary recognition and correction
- Academic terminology handling
- Question-answer pattern detection
- Speaker contribution analysis
- Citation and reference formatting

### 2. Hardware Adaptability

- Dynamic resource detection
- Model selection based on available hardware
- Precision scaling (FP32, FP16, INT8)
- Batch size optimisation
- Graceful degradation for memory constraints

### 3. Cross-Platform Portability

- OS-agnostic design using standard libraries
- Path handling with `pathlib` for cross-platform compatibility
- Environment variable support for configuration
- CPU fallback for environments without GPUs
- ARM architecture support (Raspberry Pi)

### 4. Future-Proofed Features

- Named speaker identification
- Technical term correction with academic dictionaries
- Multiple speaker handling optimized for discussions
- Comprehensive metadata headers
- Academic discussion flow analysis

### 5. Containerisation Support

- Clear separation of configuration from code
- Environment variable integration
- Relative path support
- Clean separation of processing steps
- Volume mapping considerations

## WhisperX Integration Details

### Installation Requirements

```bash
# Basic installation (to include in setup.sh)
pip install git+https://github.com/m-bain/whisperx.git

# Dependencies
pip install torch torchaudio pyannote.audio
```

### Core Pattern

```python
import whisperx
import torch

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisperx.load_model("large-v3", device, compute_type="float16")

# Transcribe with word-level timestamps
result = model.transcribe("audio.mp3", batch_size=4)
segments = result["segments"]

# Speaker diarisation with WhisperX
diarize_model = whisperx.load_diarization_model(device)
diarization = diarize_model({"audio": "audio.mp3"})

# Assign speaker labels
result = whisperx.assign_speakers(segments, diarization)
final_segments = result["segments"]
```

### Research-Specific Optimizations

```python
# Initial prompt to guide WhisperX towards academic content
initial_prompt = """This is an academic discussion with technical terminology."""

# If we know the topic area
if config.research_area:
    initial_prompt += f" The topic is {config.research_area}."

# Add technical terms awareness
if config.technical_terms:
    term_examples = ", ".join(list(config.technical_terms.values())[:5])
    initial_prompt += f" Technical terms include: {term_examples}."

# Use the prompt with WhisperX
result = model.transcribe(
    audio_file,
    batch_size=batch_size,
    initial_prompt=initial_prompt
)
```

### Key WhisperX Advantages for Research Content

1. **Accuracy with Technical Terms**: Better handling of specialized vocabulary
2. **Speaker Diarisation**: Crucial for multi-person discussions, panels, and interviews
3. **Word-level Timestamps**: More precise segment boundaries
4. **Speed**: 4-10x faster than standard Whisper, important for longer academic discussions

## Implementation Phases

### Phase 1: Core Structure & Configuration (Week 1)

1. Set up package structure
2. Implement configuration system with hardware detection
3. Create basic utilities for device management

### Phase 2: WhisperX Integration (Week 2)

1. Implement WhisperX model with hardware adaptation
2. Create diarisation module with academic discussion optimization
3. Build segment processing system

### Phase 3: Academic/Research Processing (Week 3)

1. Implement technical term handling
2. Create academic discussion pattern detection
3. Build speaker contribution analysis
4. Develop specialized post-processing for academic content

### Phase 4: CLI & Shell Integration (Week 4)

1. Build research-focused command-line interface
2. Create shell wrappers for the new system
3. Write comprehensive documentation

## Technical Considerations

### Academic Content Processing

- Configure initial prompts for WhisperX with academic cues
- Implement customizable technical term dictionaries
- Add speaker role identification (moderator, presenter, audience)
- Create specialized post-processing for references, citations, and equations

### Memory Management

- Configure batch sizes dynamically based on available VRAM
- Implement model precision control (FP32, FP16, INT8)
- Use VAD (Voice Activity Detection) to reduce processing needs
- Consider model size vs. accuracy tradeoffs for technical content

### Platform Compatibility

- Use Python 3.8+ for compatibility
- Ensure ARM compatibility for Raspberry Pi
- Test on Linux, macOS, and Windows (WSL)
- Use cross-platform libraries

## Conclusion

This architecture provides a focused solution for academic and research transcription while ensuring future extensibility, hardware adaptability, and portability. By optimizing specifically for academic/technical content, the system will deliver significantly better transcription quality for interviews, meetings, panels, and workshops in research settings.

The design maintains a clean separation of concerns while providing the flexibility needed for diverse computing environments, from Raspberry Pi clusters to high-end cloud VMs. The modular structure supports all planned future improvements including containerisation, testing, and research workflows.

