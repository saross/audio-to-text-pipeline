# Audio Transcription Pipeline with Speaker Diarization

A containerized pipeline for transcribing audio files using whisper.cpp and identifying speakers using pyannote.audio.

## Overview

This pipeline provides:
- **High-quality transcription** using whisper.cpp (C++ implementation of OpenAI's Whisper)
- **Speaker diarization** using pyannote.audio (identifies who spoke when)
- **Audio preprocessing** with 7-step optimization for transcription accuracy
- **Automatic term corrections** for domain-specific vocabulary
- **GPU acceleration** for both transcription and diarization
- **Progress indicators** showing real-time processing status

## Quick Start

```bash
# 1. Preprocess audio (MP4 to optimized FLAC)
cd audio-preprocessing
./process_audio.sh convert input.mp4 output.flac

# 2. Transcribe with whisper.cpp
cd ../transcription-pipeline-cpp
./scripts/run_whisper.sh data/processed/interview.flac -o results/transcript.txt

# 3. Run speaker diarization
export HF_TOKEN=your_huggingface_token
./scripts/run_diarization.sh diarize data/processed/interview.flac -o results/diarization.json

# 4. Merge transcript with speakers
./scripts/run_diarization.sh merge results/transcript.txt results/diarization.json \
    --speaker-names "Interviewer,Participant"
```

## Components

### 1. Audio Preprocessing (`audio-preprocessing/`)

Converts and optimizes audio for transcription:

- **Script**: `simple_preprocess.py`
- **Features**:
  - MP4 to FLAC conversion
  - 16kHz resampling (optimal for Whisper)
  - Mono conversion
  - Noise reduction
  - Volume normalization
  - Dynamic range compression (3:1 ratio for diarization)
  - Speech enhancement EQ
  - Optional silence removal

#### Usage:
```bash
# Single file
python3 simple_preprocess.py convert input.mp4 output.flac

# Batch processing
python3 simple_preprocess.py batch /input/dir /output/dir

# With silence removal
python3 simple_preprocess.py convert input.mp4 output.flac --remove-silence
```

### 2. Transcription (`transcription-pipeline-cpp/`)

Uses whisper.cpp in Docker for fast, accurate transcription:

- **Script**: `scripts/run_whisper.sh`
- **Docker Image**: CUDA-enabled whisper.cpp
- **Features**:
  - GPU acceleration
  - Progress percentage display
  - Automatic term corrections
  - Multiple model sizes (tiny, base, small, medium)
  - Custom prompts for domain-specific terms

#### Usage:
```bash
# Build Docker image (first time only)
./scripts/run_whisper.sh --build

# Transcribe audio
./scripts/run_whisper.sh audio.flac -o transcript.txt -m medium.en

# With custom prompt
./scripts/run_whisper.sh audio.flac -p config/whisper_prompt.txt

# Skip automatic corrections
./scripts/run_whisper.sh audio.flac --no-corrections

# Verbose mode (see all output)
./scripts/run_whisper.sh audio.flac -v
```

### 3. Speaker Diarization (`scripts/diarize_audio.py`)

Identifies different speakers in the audio:

- **Technology**: pyannote.audio 3.1
- **Docker Image**: PyTorch-based with CUDA support
- **Features**:
  - Auto-detects number of speakers
  - GPU acceleration
  - Progress tracking
  - Multiple output formats (JSON, RTTM, simple text)

#### Setup:
```bash
# Build Docker image (first time only)
./scripts/run_diarization.sh --build

# Get Hugging Face token from: https://huggingface.co/settings/tokens
# Accept model at: https://huggingface.co/pyannote/speaker-diarization-3.1
export HF_TOKEN=your_token_here
```

#### Usage:

**For short files (<10 minutes):**
```bash
# Auto-detect speakers
./scripts/run_diarization.sh diarize audio.flac

# Specify number of speakers
./scripts/run_diarization.sh diarize audio.flac -n 2
```

**For long files (>10 minutes):**
```bash
# Use chunked processing to avoid hanging
./scripts/chunked_diarization.sh -c 10 -n 2 audio.flac output.json

# Smaller chunks for very long files
./scripts/chunked_diarization.sh -c 5 audio.flac output.json
```

### 4. Merge Transcripts with Speakers (`scripts/merge_diarization.py`)

Combines transcription with speaker identification:

- **Input**: Transcript + diarization JSON
- **Output**: Speaker-labeled transcript
- **Features**:
  - Custom speaker names
  - Multiple output formats (text, JSON, SRT)
  - Intelligent segment matching

#### Usage:
```bash
# Basic merge
./scripts/run_diarization.sh merge transcript.txt diarization.json

# With speaker names
./scripts/run_diarization.sh merge transcript.txt diarization.json \
    --speaker-names "Dr. Smith,Patient"

# SRT format for subtitles
./scripts/run_diarization.sh merge transcript.txt diarization.json \
    --format srt
```

### 5. Complete Pipeline (`scripts/transcribe_with_speakers.sh`)

Runs the entire pipeline end-to-end:

```bash
./scripts/transcribe_with_speakers.sh audio.flac \
    -n 2 \
    --speaker-names "Interviewer,Guest" \
    -o final_transcript.txt
```

## Configuration Files

### `config/term_corrections.txt`
Domain-specific term corrections applied automatically after transcription:
```
incorrect_term -> correct_term
```

### `config/whisper_prompt.txt`
Custom vocabulary and context for improving transcription accuracy.

## Docker Images

1. **whisper-cpp-cuda**: Whisper.cpp with CUDA support
   - Base: nvidia/cuda:12.2.0-devel-ubuntu22.04
   - Includes: whisper.cpp compiled with CUDA

2. **diarization-pyannote**: Speaker diarization
   - Base: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
   - Includes: pyannote.audio 3.1

## Progress Indicators

- **Transcription**: Shows percentage complete and elapsed time
  ```
  Transcribing... 45% [03:25]
  ```

- **Diarization**: Shows audio duration and processing progress
  ```
  Audio duration: 45:30 (2730.0 seconds)
  Diarization progress: [████████--] 80% ETA: 2:30
  ```

## Performance

Typical processing times (with GPU):
- **Preprocessing**: ~1 minute per hour of audio
- **Transcription**: ~3-5 minutes per hour of audio
- **Diarization**: ~10-15 minutes per hour of audio

File size reductions:
- **MP4 to FLAC**: ~90% reduction with quality preservation
- **Compression level**: FLAC level 8 (maximum)

## Troubleshooting

See [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) for detailed solutions to common issues.

### Quick Fixes

**GPU Issues**
```bash
./scripts/run_whisper.sh --gpu-test
./scripts/run_diarization.sh --gpu-test
```

**Diarization Token**
```bash
export HF_TOKEN=your_huggingface_token
```

## Technical Details

- [`docs/TECHNICAL_DETAILS.md`](docs/TECHNICAL_DETAILS.md) - Implementation details and architecture
- [`docs/whisper-cpp-investigation.md`](docs/whisper-cpp-investigation.md) - Historical notes on whisper.cpp setup

## Project Structure

```
audio-transcription/
├── audio-preprocessing/
│   ├── simple_preprocess.py      # Main preprocessing script
│   ├── process_audio.sh          # Wrapper script
│   └── batch_process_transcription_data.sh
├── transcription-pipeline-cpp/
│   ├── config/
│   │   ├── term_corrections.txt  # Post-processing corrections
│   │   └── whisper_prompt.txt    # Custom vocabulary
│   ├── data/
│   │   ├── raw/                  # Original MP4 files
│   │   └── processed/            # Preprocessed FLAC files
│   ├── docker/
│   │   ├── Dockerfile.whisper-cpp     # Whisper container
│   │   ├── Dockerfile.diarization     # Diarization container
│   │   └── transcribe.py              # Whisper wrapper
│   ├── results/                  # Output transcripts
│   └── scripts/
│       ├── run_whisper.sh        # Main transcription runner
│       ├── run_diarization.sh    # Diarization runner
│       ├── diarize_audio.py      # Diarization implementation
│       ├── merge_diarization.py  # Merge speakers with transcript
│       ├── apply_corrections.py  # Term corrections
│       └── transcribe_with_speakers.sh  # Complete pipeline
```

## Recent Updates

- Added progress indicators for transcription (shows %)
- Added progress tracking for diarization
- Fixed false error messages after successful transcription
- Automated term corrections (now default, use --no-corrections to skip)
- Optimized audio preprocessing for better diarization (gentler compression)
- Added high-frequency boost for better consonant recognition
- Containerized all components for portability

## License

This project uses:
- whisper.cpp (MIT License)
- pyannote.audio (MIT License)
- Various open-source tools and libraries

---
Last updated: 2024