# Audio Transcription and Speaker Diarisation System

## Overview

This documentation covers two complementary scripts for audio transcription and speaker diarisation:

1. **`transcript.py`**: GPU-optimised implementation using the Whisper 'medium.en' model
   - Faster processing
   - Works well with 6GB or more VRAM
   - Excellent quality for most English recordings
   - Uses GPU for transcription, CPU for diarisation

2. **`transcript_cpu.py`**: CPU-only implementation using the Whisper 'large-v3' model
   - Highest possible transcription quality
   - No GPU memory limitations
   - Slower processing (5-10× longer)
   - Completely CPU-based processing

Both scripts combine:
- **Whisper** (from OpenAI): For speech-to-text transcription
- **pyannote.audio**: For speaker identification (diarisation)


## Prerequisites

- **Python 3.8+**: All scripts are written in Python (tested up to 3.12)
- **NVIDIA GPU** (optional): For faster processing with `transcript.py`
- **CUDA toolkit**: For GPU acceleration
- **Hugging Face account**: For accessing the diarisation models
- **ffmpeg**: For audio processing and conversion



## Scripts Included

1. **`transcript.py`**: GPU-optimised implementation for fast processing
2. **`transcript_cpu.py`**: CPU-only implementation for highest quality 
3. **`run_transcription.sh`**: Shell script for GPU processing
4. **`run_cpu_transcription.sh`**: Shell script for CPU-only processing
5. **`audio_preprocess.py`**: Audio preprocessing tool to improve transcription accuracy

The preprocessing script enhances audio quality through:
- Noise reduction
- Volume normalisation
- Dynamic range compression
- Speech frequency enhancement
- Silence trimming
- Conversion to optimal format (mono 16kHz FLAC)

## Installation Steps

1. **Create a virtual environment**:
   ```bash
   python3 -m venv ~/transcription-env
   source ~/transcription-env/bin/activate
   ```

2. **Install required packages**:
   ```bash
   pip install openai-whisper torch pyannote.audio pydub tqdm numpy
   ```

3. **Set up Hugging Face token**:
   ```bash
   export HUGGINGFACE_TOKEN=your_token_here
   ```

4. **Accept model licenses** at:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/embedding-3.1

## Package Versions

The scripts have been tested with the following package versions:

| Package | Version | Purpose |
|---------|---------|---------|
| openai-whisper | 20240311+ | Speech recognition |
| torch | 2.2.0+ | Deep learning framework |
| pyannote.audio | 3.1.0+ | Speaker diarisation |
| pydub | 0.25.1+ | Audio file handling |
| tqdm | 4.66.1+ | Progress bars |

## Shell Scripts Explained

Both Python scripts have accompanying shell scripts that handle environment setup, resource optimization, and execution:

### 1. `run_transcription.sh` - GPU Optimised Processing

This shell script:

- **Sets up the environment**:
  ```bash
  source ~/transcription-env/bin/activate
  ```

- **Configures GPU memory optimization**:
  ```bash
  export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
  export CUDA_LAUNCH_BLOCKING="1"
  ```
  These settings help prevent GPU memory fragmentation and improve allocation efficiency.

- **Configures CPU operations**:
  ```bash
  export OMP_NUM_THREADS="4"
  export TOKENIZERS_PARALLELISM="false"
  ```
  This optimises CPU usage for operations like diarisation.

- **Checks for required environment variables**:
  ```bash
  if [ -z "$HUGGINGFACE_TOKEN" ]; then
      echo "ERROR: HUGGINGFACE_TOKEN environment variable is not set."
      exit 1
  fi
  ```

- **Creates output directories** as needed:
  ```bash
  mkdir -p "$OUTPUT_DIR"
  ```

- **Clears GPU memory** before starting:
  ```bash
  python -c "import torch; torch.cuda.empty_cache()"
  ```
  This ensures maximum available VRAM for processing.

- **Tracks processing time**:
  ```bash
  START_TIME=$(date +%s)
  # ...processing...
  ELAPSED=$((END_TIME - START_TIME))
  ```
  Reports total time in hours, minutes, and seconds.

- **Provides error handling**:
  ```bash
  if python transcript.py "$@"; then
      # Success message
  else
      # Error message and suggestion to try CPU version
  fi
  ```

### 2. `run_cpu_transcription.sh` - CPU-Only Processing

This shell script:

- **Sets up the environment**:
  ```bash
  source ~/transcription-env/bin/activate
  ```

- **Detects and configures CPU threads** intelligently:
  ```bash
  if command -v nproc &> /dev/null; then
      TOTAL_CORES=$(nproc)
      CALCULATED_THREADS=$(( TOTAL_CORES * 3 / 4 ))
      CPU_THREADS=$(( CALCULATED_THREADS < 4 ? 4 : (CALCULATED_THREADS > 10 ? 10 : CALCULATED_THREADS) ))
  fi
  ```
  This automatically uses 75% of available CPU cores (minimum 4, maximum 10).

- **Sets CPU optimization** environment variables:
  ```bash
  export OMP_NUM_THREADS="$CPU_THREADS"
  export MKL_NUM_THREADS="$CPU_THREADS"
  export OPENBLAS_NUM_THREADS="$CPU_THREADS"
  export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
  ```
  These settings balance performance and system responsiveness.

- **Explicitly disables CUDA** to ensure CPU-only operation:
  ```bash
  export CUDA_VISIBLE_DEVICES=""
  ```

- **Estimates processing time** based on audio length:
  ```bash
  DURATION_SECONDS=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$1")
  MIN_ESTIMATE=$(echo "$DURATION_SECONDS / 60 * 5" | bc)
  MAX_ESTIMATE=$(echo "$DURATION_SECONDS / 60 * 10" | bc)
  ```
  This provides users with an estimated completion time (5-10× real-time).

- **Provides detailed progress reporting** and error handling similar to the GPU script.

## Script Execution

Both scripts can be run directly or using their corresponding shell scripts:

### For GPU-Optimised Processing (Medium.en Model)

```bash
# Direct Python execution
python transcript.py data/interview.mp3 -o output/transcript.txt -s 2

# Using shell script (recommended)
./run_transcription.sh data/interview.mp3 -o output/transcript.txt -s 2
```

### For CPU-Only Processing (Large-v3 Model)

```bash
# Direct Python execution
python transcript_cpu.py data/interview.mp3 -o output/transcript.txt -s 2

# Using shell script (recommended)
./run_cpu_transcription.sh data/interview.mp3 -o output/transcript.txt -s 2
```

## Command Line Parameters

Both scripts accept the same parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `audio_file` | Path to the audio file (required) | - |
| `-o, --output FILE` | Output transcript path | `[input_filename]_transcript.txt` |
| `-s, --speakers NUM` | Number of speakers in recording | Auto-detect |

## How the Scripts Work

### GPU-Based Script (transcript.py)

The GPU script works in several distinct phases:

1. **Device selection and configuration**:
   - Checks for GPU availability and VRAM amount
   - Warns if GPU is being used for display (which reduces available VRAM)
   - Uses full precision (FP32) for maximum compatibility

2. **Whisper model loading**:
   ```python
   model = whisper.load_model(model_size, device=device)
   ```
   - Loads the medium.en model, optimised for English transcription
   - Places the model on GPU for faster processing

3. **Audio transcription**:
   ```python
   transcription = model.transcribe(audio_file, word_timestamps=True, verbose=False)
   ```
   - Processes the audio file with Whisper's built-in sequential processing
   - Records word-level timestamps for accurate alignment with speakers
   - Shows a progress bar with elapsed time

4. **Memory cleanup**:
   ```python
   del model
   torch.cuda.empty_cache()
   gc.collect()
   ```
   - Clears GPU memory before loading diarisation

5. **Diarisation on CPU**:
   ```python
   # Temporarily disable CUDA visibility
   original_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
   os.environ['CUDA_VISIBLE_DEVICES'] = ''

   # Load and run diarisation
   diarisation = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
   diarisation_result = diarisation(audio_file, num_speakers=num_speakers)

   # Restore original CUDA visibility
   if original_cuda_visible_devices:
       os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible_devices
   ```
   - Temporarily forces CPU-only operation for diarisation to avoid memory issues
   - This approach ensures diarisation works even with limited VRAM

6. **Speaker assignment and segment merging**:
   - Combines transcription segments with speaker identities
   - Merges consecutive segments from the same speaker
   - Formats and saves the final transcript

### CPU-Based Script (transcript_cpu.py)

The CPU script follows a similar workflow but with key differences:

1. **Force CPU-only processing**:
   ```python
   os.environ["CUDA_VISIBLE_DEVICES"] = ""
   ```
   - Completely disables CUDA to ensure CPU-only operation

2. **CPU optimisation**:
   ```python
   os.environ["OMP_NUM_THREADS"] = "4"
   os.environ["MKL_NUM_THREADS"] = "4"
   os.environ["OPENBLAS_NUM_THREADS"] = "4"
   ```
   - Controls thread count for various math libraries

3. **Using the large-v3 model**:
   ```python
   model = whisper.load_model("large-v3")
   ```
   - Uses the largest, most accurate Whisper model
   - This offers better handling of accents, background noise, and complex speech

4. **Processing and output**:
   - After transcription and diarisation, follows the same approach for combining results
   - The output format is identical to the GPU version

## Diarisation Process

Both scripts use the same approach for speaker identification:

1. **For each transcription segment**:
   ```python
   for segment in transcription["segments"]:
       # Get the time range for this segment
       segment_range = Segment(segment["start"], segment["end"])
       
       # Find which speaker was active during this segment
       speaker_times = {}
       for turn, _, speaker in diarisation_result.itertracks(yield_label=True):
           if segment_range.intersects(turn):
               # Calculate overlap duration
               overlap_duration = overlap_end - overlap_start
               speaker_times[speaker] = speaker_times.get(speaker, 0) + overlap_duration
   ```

2. **Assign dominant speaker**:
   ```python
   dominant_speaker = max(speaker_times.items(), key=lambda x: x[1])[0] if speaker_times else "UNKNOWN"
   ```
   - Selects the speaker with the most talking time during each segment

3. **Merge consecutive segments**:
   ```python
   if segment["speaker"] == current["speaker"] and segment["start"] - current["end"] < 1.0:
       current["end"] = segment["end"]
       current["text"] += " " + segment["text"]
   ```
   - Combines segments from the same speaker if there's less than 1 second gap

## Performance Considerations

### Hardware Requirements

| Script | Minimum Hardware | Recommended Hardware |
|--------|-----------------|---------------------|
| transcript.py | 6GB VRAM GPU, 8GB RAM | 8GB+ VRAM GPU, 16GB RAM |
| transcript_cpu.py | 4-core CPU, 8GB RAM | 8+ core CPU, 16GB+ RAM |

### Processing Time

| Recording Length | CPU (large-v3) | GPU (medium.en) |
|-----------------|-------------|--------------|
| 10 minutes | ~50-100 minutes | ~2-5 minutes |
| 1 hour | ~5-10 hours | ~15-30 minutes |

### Model Selection Guide

| Model | VRAM Required | Accuracy | Use When |
|-------|--------------|----------|----------|
| tiny.en | ~1GB | Basic | Limited GPU, simple audio |
| base.en | ~1-2GB | Good | Limited GPU, basic needs |
| small.en | ~2-3GB | Very Good | 4GB GPUs, general use |
| medium.en | ~5GB | Excellent | 6GB+ GPUs, standard needs |
| large-v3 | ~10GB | Superior | CPU only, maximum quality |

## Output Format

Both scripts generate the same output format:

```
# Transcript of recording.mp3

[00:00:05] SPEAKER_00: This is the first speaker talking.

[00:00:10] SPEAKER_01: This is the second speaker responding.
```

## Troubleshooting

### GPU-Related Issues

1. **"CUDA out of memory"**:
   - Your GPU doesn't have enough VRAM
   - Try the CPU version: `./run_cpu_transcription.sh`
   - Or use a smaller Whisper model

2. **"Expected scalar type Float but found Half"**:
   - This indicates a precision mismatch in the model
   - The script has been updated to use full precision (FP32)
   - If you encounter this, ensure you're using the latest version

3. **"Expected a cuda device, but got: cpu"**:
   - This indicates a conflict in device specification for the diarisation model
   - The script uses environment variables to control this properly

4. **Slow GPU performance**:
   - Check if your display is using the GPU: `glxinfo | grep renderer`
   - Connect your monitor to integrated graphics if possible
   - Close other GPU-intensive applications

### Diarisation Issues

1. **Poor speaker identification**:
   - Specify the exact number of speakers: `-s 2`
   - Ensure clear audio with minimal background noise
   - For very similar voices, manual correction might be needed

2. **"Error loading diarisation model"**:
   - Check your Hugging Face token: `echo $HUGGINGFACE_TOKEN`
   - Ensure you've accepted all model licenses
   - Check your internet connection

### CPU Performance

1. **Very slow processing**:
   - Adjust thread count in `run_cpu_transcription.sh`
   - Close other CPU-intensive applications
   - Consider using the GPU version if possible

2. **Script crashes with memory error**:
   - The large-v3 model requires ~10GB RAM
   - Try reducing thread count to reduce memory pressure
   - Consider using smaller Whisper models

## Advanced Customization

### Changing Model Size

In `transcript.py`, modify:
```python
model_size = "medium.en"  # Options: "tiny.en", "base.en", "small.en", "medium.en"
```

In `transcript_cpu.py`, modify:
```python
model = whisper.load_model("large-v3")  # Options: "tiny", "base", "small", "medium", "large-v3"
```

### Adjusting Merging Threshold

Both scripts merge consecutive segments from the same speaker with gaps less than 1 second:
```python
if segment["speaker"] == current["speaker"] and segment["start"] - current["end"] < 1.0:
```

Increase the value (1.0) to merge segments with larger gaps.

### Output Format Customization

Modify the timestamp and output format:
```python
timestamp = f"[{int(h):02d}:{int(m):02d}:{int(s):02d}]"
f.write(f"{timestamp} {segment['speaker']}: {segment['text']}\n\n")
```

## Tips for Best Results

1. **Determine speaker count in advance**:
   - Using `-s` with the exact number of speakers improves diarisation accuracy

2. **Use appropriate model for your needs**:
   - For standard recordings, medium.en on GPU is usually sufficient
   - For difficult audio (accents, noise), use large-v3 on CPU

3. **Shell scripts vs direct Python calls**:
   - Always use the shell scripts when possible
   - They handle environment setup, memory management, and error reporting

4. **Ensure your HuggingFace token is set**:
   ```bash
   export HUGGINGFACE_TOKEN=your_token_here
   ```

5. **Watch memory usage**:
   - For GPU: Use `nvidia-smi` to monitor VRAM usage
   - For CPU: Use `htop` to monitor RAM and CPU usage

## Future Improvements

Potential extensions to consider:
- Audio pre-processing for noise reduction
- Custom vocabulary for domain-specific terminology
- Integration with audio editors for timestamp-based editing
- Multi-language support for non-English recordings
