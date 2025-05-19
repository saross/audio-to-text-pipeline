# Audio Transcription and Speaker Diarisation System

## Overview

This documentation covers a complete audio processing pipeline for transcription and speaker diarisation:

1. **Preprocessing Pipeline**: Audio preprocessing to enhance quality
   - Noise reduction
   - Volume normalisation
   - Speech enhancement
   - Format conversion

2. **Transcription Pipeline**: Two complementary implementation options
   - **GPU-optimised**: Using Whisper 'medium.en' model (faster processing)
   - **CPU-only**: Using Whisper 'large-v3' model (highest quality)

3. **Complete Pipeline**: Combined preprocessing and transcription
   - Flexible workflow with customisable options
   - Preset configurations for different types of recordings

The entire system combines:
- **FFmpeg**: For audio conversion and enhancement
- **Python audio libraries**: For sophisticated audio processing
- **Whisper** (from OpenAI): For speech-to-text transcription
- **pyannote.audio**: For speaker identification (diarisation)

## Preprocessing Pipeline

The preprocessing pipeline enhances audio quality for better transcription accuracy through:

- **Format conversion**: MP4 to FLAC, standardisation to 16kHz mono
- **Noise reduction**: Spectral gating to reduce background noise
- **Volume normalisation**: Consistent audio levels
- **Dynamic range compression**: Clearer speech
- **Speech enhancement**: EQ adjustments for better clarity
- **Silence trimming**: Removing leading/trailing silence

### Preprocessing Presets

The preprocessing pipeline includes preset configurations:

| Preset | Description | Best For |
|--------|-------------|----------|
| interview | Balanced settings (default) | One-on-one interviews |
| lecture | Prioritises speech clarity | Presentations, lectures |
| noisy | Stronger noise reduction | Recordings in noisy environments |
| music | Careful processing for music | Content with music or singing |

## Transcription Pipeline

Two complementary implementations provide flexibility depending on hardware and quality requirements:

### 1. GPU-Optimised Implementation (`transcript.py`)
- Uses Whisper 'medium.en' model
- Faster processing (typically 4-10× real-time)
- Works well with 6GB or more VRAM
- Excellent quality for most English recordings
- Uses GPU for transcription, CPU for diarisation

### 2. CPU-Only Implementation (`transcript_cpu.py`)
- Uses Whisper 'large-v3' model
- Highest possible transcription quality
- No GPU memory limitations
- Slower processing (typically 5-10× longer than real-time)
- Completely CPU-based processing

## Complete Pipeline

The `transcribe_with_preprocessing.sh` script combines preprocessing and transcription:

1. **Audio preprocessing**: Enhances audio quality using customisable settings
2. **Transcription**: Using either GPU or CPU transcription
3. **Speaker diarisation**: Identifies different speakers
4. **Post-processing**: Technical term corrections and format improvements

## Prerequisites

- **Python 3.8+**: All scripts are written in Python (tested up to 3.12)
- **NVIDIA GPU** (optional): For faster processing with GPU implementation
- **CUDA toolkit**: For GPU acceleration
- **FFmpeg**: For audio processing and conversion
- **Hugging Face account**: For accessing the diarisation models

## Scripts Included

### Core Scripts
1. **`transcript.py`**: GPU-optimised transcription
2. **`transcript_cpu.py`**: CPU-only transcription 
3. **`run_transcription.sh`**: Shell script for GPU processing
4. **`run_cpu_transcription.sh`**: Shell script for CPU-only processing

### Preprocessing Scripts
5. **`audio_preprocess.py`**: Audio preprocessing library
6. **`preprocess_audio.sh`**: Standalone preprocessing script
7. **`transcribe_with_preprocessing.sh`**: Combined preprocessing and transcription

### Setup and Utility Scripts
8. **`setup_transcription.sh`**: Environment setup script
9. **`term_corrections.txt`**: Technical term corrections
10. **`whisper_prompt.txt`**: Custom prompt for domain-specific transcription

## Installation Steps

For easy setup, use the provided setup script:

```bash
./setup_transcription.sh
```

This script:
1. Checks Python installation
2. Installs FFmpeg if needed
3. Creates a Python virtual environment
4. Installs required packages
5. Sets up Hugging Face token
6. Creates environment configurations

### Manual Installation

1. **Create a virtual environment**:
   ```bash
   python3 -m venv ~/transcription-env
   source ~/transcription-env/bin/activate
   ```

2. **Install required packages**:
   ```bash
   pip install openai-whisper torch pyannote.audio pydub librosa soundfile tqdm numpy scipy
   ```

3. **Set up Hugging Face token**:
   ```bash
   export HUGGINGFACE_TOKEN=your_token_here
   ```

4. **Accept model licenses** at:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/embedding-3.1

## Usage

### For Audio Preprocessing Only

```bash
./preprocess_audio.sh interview.mp4 -p interview -o processed/interview.flac
```

Key options:
- `-p, --preset`: Select a preset (interview, lecture, noisy, music)
- `-o, --output`: Specify output path
- `--no-noise-reduction`, `--no-normalize`, etc.: Disable specific steps

### For Transcription (with Preprocessing)

```bash
./transcribe_with_preprocessing.sh interview.mp4 -o transcripts/interview.txt -s 2
```

Key options:
- `-s, --speakers`: Number of speakers (improves diarisation)
- `-o, --output`: Output transcript path
- `-p, --preset`: Preprocessing preset
- `--skip-preprocessing`: Skip preprocessing step
- `--cpu-only`: Force CPU-only mode for highest quality

### For Transcription Only (GPU mode)

```bash
./run_transcription.sh processed/interview.flac -o transcripts/interview.txt -s 2
```

### For Transcription Only (CPU mode)

```bash
./run_cpu_transcription.sh processed/interview.flac -o transcripts/interview.txt -s 2
```

## Command Line Parameters

### Preprocessing Parameters (`preprocess_audio.sh`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `audio_file` | Path to the audio file (required) | - |
| `-o, --output FILE` | Output file path | `[input_filename]_processed.flac` |
| `-p, --preset STRING` | Preset configuration | interview |
| `--no-flac` | Skip FLAC conversion | Enabled |
| `--no-resample` | Skip resampling to 16kHz | Enabled |
| `--no-noise-reduction` | Skip noise reduction | Enabled |
| `--no-normalize` | Skip volume normalisation | Enabled |
| `--no-compression` | Skip dynamic range compression | Enabled |
| `--no-enhance` | Skip speech enhancement | Enabled |
| `--no-trim` | Skip silence trimming | Enabled |
| `--no-mono` | Skip conversion to mono | Enabled |
| `-q, --quiet` | Suppress progress output | Verbose |

### Transcription Parameters (`run_transcription.sh`, `run_cpu_transcription.sh`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `audio_file` | Path to the audio file (required) | - |
| `-o, --output FILE` | Output transcript path | `[input_filename]_transcript.txt` |
| `-s, --speakers NUM` | Number of speakers | Auto-detect |

### Combined Pipeline Parameters (`transcribe_with_preprocessing.sh`)

Includes all parameters from both preprocessing and transcription, plus:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--skip-preprocessing` | Skip audio preprocessing | Preprocessing enabled |
| `--cpu-only` | Use CPU-only transcription | GPU if available |

## How the Preprocessing Works

The preprocessing pipeline applies several steps to enhance audio quality:

1. **Format conversion**:
   ```python
   convert_to_flac(input_file, output_file)
   ```
   Ensures consistent audio format with lossless compression

2. **Resampling**:
   ```python
   resample_audio(input_file, output_file, target_sr=16000)
   ```
   Standardises to 16kHz sampling rate for optimal Whisper performance

3. **Noise reduction**:
   ```python
   reduce_noise(input_file, output_file, strength=0.5)
   ```
   Uses spectral gating to reduce background noise while preserving speech

4. **Volume normalisation**:
   ```python
   normalize_volume(input_file, output_file, target_level=-18)
   ```
   Ensures consistent volume levels throughout the recording

5. **Dynamic range compression**:
   ```python
   apply_compression(input_file, output_file, threshold=-20, ratio=3)
   ```
   Makes quiet speech more audible and reduces volume differences

6. **Speech enhancement**:
   ```python
   enhance_speech(input_file, output_file)
   ```
   Applies EQ to enhance speech frequencies (1-3kHz range)

7. **Silence trimming**:
   ```python
   trim_silence(input_file, output_file)
   ```
   Removes long silences at the beginning and end

## How the Transcription Works

### GPU-Based Script (`transcript.py`)

The GPU script works in several distinct phases:

1. **Device selection and configuration**:
   - Checks for GPU availability and VRAM amount
   - Warns if GPU is being used for display (which reduces available VRAM)
   - Uses full precision (FP32) for maximum compatibility

2. **Custom prompting and term corrections**:
   ```python
   whisper_prompt = load_whisper_prompt(prompt_file)
   term_corrections = load_term_corrections(corrections_file)
   ```
   - Loads domain-specific prompts to guide Whisper
   - Configures technical term corrections

3. **Whisper model loading**:
   ```python
   model = whisper.load_model(model_size, device=device)
   ```
   - Loads the medium.en model, optimised for English transcription
   - Places the model on GPU for faster processing

4. **Audio transcription**:
   ```python
   transcription = model.transcribe(audio_file, word_timestamps=True, 
                                  initial_prompt=whisper_prompt)
   ```
   - Processes the audio file with Whisper
   - Uses custom prompt for domain-specific guidance
   - Records word-level timestamps for accurate alignment with speakers

5. **Technical term correction**:
   ```python
   for segment in transcription["segments"]:
       segment["text"] = apply_term_corrections(segment["text"], term_corrections)
   ```
   - Corrects domain-specific terminology (like "SISSVoc" or "SKOS")

6. **Memory cleanup**:
   ```python
   del model
   torch.cuda.empty_cache()
   gc.collect()
   ```
   - Clears GPU memory before loading diarisation

7. **Diarisation on CPU**:
   ```python
   diarisation_result = diarisation(audio_file, num_speakers=num_speakers)
   ```
   - Performs speaker identification using pyannote.audio

8. **Speaker assignment and segment merging**:
   - Combines transcription segments with speaker identities
   - Merges consecutive segments from the same speaker
   - Formats and saves the final transcript

### CPU-Based Script (`transcript_cpu.py`)

The CPU script follows a similar workflow but with key differences:

1. **Force CPU-only processing**:
   ```python
   os.environ["CUDA_VISIBLE_DEVICES"] = ""
   ```
   - Completely disables CUDA to ensure CPU-only operation

2. **CPU optimisation**:
   ```python
   os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
   os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
   os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
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

## Technical Term Correction

Both transcription scripts support technical term correction to improve accuracy with domain-specific terminology:

1. **Term correction file format**:
   ```
   # Format: incorrect_term|correct_term
   SysVoc|SISSVoc
   Sisk|SISSVoc
   scos|SKOS
   Research Recovery Australia|Research Vocabularies Australia
   ```

2. **Application logic**:
   ```python
   def apply_term_corrections(text, corrections):
       corrected_text = text
       for incorrect, correct in corrections.items():
           pattern = re.compile(r'\b' + re.escape(incorrect) + r'\b', re.IGNORECASE)
           corrected_text = pattern.sub(correct, corrected_text)
       return corrected_text
   ```

This allows automatic correction of misrecognised technical terms, acronyms, and proper names.

## Custom Prompting

The system supports custom prompting to guide Whisper transcription:

1. **Creating a prompt file**:
   - Create a text file with domain-specific context
   - Include key technical terms, names, and contexts
   - Specify transcription priorities

2. **Using the prompt**:
   ```python
   transcription = model.transcribe(
       audio_file,
       word_timestamps=True,
       initial_prompt=whisper_prompt
   )
   ```

This significantly improves accuracy for domain-specific content or specialised terminology.

## Performance Considerations

### Hardware Requirements

| Script | Minimum Hardware | Recommended Hardware |
|--------|-----------------|---------------------|
| preprocess_audio.sh | 2-core CPU, 4GB RAM | 4+ core CPU, 8GB+ RAM |
| transcript.py | 6GB VRAM GPU, 8GB RAM | 8GB+ VRAM GPU, 16GB RAM |
| transcript_cpu.py | 4-core CPU, 8GB RAM | 8+ core CPU, 16GB+ RAM |
| transcribe_with_preprocessing.sh | Same as above depending on mode | Same as above |

### Processing Time

| Recording Length | Preprocessing | CPU (large-v3) | GPU (medium.en) | Total (GPU+Preprocess) |
|-----------------|--------------|---------------|----------------|----------------------|
| 10 minutes | ~2-5 minutes | ~50-100 minutes | ~2-5 minutes | ~5-10 minutes |
| 1 hour | ~10-15 minutes | ~5-10 hours | ~15-30 minutes | ~25-45 minutes |

### Model Selection Guide

| Model | VRAM Required | Accuracy | Use When |
|-------|--------------|----------|----------|
| tiny.en | ~1GB | Basic | Limited GPU, simple audio |
| base.en | ~1-2GB | Good | Limited GPU, basic needs |
| small.en | ~2-3GB | Very Good | 4GB GPUs, general use |
| medium.en | ~5GB | Excellent | 6GB+ GPUs, standard needs |
| large-v3 | ~10GB | Superior | CPU only, maximum quality |

## Output Format

All scripts generate the same output format:

```
# Transcript of recording.mp3

[00:00:05] SPEAKER_00: This is the first speaker talking.

[00:00:10] SPEAKER_01: This is the second speaker responding.
```

## Troubleshooting

### Preprocessing Issues

1. **FFmpeg errors**:
   - Ensure FFmpeg is installed: `ffmpeg -version`
   - Install with: `sudo apt-get install ffmpeg` (Ubuntu/Debian)
   - Or use the provided setup script: `./setup_transcription.sh`

2. **Python errors**:
   - Ensure the environment is activated: `source ~/transcription-env/bin/activate`
   - Check for all dependencies: `pip list`
   - Reinstall if needed: `pip install librosa soundfile pydub`

3. **Permissions issues**:
   - Ensure scripts are executable: `chmod +x *.sh *.py`
   - Check output directory permissions

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

## Tips for Best Results

1. **Use appropriate preprocessing for your audio**:
   - Choose the right preset for your recording type
   - For interviews, use the "interview" preset (default)
   - For noisy environments, use the "noisy" preset
   - For music or singing, use the "music" preset

2. **Determine speaker count in advance**:
   - Using `-s` with the exact number of speakers improves diarisation accuracy

3. **Use appropriate model for your needs**:
   - For standard recordings, medium.en on GPU is usually sufficient
   - For difficult audio (accents, noise), use large-v3 on CPU

4. **Use domain-specific prompt and corrections**:
   - Create a custom whisper_prompt.txt with context about your domain
   - Add domain-specific terminology to term_corrections.txt

5. **Shell scripts vs direct Python calls**:
   - Always use the shell scripts when possible
   - They handle environment setup, memory management, and error reporting

6. **Ensure your HuggingFace token is set**:
   ```bash
   export HUGGINGFACE_TOKEN=your_token_here
   ```

## Advanced Customisation

### Changing Model Size

In `transcript.py`, modify:
```python
model_size = "medium.en"  # Options: "tiny.en", "base.en", "small.en", "medium.en"
```

In `transcript_cpu.py`, modify:
```python
model = whisper.load_model("large-v3")  # Options: "tiny", "base", "small", "medium", "large-v3"
```

### Adjusting Preprocessing Parameters

For custom noise reduction strength, modify in `audio_preprocess.py`:
```python
reduce_noise(input_file, output_file, strength=0.7)  # Increase from default 0.5
```

For EQ adjustments, modify in `audio_preprocess.py`:
```python
# Custom EQ for enhancing a different frequency range
command = [
    "ffmpeg", "-i", input_file,
    "-filter:a", "equalizer=f=200:width_type=h:width=100:g=-3, equalizer=f=2000:width_type=q:width=1:g=4",
    "-y",
    output_file
]
```

### Adjusting Merging Threshold

Both scripts merge consecutive segments from the same speaker with gaps less than 1 second:
```python
if segment["speaker"] == current["speaker"] and segment["start"] - current["end"] < 1.0:
```

Increase the value (1.0) to merge segments with larger gaps.

### Custom Output Format

Modify the timestamp and output format:
```python
timestamp = f"[{int(h):02d}:{int(m):02d}:{int(s):02d}]"
f.write(f"{timestamp} {segment['speaker']}: {segment['text']}\n\n")
```

## Future Developments

Based on the to-do lists and planning documents, future improvements will include:

1. **Migration to WhisperX**:
   - Significantly faster processing (4-10× faster than standard Whisper)
   - Better speaker diarisation integration
   - More accurate word-level timestamps
   - Better handling of overlapping speech

2. **Additional output formats**:
   - SRT for subtitling
   - JSON for programmatic use
   - Other structured formats

3. **Architectural improvements**:
   - Modular Python architecture
   - Comprehensive testing regime
   - Docker containerisation
   - CI/CD integration

4. **Enhanced features**:
   - Named speaker identification (replacing SPEAKER_XX with actual names)
   - Non-speech audio annotation (laughter, pauses, etc.)
   - Confidence indicators for uncertain transcriptions
   - Speech rate indicators
