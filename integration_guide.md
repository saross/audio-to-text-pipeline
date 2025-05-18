# Audio Preprocessing Integration Guide

## Environment Setup

The audio preprocessing scripts are designed to work seamlessly with your existing transcription environment. They use the same Python virtual environment and dependencies as your transcription scripts.

## Environment Details

- **Python Version**: 3.8+ (3.12 as per your configuration)
- **Virtual Environment Path**: ~/transcription-env
- **Key Libraries**:
  - openai-whisper (for speech recognition)
  - torch (machine learning framework)
  - pyannote.audio (for speaker diarisation)
  - pydub, librosa, soundfile (audio processing)
  - numpy, scipy (numerical operations)
  - tqdm (progress bars)

## Environment Variables

The scripts use the following environment variables:

```bash
# Common CPU optimizations
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# GPU optimizations
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
export CUDA_LAUNCH_BLOCKING="1"

# Authentication for pyannote.audio models
export HUGGINGFACE_TOKEN="your_token_here"
```

## Integration Instructions

1. **No Installation Needed**: Since you already have the environment set up, there's no need to create a new one. The audio preprocessing scripts will use your existing environment.

2. **Activating the Environment**: All shell scripts automatically activate the `~/transcription-env` environment.

3. **Using the Preprocessing Pipeline**:
   - Standalone: `./preprocess_audio.sh your_file.mp4`
   - With transcription: `./transcribe_with_preprocessing.sh your_file.mp4`

4. **Processing Order**:
   1. Audio file preprocessing: Convert, denoise, normalize, enhance
   2. Audio transcription: Using your existing Whisper scripts
   3. Speaker diarisation: Using your existing pyannote.audio setup

## File Management

- **Input Files**: Original MP4 or other format files
- **Intermediate Files**: Processed FLAC files (saved in temp directory by default)
- **Output Files**: 
  - Processed audio: `[input_filename]_processed.flac`
  - Transcript: `[input_filename]_transcript.txt`

## Troubleshooting

If you encounter any issues:

1. **Environment Activation**: Ensure the environment is activated with:
   ```bash
   source ~/transcription-env/bin/activate
   ```

2. **Dependencies**: Verify all required packages are installed:
   ```bash
   pip list | grep -E 'whisper|torch|pyannote|librosa|soundfile|pydub|numpy|scipy'
   ```

3. **FFmpeg**: Ensure FFmpeg is installed and in your PATH:
   ```bash
   ffmpeg -version
   ```

4. **Permissions**: Make sure all scripts are executable:
   ```bash
   chmod +x *.sh *.py
   ```

5. **Hugging Face Token**: Ensure your HUGGINGFACE_TOKEN is set:
   ```bash
   echo $HUGGINGFACE_TOKEN
   ```

## Customization

All preprocessing options can be adjusted via command-line parameters. See each script's help text for details:

```bash
./preprocess_audio.sh --help
./transcribe_with_preprocessing.sh --help
```