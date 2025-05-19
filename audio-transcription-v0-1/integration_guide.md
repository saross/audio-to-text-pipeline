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
# Common CPU optimisations
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# GPU optimisations
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
   1. Audio file preprocessing: Convert, denoise, normalise, enhance
   2. Audio transcription: Using your existing Whisper scripts
   3. Speaker diarisation: Using your existing pyannote.audio setup

## Preset Configurations

The preprocessing pipeline includes preset configurations that you can use for different types of recordings:

| Preset | Description | Best For |
|--------|-------------|----------|
| interview | Balanced settings (default) | One-on-one interviews, general recordings |
| lecture | Prioritises speech clarity | Presentations, lectures, single speaker |
| noisy | Stronger noise reduction | Recordings in noisy environments |
| music | Careful processing for music | Content with music or singing |

Example usage:
```bash
# For a noisy environment recording
./preprocess_audio.sh recording.mp4 -p noisy

# For a lecture with the combined pipeline
./transcribe_with_preprocessing.sh lecture.mp4 -p lecture -s 1
```

## File Management

- **Input Files**: Original MP4 or other format files
- **Intermediate Files**: Processed FLAC files (saved in temp directory by default)
- **Output Files**: 
  - Processed audio: `[input_filename]_processed.flac`
  - Transcript: `[input_filename]_transcript.txt`

The combined pipeline (`transcribe_with_preprocessing.sh`) automatically manages temporary files, creating a temporary directory that is cleaned up when processing completes or if the script exits unexpectedly.

## Real-world Usage Examples

### Example 1: Process a Noisy Interview

```bash
# Process and transcribe a noisy interview with 2 speakers
./transcribe_with_preprocessing.sh interviews/noisy_interview.mp4 -p noisy -s 2 -o transcripts/interview_clean.txt
```

### Example 2: Preprocess Multiple Files for Later Transcription

```bash
# Batch process audio files with a for loop
for file in raw_audio/*.mp4; do
  filename=$(basename "$file")
  ./preprocess_audio.sh "$file" -o processed_audio/"${filename%.*}.flac" -p interview
done

# Later transcribe the processed files
for file in processed_audio/*.flac; do
  filename=$(basename "$file")
  ./run_transcription.sh "$file" -o transcripts/"${filename%.*}.txt"
done
```

### Example 3: High-quality Processing for Music Content

```bash
# For audio with music sections, use the music preset and CPU-only mode
./transcribe_with_preprocessing.sh concert_recording.mp4 -p music --cpu-only -o transcripts/concert.txt
```

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

6. **Temporary Directory Issues**: If the process fails and leaves temporary files:
   ```bash
   # List any temporary directories that might need cleaning
   ls -la /tmp/tmp*
   
   # Remove only those created by the script if necessary
   rm -rf /tmp/tmp*
   ```

## Advanced Integration

### Adding to Existing Workflows

You can integrate the preprocessing scripts into more complex workflows:

```bash
# Example: Record audio, preprocess, transcribe, and archive
record_audio.sh input.mp4
./preprocess_audio.sh input.mp4 -o processed/input.flac -p interview
./run_transcription.sh processed/input.flac -o transcripts/input.txt
archive_results.sh processed/input.flac transcripts/input.txt
```

### Customisation

All preprocessing options can be adjusted via command-line parameters. See each script's help text for details:

```bash
./preprocess_audio.sh --help
./transcribe_with_preprocessing.sh --help
```

For specific needs, you can selectively disable processing steps:

```bash
# Preserve music but reduce noise and normalise volume
./preprocess_audio.sh music_interview.mp4 -o processed.flac --no-compression --no-enhance
```

## Performance Considerations

- Preprocessing typically takes 10-20% of the total transcription time
- For very large files, preprocessing can significantly improve transcription accuracy
- GPU transcription with preprocessing is still much faster than CPU-only transcription
