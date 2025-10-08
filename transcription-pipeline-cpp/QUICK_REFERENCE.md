# Quick Reference Guide

## Common Tasks

### Process a New Interview Recording

```bash
# 1. Preprocess MP4 to FLAC (if needed)
cd audio-preprocessing
./process_audio.sh convert ../transcription-pipeline-cpp/data/raw/interview.mp4 \
    ../transcription-pipeline-cpp/data/processed/interview.flac

# 2. Go to main pipeline directory
cd ../transcription-pipeline-cpp

# 3. Set Hugging Face token (once per session)
export HF_TOKEN=your_token_here

# 4. Run complete pipeline with speakers
./scripts/transcribe_with_speakers.sh data/processed/interview.flac \
    -n 2 \
    --speaker-names "Interviewer,Participant" \
    -o results/interview_final.txt
```

### Transcribe Without Diarization

```bash
# Quick transcription only
./scripts/run_whisper.sh data/processed/interview.flac \
    -o results/interview_transcript.txt
```

### Add Speakers to Existing Transcript

```bash
# 1. Run diarization
./scripts/run_diarization.sh diarize data/processed/interview.flac \
    -o results/interview_diarization.json

# 2. Merge with transcript
./scripts/run_diarization.sh merge results/interview_transcript.txt \
    results/interview_diarization.json \
    --speaker-names "Dr. Smith,Patient"
```

### Batch Process Multiple Files

```bash
# In audio-preprocessing directory
./batch_process_transcription_data.sh

# Or manually
for file in data/raw/*.mp4; do
    basename=$(basename "$file" .mp4)
    ./scripts/run_whisper.sh "data/processed/${basename}.flac" \
        -o "results/${basename}.txt"
done
```

## Command Options

### Preprocessing Options
- `--remove-silence` - Remove silence gaps > 0.5s
- `-v, --verbose` - Show detailed output

### Transcription Options
- `-m, --model` - Model size: tiny.en, base.en, small.en, medium.en
- `-p, --prompt` - Custom prompt file
- `-v, --verbose` - Show whisper output
- `--no-corrections` - Skip automatic term corrections

### Diarization Options
- `-n, --num-speakers` - Specify number of speakers
- `--format` - Output format: json, rttm, simple
- `--min-speakers` - Minimum speakers (default: 1)
- `--max-speakers` - Maximum speakers (default: 10)

## File Locations

### Input
- Raw MP4: `data/raw/`
- Processed FLAC: `data/processed/`

### Configuration
- Term corrections: `config/term_corrections.txt`
- Whisper prompt: `config/whisper_prompt.txt`

### Output
- Transcripts: `results/`
- Diarization: `results/*_diarization.json`
- Final output: `results/*_speakers.txt`

## Troubleshooting

### Check GPU
```bash
nvidia-smi
./scripts/run_whisper.sh --gpu-test
./scripts/run_diarization.sh --gpu-test
```

### Rebuild Images
```bash
./scripts/run_whisper.sh --build
./scripts/run_diarization.sh --build
```

### Monitor Progress
- Transcription: Shows percentage (45%) and time
- Diarization: Shows progress bar with ETA

### Common Issues

**"HF_TOKEN not set"**
```bash
export HF_TOKEN=your_huggingface_token
```

**"Docker image not found"**
```bash
./scripts/run_whisper.sh --build
./scripts/run_diarization.sh --build
```

**Out of GPU memory**
- Close other GPU applications
- Process files one at a time
- Use CPU mode (slower but works)

## Performance Tips

1. **Use GPU** - 5-10x faster than CPU
2. **Preprocess first** - Optimized FLAC files process faster
3. **Batch at night** - Diarization takes 10-15 min/hour of audio
4. **Monitor progress** - Check percentage to estimate completion

## Model Selection

- `tiny.en` - Fastest, lowest accuracy (~39 WER)
- `base.en` - Fast, reasonable accuracy (~30 WER)
- `small.en` - Balanced speed/accuracy (~21 WER)
- `medium.en` - Best accuracy, slower (~17 WER)

WER = Word Error Rate (lower is better)