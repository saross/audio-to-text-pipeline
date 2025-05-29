# Whisper.cpp GPU Setup Instructions

## Quick Start

### 1. Create project structure
```bash
# Create and enter project directory
mkdir ~/whisper-transcription && cd ~/whisper-transcription

# Create directory structure
mkdir -p {config,docker,scripts,data/{input,processed},results/{raw,corrected,final},models,docs,logs}
```

### 2. Save the provided files
- Save `Dockerfile.whisper-cpp` to `docker/` directory
- Save `run_whisper.sh` to `scripts/` directory and make executable: `chmod +x scripts/run_whisper.sh`
- Save `apply_corrections.py` to `scripts/` directory and make executable: `chmod +x scripts/apply_corrections.py`

### 3. Copy your configuration files
```bash
# Copy from your existing project
cp ~/Code/audio-transcription/transcription-pipeline/config/whisper_prompt.txt config/
cp ~/Code/audio-transcription/transcription-pipeline/config/term_corrections.txt config/
```

### 4. Build the Docker image
```bash
./scripts/run_whisper.sh --build
```

This will:
- Download NVIDIA CUDA base image
- Compile whisper.cpp with GPU support
- Download models (tiny, base, small, medium, large-v3)
- Create quantized large-v3 model for 6GB GPUs
- Takes about 15-20 minutes

### 5. Test GPU access
```bash
./scripts/run_whisper.sh --gpu-test
```

### 6. Transcribe your audio
```bash
# Basic usage
./scripts/run_whisper.sh data/input/interview-2.flac

# With prompt and model selection
./scripts/run_whisper.sh data/input/interview-2.flac \
  -o results/raw/interview-2.txt \
  -m medium.en \
  -p config/whisper_prompt.txt

# Using quantized large model
./scripts/run_whisper.sh data/input/interview-2.flac \
  -o results/raw/interview-2.txt \
  -m large-v3-q5_0 \
  -p config/whisper_prompt.txt
```

### 7. Apply term corrections
```bash
python3 scripts/apply_corrections.py \
  results/raw/interview-2.txt \
  -o results/corrected/interview-2-corrected.txt \
  -c config/term_corrections.txt
```

## Project Structure

```
whisper-transcription/
├── config/
│   ├── whisper_prompt.txt         # Your technical prompt
│   └── term_corrections.txt       # Post-processing corrections
├── docker/
│   └── Dockerfile.whisper-cpp     # Docker build file
├── scripts/
│   ├── run_whisper.sh            # Main runner script
│   ├── apply_corrections.py      # Post-processing
│   └── batch_process.sh          # Batch processing
├── data/
│   ├── raw/                      # Your audio files
│   └── processed/                # Preprocessed audio
├── results/
│   ├── raw/                      # Direct whisper output
│   ├── corrected/                # After corrections
│   └── final/                    # Final formatted
├── models/                       # Model cache
├── docs/                         # Documentation
└── logs/                         # Processing logs
```

## Expected Performance

On your RTX 3050 (6GB):

| Model | Speed | Memory | Quality |
|-------|-------|---------|---------|
| tiny.en | ~2-5 min | ~1GB | Basic |
| base.en | ~3-7 min | ~1.5GB | Good |
| small.en | ~5-10 min | ~2GB | Very Good |
| medium.en | ~10-20 min | ~3GB | Excellent |
| large-v3-q5_0 | ~15-30 min | ~3.5GB | Best |

## Command Line Options

```bash
./scripts/run_whisper.sh <input_audio> [options]

Options:
  -o, --output FILE    Output file (default: input_name.txt)
  -m, --model MODEL    Model to use (tiny.en, base.en, small.en, medium.en, large-v3-q5_0)
                       Default: medium.en
  -p, --prompt FILE    Prompt file to guide transcription
  --build              Build the Docker image
  --gpu-test           Test GPU access
```

## Batch Processing

Create a batch processing script in `scripts/batch_process.sh`:

```bash
#!/bin/bash
# Process all interviews with prompt and corrections

MODEL=${1:-medium.en}

for f in data/input/*.flac; do
    if [ -f "$f" ]; then
        basename=$(basename "$f" .flac)
        echo "Processing $basename with $MODEL model..."
        
        # Transcribe with prompt
        ./scripts/run_whisper.sh "$f" \
            -o "results/raw/${basename}.txt" \
            -m "$MODEL" \
            -p config/whisper_prompt.txt
        
        # Apply corrections
        python3 scripts/apply_corrections.py \
            "results/raw/${basename}.txt" \
            -o "results/corrected/${basename}-corrected.txt" \
            -c config/term_corrections.txt
    fi
done
```

Make it executable: `chmod +x scripts/batch_process.sh`

Run all interviews: `./scripts/batch_process.sh large-v3-q5_0`

## Using Prompts

The whisper_prompt.txt file helps guide the model to recognise technical terms. Your 5570-character prompt with terms like SKOS, SISSVoc, ARDC, Kurrawong, etc., will significantly improve accuracy.

Example with prompt:
```bash
./scripts/run_whisper.sh data/input/interview.flac \
  -p config/whisper_prompt.txt \
  -m large-v3-q5_0
```

## Key Advantages

1. **No Python dependency issues** - Pure C++ implementation
2. **Excellent GPU memory management** - Quantization allows large models on 6GB
3. **Fast** - Optimised CUDA kernels
4. **Prompt support** - Guides technical term recognition
5. **Simple** - Just works

## Troubleshooting

### If build fails
- Check Docker GPU support: `docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi`
- Ensure NVIDIA container toolkit is installed
- Check disk space (need ~25GB during build)

### If GPU not detected
- The image uses CUDA 12.2 which should work with your drivers
- Try: `docker run --rm --gpus all whisper-cpp-gpu nvidia-smi`

### If out of memory
- Use quantized large model (large-v3-q5_0) instead of medium
- Close other GPU applications
- Monitor with `nvidia-smi`

### If transcription quality is poor
- Use the prompt file (`-p config/whisper_prompt.txt`)
- Try large-v3-q5_0 model for best quality
- Apply term corrections after transcription

## Notes on Speaker Diarisation

whisper.cpp doesn't include speaker diarisation. Options:
1. Use your working CPU pipeline for diarisation
2. Post-process with pyannote.audio separately
3. Consider WhisperX or similar tools that combine both

For interview transcripts, you might want to:
1. Get high-quality transcription with whisper.cpp + GPU
2. Run speaker diarisation separately
3. Combine the results

## Recommended Workflow

1. **Preprocess audio** (if needed): Ensure files are 16kHz mono FLAC
2. **Transcribe with large model**: `./scripts/run_whisper.sh input.flac -m large-v3-q5_0 -p config/whisper_prompt.txt`
3. **Apply corrections**: `python3 scripts/apply_corrections.py transcript.txt`
4. **Add speakers**: Use separate diarisation tool if needed

This approach gives you the best transcription quality while leveraging your GPU effectively.