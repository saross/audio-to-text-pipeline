#!/bin/bash

# =============================================================================
# AUDIO TRANSCRIPTION PIPELINE
# =============================================================================
#
# This script runs audio transcription optimised for 6GB VRAM GPUs.
# Uses faster-whisper (CTranslate2) for efficient memory usage and 4x faster performance.
#
# PREREQUISITES:
# - Python environment with faster-whisper installed
# - NVIDIA GPU with 6GB+ VRAM (recommended) or CPU fallback
#
# USAGE:
#   ./transcribe.sh [audio_file] [options]
#
# REQUIRED ARGUMENTS:
#   audio_file              Path to the audio file to transcribe
#
# OPTIONS:
#   -o, --output FILE       Output transcript file path
#                           Default: [input_filename]_transcript.txt
#   -m, --model MODEL       Model size: tiny, small, medium, large-v3
#                           Default: medium
#   -l, --language LANG     Language code (default: en)
#
# EXAMPLES:
#   # Basic usage with medium model:
#   ./transcribe.sh data/interview.mp3
#
#   # Use large-v3 model (fits in 6GB with faster-whisper!):
#   ./transcribe.sh data/interview.mp3 -m large-v3
#
#   # Specify output location:
#   ./transcribe.sh data/interview.mp3 -o results/interview-transcript.txt
#
# =============================================================================

# Activate virtual environment
source ~/transcription-env/bin/activate

# ===== HARDWARE OPTIMISATION =====
echo "Setting up optimised environment for transcription..."

# GPU memory optimisation
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

# CPU optimisation for fallback operations
CPU_THREADS=4  # Default for 4-core system

# Get CPU core count if possible
if command -v nproc &> /dev/null; then
    TOTAL_CORES=$(nproc)
    # Use 75% of cores (rounded down) but minimum 4, maximum 10
    CALCULATED_THREADS=$(( TOTAL_CORES * 3 / 4 ))
    CPU_THREADS=$(( CALCULATED_THREADS < 4 ? 4 : (CALCULATED_THREADS > 10 ? 10 : CALCULATED_THREADS) ))
    echo "Detected $TOTAL_CORES CPU cores, using $CPU_THREADS threads for CPU operations"
else
    echo "Using default thread count: $CPU_THREADS"
fi

# Set threading environment variables
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"

# Memory optimisation
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# General optimisation
export TOKENIZERS_PARALLELISM="false"  # Prevent tokeniser warnings
export PYTHONOPTIMISE="1"  # Optimise Python

# ===== CREATE OUTPUT DIRECTORY =====
# Create output directory if specified
if [[ "$*" =~ -o\ ([^\ ]+) || "$*" =~ --output\ ([^\ ]+) ]]; then
    OUTPUT_DIR=$(dirname "${BASH_REMATCH[1]}")
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
else
    # Default output directory
    echo "Creating default output directory"
    mkdir -p output
fi

# ===== CLEAR GPU MEMORY =====
echo "Clearing GPU memory..."
python -c "
import gc
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f'GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB')
    else:
        print('No CUDA GPU available, using CPU')
except Exception as e:
    print(f'Could not clear GPU memory: {e}')
gc.collect()
" 2>/dev/null || true

# ===== CHECK TRANSCRIPTION ENGINE =====
echo "Checking transcription engine..."
python -c "
try:
    from faster_whisper import WhisperModel
    print('✅ Transcription engine ready')
except ImportError:
    print('❌ ERROR: faster-whisper not installed!')
    print('Please install with: pip install faster-whisper')
    exit(1)
" || exit 1

# ===== RUN TRANSCRIPTION =====
echo "Starting audio transcription..."

# Record start time
START_TIME=$(date +%s)

# Run with basic error handling
if python transcribe.py "$@"; then
    # Calculate elapsed time
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(( (ELAPSED % 3600) / 60 ))
    SECONDS=$((ELAPSED % 60))
    
    echo "---------------------------------------------------"
    echo "Audio transcription completed successfully!"
    echo "Total processing time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "---------------------------------------------------"
else
    echo "---------------------------------------------------"
    echo "ERROR: Audio transcription failed."
    echo "Check the error messages above for details."
    echo "---------------------------------------------------"
    exit 1
fi