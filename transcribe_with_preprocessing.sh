#!/bin/bash

# =============================================================================
# COMPLETE AUDIO TRANSCRIPTION PIPELINE WITH PREPROCESSING
# =============================================================================
#
# This script processes audio files to enhance quality, then transcribes them
# using Whisper and pyannote.audio for speaker diarisation.
#
# The workflow consists of two main phases:
# 1. Audio preprocessing (MP4 to FLAC, noise reduction, volume normalisation, etc.)
# 2. Transcription with speaker diarisation
#
# PREREQUISITES:
# - Python environment with required packages
# - FFmpeg installed and in PATH
# - A valid HUGGINGFACE_TOKEN environment variable for diarisation model access
#
# USAGE:
#   ./transcribe_with_preprocessing.sh [audio_file] [options]
#
# REQUIRED ARGUMENTS:
#   audio_file              Path to the audio file to process and transcribe
#
# OPTIONS:
#   -o, --output FILE       Output transcript file path
#                           Default: [input_filename]_transcript.txt in ./output/
#   -s, --speakers NUM      Number of speakers in the recording (if known)
#                           Default: Auto-detect
#   -p, --preset STRING     Audio preprocessing preset (default: interview)
#                           Available presets: interview, lecture, noisy, music
#   --skip-preprocessing    Skip audio preprocessing phase
#   --cpu-only              Use CPU for transcription (highest quality, but slower)
#                           Default: Use GPU if available
#   -q, --quiet             Suppress progress output
#
# AUDIO PREPROCESSING OPTIONS:
#   --no-flac               Skip FLAC conversion
#   --no-resample           Skip resampling to 16kHz
#   --no-noise-reduction    Skip noise reduction
#   --no-normalize          Skip volume normalisation
#   --no-compression        Skip dynamic range compression
#   --no-enhance            Skip speech enhancement
#   --no-trim               Skip silence trimming
#   --no-mono               Skip conversion to mono
#
# EXAMPLES:
#   # Basic usage:
#   ./transcribe_with_preprocessing.sh data/interview.mp4
#
#   # Specify output location:
#   ./transcribe_with_preprocessing.sh data/interview.mp4 -o results/interview-transcript.txt
#
#   # Specify number of speakers:
#   ./transcribe_with_preprocessing.sh data/interview.mp4 -s 2
#
#   # Use CPU-only mode:
#   ./transcribe_with_preprocessing.sh data/interview.mp4 --cpu-only
#
#   # Skip preprocessing:
#   ./transcribe_with_preprocessing.sh data/interview.flac --skip-preprocessing
#
# =============================================================================

# Activate transcription environment
source ~/transcription-env/bin/activate

# Set environment variables for optimal performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# GPU optimization if using GPU mode
if [ -z "$CPU_ONLY" ]; then
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
    export CUDA_LAUNCH_BLOCKING="1"
fi

# Check for required commands
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: FFmpeg is required but not found"
    echo "Please install FFmpeg: https://ffmpeg.org/download.html"
    exit 1
fi

# Check for required environment variables
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "ERROR: HUGGINGFACE_TOKEN environment variable is not set."
    echo "Please set it with: export HUGGINGFACE_TOKEN=your_token_here"
    exit 1
fi

# Set default values
OUTPUT_FILE=""
SPEAKERS=""
PRESET="interview"
SKIP_PREPROCESSING=""
CPU_ONLY=""
QUIET=""
PREPROCESS_ARGS=""

# Parse arguments
INPUT_FILE=""
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -s|--speakers)
            SPEAKERS="$2"
            shift 2
            ;;
        -p|--preset)
            PRESET="$2"
            PREPROCESS_ARGS="$PREPROCESS_ARGS -p $PRESET"
            shift 2
            ;;
        --skip-preprocessing)
            SKIP_PREPROCESSING="yes"
            shift
            ;;
        --cpu-only)
            CPU_ONLY="yes"
            shift
            ;;
        -q|--quiet)
            QUIET="--quiet"
            PREPROCESS_ARGS="$PREPROCESS_ARGS -q"
            shift
            ;;
        --no-flac|--no-resample|--no-noise-reduction|--no-normalize|--no-compression|--no-enhance|--no-trim|--no-mono)
            PREPROCESS_ARGS="$PREPROCESS_ARGS $1"
            shift
            ;;
        --help)
            # Print help
            sed -n '/^# ==============/,/^# ==============/p' "$0" | sed 's/^# //g'
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional arguments
set -- "${POSITIONAL_ARGS[@]}"

# Check for input file
if [ $# -eq 0 ]; then
    echo "ERROR: No input file specified"
    echo "Run with --help for usage information"
    exit 1
fi

INPUT_FILE="$1"

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Set output file if not specified
if [ -z "$OUTPUT_FILE" ]; then
    # Get the base name without extension
    BASE_NAME="${INPUT_FILE%.*}"
    OUTPUT_FILE="${BASE_NAME}_transcript.txt"
fi

# Create output directory if needed
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
if [ ! -d "$OUTPUT_DIR" ] && [ "$OUTPUT_DIR" != "." ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Create temporary directory for intermediate files
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# Print summary of processing
if [ -z "$QUIET" ]; then
    echo "======================================================================"
    echo "COMPLETE AUDIO TRANSCRIPTION PIPELINE"
    echo "======================================================================"
    echo "Input file: $INPUT_FILE"
    echo "Output transcript: $OUTPUT_FILE"
    echo "Preprocessing: $([ -z "$SKIP_PREPROCESSING" ] && echo "Yes (preset: $PRESET)" || echo "No")"
    echo "Transcription mode: $([ -z "$CPU_ONLY" ] && echo "GPU (if available)" || echo "CPU-only")"
    echo "Speaker count: $([ -z "$SPEAKERS" ] && echo "Auto-detect" || echo "$SPEAKERS")"
    echo "======================================================================"
fi

# Record start time
START_TIME=$(date +%s)

# PHASE 1: Audio Preprocessing
if [ -z "$SKIP_PREPROCESSING" ]; then
    if [ -z "$QUIET" ]; then
        echo ""
        echo "PHASE 1: Audio Preprocessing"
        echo "-------------------------------------------------------------------"
    fi
    
    # Generate intermediate file path
    PROCESSED_AUDIO="$TEMP_DIR/processed_audio.flac"
    
    # Run preprocessing script
    ./preprocess_audio.sh "$INPUT_FILE" -o "$PROCESSED_AUDIO" $PREPROCESS_ARGS
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Audio preprocessing failed"
        exit 1
    fi
    
    # Use processed audio file for transcription
    TRANSCRIPTION_INPUT="$PROCESSED_AUDIO"
else
    # Use original input file for transcription
    TRANSCRIPTION_INPUT="$INPUT_FILE"
fi

# PHASE 2: Transcription with Speaker Diarisation
if [ -z "$QUIET" ]; then
    echo ""
    echo "PHASE 2: Transcription with Speaker Diarisation"
    echo "-------------------------------------------------------------------"
fi

# Prepare transcription command arguments
TRANSCRIBE_ARGS=""
if [ -n "$OUTPUT_FILE" ]; then
    TRANSCRIBE_ARGS="$TRANSCRIBE_ARGS -o $OUTPUT_FILE"
fi
if [ -n "$SPEAKERS" ]; then
    TRANSCRIBE_ARGS="$TRANSCRIBE_ARGS -s $SPEAKERS"
fi

# Run transcription script based on mode
if [ -n "$CPU_ONLY" ]; then
    # CPU-only mode
    ./run_cpu_transcription.sh "$TRANSCRIPTION_INPUT" $TRANSCRIBE_ARGS
else
    # GPU mode (if available)
    ./run_transcription.sh "$TRANSCRIPTION_INPUT" $TRANSCRIBE_ARGS
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Transcription failed"
    exit 1
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))

if [ -z "$QUIET" ]; then
    echo ""
    echo "======================================================================"
    echo "Complete pipeline finished successfully!"
    echo "Total processing time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "Transcript saved to: $OUTPUT_FILE"
    echo "======================================================================"
fi

exit 0
