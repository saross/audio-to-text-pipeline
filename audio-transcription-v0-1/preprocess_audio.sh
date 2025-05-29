#!/bin/bash

# =============================================================================
# AUDIO PRE-PROCESSING PIPELINE FOR TRANSCRIPTION
# =============================================================================
#
# This script enhances audio quality for better transcription and diarisation
# by applying a series of processing steps:
#
# 1. MP4 to FLAC conversion
# 2. Sample rate standardisation (16kHz)
# 3. Stereo to mono conversion
# 4. Background noise reduction
# 5. Volume normalisation
# 6. Dynamic range compression
# 7. Speech enhancement (EQ)
# 8. Silence trimming
#
# PREREQUISITES:
# - Python environment with librosa, soundfile, numpy, scipy
# - FFmpeg installed and in PATH
#
# USAGE:
#   ./preprocess_audio.sh [audio_file] [options]
#
# REQUIRED ARGUMENTS:
#   audio_file              Path to the audio file to process
#
# OPTIONS:
#   -o, --output FILE       Output file path (default: [input]_processed.flac)
#   -p, --preset STRING     Preset configuration (default: interview)
#                           Available presets: interview, lecture, noisy, music
#   --diagnose              Only run diagnostics on the audio file
#   --no-flac               Skip FLAC conversion
#   --no-resample           Skip resampling to 16kHz
#   --no-noise-reduction    Skip noise reduction
#   --no-normalize          Skip volume normalisation
#   --no-compression        Skip dynamic range compression
#   --no-enhance            Skip speech enhancement
#   --no-trim               Skip silence trimming
#   --no-mono               Skip conversion to mono
#   -q, --quiet             Suppress progress output
#
# EXAMPLES:
#   # Basic usage:
#   ./preprocess_audio.sh data/interview.mp4
#
#   # Diagnose a problematic file:
#   ./preprocess_audio.sh data/interview.mp4 --diagnose
#
#   # Specify output location:
#   ./preprocess_audio.sh data/interview.mp4 -o cleaned/interview.flac
#
#   # Use a preset:
#   ./preprocess_audio.sh data/interview.mp4 -p noisy
#
#   # Custom options:
#   ./preprocess_audio.sh data/lecture.mp4 --no-noise-reduction --no-mono
#
# =============================================================================

# Activate transcription environment
source ~/transcription-env/bin/activate

# Set environment variables for optimal performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Check for required commands
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: FFmpeg is required but not found"
    echo "Please install FFmpeg: https://ffmpeg.org/download.html"
    exit 1
fi

if ! command -v ffprobe &> /dev/null; then
    echo "ERROR: FFprobe is required but not found"
    echo "Please install FFmpeg: https://ffmpeg.org/download.html"
    exit 1
fi

# Set default values
OUTPUT_FILE=""
PRESET="interview"
DIAGNOSE_ONLY=""
SKIP_FLAC=""
SKIP_RESAMPLE=""
SKIP_NOISE_REDUCTION=""
SKIP_NORMALIZE=""
SKIP_COMPRESSION=""
SKIP_ENHANCE=""
SKIP_TRIM=""
SKIP_MONO=""
QUIET=""

# Parse arguments
INPUT_FILE=""
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -p|--preset)
            PRESET="$2"
            shift 2
            ;;
        --diagnose)
            DIAGNOSE_ONLY="--diagnose"
            shift
            ;;
        --no-flac)
            SKIP_FLAC="--no-flac"
            shift
            ;;
        --no-resample)
            SKIP_RESAMPLE="--no-resample"
            shift
            ;;
        --no-noise-reduction)
            SKIP_NOISE_REDUCTION="--no-noise-reduction"
            shift
            ;;
        --no-normalize)
            SKIP_NORMALIZE="--no-normalize"
            shift
            ;;
        --no-compression)
            SKIP_COMPRESSION="--no-compression"
            shift
            ;;
        --no-enhance)
            SKIP_ENHANCE="--no-enhance"
            shift
            ;;
        --no-trim)
            SKIP_TRIM="--no-trim"
            shift
            ;;
        --no-mono)
            SKIP_MONO="--no-mono"
            shift
            ;;
        -q|--quiet)
            QUIET="--quiet"
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

# If diagnose-only mode, run diagnostics and exit
if [ -n "$DIAGNOSE_ONLY" ]; then
    echo "======================================================================"
    echo "AUDIO FILE DIAGNOSTICS"
    echo "======================================================================"
    echo "File: $INPUT_FILE"
    echo ""
    
    # Run Python script in diagnose mode
    python audio_preprocess.py "$INPUT_FILE" --diagnose
    exit $?
fi

# Apply presets (override default options)
case $PRESET in
    interview)
        # Default balanced settings (all options enabled)
        ;;
    lecture)
        # For lectures, prioritise speech clarity
        SKIP_NOISE_REDUCTION=""
        SKIP_ENHANCE=""
        ;;
    noisy)
        # For very noisy environments, stronger noise reduction
        SKIP_NOISE_REDUCTION=""
        SKIP_ENHANCE=""
        ;;
    music)
        # For audio with music, be careful with noise reduction and compression
        SKIP_NOISE_REDUCTION="--no-noise-reduction"
        SKIP_COMPRESSION="--no-compression"
        ;;
    *)
        echo "WARNING: Unknown preset '$PRESET', using default settings"
        ;;
esac

# Set output file if not specified
if [ -z "$OUTPUT_FILE" ]; then
    # Get the base name without extension
    BASE_NAME="${INPUT_FILE%.*}"
    OUTPUT_FILE="${BASE_NAME}_processed.flac"
fi

# Create output directory if needed
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
if [ ! -d "$OUTPUT_DIR" ] && [ "$OUTPUT_DIR" != "." ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Print summary of processing
if [ -z "$QUIET" ]; then
    echo "======================================================================"
    echo "AUDIO PRE-PROCESSING PIPELINE"
    echo "======================================================================"
    echo "Input file: $INPUT_FILE"
    echo "Output file: $OUTPUT_FILE"
    echo "Preset: $PRESET"
    echo ""
    echo "Processing steps:"
    echo "- Convert to FLAC: $([ -z "$SKIP_FLAC" ] && echo "Yes" || echo "No")"
    echo "- Resample to 16kHz: $([ -z "$SKIP_RESAMPLE" ] && echo "Yes" || echo "No")"
    echo "- Convert to mono: $([ -z "$SKIP_MONO" ] && echo "Yes" || echo "No")"
    echo "- Noise reduction: $([ -z "$SKIP_NOISE_REDUCTION" ] && echo "Yes" || echo "No")"
    echo "- Volume normalisation: $([ -z "$SKIP_NORMALIZE" ] && echo "Yes" || echo "No")"
    echo "- Dynamic compression: $([ -z "$SKIP_COMPRESSION" ] && echo "Yes" || echo "No")"
    echo "- Speech enhancement: $([ -z "$SKIP_ENHANCE" ] && echo "Yes" || echo "No")"
    echo "- Silence trimming: $([ -z "$SKIP_TRIM" ] && echo "Yes" || echo "No")"
    echo "======================================================================"
fi

# Record start time
START_TIME=$(date +%s)

# Run the Python script with all the options
python audio_preprocess.py \
    "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    $SKIP_FLAC \
    $SKIP_RESAMPLE \
    $SKIP_NOISE_REDUCTION \
    $SKIP_NORMALIZE \
    $SKIP_COMPRESSION \
    $SKIP_ENHANCE \
    $SKIP_TRIM \
    $SKIP_MONO \
    $QUIET

# Check if processing was successful
if [ $? -eq 0 ]; then
    # Calculate elapsed time
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))
    
    if [ -z "$QUIET" ]; then
        echo "======================================================================"
        echo "Audio processing completed successfully!"
        echo "Processing time: ${MINUTES}m ${SECONDS}s"
        echo "Output saved to: $OUTPUT_FILE"
        echo "======================================================================"
    fi
    
    # Check file size reduction/increase
    if [ -z "$QUIET" ] && [ -f "$OUTPUT_FILE" ]; then
        INPUT_SIZE=$(du -h "$INPUT_FILE" | cut -f1)
        OUTPUT_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo "Original size: $INPUT_SIZE"
        echo "Processed size: $OUTPUT_SIZE"
    fi
else
    echo "ERROR: Audio processing failed"
    
    # Suggest running diagnostics
    echo ""
    echo "TROUBLESHOOTING SUGGESTIONS:"
    echo "1. Run diagnostics: ./preprocess_audio.sh \"$INPUT_FILE\" --diagnose"
    echo "2. Try converting just to FLAC: ./preprocess_audio.sh \"$INPUT_FILE\" --no-resample --no-noise-reduction --no-normalize --no-compression --no-enhance --no-trim --no-mono"
    echo "3. Test with a smaller audio file first"
    echo "4. Check available disk space and RAM"
    
    exit 1
fi