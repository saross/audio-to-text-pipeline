#!/bin/bash
#
# Complete Audio Transcription Pipeline with Speaker Identification
#
# PURPOSE:
#     Runs the complete pipeline: transcription followed by speaker diarization,
#     then merges the results to create a speaker-labeled transcript.
#     This is a convenience wrapper that chains together the individual steps.
#
# FEATURES:
#     - Automatic transcription with whisper.cpp
#     - Speaker diarization with pyannote.audio
#     - Merged output with speaker labels
#     - Optional custom speaker names
#     - Progress tracking throughout
#
# REQUIREMENTS:
#     - Both Docker images built (whisper and diarization)
#     - HF_TOKEN environment variable set
#     - GPU recommended for performance
#
# USAGE:
#     ./transcribe_with_speakers.sh audio.flac [options]
#
# OPTIONS:
#     -o, --output FILE       Final output file
#     -m, --model MODEL       Whisper model (default: medium.en)
#     -n, --num-speakers INT  Number of speakers (auto-detect if not specified)
#     --speaker-names NAMES   Custom names (e.g., 'Alice,Bob')
#     --keep-intermediate     Keep intermediate transcript and diarization files
#     -v, --verbose          Show detailed output
#
# EXAMPLE:
#     export HF_TOKEN=your_token
#     ./transcribe_with_speakers.sh interview.flac -n 2 --speaker-names 'Host,Guest'
#
# AUTHOR: Audio transcription pipeline
# DATE: 2024

# Color codes for terminal output
GREEN='\033[0;32m'   # Success
BLUE='\033[0;34m'    # Information
RED='\033[0;31m'     # Errors
YELLOW='\033[1;33m'  # Warnings
NC='\033[0m'         # No color

# Get script directory for relative paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

print_usage() {
    echo "Usage: $0 <audio_file> [options]"
    echo ""
    echo "Complete transcription pipeline with speaker identification"
    echo ""
    echo "Options:"
    echo "  -o, --output FILE       Final output file (default: input_speakers.txt)"
    echo "  -m, --model MODEL       Whisper model (default: medium.en)"
    echo "  -n, --num-speakers INT  Number of speakers (default: auto-detect)"
    echo "  --speaker-names NAMES   Speaker names (e.g., 'Alice,Bob')"
    echo "  --keep-intermediate     Keep intermediate files"
    echo "  -v, --verbose           Verbose output"
    echo ""
    echo "Example:"
    echo "  export HF_TOKEN=your_token_here"
    echo "  $0 interview.flac -n 2 --speaker-names 'Interviewer,Guest'"
}

# Check dependencies
check_dependencies() {
    # Check HF_TOKEN
    if [ -z "$HF_TOKEN" ]; then
        echo -e "${RED}Error: HF_TOKEN not set${NC}"
        echo "Please: export HF_TOKEN=your_huggingface_token"
        exit 1
    fi
    
    # Check Docker images
    if ! docker image inspect whisper-cpp-cuda >/dev/null 2>&1; then
        echo -e "${RED}Error: Whisper Docker image not found${NC}"
        echo "Please run: $SCRIPT_DIR/run_whisper.sh --build"
        exit 1
    fi
    
    if ! docker image inspect diarization-pyannote >/dev/null 2>&1; then
        echo -e "${RED}Error: Diarization Docker image not found${NC}"
        echo "Please run: $SCRIPT_DIR/run_diarization.sh --build"
        exit 1
    fi
}

# Parse arguments
if [ "$1" == "--help" ] || [ "$1" == "-h" ] || [ -z "$1" ]; then
    print_usage
    exit 0
fi

INPUT_FILE="$1"
shift

# Check input file
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file not found: $INPUT_FILE${NC}"
    exit 1
fi

# Get base name for outputs
BASE_NAME=$(basename "$INPUT_FILE" | sed 's/\.[^.]*$//')
WORK_DIR=$(dirname "$INPUT_FILE")

# Defaults
OUTPUT_FILE="${BASE_NAME}_speakers.txt"
MODEL="medium.en"
NUM_SPEAKERS=""
SPEAKER_NAMES=""
KEEP_INTERMEDIATE=false
VERBOSE=""

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -n|--num-speakers)
            NUM_SPEAKERS="-n $2"
            shift 2
            ;;
        --speaker-names)
            SPEAKER_NAMES="--speaker-names \"$2\""
            shift 2
            ;;
        --keep-intermediate)
            KEEP_INTERMEDIATE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Check dependencies
check_dependencies

# Set intermediate file names
TRANSCRIPT_FILE="${WORK_DIR}/${BASE_NAME}_transcript.txt"
DIARIZATION_FILE="${WORK_DIR}/${BASE_NAME}_diarization.json"

echo -e "${BLUE}Starting speaker-aware transcription pipeline${NC}"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo ""

# Step 1: Transcription
echo -e "${BLUE}Step 1/3: Transcribing audio...${NC}"
"$SCRIPT_DIR/run_whisper.sh" "$INPUT_FILE" \
    -o "$TRANSCRIPT_FILE" \
    -m "$MODEL" \
    $VERBOSE

if [ $? -ne 0 ]; then
    echo -e "${RED}Transcription failed${NC}"
    exit 1
fi

# Step 2: Diarization
echo ""
echo -e "${BLUE}Step 2/3: Identifying speakers...${NC}"
"$SCRIPT_DIR/run_diarization.sh" diarize "$INPUT_FILE" \
    -o "$DIARIZATION_FILE" \
    $NUM_SPEAKERS

if [ $? -ne 0 ]; then
    echo -e "${RED}Diarization failed${NC}"
    exit 1
fi

# Step 3: Merge
echo ""
echo -e "${BLUE}Step 3/3: Merging transcript with speakers...${NC}"
eval "$SCRIPT_DIR/run_diarization.sh merge \"$TRANSCRIPT_FILE\" \"$DIARIZATION_FILE\" \
    -o \"$OUTPUT_FILE\" \
    $SPEAKER_NAMES"

if [ $? -ne 0 ]; then
    echo -e "${RED}Merge failed${NC}"
    exit 1
fi

# Cleanup intermediate files
if [ "$KEEP_INTERMEDIATE" = false ]; then
    rm -f "$TRANSCRIPT_FILE" "$DIARIZATION_FILE"
fi

echo ""
echo -e "${GREEN}✓ Pipeline complete!${NC}"
echo "Speaker-labeled transcript: $OUTPUT_FILE"

# Show summary
echo ""
echo -e "${BLUE}Summary:${NC}"
if [ -f "$OUTPUT_FILE" ]; then
    # Count unique speakers
    SPEAKERS=$(grep -E "^[A-Z_]+[0-9]*:" "$OUTPUT_FILE" | cut -d: -f1 | sort -u)
    NUM_DETECTED=$(echo "$SPEAKERS" | wc -l)
    echo "Detected speakers: $NUM_DETECTED"
    echo "$SPEAKERS" | while read speaker; do
        COUNT=$(grep -c "^$speaker:" "$OUTPUT_FILE")
        echo "  $speaker: $COUNT segments"
    done
fi