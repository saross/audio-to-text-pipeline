#!/bin/bash
#
# Docker-based Speaker Diarization Runner
#
# PURPOSE:
#     Provides containerized speaker diarization using pyannote.audio.
#     Identifies different speakers in audio files and can merge results
#     with existing transcripts to create speaker-labeled output.
#
# FEATURES:
#     - GPU acceleration (automatic detection)
#     - Progress tracking during diarization
#     - Multiple output formats (JSON, RTTM, simple text)
#     - Speaker name customization
#     - Fully containerized for reproducibility
#
# REQUIREMENTS:
#     - Docker with GPU support (optional but recommended)
#     - Hugging Face token (export HF_TOKEN=your_token)
#     - Built Docker image (run with --build first)
#
# AUTHOR: Audio transcription pipeline
# DATE: 2024

# Colors for output formatting
GREEN='\033[0;32m'   # Success messages
BLUE='\033[0;34m'    # Information messages
RED='\033[0;31m'     # Error messages
YELLOW='\033[1;33m'  # Warning messages
NC='\033[0m'         # No color (reset)

# Get script directory for relative paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Docker image name for diarization container
IMAGE_NAME="diarization-pyannote"

print_usage() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  diarize <audio_file>    Perform speaker diarization"
    echo "  merge <transcript> <diarization>  Merge diarization with transcript"
    echo "  --build                 Build the Docker image"
    echo "  --gpu-test              Test GPU access"
    echo ""
    echo "Diarization options:"
    echo "  -o, --output FILE       Output file"
    echo "  -n, --num-speakers INT  Number of speakers (default: auto)"
    echo "  --format FORMAT         Output format: json, rttm, simple (default: json)"
    echo "  --show-plot             Save diarization plot"
    echo ""
    echo "Merge options:"
    echo "  -o, --output FILE       Output file"
    echo "  --format FORMAT         Output format: text, json, srt (default: text)"
    echo "  --speaker-names NAMES   Comma-separated names (e.g., 'Alice,Bob')"
    echo ""
    echo "Environment:"
    echo "  HF_TOKEN                Hugging Face token (required)"
    echo ""
    echo "Examples:"
    echo "  export HF_TOKEN=your_token_here"
    echo "  $0 diarize interview.flac -o speakers.json"
    echo "  $0 merge transcript.txt speakers.json --speaker-names 'Dr. Smith,Patient'"
}

# Build function
build_image() {
    echo -e "${BLUE}Building diarization Docker image...${NC}"
    cd "$PROJECT_ROOT"
    
    # Check if scripts exist
    if [ ! -f "scripts/diarize_audio.py" ] || [ ! -f "scripts/merge_diarization.py" ]; then
        echo -e "${RED}Error: Diarization scripts not found in scripts/ directory${NC}"
        exit 1
    fi
    
    # Build image
    docker build -f docker/Dockerfile.diarization -t "$IMAGE_NAME" .
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Build complete!${NC}"
    else
        echo -e "${RED}Build failed!${NC}"
        exit 1
    fi
}

# GPU test function
test_gpu() {
    echo -e "${BLUE}Testing GPU access in diarization container...${NC}"
    
    # Check if image exists
    if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
        echo -e "${RED}Error: Docker image '$IMAGE_NAME' not found.${NC}"
        echo "Please run: $0 --build"
        exit 1
    fi
    
    # Test GPU access
    docker run --rm --gpus all "$IMAGE_NAME" -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('Running on CPU')
"
}

# Check HF token
check_hf_token() {
    if [ -z "$HF_TOKEN" ]; then
        echo -e "${RED}Error: HF_TOKEN environment variable not set${NC}"
        echo ""
        echo "Please set your Hugging Face token:"
        echo "  export HF_TOKEN=your_token_here"
        echo ""
        echo "Get token from: https://huggingface.co/settings/tokens"
        echo "Accept model at: https://huggingface.co/pyannote/speaker-diarization-3.1"
        exit 1
    fi
}

# Parse arguments
if [ "$1" == "--build" ]; then
    build_image
    exit 0
elif [ "$1" == "--gpu-test" ]; then
    test_gpu
    exit 0
elif [ "$1" == "--help" ] || [ "$1" == "-h" ] || [ -z "$1" ]; then
    print_usage
    exit 0
fi

# Check if Docker image exists
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo -e "${RED}Error: Docker image '$IMAGE_NAME' not found.${NC}"
    echo "Please run: $0 --build"
    exit 1
fi

# Main command processing
COMMAND="$1"
shift

case $COMMAND in
    diarize)
        # Check for input file
        if [ -z "$1" ]; then
            echo -e "${RED}Error: No input file specified${NC}"
            print_usage
            exit 1
        fi
        
        INPUT_FILE="$1"
        shift
        
        # Check if file exists
        if [ ! -f "$INPUT_FILE" ]; then
            echo -e "${RED}Error: Input file not found: $INPUT_FILE${NC}"
            exit 1
        fi
        
        # Check HF token
        check_hf_token
        
        # Get absolute paths
        INPUT_PATH=$(realpath "$INPUT_FILE")
        INPUT_DIR=$(dirname "$INPUT_PATH")
        INPUT_NAME=$(basename "$INPUT_PATH")
        
        # Default output
        OUTPUT_NAME="${INPUT_NAME%.*}_diarization.json"
        OUTPUT_DIR="$PWD"
        
        # Parse diarization options
        EXTRA_ARGS=""
        while [[ $# -gt 0 ]]; do
            case $1 in
                -o|--output)
                    if [[ "$2" == */* ]]; then
                        OUTPUT_DIR=$(dirname "$2")
                        OUTPUT_NAME=$(basename "$2")
                        mkdir -p "$OUTPUT_DIR"
                        OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
                    else
                        OUTPUT_NAME="$2"
                    fi
                    shift 2
                    ;;
                -n|--num-speakers)
                    EXTRA_ARGS="$EXTRA_ARGS -n $2"
                    shift 2
                    ;;
                --format)
                    EXTRA_ARGS="$EXTRA_ARGS --format $2"
                    shift 2
                    ;;
                --show-plot)
                    EXTRA_ARGS="$EXTRA_ARGS --show-plot"
                    shift
                    ;;
                *)
                    EXTRA_ARGS="$EXTRA_ARGS $1"
                    shift
                    ;;
            esac
        done
        
        # Run diarization
        echo -e "${BLUE}Starting speaker diarization...${NC}"
        echo "Input:  $INPUT_FILE"
        echo "Output: $OUTPUT_DIR/$OUTPUT_NAME"
        echo ""
        
        # Determine if GPU is available
        GPU_FLAG=""
        if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
            GPU_FLAG="--gpus all"
            echo "GPU detected, using CUDA acceleration"
        else
            echo "No GPU detected, using CPU"
        fi
        
        # Run container
        docker run --rm $GPU_FLAG \
            -v "$INPUT_DIR":/input:ro \
            -v "$OUTPUT_DIR":/output \
            -e HF_TOKEN="$HF_TOKEN" \
            "$IMAGE_NAME" \
            "/input/$INPUT_NAME" \
            -o "/output/$OUTPUT_NAME" \
            $EXTRA_ARGS
        
        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}✓ Diarization complete!${NC}"
            echo "Output saved to: $OUTPUT_DIR/$OUTPUT_NAME"
        else
            echo -e "${RED}✗ Diarization failed${NC}"
            exit 1
        fi
        ;;
        
    merge)
        # Check for input files
        if [ -z "$1" ] || [ -z "$2" ]; then
            echo -e "${RED}Error: Need transcript and diarization files${NC}"
            print_usage
            exit 1
        fi
        
        TRANSCRIPT_FILE="$1"
        DIARIZATION_FILE="$2"
        shift 2
        
        # Check if files exist
        if [ ! -f "$TRANSCRIPT_FILE" ]; then
            echo -e "${RED}Error: Transcript file not found: $TRANSCRIPT_FILE${NC}"
            exit 1
        fi
        
        if [ ! -f "$DIARIZATION_FILE" ]; then
            echo -e "${RED}Error: Diarization file not found: $DIARIZATION_FILE${NC}"
            exit 1
        fi
        
        # Get absolute paths
        TRANSCRIPT_PATH=$(realpath "$TRANSCRIPT_FILE")
        TRANSCRIPT_DIR=$(dirname "$TRANSCRIPT_PATH")
        TRANSCRIPT_NAME=$(basename "$TRANSCRIPT_PATH")
        
        DIARIZATION_PATH=$(realpath "$DIARIZATION_FILE")
        DIARIZATION_DIR=$(dirname "$DIARIZATION_PATH")
        DIARIZATION_NAME=$(basename "$DIARIZATION_PATH")
        
        # Default output
        OUTPUT_NAME="${TRANSCRIPT_NAME%.*}_speakers.txt"
        OUTPUT_DIR="$PWD"
        
        # Parse merge options
        EXTRA_ARGS=""
        while [[ $# -gt 0 ]]; do
            case $1 in
                -o|--output)
                    if [[ "$2" == */* ]]; then
                        OUTPUT_DIR=$(dirname "$2")
                        OUTPUT_NAME=$(basename "$2")
                        mkdir -p "$OUTPUT_DIR"
                        OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
                    else
                        OUTPUT_NAME="$2"
                    fi
                    shift 2
                    ;;
                --format)
                    EXTRA_ARGS="$EXTRA_ARGS --format $2"
                    shift 2
                    ;;
                --speaker-names)
                    EXTRA_ARGS="$EXTRA_ARGS --speaker-names \"$2\""
                    shift 2
                    ;;
                *)
                    EXTRA_ARGS="$EXTRA_ARGS $1"
                    shift
                    ;;
            esac
        done
        
        # Run merge
        echo -e "${BLUE}Merging diarization with transcript...${NC}"
        echo "Transcript:   $TRANSCRIPT_FILE"
        echo "Diarization:  $DIARIZATION_FILE"
        echo "Output:       $OUTPUT_DIR/$OUTPUT_NAME"
        echo ""
        
        # Need to mount both input directories
        MOUNT_ARGS="-v $TRANSCRIPT_DIR:/transcript:ro"
        if [ "$TRANSCRIPT_DIR" != "$DIARIZATION_DIR" ]; then
            MOUNT_ARGS="$MOUNT_ARGS -v $DIARIZATION_DIR:/diarization:ro"
            DIARIZATION_CONTAINER_PATH="/diarization/$DIARIZATION_NAME"
        else
            DIARIZATION_CONTAINER_PATH="/transcript/$DIARIZATION_NAME"
        fi
        
        # Run merge in container
        docker run --rm \
            $MOUNT_ARGS \
            -v "$OUTPUT_DIR":/output \
            --entrypoint python3 \
            "$IMAGE_NAME" \
            /app/merge_diarization.py \
            "/transcript/$TRANSCRIPT_NAME" \
            "$DIARIZATION_CONTAINER_PATH" \
            -o "/output/$OUTPUT_NAME" \
            $EXTRA_ARGS
        
        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}✓ Merge complete!${NC}"
            echo "Output saved to: $OUTPUT_DIR/$OUTPUT_NAME"
        else
            echo -e "${RED}✗ Merge failed${NC}"
            exit 1
        fi
        ;;
        
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        print_usage
        exit 1
        ;;
esac