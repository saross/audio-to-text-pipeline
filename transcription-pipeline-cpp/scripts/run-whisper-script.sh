#!/bin/bash

# Simple whisper.cpp Docker runner

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_usage() {
    echo "Usage: $0 <input_audio> [options]"
    echo "Options:"
    echo "  -o, --output FILE    Output file (default: input_name.txt)"
    echo "  -m, --model MODEL    Model to use (tiny.en, base.en, small.en, medium.en, large-v3-q5_0)"
    echo "                       Default: medium.en"
    echo "  -p, --prompt FILE    Prompt file to guide transcription"
    echo "  --build              Build the Docker image"
    echo "  --gpu-test           Test GPU access"
    echo ""
    echo "Examples:"
    echo "  $0 interview.flac"
    echo "  $0 interview.flac -o transcript.txt -m small.en"
    echo "  $0 interview.flac -p whisper_prompt.txt"
    echo "  $0 --build"
}

# Build function
build_image() {
    echo -e "${BLUE}Building whisper.cpp Docker image...${NC}"
    cd "$PROJECT_ROOT"
    # Build with docker directory as context so COPY can find transcribe.py
    docker build -f docker/Dockerfile.whisper-cpp -t whisper-cpp-gpu docker/
    echo -e "${GREEN}Build complete!${NC}"
}

# GPU test function
test_gpu() {
    echo -e "${BLUE}Testing GPU access...${NC}"
    docker run --rm --gpus all whisper-cpp-gpu nvidia-smi
}

# Parse arguments
if [ "$1" == "--build" ]; then
    build_image
    exit 0
elif [ "$1" == "--gpu-test" ]; then
    test_gpu
    exit 0
elif [ "$1" == "--help" ] || [ -z "$1" ]; then
    print_usage
    exit 0
fi

INPUT_FILE="$1"
shift

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file '$INPUT_FILE' not found${NC}"
    exit 1
fi

# Get absolute paths
INPUT_PATH=$(realpath "$INPUT_FILE")
INPUT_DIR=$(dirname "$INPUT_PATH")
INPUT_NAME=$(basename "$INPUT_PATH")

# Default values
OUTPUT_NAME="${INPUT_NAME%.*}.txt"
OUTPUT_DIR="$PWD"
MODEL="medium.en"
PROMPT_FILE=""

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            if [[ "$2" == */* ]]; then
                # Full path provided
                OUTPUT_DIR=$(dirname "$(realpath "$2")")
                OUTPUT_NAME=$(basename "$2")
            else
                # Just filename provided
                OUTPUT_NAME="$2"
            fi
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -p|--prompt)
            PROMPT_FILE="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Validate model choice
case $MODEL in
    tiny.en|base.en|small.en|medium.en|large-v3-q5_0)
        ;;
    *)
        echo -e "${RED}Invalid model: $MODEL${NC}"
        echo "Valid models: tiny.en, base.en, small.en, medium.en, large-v3-q5_0"
        exit 1
        ;;
esac

# Build docker run command
DOCKER_CMD="docker run --rm --gpus all"
DOCKER_CMD="$DOCKER_CMD -v \"$INPUT_DIR\":/input"
DOCKER_CMD="$DOCKER_CMD -v \"$OUTPUT_DIR\":/output"

# Add prompt volume if specified
PROMPT_ARG=""
if [ -n "$PROMPT_FILE" ]; then
    if [ ! -f "$PROMPT_FILE" ]; then
        echo -e "${RED}Error: Prompt file '$PROMPT_FILE' not found${NC}"
        exit 1
    fi
    PROMPT_PATH=$(realpath "$PROMPT_FILE")
    PROMPT_DIR=$(dirname "$PROMPT_PATH")
    PROMPT_NAME=$(basename "$PROMPT_PATH")
    DOCKER_CMD="$DOCKER_CMD -v \"$PROMPT_DIR\":/prompts"
    PROMPT_ARG="-p /prompts/$PROMPT_NAME"
    echo -e "${BLUE}Using prompt: $PROMPT_FILE${NC}"
fi

# Run transcription
echo -e "${BLUE}Starting transcription...${NC}"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_DIR/$OUTPUT_NAME"
echo "Model: $MODEL"
echo ""

# Complete command
FULL_CMD="$DOCKER_CMD whisper-cpp-gpu /input/$INPUT_NAME -o /output/$OUTPUT_NAME -m $MODEL $PROMPT_ARG"

# Execute
eval "$FULL_CMD"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}Transcription complete!${NC}"
    echo "Output saved to: $OUTPUT_DIR/$OUTPUT_NAME"
    
    # Suggest next step
    echo -e "\n${BLUE}Next step:${NC}"
    echo "Apply corrections with:"
    echo "  python3 $SCRIPT_DIR/apply_corrections.py $OUTPUT_DIR/$OUTPUT_NAME"
else
    echo -e "\n${RED}Transcription failed${NC}"
    exit 1
fi