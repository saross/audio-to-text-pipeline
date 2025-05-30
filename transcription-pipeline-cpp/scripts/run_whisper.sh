#!/bin/bash

# Simple whisper.cpp Docker runner

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_usage() {
    echo "Usage: $0 <input_audio> [options]"
    echo "Options:"
    echo "  -o, --output FILE    Output file (default: input_name.txt)"
    echo "  -m, --model MODEL    Model to use (tiny.en, base.en, small.en, medium.en)"
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
    echo "  $0 --gpu-test"
}

# Build function
build_image() {
    echo -e "${BLUE}Building whisper.cpp Docker image...${NC}"
    cd "$PROJECT_ROOT"
    # Build with docker directory as context so COPY can find transcribe.py
    docker build -f docker/Dockerfile.whisper-cpp -t whisper-cpp-cuda .
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Build complete!${NC}"
    else
        echo -e "${RED}Build failed!${NC}"
        exit 1
    fi
}

# GPU test function
test_gpu() {
    echo -e "${BLUE}Testing GPU access in Docker container...${NC}"
    echo ""
    
    # First check if Docker image exists
    if ! docker image inspect whisper-cpp-cuda >/dev/null 2>&1; then
        echo -e "${RED}Error: Docker image 'whisper-cpp-cuda' not found.${NC}"
        echo "Please run: $0 --build"
        exit 1
    fi
    
    # Test nvidia-smi
    echo -e "${BLUE}Running nvidia-smi in container:${NC}"
    docker run --rm --gpus all --entrypoint nvidia-smi whisper-cpp-cuda
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ GPU access successful!${NC}"
        
        # Additional GPU info
        echo ""
        echo -e "${BLUE}GPU Memory Info:${NC}"
        docker run --rm --gpus all --entrypoint nvidia-smi whisper-cpp-cuda --query-gpu=name,memory.total,memory.free --format=csv,noheader
        
        # Test CUDA
        echo ""
        echo -e "${BLUE}Testing CUDA in whisper.cpp:${NC}"
        docker run --rm --gpus all --entrypoint /bin/bash whisper-cpp-cuda -c "cd /app/whisper.cpp && ./build/bin/main --help 2>&1 | grep -i cuda || echo 'CUDA info not found in help output'"
        
        echo ""
        echo -e "${GREEN}GPU is ready for transcription!${NC}"
    else
        echo ""
        echo -e "${RED}✗ GPU access failed!${NC}"
        echo ""
        echo "Troubleshooting steps:"
        echo "1. Check if NVIDIA drivers are installed: nvidia-smi"
        echo "2. Check if Docker GPU support is installed:"
        echo "   docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi"
        echo "3. Check if nvidia-container-toolkit is installed:"
        echo "   dpkg -l | grep nvidia-container"
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

INPUT_FILE="$1"
shift

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file '$INPUT_FILE' not found${NC}"
    exit 1
fi

# Check if Docker image exists
if ! docker image inspect whisper-cpp-cuda >/dev/null 2>&1; then
    echo -e "${RED}Error: Docker image 'whisper-cpp-cuda' not found.${NC}"
    echo "Please run: $0 --build"
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
                OUTPUT_DIR=$(dirname "$2")
                OUTPUT_NAME=$(basename "$2")
                # Create directory if it doesn't exist
                mkdir -p "$OUTPUT_DIR"
                OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
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

# Validate model choice (removed large-v3-q5_0)
case $MODEL in
    tiny.en|base.en|small.en|medium.en)
        ;;
    *)
        echo -e "${RED}Invalid model: $MODEL${NC}"
        echo "Valid models: tiny.en, base.en, small.en, medium.en"
        exit 1
        ;;
esac

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

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
    PROMPT_ARG="--prompt /prompts/$PROMPT_NAME"
    echo -e "${BLUE}Using prompt: $PROMPT_FILE${NC}"
fi

# Run transcription
echo -e "${BLUE}Starting transcription...${NC}"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_DIR/$OUTPUT_NAME"
echo "Model: $MODEL"
echo ""

# Show progress hint
echo -e "${YELLOW}Progress will be shown below. This may take a few minutes...${NC}"
echo ""

# Complete command
# Note: transcribe.py expects different argument format than whisper.cpp
FULL_CMD="$DOCKER_CMD whisper-cpp-cuda-fixed /input/$INPUT_NAME -o /output/$OUTPUT_NAME -m $MODEL $PROMPT_ARG"

# Execute
eval "$FULL_CMD"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Transcription complete!${NC}"
    echo "Output saved to: $OUTPUT_DIR/$OUTPUT_NAME"
    
    # Show file size
    if [ -f "$OUTPUT_DIR/$OUTPUT_NAME" ]; then
        FILE_SIZE=$(wc -c < "$OUTPUT_DIR/$OUTPUT_NAME")
        echo "File size: $FILE_SIZE bytes"
    fi
    
    # Suggest next step
    echo ""
    echo -e "${BLUE}Next step:${NC}"
    echo "Apply corrections with:"
    echo "  python3 $SCRIPT_DIR/apply_corrections.py $OUTPUT_DIR/$OUTPUT_NAME"
else
    echo ""
    echo -e "${RED}✗ Transcription failed${NC}"
    echo "Check the error messages above for details."
    exit 1
fi