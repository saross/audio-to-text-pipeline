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
    echo "  -v, --verbose        Show detailed whisper output (default: quiet with progress)"
    echo "  --no-corrections     Skip automatic term corrections"
    echo "  --build              Build the Docker image"
    echo "  --gpu-test           Test GPU access"
    echo ""
    echo "Examples:"
    echo "  $0 interview.flac"
    echo "  $0 interview.flac -o transcript.txt -m small.en"
    echo "  $0 interview.flac -p whisper_prompt.txt --verbose"
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
VERBOSE=false
AUTO_CORRECT=true

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
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --no-corrections)
            AUTO_CORRECT=false
            shift
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

# Enhanced progress monitoring function
# Displays real-time transcription progress by monitoring whisper.cpp output
# Shows percentage complete when available, otherwise shows animated spinner
show_progress() {
    local pid=$1        # Process ID to monitor
    local log_file=$2   # Log file to extract progress from
    local delay=0.5     # Update frequency in seconds
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'  # Unicode spinner characters
    local start_time=$(date +%s)
    local last_progress=""
    
    echo -ne "${YELLOW}Starting transcription...${NC}"
    
    # Monitor while process is running
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        local elapsed=$(($(date +%s) - start_time))
        local mins=$((elapsed / 60))
        local secs=$((elapsed % 60))
        
        # Try to extract progress percentage from whisper.cpp output
        if [ -f "$log_file" ]; then
            # Look for whisper progress output (e.g., "progress = 45%")
            local current_progress=$(grep -o "progress = [0-9]*%" "$log_file" | tail -1 | grep -o "[0-9]*")
            if [ -n "$current_progress" ] && [ "$current_progress" != "$last_progress" ]; then
                last_progress=$current_progress
                # Show percentage and elapsed time
                printf "\r${YELLOW}Transcribing... ${current_progress}%% [%02d:%02d]${NC}        " "$mins" "$secs"
            else
                # Show spinner if no percentage available
                printf "\r${YELLOW}Transcribing... %c [%02d:%02d]${NC} " "$spinstr" "$mins" "$secs"
            fi
        else
            # Log file not yet created, show spinner
            printf "\r${YELLOW}Transcribing... %c [%02d:%02d]${NC} " "$spinstr" "$mins" "$secs"
        fi
        
        # Rotate spinner
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
    done
    
    # Final update showing total time
    local elapsed=$(($(date +%s) - start_time))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))
    printf "\r${GREEN}Transcription complete! [%02d:%02d]${NC}        \n" "$mins" "$secs"
}

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
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_DIR/$OUTPUT_NAME"
echo "Model:  $MODEL"
if [ -n "$PROMPT_FILE" ]; then
    echo "Prompt: $PROMPT_FILE"
fi
echo ""

# Complete command
# Note: transcribe.py expects different argument format than whisper.cpp
FULL_CMD="$DOCKER_CMD whisper-cpp-cuda-fixed /input/$INPUT_NAME -o /output/$OUTPUT_NAME -m $MODEL $PROMPT_ARG"

if [ "$VERBOSE" = true ]; then
    # Verbose mode - show all output
    echo -e "${YELLOW}Verbose mode - showing detailed output...${NC}"
    echo ""
    eval "$FULL_CMD"
    EXIT_CODE=$?
else
    # Quiet mode with progress indicator (default)
    # Create a temporary file for capturing output
    TEMP_LOG=$(mktemp)
    
    # Execute in background with output redirected
    eval "$FULL_CMD > $TEMP_LOG 2>&1" &
    DOCKER_PID=$!
    
    # Show progress while running
    show_progress $DOCKER_PID $TEMP_LOG
    
    # Wait for completion and get exit code
    wait $DOCKER_PID
    EXIT_CODE=$?
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Transcription complete!${NC}"
    echo "Output saved to: $OUTPUT_DIR/$OUTPUT_NAME"
    
    # Show file size and word count
    if [ -f "$OUTPUT_DIR/$OUTPUT_NAME" ]; then
        FILE_SIZE=$(wc -c < "$OUTPUT_DIR/$OUTPUT_NAME")
        WORD_COUNT=$(wc -w < "$OUTPUT_DIR/$OUTPUT_NAME")
        echo "File size: $(numfmt --to=iec --suffix=B $FILE_SIZE 2>/dev/null || echo "$FILE_SIZE bytes")"
        echo "Word count: $(printf "%'d" $WORD_COUNT 2>/dev/null || echo "$WORD_COUNT")"
    fi
    
    # Apply corrections if enabled
    if [ "$AUTO_CORRECT" = true ] && [ -f "$OUTPUT_DIR/$OUTPUT_NAME" ]; then
        echo ""
        echo -e "${BLUE}Applying term corrections...${NC}"
        
        # Check if corrections script exists
        if [ -f "$SCRIPT_DIR/apply_corrections.py" ]; then
            python3 "$SCRIPT_DIR/apply_corrections.py" "$OUTPUT_DIR/$OUTPUT_NAME"
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✓ Corrections applied${NC}"
            else
                echo -e "${YELLOW}Warning: Corrections failed but transcript is available${NC}"
            fi
        else
            echo -e "${YELLOW}Warning: Corrections script not found${NC}"
        fi
    fi
    
    # Clean up temp file in quiet mode
    if [ "$VERBOSE" = false ] && [ -n "$TEMP_LOG" ]; then
        rm -f "$TEMP_LOG"
    fi
else
    echo -e "${RED}✗ Transcription failed${NC}"
    
    if [ "$VERBOSE" = false ]; then
        # In quiet mode, show last 20 lines of error output
        echo ""
        echo "Error output:"
        echo "---"
        if [ -f "$TEMP_LOG" ]; then
            tail -20 "$TEMP_LOG"
        fi
        echo "---"
        echo ""
        echo "Full log saved to: $TEMP_LOG"
        echo "(Remove with: rm $TEMP_LOG)"
    else
        # In verbose mode, error was already shown
        echo "Check the error messages above for details."
    fi
    exit 1
fi

