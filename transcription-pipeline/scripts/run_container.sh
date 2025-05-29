#!/bin/bash

# =============================================================================
# GPU-Accelerated Transcription Pipeline - Docker Runner
# =============================================================================

set -e

# Configuration
IMAGE_NAME="transcription-pipeline"
CONTAINER_NAME="transcription-work"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Check for NVIDIA Docker support with better error reporting
print_status "Checking basic Docker and NVIDIA setup..."

# First check if nvidia-smi works on host
if ! nvidia-smi > /dev/null 2>&1; then
    print_error "nvidia-smi not working on host system. Check NVIDIA drivers."
    exit 1
fi

# Test basic Docker functionality
if ! docker run --rm hello-world > /dev/null 2>&1; then
    print_error "Basic Docker functionality not working. Check Docker permissions."
    print_error "Try: sudo usermod -aG docker $USER && newgrp docker"
    exit 1
fi

print_status "✅ Basic Docker and NVIDIA drivers working"
print_status "GPU verification will be done after building the pipeline container"

# Function to build the container
build_container() {
    print_status "Building transcription pipeline container..."
    docker build -t $IMAGE_NAME .
    
    if [ $? -eq 0 ]; then
        print_status "Container built successfully: $IMAGE_NAME"
    else
        print_error "Container build failed"
        exit 1
    fi
}

# Function to test GPU access
test_gpu() {
    print_status "Testing GPU access with transcription pipeline..."
    docker run --rm --gpus all \
        -v $(pwd)/config:/workspace/config \
        $IMAGE_NAME --gpu-check
}

# Function to check pipeline status
check_status() {
    print_status "Checking pipeline status..."
    docker run --rm --gpus all \
        -v $(pwd)/config:/workspace/config \
        $IMAGE_NAME --gpu-check
}

# Function to run transcription
run_transcription() {
    local input_file="$1"
    local output_file="$2"
    local extra_args="${@:3}"
    
    if [ -z "$input_file" ]; then
        print_error "Input file not specified"
        echo "Usage: $0 transcribe <input_file> [output_file] [extra_args]"
        exit 1
    fi
    
    if [ ! -f "$input_file" ]; then
        print_error "Input file not found: $input_file"
        exit 1
    fi
    
    # Set default output file
    if [ -z "$output_file" ]; then
        output_file="${input_file%.*}.txt"
    fi
    
    # Get absolute paths
    input_dir=$(dirname "$(realpath "$input_file")")
    input_name=$(basename "$input_file")
    output_dir=$(dirname "$(realpath "$output_file")")
    output_name=$(basename "$output_file")
    
    print_status "Starting transcription..."
    print_status "Input: $input_file"
    print_status "Output: $output_file"
    
    # Set up Hugging Face token if available
    HF_TOKEN_ARG=""
    if [ -n "$HUGGINGFACE_TOKEN" ]; then
        HF_TOKEN_ARG="-e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN"
    else
        print_warning "HUGGINGFACE_TOKEN not set. Speaker diarisation may fail."
        print_warning "Set with: export HUGGINGFACE_TOKEN=your_token_here"
    fi
    
    # Run transcription
    docker run --rm --gpus all \
        $HF_TOKEN_ARG \
        -v "$input_dir":/workspace/input \
        -v "$output_dir":/workspace/output \
        -v $(pwd)/config:/workspace/config \
        $IMAGE_NAME \
        "/workspace/input/$input_name" \
        -o "/workspace/output/$output_name" \
        $extra_args
    
    if [ $? -eq 0 ]; then
        print_status "Transcription completed successfully!"
        print_status "Output saved to: $output_file"
    else
        print_error "Transcription failed"
        exit 1
    fi
}

# Main script logic
case "$1" in
    "build")
        build_container
        ;;
    "test")
        test_gpu
        ;;
    "status")
        check_status
        ;;
    "transcribe")
        shift
        run_transcription "$@"
        ;;
    "shell")
        print_status "Starting interactive shell in container..."
        docker run --rm -it --gpus all \
            -v $(pwd):/workspace \
            --entrypoint /bin/bash \
            $IMAGE_NAME
        ;;
    *)
        echo "GPU-Accelerated Transcription Pipeline"
        echo ""
        echo "Usage: $0 <command> [arguments]"
        echo ""
        echo "Commands:"
        echo "  build                          Build the Docker container"
        echo "  test                           Test GPU access"
        echo "  status                         Check pipeline status"
        echo "  transcribe <input> [output]    Run transcription"
        echo "  shell                          Open interactive shell"
        echo ""
        echo "Examples:"
        echo "  $0 build"
        echo "  $0 test"
        echo "  $0 transcribe interview.flac"
        echo "  $0 transcribe interview.flac transcript.txt --model large-v3"
        echo ""
        echo "Environment Variables:"
        echo "  HUGGINGFACE_TOKEN             Required for speaker diarisation"
        ;;
esac