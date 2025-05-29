#!/bin/bash

# Comprehensive cleanup script for all previous transcription attempts

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_info() { echo -e "${BLUE}[i]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }

echo "=== Cleanup Script for Old Transcription Attempts ==="
echo "This will remove Docker images, containers, and old project files"
echo

# 1. Stop and remove all containers related to transcription
print_info "Stopping and removing old containers..."

# Find and stop containers
CONTAINERS=$(docker ps -a | grep -E "(transcription|whisper)" | awk '{print $1}')
if [ ! -z "$CONTAINERS" ]; then
    print_warning "Found containers to remove:"
    docker ps -a | grep -E "(transcription|whisper)"
    docker stop $CONTAINERS 2>/dev/null || true
    docker rm $CONTAINERS 2>/dev/null || true
    print_status "Containers removed"
else
    print_status "No old containers found"
fi

# 2. Remove Docker images
print_info "Looking for old Docker images..."

# List all transcription-related images
IMAGES=$(docker images | grep -E "(transcription-pipeline|whisper|pytorch|faster-whisper)" | awk '{print $3}')
if [ ! -z "$IMAGES" ]; then
    print_warning "Found images to remove:"
    docker images | grep -E "(transcription-pipeline|whisper|pytorch|faster-whisper)"
    
    read -p "Remove these images? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo $IMAGES | xargs docker rmi -f 2>/dev/null || true
        print_status "Images removed"
    fi
else
    print_status "No old images found"
fi

# 3. Remove dangling images and volumes
print_info "Cleaning up dangling images and volumes..."
docker image prune -f
docker volume prune -f
print_status "Dangling resources cleaned"

# 4. Clean build cache
print_info "Cleaning Docker build cache..."
docker builder prune -f
print_status "Build cache cleaned"

# 5. Remove old project directories
print_info "Looking for old project directories..."

OLD_DIRS=(
    "$HOME/Code/audio-transcription/transcription-pipeline"
    "$HOME/transcription-env"
    "$HOME/whisper-transcription"  # If you started but didn't complete
)

for dir in "${OLD_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        print_warning "Found: $dir"
        read -p "Remove this directory? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$dir"
            print_status "Removed $dir"
        fi
    fi
done

# 6. Clean pip cache (if virtual environment exists)
if [ -d "$HOME/transcription-env" ]; then
    print_info "Cleaning pip cache..."
    pip cache purge 2>/dev/null || true
fi

# 7. Remove any downloaded models
print_info "Looking for cached models..."
MODEL_DIRS=(
    "$HOME/.cache/whisper"
    "$HOME/.cache/huggingface"
    "$HOME/.cache/torch"
)

for dir in "${MODEL_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
        print_warning "Found model cache: $dir (Size: $SIZE)"
        read -p "Remove this cache? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$dir"
            print_status "Removed $dir"
        fi
    fi
done

# 8. Show disk space recovery
print_info "Checking disk space..."
echo
echo "Docker disk usage:"
docker system df
echo

# 9. Final system prune (optional)
print_warning "Optional: Full Docker system prune"
echo "This will remove ALL unused Docker resources (not just transcription-related)"
read -p "Run full system prune? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker system prune -a -f --volumes
    print_status "Full system prune complete"
fi

# 10. Summary
echo
print_status "Cleanup complete!"
echo
echo "Space recovered:"
df -h . | grep -E "Filesystem|$(pwd)"
echo
echo "Remaining Docker images:"
docker images | head -5
echo
print_info "You now have a clean system for the whisper.cpp implementation!"
print_info "Your audio files and config files are preserved."

# Create a fresh directory for the new approach
echo
read -p "Create fresh directory ~/whisper-transcription for new setup? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p ~/whisper-transcription
    print_status "Created ~/whisper-transcription"
    echo "Next steps:"
    echo "1. cd ~/whisper-transcription"
    echo "2. Copy your artifacts from the new chat"
    echo "3. Follow the whisper.cpp setup instructions"
fi