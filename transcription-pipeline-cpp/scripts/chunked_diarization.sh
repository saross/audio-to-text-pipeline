#!/bin/bash
#
# Chunked Speaker Diarization Script
#
# PURPOSE:
#     Performs speaker diarization on long audio files by processing them in chunks.
#     This avoids memory issues and hanging that occur with files >10-15 minutes.
#
# FEATURES:
#     - Processes audio in configurable chunks (default: 10 minutes)
#     - Handles file permission issues from Docker containers
#     - Shows real-time progress for each chunk
#     - Continues processing even if some chunks fail
#     - Merges all chunks into final output
#
# USAGE:
#     ./chunked_diarization.sh [options] <input_audio> <output_json>
#
# OPTIONS:
#     -c, --chunk-size MINUTES   Size of chunks in minutes (default: 10)
#     -n, --num-speakers N       Number of speakers (default: auto-detect)
#
# REQUIREMENTS:
#     - Docker with diarization-pyannote image
#     - HF_TOKEN environment variable set
#     - sudo access (for fixing file permissions)
#
# EXAMPLE:
#     export HF_TOKEN=your_token_here
#     ./chunked_diarization.sh -c 10 -n 2 interview.flac output.json

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Default values
CHUNK_MINUTES=10
NUM_SPEAKERS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--chunk-size)
            CHUNK_MINUTES="$2"
            shift 2
            ;;
        -n|--num-speakers)
            NUM_SPEAKERS="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

INPUT_FILE="$1"
OUTPUT_FILE="$2"

if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 [-c CHUNK_MINUTES] [-n NUM_SPEAKERS] <input_audio> <output_json>"
    exit 1
fi

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}Error: HF_TOKEN not set${NC}"
    exit 1
fi

# Get absolute paths
INPUT_FILE=$(realpath "$INPUT_FILE")
OUTPUT_FILE=$(realpath "$OUTPUT_FILE")

# Create temp directory
TEMP_DIR="/tmp/diarization_$$"
mkdir -p "$TEMP_DIR"

echo -e "${BLUE}=== Diarization with Fixed Permissions ===${NC}"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Chunk size: ${CHUNK_MINUTES} minutes"
[ -n "$NUM_SPEAKERS" ] && echo "Speakers: $NUM_SPEAKERS"

# Get duration
DURATION=$(ffprobe -i "$INPUT_FILE" -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 2>/dev/null)
DURATION_INT=${DURATION%.*}
DURATION_MIN=$((DURATION_INT / 60))
echo "Duration: ${DURATION_MIN} minutes"

# Calculate chunks
CHUNK_SIZE=$((CHUNK_MINUTES * 60))
NUM_CHUNKS=$(( (DURATION_INT + CHUNK_SIZE - 1) / CHUNK_SIZE ))
echo -e "${BLUE}Processing $NUM_CHUNKS chunks...${NC}"

# Process chunks
SUCCESSFUL=0
TOTAL_START=$(date +%s)

for i in $(seq 0 $((NUM_CHUNKS-1))); do
    START=$((i * CHUNK_SIZE))
    END=$((START + CHUNK_SIZE))
    [ $END -gt $DURATION_INT ] && END=$DURATION_INT
    
    CHUNK_FILE="$TEMP_DIR/chunk_${i}.flac"
    CHUNK_JSON="$TEMP_DIR/chunk_${i}_diarization.json"
    
    echo -e "\n${BLUE}[Chunk $((i+1))/$NUM_CHUNKS]${NC} ${START}s-${END}s"
    
    # Extract chunk
    echo -n "  Extracting..."
    ffmpeg -ss $START -i "$INPUT_FILE" -t $CHUNK_SIZE -c copy "$CHUNK_FILE" -y -loglevel error 2>/dev/null
    echo -e "\r  ${GREEN}✓ Extracted${NC}   "
    
    # Build Docker command (without user restrictions)
    DOCKER_CMD="docker run --rm --gpus all"
    DOCKER_CMD="$DOCKER_CMD -e HF_TOKEN=$HF_TOKEN"
    DOCKER_CMD="$DOCKER_CMD -e MPLCONFIGDIR=/tmp"
    DOCKER_CMD="$DOCKER_CMD -v $CHUNK_FILE:/input/audio.flac:ro"
    DOCKER_CMD="$DOCKER_CMD -v $TEMP_DIR:/output"
    DOCKER_CMD="$DOCKER_CMD diarization-pyannote"
    DOCKER_CMD="$DOCKER_CMD /input/audio.flac"
    DOCKER_CMD="$DOCKER_CMD -o /output/chunk_${i}_diarization.json"
    [ -n "$NUM_SPEAKERS" ] && DOCKER_CMD="$DOCKER_CMD -n $NUM_SPEAKERS"
    
    echo "  Running diarization..."
    CHUNK_START=$(date +%s)
    
    # Run diarization
    if timeout 600 bash -c "$DOCKER_CMD" 2>&1 | while IFS= read -r line; do
        if echo "$line" | grep -q "━━━"; then
            STAGE=$(echo "$line" | awk '{print $1}')
            echo -ne "\r  ${STAGE}...                              "
        fi
    done; then
        CHUNK_TIME=$(($(date +%s) - CHUNK_START))
        echo -e "\r  ${GREEN}✓ Complete${NC} (${CHUNK_TIME}s)                    "
        
        # Fix permissions on the output file
        if [ -f "$CHUNK_JSON" ]; then
            # Change ownership to current user
            sudo chown $(whoami):$(whoami) "$CHUNK_JSON" 2>/dev/null || \
                chmod 644 "$CHUNK_JSON" 2>/dev/null || true
            
            # Now adjust timestamps
            if python3 -c "
import json
try:
    with open('$CHUNK_JSON', 'r') as f:
        data = json.load(f)
    for seg in data.get('segments', []):
        seg['start'] += $START
        seg['end'] += $START
    with open('$CHUNK_JSON', 'w') as f:
        json.dump(data, f)
    print('✓ Timestamps adjusted')
except Exception as e:
    print(f'✗ Error adjusting timestamps: {e}')
    exit(1)
" 2>/dev/null; then
                SUCCESSFUL=$((SUCCESSFUL + 1))
            else
                echo "  ${YELLOW}Warning: Could not adjust timestamps${NC}"
            fi
        else
            echo "  ${RED}✗ Output file not found${NC}"
        fi
    else
        echo -e "\r  ${RED}✗ Failed${NC}                              "
    fi
    
    # Clean up audio chunk
    rm -f "$CHUNK_FILE"
    
    # Progress
    ELAPSED=$(($(date +%s) - TOTAL_START))
    echo "  Progress: ${GREEN}$SUCCESSFUL${NC}/$((i+1)) successful | Time: $((ELAPSED/60))m"
done

echo -e "\n${BLUE}Merging results...${NC}"

# Merge with proper error handling
python3 << EOF
import json
import glob
import os

all_segments = []
all_speakers = set()
files_read = 0

for f in sorted(glob.glob('$TEMP_DIR/chunk_*_diarization.json')):
    try:
        # Try to read with different methods if permission denied
        try:
            with open(f, 'r') as file:
                data = json.load(file)
        except PermissionError:
            # Try with sudo cat
            import subprocess
            result = subprocess.run(['sudo', 'cat', f], capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
            else:
                raise Exception("Cannot read file")
        
        all_segments.extend(data.get('segments', []))
        all_speakers.update(data.get('speakers', []))
        files_read += 1
    except Exception as e:
        print(f"Warning: Could not read {os.path.basename(f)}: {e}")

all_segments.sort(key=lambda x: x['start'])

output = {
    'speakers': sorted(list(all_speakers)),
    'num_speakers': len(all_speakers),
    'total_segments': len(all_segments),
    'segments': all_segments,
    'chunks_processed': files_read,
    'chunks_total': $NUM_CHUNKS
}

with open('$OUTPUT_FILE', 'w') as f:
    json.dump(output, f, indent=2)

print(f'Merged {len(all_segments)} segments from {files_read} chunks')
EOF

# Cleanup - use sudo if needed
sudo rm -rf "$TEMP_DIR" 2>/dev/null || rm -rf "$TEMP_DIR"

echo -e "\n${GREEN}✓ Complete!${NC}"
echo "Output: $OUTPUT_FILE"

# Summary
if [ -f "$OUTPUT_FILE" ]; then
    python3 -c "
import json
with open('$OUTPUT_FILE') as f:
    data = json.load(f)
if data['segments']:
    first = min(s['start'] for s in data['segments'])
    last = max(s['end'] for s in data['segments'])
    coverage = (last - first) / $DURATION * 100
    print(f'Coverage: {coverage:.1f}%')
    print(f'Segments: {len(data[\"segments\"])}')
    print(f'Speakers: {\", \".join(data[\"speakers\"])}')
    print(f'Success rate: {data[\"chunks_processed\"]}/{data[\"chunks_total\"]} chunks')
"
fi