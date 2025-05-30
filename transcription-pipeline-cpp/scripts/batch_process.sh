#!/bin/bash
# Batch process multiple audio files with whisper.cpp

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default model
MODEL=${1:-medium.en}

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=== Batch Transcription Processing ===${NC}"
echo -e "Model: ${YELLOW}$MODEL${NC}"
echo ""

# Check if input directory exists
if [ ! -d "data/input" ]; then
    echo -e "${RED}Error: data/input directory not found${NC}"
    echo "Please create it and add your audio files"
    exit 1
fi

# Count files
FILE_COUNT=$(find data/input -name "*.flac" -o -name "*.mp3" -o -name "*.wav" -o -name "*.m4a" | wc -l)

if [ $FILE_COUNT -eq 0 ]; then
    echo -e "${YELLOW}No audio files found in data/input${NC}"
    echo "Supported formats: .flac, .mp3, .wav, .m4a"
    exit 1
fi

echo -e "Found ${GREEN}$FILE_COUNT${NC} audio files to process"
echo ""

# Process each file
PROCESSED=0
FAILED=0

for f in data/input/*.{flac,mp3,wav,m4a} 2>/dev/null; do
    if [ -f "$f" ]; then
        basename=$(basename "$f")
        name="${basename%.*}"
        
        echo -e "${BLUE}[$((PROCESSED + FAILED + 1))/$FILE_COUNT]${NC} Processing: $basename"
        
        # Check if already processed
        if [ -f "results/corrected/${name}-corrected.txt" ]; then
            echo -e "  ${YELLOW}Skipping - already processed${NC}"
            PROCESSED=$((PROCESSED + 1))
            continue
        fi
        
        # Transcribe with prompt
        echo -e "  ${BLUE}Transcribing...${NC}"
        if "$SCRIPT_DIR/run_whisper.sh" "$f" \
            -o "results/raw/${name}.txt" \
            -m "$MODEL" \
            -p config/whisper_prompt.txt > "logs/${name}_transcribe.log" 2>&1; then
            
            # Apply corrections if transcription succeeded
            if [ -f "results/raw/${name}.txt" ]; then
                echo -e "  ${BLUE}Applying corrections...${NC}"
                if python3 "$SCRIPT_DIR/apply_corrections.py" \
                    "results/raw/${name}.txt" \
                    -o "results/corrected/${name}-corrected.txt" \
                    -c config/term_corrections.txt > "logs/${name}_corrections.log" 2>&1; then
                    
                    echo -e "  ${GREEN}✓ Complete${NC}"
                    PROCESSED=$((PROCESSED + 1))
                else
                    echo -e "  ${RED}✗ Correction failed${NC}"
                    FAILED=$((FAILED + 1))
                fi
            else
                echo -e "  ${RED}✗ Transcription output not found${NC}"
                FAILED=$((FAILED + 1))
            fi
        else
            echo -e "  ${RED}✗ Transcription failed${NC}"
            FAILED=$((FAILED + 1))
        fi
        
        echo ""
    fi
done

# Summary
echo -e "${BLUE}=== Processing Complete ===${NC}"
echo -e "Processed: ${GREEN}$PROCESSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "Failed: ${RED}$FAILED${NC}"
fi

# Show results location
if [ $PROCESSED -gt 0 ]; then
    echo ""
    echo -e "${BLUE}Results saved in:${NC}"
    echo "  - Raw transcripts: results/raw/"
    echo "  - Corrected transcripts: results/corrected/"
    echo "  - Processing logs: logs/"
fi