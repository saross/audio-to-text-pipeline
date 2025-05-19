#!/bin/bash

# =============================================================================
# HIGH-QUALITY CPU-ONLY AUDIO TRANSCRIPTION PIPELINE
# =============================================================================
#
# This script runs a CPU-optimized audio transcription pipeline using the Whisper
# 'large-v3' model for transcription and pyannote.audio for speaker diarisation.
# It prioritizes transcription quality over speed by using CPU-only processing
# with the most accurate model available.
#
# PREREQUISITES:
# - Python environment with whisper, pyannote.audio, torch installed
# - A valid HUGGINGFACE_TOKEN (will prompt if not set as environment variable)
#
# USAGE:
#   ./run_cpu_transcription.sh [audio_file] [options]
#
# REQUIRED ARGUMENTS:
#   audio_file              Path to the audio file to transcribe
#
# OPTIONS:
#   -o, --output FILE       Output transcript file path
#                           Default: [input_filename]_transcript.txt in ./output/
#   -s, --speakers NUM      Number of speakers in the recording (if known)
#                           Default: Auto-detect
#
# EXAMPLES:
#   # Basic usage:
#   ./run_cpu_transcription.sh data/interview.mp3
#
#   # Specify output location:
#   ./run_cpu_transcription.sh data/interview.mp3 -o results/interview-transcript.txt
#
#   # Specify number of speakers:
#   ./run_cpu_transcription.sh data/interview.mp3 -s 2
#
# =============================================================================

# Activate transcription environment
source ~/transcription-env/bin/activate

# ===== CHECK/REQUEST HUGGINGFACE TOKEN =====
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    # Check if token is saved in shell config files
    TOKEN_FOUND=false
    POTENTIAL_CONFIGS=("$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.zshrc" "$HOME/.profile")
    
    for CONFIG_FILE in "${POTENTIAL_CONFIGS[@]}"; do
        if [ -f "$CONFIG_FILE" ]; then
            # Check if token exists in this config file
            if grep -q "HUGGINGFACE_TOKEN" "$CONFIG_FILE"; then
                echo "Found saved token in $CONFIG_FILE"
                # Extract only the token line instead of sourcing the entire file
                TOKEN_LINE=$(grep "export HUGGINGFACE_TOKEN" "$CONFIG_FILE" | grep -v "#" | tail -n 1)
                
                # If a token line was found, evaluate just that line
                if [ -n "$TOKEN_LINE" ]; then
                    eval "$TOKEN_LINE"
                    
                    # Verify the token was actually loaded
                    if [ -n "$HUGGINGFACE_TOKEN" ]; then
                        TOKEN_FOUND=true
                        break
                    fi
                fi
            fi
        fi
    done
    
    # If token is still not set after checking config files, ask for it
    if [ -z "$HUGGINGFACE_TOKEN" ]; then
        if [ "$TOKEN_FOUND" = true ]; then
            echo "Unable to load token from config file. It may be commented out or have special characters."
        else
            echo "HUGGINGFACE_TOKEN environment variable is not set."
        fi
        
        echo "A Hugging Face token is required for speaker diarisation."
        echo ""
        read -p "Please enter your Hugging Face token: " HUGGINGFACE_TOKEN
        
        if [ -z "$HUGGINGFACE_TOKEN" ]; then
            echo "No token provided. Exiting."
            exit 1
        fi
        
        # Set token for this session
        export HUGGINGFACE_TOKEN
        
        # Ask if user wants to save token to shell config
        read -p "Would you like to save this token for future use? (y/n): " SAVE_TOKEN
        if [[ "$SAVE_TOKEN" =~ ^[Yy] ]]; then
            SHELL_CONFIG=""
            if [[ "$SHELL" == *"bash"* ]]; then
                SHELL_CONFIG="$HOME/.bashrc"
            elif [[ "$SHELL" == *"zsh"* ]]; then
                SHELL_CONFIG="$HOME/.zshrc"
            fi
            
            if [ -n "$SHELL_CONFIG" ] && [ -f "$SHELL_CONFIG" ]; then
                # Check if token already exists in file to avoid duplicates
                if grep -q "HUGGINGFACE_TOKEN" "$SHELL_CONFIG"; then
                    echo "Token entry already exists in $SHELL_CONFIG. Updating it..."
                    # Use sed to replace the existing token line
                    sed -i "s/export HUGGINGFACE_TOKEN=.*/export HUGGINGFACE_TOKEN=\"$HUGGINGFACE_TOKEN\"/" "$SHELL_CONFIG"
                else
                    echo "" >> "$SHELL_CONFIG"
                    echo "# Hugging Face token for audio transcription" >> "$SHELL_CONFIG"
                    echo "export HUGGINGFACE_TOKEN=\"$HUGGINGFACE_TOKEN\"" >> "$SHELL_CONFIG"
                fi
                echo "Token saved to $SHELL_CONFIG. It will be available in new terminal sessions."
                echo "To make it available in this session too, run: source $SHELL_CONFIG"
            else
                echo "Couldn't determine shell config file. Please add manually:"
                echo "export HUGGINGFACE_TOKEN=\"$HUGGINGFACE_TOKEN\""
            fi
        fi
    fi
fi

# Remove duplicate token display line
echo "Using Hugging Face token: ${HUGGINGFACE_TOKEN:0:5}...${HUGGINGFACE_TOKEN: -5}"

# ===== CPU OPTIMIZATION =====
echo "Configuring CPU optimisation settings..."

# Set thread count - adjust these numbers based on your CPU
CPU_THREADS=4  # Default for 4-core system

# Get CPU core count if possible
if command -v nproc &> /dev/null; then
    TOTAL_CORES=$(nproc)
    # Use 75% of cores (rounded down) but minimum 4, maximum 10
    CALCULATED_THREADS=$(( TOTAL_CORES * 3 / 4 ))
    CPU_THREADS=$(( CALCULATED_THREADS < 4 ? 4 : (CALCULATED_THREADS > 10 ? 10 : CALCULATED_THREADS) ))
    echo "Detected $TOTAL_CORES CPU cores, using $CPU_THREADS threads"
else
    echo "Using default thread count: $CPU_THREADS"
fi

# Set threading environment variables
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
export TOKENIZERS_PARALLELISM="false"  # Prevent huggingface tokenizer warnings
export PYTHONOPTIMIZE="1"  # Optimize Python

# Ensure CUDA is disabled for CPU-only operation
export CUDA_VISIBLE_DEVICES=""

# ===== CREATE OUTPUT DIRECTORY =====
# Create output directory if specified in arguments
if [[ "$*" =~ -o\ ([^\ ]+) || "$*" =~ --output\ ([^\ ]+) ]]; then
    OUTPUT_PATH="${BASH_REMATCH[1]}"
    OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
else
    # Default output directory
    echo "Creating default output directory"
    mkdir -p output
fi

# ===== ESTIMATE PROCESSING TIME =====
# Try to estimate duration of audio file if possible
DURATION_ESTIMATE=""
if command -v ffprobe &> /dev/null && [[ "$1" != "" && -f "$1" ]]; then
    DURATION_SECONDS=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$1" 2>/dev/null)
    if [[ "$DURATION_SECONDS" != "" ]]; then
        DURATION_MINUTES=$(echo "$DURATION_SECONDS / 60" | bc)
        MIN_ESTIMATE=$(echo "$DURATION_SECONDS / 60 * 5" | bc)
        MAX_ESTIMATE=$(echo "$DURATION_SECONDS / 60 * 10" | bc)
        echo "Audio duration: ~${DURATION_MINUTES} minutes. Estimated processing time: ${MIN_ESTIMATE}-${MAX_ESTIMATE} minutes"
    fi
fi

# ===== RUN TRANSCRIPTION =====
echo ""
echo "========================================================================"
echo "Starting CPU-only high-quality transcription with large-v3 model..."
echo "This will take longer but produce the highest quality results."
echo "Running with thread count: ${CPU_THREADS}"
echo "========================================================================" 
echo ""

# Record start time
START_TIME=$(date +%s)

# Run the script with proper error handling
if python transcript_cpu.py "$@"; then
    # Calculate elapsed time
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(( (ELAPSED % 3600) / 60 ))
    SECONDS=$((ELAPSED % 60))
    
    echo ""
    echo "========================================================================"
    echo "High-quality CPU transcription completed successfully!"
    echo "Total processing time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "ERROR: Transcription failed!"
    echo "Check error messages above for details."
    echo "========================================================================"
    exit 1
fi