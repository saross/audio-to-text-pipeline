#!/bin/bash

# =============================================================================
# AUDIO TRANSCRIPTION AND DIARISATION PIPELINE
# =============================================================================
#
# This script runs an optimized audio transcription pipeline using the Whisper
# 'medium' model for transcription (on GPU) and pyannote.audio for speaker diarisation
# (on CPU). This balanced approach provides excellent quality with good performance.
#
# PREREQUISITES:
# - Python environment with whisper, pyannote.audio, torch installed
# - A valid HUGGINGFACE_TOKEN (will prompt if not set as environment variable)
# - NVIDIA GPU with 6GB+ VRAM (for optimal performance)
#
# USAGE:
#   ./run_transcription.sh [audio_file] [options]
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
#   ./run_transcription.sh data/interview.mp3
#
#   # Specify output location:
#   ./run_transcription.sh data/interview.mp3 -o results/interview-transcript.txt
#
#   # Specify number of speakers:
#   ./run_transcription.sh data/interview.mp3 -s 2
#
# ENVIRONMENT VARIABLES:
#   HUGGINGFACE_TOKEN       Required for diarisation model access
#   PYTORCH_CUDA_ALLOC_CONF Memory management for PyTorch
#
# OUTPUT:
#   A text file containing the transcript with:
#   - Timestamps in format [HH:MM:SS]
#   - Speaker labels (SPEAKER_00, SPEAKER_01, etc.)
#   - Transcribed text for each segment
#
# =============================================================================

# Activate virtual environment
source ~/transcription-env/bin/activate

# ===== CHECK/REQUEST HUGGINGFACE TOKEN =====
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    # Check if token is saved in shell config files
    TOKEN_FOUND=false
    POTENTIAL_CONFIGS=("$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.zshrc" "$HOME/.profile")
    
    for CONFIG_FILE in "${POTENTIAL_CONFIGS[@]}"; do
        if [ -f "$CONFIG_FILE" ]; then
            # Check if token exists in this config file
            TOKEN_LINE=$(grep "export HUGGINGFACE_TOKEN" "$CONFIG_FILE" | grep -v "#" | tail -n 1)
            
            if [ -n "$TOKEN_LINE" ]; then
                echo "Found saved token in $CONFIG_FILE"
                # Extract and evaluate just the token line, not the whole file
                eval "$TOKEN_LINE"
                
                # Verify that token was loaded
                if [ -n "$HUGGINGFACE_TOKEN" ]; then
                    TOKEN_FOUND=true
                    break
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

echo "Using Hugging Face token: ${HUGGINGFACE_TOKEN:0:5}...${HUGGINGFACE_TOKEN: -5}"

# ===== HARDWARE OPTIMIZATION =====
echo "Setting up optimized environment for transcription..."

# GPU memory optimization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
export CUDA_LAUNCH_BLOCKING="1"

# CPU optimization for diarisation and fallback operations
CPU_THREADS=4  # Default for 4-core system

# Get CPU core count if possible
if command -v nproc &> /dev/null; then
    TOTAL_CORES=$(nproc)
    # Use 75% of cores (rounded down) but minimum 4, maximum 10
    CALCULATED_THREADS=$(( TOTAL_CORES * 3 / 4 ))
    CPU_THREADS=$(( CALCULATED_THREADS < 4 ? 4 : (CALCULATED_THREADS > 10 ? 10 : CALCULATED_THREADS) ))
    echo "Detected $TOTAL_CORES CPU cores, using $CPU_THREADS threads for CPU operations"
else
    echo "Using default thread count: $CPU_THREADS"
fi

# Set threading environment variables
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"

# General optimization
export TOKENIZERS_PARALLELISM="false"  # Prevent huggingface tokenizer warnings
export PYTHONOPTIMIZE="1"  # Optimize Python

# ===== CREATE OUTPUT DIRECTORY =====
# Create output directory if specified
if [[ "$*" =~ -o\ ([^\ ]+) || "$*" =~ --output\ ([^\ ]+) ]]; then
    OUTPUT_DIR=$(dirname "${BASH_REMATCH[1]}")
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
else
    # Default output directory
    echo "Creating default output directory"
    mkdir -p output
fi

# ===== CLEAR GPU MEMORY =====
echo "Clearing GPU memory..."
python -c "
import gc
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f'GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB')
except Exception as e:
    print(f'Could not clear GPU memory: {e}')
gc.collect()
" 2>/dev/null || true

# ===== RUN TRANSCRIPTION =====
echo "Starting GPU-accelerated transcription..."

# Record start time
START_TIME=$(date +%s)

# Run with basic error handling
if python transcript.py "$@"; then
    # Calculate elapsed time
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(( (ELAPSED % 3600) / 60 ))
    SECONDS=$((ELAPSED % 60))
    
    echo "---------------------------------------------------"
    echo "Transcription completed successfully!"
    echo "Total processing time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "---------------------------------------------------"
else
    echo "---------------------------------------------------"
    echo "ERROR: Transcription failed."
    echo "Try running with CPU only: CUDA_VISIBLE_DEVICES='' ./run_cpu_transcription.sh $@"
    echo "---------------------------------------------------"
    exit 1
fi