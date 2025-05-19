#!/bin/bash

# =============================================================================
# SETUP SCRIPT FOR AUDIO TRANSCRIPTION PIPELINE
# =============================================================================
#
# This script sets up the environment required for the audio preprocessing,
# transcription, and diarisation pipeline.
#
# It installs:
# 1. FFmpeg (if not already installed)
# 2. Python dependencies in a virtual environment
# 3. Configures access to Hugging Face models
#
# USAGE:
#   ./setup_transcription.sh
#
# REQUIREMENTS:
# - Python 3.8+ installed
# - Internet connection for downloading packages
# - System with apt package manager (for Ubuntu/Debian)
#   (modify for other systems as needed)
#
# =============================================================================

echo "======================================================================"
echo "Setting up Audio Transcription Pipeline Environment"
echo "======================================================================"

# Check for Python 3.8+
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required but not found"
    echo "Please install Python 3.8 or newer: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "ERROR: Python 3.8+ is required, but you have Python $PYTHON_VERSION"
    echo "Please install Python 3.8 or newer: https://www.python.org/downloads/"
    exit 1
fi

echo "Python $PYTHON_VERSION found"

# Check and install FFmpeg if needed
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg not found. Attempting to install..."
    
    # Detect OS and install FFmpeg
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        sudo apt-get update
        sudo apt-get install -y ffmpeg
    elif command -v brew &> /dev/null; then
        # macOS with Homebrew
        brew install ffmpeg
    elif command -v dnf &> /dev/null; then
        # Fedora/RHEL
        sudo dnf install -y ffmpeg
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        sudo yum install -y epel-release
        sudo yum install -y ffmpeg
    elif command -v pacman &> /dev/null; then
        # Arch Linux
        sudo pacman -S ffmpeg
    else
        echo "WARNING: Could not automatically install FFmpeg."
        echo "Please install FFmpeg manually: https://ffmpeg.org/download.html"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Verify FFmpeg installation
if command -v ffmpeg &> /dev/null; then
    echo "FFmpeg installed successfully: $(ffmpeg -version | head -n1)"
else
    echo "WARNING: FFmpeg installation could not be verified"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv ~/transcription-env

# Activate virtual environment
source ~/transcription-env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing Python dependencies..."
pip install openai-whisper torch numpy scipy librosa soundfile pydub tqdm

# Install pyannote.audio
echo "Installing pyannote.audio for speaker diarisation..."
pip install pyannote.audio

# ===== CHECK/REQUEST HUGGINGFACE TOKEN =====
echo "======================================================================"
echo "Hugging Face Token Setup"
echo "======================================================================"

# Check for existing token in environment variables or config files
HF_TOKEN=""
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    echo "Found HUGGINGFACE_TOKEN in environment variables."
    HF_TOKEN="$HUGGINGFACE_TOKEN"
else
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
                        HF_TOKEN="$HUGGINGFACE_TOKEN"
                        TOKEN_FOUND=true
                        break
                    fi
                fi
            fi
        fi
    done
fi

# Prompt for token only if not found
if [ -z "$HF_TOKEN" ]; then
    echo "No existing Hugging Face token found."
    echo "You need a Hugging Face token to access the diarisation models."
    echo "1. Go to: https://huggingface.co/settings/tokens"
    echo "2. Create a new token if you don't have one"
    echo "3. Enter your token below"
    echo ""
    
    read -p "Enter your Hugging Face token: " HF_TOKEN
    
    # Ask if user wants to save token to shell config
    if [ -n "$HF_TOKEN" ]; then
        read -p "Would you like to save this token for future use? (y/n): " SAVE_TOKEN
        if [[ "$SAVE_TOKEN" =~ ^[Yy]$ ]]; then
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
                    sed -i "s/export HUGGINGFACE_TOKEN=.*/export HUGGINGFACE_TOKEN=\"$HF_TOKEN\"/" "$SHELL_CONFIG"
                else
                    echo "" >> "$SHELL_CONFIG"
                    echo "# Hugging Face token for audio transcription" >> "$SHELL_CONFIG"
                    echo "export HUGGINGFACE_TOKEN=\"$HF_TOKEN\"" >> "$SHELL_CONFIG"
                fi
                echo "Token saved to $SHELL_CONFIG. It will be available in new terminal sessions."
            else
                echo "Couldn't determine shell config file. Please add manually:"
                echo "export HUGGINGFACE_TOKEN=\"$HF_TOKEN\""
            fi
        fi
    fi
else
    echo "Using existing Hugging Face token: ${HF_TOKEN:0:5}...${HF_TOKEN: -5}"
fi

# If token is still empty after all of that, exit
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: No Hugging Face token provided. This is required for the diarisation models."
    exit 1
fi

# Create environment setup script
echo "Creating environment setup script..."
cat > ~/transcription-env/setup_env.sh << 'EOF'
#!/bin/bash

# Environment setup for transcription pipeline
# Determine optimal thread count
echo "Detecting CPU cores..."
if command -v nproc &> /dev/null; then
    TOTAL_CORES=$(nproc)
    # Use 75% of cores but minimum 4
    CALCULATED_THREADS=$(( TOTAL_CORES * 3 / 4 ))
    CPU_THREADS=$(( CALCULATED_THREADS < 4 ? 4 : CALCULATED_THREADS ))
    echo "Detected $TOTAL_CORES CPU cores, configuring for $CPU_THREADS threads"
else
    CPU_THREADS=4
    echo "Could not detect CPU cores, using default of 4 threads"
fi

# Create environment setup script with dynamic thread count
echo "Creating environment setup script..."
cat > ~/transcription-env/setup_env.sh << EOF
#!/bin/bash

# Environment setup for transcription pipeline
export HUGGINGFACE_TOKEN="$HF_TOKEN"
export OMP_NUM_THREADS=$CPU_THREADS
export MKL_NUM_THREADS=$CPU_THREADS
export OPENBLAS_NUM_THREADS=$CPU_THREADS
export VECLIB_MAXIMUM_THREADS=$CPU_THREADS
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
export CUDA_LAUNCH_BLOCKING="1"
EOF

chmod +x ~/transcription-env/setup_env.sh

# Export for current session
export HUGGINGFACE_TOKEN="$HF_TOKEN"
export OMP_NUM_THREADS=$CPU_THREADS
export MKL_NUM_THREADS=$CPU_THREADS
export OPENBLAS_NUM_THREADS=$CPU_THREADS
export VECLIB_MAXIMUM_THREADS=$CPU_THREADS
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
export CUDA_LAUNCH_BLOCKING="1"

# Accept model licenses
echo "======================================================================"
echo "Accept Model Licenses"
echo "======================================================================"
echo "You need to accept the licenses for the pyannote.audio models."
echo "Please go to the following URLs and manually accept the licenses:"
echo "1. https://huggingface.co/pyannote/speaker-diarization-3.1"
echo "2. https://huggingface.co/pyannote/segmentation-3.0"
echo "3. https://huggingface.co/pyannote/embedding-3.1"
echo ""
echo "After accepting the licenses, the setup will be complete."
echo ""

read -p "Have you accepted the licenses? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please accept the licenses before using the scripts."
    exit 1
fi

# Make scripts executable
echo "Making scripts executable..."
chmod +x audio_preprocess.py
chmod +x preprocess_audio.sh
chmod +x transcribe_with_preprocessing.sh
chmod +x run_transcription.sh
chmod +x run_cpu_transcription.sh
chmod +x transcript.py
chmod +x transcript_cpu.py

echo "======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo "To activate the environment in a new terminal:"
echo "  source ~/transcription-env/bin/activate"
echo ""
echo "To verify your setup, run a test transcription:"
echo "  ./transcribe_with_preprocessing.sh /path/to/your/audio.mp4"
echo ""
echo "Happy transcribing!"
echo "======================================================================"
