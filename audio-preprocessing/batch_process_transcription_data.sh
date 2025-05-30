#!/bin/bash
#
# Batch Audio Preprocessing Script
#
# PURPOSE:
#     Batch process all MP4 files in the transcription-pipeline-cpp data folder,
#     converting them to optimized FLAC files for transcription and diarization.
#
# USAGE:
#     ./batch_process_transcription_data.sh
#
# WHAT IT DOES:
#     1. Finds all MP4 files in the raw data directory
#     2. Converts each to FLAC with 7-step preprocessing
#     3. Saves output to processed data directory
#     4. Shows progress for each file
#
# DIRECTORIES:
#     Input:  transcription-pipeline-cpp/data/raw/*.mp4
#     Output: transcription-pipeline-cpp/data/processed/*.flac
#
# PROCESSING APPLIED:
#     - MP4 to FLAC conversion
#     - 16kHz resampling
#     - Stereo to mono
#     - Noise reduction
#     - Volume normalization
#     - Dynamic compression
#     - Speech enhancement EQ
#
# NOTE: Creates output directory if it doesn't exist

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define input and output paths
INPUT_DIR="/home/shawn/Code/audio-transcription/transcription-pipeline-cpp/data/raw"
OUTPUT_DIR="/home/shawn/Code/audio-transcription/transcription-pipeline-cpp/data/processed"

# Display processing information
echo "=================================================="
echo "Audio Preprocessing Batch Process"
echo "=================================================="
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Run batch processing using the Python script
# -p "*.mp4" specifies the file pattern to match
python3 "$SCRIPT_DIR/simple_preprocess.py" batch "$INPUT_DIR" "$OUTPUT_DIR" -p "*.mp4"