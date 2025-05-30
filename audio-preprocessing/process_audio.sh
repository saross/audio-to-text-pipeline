#!/bin/bash
#
# Audio Preprocessing Wrapper Script
#
# PURPOSE:
#     Simple wrapper that forwards all arguments to the Python preprocessing script.
#     Provides a convenient shell interface for the preprocessing pipeline.
#
# USAGE:
#     Convert single file:
#         ./process_audio.sh convert input.mp4 output.flac
#     
#     Batch process directory:
#         ./process_audio.sh batch /input/dir /output/dir
#     
#     Test with 30 seconds:
#         ./process_audio.sh test input.mp4
#
# NOTES:
#     - All arguments are passed directly to simple_preprocess.py
#     - Run without arguments to see help message
#     - Requires Python 3 and FFmpeg

# Get the directory where this script is located
# This ensures the script works regardless of where it's called from
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Forward all arguments to the Python preprocessing script
# "$@" preserves all arguments exactly as provided
python3 "$SCRIPT_DIR/simple_preprocess.py" "$@"