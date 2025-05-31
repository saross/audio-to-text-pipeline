#!/bin/bash

# Wrapper script to ensure transcription logs go to the correct directory
# Usage: ./scripts/transcribe_with_logging.sh <same args as run_whisper.sh>

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/logs"

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_ROOT/logs/transcription_${TIMESTAMP}.log"

# Run the actual transcription script and log output
echo "Logging to: $LOG_FILE"
"$SCRIPT_DIR/run_whisper.sh" "$@" 2>&1 | tee "$LOG_FILE"

# Exit with the same code as the transcription script
exit ${PIPESTATUS[0]}