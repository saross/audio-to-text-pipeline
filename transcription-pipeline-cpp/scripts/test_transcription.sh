#!/bin/bash

# Test the complete transcription pipeline

echo "=== Testing Whisper-CPP Docker Transcription Pipeline ==="

# Create test audio
echo -e "\n1. Creating test audio file..."
docker run --rm \
    -v "$(pwd)/data:/data" \
    --entrypoint ffmpeg \
    whisper-cpp-cuda \
    -f lavfi -i "sine=frequency=1000:duration=3" \
    -c:a pcm_s16le -ar 16000 \
    -y /data/test_tone.wav

# Test transcription with Python wrapper
echo -e "\n2. Testing transcription with Python wrapper..."
docker run --rm \
    -v "$(pwd)/data:/data" \
    -v "$(pwd)/results:/results" \
    -v "$(pwd)/config:/config" \
    --gpus all \
    whisper-cpp-cuda \
    /data/test_tone.wav \
    -o /results/test_output.txt \
    -m tiny.en \
    -t 4

# Check results
echo -e "\n3. Checking results..."
if [ -f "results/test_output.txt" ]; then
    echo "Success! Transcription completed."
    echo "Output:"
    cat results/test_output.txt
else
    echo "Failed: No output file created"
    ls -la results/
fi

# Clean up
echo -e "\n4. Cleaning up test files..."
rm -f data/test_tone.wav results/test_output.txt

echo -e "\n=== Test Complete ==="