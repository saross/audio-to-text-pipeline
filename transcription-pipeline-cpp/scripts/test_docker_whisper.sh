#!/bin/bash

# Test script for whisper-cpp Docker container

echo "Testing whisper-cpp Docker container..."

# Check what executables are available
echo -e "\n1. Checking available executables in /app/whisper.cpp/build/bin/:"
docker run --rm --entrypoint /bin/bash whisper-cpp-cuda -c "ls -la /app/whisper.cpp/build/bin/"

echo -e "\n2. Checking for whisper-cli specifically:"
docker run --rm --entrypoint /bin/bash whisper-cpp-cuda -c "find /app/whisper.cpp/build -name 'whisper-cli' -type f -ls"

echo -e "\n3. Testing whisper-cli help (if available):"
docker run --rm --entrypoint /bin/bash whisper-cpp-cuda -c "/app/whisper.cpp/build/bin/whisper-cli --help 2>&1 | head -20" || echo "whisper-cli not found"

echo -e "\n4. Testing fallback main executable:"
docker run --rm --entrypoint /bin/bash whisper-cpp-cuda -c "/app/whisper.cpp/build/bin/main 2>&1 | grep -v 'deprecated' | head -20" || echo "main not working"

echo -e "\n5. Checking GPU support:"
docker run --rm --gpus all --entrypoint /bin/bash whisper-cpp-cuda -c "nvidia-smi -L" || echo "GPU not available"

echo -e "\n6. Testing transcription wrapper script:"
docker run --rm whisper-cpp-cuda --help

echo -e "\nDone!"