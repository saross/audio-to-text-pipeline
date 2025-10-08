# Whisper.cpp Docker Investigation Results

## Summary

The whisper.cpp project has deprecated the `main` executable in favor of `whisper-cli`. The Docker build process needed to be updated to explicitly build this new executable.

## Key Findings

1. **Deprecated Executable**: The `main` executable shows a deprecation warning and doesn't function properly
2. **Missing Build Target**: The original Dockerfile didn't build the `whisper-cli` executable
3. **Build Location**: `whisper-cli` is built to `/app/whisper.cpp/build/bin/whisper-cli`
4. **CMake Target**: Need to explicitly run `cmake --build build --target whisper-cli`

## Solution

### Updated Dockerfile

The Dockerfile was updated to:
1. Enable building examples with `-DWHISPER_BUILD_EXAMPLES=ON`
2. Explicitly build the `whisper-cli` target after the main build
3. Verify the executable exists after building

### Updated transcribe.py

The Python wrapper was updated to:
1. Look for `whisper-cli` in the correct location first
2. Handle the deprecated `main` executable gracefully
3. Suppress stderr when using the deprecated binary

## Docker Commands for Testing

```bash
# List executables in the container
docker run --rm --entrypoint /bin/bash whisper-cpp-cuda -c "ls -la /app/whisper.cpp/build/bin/"

# Find whisper executables
docker run --rm --entrypoint /bin/bash whisper-cpp-cuda -c "find /app/whisper.cpp -name '*whisper*' -type f -executable"

# Test whisper-cli help
docker run --rm --entrypoint /bin/bash whisper-cpp-cuda -c "/app/whisper.cpp/build/bin/whisper-cli --help"

# Check GPU support
docker run --rm --gpus all --entrypoint /bin/bash whisper-cpp-cuda -c "nvidia-smi -L"
```

## Building whisper-cli Manually

If the Dockerfile doesn't build whisper-cli, it can be built manually:

```bash
# Start a container
docker run -d --entrypoint /bin/bash --name whisper-temp whisper-cpp-cuda -c "sleep 3600"

# Build whisper-cli
docker exec whisper-temp bash -c "cd /app/whisper.cpp/build && cmake --build . --target whisper-cli --config Release"

# Commit the changes
docker commit whisper-temp whisper-cpp-cuda:latest

# Clean up
docker stop whisper-temp && docker rm whisper-temp
```

## Transcription Usage

With the updated container, transcriptions can be run as:

```bash
docker run --rm \
    -v "$(pwd)/data:/data" \
    -v "$(pwd)/results:/results" \
    -v "$(pwd)/config:/config" \
    --gpus all \
    whisper-cpp-cuda \
    /data/audio.wav \
    -o /results/output.txt \
    -m medium.en \
    -t 4 \
    -p /config/whisper_prompt.txt
```

## Notes

- GPU acceleration is enabled with CUDA support
- The container includes models: tiny.en, base.en, small.en, medium.en
- Prompts can be provided to guide transcription accuracy
- The Python wrapper handles output file management and GPU detection