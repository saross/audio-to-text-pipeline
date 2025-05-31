# Technical Details

## Architecture Overview

The pipeline consists of three main components running in separate Docker containers:

1. **Audio Preprocessing** - Python/FFmpeg based
2. **Transcription** - whisper.cpp with CUDA
3. **Speaker Diarization** - pyannote.audio with PyTorch

## Implementation Details

### Audio Preprocessing

**Technology Stack:**
- FFmpeg for audio processing
- Python for orchestration
- No audio libraries (avoids memory issues)

**Processing Pipeline:**
```
MP4 → FFmpeg → FLAC
     ↓
   Filters:
   - Resample to 16kHz
   - Convert to mono
   - Noise reduction
   - Normalization
   - Compression (3:1)
   - EQ enhancement
```

**Key Design Decisions:**
- Stream processing (no full file loading)
- FLAC compression level 8 for maximum compression
- 3:1 compression ratio (gentler for diarization)
- 16kHz sample rate (Whisper training standard)

### Transcription (whisper.cpp)

**Docker Image:**
- Base: `nvidia/cuda:12.2.0-devel-ubuntu22.04`
- Whisper.cpp compiled with CUDA support
- Models: tiny.en, base.en, small.en, medium.en

**GPU Optimization:**
```cmake
-DGGML_CUDA=ON
-DCMAKE_CUDA_ARCHITECTURES="60;70;75;80;86"
```

**Binary Evolution:**
- Originally used `main` (now deprecated)
- Migrated to `whisper-cli`
- Python wrapper handles both for compatibility

**Progress Tracking:**
- Monitors whisper.cpp output for "progress = X%"
- Falls back to time-based spinner
- Updates every 0.5 seconds

### Speaker Diarization

**Docker Image:**
- Base: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
- pyannote.audio 3.1.1
- Automatic GPU detection

**Pipeline Stages:**
1. Voice Activity Detection (VAD)
2. Speaker embedding extraction
3. Clustering
4. Resegmentation

**Output Formats:**
- JSON: Structured segments with timestamps
- RTTM: Standard evaluation format
- Simple: Human-readable text

### Integration Layer

**Merge Algorithm:**
1. Parse transcript segments
2. Find speaker at segment midpoint
3. Assign speaker label
4. Optional: Apply custom names

**File Handling:**
- All paths converted to absolute
- Docker volume mounts for data access
- Automatic output directory creation

## Performance Characteristics

### GPU Memory Usage

| Component | Model/Mode | GPU Memory |
|-----------|------------|------------|
| Whisper tiny.en | - | ~1 GB |
| Whisper base.en | - | ~1.5 GB |
| Whisper small.en | - | ~2 GB |
| Whisper medium.en | - | ~3 GB |
| Diarization | pyannote 3.1 | ~2.5 GB |

### Processing Speed (with RTX 3050 6GB)

| Task | GPU | CPU |
|------|-----|-----|
| Preprocessing | N/A | ~1 min/hour |
| Transcription | ~3-5 min/hour | ~20-30 min/hour |
| Diarization | ~10-15 min/hour | ~30-60 min/hour |

### Optimization Techniques

**Memory Optimization:**
- Stream processing for large files
- Chunked reading in diarization
- Docker memory limits prevent OOM

**Speed Optimization:**
- GPU acceleration where possible
- Parallel processing in preprocessing
- Optimized FFmpeg filter chains

**Quality Optimization:**
- Custom prompts for domain terms
- Post-processing corrections
- Careful audio preprocessing

## Docker Implementation

### Volume Mounts

```bash
docker run --rm --gpus all \
  -v "$INPUT_DIR:/input:ro" \    # Read-only input
  -v "$OUTPUT_DIR:/output" \      # Writable output
  -v "$CONFIG_DIR:/config:ro" \   # Read-only config
  image_name
```

### GPU Passthrough

- Uses `--gpus all` flag
- Requires nvidia-container-toolkit
- Falls back to CPU if unavailable

### Image Layers

1. Base CUDA/PyTorch image
2. System dependencies
3. Python/compiled binaries
4. Application scripts

## File Formats

### Whisper Prompt Format
```
This is a technical interview about semantic web technologies.
Technical terms: SKOS, RDF, SPARQL, ontology...
```
- Maximum ~2048 characters
- Guides model vocabulary
- Improves technical term accuracy

### Term Corrections Format
```
# Pattern matching replacements
sissvok -> SISSVoc
rda -> RDA
ardc -> ARDC

# Case-sensitive exact matches
/Exact:pool party -> PoolParty
/Exact:RDF -> RDF
```

### Diarization JSON Format
```json
{
  "speakers": ["SPEAKER_0", "SPEAKER_1"],
  "num_speakers": 2,
  "segments": [
    {
      "speaker": "SPEAKER_0",
      "start": 0.5,
      "end": 10.2,
      "duration": 9.7
    }
  ]
}
```

## Security Considerations

- Docker containers run with minimal privileges
- Input directories mounted read-only
- No network access during processing
- HF_TOKEN passed as environment variable

## Future Architecture Considerations

### Potential Improvements

1. **Streaming Pipeline**
   - Real-time transcription
   - Live diarization
   - WebSocket output

2. **Distributed Processing**
   - Kubernetes deployment
   - Job queue system
   - Parallel file processing

3. **Model Optimization**
   - Quantized models for larger sizes
   - Fine-tuned models for domain
   - Speaker embeddings database

### Scalability

Current architecture scales by:
- Processing files sequentially
- Using larger GPU for speed
- Batching overnight

Could scale to:
- Multiple GPU workers
- Cloud deployment
- API service