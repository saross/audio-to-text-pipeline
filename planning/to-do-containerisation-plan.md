# Containerisation Plan for the Transcription Pipeline

## Introduction: Why Containerisation Matters for This Project

Containerisation offers significant benefits for your transcription pipeline, particularly for research applications:

- **Reproducibility**: Ensures the exact same environment across different systems
- **Portability**: Run on different systems (cloud VMs, Raspberry Pi, etc.) without reconfiguration
- **Dependency isolation**: Avoid conflicts between different software versions
- **Scalability**: Easily scale to process multiple files in parallel
- **Version control**: Package specific versions of the pipeline for different research projects

For computational research specifically, containerisation addresses the "works on my machine" problem by providing a consistent, documented execution environment that others can use to verify your results.

## Learning Path for Containerisation

### Stage 1: Docker Basics (2-3 weeks)
**Goal**: Learn Docker fundamentals and create basic container for the pipeline

**Activities**:
- Install Docker on your development machine
- Learn basic Docker commands
- Create a simple Dockerfile for the pipeline
- Build and test the initial container locally

### Stage 2: Multi-Container Architecture (2-3 weeks)
**Goal**: Create a more sophisticated containerised solution with separate services

**Activities**:
- Learn Docker Compose for multi-container applications
- Separate preprocessing, transcription, and storage into distinct services
- Handle volume mapping for input/output data
- Create a unified workflow across containers

### Stage 3: Optimisation & GPU Support (2-3 weeks)
**Goal**: Optimise containers and add GPU support

**Activities**:
- Learn about GPU access in Docker (NVIDIA Docker)
- Optimise container size and performance
- Implement caching strategies
- Configure appropriate resource limits

### Stage 4: Deployment (2-3 weeks)
**Goal**: Deploy to different environments including Raspberry Pi

**Activities**:
- Build multi-architecture images (x86_64 and ARM)
- Set up deployment workflows 
- Test on cloud VMs and Raspberry Pi
- Document deployment process for others

## Containerisation Architecture for Transcription Pipeline

### Proposed Architecture

```
┌─────────────────────────────────────────────────────┐
│                Docker Host System                    │
│                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────┐ │
│  │ Preprocessing│    │Transcription│    │ Storage/ │ │
│  │   Service    │    │   Service   │    │ Results  │ │
│  │             │    │ (GPU/CPU)   │    │          │ │
│  └──────┬──────┘    └──────┬──────┘    └─────┬────┘ │
│         │                  │                 │      │
│  ┌──────┴──────────────────┴─────────────────┴────┐ │
│  │               Shared Volumes                    │ │
│  │  ┌──────────────┐  ┌───────────────┐  ┌──────┐ │ │
│  │  │  Input Audio │  │ Processed Audio│  │Results│ │ │
│  │  └──────────────┘  └───────────────┘  └──────┘ │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### Key Components:

1. **Preprocessing Service Container**:
   - Contains audio preprocessing tools
   - Handles format conversion, noise reduction, etc.
   - CPU-based processing only

2. **Transcription Service Container**:
   - Contains Whisper/WhisperX and diarisation models
   - Can be configured for GPU or CPU processing
   - Optionally has model variants (lightweight for Raspberry Pi)

3. **Storage/Results Container** (optional):
   - Manages results database/files
   - Provides web interface for browsing transcripts
   - Handles exporting to different formats

4. **Shared Volumes**:
   - Input audio files (mounted from host)
   - Intermediate processed audio
   - Output transcripts and metadata

### Container Coordination:

1. **Simple Approach**: Shell scripts to orchestrate container execution
2. **Advanced Approach**: Docker Compose for defining multi-container applications
3. **Enterprise Approach**: Kubernetes for orchestration at scale (future consideration)

## Implementation Plan

### Stage 1: Basic Dockerfile

First, create a single container that includes the entire pipeline:

```dockerfile
# Base image with Python and audio tools
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline code
COPY transcription_pipeline/ /app/transcription_pipeline/
COPY scripts/ /app/scripts/

# Make scripts executable
RUN chmod +x /app/scripts/*.sh

# Default command
ENTRYPOINT ["/app/scripts/transcribe_with_preprocessing.sh"]
CMD ["--help"]
```

#### Creating a requirements.txt file:

```
openai-whisper
torch
pyannote.audio
librosa
soundfile
pydub
tqdm
numpy
scipy
```

### Stage 2: Multi-Container Setup with Docker Compose

After getting a basic container working, separate concerns with Docker Compose:

```yaml
# docker-compose.yml
version: '3.8'

services:
  preprocess:
    build: 
      context: .
      dockerfile: Dockerfile.preprocess
    volumes:
      - ./input:/input
      - processed_audio:/processed_audio
    command: "./scripts/preprocess_audio.sh /input/audio.mp3 -o /processed_audio/processed.flac"

  transcribe:
    build:
      context: .
      dockerfile: Dockerfile.transcribe
    volumes:
      - processed_audio:/input
      - ./output:/output
    command: "./scripts/run_transcription.sh /input/processed.flac -o /output/transcript.txt"

volumes:
  processed_audio:
```

### Stage 3: Adding GPU Support

For GPU-accelerated transcription, modify the Dockerfile:

```dockerfile
# Dockerfile.transcribe.gpu
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy pipeline code
COPY transcription_pipeline/ /app/transcription_pipeline/
COPY scripts/ /app/scripts/

# Make scripts executable
RUN chmod +x /app/scripts/*.sh

# Default command
ENTRYPOINT ["/app/scripts/run_transcription.sh"]
CMD ["--help"]
```

And update the Docker Compose file to use GPU:

```yaml
# In docker-compose.yml
transcribe:
  build:
    context: .
    dockerfile: Dockerfile.transcribe.gpu
  volumes:
    - processed_audio:/input
    - ./output:/output
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  command: "./scripts/run_transcription.sh /input/processed.flac -o /output/transcript.txt"
```

### Stage 4: Multi-Architecture Support for Raspberry Pi

Build images for both x86_64 (standard computers) and ARM (Raspberry Pi):

```bash
# Build and push multi-arch images
docker buildx create --name multiarch --driver docker-container --use
docker buildx build --platform linux/amd64,linux/arm64 -t yourusername/transcription:latest -f Dockerfile --push .
```

Create a lightweight version for Raspberry Pi:

```dockerfile
# Dockerfile.rpi
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies - optimized for Raspberry Pi
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies - use smaller models
COPY requirements.rpi.txt .
RUN pip install --no-cache-dir -r requirements.rpi.txt

# Copy pipeline code
COPY transcription_pipeline/ /app/transcription_pipeline/
COPY scripts/ /app/scripts/

# Make scripts executable
RUN chmod +x /app/scripts/*.sh

# Configure for lower resource usage
ENV MODEL_SIZE="tiny"
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=2

# Default command
ENTRYPOINT ["/app/scripts/transcribe_with_preprocessing.sh"]
CMD ["--help"]
```

## Using the Containerised Pipeline

### Basic Usage

```bash
# Simple run with input and output directories
docker run -v /path/to/audio:/input -v /path/to/output:/output \
  transcript-pipeline:latest /input/audio.mp3 -o /output/transcript.txt
```

### Using Docker Compose

```bash
# Start the pipeline
docker-compose up

# Process a specific file
docker-compose run --rm preprocess ./scripts/preprocess_audio.sh /input/audio.mp3 -o /processed_audio/processed.flac
docker-compose run --rm transcribe ./scripts/run_transcription.sh /processed_audio/processed.flac -o /output/transcript.txt
```

### Raspberry Pi Deployment

```bash
# Pull the ARM version
docker pull yourusername/transcription:latest

# Run on Raspberry Pi
docker run -v /path/to/audio:/input -v /path/to/output:/output \
  yourusername/transcription:latest /input/audio.mp3 -o /output/transcript.txt --cpu-only
```

## Enhancing the Pipeline with Container Orchestration

As your containerised pipeline grows, you might want to explore more advanced orchestration:

### For Single-Machine Deployment

Docker Compose is sufficient for managing multiple containers on a single machine.

### For Multi-Machine Deployment

Kubernetes provides more advanced orchestration capabilities:

- **Resource Management**: Efficient allocation of CPU/RAM/GPU
- **Scaling**: Automatically scale containers based on load
- **Self-Healing**: Restart failed containers
- **Storage Orchestration**: Manage persistent storage across machines

For a Raspberry Pi cluster, K3s (lightweight Kubernetes) is ideal:

```bash
# Install K3s on the main Raspberry Pi
curl -sfL https://get.k3s.io | sh -

# Get the token
TOKEN=$(sudo cat /var/lib/rancher/k3s/server/node-token)

# Install as agent on other Raspberry Pis
curl -sfL https://get.k3s.io | K3S_URL=https://main-pi:6443 K3S_TOKEN=$TOKEN sh -
```

## Reproducibility for Research

Containerisation significantly enhances research reproducibility:

1. **Environment Documentation**: Dockerfile provides precise documentation of all dependencies
2. **Version Pinning**: Exact versions of all software components are captured
3. **Portability**: Same container works across different computing environments
4. **Long-term Preservation**: Container images can be archived with research outputs
5. **Collaboration**: Easy sharing of exact research environments with colleagues

### Research Data Management Best Practices

When using containers for research:

1. **Tag container images with DOIs**: Reference specific versions in publications
2. **Archive containers with data**: Store container images alongside research data
3. **Document container usage**: Include container run commands in methodology
4. **Verify reproducibility**: Test containers in different environments before publishing

## Progressive Learning Resources

### Stage 1: Docker Basics

1. **Docker's Official Getting Started**: https://docs.docker.com/get-started/
2. **Docker for Beginners**: https://docker-curriculum.com/
3. **Basic Docker Commands Cheatsheet**: https://www.docker.com/sites/default/files/d8/2019-09/docker-cheat-sheet.pdf

### Stage 2: Docker Compose

1. **Docker Compose Overview**: https://docs.docker.com/compose/
2. **Docker Compose in Production**: https://docs.docker.com/compose/production/
3. **Docker Compose Cheatsheet**: https://devhints.io/docker-compose

### Stage 3: GPU Support & Optimisation

1. **NVIDIA Docker Documentation**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html
2. **Docker Optimisation Guide**: https://docs.docker.com/develop/develop-images/dockerfile_best-practices/
3. **Multi-stage Builds**: https://docs.docker.com/build/building/multi-stage/

### Stage 4: Multi-Architecture & Deployment

1. **Docker Buildx**: https://docs.docker.com/buildx/working-with-buildx/
2. **Multi-architecture Images**: https://www.docker.com/blog/multi-arch-build-and-images-the-simple-way/
3. **Raspberry Pi Docker**: https://www.raspberrypi.com/news/docker-comes-to-raspberry-pi/

### Advanced: Kubernetes (if needed)

1. **K3s for Edge Computing**: https://k3s.io/
2. **Raspberry Pi Kubernetes Cluster**: https://ubuntu.com/tutorials/how-to-kubernetes-cluster-on-raspberry-pi

## Conclusion: From Development to Reproducible Research

Containerising your transcription pipeline creates a bridge between software development and scientific reproducibility. The approach outlined here will:

1. Make your research methods transparent and repeatable
2. Ensure consistent results across different computing environments
3. Simplify deployment to cloud computing and Raspberry Pi clusters
4. Create shareable components that other researchers can build upon

By following this containerisation plan alongside your modular architecture and testing regime, you'll create not just a useful tool, but a fully reproducible research asset that demonstrates best practices in computational science.
