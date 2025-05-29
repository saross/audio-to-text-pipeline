# Use the original PyTorch image that worked for basic GPU test
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0
ENV OMP_NUM_THREADS=4

WORKDIR /workspace

# Skip system packages - not needed for FLAC
RUN pip install --upgrade pip wheel

# Create requirements file inline with compatible versions
RUN echo "faster-whisper==0.7.1" > requirements.txt && \
    echo "librosa==0.10.1" >> requirements.txt && \
    echo "soundfile==0.12.1" >> requirements.txt && \
    echo "pydub==0.25.1" >> requirements.txt && \
    echo "numpy<1.24" >> requirements.txt && \
    echo "scipy>=1.7.0" >> requirements.txt && \
    echo "tqdm>=4.60.0" >> requirements.txt && \
    echo "pyyaml>=6.0" >> requirements.txt && \
    echo "psutil>=5.8.0" >> requirements.txt && \
    echo "nvidia-ml-py3" >> requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

RUN chmod +x scripts/*.py
RUN mkdir -p /workspace/models

ENTRYPOINT ["python", "scripts/transcribe_pipeline.py"]
CMD ["--help"]