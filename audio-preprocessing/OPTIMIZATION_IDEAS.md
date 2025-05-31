# Potential Preprocessing Optimizations for Whisper.cpp and Pyannote

## Current Pipeline Performance
- File sizes: 50-115 MB (good range for ~1-2 hour interviews)
- Processing speed: Much faster than before
- No memory crashes

## Potential Improvements for Better Transcription & Diarization

### 1. **Voice Activity Detection (VAD) Pre-filtering**
```bash
# Add to filter chain:
"silenceremove=start_periods=1:start_duration=0.5:start_threshold=-40dB:stop_periods=-1:stop_duration=0.5:stop_threshold=-40dB"
```
- **Benefit**: Removes long silences, reducing file size and processing time
- **Risk**: May cut off very quiet speakers
- **Recommendation**: Test carefully with your audio

### 2. **Adaptive Noise Reduction**
```bash
# Replace current noise gate with:
"anlmdn=s=7:p=0.002:r=0.002:m=15"  # FFmpeg's AI noise reduction
```
- **Benefit**: Better at removing consistent background noise
- **Risk**: Requires newer FFmpeg version
- **Alternative**: Use RNNoise if available

### 3. **Whisper.cpp Specific Optimizations**

#### A. Optimal Sample Rate
- Current: 16kHz ✓ (Whisper's native rate)
- No change needed

#### B. Pre-emphasis Filter
```bash
# Add to enhance high frequencies:
"equalizer=f=4000:t=h:width=2000:g=1"
```
- Slight boost to consonants that Whisper sometimes misses

### 4. **Pyannote Diarization Optimizations**

#### A. Preserve Speaker Characteristics
Current compression (4:1) is good. For better diarization:
```bash
# Gentler compression for better speaker separation:
"acompressor=threshold=-24dB:ratio=3:attack=10:release=100"
```

#### B. Preserve Frequency Range
```bash
# Widen the frequency range slightly:
"highpass=f=70,lowpass=f=10000"  # Instead of implied 8kHz cutoff
```

#### C. Channel Separation (if stereo has different speakers)
```bash
# If left/right channels have different speakers:
# Process channels separately before mixing
```

### 5. **Two-Stage Processing Option**

Create two versions:
1. **Whisper version** (current settings) - Optimized for transcription
2. **Diarization version** - Less processing, preserves speaker characteristics

### 6. **Smart Normalization**
```bash
# Replace loudnorm with dual-stage:
"dynaudnorm=f=150:g=15:p=0.95,loudnorm=I=-16:TP=-1.5:LRA=11"
```
- Dynamic normalization before loudness normalization
- Better handles varying speaker volumes

## Recommended Minimal Changes

For immediate improvement without risk:

```python
# In simple_preprocess.py, modify the filter chain:

# Add after the current noise gate:
"silenceremove=start_periods=1:start_duration=0.3:start_threshold=-35dB",

# Modify compression for better diarization:
"acompressor=threshold=-22dB:ratio=3:attack=10:release=100",

# Add slight pre-emphasis for Whisper:
"equalizer=f=4000:t=h:width=2000:g=1",
```

## Testing Recommendations

1. Process a sample file with current vs modified settings
2. Run through Whisper.cpp and compare WER (Word Error Rate)
3. Run through pyannote and check DER (Diarization Error Rate)
4. Only adopt changes that improve both metrics

## Advanced Options

### GPU-Accelerated Preprocessing
If you have NVIDIA GPU with FFmpeg CUDA support:
```bash
ffmpeg -hwaccel cuda -i input.mp4 ...
```

### Parallel Processing
For batch jobs, process multiple files simultaneously:
```python
from concurrent.futures import ProcessPoolExecutor
```

Would you like me to implement any of these optimizations?