# Troubleshooting Guide

## Common Issues and Solutions

### GPU Issues

#### GPU not detected
```bash
# Test Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# Test whisper GPU
./scripts/run_whisper.sh --gpu-test

# Test diarization GPU
./scripts/run_diarization.sh --gpu-test
```

**Solutions:**
- Ensure NVIDIA container toolkit is installed: `sudo apt-get install nvidia-container-toolkit`
- Restart Docker daemon: `sudo systemctl restart docker`
- Check NVIDIA drivers: `nvidia-smi`

#### Out of GPU memory

**For transcription:**
- Use smaller model (medium.en instead of large)
- Close other GPU applications
- Monitor with `nvidia-smi`

**For diarization:**
- Process shorter segments
- Ensure no other GPU processes running
- Falls back to CPU automatically

### Docker Build Issues

#### Build fails with "no space left on device"
- Need ~25GB during build
- Clean Docker cache: `docker system prune -a`
- Check disk space: `df -h`

#### Timezone prompt during build
- Already fixed with `ENV TZ=Australia/Melbourne`
- If persists, rebuild: `docker build --no-cache`

### Transcription Issues

#### Poor transcription quality
1. Use prompt file: `-p config/whisper_prompt.txt`
2. Try larger model: `-m medium.en`
3. Ensure audio is preprocessed (16kHz, mono)
4. Check audio quality - very noisy audio needs preprocessing

#### False error after successful transcription
- Known issue: "NameError: name 'output_dir' is not defined"
- **This is harmless** - transcription completed successfully
- Check output file exists with correct content
- Will be fixed in next rebuild

#### Transcription hangs
- Normal for long files (3-5 min per hour of audio)
- Check progress indicator for percentage
- If no progress after 10 minutes, check `docker ps`

### Diarization Issues

#### "HF_TOKEN not set"
```bash
# Set token
export HF_TOKEN=your_huggingface_token

# Get token from:
# https://huggingface.co/settings/tokens

# Accept model conditions at:
# https://huggingface.co/pyannote/speaker-diarization-3.1
```

#### Diarization takes too long
- Normal: 10-15 minutes per hour of audio with GPU
- Check if using GPU: `nvidia-smi` should show python process
- For 2-3 hour files, expect 30-45 minutes

#### Wrong number of speakers detected
- Specify manually: `-n 2` for 2 speakers
- Adjust min/max: `--min-speakers 2 --max-speakers 4`
- Very similar voices may be merged

### Merge Issues

#### Speaker names not applied
- Use without quotes: `--speaker-names Alice,Bob`
- Or single quotes: `--speaker-names 'Alice,Bob'`
- Names map in order to SPEAKER_0, SPEAKER_1, etc.

#### Segments don't align with transcript
- Ensure using same audio file for both
- Transcript must have timestamps for best results
- Check diarization JSON has segments

### Audio Preprocessing Issues

#### FFmpeg crashes on large files
- Use simple_preprocess.py (streaming, no memory loading)
- Process one file at a time
- Check available disk space

#### Processed files too large
- Ensure using compression level 8 (maximum)
- Consider --remove-silence flag
- 90% reduction is typical

### Performance Issues

#### Everything runs slowly
- Check if using GPU: `nvidia-smi`
- Transcription: ~3-5 min/hour with GPU, ~20-30 min/hour CPU
- Diarization: ~10-15 min/hour with GPU, ~30-60 min/hour CPU

#### Progress indicators not showing
- Rebuild images to get latest updates
- For transcription: shows percentage when available
- For diarization: shows progress bar

## Quick Diagnostic Commands

```bash
# Check Docker
docker --version
docker ps

# Check GPU
nvidia-smi
nvidia-container-cli info

# Check disk space
df -h

# Check memory
free -h

# Check processes
htop

# Check audio file
ffprobe -i audio.flac

# Test components
./scripts/run_whisper.sh --gpu-test
./scripts/run_diarization.sh --gpu-test
```

## Getting Help

If issues persist:

1. Check error messages carefully
2. Look for output files even if error shown
3. Try with a small test file first
4. Run with verbose flag: `-v` or `--verbose`
5. Check Docker logs: `docker logs <container_id>`

## Known Limitations

- Whisper.cpp doesn't include timestamps in output (use whisper Python for that)
- Diarization may struggle with very similar voices
- Phone quality audio may have reduced accuracy
- Background music can interfere with speech detection