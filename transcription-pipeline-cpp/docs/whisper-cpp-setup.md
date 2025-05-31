# Whisper.cpp Setup - Historical Notes

**⚠️ This document is historical/archival. For current usage, see the main [README.md](../README.md).**

This document preserves the learning process and challenges encountered during the initial whisper.cpp implementation.

## Original Setup Challenges

When first implementing whisper.cpp, we encountered several challenges that have since been resolved:

1. **Deprecated main executable** - whisper.cpp moved from `main` to `whisper-cli`
2. **Missing build targets** - Had to explicitly build the CLI tool
3. **Large model memory issues** - Resolved by removing large-v3 (doesn't fit in 6GB)
4. **Progress tracking** - Initially had no progress indicators

## Key Learnings

### Model Selection for 6GB GPUs
After testing, we determined the optimal models for 6GB GPUs (like RTX 3050):
- `tiny.en` - Very fast, basic quality
- `base.en` - Fast, good quality
- `small.en` - Balanced speed/quality
- `medium.en` - Best quality that fits comfortably

The large-v3 model and its quantized versions were removed as they exceed 6GB memory limits.

### Prompt Optimization
Discovered that whisper.cpp has a ~2048 character limit for prompts, requiring condensed technical term lists.

### Docker Build Evolution

The Dockerfile evolved through several iterations:

1. **Initial version** - Basic CUDA setup, downloaded all models
2. **Memory optimization** - Removed large models, added specific CUDA architectures
3. **Build fix** - Added explicit whisper-cli target
4. **Current version** - Streamlined with only models that fit in 6GB

### Integration with Existing Pipeline

The whisper.cpp implementation was designed to slot into the existing pipeline:
- Maintains same directory structure
- Compatible with existing term corrections
- Produces similar output format to Python Whisper

### Performance Observations

On RTX 3050 (6GB VRAM):
- **Medium model** proved optimal - best quality/speed trade-off
- **Batch processing** works well overnight
- **Progress tracking** essential for long files

### Lessons for Future Implementations

1. **Always test Docker GPU access first**
2. **Start with smaller models** and work up
3. **Monitor GPU memory** during first runs
4. **Keep prompts under 2048 chars** for whisper.cpp
5. **Plan for progress indicators** from the start