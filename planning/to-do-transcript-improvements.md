# Transcript Pipeline Improvements & WhisperX Migration To-Do List

## Current Improvements for Existing Pipeline

### 1. Empty UNKNOWN Segment Removal
```python
# Filter out empty UNKNOWN segments
print("Removing empty UNKNOWN speaker segments...")
merged_segments = [segment for segment in merged_segments 
                   if not (segment["speaker"] == "UNKNOWN" and not segment["text"].strip())]
```

## Planned Improvements When Migrating to WhisperX

### 1. Named Speaker Identification
Add speaker name mapping to replace generic SPEAKER_XX labels:
```python
# Add to script arguments
parser.add_argument("--speaker-names", "-n", help="Comma-separated list of speaker names (e.g., 'John,Sarah')")

# Modify output section
if args.speaker_names:
    speaker_map = {f"SPEAKER_{i:02d}": name for i, name in enumerate(args.speaker_names.split(','))}
    speaker_label = speaker_map.get(segment["speaker"], segment["speaker"])
else:
    speaker_label = segment["speaker"]
```

### 2. Non-Speech Audio Annotation
Add markers for laughter, pauses, etc:
```python
def add_non_speech_markers(text):
    """Add markers for detected non-speech elements."""
    patterns = [
        (r'\b(haha|hehe|ha ha)\b', '[laughter]'),
        (r'\b(hmm|hmmmm|mm-hmm)\b', '[agreement]'),
        (r'\buh+\b', '[hesitation]'),
        (r'\.\.\.|…', '[pause]')
    ]
    
    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    return result
```

### 3. Confidence Indicators for Uncertain Transcriptions
Mark low-confidence sections:
```python
# Add to combined_segment dictionary
combined_segment = {
    # other fields...
    "confidence": segment.get("confidence", 0)
}

# In output section
if segment.get("confidence", 1.0) < 0.75:
    f.write(f"{timestamp} {segment['speaker']}: [uncertain] {segment['text']}\n\n")
```

### 4. Multiple Output Formats
Support for SRT, JSON, etc:
```python
parser.add_argument("--format", choices=["txt", "json", "srt"], default="txt",
                   help="Output format (default: txt)")
```

### 5. Comprehensive Metadata Header
```python
f.write(f"# Transcript of {os.path.basename(audio_file)}\n")
f.write(f"# Transcribed on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
f.write(f"# Model: WhisperX {model_size}\n")
f.write(f"# Duration: {audio_duration:.2f} seconds\n")
f.write(f"# Speakers: {num_speakers if num_speakers else 'auto-detected'}\n\n")
```

### 6. Paragraph Breaks Based on Silence
```python
parser.add_argument("--paragraph-break", type=float, default=2.0,
                   help="Seconds of silence to trigger a paragraph break (default: 2.0)")

# In output section
prev_segment_end = 0
for segment in merged_segments:
    if segment["start"] - prev_segment_end > args.paragraph_break:
        f.write("\n")  # Extra line for paragraph break
    # Write segment as normal
    prev_segment_end = segment["end"]
```

### 7. Minimum Duration Filtering
```python
parser.add_argument("--min-duration", type=float, default=0.5,
                   help="Minimum duration in seconds for a segment (default: 0.5)")

filtered_segments = [s for s in merged_segments if s["end"] - s["start"] >= args.min_duration]
```

### 8. Speech Rate Indicator
```python
# Calculate words per minute
for segment in merged_segments:
    duration = segment["end"] - segment["start"]
    word_count = len(segment["text"].split())
    if duration > 0:
        wpm = word_count / duration * 60
        if wpm > 180:
            segment["text"] = "[fast] " + segment["text"]
        elif wpm < 100:
            segment["text"] = "[slow] " + segment["text"]
```

## WhisperX Migration Notes

### Benefits of WhisperX
- Significantly faster processing (sometimes 4-10x faster than standard Whisper)
- Better speaker diarisation integration
- More accurate word-level timestamps
- Better handling of overlapping speech

### Installation Requirements
```bash
# Basic installation
pip install git+https://github.com/m-bain/whisperx.git

# Dependencies
pip install torch torchvision torchaudio pyannote.audio
```

### Sample WhisperX Usage Pattern
```python
import whisperx
import torch

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisperx.load_model("large-v3", device)

# Transcribe with word-level timestamps
result = model.transcribe("audio.mp3", batch_size=16)
segments = result["segments"]

# Speaker diarisation with WhisperX
diarize_model = whisperx.load_diarization_model(device)
diarization = diarize_model({"audio": "audio.mp3"})

# Assign speaker labels
result = whisperx.assign_speakers(segments, diarization)
final_segments = result["segments"]
```

### Additional WhisperX Options to Explore
- VAD (Voice Activity Detection) filters for better silence handling
- Speaker embedding for improved speaker differentiation
- Language identification for multilingual content
- Different batch sizes for performance tuning

### Compatibility Notes
- Review current term_corrections.txt to ensure compatibility
- Ensure any custom post-processing is adapted to WhisperX's output format
- Test memory requirements as WhisperX may have different GPU memory usage patterns

### References
- WhisperX GitHub: https://github.com/m-bain/whisperx
- Pyannote Audio: https://github.com/pyannote/pyannote-audio
