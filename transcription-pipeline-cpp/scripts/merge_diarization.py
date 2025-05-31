#!/usr/bin/env python3
"""
Merge Speaker Diarization with Transcripts

PURPOSE:
    Combines speaker diarization results with whisper transcripts to produce
    speaker-labeled transcripts. Works with the existing whisper.cpp output.

USAGE:
    python3 merge_diarization.py <transcript_file> <diarization_file> [options]
    
    Options:
        -o, --output FILE       Output file (default: transcript_speakers.txt)
        --format FORMAT         Output format: text, json, or srt (default: text)
        --speaker-names NAMES   Comma-separated speaker names (e.g., "Alice,Bob")

EXAMPLE:
    python3 merge_diarization.py transcript.txt diarization.json -o final.txt
    python3 merge_diarization.py transcript.txt diarization.json --speaker-names "Dr. Smith,Patient"

AUTHOR: Audio transcription merge module
DATE: 2024
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


def parse_transcript(transcript_file: str) -> List[Dict]:
    """
    Parse whisper transcript file to extract text segments with timestamps
    
    Handles formats like:
    [00:00.000 --> 00:05.000]  Hello, this is the beginning.
    or just plain text without timestamps
    """
    segments = []
    
    with open(transcript_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try to detect timestamp format
    # Common formats: [00:00.000 --> 00:05.000] or [00:00 - 00:05]
    timestamp_pattern = r'\[(\d{2}:\d{2}(?:\.\d{3})?)\s*(?:-->|-)\s*(\d{2}:\d{2}(?:\.\d{3})?)\]'
    
    lines = content.strip().split('\n')
    current_text = []
    current_start = None
    current_end = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for timestamp
        match = re.match(timestamp_pattern, line)
        if match:
            # Save previous segment if exists
            if current_text and current_start is not None:
                segments.append({
                    'start': current_start,
                    'end': current_end,
                    'text': ' '.join(current_text).strip()
                })
            
            # Parse new timestamp
            start_str, end_str = match.groups()
            current_start = timestamp_to_seconds(start_str)
            current_end = timestamp_to_seconds(end_str)
            
            # Get text after timestamp
            text_after = line[match.end():].strip()
            current_text = [text_after] if text_after else []
        else:
            # Add to current text
            current_text.append(line)
    
    # Don't forget last segment
    if current_text and current_start is not None:
        segments.append({
            'start': current_start,
            'end': current_end,
            'text': ' '.join(current_text).strip()
        })
    
    # If no timestamps found, treat as one big segment
    if not segments and content.strip():
        segments.append({
            'start': 0,
            'end': None,
            'text': content.strip()
        })
    
    return segments


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert MM:SS or MM:SS.mmm to seconds"""
    parts = timestamp.split(':')
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")


def seconds_to_timestamp(seconds: float, include_ms: bool = False) -> str:
    """Convert seconds to MM:SS or MM:SS.mmm format"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    
    if include_ms:
        return f"{minutes:02d}:{secs:06.3f}"
    else:
        return f"{minutes:02d}:{int(secs):02d}"


def find_speaker_at_time(diarization: Dict, time: float) -> str:
    """Find which speaker is active at a given time"""
    for segment in diarization['segments']:
        if segment['start'] <= time <= segment['end']:
            return segment['speaker']
    return 'UNKNOWN'


def merge_transcript_diarization(
    transcript_segments: List[Dict],
    diarization: Dict,
    speaker_names: Dict[str, str] = None
) -> List[Dict]:
    """
    Merge transcript segments with speaker diarization
    
    Returns list of segments with speaker information added
    """
    merged = []
    
    for segment in transcript_segments:
        # Find speaker for this segment
        # Use middle of segment for better accuracy
        if segment['end'] is not None:
            check_time = (segment['start'] + segment['end']) / 2
        else:
            check_time = segment['start']
        
        speaker_id = find_speaker_at_time(diarization, check_time)
        
        # Map to custom name if provided
        if speaker_names and speaker_id in speaker_names:
            speaker_name = speaker_names[speaker_id]
        else:
            speaker_name = speaker_id
        
        merged.append({
            'start': segment['start'],
            'end': segment['end'],
            'speaker': speaker_name,
            'text': segment['text']
        })
    
    return merged


def format_output_text(segments: List[Dict]) -> str:
    """Format as readable text with speaker labels"""
    lines = []
    current_speaker = None
    
    for segment in segments:
        # Add speaker label when speaker changes
        if segment['speaker'] != current_speaker:
            current_speaker = segment['speaker']
            lines.append(f"\n{current_speaker}:")
        
        # Add timestamp and text
        timestamp = f"[{seconds_to_timestamp(segment['start'])}]"
        lines.append(f"{timestamp} {segment['text']}")
    
    return '\n'.join(lines).strip()


def format_output_srt(segments: List[Dict]) -> str:
    """Format as SRT subtitle file with speaker names"""
    srt_lines = []
    
    for i, segment in enumerate(segments, 1):
        # SRT index
        srt_lines.append(str(i))
        
        # Timestamps (SRT uses HH:MM:SS,mmm format)
        start_time = seconds_to_srt_time(segment['start'])
        end_time = seconds_to_srt_time(segment['end'] or segment['start'] + 5)
        srt_lines.append(f"{start_time} --> {end_time}")
        
        # Text with speaker
        srt_lines.append(f"[{segment['speaker']}] {segment['text']}")
        srt_lines.append("")  # Empty line between entries
    
    return '\n'.join(srt_lines)


def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs % 1) * 1000)
    secs = int(secs)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def format_output_json(segments: List[Dict]) -> str:
    """Format as JSON with all information"""
    output = {
        'segments': segments,
        'speakers': sorted(set(s['speaker'] for s in segments)),
        'total_segments': len(segments)
    }
    return json.dumps(output, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Merge speaker diarization with transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('transcript_file', help='Transcript file from whisper')
    parser.add_argument('diarization_file', help='Diarization JSON file')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('--format', choices=['text', 'json', 'srt'],
                       default='text', help='Output format')
    parser.add_argument('--speaker-names', 
                       help='Comma-separated speaker names (e.g., "Alice,Bob")')
    
    args = parser.parse_args()
    
    # Check input files
    if not os.path.exists(args.transcript_file):
        print(f"Error: Transcript file not found: {args.transcript_file}")
        sys.exit(1)
    
    if not os.path.exists(args.diarization_file):
        print(f"Error: Diarization file not found: {args.diarization_file}")
        sys.exit(1)
    
    # Set output file if not specified
    if not args.output:
        base_name = Path(args.transcript_file).stem
        extension = {'text': 'txt', 'json': 'json', 'srt': 'srt'}[args.format]
        args.output = f"{base_name}_speakers.{extension}"
    
    try:
        # Load transcript
        print(f"Loading transcript: {args.transcript_file}")
        transcript_segments = parse_transcript(args.transcript_file)
        print(f"Found {len(transcript_segments)} transcript segments")
        
        # Load diarization
        print(f"Loading diarization: {args.diarization_file}")
        with open(args.diarization_file, 'r') as f:
            diarization = json.load(f)
        print(f"Found {len(diarization['speakers'])} speakers")
        
        # Parse speaker names if provided
        speaker_name_map = {}
        if args.speaker_names:
            names = args.speaker_names.split(',')
            speakers = sorted(diarization['speakers'])
            for i, name in enumerate(names):
                if i < len(speakers):
                    speaker_name_map[speakers[i]] = name.strip()
            print(f"Speaker mapping: {speaker_name_map}")
        
        # Merge
        print("Merging transcript with speaker information...")
        merged_segments = merge_transcript_diarization(
            transcript_segments, diarization, speaker_name_map
        )
        
        # Format output
        if args.format == 'text':
            output = format_output_text(merged_segments)
        elif args.format == 'json':
            output = format_output_json(merged_segments)
        elif args.format == 'srt':
            output = format_output_srt(merged_segments)
        
        # Save
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        
        print(f"\nMerge complete!")
        print(f"Output saved to: {args.output}")
        
        # Print summary
        speakers = set(s['speaker'] for s in merged_segments)
        print(f"Total speakers in output: {len(speakers)}")
        for speaker in sorted(speakers):
            count = sum(1 for s in merged_segments if s['speaker'] == speaker)
            print(f"  {speaker}: {count} segments")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()