#!/usr/bin/env python3
"""Recover diarization results from permission-denied files"""

import json
import glob
import subprocess
import sys

if len(sys.argv) < 3:
    print("Usage: python3 recover_diarization.py <temp_dir> <output_file> [duration_seconds]")
    sys.exit(1)

temp_dir = sys.argv[1]
output_file = sys.argv[2]
duration = float(sys.argv[3]) if len(sys.argv) > 3 else None

print(f"Recovering diarization from: {temp_dir}")

# Find all chunk files
chunk_files = sorted(glob.glob(f"{temp_dir}/chunk_*_diarization.json"))
print(f"Found {len(chunk_files)} chunk files")

all_segments = []
all_speakers = set()
chunks_processed = 0
chunk_size = 600  # 10 minutes

for i, chunk_file in enumerate(chunk_files):
    print(f"Processing {chunk_file}...")
    try:
        # Try normal read first
        try:
            with open(chunk_file, 'r') as f:
                data = json.load(f)
        except PermissionError:
            # Use sudo to read
            print("  Using sudo to read file...")
            result = subprocess.run(['sudo', 'cat', chunk_file], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
            else:
                print(f"  Error: Could not read file")
                continue
        
        # Adjust timestamps based on chunk index
        chunk_index = int(chunk_file.split('chunk_')[1].split('_')[0])
        time_offset = chunk_index * chunk_size
        
        # Adjust segment timestamps
        for segment in data.get('segments', []):
            segment['start'] += time_offset
            segment['end'] += time_offset
        
        all_segments.extend(data.get('segments', []))
        all_speakers.update(data.get('speakers', []))
        chunks_processed += 1
        
        print(f"  ✓ Recovered {len(data.get('segments', []))} segments")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")

# Sort segments by time
all_segments.sort(key=lambda x: x['start'])

# Create output
output = {
    'speakers': sorted(list(all_speakers)),
    'num_speakers': len(all_speakers),
    'total_segments': len(all_segments),
    'segments': all_segments,
    'chunks_processed': chunks_processed,
    'chunks_total': len(chunk_files)
}

# Save output
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Recovered {len(all_segments)} segments from {chunks_processed} chunks")
print(f"Output saved to: {output_file}")

# Show summary
if all_segments and duration:
    first = min(s['start'] for s in all_segments)
    last = max(s['end'] for s in all_segments)
    coverage = (last - first) / duration * 100
    print(f"Coverage: {coverage:.1f}% of audio")
    print(f"Time range: {first/60:.1f}m to {last/60:.1f}m")
    print(f"Speakers: {', '.join(output['speakers'])}")