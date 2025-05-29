#!/bin/bash
# batch_preprocess.sh

# List your files here
files=(
    #"/home/shawn/Code/audio-transcription/data/raw/interview-1.mp4"
    "/home/shawn/Code/audio-transcription/data/raw/interview-2.mp4" 
    # "/home/shawn/Code/audio-transcription/data/raw/interview-3.mp4"
    # "/home/shawn/Code/audio-transcription/data/raw/interview-4.mp4"
    # "/home/shawn/Code/audio-transcription/data/raw/interview-5.mp4"
    # "/home/shawn/Code/audio-transcription/data/raw/interview-6.mp4"

)

# Output directory
output_dir="/home/shawn/Code/audio-transcription/data/processed"
mkdir -p "$output_dir"

# Process each file
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .flac)
        echo "Processing: $filename"
        
        ./preprocess_audio.sh "$file" -o "$output_dir/${filename}.flac"
        
        if [ $? -eq 0 ]; then
            echo "✅ Completed: $filename"
        else
            echo "❌ Failed: $filename"
        fi
        echo "---"
    else
        echo "⚠️  File not found: $file"
    fi
done

echo "Batch processing complete!"