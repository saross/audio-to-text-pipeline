#!/usr/bin/env python3
"""
Apply term corrections to whisper.cpp output
"""

import re
import argparse
from pathlib import Path

def load_corrections(corrections_file):
    """Load term corrections from file."""
    corrections = {}
    
    try:
        with open(corrections_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '|' in line:
                    incorrect, correct = line.split('|', 1)
                    corrections[incorrect.strip()] = correct.strip()
        
        print(f"Loaded {len(corrections)} term corrections")
        return corrections
    except FileNotFoundError:
        print(f"Corrections file not found: {corrections_file}")
        return {}

def apply_corrections(text, corrections):
    """Apply term corrections to text."""
    if not corrections:
        return text
    
    corrected_text = text
    corrections_applied = 0
    
    # Sort by length (longest first) to avoid partial replacements
    sorted_corrections = sorted(corrections.items(), key=lambda x: len(x[0]), reverse=True)
    
    for incorrect, correct in sorted_corrections:
        # Use word boundaries for whole word matching
        pattern = re.compile(r'\b' + re.escape(incorrect) + r'\b', re.IGNORECASE)
        
        # Count matches
        matches = len(pattern.findall(corrected_text))
        if matches > 0:
            corrected_text = pattern.sub(correct, corrected_text)
            corrections_applied += matches
            print(f"  {incorrect} -> {correct} ({matches} occurrences)")
    
    print(f"\nTotal corrections applied: {corrections_applied}")
    return corrected_text

def format_transcript(text):
    """Add basic formatting to transcript."""
    # Ensure sentences end with proper punctuation
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not line[-1] in '.!?':
            line += '.'
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def main():
    parser = argparse.ArgumentParser(description="Apply term corrections to transcripts")
    parser.add_argument("input_file", help="Input transcript file")
    parser.add_argument("-o", "--output", help="Output file (default: input_corrected.txt)")
    parser.add_argument("-c", "--corrections", default="term_corrections.txt",
                       help="Corrections file (default: term_corrections.txt)")
    parser.add_argument("--format", action="store_true", help="Apply basic formatting")
    
    args = parser.parse_args()
    
    # Set output file
    if args.output:
        output_file = args.output
    else:
        input_path = Path(args.input_file)
        output_file = input_path.parent / f"{input_path.stem}_corrected{input_path.suffix}"
    
    # Load corrections
    corrections = load_corrections(args.corrections)
    
    # Read input file
    print(f"\nReading: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Original length: {len(text)} characters")
    
    # Apply corrections
    print("\nApplying corrections...")
    corrected_text = apply_corrections(text, corrections)
    
    # Apply formatting if requested
    if args.format:
        print("\nApplying formatting...")
        corrected_text = format_transcript(corrected_text)
    
    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(corrected_text)
    
    print(f"\nOutput saved to: {output_file}")
    print(f"Final length: {len(corrected_text)} characters")

if __name__ == "__main__":
    main()