# Planned Testing Regime for Transcription Pipeline

## Introduction: Why Testing Matters for This Project

This document outlines a progressive testing strategy for the transcription pipeline, designed specifically for researchers with data science experience who are new to formal software testing methodologies.

As a specialized audio processing pipeline with critical requirements for accuracy and reliability, testing provides several key benefits:

- **Quality assurance**: Ensures transcriptions meet research standards
- **Regression prevention**: Maintains reliability when adding features (like WhisperX)
- **Knowledge documentation**: Tests serve as executable documentation
- **Refactoring confidence**: Supports architectural improvements

For researchers specifically, testing also:
- Documents the analysis process
- Ensures reproducibility of results
- Provides verification of methodology
- Protects against errors that could affect research conclusions

## Testing Goals & Timeline

The testing regime is organized as a progressive learning journey across four stages, each building on the previous:

### Stage 1: Basic Function Testing (1-2 weeks)
**Goal**: Test 3-5 core functions from your modules (term corrections, empty segment filtering, etc.)

**Activities**:
- Write simple unit tests for individual functions
- Focus on pure functions with clear inputs/outputs
- Learn basic pytest assertions and test structure

**Expected outcomes**:
- Foundational understanding of test writing
- Tests for critical text processing functions
- Confidence in running and interpreting test results

### Stage 2: Module & Configuration Testing (2-3 weeks)
**Goal**: Test each module's core functionality in isolation

**Activities**:
- Write tests for complete modules
- Test configuration handling
- Learn to use test fixtures
- Introduce basic mocking for dependencies

**Expected outcomes**:
- Comprehensive test coverage of individual modules
- Ability to test more complex scenarios
- Tests that verify module interfaces work correctly

### Stage 3: Integration Testing (3-4 weeks)
**Goal**: Verify how modules work together in combinations

**Activities**:
- Test interactions between multiple modules
- Verify data flows correctly between components
- Test with real (but small) audio samples
- Incorporate test coverage analysis

**Expected outcomes**:
- Tests that validate cross-module functionality
- Understanding of system behavior as a whole
- Identification of potential integration issues

### Stage 4: End-to-End Testing (4-5 weeks)
**Goal**: Verify the complete pipeline from shell scripts to output

**Activities**:
- Test the full system from input to output
- Create test cases for different audio types/scenarios
- Test shell wrappers with BATS
- Develop automated test workflows

**Expected outcomes**:
- Confidence in overall system functionality
- Tests that verify real-world use cases
- Ability to detect issues across the entire pipeline

## Testing Frameworks & Tools

### Primary Framework: pytest

```bash
pip install pytest pytest-cov
```

**Why pytest is ideal for this project**:
- **Simple syntax**: Uses standard Python assert statements
- **Minimal boilerplate**: Tests are just Python functions
- **Descriptive failures**: Clear error messages with context
- **Fixture system**: Elegant ways to set up test data
- **Plugin ecosystem**: Extensions for specific needs

### For Shell Script Testing

**ShellCheck**: Static analyzer for shell scripts
```bash
# Ubuntu/Debian
sudo apt-get install shellcheck

# macOS
brew install shellcheck
```

**BATS**: Bash Automated Testing System
```bash
git clone https://github.com/bats-core/bats-core.git
cd bats-core
./install.sh /usr/local
```

### Audio Testing Utilities

For testing audio processing components:
- Create short test audio files (3-5 seconds)
- Include samples with different characteristics:
  - Single speaker vs. multiple speakers
  - Clean audio vs. noisy audio
  - Various languages (if multi-language support is planned)

## Implementation Plan

### 1. Project Test Structure

```
transcription_pipeline/
├── tests/
│   ├── fixtures/                # Test audio files and sample data
│   │   ├── audio/               # Small audio test files
│   │   │   ├── single_speaker.flac
│   │   │   ├── two_speakers.flac
│   │   │   └── noisy_audio.flac
│   │   └── expected/            # Expected outputs for comparison
│   ├── unit/                    # Unit tests for individual functions
│   │   ├── test_postprocess.py
│   │   ├── test_segment_merger.py
│   │   └── ...
│   ├── integration/             # Tests for module interactions
│   │   ├── test_transcription_flow.py
│   │   └── ...
│   └── end_to_end/              # Full pipeline tests
│       └── test_transcribe_script.py
├── scripts/
│   └── test_pipeline.sh         # Test runner script
└── pytest.ini                   # pytest configuration
```

### 2. Establishing Baseline Tests

Before major refactoring, create tests for existing functionality:

1. **Identify critical functions**: Find the most important/complex parts of your code
2. **Create simple test fixtures**: Small audio files, expected outputs
3. **Write baseline tests**: Test current behavior to ensure it doesn't change during refactoring

### 3. Test-Driven Development for New Features

For each new feature (like empty segment filtering):

1. **Write the test first**
2. **Implement the feature** until the test passes
3. **Refactor** to improve code quality
4. **Add tests for edge cases**

### 4. Continuous Integration Setup

Set up automated test running with shell scripts:

```bash
#!/bin/bash
# scripts/test_pipeline.sh

# Activate environment
source ~/transcription-env/bin/activate

# Run Python tests
echo "Running Python tests..."
pytest tests/unit/ tests/integration/ tests/end_to_end/ -v

# Run shell script tests
echo "Running shell script tests..."
bats tests/shell/

# Report results
if [ $? -eq 0 ]; then
    echo "All tests passed!"
else
    echo "Some tests failed."
    exit 1
fi
```

## Test Types & Examples

### Unit Tests

```python
# tests/unit/test_postprocess.py
def test_apply_term_corrections():
    """Test technical term corrections functionality."""
    # Setup
    corrections = {"Sisk": "SISSVoc", "scos": "SKOS"}
    text = "The Sisk system uses scos concepts"
    
    # Execute
    from transcription_pipeline.processors.postprocess import apply_term_corrections
    result = apply_term_corrections(text, corrections)
    
    # Verify
    assert result == "The SISSVoc system uses SKOS concepts"
```

```python
# tests/unit/test_segment_merger.py
def test_filter_empty_segments():
    """Test filtering of empty UNKNOWN segments."""
    # Setup
    segments = [
        {"speaker": "SPEAKER_00", "text": "Hello", "start": 0, "end": 1},
        {"speaker": "UNKNOWN", "text": "", "start": 1, "end": 2},
        {"speaker": "SPEAKER_01", "text": "Hi there", "start": 2, "end": 3}
    ]
    
    # Execute
    from transcription_pipeline.processors.segment_merger import filter_empty_segments
    filtered = filter_empty_segments(segments)
    
    # Verify
    assert len(filtered) == 2
    assert filtered[0]["speaker"] == "SPEAKER_00"
    assert filtered[1]["speaker"] == "SPEAKER_01"
```

### Module Tests

```python
# tests/unit/test_whisper_model.py
def test_transcribe_short_audio():
    """Test basic transcription of a short audio clip."""
    # Setup
    audio_file = "tests/fixtures/audio/single_speaker.flac"
    config = MockConfig(model_size="tiny")
    
    # Execute
    from transcription_pipeline.models.whisper_model import transcribe
    result = transcribe(audio_file, config)
    
    # Verify basic structure
    assert "segments" in result
    assert len(result["segments"]) > 0
    assert "text" in result["segments"][0]
    
    # Verify content (fuzzy matching since transcription can vary slightly)
    expected_text = "This is a test audio clip"
    assert expected_text.lower() in result["segments"][0]["text"].lower()
```

### Integration Tests

```python
# tests/integration/test_transcription_flow.py
def test_transcription_and_diarisation():
    """Test transcription and diarisation combined workflow."""
    # Setup
    audio_file = "tests/fixtures/audio/two_speakers.flac"
    config = MockConfig(model_size="tiny", num_speakers=2)
    
    # Execute transcription
    from transcription_pipeline.models.whisper_model import transcribe
    from transcription_pipeline.models.diarisation import diarise
    from transcription_pipeline.processors.segment_merger import combine, merge_segments
    
    transcription = transcribe(audio_file, config)
    diarisation_result = diarise(audio_file, config)
    
    # Combine results
    combined = combine(transcription, diarisation_result)
    segments = merge_segments(combined)
    
    # Verify
    assert len(segments) >= 2  # At least two segments for two speakers
    
    # Check speaker distribution
    speakers = set(segment["speaker"] for segment in segments)
    assert len(speakers) == 2  # Two unique speakers
    
    # Verify timing sequence is logical
    for i in range(1, len(segments)):
        assert segments[i]["start"] >= segments[i-1]["end"]
```

### End-to-End Tests

```python
# tests/end_to_end/test_transcribe_script.py
import subprocess
import os

def test_end_to_end_transcription():
    """Test the complete transcription pipeline via shell script."""
    # Setup
    test_audio = "tests/fixtures/audio/two_speakers.flac"
    output_path = "tests/output/test_transcript.txt"
    
    # Delete output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Execute actual shell command
    result = subprocess.run(
        ["./scripts/transcribe.sh", test_audio, "-o", output_path],
        capture_output=True,
        text=True
    )
    
    # Verify process success
    assert result.returncode == 0, f"Process failed with error: {result.stderr}"
    
    # Verify output file exists
    assert os.path.exists(output_path), "Output transcript not created"
    
    # Verify content contains expected patterns
    with open(output_path, 'r') as f:
        content = f.read()
        assert "[00:00:" in content  # Has timestamps
        assert "SPEAKER_" in content  # Has speaker labels
        assert len(content.split("\n\n")) >= 3  # At least a few transcript lines
```

### Shell Script Tests

```bash
#!/usr/bin/env bats
# tests/shell/test_preprocess.sh

setup() {
  # Create a temporary directory for test outputs
  TEST_DIR=$(mktemp -d)
  TEST_AUDIO="tests/fixtures/audio/single_speaker.flac"
}

teardown() {
  # Clean up temporary files
  rm -rf "$TEST_DIR"
}

@test "preprocess_audio.sh creates output file" {
  ./scripts/preprocess_audio.sh "$TEST_AUDIO" -o "$TEST_DIR/processed.flac"
  [ -f "$TEST_DIR/processed.flac" ]
}

@test "preprocess_audio.sh fails on missing input" {
  run ./scripts/preprocess_audio.sh nonexistent.mp3
  [ "$status" -eq 1 ]
}

@test "preprocess_audio.sh respects --no-noise-reduction flag" {
  # This would need more sophisticated verification in real tests
  run ./scripts/preprocess_audio.sh "$TEST_AUDIO" -o "$TEST_DIR/processed.flac" --no-noise-reduction
  [ "$status" -eq 0 ]
  [ -f "$TEST_DIR/processed.flac" ]
}
```

## Testing Strategies for Audio Processing

Audio processing presents unique testing challenges:

### 1. Test with Representative Audio Files

Create a suite of test audio files that represent real-world scenarios:
- Single speakers vs. multiple speakers
- Clean vs. noisy backgrounds  
- Different languages (if applicable)
- Various audio qualities

Keep test files short (3-5 seconds) for quick test execution.

### 2. Verify Outputs with Tolerance

For audio processing, exact matches rarely work. Instead:
- Check structural properties (correct timestamps, speaker labels)
- Use fuzzy text matching for transcription content
- Verify statistical properties of audio (after noise reduction, etc.)

### 3. Golden Master Testing

For complex processing functions:
1. Process a test file with current code
2. Save the results as "golden master"
3. When refactoring, verify new code produces equivalent results

### 4. Measure & Test Performance

For audio processing pipelines, performance is often critical:
- Test processing time on standard inputs
- Verify memory usage stays within bounds
- Test with both small and large files

## Progressive Learning Path

For someone new to testing, follow this learning progression:

### Week 1-2: Basic pytest
- Install pytest
- Create simple function tests
- Learn about assertions
- Run tests with `pytest -v`

### Week 3-5: Test Organization
- Create test fixtures
- Group tests logically
- Use setup/teardown functions
- Learn about test parameterization

### Week 6-9: More Advanced Testing
- Learn mocking for dependencies
- Create integration tests
- Measure test coverage
- Implement shell script testing

### Week 10-13: Complete Testing System
- Set up automated test running
- Write end-to-end tests
- Create continuous integration workflows
- Develop performance tests

## Resources for Learning More

### Beginner-Friendly Testing Resources

1. **Python Testing with pytest** by Brian Okken
   - Excellent introduction for Python developers
   - Focuses on practical examples

2. **Real Python Testing Tutorials**: 
   - https://realpython.com/pytest-python-testing/
   - https://realpython.com/python-testing/

3. **pytest Documentation**:
   - https://docs.pytest.org/en/stable/

### Specific to Research Software

1. **Software Carpentry: Testing**:
   - https://swcarpentry.github.io/python-novice-inflammation/10-defensive/index.html

2. **The Turing Way: Testing**:
   - https://the-turing-way.netlify.app/reproducible-research/testing.html

### Audio Processing Testing

For the unique challenges of testing audio processing:

1. **LibriSpeech Test Set**:
   - Small, public domain audio clips for testing speech recognition
   - https://www.openslr.org/12/

2. **Pytest-Benchmark**:
   - For testing processing performance
   - https://pytest-benchmark.readthedocs.io/

## Conclusion: Testing in Research Software Development

As an archaeologist working with data science, integrating testing into your workflow provides several benefits:

1. **Scientific Reproducibility**: Tests document exact processing steps
2. **Verification & Validation**: Formal verification of methodology
3. **Knowledge Transfer**: Tests document how the system should behave
4. **Confidence in Results**: Reduce chances of processing errors affecting research conclusions

This testing regime is designed to progressively build your testing skills while improving the reliability of your transcription pipeline. By following the timeline and examples provided, you'll not only create a robust test suite for your current project but also develop valuable skills that transfer to other research software development.
