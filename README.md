# Turkish Speech Recognition System

This application uses OpenAI's Whisper model locally to provide real-time Turkish speech-to-text transcription with a clean, minimal interface.

## Version

**voice2text.py** - Minimalist Turkish speech transcription that outputs only clean text with natural pauses

## Requirements

- Python 3.8+
- PyAudio
- FFmpeg (for audio processing)
- OpenAI Whisper (local version)
- Minimum 4GB RAM
- CUDA-compatible GPU (optional but recommended for faster transcription)

## Installation

1. Install dependencies:
```
pip install -r requirements.txt
```

2. For GPU acceleration, install PyTorch with CUDA:
```
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
For other CUDA versions, refer to the [PyTorch installation page](https://pytorch.org/get-started/locally/).

3. Install FFmpeg (required for Whisper):
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt-get install ffmpeg`
- Windows: Download from https://ffmpeg.org/download.html

## Usage

Run the speech recognition system:
```
python fhisper.py
```

- The application will silently listen to your microphone
- It will output only the transcribed Turkish text with natural pauses
- No progress indicators, logging, or extra text will be displayed
- Press Ctrl+C to stop the application

## Language Setting

The application is configured to use Turkish by default:

```python
# Language settings
DEFAULT_LANGUAGE = "tr"  # Turkish language code
INITIAL_PROMPT = "Bu bir Türkçe ses kaydıdır."  # "This is a Turkish audio recording"
```

To use a different language, edit these values in `fhisper.py`:
- Change `DEFAULT_LANGUAGE` to the appropriate language code (e.g., "en" for English, "fr" for French)
- Update `INITIAL_PROMPT` to a phrase in the target language

## Features

- Turkish language is the default setting
- Clean, distraction-free interface that shows only transcribed text
- Natural 5-second pauses between outputs
- Background processing with no visual indicators
- GPU acceleration for faster processing (if available)

## Technical Details

- Uses the "large-v3" Whisper model for high-quality transcription
- Records in 5-second chunks
- Enforces minimum 5-second spacing between outputs
- Runs three parallel threads for recording, transcription and display
- Silent operation with no progress indicators 