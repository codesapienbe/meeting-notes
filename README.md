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

# Whisper Voice Typing

Voice typing and transcription application using Whisper.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure Docker is installed and running on your system.

3. Set up environment variables by creating a `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
```

## Running the Application

Instead of using docker-compose, you can now use the Python script to run all components:

```bash
python run.py
```

This script will:
1. Start a Redis container using Docker
2. Start the Celery worker for background processing
3. Start the FastAPI application

The API will be available at http://localhost:8000

### Redis Port Configuration

The script automatically handles Redis port configurations:

- If port 6379 is already in use, it will automatically find and use another available port
- If Redis is already running on port 6379, it will use that existing instance instead of starting a new container
- The Celery worker and FastAPI application will automatically connect to whichever port Redis is using

### Troubleshooting

If you encounter errors related to Redis or port availability:

1. You can manually modify the `REDIS_PORT` variable in `run.py` if needed
2. Check if you have an existing Redis service running (`redis-server` or another container)
3. Use `docker ps` to check for running containers that might be using the Redis port

## Stopping the Application

Press Ctrl+C to stop all processes and the Redis container.

## API Endpoints

- `POST /transcribe` - Upload an audio file for transcription
- `POST /summarize` - Summarize a text
- `GET /task/{task_id}` - Check the status of a task
- `GET /models` - Get available models 