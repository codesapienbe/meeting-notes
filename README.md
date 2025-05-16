# Meeting Notes

This application uses OpenAI's Whisper model locally to provide real-time Turkish speech-to-text transcription for meeting notes with a clean, minimal interface.

## Version

**meeting-notes** - Turkish meeting transcription that outputs clean text with natural pauses

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

# Meeting Notes

Meeting transcription and summarization application using Whisper.

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

> **Important**: The GROQ_API_KEY is required for the summarization feature to work. If you don't have a Groq API key:
> 1. Sign up at https://console.groq.com/ and obtain an API key
> 2. Create a `.env` file in the project root and add your key as shown above
> 3. Make sure the `.env` file is properly loaded when you run the application

## Running the Application

You can run the application with a single command, including the GUI option:

```bash
python run.py --gui
```

This will:
1. Start all backend components (Redis, Celery worker, FastAPI)
2. Launch the Streamlit frontend
3. Provide a unified logging view for all components
4. Handle graceful shutdown of all components when you press Ctrl+C

### Command Line Options

The `run.py` script accepts these command-line arguments:

- `--workers N` or `-w N`: Specify the number of Celery worker processes (default: 1)
- `--kill-existing`: Kill existing Celery processes before starting (default: true)
- `--reset-redis`: Completely reset Redis database before starting
- `--gui`: Start the Streamlit frontend GUI

### Backend Services Only

If you want to run only the backend services without the GUI:

```bash
python run.py
```

The API will be available at http://localhost:8000

### Frontend Only

If you already have the backend running and want to start only the frontend:

```bash
streamlit run webapp.py
```

The Streamlit interface will be available at http://localhost:8501

## Stopping the Application

Press Ctrl+C to stop all processes and the Redis container.

## API Endpoints

- `POST /transcribe` - Upload a meeting audio file for transcription
  - Optional query parameters:
    - `language`: Language code (e.g., 'tr' for Turkish, 'en' for English)
    - `initial_prompt`: Custom prompt to guide the transcription
    - `save_output`: Whether to save the output to a JSON file (default: false)

- `POST /summarize` - Summarize meeting notes

- `GET /task/{task_id}` - Check the status of a task

- `GET /task/{task_id}/text` - Get just the transcription text from a completed task

- `GET /models` - Get available models and supported languages

## Language Support

The application defaults to Turkish (tr), but you can specify a different language when transcribing:

```bash
# Example of transcribing with a different language
curl -X POST "http://localhost:8000/transcribe?language=en" \
  -F "file=@your_audio_file.wav"
```

Supported languages include:
- English (en)
- Turkish (tr)
- German (de)
- Spanish (es)
- French (fr)
- And many more...

Check the `/models` endpoint for the full list of supported languages. 