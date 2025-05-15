import whisper
import pyaudio
import wave
import os
import time
import torch
import threading
import json
import datetime
import uuid
import requests
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Optional, Dict, Any
import shutil
from celery import Celery
from celery.result import AsyncResult

# Load environment variables
load_dotenv()

# Check if GPU is available (quiet check)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Language settings for Turkish
DEFAULT_LANGUAGE = "tr"  # Turkish language code
INITIAL_PROMPT = "Bu bir Türkçe ses kaydıdır. Lütfen Türkçe konuşmayı doğru şekilde yazıya dökün. Türkiye Türkçesi kullanılıyor."

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # High quality audio
CHUNK = 1024

# Export settings
EXPORT_DIRECTORY = "transcripts"  # Directory to save JSON files
MODEL_NAME = "large-v3"  # Whisper model to use

# Groq API settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-70b-8192"  # Can be changed to other Groq models

# Setup Celery
# Define in a way that it can be imported by the Celery worker
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
celery_app = Celery("voice2text_tasks", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.task_routes = {
    "voice2text_tasks.transcribe_task": {"queue": "transcription"},
    "voice2text_tasks.summarize_task": {"queue": "summarization"}
}

# Create FastAPI app
app = FastAPI(title="Voice2Text API", description="API for transcribing audio to text and summarizing")

# Pydantic models for API
class SummarizeRequest(BaseModel):
    text: str

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TranscriptionResponse(BaseModel):
    text: str
    segments: list
    language: str
    processing_time_seconds: float

class SummaryResponse(BaseModel):
    summary: str
    model: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None

def ensure_export_directory():
    """Ensure the export directory exists"""
    Path(EXPORT_DIRECTORY).mkdir(exist_ok=True)
    Path("temp").mkdir(exist_ok=True)
    
def generate_filename(extension="json"):
    """Generate a unique filename based on current date/time"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"transcript_{timestamp}_{uuid.uuid4().hex[:6]}.{extension}"

def record_audio(duration=None):
    """Record audio from microphone for a specified duration or until Enter key is pressed"""
    frames = []
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    
    print("\nKayıt başladı... Konuşun... (Recording started...)")
    print("Durdurmak için Enter'a basın. (Press Enter to stop recording)")
    
    # Start a thread to wait for Enter key
    stop_recording = threading.Event()
    
    def wait_for_enter():
        input()  # Wait for Enter key
        stop_recording.set()
    
    input_thread = threading.Thread(target=wait_for_enter, daemon=True)
    input_thread.start()
    
    # Record metadata
    start_time = time.time()
    start_datetime = datetime.datetime.now()
    
    try:
        # Record until stop_recording is set or duration is reached
        while not stop_recording.is_set() and (duration is None or time.time() - start_time < duration):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
    finally:
        # Stop recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("\nKayıt tamamlandı. (Recording finished.)")
    
    # Record end metadata
    end_time = time.time()
    end_datetime = datetime.datetime.now()
    duration_seconds = end_time - start_time
    
    return {
        "frames": frames,
        "start_time": start_time,
        "end_time": end_time,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "duration_seconds": duration_seconds
    }

def save_audio_to_file(frames, filename):
    """Save recorded audio frames to a file"""
    if not frames:
        return None
        
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    return filename

def transcribe_audio(filename):
    """Transcribe audio file to text and return full results"""
    print("Loading Whisper model...")
    
    model = whisper.load_model(MODEL_NAME).to(device)
    print("Model loaded. Transcribing...")
    
    # Time the transcription process
    transcription_start = time.time()
    
    result = model.transcribe(
        filename,
        language=DEFAULT_LANGUAGE,  # Use Turkish
        initial_prompt=INITIAL_PROMPT,
        fp16=(device == "cuda"),
        temperature=0.0,  # Deterministic output
        beam_size=5       # Better quality
    )
    
    transcription_end = time.time()
    transcription_duration = transcription_end - transcription_start
    
    # Add timing data to the result
    result["transcription_duration"] = transcription_duration
    
    return result

def summarize_with_groq(text):
    """Use Groq API to summarize the transcribed text"""
    if not GROQ_API_KEY:
        print("Groq API key not found.")
        return None
        
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Lütfen aşağıdaki Türkçe metni özetleyin. Önemli noktaları ve ana fikirleri koruyarak kısa bir özet oluşturun:

{text}

Özet:"""
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "Sen profesyonel bir özet yapma asistanısın. Türkçe metinleri kısa ve öz bir şekilde özetliyorsun."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        summary = result["choices"][0]["message"]["content"]
        
        return {
            "summary": summary,
            "model": GROQ_MODEL,
            "processing_details": {
                "timestamp": datetime.datetime.now().isoformat(),
                "tokens_used": result.get("usage", {})
            }
        }
    except Exception as e:
        print(f"Error summarizing with Groq API: {str(e)}")
        return {"error": str(e)}

def export_to_json(recording_data, transcription_data, audio_filename=None, summary_data=None):
    """Export all data to a structured JSON file"""
    ensure_export_directory()
    
    # Create metadata structure
    metadata = {
        "recording": {
            "start_time": recording_data["start_datetime"].isoformat() if "start_datetime" in recording_data else None,
            "end_time": recording_data["end_datetime"].isoformat() if "end_datetime" in recording_data else None,
            "duration_seconds": recording_data.get("duration_seconds"),
            "sample_rate": RATE,
            "channels": CHANNELS,
            "audio_format": "WAV"
        },
        "transcription": {
            "model": MODEL_NAME,
            "language": transcription_data.get("language", DEFAULT_LANGUAGE),
            "processing_time_seconds": transcription_data.get("transcription_duration"),
            "text": transcription_data.get("text", ""),
            "segments": transcription_data.get("segments", [])
        },
        "metadata": {
            "device": device,
            "timestamp": datetime.datetime.now().isoformat(),
            "audio_file": audio_filename,
            "prompt_used": INITIAL_PROMPT,
            "uuid": str(uuid.uuid4())
        }
    }
    
    # Add summary data if available
    if summary_data and "summary" in summary_data:
        metadata["summary"] = summary_data
    
    # Save JSON to file
    filename = os.path.join(EXPORT_DIRECTORY, generate_filename())
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return filename, metadata

# Register Celery tasks
@celery_app.task(name="voice2text_tasks.transcribe_task")
def transcribe_task(file_path, save_output=True):
    """Celery task to transcribe audio file and optionally summarize"""
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
        
    print(f"Processing audio file: {file_path}")
    
    try:
        # Create metadata for recording
        start_time = time.time()
        start_datetime = datetime.datetime.now()
        
        # Get file creation time if possible
        try:
            file_stat = os.stat(file_path)
            file_creation_time = datetime.datetime.fromtimestamp(file_stat.st_ctime)
            file_modification_time = datetime.datetime.fromtimestamp(file_stat.st_mtime)
        except:
            file_creation_time = None
            file_modification_time = None
        
        # Transcribe audio
        transcription_data = transcribe_audio(file_path)
        
        # Calculate basic recording data
        end_time = time.time()
        end_datetime = datetime.datetime.now()
        processing_duration = end_time - start_time
        
        # Create minimal recording data
        recording_data = {
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "duration_seconds": processing_duration,
            "start_time": start_time,
            "end_time": end_time,
            "file_metadata": {
                "path": file_path,
                "filename": os.path.basename(file_path),
                "creation_time": file_creation_time.isoformat() if file_creation_time else None,
                "modification_time": file_modification_time.isoformat() if file_modification_time else None
            }
        }
        
        # Generate summary using Groq API if API key is available
        summary_data = None
        if GROQ_API_KEY:
            # Start summarize task
            summary_task = summarize_task.delay(transcription_data["text"])
            summary_data = summary_task.get(timeout=60)  # Wait for result with timeout
        
        # Export everything to JSON if requested
        json_file = None
        metadata = None
        
        if save_output:
            audio_filename = os.path.basename(file_path)
            json_file, metadata = export_to_json(
                recording_data, 
                transcription_data, 
                audio_filename,
                summary_data
            )
        
        result = {
            "transcription": {
                "text": transcription_data["text"],
                "language": transcription_data.get("language", DEFAULT_LANGUAGE),
                "processing_time_seconds": transcription_data.get("transcription_duration"),
                "segments": transcription_data.get("segments", [])
            },
            "summary": summary_data["summary"] if summary_data and "summary" in summary_data else None,
            "json_file": json_file
        }
        
        # Clean up temp file if it exists and is in temp directory
        if "temp/" in file_path and os.path.exists(file_path):
            os.remove(file_path)
            
        return result
            
    except Exception as e:
        # Clean up in case of error
        if "temp/" in file_path and os.path.exists(file_path):
            os.remove(file_path)
        return {"error": f"Error processing file: {str(e)}"}

@celery_app.task(name="voice2text_tasks.summarize_task")
def summarize_task(text):
    """Celery task to summarize text using Groq API"""
    return summarize_with_groq(text)

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Voice2Text API", "status": "running", "model": MODEL_NAME, "device": device}

@app.post("/transcribe", response_model=TaskResponse)
async def transcribe_endpoint(
    file: UploadFile = File(...),
    save_output: bool = Form(False)
):
    """
    Start async task to transcribe an audio file
    
    - **file**: Audio file to transcribe (WAV format recommended)
    - **save_output**: Whether to save the output to a JSON file
    """
    # Create temp directory if it doesn't exist
    ensure_export_directory()
    
    # Save uploaded file to temp directory
    temp_file_path = f"temp/{uuid.uuid4()}.wav"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Start celery task
    try:
        task = transcribe_task.delay(temp_file_path, save_output)
        return {
            "task_id": task.id,
            "status": "PENDING",
            "message": "Transcription task started"
        }
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize", response_model=TaskResponse)
async def summarize_endpoint(request: SummarizeRequest):
    """
    Start async task to summarize text using Groq API
    
    - **text**: Text to summarize
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key not configured")
    
    try:
        task = summarize_task.delay(request.text)
        return {
            "task_id": task.id,
            "status": "PENDING",
            "message": "Summarization task started"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Check the status of a task
    
    - **task_id**: ID of the task to check
    """
    task_result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": task_result.status
    }
    
    if task_result.ready():
        if task_result.successful():
            response["result"] = task_result.get()
        else:
            response["result"] = {"error": str(task_result.result)}
    
    return response

@app.get("/models")
async def get_models():
    """Get information about available models"""
    whisper_models = ["tiny", "base", "small", "medium", "large-v3"]
    
    return {
        "current_whisper_model": MODEL_NAME,
        "available_whisper_models": whisper_models,
        "current_summary_model": GROQ_MODEL,
        "device": device
    }

if __name__ == "__main__":
    ensure_export_directory()
    uvicorn.run("voice2text:app", host="0.0.0.0", port=8000, reload=True) 