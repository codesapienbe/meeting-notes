import os
import sys

# Add the parent directory to the path so imports work consistently
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, current_dir)

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
from typing import Optional, Dict, Any, List
import shutil
from celery import Celery
from celery.result import AsyncResult
from src.dbmgr import TaskDatabaseManager

# Load environment variables
load_dotenv()

# Initialize database manager
db_manager = TaskDatabaseManager()

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
celery_app = Celery("meeting_notes_tasks", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.task_routes = {
    "meeting_notes_tasks.transcribe_task": {"queue": "transcription"},
    "meeting_notes_tasks.summarize_task": {"queue": "summarization"}
}

# Create FastAPI app
app = FastAPI(title="Meeting Notes", description="API for transcribing meeting audio to text and summarizing meeting notes")

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

def transcribe_audio(filename, language=None, initial_prompt=None):
    """Transcribe audio file to text and return full results"""
    print("Loading Whisper model...")
    
    # Use provided language or default
    language = language or DEFAULT_LANGUAGE
    initial_prompt = initial_prompt or INITIAL_PROMPT
    
    print(f"Transcribing with language: {language}")
    
    model = whisper.load_model(MODEL_NAME).to(device)
    print("Model loaded. Transcribing...")
    
    # Time the transcription process
    transcription_start = time.time()
    
    result = model.transcribe(
        filename,
        language=language,
        initial_prompt=initial_prompt,
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
    
    if not text or text.strip() == "":
        print("Empty text provided to summarize_with_groq, nothing to summarize.")
        return {
            "summary": "Metin boş! Özetlenecek içerik bulunamadı.",
            "model": GROQ_MODEL,
            "processing_details": {
                "timestamp": datetime.datetime.now().isoformat(),
                "error": "Empty input text"
            }
        }
    
    print(f"Sending text to Groq API for summarization (length: {len(text)} chars)")
    
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
        print("Making request to Groq API...")
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        
        # Print status code for debugging
        print(f"Groq API response status code: {response.status_code}")
        
        # Handle different status codes
        if response.status_code == 401:
            print("Authentication error: Invalid Groq API key")
            return {"error": "Invalid Groq API key", "summary": "API kimlik doğrulama hatası!"}
            
        if response.status_code != 200:
            print(f"Groq API error: {response.text}")
            return {"error": f"Groq API error: {response.status_code}", "summary": "API hatası!"}
        
        # Parse response if status code is 200
        result = response.json()
        summary = result["choices"][0]["message"]["content"]
        
        print(f"Successfully received summary from Groq (length: {len(summary)} chars)")
        
        return {
            "summary": summary,
            "model": GROQ_MODEL,
            "processing_details": {
                "timestamp": datetime.datetime.now().isoformat(),
                "tokens_used": result.get("usage", {})
            }
        }
    except requests.exceptions.RequestException as e:
        print(f"Network error when connecting to Groq API: {str(e)}")
        return {"error": f"Network error: {str(e)}", "summary": "API bağlantı hatası!"}
    except KeyError as e:
        print(f"Unexpected Groq API response format: {str(e)}")
        return {"error": f"Unexpected response format: {str(e)}", "summary": "API yanıt hatası!"}
    except Exception as e:
        print(f"Error summarizing with Groq API: {str(e)}")
        return {"error": str(e), "summary": "Özet oluşturma hatası!"}

def export_to_json(recording_data, transcription_data, audio_filename=None, summary_data=None):
    """Export all data to a structured JSON file"""
    ensure_export_directory()
    
    # Get the language used for transcription
    language = transcription_data.get("language", DEFAULT_LANGUAGE)
    
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
            "language": language,
            "processing_time_seconds": transcription_data.get("transcription_duration"),
            "text": transcription_data.get("text", ""),
            "segments": transcription_data.get("segments", [])
        },
        "metadata": {
            "device": device,
            "timestamp": datetime.datetime.now().isoformat(),
            "audio_file": audio_filename,
            "prompt_used": transcription_data.get("initial_prompt", INITIAL_PROMPT),
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
@celery_app.task(name="meeting_notes_tasks.transcribe_task")
def transcribe_task(file_path, save_output=True, language=None, initial_prompt=None):
    """Celery task to transcribe audio file only (no summarization)"""
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
        print(f"Starting audio transcription with language: {language or DEFAULT_LANGUAGE}...")
        transcription_data = transcribe_audio(file_path, language, initial_prompt)
        transcription_text = transcription_data.get("text", "").strip()
        print(f"Transcription complete. Text length: {len(transcription_text)} chars")
        
        # Check if we got any text from transcription
        if not transcription_text:
            print("Warning: Transcription produced empty text!")
        
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
        
        # Export everything to JSON if requested
        json_file = None
        metadata = None
        
        if save_output:
            audio_filename = os.path.basename(file_path)
            json_file, metadata = export_to_json(
                recording_data, 
                transcription_data, 
                audio_filename
            )
        
        result = {
            "transcription": {
                "text": transcription_text,
                "language": transcription_data.get("language", language or DEFAULT_LANGUAGE),
                "processing_time_seconds": transcription_data.get("transcription_duration"),
                "segments": transcription_data.get("segments", [])
            },
            "json_file": json_file
        }
        
        if "temp/" in file_path and os.path.exists(file_path):
            os.remove(file_path)
            
        return result
            
    except Exception as e:
        if "temp/" in file_path and os.path.exists(file_path):
            os.remove(file_path)
        return {"error": f"Error processing file: {str(e)}"}

@celery_app.task(name="meeting_notes_tasks.summarize_task")
def summarize_task(text):
    """Celery task to summarize text using Groq API"""
    return summarize_with_groq(text)

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Meeting Notes API", "status": "running", "model": MODEL_NAME, "device": device}

@app.post("/transcribe", response_model=TaskResponse)
async def transcribe_endpoint(
    file: UploadFile = File(...),
    save_output: bool = Form(False),
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None
):
    """
    Start async task to transcribe an audio file
    
    - **file**: Audio file to transcribe (WAV format recommended)
    - **save_output**: Whether to save the output to a JSON file
    - **language**: Language code (e.g., 'tr' for Turkish, 'en' for English)
    - **initial_prompt**: Initial prompt to guide the transcription
    """
    # Create temp directory if it doesn't exist
    ensure_export_directory()
    
    # Save uploaded file to temp directory
    temp_file_path = f"temp/{uuid.uuid4()}.wav"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Start celery task
    try:
        task = transcribe_task.delay(
            temp_file_path, 
            save_output,
            language,
            initial_prompt
        )
        
        # Save task information to database
        options = {
            "save_output": save_output,
            "language": language,
            "initial_prompt": initial_prompt,
            "file_path": temp_file_path
        }
        db_manager.save_task(task.id, "transcribe", "PENDING", options)
        
        return {
            "task_id": task.id,
            "status": "PENDING",
            "message": f"Transcription task started with language: {language or DEFAULT_LANGUAGE}"
        }
    except Exception as e:
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
    
    if not request.text or request.text.strip() == "":
        raise HTTPException(status_code=400, detail="Empty text provided. Please provide text to summarize.")
    
    print(f"Starting summarization task for text of length: {len(request.text)} chars")
    
    try:
        task = summarize_task.delay(request.text)
        
        # Save task information to database
        options = {
            "text_length": len(request.text)
        }
        db_manager.save_task(task.id, "summarize", "PENDING", options)
        
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
    # First check if we have the result in the database
    task_with_response = db_manager.get_task_with_response(task_id)
    
    if task_with_response and "result" in task_with_response:
        # Return result from database
        return {
            "task_id": task_id,
            "status": task_with_response["status"],
            "result": task_with_response["result"]
        }
    
    # If not in database or no result yet, check Celery
    task_result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": task_result.status
    }
    
    # Update task status in database
    db_manager.update_task_status(task_id, task_result.status)
    
    if task_result.ready():
        if task_result.successful():
            result = task_result.get()
            response["result"] = result
            
            # Save response to database
            db_manager.save_task_response(task_id, result)
            
            # Remove task from Celery backend
            task_result.forget()
        else:
            error = {"error": str(task_result.result)}
            response["result"] = error
            
            # Save error response to database
            db_manager.save_task_response(task_id, error)
            
            # Remove task from Celery backend
            task_result.forget()
    
    return response

@app.get("/task/{task_id}/text")
async def get_transcription_text(task_id: str):
    """
    Get only the transcription text for a completed transcription task
    
    - **task_id**: ID of the transcription task
    """
    # First check if we have the result in the database
    task_with_response = db_manager.get_task_with_response(task_id)
    
    if task_with_response and "result" in task_with_response:
        result = task_with_response["result"]
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        if "transcription" not in result or "text" not in result["transcription"]:
            raise HTTPException(status_code=500, detail="No transcription text found in result")
        
        return {"text": result["transcription"]["text"]}
    
    # If not in database, check Celery
    task_result = AsyncResult(task_id, app=celery_app)
    
    if not task_result.ready():
        raise HTTPException(status_code=400, detail="Task is not complete yet")
    
    if not task_result.successful():
        error = {"error": str(task_result.result)}
        # Save error response to database
        db_manager.save_task_response(task_id, error)
        # Remove task from Celery backend
        task_result.forget()
        raise HTTPException(status_code=500, detail=f"Task failed: {str(task_result.result)}")
    
    result = task_result.get()
    
    # Save result to database
    db_manager.save_task_response(task_id, result)
    
    # Remove task from Celery backend
    task_result.forget()
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    if "transcription" not in result or "text" not in result["transcription"]:
        raise HTTPException(status_code=500, detail="No transcription text found in result")
    
    return {"text": result["transcription"]["text"]}

@app.get("/tasks", response_model=List[Dict[str, Any]])
async def list_tasks(limit: int = 100, offset: int = 0):
    """
    List recent tasks with pagination
    
    - **limit**: Maximum number of tasks to return
    - **offset**: Number of tasks to skip
    """
    return db_manager.list_tasks(limit, offset)

@app.get("/models")
async def get_models():
    """Get information about available models and supported languages"""
    whisper_models = ["tiny", "base", "small", "medium", "large-v3"]
    
    # List of supported languages in Whisper
    supported_languages = {
        "en": "English",
        "tr": "Turkish",
        "de": "German",
        "es": "Spanish",
        "fr": "French",
        "it": "Italian",
        "pt": "Portuguese",
        "nl": "Dutch",
        "ja": "Japanese",
        "zh": "Chinese",
        "ru": "Russian",
        "ar": "Arabic",
        "hi": "Hindi",
        # Add more languages as needed
    }
    
    return {
        "current_whisper_model": MODEL_NAME,
        "available_whisper_models": whisper_models,
        "current_summary_model": GROQ_MODEL,
        "device": device,
        "default_language": DEFAULT_LANGUAGE,
        "supported_languages": supported_languages
    }

if __name__ == "__main__":
    ensure_export_directory()
    uvicorn.run("restapi:app", host="0.0.0.0", port=8000, reload=True) 