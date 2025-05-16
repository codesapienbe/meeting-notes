import os
import sys
import traceback  # Add this import for exception handling

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
import asyncio
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
app = FastAPI(
    title="Meeting Notes with Vector Embeddings", 
    description="API for transcribing meeting audio to text, generating vector embeddings, and summarizing meeting notes"
)

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
    embeddings: Optional[List[float]] = None

# Import the ZeroMQ publisher from tasks module
try:
    from tasks import publish_task_update as zmq_publish_task_update
    ZMQ_AVAILABLE = True
    print("ZeroMQ publisher available - using ZeroMQ for real-time updates")
except ImportError:
    ZMQ_AVAILABLE = False
    print("ZeroMQ publisher not available - falling back to database-only updates")

# Function to send task updates via ZeroMQ
def send_task_update(task_id: str, status: str, result: Dict[str, Any] = None, progress: str = None):
    """Send a task update via ZeroMQ"""
    # Always update the database status first
    db_manager.update_task_status(task_id, status)
    
    # If we have a result, save it to the database
    if result and status == "SUCCESS":
        db_manager.save_task_response(task_id, result)
    
    # Send via ZeroMQ (sync) if available
    if ZMQ_AVAILABLE:
        # Prepare data for ZeroMQ
        task_data = db_manager.get_task(task_id)
        options = task_data.get('options') if task_data else None
        
        zmq_data = {
            "result": result,
            "progress": progress,
            "options": options
        }
        
        try:
            # Use the imported ZeroMQ publisher
            zmq_publish_task_update(task_id, status, zmq_data)
            print(f"Published ZeroMQ update for task {task_id}: {status}")
        except Exception as e:
            print(f"Error publishing ZeroMQ update: {e}")
    else:
        print(f"ZeroMQ not available - status update for task {task_id} saved to database only")

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
    
    # Generate embeddings if text is available
    transcription_text = result.get("text", "").strip()
    if transcription_text:
        try:
            print("Generating vector embeddings for transcription...")
            # Use the database manager method to generate embeddings (ensures consistency)
            result["embeddings"] = db_manager.generate_embeddings(transcription_text)
            if result["embeddings"]:
                print(f"Successfully generated embeddings (vector dimension: {len(result['embeddings'])})")
            else:
                print("Failed to generate embeddings")
        except Exception as e:
            print(f"Error generating embeddings: {e}")
    
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
def transcribe_task(file_path, save_output=True, language=None, initial_prompt=None, model=None):
    """
    Celery task to transcribe an audio file
    
    Args:
        file_path: Path to the audio file
        save_output: Whether to save the output to a file
        language: Language code to use for transcription
        initial_prompt: Initial prompt to guide transcription
        model: Whisper model to use
    
    Returns:
        Dictionary with transcription results
    """
    try:
        task_id = transcribe_task.request.id
        
        # Send status update - STARTED with task info
        model_name = model or MODEL_NAME
        status_msg = f"Starting transcription with {model_name} model"
        send_task_update(task_id, "STARTED", None, status_msg)
        
        # Validate model
        valid_models = ["tiny", "base", "small", "medium", "large-v3"]
        if model_name not in valid_models:
            raise ValueError(f"Invalid model: {model_name}. Must be one of: {', '.join(valid_models)}")
        
        # Validate and check file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        print(f"Processing audio file: {file_path} with {model_name} model")
        
        # Use default language if not specified
        if not language:
            language = DEFAULT_LANGUAGE
        
        # Send status update - PROCESSING with more details
        send_task_update(task_id, "PROGRESS", None, f"Starting transcription with {model_name} model in {language} (10%)")
        print(f"Starting audio transcription with language: {language} using {model_name} model")
        
        # Load model
        send_task_update(task_id, "PROGRESS", None, f"Loading {model_name} model (20%)")
        print(f"Loading Whisper model: {model_name}")
        model_instance = whisper.load_model(model_name, device=device)
        print(f"Model loaded. Transcribing...")
        send_task_update(task_id, "PROGRESS", None, f"Model loaded. Processing audio (30%)")
        
        # Record start time
        start_time = time.time()
        
        # Prepare transcription options
        transcribe_options = {
            "language": language,
            "task": "transcribe",
            "verbose": True
        }
        
        # Add initial prompt if provided
        if initial_prompt:
            print(f"Using initial prompt: {initial_prompt}")
            transcribe_options["initial_prompt"] = initial_prompt
            
        # Print the language we're transcribing with
        print(f"Transcribing with language: {language}")
        
        # Send progress update - 40%
        send_task_update(task_id, "PROGRESS", None, "Processing audio (40%)")
            
        # Perform transcription
        result = model_instance.transcribe(file_path, **transcribe_options)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Send progress update - 80%
        send_task_update(task_id, "PROGRESS", None, "Transcription complete (80%), finalizing result")
        
        # Prepare response
        transcription_response = {
            "text": result["text"],
            "segments": result["segments"],
            "language": result["language"],
            "processing_time_seconds": processing_time,
            "model": model_name
        }
        
        # Save output to JSON file if requested
        output_path = None
        if save_output:
            # Generate output path
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"transcript_{timestamp}.json"
            output_directory = "transcripts"
            
            # Ensure output directory exists
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
                
            # Construct full path
            output_path = os.path.join(output_directory, output_filename)
            
            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(transcription_response, f, ensure_ascii=False, indent=2)
                
            print(f"Transcription saved to {output_path}")
            transcription_response["output_file"] = output_path
        
        # Add file info to response
        transcription_response["file_path"] = file_path
        
        # Build final response
        response = {
            "transcription": transcription_response
        }
        
        # Send final update
        send_task_update(task_id, "PROGRESS", None, "Transcription complete (95%)")
        time.sleep(0.5)  # Ensure progress message is seen before SUCCESS message
        send_task_update(task_id, "SUCCESS", response, "Transcription complete (100%)")
        
        # Return response for Celery result
        return response
    except Exception as e:
        error_message = f"Error in transcription: {str(e)}"
        print(error_message)
        traceback.print_exc()
        
        # Create error response
        response = {
            "error": error_message
        }
        
        # Send error update
        task_id = transcribe_task.request.id
        send_task_update(task_id, "FAILURE", response, "Transcription failed")
        
        # Re-raise for Celery
        raise e

@celery_app.task(name="meeting_notes_tasks.summarize_task")
def summarize_task(text):
    """
    Celery task to summarize a text using LLM
    
    Args:
        text: The text to summarize
    
    Returns:
        Dictionary with summary results
    """
    try:
        task_id = summarize_task.request.id
        
        # Send status update - STARTED
        send_task_update(task_id, "STARTED", None, "Preparing summary (0%)")
        
        # Validate input
        if not text or not isinstance(text, str) or len(text) < 10:
            raise ValueError("Invalid text provided for summarization")
        
        # Send progress update - beginning processing
        send_task_update(task_id, "PROCESSING", None, "Analyzing text... (10%)")
            
        # Record start time
        start_time = time.time()
        
        # Send progress update - 25%
        send_task_update(task_id, "PROCESSING", None, "Generating summary (25%)")
        
        # Perform summarization
        summary = summarize_with_groq(text)
        
        # Send progress update - 75%
        send_task_update(task_id, "PROCESSING", None, "Summary generated (75%), post-processing...")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response
        summary_response = {
            "summary": summary,
            "original_text": text[:500] + "..." if len(text) > 500 else text,  # Truncate for response
            "model": GROQ_MODEL,
            "processing_time_seconds": processing_time
        }
        
        # Save output to JSON file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"summary_{timestamp}.json"
        output_directory = "transcripts"
        
        # Ensure output directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            
        # Construct full path
        output_path = os.path.join(output_directory, output_filename)
        
        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary_response, f, ensure_ascii=False, indent=2)
            
        print(f"Summary saved to {output_path}")
        summary_response["output_file"] = output_path
        
        # Build final response
        response = {
            "summary": summary_response
        }
        
        # Send progress update - 90%
        send_task_update(task_id, "PROCESSING", None, "Finalizing summary (90%)")
        
        # Send final update
        send_task_update(task_id, "SUCCESS", response, "Summary complete (100%)")
        
        # Return response for Celery result
        return response
    except Exception as e:
        error_message = f"Error in summarization: {str(e)}"
        print(error_message)
        traceback.print_exc()
        
        # Create error response
        response = {
            "error": error_message
        }
        
        # Send error update
        task_id = summarize_task.request.id
        send_task_update(task_id, "FAILURE", response, "Summarization failed")
        
        # Re-raise for Celery
        raise e

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Meeting Notes API", "status": "running", "model": MODEL_NAME, "device": device}

@app.post("/transcribe", response_model=TaskResponse)
async def transcribe_endpoint(
    file: UploadFile = File(...),
    save_output: bool = Form(False),
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    model: Optional[str] = Form(None)
):
    """
    Start async task to transcribe audio file
    
    - **file**: Audio file to transcribe
    - **save_output**: Whether to save the output to a file
    - **language**: Language code (e.g., 'tr' for Turkish)
    - **initial_prompt**: Custom prompt to guide the transcription
    - **model**: Whisper model size to use (tiny, base, small, medium, large-v3)
    """
    # Validate model parameter
    valid_models = ["tiny", "base", "small", "medium", "large-v3"]
    if model and model not in valid_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model: {model}. Valid models are: {', '.join(valid_models)}"
        )
    
    # Create a temporary directory to save the uploaded file
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Generate a unique filename
    temp_filename = f"{uuid.uuid4()}.{file.filename.split('.')[-1]}"
    temp_file_path = os.path.join(temp_dir, temp_filename)
    
    # Save the uploaded file
    try:
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"File saved to {temp_file_path}")
        
        # Get file size for logging
        file_size = os.path.getsize(temp_file_path)
        print(f"File size: {file_size / 1024 / 1024:.2f} MB")
        
        # Log selected model
        selected_model = model if model else MODEL_NAME
        print(f"Using model: {selected_model}")
        
        # Start async task
        task = transcribe_task.delay(
            temp_file_path, 
            save_output=save_output,
            language=language,
            initial_prompt=initial_prompt,
            model=selected_model
        )
        
        # Save task information to database
        options = {
            "file_path": temp_file_path,
            "language": language or DEFAULT_LANGUAGE,
            "initial_prompt": initial_prompt,
            "model": selected_model,
            "save_output": save_output
        }
        db_manager.save_task(task.id, "transcribe", "PENDING", options)
        
        return {
            "task_id": task.id,
            "status": "PENDING",
            "message": f"Transcription task started with model: {selected_model}"
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
async def get_task_status(task_id: str, include_embeddings: bool = False):
    """
    Check the status of a task
    
    - **task_id**: ID of the task to check
    - **include_embeddings**: Whether to include vector embeddings in the response
    """
    # First check if we have the result in the database
    task_with_response = db_manager.get_task_with_response(task_id, include_embeddings)
    
    if task_with_response and "result" in task_with_response:
        # Return result from database
        response = {
            "task_id": task_id,
            "status": task_with_response["status"],
            "result": task_with_response["result"]
        }
        
        # Include embeddings if requested and available
        if include_embeddings and "embeddings" in task_with_response:
            response["embeddings"] = task_with_response["embeddings"]
            
        return response
    
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
            
            # Include embeddings if requested
            if include_embeddings:
                # Get the response with embeddings
                saved_response = db_manager.get_task_response(task_id, True)
                if saved_response and "embeddings" in saved_response:
                    response["embeddings"] = saved_response["embeddings"]
            
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
async def list_tasks(limit: int = 100, offset: int = 0, include_embeddings: bool = False):
    """
    List all tasks with pagination
    
    - **limit**: Maximum number of tasks to return
    - **offset**: Number of tasks to skip
    - **include_embeddings**: Whether to include vector embeddings in the response
    """
    return db_manager.list_tasks(limit, offset, include_embeddings)

@app.get("/models")
async def get_models():
    """Get information about available models and supported languages"""
    whisper_models = ["tiny", "base", "small", "medium", "large-v3"]
    
    # Model size and performance information
    model_info = {
        "tiny": {"size": "39M", "speed": "Very Fast", "accuracy": "Low"},
        "base": {"size": "74M", "speed": "Fast", "accuracy": "Basic"},
        "small": {"size": "244M", "speed": "Medium", "accuracy": "Good"},
        "medium": {"size": "769M", "speed": "Slow", "accuracy": "Better"},
        "large-v3": {"size": "1.5GB", "speed": "Very Slow", "accuracy": "Best"}
    }
    
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
        "whisper_model_info": model_info,
        "current_summary_model": GROQ_MODEL,
        "device": device,
        "default_language": DEFAULT_LANGUAGE,
        "supported_languages": supported_languages
    }

if __name__ == "__main__":
    ensure_export_directory()
    uvicorn.run("restapi:app", host="0.0.0.0", port=8000, reload=True) 