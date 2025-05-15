import whisper
import pyaudio
import wave
import os
import tempfile
import time
import torch
import threading
import json
import datetime
import uuid
import requests
from pathlib import Path
from dotenv import load_dotenv

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

def ensure_export_directory():
    """Ensure the export directory exists"""
    Path(EXPORT_DIRECTORY).mkdir(exist_ok=True)
    
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
    print("Whisper modeli yükleniyor... (Loading Whisper model...)")
    model = whisper.load_model(MODEL_NAME).to(device)
    print("Model yüklendi. Çevriliyor... (Model loaded. Transcribing...)")
    
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
        print("Groq API anahtarı bulunamadı. (Groq API key not found.)")
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
        print(f"Groq API ile özet oluşturma hatası: {str(e)}")
        print(f"Error summarizing with Groq API: {str(e)}")
        return None

def export_to_json(recording_data, transcription_data, audio_filename=None, summary_data=None):
    """Export all data to a structured JSON file"""
    ensure_export_directory()
    
    # Create metadata structure
    metadata = {
        "recording": {
            "start_time": recording_data["start_datetime"].isoformat(),
            "end_time": recording_data["end_datetime"].isoformat(),
            "duration_seconds": recording_data["duration_seconds"],
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
    if summary_data:
        metadata["summary"] = summary_data
    
    # Save JSON to file
    filename = os.path.join(EXPORT_DIRECTORY, generate_filename())
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return filename

def main():
    """Main function"""
    print("\n=== TÜRKÇE SES KAYIT, ÇEVİRİ VE EXPORT UYGULAMASI ===")
    print("=== TURKISH VOICE RECORDING, TRANSCRIPTION AND EXPORT APP ===")
    
    while True:
        print("\nNe yapmak istiyorsunuz? (What would you like to do?)")
        print("1. Kayıt başlat, çevir ve JSON olarak kaydet (Start recording, transcribe and save as JSON)")
        print("2. Çıkış (Exit)")
        
        choice = input("\nSeçiminiz (Your choice) [1/2]: ")
        
        if choice == "1":
            # Record audio
            recording_data = record_audio()
            
            if recording_data["frames"]:
                print("\nKayıt işleniyor... (Processing recording...)")
                
                # Save audio to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                temp_filename = temp_file.name
                temp_file.close()
                
                # Create a permanent audio file
                audio_filename = os.path.join(EXPORT_DIRECTORY, generate_filename("wav"))
                ensure_export_directory()
                
                # Save audio to both temp and permanent files
                save_audio_to_file(recording_data["frames"], temp_filename)
                save_audio_to_file(recording_data["frames"], audio_filename)
                
                try:
                    # Transcribe audio
                    transcription_data = transcribe_audio(temp_filename)
                    
                    # Generate summary using Groq API
                    print("Metin özetleniyor... (Summarizing text...)")
                    summary_data = summarize_with_groq(transcription_data["text"])
                    
                    # Export everything to JSON
                    json_file = export_to_json(
                        recording_data, 
                        transcription_data, 
                        os.path.basename(audio_filename),
                        summary_data
                    )
                    
                    # Print transcript and summary
                    print("\n" + "=" * 80)
                    print("ÇEVİRİ SONUCU / TRANSCRIPTION RESULT:")
                    print("=" * 80)
                    print(transcription_data["text"])
                    print("=" * 80)
                    
                    if summary_data:
                        print("ÖZET / SUMMARY:")
                        print("=" * 80)
                        print(summary_data["summary"])
                        print("=" * 80)
                    
                    print(f"\nJSON dosyası kaydedildi (JSON file saved): {json_file}")
                    print(f"Ses dosyası kaydedildi (Audio file saved): {audio_filename}")
                    print("=" * 80 + "\n")
                    
                    # Clean up temporary file
                    os.unlink(temp_filename)
                except Exception as e:
                    print(f"\nİşlem sırasında hata oluştu: {str(e)}")
                    print(f"Error during processing: {str(e)}")
        
        elif choice == "2":
            print("\nProgram sonlandırılıyor... (Terminating program...)")
            break
            
        else:
            print("\nGeçersiz seçim. Lütfen tekrar deneyin.")
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram sonlandırıldı. (Program terminated.)")
    except Exception as e:
        print(f"\nBir hata oluştu: {str(e)}")
        print(f"An error occurred: {str(e)}") 