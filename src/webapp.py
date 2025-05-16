import streamlit as st
import requests
import json
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional
import tempfile
import threading
import uuid
import zmq

# Constants
API_URL = "http://localhost:8000"
ZMQ_PUB_PORT = 5556  # Must match the port in tasks.py
ZMQ_HEARTBEAT_INTERVAL = 5  # Seconds between heartbeats
ZMQ_CONNECTION_TIMEOUT = 15  # Seconds without messages before considering connection lost
ZMQ_RECONNECT_ATTEMPT_INTERVAL = 10  # Seconds between reconnection attempts

# Set page config
st.set_page_config(
    page_title="Meeting Notes",
    page_icon="📝",
    layout="centered"
)

# Initialize session state
if 'current_task_id' not in st.session_state:
    st.session_state.current_task_id = None
if 'task_status' not in st.session_state:
    st.session_state.task_status = {}
if 'transcription_text' not in st.session_state:
    st.session_state.transcription_text = ""
if 'summary_text' not in st.session_state:
    st.session_state.summary_text = ""
if 'progress_status' not in st.session_state:
    st.session_state.progress_status = ""
if 'progress_value' not in st.session_state:
    st.session_state.progress_value = 0.0
if 'zmq_subscriber' not in st.session_state:
    st.session_state.zmq_subscriber = None
if 'zmq_context' not in st.session_state:
    st.session_state.zmq_context = None
if 'zmq_connected' not in st.session_state:
    st.session_state.zmq_connected = False
if 'zmq_thread' not in st.session_state:
    st.session_state.zmq_thread = None
if 'last_zmq_message_time' not in st.session_state:
    st.session_state.last_zmq_message_time = time.time()
if 'detailed_status' not in st.session_state:
    st.session_state.detailed_status = "Waiting for task to start..."
if 'status_log' not in st.session_state:
    st.session_state.status_log = []  # List to store multiple status messages
if 'last_status_update' not in st.session_state:
    st.session_state.last_status_update = time.time()
if 'available_models' not in st.session_state:
    st.session_state.available_models = []
if 'whisper_model_sizes' not in st.session_state:
    # Information about model sizes and speeds (rough estimates)
    st.session_state.whisper_model_sizes = {
        "tiny": {"size": "39M", "speed": "Very Fast", "accuracy": "Low"},
        "base": {"size": "74M", "speed": "Fast", "accuracy": "Basic"},
        "small": {"size": "244M", "speed": "Medium", "accuracy": "Good"},
        "medium": {"size": "769M", "speed": "Slow", "accuracy": "Better"},
        "large-v3": {"size": "1.5GB", "speed": "Very Slow", "accuracy": "Best"}
    }

# Use thread-safe events instead of session state for thread control
zmq_stop_event = threading.Event()
zmq_message_received_event = threading.Event()

def fetch_available_models():
    """Fetch available models from the API"""
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            st.session_state.available_models = data.get("available_whisper_models", [])
            st.session_state.current_model = data.get("current_whisper_model", "large-v3")
            st.session_state.languages = data.get("supported_languages", {})
            return data
        return None
    except:
        return None

def check_api_connection() -> bool:
    """Check if the API is accessible"""
    try:
        response = requests.get(f"{API_URL}/", timeout=3)
        if response.status_code == 200:
            # While we're at it, fetch available models
            fetch_available_models()
            return True
        return False
    except:
        return False

def zmq_heartbeat_thread(stop_event, subscriber):
    """Thread function to ping the ZMQ connection periodically"""
    print("Starting ZeroMQ heartbeat thread")
    
    while not stop_event.is_set():
        try:
            # Check if subscriber is still valid
            if subscriber and not stop_event.is_set():
                # Try to send a ping by checking for messages
                try:
                    if subscriber.poll(100) != 0:
                        # There's a message available, receive it in the main listener thread
                        zmq_message_received_event.set()
                except zmq.ZMQError as e:
                    print(f"ZMQ heartbeat error: {e}")
        except Exception as e:
            print(f"Error in heartbeat thread: {e}")
        
        # Set a flag to indicate the message time is being updated by the heartbeat
        # Sleep for the heartbeat interval
        for _ in range(ZMQ_HEARTBEAT_INTERVAL * 10):  # Use smaller sleeps for more responsive stopping
            if stop_event.is_set():
                break
            time.sleep(0.1)
    
    print("ZeroMQ heartbeat thread stopping")

def initialize_zmq_subscriber(task_id):
    """Initialize ZeroMQ subscriber for a specific task"""
    # Clean up existing resources
    cleanup_zmq_resources()
    
    try:
        # Create a new context
        st.session_state.zmq_context = zmq.Context()
        
        # Create and configure subscriber socket
        subscriber = st.session_state.zmq_context.socket(zmq.SUB)
        subscriber.setsockopt(zmq.RCVTIMEO, 1000)  # 1-second timeout for non-blocking receives
        subscriber.setsockopt(zmq.LINGER, 0)       # Don't wait on close
        
        # Set up socket options for heartbeats and reliability
        subscriber.setsockopt(zmq.TCP_KEEPALIVE, 1)
        subscriber.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
        subscriber.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 15)
        subscriber.setsockopt(zmq.TCP_KEEPALIVE_CNT, 4)
        subscriber.setsockopt(zmq.RECONNECT_IVL, 1000)  # Reconnect interval in ms
        subscriber.setsockopt(zmq.RECONNECT_IVL_MAX, 5000)  # Max reconnect interval
        
        # Connect to the publisher
        try:
            subscriber.connect(f"tcp://localhost:{ZMQ_PUB_PORT}")
            print(f"ZeroMQ subscriber connected to port {ZMQ_PUB_PORT}")
        except zmq.ZMQError as e:
            print(f"Failed to connect ZeroMQ subscriber: {e}")
            subscriber.close()
            return False
        
        # Subscribe to specific task updates
        topic = f"task.{task_id}"
        subscriber.setsockopt_string(zmq.SUBSCRIBE, topic)
        
        # Store the subscriber
        st.session_state.zmq_subscriber = subscriber
        st.session_state.zmq_connected = True
        st.session_state.last_zmq_message_time = time.time()  # Initialize with current time
        
        # Reset the heartbeat stop event
        global zmq_stop_event, zmq_message_received_event
        zmq_stop_event.clear()
        zmq_message_received_event.clear()
        
        # Start heartbeat thread
        heartbeat_thread = threading.Thread(
            target=zmq_heartbeat_thread,
            args=(zmq_stop_event, subscriber),
            daemon=True
        )
        heartbeat_thread.start()
        
        # Start listener thread
        zmq_thread = threading.Thread(
            target=zmq_listener_thread, 
            args=(subscriber, task_id, zmq_stop_event, zmq_message_received_event),
            daemon=True
        )
        zmq_thread.start()
        st.session_state.zmq_thread = zmq_thread
        
        print(f"ZeroMQ initialization complete for task {task_id}")
        return True
    except Exception as e:
        print(f"Error initializing ZeroMQ subscriber: {e}")
        cleanup_zmq_resources()
        return False

def cleanup_zmq_resources():
    """Clean up ZeroMQ resources"""
    # Signal heartbeat thread to stop
    global zmq_stop_event
    zmq_stop_event.set()
    
    # Close subscriber
    if hasattr(st.session_state, 'zmq_subscriber') and st.session_state.zmq_subscriber:
        try:
            st.session_state.zmq_subscriber.close()
        except:
            pass
        st.session_state.zmq_subscriber = None
    
    # Terminate context
    if hasattr(st.session_state, 'zmq_context') and st.session_state.zmq_context:
        try:
            st.session_state.zmq_context.term()
        except:
            pass
        st.session_state.zmq_context = None
    
    # Reset connection state
    st.session_state.zmq_connected = False

def update_detailed_status(status, progress_message=None):
    """Update the detailed status display with timestamps and more information"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    status_message = f"[{timestamp}] "
    
    # Extract percentage from progress message if available
    if progress_message and "%" in progress_message:
        import re
        percent_match = re.search(r'(\d+)%', progress_message)
        if percent_match:
            try:
                percent = int(percent_match.group(1))
                st.session_state.progress_value = percent / 100.0
                print(f"Progress bar updated: {percent}%")
            except ValueError:
                pass
    
    # Create a more descriptive message based on status
    if status == "PENDING":
        status_message += "Task pending, waiting to start..."
    elif status == "STARTED":
        status_message += "Task started, processing..."
        # Set minimum progress for STARTED status
        if st.session_state.progress_value < 0.1:
            st.session_state.progress_value = 0.1
    elif status == "PROGRESS":
        if progress_message:
            status_message += f"In progress: {progress_message}"
        else:
            status_message += "Processing audio..."
        # Set minimum progress for PROGRESS status
        if st.session_state.progress_value < 0.2:
            st.session_state.progress_value = 0.2
    elif status == "SUCCESS":
        status_message += "✅ Task completed successfully!"
        # Force progress to 100% when complete
        st.session_state.progress_value = 1.0
    elif status == "FAILURE":
        status_message += "❌ Task failed! See details below."
    else:
        status_message += f"Status: {status}"
        if progress_message:
            status_message += f" - {progress_message}"
    
    # Update the status
    st.session_state.detailed_status = status_message
    print(f"Updated detailed status to: {status_message}")
    
    # Add the message to the status log for history
    if len(st.session_state.status_log) > 0:
        # Don't add duplicate messages, but do update if the content is different
        if st.session_state.status_log[-1].split("] ", 1)[1] != status_message.split("] ", 1)[1]:
            st.session_state.status_log.append(status_message)
            print(f"Added status log entry: {status_message}")
    else:
        # First message, just add it
        st.session_state.status_log.append(status_message)
        print(f"Added first status log entry: {status_message}")
    
    # Keep the log at a reasonable size
    if len(st.session_state.status_log) > 10:
        st.session_state.status_log = st.session_state.status_log[-10:]
    
    st.session_state.last_status_update = time.time()
    
    # Force an immediate rerun for important status changes
    if status in ["SUCCESS", "FAILURE"]:
        print(f"Critical status update ({status}), forcing rerun")
        st.rerun()

def zmq_listener_thread(subscriber, task_id, stop_event, message_received_event):
    """Thread function to listen for ZeroMQ messages"""
    print(f"Starting ZeroMQ listener thread for task {task_id}")
    last_rerun_time = time.time()
    consecutive_errors = 0
    
    try:
        while not stop_event.is_set():
            try:
                # Wait for notification from heartbeat thread or check for messages
                if message_received_event.wait(0.1):  # Short timeout to be responsive
                    message_received_event.clear()
                
                # Poll with timeout to allow thread termination
                if subscriber.poll(100) == 0:  # 100ms timeout
                    continue
                
                # Receive multipart message (topic, data)
                topic, message_data = subscriber.recv_multipart(zmq.NOBLOCK)
                topic = topic.decode('utf-8')
                
                # Update timestamp to track connection status
                st.session_state.last_zmq_message_time = time.time()
                st.session_state.zmq_connected = True
                consecutive_errors = 0  # Reset error counter on successful message
                
                try:
                    data = json.loads(message_data.decode('utf-8'))
                    print(f"ZeroMQ received: {topic} - {data.get('status')}")
                    
                    # Process the message
                    if data.get('task_id') == task_id:
                        # Update task status
                        st.session_state.task_status = data
                        current_status = data.get('status', 'PENDING')
                        
                        # Update progress if available
                        progress_message = None
                        if 'data' in data and 'data' in data['data'] and 'progress' in data['data']['data']:
                            progress_message = data['data']['data']['progress']
                            st.session_state.progress_status = progress_message
                            
                            # Extract percentage from the progress message if available
                            if isinstance(progress_message, str):
                                # Try to find percentage in the format "(X%)" or "X%"
                                if "%" in progress_message:
                                    # First try to match pattern "... (X%)"
                                    import re
                                    percent_match = re.search(r'\((\d+)%\)', progress_message)
                                    if percent_match:
                                        try:
                                            percent = int(percent_match.group(1))
                                            st.session_state.progress_value = percent / 100.0
                                            print(f"Parsed progress: {percent}%")
                                        except (ValueError, IndexError) as e:
                                            print(f"Error parsing matched progress percentage: {e}")
                                    else:
                                        # If no match for (X%), try to find any number followed by %
                                        percent_match = re.search(r'(\d+)%', progress_message)
                                        if percent_match:
                                            try:
                                                percent = int(percent_match.group(1))
                                                st.session_state.progress_value = percent / 100.0
                                                print(f"Parsed progress: {percent}%")
                                            except (ValueError, IndexError) as e:
                                                print(f"Error parsing progress percentage: {e}")
                                        else:
                                            # Increment progress for animation effect if no percentage found
                                            st.session_state.progress_value = min(st.session_state.progress_value + 0.05, 0.95)
                                else:
                                    # Increment progress for animation effect if no percentage found
                                    st.session_state.progress_value = min(st.session_state.progress_value + 0.05, 0.95)
                        
                        # Update detailed status for better visibility
                        update_detailed_status(current_status, progress_message)
                        
                        # Update transcription or summary text if available
                        result = data.get('data', {}).get('data', {}).get('result', {})
                        if result:
                            if 'transcription' in result and 'text' in result['transcription']:
                                st.session_state.transcription_text = result['transcription']['text']
                            elif 'summary' in result and isinstance(result['summary'], dict) and 'summary' in result['summary']:
                                st.session_state.summary_text = result['summary']['summary']
                        
                        # Set progress to 100% when task is complete
                        if current_status == "SUCCESS":
                            st.session_state.progress_value = 1.0
                            
                        # Force UI update on every status change, but limit rerun frequency
                        current_time = time.time()
                        if current_time - last_rerun_time > 0.5:  # Limit reruns to every 0.5 second
                            print(f"Status changed to {current_status}, triggering UI update")
                            last_rerun_time = current_time
                            st.rerun()
                except json.JSONDecodeError:
                    print(f"Invalid JSON received: {message_data}")
                except Exception as e:
                    print(f"Error processing ZMQ message: {e}")
            
            except zmq.ZMQError as e:
                if e.errno == zmq.EAGAIN:
                    # No message available, just continue
                    pass
                else:
                    print(f"ZMQ error in listener thread: {e}")
                    consecutive_errors += 1
                    if consecutive_errors > 5:
                        print("Too many consecutive ZMQ errors, reconnecting...")
                        st.session_state.zmq_connected = False
                        # Let the heartbeat thread handle reconnection
                        time.sleep(1)
                        consecutive_errors = 0
                    continue
            except Exception as e:
                print(f"Unexpected error in ZMQ listener: {e}")
                consecutive_errors += 1
                if consecutive_errors > 5:
                    st.session_state.zmq_connected = False
                    print("Too many errors in listener, reconnecting...")
                time.sleep(0.5)
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.01)
    except Exception as e:
        print(f"Critical error in ZeroMQ listener thread: {e}")
    finally:
        print("ZeroMQ listener thread exiting")
        st.session_state.zmq_connected = False

def is_realtime_connected():
    """Check if ZeroMQ connection is active"""
    if st.session_state.zmq_connected:
        # Check if we've received a message recently
        current_time = time.time()
        if current_time - st.session_state.last_zmq_message_time < ZMQ_CONNECTION_TIMEOUT:
            return True
    
    return False

def submit_audio_for_transcription(audio_file, language=None, initial_prompt=None, model=None) -> Optional[str]:
    """Submit an audio file for transcription"""
    try:
        files = {"file": (os.path.basename(audio_file.name), audio_file, "audio/wav")}
        data = {"save_output": "true"}
        
        if language:
            data["language"] = language
        if initial_prompt:
            data["initial_prompt"] = initial_prompt
        if model:
            data["model"] = model
            
        response = requests.post(
            f"{API_URL}/transcribe",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("task_id")
        else:
            st.error(f"Error submitting transcription: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error submitting transcription: {e}")
        return None

def request_summary(text: str) -> Optional[str]:
    """Request a summary of the text"""
    try:
        response = requests.post(
            f"{API_URL}/summarize",
            json={"text": text}
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("task_id")
        else:
            st.error(f"Error requesting summary: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error requesting summary: {e}")
        return None

def format_duration(seconds):
    """Format duration in seconds to a human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    else:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes} min {seconds:.1f} sec"

def clear_current_task():
    """Clear the current task and reset the state"""
    # Clean up ZeroMQ resources
    cleanup_zmq_resources()
    
    # Reset state variables
    st.session_state.current_task_id = None
    st.session_state.transcription_text = ""
    st.session_state.summary_text = ""
    st.session_state.progress_status = ""
    st.session_state.task_status = {}
    st.session_state.progress_value = 0.0
    st.session_state.detailed_status = "Waiting for task to start..."

def check_task_status(task_id):
    """Check the current status of a task via the API"""
    try:
        response = requests.get(f"{API_URL}/task/{task_id}")
        if response.status_code == 200:
            result = response.json()
            return result
        return None
    except:
        return None

def format_model_info(model_name):
    """Format the model info for display"""
    model_info = st.session_state.whisper_model_sizes.get(model_name, {"size": "Unknown", "speed": "Unknown", "accuracy": "Unknown"})
    return f"{model_name} ({model_info['size']}, {model_info['speed']}, Accuracy: {model_info['accuracy']})"

def poll_task_status():
    """
    This function automatically runs when displaying the active task page
    to check task status from API and fallback Redis channel
    """
    task_id = st.session_state.current_task_id
    task_status = st.session_state.task_status
    status = task_status.get("status", "PENDING")
    
    # Always check API status on page load for reliable updates
    updated = False
    
    # First, try to get updates from Redis fallback mechanism
    try:
        # Import Redis client
        from redis import Redis
        import json
        
        # Get Redis connection info from environment
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        
        # Parse Redis URL to components (very basic parser)
        if redis_url.startswith("redis://"):
            redis_url = redis_url[8:]  # Remove redis:// prefix
            
        # Default host and port
        redis_host = "localhost"
        redis_port = 6379
        redis_db = 0
        
        # Extract host, port, db if provided
        if "/" in redis_url:
            redis_address, redis_db = redis_url.rsplit("/", 1)
            redis_db = int(redis_db)
            if ":" in redis_address:
                redis_host, redis_port = redis_address.split(":", 1)
                redis_port = int(redis_port)
            else:
                redis_host = redis_address
        
        # Connect to Redis
        redis_client = Redis(host=redis_host, port=redis_port, db=redis_db)
        
        # CRITICAL FIX: Check the main status key directly - match the format in tasks.py
        status_key = f"task_status:{task_id}"
        message_data = redis_client.get(status_key)
        
        if message_data:
            # Found a status update in Redis
            try:
                message = json.loads(message_data.decode('utf-8'))
                print(f"Found Redis status update for task {task_id}: {message.get('status', 'unknown')}")
                
                # Update our status and progress
                new_status = message.get("status", status)
                
                # Force status change
                if "task_status" in st.session_state:
                    st.session_state.task_status["status"] = new_status
                
                # Update progress based on status
                if new_status == "SUCCESS":
                    st.session_state.progress_value = 1.0
                elif new_status in ["PROGRESS", "STARTED"] and st.session_state.progress_value < 0.2:
                    st.session_state.progress_value = 0.2
                
                # Update status message and extract progress percentage if available
                progress_message = None
                if "data" in message and message["data"] and "progress" in message["data"]:
                    progress_message = message["data"]["progress"]
                    # Try to parse percentage from progress message
                    if progress_message and "%" in progress_message:
                        import re
                        percent_match = re.search(r'(\d+)%', progress_message)
                        if percent_match:
                            try:
                                percent = int(percent_match.group(1))
                                st.session_state.progress_value = percent / 100.0
                                print(f"Progress updated from Redis: {percent}%")
                            except ValueError:
                                pass
                
                # Update the detailed status and status log
                update_detailed_status(new_status, progress_message)
                print(f"Updated status from Redis: {new_status}")
                updated = True
            except json.JSONDecodeError:
                # Ignore invalid JSON
                print(f"Invalid JSON in Redis key {status_key}")
            except Exception as e:
                print(f"Error processing Redis status: {e}")
        else:
            print(f"No Redis status found for key {status_key}")
            
            # Fallback to checking by status type as before
            for check_status in ["SUCCESS", "FAILURE", "PROGRESS", "STARTED", "PENDING"]:
                status_type_key = f"task_status:{task_id}:{check_status}"
                message_data = redis_client.get(status_type_key)
                
                if message_data:
                    print(f"Found status by type: {status_type_key}")
                    try:
                        message = json.loads(message_data.decode('utf-8'))
                        
                        # Update our status and progress
                        new_status = message.get("status", status)
                        
                        # Force status change
                        if "task_status" in st.session_state:
                            st.session_state.task_status["status"] = new_status
                        
                        # Update progress based on status
                        if new_status == "SUCCESS":
                            st.session_state.progress_value = 1.0
                        elif new_status in ["PROGRESS", "STARTED"]:
                            st.session_state.progress_value = max(0.2, st.session_state.progress_value)
                        
                        # Update status message
                        progress_message = None
                        if "data" in message and message["data"] and "progress" in message["data"]:
                            progress_message = message["data"]["progress"]
                        
                        update_detailed_status(new_status, progress_message)
                        print(f"Updated status from Redis type key: {new_status}")
                        updated = True
                        break
                    except Exception as e:
                        print(f"Error processing Redis type status: {e}")
    except ImportError:
        print("Redis not available for fallback checks")
    except Exception as e:
        print(f"Error checking Redis fallback: {e}")
    
    # Then check the API
    try:
        api_status = check_task_status(task_id)
        if api_status:
            # Get the status from the API response
            api_task_status = api_status.get("status", status) 
            
            # Update the progress based on status
            if api_task_status in ["PROGRESS", "STARTED", "PROCESSING"] and st.session_state.progress_value == 0:
                # Set a default progress value to show activity
                st.session_state.progress_value = 0.2
            elif api_task_status == "SUCCESS":
                st.session_state.progress_value = 1.0
            
            # If we get a new status, update it in the session
            if api_task_status != status:
                # Update session with new status
                if "task_status" in st.session_state:
                    st.session_state.task_status["status"] = api_task_status
                
                # Update the progress message if we can
                update_detailed_status(api_task_status, "Retrieved latest status from API")
                
                # Get result data if task is complete
                if api_task_status == "SUCCESS" and "result" in api_status:
                    result = api_status["result"]
                    # Update transcription result
                    if "transcription" in result and "text" in result["transcription"]:
                        st.session_state.transcription_text = result["transcription"]["text"]
                    # Update summary result
                    elif "summary" in result and "summary" in result["summary"]:
                        st.session_state.summary_text = result["summary"]["summary"]
                
                updated = True
                # Always force a rerun when API status changes for better responsiveness
                st.rerun()
    except Exception as e:
        print(f"Error in poll_task_status: {e}")
    
    # Force a UI update if anything changed
    if updated:
        return True
    
    # No updates, continue normal operation
    return False

# Add a timer to ensure regular updates even when ZeroMQ is not connecting
def auto_refresh():
    """Automatically refreshes the app periodically to poll for updates"""
    # Check if we have an active task that needs refreshing
    if st.session_state.current_task_id:
        if 'last_auto_refresh' not in st.session_state:
            st.session_state.last_auto_refresh = time.time()
            
        # Get current status to determine refresh frequency
        current_status = "PENDING"
        if "task_status" in st.session_state:
            current_status = st.session_state.task_status.get("status", "PENDING")
            
        # Check if it's time for a refresh
        current_time = time.time()
        
        # Use more aggressive refresh for PENDING status (every 1 second)
        refresh_interval = 1.0 if current_status == "PENDING" else 2.0
        
        if current_time - st.session_state.last_auto_refresh > refresh_interval:
            # Print info about auto refresh
            print(f"Auto-refreshing for status {current_status} (interval: {refresh_interval}s)")
            
            # Update refresh timestamp
            st.session_state.last_auto_refresh = current_time
            
            # Trigger a rerun to check for updates
            st.rerun()

# App Header
st.title("📝 Meeting Notes")

# Check backend connectivity and realtime status in the same area
backend_col, realtime_col = st.columns(2)

with backend_col:
    with st.spinner("Connecting to backend..."):
        api_connected = check_api_connection()
        
        if not api_connected:
            st.error("⚠️ Cannot connect to backend")
            st.info("Run `python run.py` in terminal")
            st.stop()
        
        st.success("✅ Backend connected")

# Show real-time status
with realtime_col:
    # Check real-time status if we have an active task
    if st.session_state.current_task_id:
        if is_realtime_connected():
            st.success("✅ Real-time updates active")
        else:
            st.warning("⚠️ Connection lost")
            # Try to reconnect to ZMQ
            if hasattr(st.session_state, 'zmq_connected') and st.session_state.zmq_connected == False:
                if st.session_state.current_task_id:
                    st.info("Reconnecting...")
                    initialize_zmq_subscriber(st.session_state.current_task_id)
    else:
        st.info("Ready for transcription")

# Main content
if not st.session_state.current_task_id:
    # No active task - Show upload form
    
    # If we don't have any models, try to fetch them again
    if not st.session_state.available_models:
        fetch_available_models()
    
    # Model and language selection in columns
    col1, col2 = st.columns(2)
    
    # Model selection first
    with col1:
        # Use format_model_info to display model details
        if st.session_state.available_models:
            model_options = st.session_state.available_models
            model_labels = {model: format_model_info(model) for model in model_options}
            selected_model = st.selectbox(
                "Model Size (smaller = faster)",
                options=model_options,
                format_func=lambda x: model_labels[x],
                index=0 if "tiny" in model_options else len(model_options)-1  # Default to tiny if available, otherwise last model
            )
        else:
            selected_model = "tiny"  # Default to tiny if no models are fetched
            st.warning("Could not fetch available models. Using default model.")
    
    # Language selection
    with col2:
        languages = st.session_state.get("languages", {
            "tr": "Turkish",
            "en": "English",
            "de": "German",
            "fr": "French",
            "es": "Spanish",
            "it": "Italian"
        })
        
        selected_language = st.selectbox(
            "Language", 
            options=list(languages.keys()), 
            format_func=lambda x: languages.get(x)
        )
    
    # Advanced options
    with st.expander("Advanced Options"):
        initial_prompt = st.text_area(
            "Initial Prompt (helps guide transcription)",
            value="Bu bir Türkçe ses kaydıdır. Lütfen Türkçe konuşmayı doğru şekilde yazıya dökün." if selected_language == "tr" else "",
            help="Provide context to help Whisper understand the audio better"
        )
    
    # File uploader for audio files - with automatic transcription
    st.info("Transcription will start automatically when you select a file")
    audio_file = st.file_uploader("Upload audio file (WAV, MP3, etc.)", type=["wav", "mp3", "m4a", "ogg"])
    
    # Show model performance note
    if selected_model:
        model_info = st.session_state.whisper_model_sizes.get(selected_model, {})
        st.caption(f"**Model Performance**: Size: {model_info.get('size', 'Unknown')}, "
                  f"Speed: {model_info.get('speed', 'Unknown')}, "
                  f"Accuracy: {model_info.get('accuracy', 'Unknown')}")
    
    # Auto-start transcription when file is uploaded
    if audio_file:
        with st.spinner(f"Starting transcription with {selected_model} model..."):
            task_id = submit_audio_for_transcription(
                audio_file,
                language=selected_language,
                initial_prompt=initial_prompt if initial_prompt else None,
                model=selected_model
            )
            
            if task_id:
                st.session_state.current_task_id = task_id
                # Reset status for new task
                st.session_state.detailed_status = f"Task started using {selected_model} model, connecting to real-time updates..."
                # Connect to ZeroMQ for real-time updates
                if initialize_zmq_subscriber(task_id):
                    st.success(f"Transcription started! Task ID: {task_id}")
                    
                    # For very fast models, immediately check status to avoid UI lag
                    if selected_model in ["tiny", "base"]:
                        time.sleep(0.5)  # Give the task a moment to start/complete
                        api_status = check_task_status(task_id)
                        if api_status and api_status.get("status") == "SUCCESS":
                            # Task already completed! Update UI
                            st.session_state.progress_value = 1.0
                            update_detailed_status("SUCCESS", "Task completed very quickly!")
                            
                            # Get result if available
                            if "result" in api_status:
                                result = api_status["result"]
                                if "transcription" in result and "text" in result["transcription"]:
                                    st.session_state.transcription_text = result["transcription"]["text"]
                                    
                                    # Update task status object
                                    st.session_state.task_status = {
                                        "task_id": task_id,
                                        "status": "SUCCESS",
                                        "data": {
                                            "data": {"result": result}
                                        }
                                    }
                else:
                    st.warning("Connected to API but real-time updates disabled. Refresh for updates.")
                st.rerun()
            else:
                st.error("Failed to start transcription. Please try again.")
else:
    # Active task - Show results
    
    # Call the polling function to ensure we get updates
    status_updated = poll_task_status()
    
    # Auto-refresh for live updates
    auto_refresh()
    
    # Display task ID and status
    task_id = st.session_state.current_task_id 
    task_status = st.session_state.task_status
    status = task_status.get("status", "PENDING")
    
    # Display task ID and status
    task_id = st.session_state.current_task_id 
    task_status = st.session_state.task_status
    status = task_status.get("status", "PENDING")
    
    # Show detailed status in a collapsible component
    status_box = st.container()
    with status_box:
        # Show the current status in an info box
        st.info(st.session_state.detailed_status)
        
        # Add a collapsible section for detailed logs
        with st.expander("View Detailed Log History", expanded=False):
            # Create a container for the status log
            log_container = st.empty()
            
            # Format all logs as a single string with appropriate styling
            log_html = ""
            for log_entry in st.session_state.status_log:
                if "completed successfully" in log_entry:
                    log_html += f"<div style='color:green; margin-bottom:4px;'>{log_entry}</div>"
                elif "failed" in log_entry:
                    log_html += f"<div style='color:red; margin-bottom:4px;'>{log_entry}</div>"
                elif "started" in log_entry or "progress" in log_entry or "Processing" in log_entry:
                    log_html += f"<div style='color:blue; margin-bottom:4px;'>{log_entry}</div>"
                else:
                    log_html += f"<div style='margin-bottom:4px;'>{log_entry}</div>"
            
            # Display all logs in a scrollable container
            if log_html:
                log_container.markdown(f"""
                <div style="max-height: 200px; overflow-y: auto; border: 1px solid #f0f0f0; padding: 10px; border-radius: 4px; background-color: #f9f9f9;">
                    {log_html}
                </div>
                """, unsafe_allow_html=True)
            else:
                log_container.info("No log entries available yet.")
    
    # Progress indicator for pending/processing tasks
    if status in ["PENDING", "STARTED", "PROCESSING"]:
        # Use progress value if available, otherwise animate
        if st.session_state.progress_value > 0:
            progress_color = "orange"
            st.markdown(
                f"""
                <div class="stProgress">
                    <div style="width:{st.session_state.progress_value*100}%;
                                height:20px;
                                background-color:{progress_color};
                                border-radius:5px;">
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Increment progress value for animation effect (loops between 0 and 1)
            st.session_state.progress_value = (st.session_state.progress_value + 0.01) % 1.0
            st.progress(st.session_state.progress_value)
    else:
        # For completed or failed tasks, show final progress
        progress_value = 1.0 if status == "SUCCESS" else st.session_state.progress_value
        progress_color = "green" if status == "SUCCESS" else "red" if status == "FAILURE" else "orange"
        
        st.markdown(
            f"""
            <div class="stProgress">
                <div style="width:{progress_value*100}%;
                            height:20px;
                            background-color:{progress_color};
                            border-radius:5px;">
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Task details in a more compact format
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.markdown(f"**ID**: {task_id[-8:]}")
    with info_col2:
        status_color = "green" if status == "SUCCESS" else "red" if status == "FAILURE" else "orange"
        st.markdown(f"**Status**: <span style='color:{status_color}'>{status}</span>", unsafe_allow_html=True)
    with info_col3:
        if status == "SUCCESS" and "result" in task_status.get("data", {}).get("data", {}):
            result = task_status["data"]["data"]["result"]
            if "transcription" in result and "processing_time_seconds" in result["transcription"]:
                st.markdown(f"**Processing Time**: {format_duration(result['transcription']['processing_time_seconds'])}")
    
    # Display transcription if available
    if st.session_state.transcription_text:
        # Show the transcription
        transcription_text = st.session_state.transcription_text.strip()
        
        if transcription_text:
            st.subheader("Transcription")
            st.text_area("", 
                        value=transcription_text, 
                        height=300, 
                        key="transcription_display",
                        help="The transcribed text from your audio file")
            
            # Add copy button for transcription
            if st.button("📋 Copy Transcription", key="copy_transcription"):
                st.code(f"echo '{transcription_text}' | clip")  # Show command for copying
                st.success("Text copied to clipboard!")
            
            # Option to request summary
            if st.button("Generate Summary") and GROQ_API_KEY:
                with st.spinner("Generating summary..."):
                    # Submit for summarization
                    summary_task_id = request_summary(transcription_text)
                    if summary_task_id:
                        # Store summary task ID and reset summary text
                        st.session_state.current_summary_task_id = summary_task_id
                        st.session_state.summary_text = ""
                        st.success("Summary requested. Processing...")
                        
                        # Check status immediately
                        time.sleep(1)
                        summary_status = check_task_status(summary_task_id)
                        if summary_status and summary_status.get("status") == "SUCCESS":
                            # Summary already completed!
                            if "result" in summary_status:
                                result = summary_status["result"]
                                if "summary" in result and "summary" in result["summary"]:
                                    st.session_state.summary_text = result["summary"]["summary"]
                        
                        st.rerun()
            elif not GROQ_API_KEY:
                st.warning("Summary generation requires a Groq API key. Please add it to your .env file.")
    
    # Display summary if available
    if st.session_state.summary_text:
        st.subheader("Summary")
        st.text_area("", 
                    value=st.session_state.summary_text, 
                    height=200, 
                    key="summary_display",
                    help="AI-generated summary of the transcription")
        
        # Add copy button for summary
        if st.button("📋 Copy Summary", key="copy_summary"):
            st.code(f"echo '{st.session_state.summary_text}' | clip")  # Show command for copying
            st.success("Summary copied to clipboard!")
    
    # Option to start a new transcription
    if st.button("Start New Transcription", key="new_transcription"):
        clear_current_task()
        st.rerun()

# On app exit, clean up resources
def on_app_exit():
    cleanup_zmq_resources()

# Register the cleanup function to run on app exit
import atexit
atexit.register(on_app_exit) 