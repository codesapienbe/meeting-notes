#!/usr/bin/env python3
import os
import subprocess
import docker
import time
import threading
import signal
import sys
import socket
import argparse
import psutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
REDIS_CONTAINER_NAME = "redis-server"
REDIS_PORT = 6379
REDIS_IMAGE = "redis:latest"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run the Whisper transcription application")
    parser.add_argument("-w", "--workers", type=int, default=1, help="Number of Celery worker processes (default: 1)")
    parser.add_argument("--kill-existing", action="store_true", default=True, help="Kill existing Celery processes before starting")
    parser.add_argument("--reset-redis", action="store_true", help="Completely reset Redis database before starting")
    return parser.parse_args()

def is_port_in_use(port):
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port=6379, max_attempts=10):
    """Find an available port starting from start_port"""
    port = start_port
    for _ in range(max_attempts):
        if not is_port_in_use(port):
            return port
        port += 1
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

def check_redis_connection(host='localhost', port=6379):
    """Check if Redis is already running and accessible"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            if s.connect_ex((host, port)) == 0:
                return True
    except:
        pass
    return False

def kill_existing_celery_processes():
    """Kill any existing Celery worker processes"""
    killed = 0
    
    # Method 1: Use psutil to find and kill processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            # Look for both 'celery' and 'tasks.py' to catch our specific processes
            if cmdline and (
                (any('celery' in cmd.lower() for cmd in cmdline) and 'worker' in cmdline) or
                (any('tasks.py' in cmd for cmd in cmdline))
            ):
                print(f"Killing existing Celery worker process: {proc.pid}")
                proc.kill()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    # Method 2: Use system commands as backup
    try:
        if sys.platform == 'darwin' or sys.platform.startswith('linux'):
            # On macOS/Linux, use pkill
            subprocess.run(["pkill", "-f", "celery worker"], stderr=subprocess.DEVNULL)
            subprocess.run(["pkill", "-f", "tasks.py"], stderr=subprocess.DEVNULL)
        elif sys.platform == 'win32':
            # On Windows, use taskkill
            subprocess.run(["taskkill", "/f", "/im", "celery.exe"], stderr=subprocess.DEVNULL)
            subprocess.run(["taskkill", "/f", "/fi", "IMAGENAME eq python.exe", "/fi", "WINDOWTITLE eq *celery*"], stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Warning: System command process killing failed: {e}")
    
    if killed > 0:
        print(f"Killed {killed} existing Celery worker processes")
        # Give processes time to fully terminate
        time.sleep(2)
    else:
        print("No existing Celery worker processes found")
    
    return killed

def start_redis():
    """Start Redis container using Docker Python SDK"""
    global REDIS_PORT  # Allow modifying the global port variable
    
    print("Checking Redis availability...")
    
    # If Redis is already running on the default port, use it
    if check_redis_connection(port=REDIS_PORT):
        print(f"Redis is already running on port {REDIS_PORT}, using existing instance")
        return None  # Return None to indicate we're using existing Redis
    
    # Find an available port if the default is in use
    if is_port_in_use(REDIS_PORT):
        old_port = REDIS_PORT
        REDIS_PORT = find_available_port(REDIS_PORT)
        print(f"Port {old_port} is already in use, using port {REDIS_PORT} instead")
    
    print("Starting Redis container...")
    client = docker.from_env()
    
    # Check if container already exists
    try:
        container = client.containers.get(REDIS_CONTAINER_NAME)
        if container.status != "running":
            print(f"Container {REDIS_CONTAINER_NAME} exists but not running. Starting it...")
            # Remove container if it exists but with wrong port mapping
            try:
                container.remove(force=True)
                print(f"Removed existing container to create one with correct port")
                raise docker.errors.NotFound("Forced container recreation")
            except Exception as e:
                print(f"Error removing container: {str(e)}")
        else:
            print(f"Container {REDIS_CONTAINER_NAME} is already running")
            return container
    except docker.errors.NotFound:
        pass
    
    # Create and start a new container
    try:
        container = client.containers.run(
            REDIS_IMAGE,
            name=REDIS_CONTAINER_NAME,
            ports={'6379/tcp': REDIS_PORT},
            detach=True,
            restart_policy={"Name": "unless-stopped"},
            volumes={
                'redis-data': {'bind': '/data', 'mode': 'rw'}
            }
        )
        
        print(f"Redis container started with ID: {container.id}")
        return container
    except docker.errors.APIError as e:
        if "address already in use" in str(e):
            print(f"Error: Port {REDIS_PORT} is already in use.")
            print("You may have another Redis instance or container already running.")
            print("Options:")
            print("1. Stop the existing Redis service")
            print("2. Use a different port by modifying REDIS_PORT in run.py")
            sys.exit(1)
        else:
            print(f"Docker API error: {str(e)}")
            sys.exit(1)
    except Exception as e:
        print(f"Error starting Redis container: {str(e)}")
        sys.exit(1)

def flush_redis_celery_keys():
    """Flush only Celery-related keys from Redis to clear stuck tasks"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=REDIS_PORT, db=0)
        
        # Clear all Celery-related keys (more comprehensive patterns)
        patterns = ['celery*', 'unacked*', '_kombu*', 'worker*', 'task*']
        total_deleted = 0
        
        for pattern in patterns:
            keys = r.keys(pattern)
            if keys:
                count = r.delete(*keys)
                total_deleted += count
                print(f"Deleted {count} Redis keys matching '{pattern}'")
        
        print(f"Total: Flushed {total_deleted} potentially stale Redis keys")
        return True
    except Exception as e:
        print(f"Failed to flush Redis keys: {e}")
        return False

def start_celery_worker(worker_count=1):
    """Start Celery worker from tasks.py"""
    print(f"Starting Celery worker with {worker_count} worker processes...")
    env = os.environ.copy()
    env["REDIS_URL"] = f"redis://localhost:{REDIS_PORT}/0"
    env["REDIS_HOST"] = "localhost"
    env["REDIS_PORT"] = str(REDIS_PORT)
    env["WORKER_COUNT"] = str(worker_count)
    
    # Ensure GROQ_API_KEY is passed to the worker
    if "GROQ_API_KEY" in os.environ:
        print("Using GROQ_API_KEY from environment")
    else:
        print("Warning: GROQ_API_KEY not found in environment. Summarization will not work.")
    
    return subprocess.Popen(
        ["python", "src/tasks.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

def start_fastapi():
    """Start FastAPI application from restapi.py"""
    print("Starting FastAPI application...")
    env = os.environ.copy()
    env["REDIS_URL"] = f"redis://localhost:{REDIS_PORT}/0"
    env["REDIS_HOST"] = "localhost"
    env["REDIS_PORT"] = str(REDIS_PORT)
    
    # Ensure GROQ_API_KEY is passed to the API
    if "GROQ_API_KEY" in os.environ:
        print("Using GROQ_API_KEY from environment")
    else:
        print("Warning: GROQ_API_KEY not found in environment. Summarization will not work.")
    
    return subprocess.Popen(
        ["uvicorn", "src.restapi:app", "--host", "0.0.0.0", "--port", "8000"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

def log_output(process, prefix):
    """Log the output of a process with a prefix"""
    for line in iter(process.stdout.readline, ''):
        if not line:
            break
        print(f"{prefix}: {line.strip()}")

def cleanup(redis_container, processes):
    """Cleanup function to stop all processes and containers"""
    print("\nShutting down...")
    
    for process in processes:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    
    if redis_container:
        print("Stopping Redis container...")
        try:
            redis_container.stop()
            print("Redis container stopped")
        except Exception as e:
            print(f"Error stopping Redis container: {str(e)}")

def nuke_redis_if_worker_duplicates():
    """Nuclear option: reset Redis completely if we've had persistent issues with duplicates"""
    try:
        redis_keys_file = "data/redis_reset_tracker.txt"
        duplicate_warnings_count = 0
        
        # Check if we have a tracking file
        if os.path.exists(redis_keys_file):
            with open(redis_keys_file, "r") as f:
                try:
                    duplicate_warnings_count = int(f.read().strip())
                except:
                    duplicate_warnings_count = 0
        
        # Increment the count - we only get here if we've seen a DuplicateNodenameWarning
        duplicate_warnings_count += 1
        
        # Store the updated count
        os.makedirs(os.path.dirname(redis_keys_file), exist_ok=True)
        with open(redis_keys_file, "w") as f:
            f.write(str(duplicate_warnings_count))
        
        # If we've seen multiple warnings in a row, take drastic action
        if duplicate_warnings_count >= 2:
            print(f"WARNING: Detected {duplicate_warnings_count} consecutive duplicate warnings.")
            print("Performing FULL Redis flush to resolve persistent node duplicates.")
            import redis
            r = redis.Redis(host='localhost', port=REDIS_PORT, db=0)
            r.flushall()
            print("✓ Redis database COMPLETELY reset. All keys removed.")
            
            # Reset the counter after taking action
            with open(redis_keys_file, "w") as f:
                f.write("0")
            
            return True
    except Exception as e:
        print(f"Error in nuke_redis function: {e}")
    
    return False

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Kill existing Celery processes if requested
    if args.kill_existing:
        kill_existing_celery_processes()
    
    # Start Redis container
    redis_container = start_redis()
    
    # Wait for Redis to be fully up
    time.sleep(3)
    
    # Check if we need to reset Redis completely
    if args.reset_redis:
        print("Performing complete Redis reset as requested...")
        import redis
        r = redis.Redis(host='localhost', port=REDIS_PORT, db=0)
        r.flushall()
        print("✓ Redis database completely reset")
    else:
        # Clean stale Celery tasks
        flush_redis_celery_keys()
    
    # Start Celery worker and FastAPI app
    celery_process = start_celery_worker(args.workers)
    fastapi_process = start_fastapi()
    
    processes = [celery_process, fastapi_process]
    
    # Setup log threads
    celery_log_thread = threading.Thread(
        target=log_output, 
        args=(celery_process, "MEETING-NOTES-WORKER"),
        daemon=True
    )
    
    fastapi_log_thread = threading.Thread(
        target=log_output, 
        args=(fastapi_process, "MEETING-NOTES-API"),
        daemon=True
    )
    
    celery_log_thread.start()
    fastapi_log_thread.start()
    
    # Handle clean shutdown
    def signal_handler(sig, frame):
        cleanup(redis_container, processes)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if celery_process.poll() is not None:
                print("Celery worker process exited with code:", celery_process.returncode)
                break
                
            if fastapi_process.poll() is not None:
                print("FastAPI process exited with code:", fastapi_process.returncode)
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cleanup(redis_container, processes)

if __name__ == "__main__":
    main() 