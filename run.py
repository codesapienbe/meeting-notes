#!/usr/bin/env python3
import os
import subprocess
import docker
import time
import threading
import signal
import sys
import socket
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
REDIS_CONTAINER_NAME = "redis-server"
REDIS_PORT = 6379
REDIS_IMAGE = "redis:latest"

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

def start_celery_worker():
    """Start Celery worker from tasks.py"""
    print("Starting Celery worker...")
    env = os.environ.copy()
    env["REDIS_URL"] = f"redis://localhost:{REDIS_PORT}/0"
    env["REDIS_HOST"] = "localhost"
    env["REDIS_PORT"] = str(REDIS_PORT)
    
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

def main():
    # Start Redis container
    redis_container = start_redis()
    
    # Wait for Redis to be fully up
    time.sleep(3)
    
    # Start Celery worker and FastAPI app
    celery_process = start_celery_worker()
    fastapi_process = start_fastapi()
    
    processes = [celery_process, fastapi_process]
    
    # Setup log threads
    celery_log_thread = threading.Thread(
        target=log_output, 
        args=(celery_process, "CELERY"),
        daemon=True
    )
    
    fastapi_log_thread = threading.Thread(
        target=log_output, 
        args=(fastapi_process, "API"),
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