import redis
import os
import docker
from dotenv import load_dotenv
import time
import socket
import sys

# Load environment variables
load_dotenv()

# Get Redis configuration from environment variables or use defaults
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_CONTAINER_NAME = os.getenv("REDIS_CONTAINER_NAME", "redis-server")

def is_redis_running():
    """Check if Redis is already running at the specified host and port"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect((REDIS_HOST, REDIS_PORT))
        s.close()
        return True
    except (socket.error, ConnectionRefusedError):
        return False

def start_redis_server():
    # First, check if Redis is already running somewhere
    if is_redis_running():
        print(f"Redis server is already running at {REDIS_HOST}:{REDIS_PORT}")
        try:
            # Create Redis connection
            redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True
            )
            
            # Test connection
            redis_client.ping()
            print(f"Connected to existing Redis server at {REDIS_HOST}:{REDIS_PORT}")
            return redis_client
        except redis.ConnectionError as e:
            print(f"Error connecting to Redis: {e}")
            return None
        
    # If Redis is not running, try to start it
    print(f"No Redis server found at {REDIS_HOST}:{REDIS_PORT}. Attempting to start one with Docker...")
    
    try:
        # Initialize Docker client
        docker_client = docker.from_env()
        
        # Check if container already exists
        try:
            container = docker_client.containers.get(REDIS_CONTAINER_NAME)
            
            # If container exists but not running, start it
            if container.status != "running":
                print(f"Starting existing Redis container: {REDIS_CONTAINER_NAME}")
                container.start()
            else:
                print(f"Redis container '{REDIS_CONTAINER_NAME}' is already running")
                
        except docker.errors.NotFound:
            # Container doesn't exist, create and start a new one
            print(f"Creating new Redis container: {REDIS_CONTAINER_NAME}")
            container = docker_client.containers.run(
                "redis:latest",
                name=REDIS_CONTAINER_NAME,
                ports={6379: REDIS_PORT},
                detach=True
            )
        
        # Wait for Redis to initialize
        print("Waiting for Redis server to initialize...")
        time.sleep(2)
        
        # Try to connect again
        for attempt in range(5):  # Try a few times
            try:
                redis_client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    decode_responses=True
                )
                redis_client.ping()
                print(f"Redis server started and connected successfully at {REDIS_HOST}:{REDIS_PORT}")
                return redis_client
            except redis.ConnectionError:
                print(f"Redis not ready yet, waiting... (attempt {attempt+1}/5)")
                time.sleep(1)
        
        print("Failed to connect to Redis server after multiple attempts")
        return None
    
    except docker.errors.DockerException as e:
        print(f"Docker error: {e}")
        print("Docker might not be available. Falling back to a direct connection attempt...")
        
        # If Docker fails (e.g., when running inside a container without Docker socket)
        # Just try to connect to Redis directly in case it's available elsewhere
        try:
            redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True
            )
            redis_client.ping()
            print(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            return redis_client
        except redis.ConnectionError as e:
            print(f"Could not connect to Redis: {e}")
            print("Please ensure Redis is running and accessible, or modify REDIS_HOST in environment variables.")
            return None

if __name__ == "__main__":
    redis_client = start_redis_server()
    if redis_client is None:
        print("Exiting due to Redis connection failure")
        sys.exit(1)
