from restapi import celery_app
import uuid
import os
import time
import socket
import zmq
import threading
import json

# ZeroMQ configuration
ZMQ_PUB_PORT = 5556  # Port for the ZMQ publisher
zmq_context = zmq.Context()
zmq_publisher = None
zmq_lock = threading.Lock()
zmq_heartbeat_event = threading.Event()
zmq_heartbeat_thread = None
zmq_fallback_mode = False  # Flag to indicate if we're using fallback mode

def initialize_zmq_publisher():
    """Initialize the ZeroMQ publisher socket"""
    global zmq_publisher, zmq_heartbeat_thread, zmq_fallback_mode
    
    # If we already have a publisher or we're in fallback mode, return the current state
    if zmq_publisher is not None or zmq_fallback_mode:
        return zmq_publisher
        
    with zmq_lock:
        # Double-check after acquiring lock
        if zmq_publisher is not None:
            return zmq_publisher
            
        print(f"Initializing ZeroMQ publisher on port {ZMQ_PUB_PORT}")
        try:
            # Check if port is already in use by another process
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 100)  # Set receive timeout to 100ms
            
            # Try to connect to the port first to check if it's in use
            try:
                socket.connect(f"tcp://localhost:{ZMQ_PUB_PORT}")
                socket.send_string("ping")
                socket.recv_string()  # This will time out if no one is listening
                # If we reach here, someone is listening and port is in use
                print(f"Port {ZMQ_PUB_PORT} is already in use by another process")
                
                # Instead of failing, we'll try to use the existing publisher
                # by sending messages through Redis instead
                print("Switching to Redis-based fallback for message delivery")
                zmq_fallback_mode = True
                socket.close()
                context.term()
                return None
            except zmq.error.Again:
                # No one responded - port is probably free
                pass
            except Exception as e:
                # Some other error occurred during the check
                print(f"Error checking if port is in use: {e}")
            finally:
                socket.close()
                context.term()
            
            # Create publisher socket
            zmq_publisher = zmq_context.socket(zmq.PUB)
            # Set socket options for reliability
            zmq_publisher.setsockopt(zmq.LINGER, 500)  # Wait 500ms on close
            zmq_publisher.setsockopt(zmq.SNDHWM, 1000)  # High water mark
            
            # Try to bind with a timeout
            try:
                # Set the socket to non-blocking for bind
                zmq_publisher.setsockopt(zmq.IMMEDIATE, 1)
                zmq_publisher.bind(f"tcp://*:{ZMQ_PUB_PORT}")
                # Give the socket time to bind
                time.sleep(0.2)
                zmq_fallback_mode = False  # We successfully bound, no fallback needed
            except zmq.error.ZMQError as e:
                print(f"Could not bind to port {ZMQ_PUB_PORT}: {e}")
                zmq_publisher.close()
                zmq_publisher = None
                
                # Use fallback mode with Redis
                print("Switching to Redis-based fallback for message delivery")
                zmq_fallback_mode = True
                return None
            
            # Start heartbeat thread if not already running
            if not zmq_fallback_mode and (zmq_heartbeat_thread is None or not zmq_heartbeat_thread.is_alive()):
                zmq_heartbeat_event.clear()
                zmq_heartbeat_thread = threading.Thread(
                    target=zmq_heartbeat_loop, 
                    daemon=True
                )
                zmq_heartbeat_thread.start()
                
        except Exception as e:
            print(f"Error initializing ZeroMQ publisher: {e}")
            if zmq_publisher:
                zmq_publisher.close()
                zmq_publisher = None
            zmq_fallback_mode = True
            
    return zmq_publisher

def zmq_heartbeat_loop():
    """Send periodic heartbeats to keep connections alive"""
    print("Starting ZeroMQ heartbeat thread")
    while not zmq_heartbeat_event.is_set():
        try:
            # Send a heartbeat message
            publish_task_update("heartbeat", "HEARTBEAT", {"timestamp": time.time()})
        except Exception as e:
            print(f"Error in ZMQ heartbeat: {e}")
        
        # Sleep for 5 seconds
        for _ in range(50):  # Check for stop event every 100ms
            if zmq_heartbeat_event.is_set():
                break
            time.sleep(0.1)
    
    print("ZeroMQ heartbeat thread stopping")

def cleanup_zmq_publisher():
    """Clean up ZeroMQ resources"""
    global zmq_publisher, zmq_heartbeat_thread
    
    # Stop heartbeat thread
    zmq_heartbeat_event.set()
    if zmq_heartbeat_thread and zmq_heartbeat_thread.is_alive():
        zmq_heartbeat_thread.join(timeout=1.0)
    
    # Close publisher
    with zmq_lock:
        if zmq_publisher:
            try:
                zmq_publisher.close()
            except Exception as e:
                print(f"Error closing ZMQ publisher: {e}")
            finally:
                zmq_publisher = None
    
    print("ZeroMQ publisher cleaned up")

def publish_task_update(task_id, status, data=None):
    """Publish a task update via ZeroMQ or fallback mechanism"""
    global zmq_fallback_mode
    
    # Create the message data structure
    message = {
        "task_id": task_id,
        "status": status,
        "timestamp": time.time(),
        "data": data or {}
    }
    
    # First try to use ZeroMQ if available
    success = False
    if not zmq_fallback_mode:
        try:
            publisher = initialize_zmq_publisher()
            if publisher:
                # Topic is the task_id for targeted subscriptions
                topic = f"task.{task_id}"
                
                # Send message as JSON
                serialized = json.dumps(message)
                
                with zmq_lock:  # Use lock for thread safety
                    publisher.send_multipart([topic.encode('utf-8'), serialized.encode('utf-8')])
                
                success = True
        except Exception as e:
            print(f"Error publishing ZeroMQ update: {e}")
            # Try to reinitialize the publisher
            with zmq_lock:
                global zmq_publisher
                if zmq_publisher:
                    try:
                        zmq_publisher.close()
                    except:
                        pass
                    zmq_publisher = None
            zmq_fallback_mode = True
    
    # If ZeroMQ failed or not available, use Redis as fallback
    if not success:
        try:
            # Import the Redis client only when needed to avoid circular imports
            from redis import Redis
            from celery.exceptions import CeleryError
            
            # Get Redis connection info from Celery
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
            
            # Use Redis pub/sub as fallback
            channel = f"zmq.fallback.{task_id}"
            redis_client.publish(channel, json.dumps(message))
            
            # Also store the message in a Redis key for reliable retrieval
            # Critical fix: Use a consistent format for status keys - SAME format must be used in webapp.py
            status_key = f"task_status:{task_id}"
            redis_client.set(status_key, json.dumps(message))
            print(f"Saved task status to Redis key: {status_key}")
            
            # Also store by status type for easier filtering
            status_type_key = f"task_status:{task_id}:{status}"
            redis_client.set(status_type_key, json.dumps(message))
            print(f"Saved status type to Redis key: {status_type_key}")
            
            # Set expiry on all Redis keys (24 hours)
            redis_client.expire(status_key, 86400)
            redis_client.expire(status_type_key, 86400)
            
            success = True
        except (ImportError, CeleryError) as e:
            print(f"Fallback Redis publish also failed: {e}")
            # At this point we can't do much but acknowledge the failure
        except Exception as e:
            print(f"Unexpected error in Redis fallback: {e}")
    
    # Log the update (except for heartbeats)
    if task_id != "heartbeat" or status != "HEARTBEAT":
        print(f"Published update for task {task_id}: {status}")
        if success:
            print(f"Published ZeroMQ update for task {task_id}: {status}")
        else:
            print(f"WARNING: Failed to publish update for task {task_id}: {status}")
            
    return success

def ensure_no_existing_default_workers():
    """Try to help the worker discover and handle any existing workers"""
    try:
        # Create a quick/throwaway Celery inspector to discover existing nodes
        from celery.app.control import Inspect
        inspector = Inspect(app=celery_app)
        
        # Allow a short time for discovery
        time.sleep(1)
        
        # Check for active nodes
        active_nodes = inspector.active()
        if active_nodes:
            node_names = list(active_nodes.keys())
            print(f"Found existing Celery nodes: {node_names}")
        
        # No action needed here - we just want to trigger discovery
        return True
    except Exception as e:
        print(f"Error checking for existing nodes: {e}")
        return False

# Make the publisher available to other modules
def get_zmq_publisher():
    return initialize_zmq_publisher()

if __name__ == "__main__":
    initialize_zmq_publisher()
    
    worker_count = int(os.environ.get("WORKER_COUNT", 1))
    hostname = socket.gethostname().replace('.', '-')
    timestamp = int(time.time())
    unique_id = f"{uuid.uuid4().hex[:6]}-{timestamp}"
    worker_name = f"worker-{unique_id}@{hostname}"
    
    ensure_no_existing_default_workers()
    
    command = [
        "worker",
        f"--hostname={worker_name}",
        f"--concurrency={worker_count}",
        "--loglevel=info",
        "-Q", "transcription,summarization",
        "--purge",
    ]
    
    print(f"Starting Celery worker with {worker_count} concurrent processes")
    print(f"Worker unique name: {worker_name}")
    
    try:
        celery_app.worker_main(command)
    finally:
        # Clean up ZMQ resources when worker exits
        cleanup_zmq_publisher() 