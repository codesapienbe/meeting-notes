from restapi import celery_app
import uuid
import os
import time
import socket

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

if __name__ == "__main__":
    # Get worker count from environment, default to 1
    worker_count = int(os.environ.get("WORKER_COUNT", 1))
    
    # Generate a truly unique ID with hostname and timestamp
    hostname = socket.gethostname().replace('.', '-')
    timestamp = int(time.time())
    unique_id = f"{uuid.uuid4().hex[:6]}-{timestamp}"
    
    # Ensure our worker name is completely unique and doesn't resemble the default
    worker_name = f"worker-{unique_id}@{hostname}"
    
    # Pre-check for any existing nodes with default names
    ensure_no_existing_default_workers()
    
    # Create command with appropriate concurrency
    command = [
        "worker",
        f"--hostname={worker_name}",
        f"--concurrency={worker_count}",
        "--loglevel=info",
        "-Q", "transcription,summarization",
        "--purge",  # Purge all tasks at startup
    ]
    
    print(f"Starting Celery worker with {worker_count} concurrent processes")
    print(f"Worker unique name: {worker_name}")
    celery_app.worker_main(command) 