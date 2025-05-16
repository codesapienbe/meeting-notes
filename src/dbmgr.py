import sqlite3
import json
import os
import datetime
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

class TaskDatabaseManager:
    """SQLite database manager for tasks and their responses"""
    
    def __init__(self, db_path="data/tasks.db"):
        """Initialize database connection and create tables if needed"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Connect to the SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON")
        # Return rows as dictionaries
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        # Task table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            options TEXT
        )
        ''')
        
        # Check if the responses table exists at all
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='responses'")
        responses_table_exists = self.cursor.fetchone() is not None
        
        if not responses_table_exists:
            # Responses table doesn't exist, create it with the embeddings column
            print("Creating responses table with embeddings column")
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                response_data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                embeddings TEXT,
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            )
            ''')
        else:
            # Check if the embeddings column exists in the responses table
            self.cursor.execute("PRAGMA table_info(responses)")
            columns = self.cursor.fetchall()
            column_names = [column['name'] for column in columns]
            
            # Debug print to verify column names
            print(f"Existing columns in responses table: {column_names}")
            
            if 'embeddings' not in column_names:
                # Response table exists but needs embeddings column added
                print("Adding embeddings column to responses table")
                self.cursor.execute('ALTER TABLE responses ADD COLUMN embeddings TEXT')
            else:
                print("Embeddings column already exists in responses table")
        
        self.conn.commit()
    
    def save_task(self, task_id: str, task_type: str, status: str, options: Dict[str, Any] = None) -> bool:
        """
        Save a new task to the database
        
        Args:
            task_id: Unique identifier for the task
            task_type: Type of task (transcribe, summarize, etc.)
            status: Current status of the task (PENDING, SUCCESS, FAILURE)
            options: Optional dictionary of task options
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            now = datetime.datetime.now().isoformat()
            
            self.cursor.execute(
                "INSERT INTO tasks (id, type, status, created_at, updated_at, options) VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, task_type, status, now, now, json.dumps(options) if options else None)
            )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error saving task: {e}")
            return False
    
    def update_task_status(self, task_id: str, status: str) -> bool:
        """
        Update the status of an existing task
        
        Args:
            task_id: ID of the task to update
            status: New status value
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            now = datetime.datetime.now().isoformat()
            
            self.cursor.execute(
                "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, task_id)
            )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error updating task status: {e}")
            return False
    
    def generate_embeddings(self, text: str) -> Optional[List[float]]:
        """
        Generate vector embeddings for text
        
        Args:
            text: Text to generate embeddings for
            
        Returns:
            List of float values representing the embedding or None if failed
        """
        try:
            import os
            import torch
            from sentence_transformers import SentenceTransformer
            
            # Force CPU usage to avoid MPS/GPU-related crashes
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # Set PyTorch to use CPU explicitly
            device = torch.device("cpu")
            
            # Load model with explicit CPU device
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
            
            # Use smaller chunks if text is very long
            max_seq_length = 256
            if len(text) > max_seq_length * 10:
                # Process in smaller chunks for long text
                chunks = [text[i:i+max_seq_length*5] for i in range(0, len(text), max_seq_length*5)]
                embeddings = model.encode(chunks, convert_to_numpy=True)
                # Average the embeddings of all chunks
                embedding = np.mean(embeddings, axis=0)
            else:
                embedding = model.encode(text, convert_to_numpy=True)
            
            return embedding.tolist()
        except ImportError as e:
            print(f"Missing dependency for embeddings: {e}")
            print("Skipping embeddings generation. Install required packages with: pip install sentence-transformers")
            return None
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            print("Continuing without embeddings")
            return None
    
    def save_task_response(self, task_id: str, response_data: Dict[str, Any]) -> bool:
        """
        Save a task response to the database
        
        Args:
            task_id: ID of the task this response belongs to
            response_data: Dictionary containing response data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            now = datetime.datetime.now().isoformat()
            
            # Update task status to SUCCESS
            self.update_task_status(task_id, "SUCCESS")
            
            # Check if we should generate embeddings
            generate_embeddings = os.environ.get("GENERATE_EMBEDDINGS", "false").lower() == "true"
            
            # Generate embeddings for transcription text if available and enabled
            embeddings = None
            if generate_embeddings and 'transcription' in response_data and 'text' in response_data['transcription']:
                text = response_data['transcription']['text']
                if text.strip():  # Only generate for non-empty text
                    embeddings = self.generate_embeddings(text)
            
            # Save response data with embeddings
            if embeddings:
                self.cursor.execute(
                    "INSERT INTO responses (task_id, response_data, created_at, embeddings) VALUES (?, ?, ?, ?)",
                    (task_id, json.dumps(response_data), now, json.dumps(embeddings))
                )
            else:
                self.cursor.execute(
                    "INSERT INTO responses (task_id, response_data, created_at) VALUES (?, ?, ?)",
                    (task_id, json.dumps(response_data), now)
                )
                
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error saving task response: {e}")
            return False
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task details by ID
        
        Args:
            task_id: ID of the task to retrieve
            
        Returns:
            Dict containing task details or None if not found
        """
        try:
            self.cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            task = self.cursor.fetchone()
            
            if not task:
                return None
            
            # Convert Row to dict
            task_dict = dict(task)
            
            # Parse options JSON
            if task_dict['options']:
                task_dict['options'] = json.loads(task_dict['options'])
            
            return task_dict
        except Exception as e:
            print(f"Error retrieving task: {e}")
            return None
    
    def get_task_response(self, task_id: str, include_embeddings: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get the latest response for a task
        
        Args:
            task_id: ID of the task
            include_embeddings: Whether to include embeddings in the response
            
        Returns:
            Dict containing response data or None if not found
        """
        try:
            self.cursor.execute(
                "SELECT * FROM responses WHERE task_id = ? ORDER BY created_at DESC LIMIT 1", 
                (task_id,)
            )
            response = self.cursor.fetchone()
            
            if not response:
                return None
            
            # Convert Row to dict
            response_dict = dict(response)
            
            # Parse response_data JSON
            if response_dict['response_data']:
                response_dict['response_data'] = json.loads(response_dict['response_data'])
            
            # Parse embeddings JSON if available and requested
            if include_embeddings and response_dict['embeddings']:
                response_dict['embeddings'] = json.loads(response_dict['embeddings'])
            elif not include_embeddings:
                # Remove embeddings if not requested to reduce data transfer
                response_dict.pop('embeddings', None)
            
            return response_dict
        except Exception as e:
            print(f"Error retrieving task response: {e}")
            return None
    
    def get_task_with_response(self, task_id: str, include_embeddings: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get task with its latest response
        
        Args:
            task_id: ID of the task to retrieve
            include_embeddings: Whether to include embeddings in the response
            
        Returns:
            Dict containing combined task and response data or None if not found
        """
        task = self.get_task(task_id)
        if not task:
            return None
        
        response = self.get_task_response(task_id, include_embeddings)
        if response:
            task['result'] = response['response_data']
            if include_embeddings and 'embeddings' in response:
                task['embeddings'] = response['embeddings']
        
        return task
    
    def list_tasks(self, limit: int = 100, offset: int = 0, include_embeddings: bool = False) -> List[Dict[str, Any]]:
        """
        List tasks with pagination
        
        Args:
            limit: Maximum number of tasks to return
            offset: Number of tasks to skip
            include_embeddings: Whether to include embeddings in the responses
            
        Returns:
            List of task dictionaries
        """
        try:
            self.cursor.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ? OFFSET ?", 
                (limit, offset)
            )
            tasks = self.cursor.fetchall()
            
            # Convert Row objects to dicts
            task_list = []
            for task in tasks:
                task_dict = dict(task)
                if task_dict['options']:
                    task_dict['options'] = json.loads(task_dict['options'])
                
                # Get response for this task if available
                response = self.get_task_response(task_dict['id'], include_embeddings)
                if response:
                    task_dict['result'] = response['response_data']
                    if include_embeddings and 'embeddings' in response:
                        task_dict['embeddings'] = response['embeddings']
                
                task_list.append(task_dict)
            
            return task_list
        except Exception as e:
            print(f"Error listing tasks: {e}")
            return []
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        self.close() 