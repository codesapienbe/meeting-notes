import sqlite3
import json
import os
import datetime
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
        
        # Response table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            response_data TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (task_id) REFERENCES tasks(id)
        )
        ''')
        
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
            
            # Save response data
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
    
    def get_task_response(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest response for a task
        
        Args:
            task_id: ID of the task
            
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
            
            return response_dict
        except Exception as e:
            print(f"Error retrieving task response: {e}")
            return None
    
    def get_task_with_response(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task with its latest response
        
        Args:
            task_id: ID of the task to retrieve
            
        Returns:
            Dict containing combined task and response data or None if not found
        """
        task = self.get_task(task_id)
        if not task:
            return None
        
        response = self.get_task_response(task_id)
        if response:
            task['result'] = response['response_data']
        
        return task
    
    def list_tasks(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List tasks with pagination
        
        Args:
            limit: Maximum number of tasks to return
            offset: Number of tasks to skip
            
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
                task_list.append(task_dict)
            
            return task_list
        except Exception as e:
            print(f"Error listing tasks: {e}")
            return []
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
    
    def __del__(self):
        """Ensure connection is closed when object is destroyed"""
        self.close() 