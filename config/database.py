# config/database.py
import sqlite3
from typing import Optional
from pathlib import Path

def get_sqlite_connection(db_path: str = "recruiter.db") -> sqlite3.Connection:
    """Get SQLite database connection"""
    db_file = Path(db_path)
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
    return conn

def init_sqlite_database(db_path: str = "recruiter.db"):
    """Initialize SQLite database with required tables"""
    conn = get_sqlite_connection(db_path)
    cursor = conn.cursor()
    
    # Create candidates table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            yoe INTEGER,
            current_title TEXT,
            industry TEXT,
            email TEXT,
            phone TEXT,
            status TEXT DEFAULT 'new',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create parsed_resumes table with embeddings support
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parsed_resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id INTEGER,
            name TEXT NOT NULL,
            title TEXT,
            phone TEXT,
            email TEXT,
            skills TEXT,  -- JSON array
            yoe INTEGER,
            relevant_experience TEXT,  -- JSON array
            industry TEXT,
            raw_text TEXT,
            embeddings TEXT,  -- JSON array of floats
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (candidate_id) REFERENCES candidates (id)
        )
    ''')
    
    # Create job_requirements table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS job_requirements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            required_skills TEXT,  -- JSON array
            min_yoe INTEGER,
            max_yoe INTEGER,
            industry TEXT,
            description TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

# Keep backward compatibility with Supabase if needed
try:
    from supabase import create_client, Client
    from .settings import SUPABASE_URL, SUPABASE_KEY

    def get_supabase_client() -> Client:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
except ImportError:
    def get_supabase_client():
        raise ImportError("Supabase client not available. Install supabase-py if needed.")