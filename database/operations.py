# database/operations.py
from typing import List, Optional
from datetime import datetime
from .models import Candidate, ParsedResume, JobRequirement
from config.database import get_sqlite_connection, init_sqlite_database
import json


class DatabaseOperations:
    def __init__(self, db_path: str = "recruiter.db"):
        self.db_path = db_path
        # Initialize database on first use
        init_sqlite_database(db_path)

    def create_candidate(self, candidate: Candidate) -> int:
        """Create new candidate record"""
        conn = get_sqlite_connection(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO candidates (name, yoe, current_title, industry, email, phone, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            candidate.name,
            candidate.yoe,
            candidate.current_title,
            candidate.industry,
            candidate.email,
            candidate.phone,
            candidate.status,
            candidate.created_at or datetime.now()
        ))
        
        candidate_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return candidate_id

    def get_candidates_by_status(self, status: str) -> List[Candidate]:
        """Fetch candidates by status"""
        conn = get_sqlite_connection(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM candidates WHERE status = ?', (status,))
        rows = cursor.fetchall()
        conn.close()
        
        return [Candidate(
            id=row['id'],
            name=row['name'],
            yoe=row['yoe'],
            current_title=row['current_title'],
            industry=row['industry'],
            email=row['email'],
            phone=row['phone'],
            status=row['status'],
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
        ) for row in rows]

    def update_candidate_status(self, candidate_id: int, status: str):
        """Update candidate status"""
        conn = get_sqlite_connection(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('UPDATE candidates SET status = ? WHERE id = ?', (status, candidate_id))
        conn.commit()
        conn.close()

    def create_parsed_resume(self, resume: ParsedResume) -> int:
        """Store parsed resume data with embeddings"""
        conn = get_sqlite_connection(self.db_path)
        cursor = conn.cursor()
        
        resume_dict = resume.to_dict()
        cursor.execute('''
            INSERT INTO parsed_resumes (
                candidate_id, name, title, phone, email, skills, yoe, 
                relevant_experience, industry, raw_text, embeddings, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            resume_dict['candidate_id'],
            resume_dict['name'],
            resume_dict['title'],
            resume_dict['phone'],
            resume_dict['email'],
            resume_dict['skills'],
            resume_dict['yoe'],
            resume_dict['relevant_experience'],
            resume_dict['industry'],
            resume_dict['raw_text'],
            resume_dict['embeddings'],
            resume_dict['created_at']
        ))
        
        resume_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return resume_id

    def get_parsed_resumes(self) -> List[ParsedResume]:
        """Get all parsed resumes"""
        conn = get_sqlite_connection(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM parsed_resumes')
        rows = cursor.fetchall()
        conn.close()
        
        return [ParsedResume.from_dict(dict(row)) for row in rows]

    def get_all_candidates(self) -> List[Candidate]:
        """Fetch all candidates from database"""
        conn = get_sqlite_connection(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM candidates')
        rows = cursor.fetchall()
        conn.close()
        
        return [Candidate(
            id=row['id'],
            name=row['name'],
            yoe=row['yoe'],
            current_title=row['current_title'],
            industry=row['industry'],
            email=row['email'],
            phone=row['phone'],
            status=row['status'],
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
        ) for row in rows]

    def search_resumes_by_embedding_similarity(self, query_embedding: List[float], limit: int = 10) -> List[ParsedResume]:
        """Search for similar resumes using embedding similarity (basic implementation)"""
        # Note: This is a basic implementation. For production, consider using a vector database
        # like Chroma, Weaviate, or adding vector search capabilities to SQLite with extensions
        
        conn = get_sqlite_connection(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM parsed_resumes WHERE embeddings IS NOT NULL LIMIT ?', (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        return [ParsedResume.from_dict(dict(row)) for row in rows]

    def get_resume_by_candidate_id(self, candidate_id: int) -> Optional[ParsedResume]:
        """Get parsed resume by candidate ID"""
        conn = get_sqlite_connection(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM parsed_resumes WHERE candidate_id = ?', (candidate_id,))
        row = cursor.fetchone()
        conn.close()
        
        return ParsedResume.from_dict(dict(row)) if row else None