# database/operations.py
from typing import List, Optional
from .models import Candidate, ParsedResume, JobRequirement
from config.database import get_supabase_client


class DatabaseOperations:
    def __init__(self):
        self.client = get_supabase_client()

    def create_candidate(self, candidate: Candidate) -> int:
        """Create new candidate record"""
        result = self.client.table('candidates').insert(candidate.__dict__).execute()
        return result.data[0]['id']

    def get_candidates_by_status(self, status: str) -> List[Candidate]:
        """Fetch candidates by status"""
        result = self.client.table('candidates').select('*').eq('status', status).execute()
        return [Candidate(**row) for row in result.data]

    def update_candidate_status(self, candidate_id: int, status: str):
        """Update candidate status"""
        self.client.table('candidates').update({'status': status}).eq('id', candidate_id).execute()

    def create_parsed_resume(self, resume: ParsedResume) -> int:
        """Store parsed resume data"""
        result = self.client.table('parsed_resumes').insert(resume.__dict__).execute()
        return result.data[0]['id']

    def get_parsed_resumes(self) -> List[ParsedResume]:
        """Get all parsed resumes"""
        result = self.client.table('parsed_resumes').select('*').execute()
        return [ParsedResume(**row) for row in result.data]