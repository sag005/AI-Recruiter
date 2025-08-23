# tools/candidate_search_tool.py
from crewai_tools import BaseTool
from database.operations import DatabaseOperations
from database.models import Candidate
from .resume_parser_tool import ResumeParserTool
import random


class CandidateSearchTool(BaseTool):
    name: str = "candidate_search"
    description: str = "Search for candidates in database and parse their resumes"

    def __init__(self):
        super().__init__()
        self.db = DatabaseOperations()
        self.resume_parser = ResumeParserTool()

    def _run(self, max_candidates: int = 10) -> str:
        # For POC: Generate hardcoded candidates
        hardcoded_candidates = self._generate_sample_candidates(max_candidates)

        candidate_ids = []
        for candidate_data in hardcoded_candidates:
            candidate = Candidate(**candidate_data)
            candidate_id = self.db.create_candidate(candidate)
            candidate_ids.append(candidate_id)

            # Trigger resume parsing
            self.resume_parser._run(candidate_id, candidate_data['resume_file'])

        return f"Successfully sourced {len(candidate_ids)} candidates"

    def _generate_sample_candidates(self, count: int) -> list:
        # Hardcoded sample data for POC
        sample_data = [
            {
                'name': f'Candidate_{i}',
                'yoe': random.randint(1, 15),
                'current_title': random.choice(['Software Engineer', 'Senior Developer', 'Tech Lead']),
                'industry': 'Technology',
                'email': f'candidate{i}@example.com',
                'phone': f'555-000{i:04d}',
                'status': 'new',
                'resume_file': f'resume_{i}.pdf'
            }
            for i in range(1, count + 1)
        ]
        return sample_data