# tools/resume_parser_tool.py
from crewai_tools import BaseTool
from database.operations import DatabaseOperations
from database.models import ParsedResume
import random


class ResumeParserTool(BaseTool):
    name: str = "resume_parser"
    description: str = "Parse resume documents and extract structured information"

    def __init__(self):
        super().__init__()
        self.db = DatabaseOperations()

    def _run(self, candidate_id: int, resume_file: str) -> str:
        # For POC: Generate hardcoded parsed data
        parsed_data = self._mock_parse_resume(candidate_id, resume_file)

        resume = ParsedResume(**parsed_data)
        resume_id = self.db.create_parsed_resume(resume)

        return f"Successfully parsed resume for candidate {candidate_id}"

    def _mock_parse_resume(self, candidate_id: int, resume_file: str) -> dict:
        skills_pool = ['Python', 'JavaScript', 'React', 'Django', 'FastAPI', 'PostgreSQL', 'AWS', 'Docker']
        return {
            'candidate_id': candidate_id,
            'name': f'Parsed_Name_{candidate_id}',
            'title': random.choice(['Software Engineer', 'Full Stack Developer']),
            'phone': f'555-{candidate_id:04d}',
            'email': f'parsed{candidate_id}@example.com',
            'skills': random.sample(skills_pool, random.randint(3, 6)),
            'yoe': random.randint(1, 12),
            'relevant_experience': [f'Experience_{i}' for i in range(1, 4)],
            'industry': 'Technology',
            'raw_text': f'Raw resume text for candidate {candidate_id}...'
        }