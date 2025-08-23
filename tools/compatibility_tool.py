# tools/compatibility_tool.py
from crewai_tools import BaseTool
from database.operations import DatabaseOperations
from utils.embeddings import calculate_similarity
from typing import Dict, List


class CompatibilityTool(BaseTool):
    name: str = "compatibility_analyzer"
    description: str = "Analyze candidate compatibility with job requirements"

    def __init__(self):
        super().__init__()
        self.db = DatabaseOperations()

    def _run(self, job_description: str) -> Dict[str, List]:
        candidates = self.db.get_parsed_resumes()

        rankings = {
            'highly_desirable': [],
            'desirable': [],
            'not_sure': [],
            'undesirable': [],
            'highly_undesirable': []
        }

        for candidate in candidates:
            score = self._calculate_compatibility_score(candidate, job_description)
            category = self._categorize_score(score)
            rankings[category].append({
                'candidate_id': candidate.candidate_id,
                'name': candidate.name,
                'score': score
            })

        # Update database with rankings
        self._update_candidate_rankings(rankings)

        return rankings

    def _calculate_compatibility_score(self, candidate, job_description: str) -> float:
        # Combine semantic and quantitative scoring
        semantic_score = calculate_similarity(candidate.raw_text, job_description)
        skill_score = self._calculate_skill_match(candidate.skills, job_description)
        yoe_score = self._calculate_yoe_score(candidate.yoe)

        return (semantic_score * 0.4 + skill_score * 0.4 + yoe_score * 0.2)

    def _calculate_skill_match(self, candidate_skills: List[str], job_description: str) -> float:
        # Mock skill matching logic
        job_keywords = ['python', 'django', 'react', 'postgresql']
        matches = sum(1 for skill in candidate_skills if skill.lower() in job_keywords)
        return matches / len(job_keywords)

    def _calculate_yoe_score(self, yoe: int) -> float:
        # Mock YOE scoring (5+ years preferred)
        if yoe >= 5:
            return 1.0
        return yoe / 5.0

    def _categorize_score(self, score: float) -> str:
        if score >= 0.8:
            return 'highly_desirable'
        elif score >= 0.6:
            return 'desirable'
        elif score >= 0.4:
            return 'not_sure'
        elif score >= 0.2:
            return 'undesirable'
        else:
            return 'highly_undesirable'

    def _update_candidate_rankings(self, rankings: Dict):
        # Update candidate status based on ranking
        for category, candidates in rankings.items():
            if category == 'highly_desirable':
                for candidate in candidates:
                    self.db.update_candidate_status(candidate['candidate_id'], 'highly_desirable')