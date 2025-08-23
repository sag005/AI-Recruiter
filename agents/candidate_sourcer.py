# agents/candidate_sourcer.py
from crewai import Agent
from tools.candidate_search_tool import CandidateSearchTool

def create_candidate_sourcer() -> Agent:
    return Agent(
        role='Candidate Sourcer',
        goal='Find and collect potential candidates for job positions',
        backstory='You are an expert recruiter who knows how to identify potential candidates and gather their information efficiently.',
        tools=[CandidateSearchTool()],
        verbose=True
    )