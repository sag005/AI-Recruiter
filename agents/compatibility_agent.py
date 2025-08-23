# agents/compatibility_agent.py
from crewai import Agent
from tools.compatibility_tool import CompatibilityTool

def create_compatibility_agent() -> Agent:
    return Agent(
        role='Compatibility Analyzer',
        goal='Evaluate how well candidates match job requirements using both semantic analysis and quantitative metrics',
        backstory='You are an AI specialist in candidate assessment, skilled at analyzing resumes against job descriptions using advanced matching algorithms.',
        tools=[CompatibilityTool()],
        verbose=True
    )