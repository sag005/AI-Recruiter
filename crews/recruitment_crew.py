# crews/recruitment_crew.py
from crewai import Crew, Task
from agents.candidate_sourcer import create_candidate_sourcer
from agents.compatibility_agent import create_compatibility_agent
from agents.progress_agent import create_progress_agent


class RecruitmentCrew:
    def __init__(self):
        self.candidate_sourcer = create_candidate_sourcer()
        self.compatibility_agent = create_compatibility_agent()
        self.progress_agent = create_progress_agent()

    def kickoff(self, inputs: dict):
        tasks = [
            Task(
                description=f"Search for {inputs['max_candidates']} candidates for the position",
                agent=self.candidate_sourcer,
                expected_output="List of sourced candidates with parsed resume data"
            ),
            Task(
                description=f"Analyze compatibility of candidates with job description: {inputs['job_description']}",
                agent=self.compatibility_agent,
                expected_output="Ranked candidates by compatibility categories"
            ),
            Task(
                description="Generate comprehensive status report for HR review",
                agent=self.progress_agent,
                expected_output="Detailed recruitment pipeline status report"
            )
        ]

        crew = Crew(
            agents=[self.candidate_sourcer, self.compatibility_agent, self.progress_agent],
            tasks=tasks,
            verbose=True
        )

        return crew.kickoff(inputs)