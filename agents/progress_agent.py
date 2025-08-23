# agents/progress_agent.py
from crewai import Agent
from database.operations import DatabaseOperations


class ProgressStatusTool:
    def __init__(self):
        self.db = DatabaseOperations()

    def generate_status_report(self) -> str:
        highly_desirable = self.db.get_candidates_by_status('highly_desirable')
        contacted = self.db.get_candidates_by_status('contacted')
        responded = self.db.get_candidates_by_status('responded')
        scheduled = self.db.get_candidates_by_status('scheduled')

        report = f"""
        RECRUITMENT STATUS REPORT
        ========================

        New Highly Desirable Candidates: {len(highly_desirable)}
        Candidates Contacted: {len(contacted)}
        Candidates Responded: {len(responded)}
        Candidates Scheduled: {len(scheduled)}

        Ready to proceed with outreach campaign.
        """

        return report


def create_progress_agent() -> Agent:
    return Agent(
        role='Progress Status Reporter',
        goal='Monitor recruitment pipeline status and generate comprehensive reports for HR review',
        backstory='You are a project manager specialized in recruitment operations, providing clear status updates and insights.',
        tools=[ProgressStatusTool()],
        verbose=True
    )