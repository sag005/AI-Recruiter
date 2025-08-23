# main.py
from crews.recruitment_crew import RecruitmentCrew
from crews.monitoring_crew import MonitoringCrew
import asyncio


def main():
    # Initialize recruitment crew (sequential phase)
    recruitment_crew = RecruitmentCrew()

    # Run initial recruitment pipeline
    result = recruitment_crew.kickoff({
        'job_description': 'Senior Python Developer with 5+ years experience...',
        'max_candidates': 20
    })

    # Wait for HR approval
    approval = input("HR Approval (y/n): ")

    if approval.lower() == 'y':
        # Start monitoring crew (parallel phase)
        monitoring_crew = MonitoringCrew()
        asyncio.run(monitoring_crew.start_monitoring())


if __name__ == "__main__":
    main()