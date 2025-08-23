# crews/monitoring_crew.py
from crewai import Crew
from agents.email_agent import create_email_agent
from agents.scheduler_agent import create_scheduler_agent
import asyncio


class MonitoringCrew:
    def __init__(self):
        self.email_agent = create_email_agent()
        self.scheduler_agent = create_scheduler_agent()

    async def start_monitoring(self):
        # Run both agents in parallel
        email_task = asyncio.create_task(self._run_email_monitoring())
        scheduler_task = asyncio.create_task(self._run_scheduler_service())

        await asyncio.gather(email_task, scheduler_task)

    async def _run_email_monitoring(self):
        print("Email agent monitoring started...")
        # Continuous email monitoring logic

    async def _run_scheduler_service(self):
        print("Scheduler agent service started...")
        # Scheduler service logic