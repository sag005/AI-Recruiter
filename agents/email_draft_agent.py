"""
Email Draft Agent

Creates targeted email drafts for candidates explaining:
1. Why they are a good match for the position
2. Why the job is beneficial for them
3. Includes 3 time slots for scheduling from scheduler_agent.py
"""

import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

from .candidate_matcher import CandidateMatch, CandidateRanking, CandidateMatcherAgent
from .scheduler_agent import get_available_slots_direct


class Settings(BaseSettings):
    """Application settings from environment variables."""
    anthropic_api_key: str = ""
    
    model_config = ConfigDict(env_file=".env", extra="ignore")


class EmailDraft(BaseModel):
    """Email draft for a candidate."""
    candidate_id: str
    candidate_name: str
    subject_line: str
    email_body: str
    time_slots: List[str] = Field(default_factory=list)


class EmailDraftAgent:
    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        
        # Configure LLM to use Anthropic Claude
        self.llm = LLM(
            model="claude-3-5-sonnet-20241022", 
            api_key=self.settings.anthropic_api_key
        )
        
        # Initialize candidate matcher
        self.candidate_matcher = CandidateMatcherAgent(self.settings)
        
        # Create the time slot tool
        self.schedule_tool = self._create_schedule_tool()
        
        # Create CrewAI agents
        self.email_writer = Agent(
            role="Personalized Email Writer",
            goal="Create compelling, personalized emails that highlight candidate strengths and job benefits",
            backstory="You are an expert recruiter who excels at writing personalized, engaging emails that make candidates excited about opportunities. You understand how to highlight why someone is a perfect match while also explaining what makes a role attractive to them specifically.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.schedule_tool]
        )
    
    def _create_schedule_tool(self):
        """Create a tool to get available time slots."""
        
        @tool("Get Available Time Slots")
        def get_time_slots() -> str:
            """
            Get 3 available 30-minute time slots for candidate interviews.
            
            Returns:
                String containing 3 available time slots
            """
            try:
                slots_result = get_available_slots_direct()
                return slots_result
            except Exception as e:
                return f"Error getting time slots: {str(e)}"
        
        return get_time_slots
    
    def create_email_draft(self, candidate: CandidateMatch, job_description: str, job_title: str, company_name: str = "Our Company") -> EmailDraft:
        """
        Create a personalized email draft for a specific candidate.
        
        Args:
            candidate: CandidateMatch object with candidate details
            job_description: The full job description
            job_title: The job title
            company_name: Company name (default: "Our Company")
            
        Returns:
            EmailDraft object with personalized email content and time slots
        """
        
        # Task to create personalized email
        email_task = Task(
            description=f"""
            Create a personalized email draft for the following candidate:
            
            Candidate Name: {candidate.name}
            Match Score: {candidate.match_score}
            Key Strengths: {', '.join(candidate.key_strengths)}
            Relevant Experience: {', '.join(candidate.relevant_experience)}
            Rationale: {candidate.rationale}
            
            Job Details:
            Company: {company_name}
            Position: {job_title}
            Job Description: {job_description}
            
            Instructions:
            1. First, use the "Get Available Time Slots" tool to retrieve 3 available interview times
            2. Write a compelling subject line (max 60 characters)
            3. Create an email body that includes:
               a) Personal greeting using their name
               b) Specific mention of why they're a strong match (reference their key strengths and experience)
               c) Explanation of why this role would be beneficial for their career (growth opportunities, learning, impact)
               d) Brief overview of what makes the company/role attractive
               e) Request to schedule an interview with the 3 time slot options
               f) Professional but warm closing
            
            Tone: Professional yet personalized, enthusiastic but not overly salesy, respectful of their time
            Length: Keep email concise (200-300 words max)
            
            Return the result as a JSON object with this exact structure:
            {{
                "candidate_id": "{candidate.candidate_id}",
                "candidate_name": "{candidate.name}",
                "subject_line": "<compelling subject line>",
                "email_body": "<complete email body with time slots included>",
                "time_slots": ["slot1", "slot2", "slot3"]
            }}
            """,
            agent=self.email_writer,
            expected_output="JSON object with personalized email draft and time slots"
        )
        
        # Create and run the crew
        crew = Crew(
            agents=[self.email_writer],
            tasks=[email_task],
            verbose=True
        )
        
        # Execute the crew
        result = crew.kickoff()
        
        # Parse the result
        try:
            result_str = str(result)
            # Extract JSON from the result
            if "{" in result_str:
                json_start = result_str.find("{")
                json_end = result_str.rfind("}") + 1
                json_str = result_str[json_start:json_end]
                result_data = json.loads(json_str)
                return EmailDraft(**result_data)
            else:
                raise ValueError("No JSON found in result")
        except Exception as e:
            # Fallback: create basic email
            return EmailDraft(
                candidate_id=candidate.candidate_id,
                candidate_name=candidate.name,
                subject_line=f"Exciting {job_title} Opportunity at {company_name}",
                email_body=f"Dear {candidate.name},\n\nI hope this email finds you well. I came across your profile and was impressed by your background, particularly your {', '.join(candidate.key_strengths[:2])}.\n\nWe have an exciting {job_title} position at {company_name} that I believe would be a great fit for your skills and career goals.\n\nWould you be available for a brief conversation to discuss this opportunity? Please let me know what works best for your schedule.\n\nBest regards,\nRecruiting Team",
                time_slots=["9:00 AM - 9:30 AM", "2:00 PM - 2:30 PM", "4:00 PM - 4:30 PM"]
            )
    
    def create_bulk_email_drafts(self, job_description: str, job_title: str, company_name: str = "Our Company", max_candidates: int = 5) -> List[EmailDraft]:
        """
        Create email drafts for multiple top candidates for a job.
        
        Args:
            job_description: The full job description
            job_title: The job title
            company_name: Company name
            max_candidates: Maximum number of candidates to create emails for
            
        Returns:
            List of EmailDraft objects
        """
        
        # First find the best candidates
        ranking = self.candidate_matcher.find_best_candidates(job_description, job_title)
        
        # Create email drafts for top candidates
        email_drafts = []
        for candidate in ranking.top_matches[:max_candidates]:
            try:
                draft = self.create_email_draft(candidate, job_description, job_title, company_name)
                email_drafts.append(draft)
            except Exception as e:
                print(f"Error creating email for candidate {candidate.name}: {e}")
                continue
        
        return email_drafts


def main():
    """Test the email draft agent with a sample job."""
    
    job_description = """
    We are seeking a Senior Software Engineer to join our growing engineering team.
    
    Requirements:
    - 5+ years of software development experience
    - Strong proficiency in Python, JavaScript, and modern web frameworks
    - Experience with cloud platforms (AWS, GCP, or Azure)
    - Knowledge of database design and optimization
    - Experience with microservices architecture
    - Strong problem-solving skills and attention to detail
    
    Responsibilities:
    - Design and implement scalable software solutions
    - Collaborate with cross-functional teams
    - Mentor junior developers
    - Participate in code reviews and architectural decisions
    - Drive technical innovation and best practices
    
    Benefits:
    - Competitive salary and equity
    - Flexible work arrangements
    - Professional development budget
    - Health, dental, and vision insurance
    - 401k matching
    """
    
    job_title = "Senior Software Engineer"
    company_name = "TechCorp"
    
    try:
        agent = EmailDraftAgent()
        email_drafts = agent.create_bulk_email_drafts(job_description, job_title, company_name, max_candidates=3)
        
        print("\n" + "="*50)
        print("EMAIL DRAFTS GENERATED")
        print("="*50)
        
        for i, draft in enumerate(email_drafts, 1):
            print(f"\n--- Email #{i} ---")
            print(f"Candidate: {draft.candidate_name} (ID: {draft.candidate_id})")
            print(f"Subject: {draft.subject_line}")
            print(f"Time Slots: {', '.join(draft.time_slots)}")
            print(f"\nEmail Body:\n{draft.email_body}")
            print("\n" + "-"*30)
            
    except Exception as e:
        print(f"Error running email draft agent: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()