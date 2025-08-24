"""
Crew AI Agent that takes in a Job Description
- parses the description
- Uses the Chroma DB RAG Tool to query the `chroma_db` to get the top 5 candidates
- Generates a list of ranked candidates with rationale.
"""

import os
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings

import chromadb
import voyageai
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

from .scheduler_agent import get_available_slots_direct


class Settings(BaseSettings):
    """Application settings from environment variables."""
    anthropic_api_key: str = ""
    voyage_api_key: str = ""
    chroma_db_path: str = "./chroma_db"
    
    model_config = ConfigDict(env_file=".env", extra="ignore")


class CandidateMatch(BaseModel):
    """Individual candidate match result."""
    candidate_id: str
    name: str
    match_score: float = Field(ge=0.0, le=1.0, description="Match score between 0.0 and 1.0")
    rationale: str
    key_strengths: List[str] = Field(default_factory=list)
    potential_concerns: List[str] = Field(default_factory=list)
    relevant_experience: List[str] = Field(default_factory=list)


class CandidateRanking(BaseModel):
    """Complete ranking of candidates for a job."""
    job_title: str
    total_candidates_found: int
    top_matches: List[CandidateMatch] = Field(max_length=5)
    search_query_used: str
    summary: str


class EmailDraft(BaseModel):
    """Email draft for a candidate."""
    candidate_id: str
    candidate_name: str
    subject_line: str
    email_body: str
    time_slots: List[str] = Field(default_factory=list)


class CandidateMatcherAgent:
    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        
        # Configure LLM to use Anthropic Claude
        self.llm = LLM(
            model="claude-3-5-sonnet-20241022", 
            api_key=self.settings.anthropic_api_key
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=self.settings.chroma_db_path)
        try:
            self.collection = self.chroma_client.get_collection("resume_embeddings")
        except Exception as e:
            raise Exception(f"Resume embeddings collection not found. Run resume_ingress.py first. Error: {e}")
        
        # Initialize Voyage client for consistent embeddings
        if self.settings.voyage_api_key:
            self.voyage_client = voyageai.Client(api_key=self.settings.voyage_api_key)
        else:
            raise Exception("Voyage API key required for consistent embeddings")
        
        # Create the RAG search tool and scheduling tool
        self.search_tool = self._create_search_tool()
        self.schedule_tool = self._create_schedule_tool()
        
        # Create CrewAI agents
        self.job_analyzer = Agent(
            role="Job Requirements Analyzer",
            goal="Extract and analyze key requirements, skills, and qualifications from job descriptions",
            backstory="You are an expert HR analyst who understands how to break down job requirements into searchable criteria. You excel at identifying the most important skills, experience levels, and qualifications needed for a role.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        self.candidate_ranker = Agent(
            role="Candidate Ranking Specialist", 
            goal="Rank and evaluate candidates based on job requirements with detailed rationale",
            backstory="You are a senior recruiter with expertise in matching candidates to roles. You provide detailed analysis of why candidates are good matches, including strengths, concerns, and specific relevant experience.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.search_tool]
        )
        
        self.email_writer = Agent(
            role="Personalized Email Writer",
            goal="Create compelling, personalized emails that highlight candidate strengths and job benefits with scheduling options",
            backstory="You are an expert recruiter who excels at writing personalized, engaging emails that make candidates excited about opportunities. You understand how to highlight why someone is a perfect match while explaining what makes a role attractive to them specifically.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.schedule_tool]
        )
    
    def _create_search_tool(self):
        """Create a custom ChromaDB search tool for finding candidates."""
        
        @tool("Search Resume Database")
        def search_candidates(query: str) -> str:
            """
            Search the resume database for candidates matching the given criteria.
            
            Args:
                query: Search query describing the skills, experience, or qualifications needed
                
            Returns:
                JSON string containing candidate information and resume data
            """
            try:
                # Generate embedding using the same model as resume ingestion (Voyage AI)
                embedding_result = self.voyage_client.embed(
                    texts=[query], 
                    model="voyage-2",  # Same model used in resume_ingress.py
                    input_type="query"  # Use 'query' for search queries
                )
                query_embedding = embedding_result.embeddings[0]
                
                # Query ChromaDB for similar resumes using embedding
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=10,  # Get top 10 to have more options for ranking
                    include=["documents", "metadatas", "distances"]
                )
                
                candidates = []
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results.get('metadatas', [[{}] * len(results['documents'][0])])[0],
                    results['distances'][0]
                )):
                    # Parse the resume JSON
                    try:
                        resume_data = json.loads(doc)
                        similarity_score = max(0, 1 - distance)  # Convert distance to similarity
                        
                        candidate_info = {
                            "candidate_id": results['ids'][0][i],
                            "similarity_score": similarity_score,
                            "resume_data": resume_data
                        }
                        candidates.append(candidate_info)
                    except json.JSONDecodeError:
                        continue
                
                return json.dumps({
                    "total_found": len(candidates),
                    "candidates": candidates,
                    "query_used": query
                }, indent=2)
                
            except Exception as e:
                return f"Error searching candidates: {str(e)}"
        
        return search_candidates
    
    def _create_schedule_tool(self):
        """Create a tool to get available time slots."""
        
        @tool("Get Available Time Slots")
        def get_time_slots() -> str:
            """
            Get 3 available 30-minute time slots for candidate interviews (weekdays only).
            
            Returns:
                String containing 3 available time slots during business hours
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
               b) Write in FIRST PERSON from the perspective of "Sarah Chen, Senior Recruiter"
               c) Specific mention of why they're a strong match (reference their key strengths and experience)
               d) Explanation of why this role would be beneficial for their career (growth opportunities, learning, impact)
               e) Brief overview of what makes the company/role attractive
               f) Request to schedule an interview with the 3 time slot options
               g) Professional but warm closing signed "Best regards, Sarah Chen"
            
            Tone: Professional yet personalized, enthusiastic but not overly salesy, respectful of their time, written as if from a real recruiter
            Length: Keep email concise (200-300 words max)
            Voice: First person ("I noticed", "I believe", "I'd love to discuss") from Sarah Chen's perspective
            
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
                email_body=f"Dear {candidate.name},\n\nI hope this email finds you well. I came across your profile and was impressed by your background, particularly your {', '.join(candidate.key_strengths[:2])}.\n\nI believe you'd be a great fit for our {job_title} position at {company_name}. This role offers excellent opportunities for growth and would leverage your existing expertise while expanding your skill set.\n\nI'd love to discuss this opportunity with you. Are you available for a brief conversation?\n\nBest regards,\nSarah Chen\nSenior Recruiter",
                time_slots=["Monday: 9:00 AM - 9:30 AM", "Tuesday: 2:00 PM - 2:30 PM", "Wednesday: 4:00 PM - 4:30 PM"]
            )
    
    def find_best_candidates(self, job_description: str, job_title: str = "") -> CandidateRanking:
        """
        Main method to find and rank the best candidates for a job.
        
        Args:
            job_description: The job description text
            job_title: Optional job title
            
        Returns:
            CandidateRanking object with top 5 ranked candidates
        """
        
        # Task 1: Analyze job requirements
        job_analysis_task = Task(
            description=f"""
            Analyze the following job description and extract the key requirements:
            
            Job Title: {job_title}
            Job Description: {job_description}
            
            Extract and identify:
            1. Required technical skills and technologies
            2. Years of experience needed
            3. Industry/domain requirements
            4. Education requirements
            5. Soft skills and leadership requirements
            6. Key responsibilities
            
            Create a search query that would help find the most relevant candidates.
            Focus on the most important 3-5 criteria that would differentiate strong candidates.
            
            Return your analysis in this format:
            SEARCH_QUERY: [optimized search query for finding relevant candidates]
            KEY_REQUIREMENTS: [list of the most important requirements]
            """,
            agent=self.job_analyzer,
            expected_output="Search query and key requirements analysis"
        )
        
        # Task 2: Search and rank candidates
        ranking_task = Task(
            description=f"""
            Using the job analysis, search for candidates and provide a detailed ranking.
            
            Job Description: {job_description}
            
            Steps:
            1. Use the Search Resume Database tool with the search query from the job analysis
            2. Evaluate each candidate against the job requirements
            3. Rank the top 5 candidates with match scores (0.0 to 1.0)
            4. For each candidate provide:
               - Match score and detailed rationale
               - Key strengths that align with the role
               - Potential concerns or gaps
               - Specific relevant experience
            
            Return the results as a JSON object with this exact structure:
            {{
                "job_title": "{job_title}",
                "total_candidates_found": <number>,
                "top_matches": [
                    {{
                        "candidate_id": "<uuid>",
                        "name": "<candidate_name>",
                        "match_score": <score_0_to_1>,
                        "rationale": "<detailed explanation of why this candidate is a good match>",
                        "key_strengths": ["strength1", "strength2", "strength3"],
                        "potential_concerns": ["concern1", "concern2"],
                        "relevant_experience": ["experience1", "experience2"]
                    }}
                ],
                "search_query_used": "<the search query used>",
                "summary": "<overall summary of the candidate pool and top recommendations>"
            }}
            """,
            agent=self.candidate_ranker,
            expected_output="JSON object with ranked candidates",
            context=[job_analysis_task]
        )
        
        # Create and run the crew
        crew = Crew(
            agents=[self.job_analyzer, self.candidate_ranker],
            tasks=[job_analysis_task, ranking_task],
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
                return CandidateRanking(**result_data)
            else:
                raise ValueError("No JSON found in result")
        except Exception as e:
            # Fallback: return empty result
            return CandidateRanking(
                job_title=job_title or "Unknown Position",
                total_candidates_found=0,
                top_matches=[],
                search_query_used="",
                summary=f"Error processing results: {str(e)}"
            )


def main():
    """Test the candidate matcher with a sample job description."""
    
    # Adobe Creative Suite focused job description
    job_description = """
    We are seeking a highly skilled Senior Creative Designer to join our dynamic marketing team. 
    
    Requirements:
    - 5+ years of professional design experience in advertising, branding, or digital media
    - Expert-level proficiency in Adobe Creative Suite, particularly Photoshop, Illustrator, and InDesign
    - Advanced Photoshop skills including photo retouching, compositing, and digital manipulation
    - Strong experience in 3D modeling and rendering software (Cinema 4D, Blender, or Maya preferred)
    - Experience with video editing and motion graphics (After Effects, Premiere Pro)
    - Portfolio demonstrating expertise in photo editing, digital art creation, and brand design
    - Experience with print and digital media design
    - Bachelor's degree in Graphic Design, Visual Arts, or related creative field
    - Strong understanding of color theory, typography, and layout principles
    - Experience working with photographers and managing photo shoots
    
    Responsibilities:
    - Create stunning visual content for marketing campaigns and brand materials
    - Perform advanced photo retouching and image manipulation for product photography
    - Design digital assets for web, social media, and mobile platforms
    - Collaborate with photographers on creative concepts and post-production workflows
    - Develop 3D models and renders for product visualization
    - Maintain brand consistency across all visual communications
    - Mentor junior designers and provide creative direction
    - Stay current with design trends and emerging creative technologies
    
    Preferred Qualifications:
    - Experience with fashion, beauty, or lifestyle photography editing
    - Knowledge of web design principles and UI/UX basics
    - Experience with digital asset management systems
    - Portfolio showcasing both creative and technical excellence
    """
    
    job_title = "Senior Creative Designer"
    
    try:
        matcher = CandidateMatcherAgent()
        result = matcher.find_best_candidates(job_description, job_title)
        
        print("\n" + "="*50)
        print("CANDIDATE MATCHING RESULTS")
        print("="*50)
        print(f"Job Title: {result.job_title}")
        print(f"Total Candidates Found: {result.total_candidates_found}")
        print(f"Search Query Used: {result.search_query_used}")
        print(f"\nSummary: {result.summary}")
        
        print(f"\nTOP {len(result.top_matches)} CANDIDATES:")
        print("-" * 40)
        
        for i, candidate in enumerate(result.top_matches, 1):
            print(f"\n#{i} - {candidate.name} (ID: {candidate.candidate_id})")
            print(f"Match Score: {candidate.match_score:.2f}")
            print(f"Rationale: {candidate.rationale}")
            print(f"Key Strengths: {', '.join(candidate.key_strengths)}")
            if candidate.potential_concerns:
                print(f"Potential Concerns: {', '.join(candidate.potential_concerns)}")
            print(f"Relevant Experience: {', '.join(candidate.relevant_experience)}")
            
    except Exception as e:
        print(f"Error running candidate matcher: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()