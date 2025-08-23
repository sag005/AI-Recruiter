"""
Crew AI Agent that takes in a Resume
- parses the PDF
- extracts data in a structured format as defined in ResumeData model (using CrewAI task)
- embeds the structured information to an embedding model (Voyage)
- inserts the structured data to a sqlite db with UUID and returns the id
- inserts the embedding to chroma db with the UUID as the key.
"""

import json
import sqlite3
import uuid
from typing import List
import json

import PyPDF2
import voyageai
import chromadb
from crewai import Agent, Task, Crew, LLM
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables."""
    anthropic_api_key: str = ""
    voyage_api_key: str = ""
    sqlite_db_path: str = "recruiter.db"
    chroma_db_path: str = "./chroma_db"
    
    class Config:
        env_file = ".env"


class PersonalInfo(BaseModel):
    """Personal information from resume."""
    name: str = None
    email: str = None
    phone: str = None
    location: str = None
    linkedin: str = None


class WorkExperience(BaseModel):
    """Work experience entry."""
    job_title: str
    company: str
    duration: str = None
    location: str = None
    description: str = None
    responsibilities: List[str] = Field(default_factory=list)


class Education(BaseModel):
    """Education entry."""
    degree: str
    field_of_study: str = None
    institution: str
    graduation_year: str = None
    gpa: str = None
    location: str = None


class Certification(BaseModel):
    """Professional certification."""
    name: str
    issuer: str = None
    year: str = None
    expiry: str = None


class ResumeData(BaseModel):
    """Complete structured resume data."""
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo)
    professional_summary: str = None
    work_experience: List[WorkExperience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    category: str = None
    years_of_experience: int = None
    seniority_level: str = None
    keywords: List[str] = Field(default_factory=list)
    pdf_path: str = None


class ResumeIngressAgent:
    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        
        # Configure LLM to use Anthropic Claude
        self.llm = LLM(
            model="claude-3-5-sonnet-20241022", 
            api_key=self.settings.anthropic_api_key
        )
        
        # Initialize Voyage client for embeddings
        if self.settings.voyage_api_key:
            self.voyage_client = voyageai.Client(api_key=self.settings.voyage_api_key)
        else:
            self.voyage_client = None
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=self.settings.chroma_db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="resume_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Create CrewAI agent for resume parsing with Anthropic Claude
        self.resume_parser_agent = Agent(
            role="Resume Data Extractor",
            goal="Extract structured information from resume text",
            backstory="You are an expert at parsing resumes and extracting structured data. You always return valid JSON that matches the requested schema.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
    def process_resume(self, pdf_path: str) -> str:
        """Main method to process a resume PDF and return the generated UUID."""
        
        # Extract text from PDF
        raw_text = self._extract_pdf_text(pdf_path)
        
        # Extract structured data using CrewAI task
        resume_id = self._extract_structured_data_with_crew(raw_text, pdf_path)
        embedding_text = self._get_embedding_text(resume_id)
        # Generate embeddings after successful DB insertion
        embedding = self._generate_embedding(embedding_text)
        
        # Insert embedding to ChromaDB with same UUID
        if embedding:
            self._insert_to_chromadb(resume_id, embedding, embedding_text)
        else:
            print(f"Skipping ChromaDB insertion - no embedding generated")
        
        return resume_id
    
    
    def _get_embedding_text(self, resume_id: str) -> str:
        with open(f"parsed_resumes/{resume_id}.json",'r') as f:
            text = f.read()
        return text
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    
    def _extract_structured_data_with_crew(self, raw_text: str, pdf_path: str) -> str:
        """Extract structured data from resume text using CrewAI task."""
        
        # Create the task for resume parsing
        parse_task = Task(
            description=f"""
            Extract structured information from this resume text and return it as valid JSON.
            
            Use these placeholder names like these for personal information:
            - Name: "John Doe" 
            - Email: "john.doe@example.com"
            - Phone: "(555) 123-4567"
            - Location: "San Francisco, CA"

            Make sure you change the names so they appear unique.
            
            Resume text:
            {raw_text[:3000]}  # Limit text to avoid token limits
            
            Return ONLY valid JSON matching this exact structure:
            {{
                "personal_info": {{
                    "name": "John Doe",
                    "email": "john.doe@example.com", 
                    "phone": "(555) 123-4567",
                    "location": "San Francisco, CA",
                    "linkedin": null
                }},
                "professional_summary": "Brief professional summary based on the resume",
                "work_experience": [
                    {{
                        "job_title": "Job Title",
                        "company": "Company Name",
                        "duration": "YYYY-YYYY",
                        "location": "City, State",
                        "description": "Brief job description",
                        "responsibilities": ["Responsibility 1", "Responsibility 2"]
                    }}
                ],
                "education": [
                    {{
                        "degree": "Degree Name",
                        "field_of_study": "Field of Study",
                        "institution": "Institution Name",
                        "graduation_year": "YYYY",
                        "gpa": null,
                        "location": "City, State"
                    }}
                ],
                "skills": ["Skill1", "Skill2", "Skill3"],
                "certifications": [
                    {{
                        "name": "Certification Name",
                        "issuer": "Issuing Organization",
                        "year": "YYYY",
                        "expiry": null
                    }}
                ],
                "languages": ["English"],
                "category": "Engineering",
                "years_of_experience": 0,
                "seniority_level": "Junior",
                "keywords": ["keyword1", "keyword2"],
                "pdf_path": "{pdf_path}"
            }}
            """,
            agent=self.resume_parser_agent,
            expected_output="Valid JSON object containing structured resume data",
        )
        
        # Create and run the crew
        crew = Crew(
            agents=[self.resume_parser_agent],
            tasks=[parse_task],
            verbose=True
        )
        
        # Execute the crew - no fallback, fail if it fails
        result = crew.kickoff()
        resume_id = uuid.uuid4()
        with open(f'parsed_resumes/{resume_id}.json','w') as f:
            f.write(str(result))
        return str(resume_id)
    
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Voyage AI."""
        if not self.voyage_client:
            print("Voyage client not available, skipping embedding generation")
            return None
            
        try:
            result = self.voyage_client.embed(
                texts=[text], 
                model="voyage-2", 
                input_type="document"
            )
            return result.embeddings[0]
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            return None
    
   
    def _insert_to_chromadb(self, resume_id: str, embedding: List[float], embedding_text: str) -> None:
        """Insert embedding to ChromaDB with resume UUID as key."""
        # Insert into ChromaDB
        self.collection.add(
            embeddings=[embedding],
            documents=[embedding_text],
            ids=[resume_id]
        )


def main():
    """Process 20 sample resumes from the data folder."""
    agent = ResumeIngressAgent()
    
    # Get all PDF files from all categories
    data_folder = "/Users/ada/fun/AI-Recruiter/data"
    pdf_files = []
    
    # Collect PDFs from all category folders
    for category in ["ENGINEERING", "DESIGNER", "BUSINESS-DEVELOPMENT"]:
        category_path = os.path.join(data_folder, category)
        if os.path.exists(category_path):
            category_pdfs = [os.path.join(category_path, f) for f in os.listdir(category_path) if f.endswith('.pdf')]
            pdf_files.extend(category_pdfs)
    
    # Sample 20 resumes
    import random
    sample_size = min(20, len(pdf_files))
    sample_pdfs = random.sample(pdf_files, sample_size)
    
    print(f"Processing {sample_size} resumes from {len(pdf_files)} total PDFs")
    
    successful_count = 0
    failed_count = 0
    
    for i, pdf_path in enumerate(sample_pdfs, 1):
        print(f"\n[{i}/{sample_size}] Processing: {os.path.basename(pdf_path)}")
        try:
            resume_id = agent.process_resume(pdf_path)
            print(f"✅ Successfully processed. Generated ID: {resume_id}")
            successful_count += 1
        except Exception as e:
            print(f"❌ Error processing resume: {e}")
            failed_count += 1
    
    print(f"\n=== Summary ===")
    print(f"Successfully processed: {successful_count}")
    print(f"Failed: {failed_count}")
    print(f"Total: {sample_size}")


if __name__ == "__main__":
    import os
    import random
    main()