"""CrewAI script for processing resumes with real Anthropic parsing and Voyage embeddings."""

import json
import os
from pathlib import Path
from typing import Dict, List, Any

from crewai import Agent, Task, Crew
from crewai.tools import tool
from dotenv import load_dotenv

from real_resume_parser import ResumeParser
from src.ai_recruiter.models import ResumeData

load_dotenv()


class ResumeProcessingCrew:
    """CrewAI-based resume processing system."""
    
    def __init__(self):
        self.parser = ResumeParser()
        self.processed_resumes = {}
        self.embeddings_database = {}
        
        # Define tools first
        self.parser_tool = self._create_parser_tool()
        self.embedding_tool = self._create_embedding_tool()
        self.analysis_tool = self._create_analysis_tool()
        
        # Create agents
        self.setup_agents()
    
    def _create_parser_tool(self):
        """Create PDF parsing tool."""
        @tool("Resume Parser")
        def parse_resume(pdf_path: str) -> Dict[str, Any]:
            """Parse a PDF resume and extract structured information."""
            try:
                resume_data = self.parser.parse_resume(pdf_path)
                return resume_data.model_dump()
            except Exception as e:
                return {"error": f"Failed to parse resume: {str(e)}"}
        
        return parse_resume
    
    def _create_embedding_tool(self):
        """Create embedding generation tool."""
        @tool("Embedding Generator")
        def create_embedding(pdf_path: str) -> Dict[str, Any]:
            """Create vector embeddings for a resume PDF."""
            try:
                return self.parser.create_embeddings(pdf_path)
            except Exception as e:
                return {"error": f"Failed to create embeddings: {str(e)}"}
        
        return create_embedding
    
    def _create_analysis_tool(self):
        """Create resume analysis tool."""
        @tool("Resume Analyzer")
        def analyze_resume(resume_data: Dict[str, Any]) -> Dict[str, Any]:
            """Analyze parsed resume data and provide insights."""
            try:
                # Calculate experience level
                years = resume_data.get('years_of_experience', 0)
                if years < 3:
                    level = "Junior"
                elif years < 7:
                    level = "Mid-level"
                elif years < 12:
                    level = "Senior"
                else:
                    level = "Executive"
                
                # Analyze skills
                skills = resume_data.get('skills', [])
                tech_skills = [s for s in skills if any(tech in s.lower() for tech in 
                              ['python', 'java', 'react', 'sql', 'aws', 'docker', 'kubernetes', 'git'])]
                
                # Provide recommendations
                strengths = []
                if len(resume_data.get('work_experience', [])) >= 3:
                    strengths.append("Strong work experience with multiple roles")
                if len(skills) >= 10:
                    strengths.append("Comprehensive skill set")
                if resume_data.get('certifications'):
                    strengths.append("Professional certifications")
                
                recommendations = []
                if not resume_data.get('personal_info', {}).get('email'):
                    recommendations.append("Add contact email to resume")
                if len(skills) < 5:
                    recommendations.append("Expand technical skills section")
                
                return {
                    "seniority_level": level,
                    "technical_skills_count": len(tech_skills),
                    "strengths": strengths,
                    "recommendations": recommendations,
                    "overall_score": min(10, max(1, 3 + len(strengths) + len(skills)//3))
                }
                
            except Exception as e:
                return {"error": f"Analysis failed: {str(e)}"}
        
        return analyze_resume
    
    def setup_agents(self):
        """Initialize CrewAI agents."""
        self.parser_agent = Agent(
            role='Resume Parser Specialist',
            goal='Extract comprehensive structured data from PDF resumes',
            backstory='''You are an expert HR data extraction specialist with years of experience 
                        in parsing resumes and organizing candidate information. You excel at identifying 
                        and extracting key details from various resume formats.''',
            tools=[self.parser_tool],
            verbose=True
        )
        
        self.embedding_agent = Agent(
            role='Document Embedding Specialist', 
            goal='Generate high-quality vector embeddings for resume similarity matching',
            backstory='''You are a machine learning engineer specialized in creating vector embeddings 
                        for document retrieval systems. You understand how to optimize embeddings for 
                        HR and recruitment use cases.''',
            tools=[self.embedding_tool],
            verbose=True
        )
        
        self.analyst_agent = Agent(
            role='Resume Analysis Expert',
            goal='Provide detailed insights and recommendations about candidate profiles',
            backstory='''You are a senior HR analyst with expertise in talent assessment. You can 
                        quickly identify candidate strengths, experience levels, and provide actionable 
                        recommendations for both recruiters and candidates.''',
            tools=[self.analysis_tool],
            verbose=True
        )
    
    def process_resume(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single resume through the CrewAI pipeline."""
        
        # Task 1: Parse the resume
        parse_task = Task(
            description=f"""Parse the PDF resume located at {pdf_path} and extract all relevant information including:
            - Personal information (name, email, phone, location)
            - Work experience with job titles, companies, and durations
            - Education background
            - Skills and certifications
            - Professional summary
            
            Ensure all extracted data is accurate and complete.""",
            expected_output="Structured JSON data containing all resume information",
            agent=self.parser_agent
        )
        
        # Task 2: Generate embeddings
        embedding_task = Task(
            description=f"""Create vector embeddings for the resume at {pdf_path} using state-of-the-art 
            embedding models. The embeddings should capture the semantic meaning of the resume content 
            for effective similarity matching and retrieval.""",
            expected_output="Vector embeddings with metadata about the embedding process",
            agent=self.embedding_agent
        )
        
        # Task 3: Analyze the resume
        analysis_task = Task(
            description="""Based on the parsed resume data, provide a comprehensive analysis including:
            1. Seniority level assessment (Junior/Mid-level/Senior/Executive)
            2. Key professional strengths
            3. Technical skills evaluation
            4. Recommendations for improvement
            5. Overall resume quality score (1-10)
            
            Consider industry standards and best practices in your analysis.""",
            expected_output="Detailed analysis report with actionable insights",
            agent=self.analyst_agent,
            context=[parse_task]
        )
        
        # Create and run the crew
        crew = Crew(
            agents=[self.parser_agent, self.embedding_agent, self.analyst_agent],
            tasks=[parse_task, embedding_task, analysis_task],
            verbose=True
        )
        
        # Execute the crew
        result = crew.kickoff()
        
        # Store results
        parsed_data = self.parser.parse_resume(pdf_path)
        embedding_data = self.parser.create_embeddings(pdf_path)
        
        combined_result = {
            "pdf_path": pdf_path,
            "parsed_data": parsed_data.model_dump(),
            "embedding": embedding_data,
            "crew_analysis": str(result),
            "timestamp": str(pd.Timestamp.now()) if 'pd' in globals() else "N/A"
        }
        
        # Update databases
        self.processed_resumes[pdf_path] = combined_result
        if "error" not in embedding_data:
            self.embeddings_database[pdf_path] = embedding_data.get("embedding", [])
        
        return combined_result
    
    def process_batch(self, directory_path: str, limit: int = None) -> List[Dict[str, Any]]:
        """Process multiple resumes from a directory."""
        results = []
        pdf_files = list(Path(directory_path).rglob("*.pdf"))
        
        if limit:
            pdf_files = pdf_files[:limit]
        
        print(f"\n🚀 Processing {len(pdf_files)} resumes with CrewAI...")
        
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n{'='*60}")
            print(f"Processing Resume {i}/{len(pdf_files)}: {pdf_path.name}")
            print(f"{'='*60}")
            
            try:
                result = self.process_resume(str(pdf_path))
                results.append(result)
                
                # Show summary
                parsed = result["parsed_data"]
                print(f"\n📋 SUMMARY:")
                print(f"  Name: {parsed.get('personal_info', {}).get('name', 'Not found')}")
                print(f"  Category: {parsed.get('category', 'Unknown')}")
                print(f"  Experience: {parsed.get('years_of_experience', 'Unknown')} years")
                print(f"  Skills: {len(parsed.get('skills', []))} total")
                
                if "error" not in result["embedding"]:
                    print(f"  Embedding: ✅ {result['embedding']['embedding_dimension']}D vector")
                else:
                    print(f"  Embedding: ❌ {result['embedding']['error']}")
                    
            except Exception as e:
                error_result = {"pdf_path": str(pdf_path), "error": str(e)}
                results.append(error_result)
                print(f"❌ Error processing {pdf_path.name}: {str(e)}")
        
        return results
    
    def save_results(self, output_path: str):
        """Save all processed results to a JSON file."""
        output_data = {
            "total_processed": len(self.processed_resumes),
            "resumes": list(self.processed_resumes.values()),
            "embedding_count": len(self.embeddings_database)
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {output_path}")


def main():
    """Main demo function."""
    print("🎯 CrewAI Resume Processing System")
    print("=" * 60)
    
    # Initialize the crew
    crew_processor = ResumeProcessingCrew()
    
    # Process sample resumes from each category
    data_dir = Path("/Users/ada/fun/AI-Recruiter/data_filtered")
    categories = ["ENGINEERING", "BUSINESS-DEVELOPMENT", "DESIGNER"]
    
    all_results = []
    
    for category in categories:
        category_path = data_dir / category
        if category_path.exists():
            print(f"\n📂 Processing {category} resumes...")
            results = crew_processor.process_batch(str(category_path), limit=1)  # Process 1 per category
            all_results.extend(results)
    
    # Save results
    crew_processor.save_results("crewai_results.json")
    
    print(f"\n🎉 Processing Complete!")
    print(f"Total resumes processed: {len(all_results)}")
    print(f"Embeddings created: {len(crew_processor.embeddings_database)}")


if __name__ == "__main__":
    main()