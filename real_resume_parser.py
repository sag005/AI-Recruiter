"""Real resume parser that extracts actual data from PDFs."""

import json
import os
import re
from pathlib import Path
from typing import Dict, Any

import PyPDF2
import voyageai
from anthropic import Anthropic
from dotenv import load_dotenv

from src.ai_recruiter.models import ResumeData, ResumeAnalysis

load_dotenv()


class ResumeParser:
    """Parse resumes and extract structured data."""
    
    def __init__(self):
        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # Initialize Voyage client only if API key is available
        try:
            if os.getenv("VOYAGE_API_KEY"):
                self.voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
            else:
                self.voyage_client = None
        except Exception:
            self.voyage_client = None
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def parse_resume(self, pdf_path: str) -> ResumeData:
        """Parse a resume PDF and return structured data."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Extract text
        pdf_text = self.extract_text_from_pdf(pdf_path)
        
        # Create detailed prompt for Anthropic
        prompt = f"""
        You are an expert resume parser. Analyze the following resume text and extract structured information.
        
        Return a JSON object that matches this exact structure (fill in actual values from the resume, use null for missing data):
        
        {{
            "personal_info": {{
                "name": "actual name from resume",
                "email": "email@example.com or null",
                "phone": "phone number or null",
                "location": "city, state or null",
                "linkedin": "linkedin url or null"
            }},
            "professional_summary": "actual professional summary/objective from resume or null",
            "work_experience": [
                {{
                    "job_title": "actual job title",
                    "company": "actual company name", 
                    "duration": "actual dates worked",
                    "location": "job location or null",
                    "description": "brief description or null",
                    "responsibilities": ["list", "of", "key", "responsibilities"]
                }}
            ],
            "education": [
                {{
                    "degree": "actual degree type",
                    "field_of_study": "major/field or null",
                    "institution": "actual school name",
                    "graduation_year": "year or null",
                    "gpa": "gpa if mentioned or null",
                    "location": "school location or null"
                }}
            ],
            "skills": ["list", "of", "actual", "skills", "from", "resume"],
            "certifications": [
                {{
                    "name": "certification name",
                    "issuer": "issuing org or null", 
                    "year": "year or null",
                    "expiry": "expiry date or null"
                }}
            ],
            "languages": ["list", "of", "languages", "if", "mentioned"],
            "category": "Engineering OR Business Development OR Design (choose the best fit)",
            "years_of_experience": estimated_total_years_as_number,
            "seniority_level": "Junior OR Mid-level OR Senior OR Executive",
            "keywords": ["important", "keywords", "for", "job", "matching"]
        }}
        
        IMPORTANT: Extract ONLY actual information from the resume. Do not make up or infer data that isn't explicitly stated. Use null for missing information.
        
        Resume text:
        {pdf_text[:12000]}
        
        Return only the JSON object, no additional text or formatting.
        """
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Using current model for better extraction
                max_tokens=2000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            
            # Clean up the response to extract JSON
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            try:
                parsed_data = json.loads(response_text)
                
                # Create ResumeData object
                resume_data = ResumeData(**parsed_data)
                resume_data.pdf_path = pdf_path
                
                return resume_data
                
            except json.JSONDecodeError as e:
                # Try to extract JSON with regex if direct parsing fails
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        parsed_data = json.loads(json_match.group())
                        resume_data = ResumeData(**parsed_data)
                        resume_data.pdf_path = pdf_path
                        return resume_data
                    except (json.JSONDecodeError, Exception):
                        pass
                
                # If all parsing fails, create minimal structure
                resume_data = ResumeData(
                    pdf_path=pdf_path,
                    processing_notes=[f"JSON parsing failed: {str(e)}", f"Raw response: {response_text[:200]}..."]
                )
                return resume_data
                
        except Exception as e:
            resume_data = ResumeData(
                pdf_path=pdf_path,
                processing_notes=[f"API call failed: {str(e)}"]
            )
            return resume_data
    
    def create_embeddings(self, pdf_path: str) -> Dict[str, Any]:
        """Create vector embeddings from PDF text using Voyage AI (Anthropic's embedding partner)."""
        if not self.voyage_client:
            return {
                "pdf_path": pdf_path,
                "error": "Voyage AI client not initialized. Set VOYAGE_API_KEY environment variable."
            }
        
        try:
            pdf_text = self.extract_text_from_pdf(pdf_path)
            
            # Voyage AI can handle longer texts, but let's be reasonable
            if len(pdf_text) > 16000:  # Voyage supports much longer context
                pdf_text = pdf_text[:16000] + "..."
            
            # Use Voyage AI's embeddings API (Anthropic's recommended partner)
            result = self.voyage_client.embed(
                texts=[pdf_text], 
                model="voyage-3.5",  # Latest model with 1024 dimensions
                input_type="document"  # This is a document for retrieval
            )
            
            embedding = result.embeddings[0]  # Get the first (and only) embedding
            
            return {
                "pdf_path": pdf_path,
                "embedding": embedding,
                "embedding_dimension": len(embedding),
                "text_length": len(pdf_text),
                "model_used": "voyage-3.5",
                "provider": "voyage_ai"
            }
            
        except Exception as e:
            return {
                "pdf_path": pdf_path,
                "error": f"Embedding generation failed: {str(e)}"
            }


def demo_real_parsing():
    """Demo with actual resume parsing."""
    print("🎯 REAL RESUME PARSING DEMO")
    print("=" * 60)
    
    parser = ResumeParser()
    
    # Find sample PDFs
    data_dir = Path("/Users/ada/fun/AI-Recruiter/data_filtered")
    categories = ["ENGINEERING", "BUSINESS-DEVELOPMENT", "DESIGNER"]
    
    for category in categories:
        category_path = data_dir / category
        if category_path.exists():
            pdf_files = list(category_path.glob("*.pdf"))
            if pdf_files:
                sample_pdf = pdf_files[0]  # Take first PDF
                
                print(f"\n📂 {category} CATEGORY")
                print(f"File: {sample_pdf.name}")
                print("-" * 40)
                
                # Parse resume
                print("📄 EXTRACTING DATA...")
                try:
                    resume_data = parser.parse_resume(str(sample_pdf))
                    
                    # Display results
                    print("✅ PARSING SUCCESSFUL!")
                    print(f"👤 Name: {resume_data.personal_info.name or 'Not found'}")
                    print(f"📧 Email: {resume_data.personal_info.email or 'Not found'}")
                    print(f"📞 Phone: {resume_data.personal_info.phone or 'Not found'}")
                    print(f"📍 Location: {resume_data.personal_info.location or 'Not found'}")
                    
                    print(f"\n🏢 WORK EXPERIENCE ({len(resume_data.work_experience)} jobs):")
                    for i, job in enumerate(resume_data.work_experience[:2], 1):  # Show first 2 jobs
                        print(f"  {i}. {job.job_title} at {job.company}")
                        print(f"     Duration: {job.duration or 'Not specified'}")
                        if job.responsibilities:
                            print(f"     Key responsibility: {job.responsibilities[0]}")
                    
                    print(f"\n🎓 EDUCATION ({len(resume_data.education)} entries):")
                    for edu in resume_data.education:
                        print(f"  • {edu.degree} from {edu.institution}")
                        if edu.graduation_year:
                            print(f"    Year: {edu.graduation_year}")
                    
                    print(f"\n🔧 SKILLS ({len(resume_data.skills)} total):")
                    print(f"  {', '.join(resume_data.skills[:10])}{'...' if len(resume_data.skills) > 10 else ''}")
                    
                    print(f"\n📊 ANALYSIS:")
                    print(f"  Category: {resume_data.category or 'Unknown'}")
                    print(f"  Experience: {resume_data.years_of_experience or 'Unknown'} years")
                    print(f"  Level: {resume_data.seniority_level or 'Unknown'}")
                    
                    if resume_data.certifications:
                        print(f"\n🏆 CERTIFICATIONS ({len(resume_data.certifications)}):")
                        for cert in resume_data.certifications[:3]:
                            print(f"  • {cert.name}")
                    
                    if resume_data.processing_notes:
                        print(f"\n⚠️  PROCESSING NOTES:")
                        for note in resume_data.processing_notes:
                            print(f"  - {note}")
                    
                    # Test embeddings
                    print(f"\n🔢 CREATING EMBEDDINGS...")
                    embedding_result = parser.create_embeddings(str(sample_pdf))
                    
                    if "error" not in embedding_result:
                        print(f"✅ Embedding created: {embedding_result['embedding_dimension']} dimensions")
                        print(f"📝 Text length: {embedding_result['text_length']} characters")
                    else:
                        print(f"❌ Embedding failed: {embedding_result['error']}")
                    
                except Exception as e:
                    print(f"❌ PARSING FAILED: {str(e)}")
                
                print("\n" + "=" * 60)
                break  # Only demo one per category for now


if __name__ == "__main__":
    demo_real_parsing()