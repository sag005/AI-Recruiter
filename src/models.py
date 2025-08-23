"""Pydantic models for structured resume data."""

from typing import List, Optional
from pydantic import BaseModel, Field


class PersonalInfo(BaseModel):
    """Personal information from resume."""
    name: Optional[str] = Field(None, description="Full name of the candidate")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="City, State or location")
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL")


class WorkExperience(BaseModel):
    """Work experience entry."""
    job_title: str = Field(..., description="Job title/position")
    company: str = Field(..., description="Company name")
    duration: Optional[str] = Field(None, description="Employment duration (e.g., '2020-2023')")
    location: Optional[str] = Field(None, description="Job location")
    description: Optional[str] = Field(None, description="Brief job description")
    responsibilities: List[str] = Field(default_factory=list, description="Key responsibilities")


class Education(BaseModel):
    """Education entry."""
    degree: str = Field(..., description="Degree type (e.g., 'Bachelor of Science')")
    field_of_study: Optional[str] = Field(None, description="Major/field of study")
    institution: str = Field(..., description="School/University name")
    graduation_year: Optional[str] = Field(None, description="Graduation year")
    gpa: Optional[str] = Field(None, description="GPA if mentioned")
    location: Optional[str] = Field(None, description="School location")


class Certification(BaseModel):
    """Professional certification."""
    name: str = Field(..., description="Certification name")
    issuer: Optional[str] = Field(None, description="Issuing organization")
    year: Optional[str] = Field(None, description="Year obtained")
    expiry: Optional[str] = Field(None, description="Expiry date if applicable")


class ResumeData(BaseModel):
    """Complete structured resume data."""
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo)
    professional_summary: Optional[str] = Field(None, description="Professional summary or objective")
    work_experience: List[WorkExperience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list, description="Technical and professional skills")
    certifications: List[Certification] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list, description="Languages spoken")
    
    # Inferred/analyzed fields
    category: Optional[str] = Field(None, description="Job category: Engineering, Business Development, or Design")
    years_of_experience: Optional[int] = Field(None, description="Estimated total years of experience")
    seniority_level: Optional[str] = Field(None, description="Junior, Mid-level, Senior, or Executive")
    keywords: List[str] = Field(default_factory=list, description="Important keywords for matching")
    
    # Metadata
    pdf_path: Optional[str] = Field(None, description="Source PDF file path")