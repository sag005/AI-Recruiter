# database/models.py
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class Candidate:
    id: Optional[int]
    name: str
    yoe: int
    current_title: str
    industry: str
    email: str
    phone: str
    status: str  # new, contacted, responded, slot_sent, scheduled, interviewed
    created_at: Optional[datetime] = None

@dataclass
class ParsedResume:
    id: Optional[int]
    candidate_id: int
    name: str
    title: str
    phone: str
    email: str
    skills: List[str]
    yoe: int
    relevant_experience: List[str]
    industry: str
    raw_text: str

@dataclass
class JobRequirement:
    id: Optional[int]
    title: str
    required_skills: List[str]
    min_yoe: int
    max_yoe: int
    industry: str
    description: str