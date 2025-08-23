# database/models.py
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import json

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
    embeddings: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self):
        """Convert to dictionary for database storage"""
        return {
            'id': self.id,
            'candidate_id': self.candidate_id,
            'name': self.name,
            'title': self.title,
            'phone': self.phone,
            'email': self.email,
            'skills': json.dumps(self.skills),
            'yoe': self.yoe,
            'relevant_experience': json.dumps(self.relevant_experience),
            'industry': self.industry,
            'raw_text': self.raw_text,
            'embeddings': json.dumps(self.embeddings) if self.embeddings else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create instance from database dictionary"""
        return cls(
            id=data.get('id'),
            candidate_id=data['candidate_id'],
            name=data['name'],
            title=data['title'],
            phone=data['phone'],
            email=data['email'],
            skills=json.loads(data['skills']) if data['skills'] else [],
            yoe=data['yoe'],
            relevant_experience=json.loads(data['relevant_experience']) if data['relevant_experience'] else [],
            industry=data['industry'],
            raw_text=data['raw_text'],
            embeddings=json.loads(data['embeddings']) if data['embeddings'] else None,
            created_at=datetime.fromisoformat(data['created_at']) if data['created_at'] else None
        )

@dataclass
class JobRequirement:
    id: Optional[int]
    title: str
    required_skills: List[str]
    min_yoe: int
    max_yoe: int
    industry: str
    description: str