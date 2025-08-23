# tools/candidate_search_tool.py
from crewai.tools import tool
from database.operations import DatabaseOperations
from typing import List
from database.models import Candidate


@tool("candidate_search")
def candidate_search_tool(max_candidates: int = 50) -> str:
    """
    Fetch all candidates from the database.
    
    Args:
        max_candidates: Maximum number of candidates to return
        
    Returns:
        String summary of all candidates found
    """
    db = DatabaseOperations()
    
    try:
        # Fetch all candidates from database
        candidates = db.get_all_candidates()
        
        if not candidates:
            return "No candidates found in database"
        
        # Limit results
        limited_candidates = candidates[:max_candidates]
        
        # Format response
        candidate_summary = []
        for candidate in limited_candidates:
            summary = f"ID: {candidate.id}, Name: {candidate.name}, Status: {candidate.status}, Title: {candidate.current_title}, YOE: {candidate.yoe}"
            candidate_summary.append(summary)
        
        result = f"Found {len(limited_candidates)} total candidates:\n"
        result += "\n".join(candidate_summary)
        
        return result
        
    except Exception as e:
        return f"Error fetching candidates: {str(e)}"