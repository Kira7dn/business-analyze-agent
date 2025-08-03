"""
Data models and schemas for Business Analyze Agent
"""

from typing import Dict, List
from pydantic import BaseModel


class RequirementAnalysis(BaseModel):
    """Model for requirement analysis results"""

    requirements: List[str]
    moscow_priority: Dict[str, List[str]]
    gaps: List[str]
    suggestions: List[str]


class QuestionGeneration(BaseModel):
    """Model for question generation results"""

    questions: List[str]
    focus_area: str
    context: str
