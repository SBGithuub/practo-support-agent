"""
schemas.py - Pydantic structured output models for CrewAI and Autogen agents.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

class SupportResponse(BaseModel):
    """
    Standard structured output schema that every CrewAI final response conforms to (Task 9).
    """
    query: str = Field(description="The original or normalized patient inquiry.")
    category: str = Field(description="Clinical or administrative category of the request.")
    answer: str = Field(description="The comprehensive, grounded response delivered to the patient.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0.")
    sources: List[str] = Field(default_factory=list, description="List of knowledge base files or record IDs consulted.")
    action_required: bool = Field(default=False, description="Whether immediate clinic staff action is needed.")
    escalation_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Escalation score from status tool if applicable.")
    follow_up_recommended: bool = Field(default=False, description="Whether patient follow-up or rebooking is recommended.")

class ReviewVerdict(BaseModel):
    """
    Structured output schema for the Autogen 2-agent review stage (Task 14).
    """
    approved: bool = Field(description="True if the draft response is fully compliant and grounded; False if revised.")
    final_answer: str = Field(description="The finalized, verified patient response (unchanged if approved, edited if revised).")
    reason: str = Field(description="Explanation of the review verdict or details of revisions made.")
