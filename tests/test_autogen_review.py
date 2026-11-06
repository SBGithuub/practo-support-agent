"""
test_autogen_review.py - Unit tests for Autogen 2-agent RoundRobinGroupChat review stage.
Tests Task 14 requirements.
"""

import pytest
from app.agents.autogen_review import run_autogen_review
from app.agents.schemas import ReviewVerdict

def test_autogen_review_approves_compliant_draft():
    draft = "Patients may cancel appointments up to two hours prior with zero penalty."
    context = "cancellation_rescheduling.txt: Patients may cancel appointments up to two hours prior with zero penalty."
    
    verdict = run_autogen_review(draft, context)
    assert isinstance(verdict, ReviewVerdict)
    assert verdict.approved is True
    assert verdict.final_answer == draft
    assert len(verdict.reason) > 10

def test_autogen_review_revises_ungrounded_claim():
    draft = "Practo guarantees 100% free surgery for all walk-in consultations across all clinics."
    context = "consultation_fees.txt: Consultation fees follow specialty pricing ranging from 500 to 2500 INR."
    
    verdict = run_autogen_review(draft, context)
    assert isinstance(verdict, ReviewVerdict)
    assert verdict.approved is False
    assert "100% free surgery" not in verdict.final_answer
    assert "ungrounded" in verdict.reason.lower() or "caught" in verdict.reason.lower()
