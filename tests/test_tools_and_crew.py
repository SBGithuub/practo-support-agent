"""
test_tools_and_crew.py - Unit tests for status lookup tool, escalation scoring, and CrewAI pipeline.
"""

import pytest
from app.tools.status_tool import compute_escalation_score, check_appointment_status_raw
from app.config import ESCALATION_THRESHOLD
from app.agents.schemas import SupportResponse
from app.agents.crew import run_practo_support_pipeline

def test_escalation_score_bounds_and_formula():
    # follow_up=False, days=0 -> 0.0
    assert compute_escalation_score(False, 0) == 0.0
    # follow_up=False, days=30 -> 0.40
    assert compute_escalation_score(False, 30) == 0.40
    # follow_up=True, days=0 -> 0.60
    assert compute_escalation_score(True, 0) == 0.60
    # follow_up=True, days=30 -> 1.00
    assert compute_escalation_score(True, 30) == 1.00

def test_escalation_threshold_behavior():
    # Without follow-up, can never exceed 0.65 threshold
    assert compute_escalation_score(False, 30) < ESCALATION_THRESHOLD
    # With follow-up, 4 days (recency 4/30 = 0.1333 -> 0.60 + 0.0533 = 0.6533 >= 0.65)
    assert compute_escalation_score(True, 4) >= ESCALATION_THRESHOLD

def test_appointment_status_lookup():
    res = check_appointment_status_raw("APT-0001")
    assert res["found"] is True
    assert res["record_id"] == "APT-0001"
    assert "status" in res
    assert "consultation_fee_inr" in res
    assert 500 <= res["consultation_fee_inr"] <= 2500

def test_appointment_status_unknown():
    res = check_appointment_status_raw("APT-9999")
    assert res["found"] is False
    assert res["status"] == "Unknown"
    assert res["escalation_score"] == 0.0

def test_crew_pipeline_policy_query():
    res = run_practo_support_pipeline("What is the consultation fee structure by specialty?")
    assert isinstance(res, SupportResponse)
    assert res.category == "Clinic Policy"
    assert len(res.answer) > 20
    assert res.confidence >= 0.50

def test_crew_pipeline_lookup_query():
    res = run_practo_support_pipeline("Check status for appointment APT-0007")
    assert isinstance(res, SupportResponse)
    assert "Cardiology" in res.category
    assert res.sources == ["APT-0007"]
    assert res.action_required is True
