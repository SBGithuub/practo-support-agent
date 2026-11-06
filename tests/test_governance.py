"""
test_governance.py - Unit tests for AI governance policies, budget caps, and response caching.
Tests Task 15 and Task 16 requirements.
"""

import pytest
from app.agents.governance import (
    enforce_runtime_budget,
    verify_least_autonomy_enforcement,
    GovernanceBudgetError,
    LeastAutonomyViolationError,
    SYSTEM_RISK_CLASSIFICATION,
    RISK_LEVEL_HIGH
)
from app.rag.grounded_gen import (
    generate_grounded_response,
    clear_cache,
    get_cache_stats,
    normalize_query
)
from app.tools.status_tool import check_appointment_status

def test_risk_classification():
    assert SYSTEM_RISK_CLASSIFICATION["level"] == RISK_LEVEL_HIGH
    assert "medical data" in SYSTEM_RISK_CLASSIFICATION["justification"].lower()
    assert len(SYSTEM_RISK_CLASSIFICATION["justification"]) > 50

def test_least_autonomy_enforcement_pass():
    valid_agents = {
        "retrieval": type("Agent", (), {"tools": [type("Tool", (), {"name": "rag_policy_lookup"})()]})(),
        "lookup": type("Agent", (), {"tools": [type("Tool", (), {"name": "check_appointment_status"})()]})(),
        "composer": type("Agent", (), {"tools": []})()
    }
    assert verify_least_autonomy_enforcement(valid_agents) is True

def test_least_autonomy_enforcement_breach():
    # Composer attempting to hold a tool
    invalid_composer = {
        "retrieval": type("Agent", (), {"tools": [type("Tool", (), {"name": "rag_policy_lookup"})()]})(),
        "lookup": type("Agent", (), {"tools": [type("Tool", (), {"name": "check_appointment_status"})()]})(),
        "composer": type("Agent", (), {"tools": [type("Tool", (), {"name": "check_appointment_status"})()]})()
    }
    with pytest.raises(LeastAutonomyViolationError):
        verify_least_autonomy_enforcement(invalid_composer)

    # Retrieval attempting to access appointment status
    invalid_retrieval = {
        "retrieval": type("Agent", (), {"tools": [type("Tool", (), {"name": "check_appointment_status"})()]})(),
        "lookup": type("Agent", (), {"tools": [type("Tool", (), {"name": "check_appointment_status"})()]})(),
        "composer": type("Agent", (), {"tools": []})()
    }
    with pytest.raises(LeastAutonomyViolationError):
        verify_least_autonomy_enforcement(invalid_retrieval)

def test_runtime_budget_cap():
    # Normal query: approved
    normal = "What is the consultation fee for cardiology?"
    res = enforce_runtime_budget(normal, max_tokens=300)
    assert res["status"] == "APPROVED"
    assert res["token_count"] < 300

    # Oversized query: rejected with GovernanceBudgetError
    oversized = "Please explain the entire policy history. " * 50
    with pytest.raises(GovernanceBudgetError):
        enforce_runtime_budget(oversized, max_tokens=100)

def test_response_cache_hit_and_normalization():
    clear_cache()
    q = "What is the cancellation window for appointments?"
    q_dirty = "   WHAT is the cancellation window for appointments???   "
    
    # 1. Cold call
    r1 = generate_grounded_response(q, use_cache=True)
    assert r1["cache_hit"] is False
    
    # 2. Warm call with punctuation/casing variations
    r2 = generate_grounded_response(q_dirty, use_cache=True)
    assert r2["cache_hit"] is True
    assert r1["answer"] == r2["answer"]
    
    stats = get_cache_stats()
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1
