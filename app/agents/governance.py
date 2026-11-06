"""
governance.py - AI Governance policies across Application, Model, System, and Runtime layers.
Implements Task 15.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel

# Governance Risk Levels
RISK_LEVEL_LOW = "Low"
RISK_LEVEL_MEDIUM = "Medium"
RISK_LEVEL_HIGH = "High"

SYSTEM_RISK_CLASSIFICATION = {
    "level": RISK_LEVEL_HIGH,
    "framework": "NIST AI RMF / EU AI Act 3-Tier Classification",
    "justification": (
        "The Practo Domain Support Agent operates in the healthcare domain, interacting with patient appointment "
        "records, specialty consultation fees, and clinical emergency protocols. Systems managing medical data, "
        "patient scheduling, and triage guidance are classified as High Risk because inaccurate triage advice or "
        "improper handling of clinical follow-up states can directly jeopardize patient well-being and violate "
        "statutory health data regulations. To mitigate these risks, the architecture enforces mandatory PII masking, "
        "least-autonomy tool segregation, an independent Autogen policy review stage, and hard runtime budget caps."
    )
}

# Runtime Layer Budget Caps
MAX_REQUEST_TOKENS = 300
MAX_REQUEST_CHARS = 1200
ESTIMATED_COST_PER_1K_TOKENS_USD = 0.002

class GovernanceBudgetError(Exception):
    """Raised when an incoming request exceeds runtime token/cost caps."""
    pass

class LeastAutonomyViolationError(Exception):
    """Raised when an unauthorized agent attempts to access restricted tools."""
    pass

def estimate_tokens(text: str) -> int:
    """Estimates token count assuming standard ~4 characters per token."""
    return max(1, len(text) // 4)

def enforce_runtime_budget(query: str, max_tokens: int = MAX_REQUEST_TOKENS) -> Dict[str, Any]:
    """
    Enforces a strict per-request token and cost budget cap at the Runtime layer.
    Rejects oversized requests rather than silently burning resources or risking buffer attacks.
    """
    token_count = estimate_tokens(query)
    estimated_cost_usd = (token_count / 1000.0) * ESTIMATED_COST_PER_1K_TOKENS_USD
    
    if token_count > max_tokens or len(query) > MAX_REQUEST_CHARS:
        raise GovernanceBudgetError(
            f"Governance Budget Violation: Incoming request of {token_count} tokens ({len(query)} characters) "
            f"exceeds the maximum per-request runtime cap of {max_tokens} tokens. "
            f"Estimated cost ${estimated_cost_usd:.5f} rejected. Please condense your inquiry."
        )
        
    return {
        "status": "APPROVED",
        "token_count": token_count,
        "character_count": len(query),
        "estimated_cost_usd": round(estimated_cost_usd, 6),
        "max_token_budget": max_tokens
    }

def verify_least_autonomy_enforcement(agents: Dict[str, Any]) -> bool:
    """
    Application Layer Guard:
    Verifies that only the Lookup Agent holds the check_appointment_status tool.
    Ensures Retrieval and Composer agents have zero access to database records.
    """
    retrieval_agent = agents.get("retrieval")
    lookup_agent = agents.get("lookup")
    composer_agent = agents.get("composer")
    
    # 1. Lookup agent must have check_appointment_status
    lookup_tool_names = [getattr(t, "name", "") for t in getattr(lookup_agent, "tools", [])]
    if "check_appointment_status" not in lookup_tool_names:
        raise LeastAutonomyViolationError("Lookup Agent is missing required status lookup privileges.")
        
    # 2. Retrieval agent must NOT have check_appointment_status
    retrieval_tool_names = [getattr(t, "name", "") for t in getattr(retrieval_agent, "tools", [])]
    if "check_appointment_status" in retrieval_tool_names:
        raise LeastAutonomyViolationError("Security Violation: Retrieval Agent possesses unauthorized access to appointment records.")
        
    # 3. Response Composer agent must have ZERO tools
    composer_tool_names = [getattr(t, "name", "") for t in getattr(composer_agent, "tools", [])]
    if len(composer_tool_names) > 0:
        raise LeastAutonomyViolationError(f"Security Violation: Response Composer has excessive tool autonomy: {composer_tool_names}.")
        
    return True
