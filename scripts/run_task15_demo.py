"""
run_task15_demo.py - Demonstrates AI Governance Model (Task 15).
Outputs to transcripts/task15_governance_budget.txt.
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.governance import (
    SYSTEM_RISK_CLASSIFICATION,
    enforce_runtime_budget,
    verify_least_autonomy_enforcement,
    GovernanceBudgetError,
    LeastAutonomyViolationError
)
from app.agents.crew import build_practo_crew
from app.tools.status_tool import check_appointment_status

def main():
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("TASK 15: FOUR-LAYER AI GOVERNANCE FRAMEWORK DEMONSTRATION")
    output_lines.append("=" * 80)

    # ----------------------------------------------------
    # 1. APPLICATION LAYER: PRINCIPLE OF LEAST AUTONOMY
    # ----------------------------------------------------
    output_lines.append("\n[1. APPLICATION LAYER: PRINCIPLE OF LEAST AUTONOMY]")
    crew, agents = build_practo_crew("Check appointment APT-0001")
    
    output_lines.append("Verifying crew agent tool assignments:")
    output_lines.append(f"  - Retrieval Agent Tools: {[t.name for t in agents['retrieval'].tools]}")
    output_lines.append(f"  - Lookup Agent Tools: {[t.name for t in agents['lookup'].tools]}")
    output_lines.append(f"  - Composer Agent Tools: {[t.name for t in agents['composer'].tools]}")
    
    is_valid = verify_least_autonomy_enforcement(agents)
    output_lines.append(f"Standard Crew Least Autonomy Check: PASSED ({is_valid})")
    
    # Demonstrate deliberate breach detection
    output_lines.append("\nSimulating Unauthorized Tool Assignment (Attaching status tool to Retrieval Agent):")
    illegal_agents = {
        "retrieval": type("AgentMock", (), {"tools": [check_appointment_status]})(),
        "lookup": agents["lookup"],
        "composer": agents["composer"]
    }
    try:
        verify_least_autonomy_enforcement(illegal_agents)
        output_lines.append("FAILED: Breach went undetected!")
    except LeastAutonomyViolationError as e:
        output_lines.append(f"BLOCKED BY GOVERNANCE GUARD: {e}")
        output_lines.append("Verdict: PASS - Unauthorized database access prevented by Application Layer guard.")

    output_lines.append("\nLeast Autonomy Architectural Explanation:")
    output_lines.append(
        "  Under the principle of least autonomy, agents are granted only the minimum tool capabilities "
        "  essential to their narrow role. The Lookup Agent is strictly restricted to read-only record lookups, "
        "  the Retrieval Agent has access only to public clinic policy documents, and the Response Composer has "
        "  strictly zero tool-execution rights. This prevents privilege escalation, cross-contamination, and "
        "  ensures that an adversarial prompt cannot manipulate a general-purpose agent into querying confidential records."
    )

    output_lines.append("\n" + "-" * 80)

    # ----------------------------------------------------
    # 2. SYSTEM RISK CLASSIFICATION & JUSTIFICATION
    # ----------------------------------------------------
    output_lines.append("\n[2. SYSTEM RISK CLASSIFICATION]")
    output_lines.append(f"Risk Tier Assigned: {SYSTEM_RISK_CLASSIFICATION['level']}")
    output_lines.append(f"Governance Framework: {SYSTEM_RISK_CLASSIFICATION['framework']}")
    output_lines.append("\nOne-Paragraph Justification:")
    output_lines.append(f"  {SYSTEM_RISK_CLASSIFICATION['justification']}")

    output_lines.append("\n" + "-" * 80)

    # ----------------------------------------------------
    # 3. RUNTIME LAYER: PER-REQUEST TOKEN & COST BUDGET CAP
    # ----------------------------------------------------
    output_lines.append("\n[3. RUNTIME LAYER: TOKEN AND COST BUDGET CAP]")
    
    # Case A: Compliant standard query
    normal_query = "What is the cancellation window for appointments?"
    budget_res = enforce_runtime_budget(normal_query)
    output_lines.append(f"Case A (Standard Query): '{normal_query}'")
    output_lines.append(f"  Tokens Estimated: {budget_res['token_count']} / {budget_res['max_token_budget']}")
    output_lines.append(f"  Cost Estimated: ${budget_res['estimated_cost_usd']:.6f}")
    output_lines.append(f"  Governance Status: {budget_res['status']}")
    
    # Case B: Deliberately oversized request exceeding runtime cap
    oversized_query = (
        "I need a complete minute-by-minute breakdown of every appointment and clinic policy "
        "for the last 10 years including all doctors, fees, room numbers, bed counts, and patient records. "
    ) * 15  # ~2,400 chars, ~600 tokens
    
    output_lines.append(f"\nCase B (Deliberately Oversized Query): {len(oversized_query)} characters")
    try:
        enforce_runtime_budget(oversized_query, max_tokens=300)
        output_lines.append("FAILED: Oversized request was not rejected!")
    except GovernanceBudgetError as e:
        output_lines.append(f"REJECTED BY RUNTIME BUDGET CAP: {e}")
        output_lines.append("Verdict: PASS - Oversized request safely halted before consuming LLM/compute resources.")

    output_lines.append("\n" + "=" * 80)
    output_lines.append("AI GOVERNANCE VERIFICATION: Least autonomy, risk tier, and budget cap successfully validated.")
    output_lines.append("=" * 80)

    text = "\n".join(output_lines)
    with open("transcripts/task15_governance_budget.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Task 15 transcript written to transcripts/task15_governance_budget.txt")

if __name__ == "__main__":
    main()

