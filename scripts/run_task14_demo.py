"""
run_task14_demo.py - Demonstrates Autogen 2-agent RoundRobinGroupChat Review Stage (Task 14).
Outputs to transcripts/task14_autogen_review.txt.
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.autogen_review import run_autogen_review
from app.agents.schemas import ReviewVerdict

def main():
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("TASK 14: AUTOGEN 2-AGENT REVIEW STAGE (ROUND ROBIN GROUP CHAT)")
    output_lines.append("=" * 80)
    output_lines.append("Team Configuration:")
    output_lines.append("  - Agent 1: PolicyComplianceReviewer (Analyzes draft against retrieved context)")
    output_lines.append("  - Agent 2: FinalEditor (Generates structured ReviewVerdict output)")
    output_lines.append("  - Orchestrator: RoundRobinGroupChat (max_turns=2, custom_message_types=[StructuredMessage[ReviewVerdict]])")
    output_lines.append("Execution Mode: Deterministic MOCK_LLM (Zero API Keys, Zero Network)")
    output_lines.append("-" * 80)

    # ----------------------------------------------------
    # SCENARIO 1: COMPLIANT DRAFT APPROVED UNCHANGED
    # ----------------------------------------------------
    draft_1 = (
        "According to Practo's official policy documentation: Patients may cancel or reschedule "
        "any confirmed appointment without penalty up to two hours before the designated consultation time."
    )
    context_1 = (
        "cancellation_rescheduling.txt: Patients may cancel or reschedule any confirmed appointment "
        "without penalty up to two hours before the designated consultation time."
    )
    
    output_lines.append("\n[SCENARIO 1: DRAFT APPROVED UNCHANGED]")
    output_lines.append(f"Input Draft Answer:\n  \"{draft_1}\"")
    output_lines.append(f"Retrieved Context:\n  \"{context_1}\"")
    output_lines.append("\nRunning Autogen Review Team...")
    
    verdict_1: ReviewVerdict = run_autogen_review(draft_1, context_1)
    
    output_lines.append("\nReview Verdict (Structured Pydantic Output):")
    output_lines.append(json.dumps(verdict_1.model_dump(), indent=2))
    output_lines.append(f"Approved: {verdict_1.approved}")
    output_lines.append(f"Reason: {verdict_1.reason}")
    output_lines.append(f"Final Answer: {verdict_1.final_answer}")
    
    assert verdict_1.approved is True, "Scenario 1 must be approved"
    assert verdict_1.final_answer == draft_1, "Final answer must match draft unchanged"

    output_lines.append("\n" + "-" * 80)

    # ----------------------------------------------------
    # SCENARIO 2: DELIBERATELY INJECTED UNGROUNDED CLAIM REVISED
    # ----------------------------------------------------
    draft_2 = (
        "Practo clinics provide specialty appointments and guarantees 100% free surgery for all walk-in patients."
    )
    context_2 = (
        "consultation_fees.txt: Consultation charges follow a tiered specialty pricing model ranging from INR 500 to 2500."
    )
    
    output_lines.append("\n[SCENARIO 2: DELIBERATELY INJECTED UNGROUNDED CLAIM REVISED]")
    output_lines.append(f"Input Draft Answer (with injected ungrounded claim):\n  \"{draft_2}\"")
    output_lines.append(f"Retrieved Context:\n  \"{context_2}\"")
    output_lines.append("\nRunning Autogen Review Team...")
    
    verdict_2: ReviewVerdict = run_autogen_review(draft_2, context_2)
    
    output_lines.append("\nReview Verdict (Structured Pydantic Output):")
    output_lines.append(json.dumps(verdict_2.model_dump(), indent=2))
    output_lines.append(f"Approved: {verdict_2.approved}")
    output_lines.append(f"Reason: {verdict_2.reason}")
    output_lines.append(f"Final Revised Answer: {verdict_2.final_answer}")
    
    assert verdict_2.approved is False, "Scenario 2 must reject ungrounded claim"
    assert "100% free surgery" not in verdict_2.final_answer, "Injected ungrounded claim must be purged"

    output_lines.append("\n" + "=" * 80)
    output_lines.append("AUTOGEN REVIEW VERIFICATION: Both approval and revision workflows verified successfully.")
    output_lines.append("=" * 80)

    text = "\n".join(output_lines)
    with open("transcripts/task14_autogen_review.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Task 14 transcript written to transcripts/task14_autogen_review.txt")

if __name__ == "__main__":
    main()

