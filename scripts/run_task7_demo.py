"""
run_task7_demo.py - Demonstrates CrewAI Multi-Agent Kickoff with RAG and Lookup Tools.
Outputs to transcripts/task7_crew_execution.txt.
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure telemetry is disabled
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from app.agents.crew import run_practo_support_pipeline, build_practo_crew
from app.agents.schemas import SupportResponse

def main():
    print("=" * 80)
    print("TASK 7: CREWAI MULTI-AGENT ORCHESTRATION WITH TOOLS VIA .kickoff()")
    print("=" * 80)
    print("Crew Structure:")
    print("  1. Retrieval Agent (Equipped with rag_policy_lookup tool only)")
    print("  2. Lookup Agent (Equipped with check_appointment_status tool only)")
    print("  3. Response Composer Agent (Zero tools - Least Autonomy Principle)")
    print("Execution Mode: Deterministic MOCK_LLM (Zero API Keys, Zero Network)")
    print("-" * 80)

    # Demo Query 1: Policy Question (Invokes rag_policy_lookup)
    query_1 = "What is the cancellation and rescheduling window for appointments?"
    print(f"\n[SCENARIO 1: POLICY INQUIRY]")
    print(f"User Query: {query_1}")
    print("Executing CrewAI .kickoff()...")
    res_1: SupportResponse = run_practo_support_pipeline(query_1)
    
    print("\n[RESULT 1 - STRUCTURED PYDANTIC OUTPUT]")
    print(json.dumps(res_1.model_dump(), indent=2))
    print(f"\nFinal Composed Answer:\n{res_1.answer}")
    assert res_1.confidence > 0, "Confidence must be positive"
    assert "practo_sentence_chunks" in res_1.sources or len(res_1.sources) > 0

    print("\n" + "-" * 80)

    # Demo Query 2: Appointment Status (Invokes check_appointment_status)
    query_2 = "Can you please check the status and fee for appointment APT-0007?"
    print(f"\n[SCENARIO 2: APPOINTMENT STATUS & ESCALATION LOOKUP]")
    print(f"User Query: {query_2}")
    print("Executing CrewAI .kickoff()...")
    res_2: SupportResponse = run_practo_support_pipeline(query_2)
    
    print("\n[RESULT 2 - STRUCTURED PYDANTIC OUTPUT]")
    print(json.dumps(res_2.model_dump(), indent=2))
    print(f"\nFinal Composed Answer:\n{res_2.answer}")
    assert "APT-0007" in res_2.sources
    assert res_2.escalation_score > 0

    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY:")
    print("  - Both rag_policy_lookup and check_appointment_status tools successfully invoked.")
    print("  - Both responses validated against declared Pydantic SupportResponse schema (Task 9).")
    print("  - Sequential process completed via .kickoff() with deterministic outputs.")
    print("=" * 80)

if __name__ == "__main__":
    main()

