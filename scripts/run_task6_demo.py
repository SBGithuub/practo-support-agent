"""
run_task6_demo.py - Demonstrates appointment status lookup and escalation score.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import APPOINTMENTS
from app.tools.status_tool import check_appointment_status_raw, compute_escalation_score
from app.config import ESCALATION_THRESHOLD

def main():
    print("=" * 70)
    print("TASK 6: APPOINTMENT STATUS LOOKUP AND CLINICAL ESCALATION SCORING")
    print("=" * 70)
    print(f"Escalation Formula: escalation_score = (0.60 * follow_up_flag) + (0.40 * (days_since_created / 30.0))")
    print(f"Escalation Threshold: {ESCALATION_THRESHOLD}")
    print("Clinical Justification:")
    print("  - follow_up_flag (60% weight) represents critical post-visit clinical care requirements.")
    print("  - days_since_created / 30.0 (40% weight) represents normalized wait time latency.")
    print("  - Any record with follow_up=True waiting > 3.75 days exceeds the 0.65 threshold (80th percentile risk).")
    print("  - Records with follow_up=False cannot exceed 0.40, preventing false positive alarms.")
    print("-" * 70)

    # 1. Distribution analysis across all dataset records
    scores = [compute_escalation_score(r["follow_up_required"], r["days_since_created"]) for r in APPOINTMENTS]
    escalated_count = sum(1 for s in scores if s >= ESCALATION_THRESHOLD)
    print(f"\nTotal Dataset Records: {len(APPOINTMENTS)}")
    print(f"Total Escalated Records: {escalated_count} ({escalated_count / len(APPOINTMENTS) * 100:.1f}%)")
    print(f"Min Score: {min(scores):.4f}, Max Score: {max(scores):.4f}, Mean Score: {sum(scores)/len(scores):.4f}")

    # 2. Targeted lookup test cases
    test_ids = [
        "APT-0001",  # scheduled
        "APT-0002",  # completed with follow up
        "APT-0007",  # high recency
        "APT-9999",  # non-existent
    ]

    print("\n--- Individual Lookup Test Cases ---")
    for rec_id in test_ids:
        res = check_appointment_status_raw(rec_id)
        print(f"\nLookup for: {rec_id}")
        print(f"  Found: {res['found']}")
        if res["found"]:
            print(f"  Category: {res['category']}")
            print(f"  Status: {res['status']}")
            print(f"  Fee (INR): {res['consultation_fee_inr']}")
            print(f"  Days Since Created: {res['days_since_created']}")
            print(f"  Follow-up Required: {res['follow_up_required']}")
            print(f"  Escalation Score: {res['escalation_score']}")
            print(f"  Escalation Recommended: {res['escalation_recommended']}")
        print(f"  Message: {res['message']}")

if __name__ == "__main__":
    main()
