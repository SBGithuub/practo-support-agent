"""
status_tool.py - Appointment status lookup with designed clinical escalation scoring.
"""

from dataset import APPOINTMENTS
from app.config import ESCALATION_THRESHOLD
from crewai.tools import tool

# Index appointments by record_id for O(1) lookup
APPOINTMENTS_BY_ID = {rec["record_id"]: rec for rec in APPOINTMENTS}

def compute_escalation_score(follow_up_required: bool, days_since_created: int) -> float:
    """
    Computes an escalation score in [0.0, 1.0] combining clinical necessity and wait recency.
    
    Formula:
      escalation_score = (0.60 * follow_up_flag) + (0.40 * (days_since_created / 30.0))
      
    Justification:
      - follow_up_required contributes 60% of the weight because pending medical care is the primary risk.
      - normalized recency signal (days_since_created / 30.0) contributes 40%, reflecting cumulative wait time.
      - The escalation threshold is 0.65.
      - Appointments without follow_up_required max out at 0.40 (even at 30 days) avoiding false alerts.
      - Appointments with follow_up_required exceed 0.65 once pending for >3.75 days (outside the standard 72h window).
    """
    follow_up_flag = 1.0 if follow_up_required else 0.0
    recency_ratio = max(0.0, min(30.0, float(days_since_created))) / 30.0
    
    score = (0.60 * follow_up_flag) + (0.40 * recency_ratio)
    return round(score, 4)

def check_appointment_status_raw(record_id: str) -> dict:
    """Core status lookup returning full structured dictionary."""
    clean_id = record_id.strip().upper()
    record = APPOINTMENTS_BY_ID.get(clean_id)
    
    if not record:
        return {
            "found": False,
            "record_id": clean_id,
            "status": "Unknown",
            "consultation_fee_inr": 0,
            "escalation_score": 0.0,
            "escalation_recommended": False,
            "message": f"Appointment record '{clean_id}' was not found in Practo database."
        }
        
    score = compute_escalation_score(record["follow_up_required"], record["days_since_created"])
    escalation_recommended = score >= ESCALATION_THRESHOLD
    
    return {
        "found": True,
        "record_id": record["record_id"],
        "category": record["category"],
        "status": record["status"],
        "consultation_fee_inr": record["consultation_fee_inr"],
        "days_since_created": record["days_since_created"],
        "follow_up_required": record["follow_up_required"],
        "escalation_score": score,
        "escalation_recommended": escalation_recommended,
        "message": (
            f"Appointment {record['record_id']} ({record['category']}) is currently '{record['status']}'. "
            f"Consultation fee: INR {record['consultation_fee_inr']}. "
            f"Escalation Score: {score} ({'Escalation Required' if escalation_recommended else 'Standard Priority'})."
        )
    }

@tool("check_appointment_status")
def check_appointment_status(record_id: str) -> str:
    """
    Lookup a specific appointment record status, fee, and clinical escalation score by record_id.
    Input argument: record_id (str) - e.g. 'APT-0001'.
    """
    res = check_appointment_status_raw(record_id)
    return res["message"]
