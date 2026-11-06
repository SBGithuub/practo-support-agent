import pytest
from collections import Counter
from dataset import APPOINTMENTS, CATEGORIES, STATUSES, generate_appointments, SEED

def test_total_record_count():
    assert len(APPOINTMENTS) >= 40
    assert len(APPOINTMENTS) == 50

def test_category_representation():
    counts = Counter(r["category"] for r in APPOINTMENTS)
    for cat in CATEGORIES:
        assert cat in counts
        assert counts[cat] >= 3

def test_status_representation():
    counts = Counter(r["status"] for r in APPOINTMENTS)
    for stat in STATUSES:
        assert stat in counts
        assert counts[stat] >= 1

def test_followup_percentage_band():
    followup_count = sum(1 for r in APPOINTMENTS if r["follow_up_required"])
    followup_pct = (followup_count / len(APPOINTMENTS)) * 100.0
    assert 10.0 <= followup_pct <= 30.0

def test_schema_fields():
    required_keys = {
        "record_id",
        "category",
        "status",
        "consultation_fee_inr",
        "days_since_created",
        "follow_up_required"
    }
    for record in APPOINTMENTS:
        assert set(record.keys()) == required_keys
        assert isinstance(record["record_id"], str)
        assert record["record_id"].startswith("APT-")
        assert record["category"] in CATEGORIES
        assert record["status"] in STATUSES
        assert 500 <= record["consultation_fee_inr"] <= 2500
        assert 0 <= record["days_since_created"] <= 30
        assert isinstance(record["follow_up_required"], bool)

def test_dataset_reproducibility():
    fresh_draw = generate_appointments(SEED, 50)
    assert fresh_draw == APPOINTMENTS

