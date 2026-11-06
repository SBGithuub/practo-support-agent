"""
test_api.py - Unit tests for FastAPI HTTP and WebSocket endpoints.
Tests Task 11 and Task 12 logging requirements.
"""

import pytest
from fastapi.testclient import TestClient
from app.api.main import app
from app.api.logger import log_request_event, LOG_FILE_PATH
import os
import json

client = TestClient(app)

def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

def test_api_ask_policy():
    resp = client.post("/ask", json={"query": "What is the cancellation window for appointments?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "cancellation" in data["query"].lower()
    assert len(data["answer"]) > 10

def test_api_ask_appointment():
    resp = client.post("/ask", json={"query": "Check status for appointment APT-0001"})
    assert resp.status_code == 200
    data = resp.json()
    assert "APT-0001" in data["sources"]

def test_api_add_document():
    payload = {
        "doc_id": "test_cardiology_policy.txt",
        "category": "Cardiology",
        "content": (
            "Cardiology consultation pre-screening requires recent electrocardiogram records. "
            "Patients with pacemakers must notify clinic triage staff upon check-in. "
            "Emergency cardiac interventions are directed to tertiary cardiac centers."
        )
    }
    resp = client.post("/add-document", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["chunks_indexed"] > 0

def test_websocket_chat_and_graceful_disconnect():
    with client.websocket_connect("/ws/chat") as ws:
        greeting = ws.receive_json()
        assert greeting["type"] == "connected"
        
        # Turn 1
        ws.send_text("Checking appointment APT-0001 status")
        r1 = ws.receive_json()
        assert r1["type"] == "response"
        assert len(r1["answer"]) > 10
        
        # Client disconnects intentionally
        ws.close()
        
    # Server remains alive
    h = client.get("/health")
    assert h.status_code == 200

def test_structured_log_pii_redaction():
    # Test that logging an entry with phone number redacts it on disk
    test_query = "Call me at +91 9876543210 about APT-0001"
    event = log_request_event(
        endpoint="TEST /endpoint",
        raw_query=test_query,
        status="SUCCESS",
        latency_ms=12.5
    )
    assert event["pii_redacted"] is True
    assert "[REDACTED_PHONE]" in event["query"]
    assert "+91 9876543210" not in event["query"]
    
    # Check that disk contains redacted token
    assert os.path.exists(LOG_FILE_PATH)
    with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
        last_line = json.loads(lines[-1])
        assert "[REDACTED_PHONE]" in last_line["query"]
