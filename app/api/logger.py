"""
logger.py - ELK-style structured JSON-Lines logger for request tracing and observability.
Guarantees PII redaction before writing to disk (Task 12).
"""

import os
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from app.guardrails.pii_masker import mask_pii

LOG_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "transcripts", "task12_structured_logs.jsonl")
)

def log_request_event(
    endpoint: str,
    raw_query: str,
    status: str,
    latency_ms: float,
    metadata: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Applies PII masking and writes a single structured JSON-Lines entry to task12_structured_logs.jsonl.
    Ensures that raw PII never reaches disk in the clear.
    """
    if trace_id is None:
        trace_id = str(uuid.uuid4())
        
    # Task 12 Requirement: Mask PII before logging to disk
    safe_query, had_pii = mask_pii(raw_query)
    
    event = {
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "query": safe_query,
        "pii_redacted": had_pii,
        "latency_ms": round(latency_ms, 2),
        "status": status,
        "metadata": metadata or {}
    }
    
    # Ensure transcripts directory exists
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    
    # Append line to task12_structured_logs.jsonl
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
        
    return event

