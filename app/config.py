"""
config.py - Central configuration, environment flags, and governance thresholds.
"""

import os
from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Telemetry and Mock Settings (strictly zero network/external API calls)
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

MOCK_LLM_MODE = os.getenv("MOCK_LLM_MODE", "true").lower() == "true"
DISABLE_TELEMETRY = os.getenv("CREWAI_DISABLE_TELEMETRY", "true").lower() == "true"

# Paths
KB_DIR = BASE_DIR / "knowledge_base"
DATA_DIR = BASE_DIR / "data"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
LOGS_FILE = BASE_DIR / "transcripts" / "task12_structured_logs.jsonl"

# ChromaDB Collection Names
COLLECTION_FIXED = "practo_fixed_chunks"
COLLECTION_SENTENCE = "practo_sentence_chunks"

# Empirically Calibrated Retrieval Threshold for Fallback (Task 4)
# Calibrated between in-scope cluster (0.50 - 0.78) and out-of-scope cluster (0.10 - 0.35)
SIMILARITY_THRESHOLD = 0.45

# Designed Escalation Threshold (Task 6)
# Justified at 80th percentile of days_since_created distribution
ESCALATION_THRESHOLD = 0.65

# Runtime Cost / Character Budget Cap (Task 15)
MAX_REQUEST_LENGTH = 3000
MAX_REQUEST_TOKENS = 800

