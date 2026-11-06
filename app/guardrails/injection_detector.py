"""
injection_detector.py - Heuristic detector for prompt injection and system jailbreak attempts.
"""

import re

INJECTION_PATTERNS = [
    r'ignore\s+(all\s+)?(previous|prior|above)\s+instructions',
    r'disregard\s+(all\s+)?(previous|prior|above)\s+rules',
    r'you\s+are\s+now\s+in\s+developer\s+mode',
    r'dan\s+mode',
    r'bypass\s+all\s+(filters|guardrails|safety)',
    r'act\s+as\s+an\s+unrestricted',
    r'system\s+override',
    r'reveal\s+your\s+system\s+prompt',
    r'drop\s+table',
    r'delete\s+from\s+appointments',
    r'jailbreak',
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

def detect_prompt_injection(text: str) -> tuple[bool, str]:
    """
    Detects if the text contains adversarial prompt injection attacks.
    Returns (is_injection, reason).
    """
    if not text:
        return False, ""
    
    clean_text = text.strip()
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(clean_text):
            return True, f"Security Violation: Potential prompt injection or system override detected matching pattern '{pattern.pattern}'."
            
    return False, ""

